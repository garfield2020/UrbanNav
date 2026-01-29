# ============================================================================
# SourceContract.jl - Source Coupling Contract (Phase G Step 1)
# ============================================================================
#
# Defines the coupling modes, gates, and configuration for how detected
# magnetic dipole sources interact with the navigation filter.
#
# Coupling Progression:
#   SOURCE_SHADOW → SOURCE_COV_ONLY → SOURCE_SUBTRACT
#
# Each escalation requires passing additional gates (evidence, observability,
# stability, predictive). This ensures sources never corrupt navigation
# unless they are well-characterized.
#
# Safety: apply_source_coupling! is bounded and reversible.
#
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Coupling Mode
# ============================================================================

"""
    SourceCouplingMode

Defines how a tracked source interacts with the navigation filter.

- `SOURCE_SHADOW`: Estimate source parameters, but zero nav feedback.
  Nav update proceeds as if source does not exist.
- `SOURCE_COV_ONLY`: Inflate innovation covariance S to account for
  source uncertainty, but do not modify innovation vector.
- `SOURCE_SUBTRACT`: Subtract predicted source field from innovation
  AND inflate covariance. Bounded by max_subtraction_field_T.
"""
@enum SourceCouplingMode begin
    SOURCE_SHADOW = 0
    SOURCE_COV_ONLY = 1
    SOURCE_SUBTRACT = 2
end

# ============================================================================
# Coupling Gates
# ============================================================================

"""
    SourceCouplingGates

Boolean gates that must ALL pass for a source to escalate coupling mode.

# Fields
- `evidence::Bool`: Gate A — accumulated log-likelihood ratio exceeds threshold
- `observability::Bool`: Gate B — source FIM position block is full-rank
- `stability::Bool`: Gate C — covariance trace is decreasing over window
- `predictive::Bool`: Gate D — held-out residual improvement exceeds threshold
"""
struct SourceCouplingGates
    evidence::Bool
    observability::Bool
    stability::Bool
    predictive::Bool
end

"""All gates pass."""
all_gates_pass(g::SourceCouplingGates) = g.evidence && g.observability && g.stability && g.predictive

"""Count number of passing gates."""
n_gates_passing(g::SourceCouplingGates) = count([g.evidence, g.observability, g.stability, g.predictive])

# ============================================================================
# Gate Thresholds
# ============================================================================

"""
    SourceGateThresholds

Thresholds for evaluating coupling gates.

# Fields
- `min_log_likelihood_ratio::Float64`: Gate A threshold [nats] (default 10.0)
- `min_fim_rank::Int`: Gate B minimum rank (6 = full for position+moment)
- `max_position_crlb_m::Float64`: Gate B maximum position CRLB [m] (default 1.0)
- `stability_window::Int`: Gate C observation window (default 10)
- `min_held_out_improvement_pct::Float64`: Gate D improvement threshold [%] (default 10.0)
"""
struct SourceGateThresholds
    min_log_likelihood_ratio::Float64
    min_fim_rank::Int
    max_position_crlb_m::Float64
    stability_window::Int
    min_held_out_improvement_pct::Float64

    function SourceGateThresholds(;
        min_log_likelihood_ratio::Float64 = 10.0,
        min_fim_rank::Int = 6,
        max_position_crlb_m::Float64 = 1.0,
        stability_window::Int = 10,
        min_held_out_improvement_pct::Float64 = 10.0
    )
        @assert min_log_likelihood_ratio > 0 "LLR threshold must be positive"
        @assert 1 <= min_fim_rank <= 6 "FIM rank must be in [1,6]"
        @assert max_position_crlb_m > 0 "Position CRLB must be positive"
        @assert stability_window >= 2 "Stability window must be >= 2"
        @assert min_held_out_improvement_pct > 0 "Improvement threshold must be positive"
        new(min_log_likelihood_ratio, min_fim_rank, max_position_crlb_m,
            stability_window, min_held_out_improvement_pct)
    end
end

const DEFAULT_SOURCE_GATE_THRESHOLDS = SourceGateThresholds()

# ============================================================================
# Coupling Configuration
# ============================================================================

"""
    SourceCouplingConfig

Configuration for source-navigation coupling.

# Fields
- `max_mode::SourceCouplingMode`: Maximum allowed coupling mode (default SOURCE_SUBTRACT)
- `gate_thresholds::SourceGateThresholds`: Thresholds for gate evaluation
- `nav_covariance_inflation_T2::Float64`: COV_ONLY σ² inflation [T²] (default 1e-16)
- `max_subtraction_field_T::Float64`: SUBTRACT bounded influence [T] (default 1e-6)
- `cross_covariance_enabled::Bool`: Enable cross-covariance (false in v1)

# Safety
- max_subtraction_field_T bounds the maximum field subtracted from innovation
  to prevent runaway source estimates from corrupting navigation.
- cross_covariance_enabled=false decouples source and nav blocks (conservative).
"""
struct SourceCouplingConfig
    max_mode::SourceCouplingMode
    gate_thresholds::SourceGateThresholds
    nav_covariance_inflation_T2::Float64
    max_subtraction_field_T::Float64
    cross_covariance_enabled::Bool

    function SourceCouplingConfig(;
        max_mode::SourceCouplingMode = SOURCE_SUBTRACT,
        gate_thresholds::SourceGateThresholds = DEFAULT_SOURCE_GATE_THRESHOLDS,
        nav_covariance_inflation_T2::Float64 = 1e-16,
        max_subtraction_field_T::Float64 = 1e-6,
        cross_covariance_enabled::Bool = false
    )
        @assert nav_covariance_inflation_T2 >= 0 "Inflation must be non-negative"
        @assert max_subtraction_field_T > 0 "Max subtraction field must be positive"
        new(max_mode, gate_thresholds, nav_covariance_inflation_T2,
            max_subtraction_field_T, cross_covariance_enabled)
    end
end

const DEFAULT_SOURCE_COUPLING_CONFIG = SourceCouplingConfig()

# ============================================================================
# Gate Evaluation
# ============================================================================

"""
    evaluate_coupling_gates(log_likelihood_ratio, fim_rank, position_crlb_m,
                            cov_trace_history, held_out_improvement_pct,
                            thresholds) -> SourceCouplingGates

Evaluate all four coupling gates against thresholds.

# Arguments
- `log_likelihood_ratio::Float64`: Accumulated LLR for source vs no-source [nats]
- `fim_rank::Int`: Numerical rank of 6×6 source FIM
- `position_crlb_m::Float64`: Position CRLB (max of xyz components) [m]
- `cov_trace_history::Vector{Float64}`: Recent covariance trace values
- `held_out_improvement_pct::Float64`: Held-out residual improvement [%]
- `thresholds::SourceGateThresholds`: Gate thresholds
"""
function evaluate_coupling_gates(log_likelihood_ratio::Float64,
                                  fim_rank::Int,
                                  position_crlb_m::Float64,
                                  cov_trace_history::Vector{Float64},
                                  held_out_improvement_pct::Float64,
                                  thresholds::SourceGateThresholds)
    # Gate A: Evidence
    evidence = log_likelihood_ratio >= thresholds.min_log_likelihood_ratio

    # Gate B: Observability (rank + CRLB)
    observability = fim_rank >= thresholds.min_fim_rank &&
                    position_crlb_m <= thresholds.max_position_crlb_m

    # Gate C: Stability (covariance trace decreasing over window)
    stability = false
    if length(cov_trace_history) >= thresholds.stability_window
        window = cov_trace_history[end - thresholds.stability_window + 1 : end]
        # Check that trace is non-increasing (allow small fluctuations)
        diffs = diff(window)
        stability = all(d -> d <= 0.01 * window[1], diffs)  # Allow 1% tolerance per step
    end

    # Gate D: Predictive (held-out improvement)
    predictive = held_out_improvement_pct >= thresholds.min_held_out_improvement_pct

    return SourceCouplingGates(evidence, observability, stability, predictive)
end

# ============================================================================
# Effective Coupling Mode
# ============================================================================

"""
    effective_coupling_mode(gates::SourceCouplingGates, config::SourceCouplingConfig)
        -> SourceCouplingMode

Determine the effective coupling mode based on gate status and max_mode cap.

# Logic
- If no gates pass → SOURCE_SHADOW
- If evidence + observability pass → SOURCE_COV_ONLY (minimum for inflation)
- If ALL 4 gates pass → SOURCE_SUBTRACT
- Result is capped at config.max_mode
"""
function effective_coupling_mode(gates::SourceCouplingGates, config::SourceCouplingConfig)
    if all_gates_pass(gates)
        mode = SOURCE_SUBTRACT
    elseif gates.evidence && gates.observability
        mode = SOURCE_COV_ONLY
    else
        mode = SOURCE_SHADOW
    end

    # Cap at max_mode
    if Int(mode) > Int(config.max_mode)
        mode = config.max_mode
    end

    return mode
end

# ============================================================================
# Apply Source Coupling
# ============================================================================

"""
    apply_source_coupling!(innovation::Vec3Map, S::Mat3Map,
                            source_fields::Vector{Vec3Map},
                            source_covariances::Vector{Mat3Map},
                            mode::SourceCouplingMode,
                            config::SourceCouplingConfig)
        -> (Vec3Map, Mat3Map)

Apply source coupling to navigation innovation and covariance.

# Arguments
- `innovation`: Raw innovation vector z - h(x) [T]
- `S`: Innovation covariance H P H' + R [T²]
- `source_fields`: Predicted field from each active source [T]
- `source_covariances`: Field covariance from each source (H_src P_src H_src') [T²]
- `mode`: Coupling mode to apply
- `config`: Coupling configuration

# Returns
- Modified (innovation, S) tuple

# Modes
- SHADOW: Returns inputs unchanged
- COV_ONLY: S += Σ source_covariances + inflation_diag; innovation unchanged
- SUBTRACT: innovation -= Σ clamp(source_fields); S += Σ source_covariances

# Safety
- SUBTRACT mode clamps each source field component to ±max_subtraction_field_T
- cross_covariance_enabled=false is enforced (no nav-source cross terms)
"""
function apply_source_coupling!(innovation::Vec3Map, S::Mat3Map,
                                 source_fields::Vector{Vec3Map},
                                 source_covariances::Vector{Mat3Map},
                                 mode::SourceCouplingMode,
                                 config::SourceCouplingConfig)
    # Enforce cross_covariance_enabled
    if !config.cross_covariance_enabled
        # No cross-covariance terms — this is enforced by design
    end

    if mode == SOURCE_SHADOW
        return (innovation, S)
    end

    # Accumulate source covariance contributions
    S_new = Matrix{Float64}(S)
    for Σ_src in source_covariances
        S_new += Matrix{Float64}(Σ_src)
    end

    if mode == SOURCE_COV_ONLY
        # Add diagonal inflation
        S_new += config.nav_covariance_inflation_T2 * I(3)
        return (innovation, Mat3Map(S_new))
    end

    # SOURCE_SUBTRACT: modify innovation with bounded subtraction
    innovation_new = Vector{Float64}(innovation)
    max_T = config.max_subtraction_field_T
    for B_src in source_fields
        for i in 1:3
            clamped = clamp(B_src[i], -max_T, max_T)
            innovation_new[i] -= clamped
        end
    end

    return (Vec3Map(innovation_new...), Mat3Map(S_new))
end

# ============================================================================
# Exports
# ============================================================================

export SourceCouplingMode, SOURCE_SHADOW, SOURCE_COV_ONLY, SOURCE_SUBTRACT
export SourceCouplingGates, all_gates_pass, n_gates_passing
export SourceGateThresholds, DEFAULT_SOURCE_GATE_THRESHOLDS
export SourceCouplingConfig, DEFAULT_SOURCE_COUPLING_CONFIG
export evaluate_coupling_gates, effective_coupling_mode
export apply_source_coupling!
