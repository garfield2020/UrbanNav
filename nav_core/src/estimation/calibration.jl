# ============================================================================
# NEES Calibration Infrastructure
# ============================================================================
#
# This module provides infrastructure for calibrating the innovation covariance
# to achieve proper NEES (Normalized Estimation Error Squared) statistics.
#
# The key insight is that Q_model (truth-model mismatch) must be decomposed
# into distinct sources, each with its own physical interpretation and
# calibration strategy.
#
# Q_model = Q_map + Q_unmodeled + Q_temporal + Q_calibration
#
# Each component has different:
# - Physical origin
# - Spatial correlation structure
# - Temporal behavior
# - Calibration approach
#
# ============================================================================

using LinearAlgebra
using Statistics

export ModelMismatchConfig, DEFAULT_MODEL_MISMATCH_CONFIG
export compute_Q_model_decomposed, compute_total_Q_model
export CalibrationState, CalibrationResult
export update_calibration!, calibrate_from_residuals
export CalibrationStats, compute_residual_statistics
export nees_from_residuals, nis_from_innovations
export CalibrationMonitor, check_calibration_status

# ============================================================================
# Model Mismatch Configuration
# ============================================================================

"""
    ModelMismatchConfig

Decomposed model mismatch configuration.

Q_model = Q_map + Q_unmodeled + Q_temporal + Q_calibration

# Components

## Q_map: Map discretization errors
- Source: Finite resolution of magnetic map tiles
- Spatial: Correlated over tile size
- Temporal: Static (doesn't change during mission)
- Calibration: From map residuals during mapping phase

## Q_unmodeled: Unmodeled field sources ("clutter")
- Source: Dipoles, geological features not in map
- Spatial: Localized near unknown sources
- Temporal: Static
- Calibration: From innovation statistics in new areas

## Q_temporal: Time-varying field components
- Source: Tidal effects, diurnal variations, space weather
- Spatial: Large-scale (affects entire region)
- Temporal: Varies over hours to days
- Calibration: From temporal residual drift

## Q_calibration: Sensor calibration drift
- Source: Temperature effects, aging, magnetic interference
- Spatial: Constant (affects all measurements equally)
- Temporal: Slow drift (minutes to hours)
- Calibration: From bias estimation in Kalman filter

# Fields
- `σ_map_B::Float64`: Map uncertainty for field (T)
- `σ_map_G::Float64`: Map uncertainty for gradient (T/m)
- `σ_unmodeled_B::Float64`: Unmodeled source uncertainty for field (T)
- `σ_unmodeled_G::Float64`: Unmodeled source uncertainty for gradient (T/m)
- `σ_temporal_B::Float64`: Temporal variation uncertainty for field (T)
- `σ_temporal_G::Float64`: Temporal variation uncertainty for gradient (T/m)
- `σ_calibration_B::Float64`: Calibration drift uncertainty for field (T)
- `σ_calibration_G::Float64`: Calibration drift uncertainty for gradient (T/m)
"""
struct ModelMismatchConfig
    # Map discretization
    σ_map_B::Float64
    σ_map_G::Float64

    # Unmodeled sources (clutter)
    σ_unmodeled_B::Float64
    σ_unmodeled_G::Float64

    # Temporal variations
    σ_temporal_B::Float64
    σ_temporal_G::Float64

    # Calibration drift
    σ_calibration_B::Float64
    σ_calibration_G::Float64
end

function ModelMismatchConfig(;
    # Map discretization: ~0.5 nT from interpolation
    σ_map_B::Float64 = 0.5e-9,
    σ_map_G::Float64 = 0.5e-9,

    # Unmodeled sources: ~1-2 nT from unknown dipoles
    σ_unmodeled_B::Float64 = 1.5e-9,
    σ_unmodeled_G::Float64 = 1.5e-9,

    # Temporal variations: ~0.5 nT from tidal/diurnal
    σ_temporal_B::Float64 = 0.5e-9,
    σ_temporal_G::Float64 = 0.2e-9,

    # Calibration drift: ~0.3 nT from sensor drift
    σ_calibration_B::Float64 = 0.3e-9,
    σ_calibration_G::Float64 = 0.1e-9
)
    ModelMismatchConfig(
        σ_map_B, σ_map_G,
        σ_unmodeled_B, σ_unmodeled_G,
        σ_temporal_B, σ_temporal_G,
        σ_calibration_B, σ_calibration_G
    )
end

const DEFAULT_MODEL_MISMATCH_CONFIG = ModelMismatchConfig()

"""
    compute_Q_model_decomposed(d::Int, config::ModelMismatchConfig)

Compute decomposed Q_model matrices for each uncertainty source.

Returns named tuple: (Q_map, Q_unmodeled, Q_temporal, Q_calibration, Q_total)
"""
function compute_Q_model_decomposed(d::Int, config::ModelMismatchConfig)
    if d == 3
        Q_map = Diagonal(fill(config.σ_map_B^2, 3))
        Q_unmodeled = Diagonal(fill(config.σ_unmodeled_B^2, 3))
        Q_temporal = Diagonal(fill(config.σ_temporal_B^2, 3))
        Q_calibration = Diagonal(fill(config.σ_calibration_B^2, 3))
    elseif d == 8
        Q_map = Diagonal(vcat(
            fill(config.σ_map_B^2, 3),
            fill(config.σ_map_G^2, 5)
        ))
        Q_unmodeled = Diagonal(vcat(
            fill(config.σ_unmodeled_B^2, 3),
            fill(config.σ_unmodeled_G^2, 5)
        ))
        Q_temporal = Diagonal(vcat(
            fill(config.σ_temporal_B^2, 3),
            fill(config.σ_temporal_G^2, 5)
        ))
        Q_calibration = Diagonal(vcat(
            fill(config.σ_calibration_B^2, 3),
            fill(config.σ_calibration_G^2, 5)
        ))
    else
        error("Unsupported dimension d=$d")
    end

    Q_total = Q_map + Q_unmodeled + Q_temporal + Q_calibration

    return (
        Q_map = Q_map,
        Q_unmodeled = Q_unmodeled,
        Q_temporal = Q_temporal,
        Q_calibration = Q_calibration,
        Q_total = Q_total
    )
end

"""
    compute_total_Q_model(d::Int, config::ModelMismatchConfig) -> Diagonal

Compute total Q_model (sum of all components).
"""
function compute_total_Q_model(d::Int, config::ModelMismatchConfig)
    components = compute_Q_model_decomposed(d, config)
    return components.Q_total
end

"""
    total_σ_B(config::ModelMismatchConfig) -> Float64

Compute total field uncertainty (RSS of all components).
"""
function total_σ_B(config::ModelMismatchConfig)
    return sqrt(
        config.σ_map_B^2 +
        config.σ_unmodeled_B^2 +
        config.σ_temporal_B^2 +
        config.σ_calibration_B^2
    )
end

"""
    total_σ_G(config::ModelMismatchConfig) -> Float64

Compute total gradient uncertainty (RSS of all components).
"""
function total_σ_G(config::ModelMismatchConfig)
    return sqrt(
        config.σ_map_G^2 +
        config.σ_unmodeled_G^2 +
        config.σ_temporal_G^2 +
        config.σ_calibration_G^2
    )
end

export total_σ_B, total_σ_G

# ============================================================================
# Residual Statistics
# ============================================================================

"""
    CalibrationStats

Statistics computed from measurement residuals for calibration.
"""
struct CalibrationStats
    n_samples::Int              # Number of residual samples
    mean_residual::Vector{Float64}      # Mean residual per component
    std_residual::Vector{Float64}       # Std of residuals per component
    mean_chi2::Float64          # Mean χ² statistic
    std_chi2::Float64           # Std of χ² statistic
    nees::Float64               # Normalized estimation error squared
    nis::Float64                # Normalized innovation squared
    acceptance_rate::Float64    # Fraction of accepted measurements
end

"""
    compute_residual_statistics(residuals, covariances) -> CalibrationStats

Compute statistics from residual sequence for calibration.

# Arguments
- `residuals`: Vector of residual vectors
- `covariances`: Vector of innovation covariance matrices
"""
function compute_residual_statistics(
    residuals::Vector{<:AbstractVector},
    covariances::Vector{<:AbstractMatrix}
)
    n = length(residuals)
    if n == 0
        d = 3
        return CalibrationStats(0, zeros(d), zeros(d), 0.0, 0.0, 0.0, 0.0, 0.0)
    end

    d = length(residuals[1])

    # Per-component statistics
    residual_matrix = hcat(residuals...)  # d × n
    mean_r = vec(mean(residual_matrix, dims=2))
    std_r = vec(std(residual_matrix, dims=2))

    # χ² statistics
    chi2_values = Float64[]
    for (r, S) in zip(residuals, covariances)
        try
            χ2 = r' * (S \ r)
            push!(chi2_values, χ2)
        catch
            # Skip if covariance is singular
        end
    end

    mean_chi2 = isempty(chi2_values) ? 0.0 : mean(chi2_values)
    std_chi2 = isempty(chi2_values) ? 0.0 : std(chi2_values)

    # NEES: should be ~d for well-calibrated filter
    nees = mean_chi2 / d

    # NIS (same as NEES for innovations)
    nis = nees

    # Acceptance rate (χ² < threshold)
    threshold = d + 3 * sqrt(2 * d)  # ~99% threshold
    acceptance_rate = count(χ2 -> χ2 < threshold, chi2_values) / max(length(chi2_values), 1)

    return CalibrationStats(n, mean_r, std_r, mean_chi2, std_chi2, nees, nis, acceptance_rate)
end

"""
    nees_from_residuals(residuals, covariances, d) -> Float64

Compute NEES (Normalized Estimation Error Squared) from residuals.

For a well-calibrated filter, NEES ≈ 1.0.
- NEES > 1: Filter is overconfident (covariance too small)
- NEES < 1: Filter is underconfident (covariance too large)
"""
function nees_from_residuals(
    residuals::Vector{<:AbstractVector},
    covariances::Vector{<:AbstractMatrix},
    d::Int
)
    stats = compute_residual_statistics(residuals, covariances)
    return stats.nees
end

"""
    nis_from_innovations(innovations, S_matrices, d) -> Float64

Compute NIS (Normalized Innovation Squared) from innovations.
Same as NEES but specifically for measurement innovations.
"""
function nis_from_innovations(
    innovations::Vector{<:AbstractVector},
    S_matrices::Vector{<:AbstractMatrix},
    d::Int
)
    return nees_from_residuals(innovations, S_matrices, d)
end

# ============================================================================
# Calibration State Machine
# ============================================================================

"""
    CalibrationState

Current state of the calibration process.
"""
@enum CalibrationState begin
    CALIBRATION_INIT = 1        # Initial state, collecting data
    CALIBRATION_LEARNING = 2    # Actively learning parameters
    CALIBRATION_CONVERGED = 3   # Parameters have converged
    CALIBRATION_FROZEN = 4      # Parameters frozen (mission mode)
end

"""
    CalibrationResult

Result of a calibration update.
"""
struct CalibrationResult
    state::CalibrationState
    config::ModelMismatchConfig
    nees_before::Float64
    nees_after::Float64
    samples_used::Int
    converged::Bool
    message::String
end

"""
    CalibrationMonitor

Monitors calibration status and triggers updates.

# Fields
- `config::ModelMismatchConfig`: Current model mismatch configuration
- `state::CalibrationState`: Current calibration state
- `residual_buffer::Vector`: Recent residuals for statistics
- `covariance_buffer::Vector`: Corresponding covariances
- `buffer_size::Int`: Maximum buffer size
- `update_interval::Int`: Samples between calibration updates
- `nees_target::Float64`: Target NEES value (should be 1.0)
- `nees_tolerance::Float64`: Acceptable deviation from target
"""
mutable struct CalibrationMonitor
    config::ModelMismatchConfig
    state::CalibrationState
    residual_buffer::Vector{Vector{Float64}}
    covariance_buffer::Vector{Matrix{Float64}}
    buffer_size::Int
    update_interval::Int
    nees_target::Float64
    nees_tolerance::Float64
    samples_since_update::Int
    last_nees::Float64
end

function CalibrationMonitor(;
    config::ModelMismatchConfig = DEFAULT_MODEL_MISMATCH_CONFIG,
    buffer_size::Int = 500,
    update_interval::Int = 100,
    nees_target::Float64 = 1.0,
    nees_tolerance::Float64 = 0.15
)
    CalibrationMonitor(
        config,
        CALIBRATION_INIT,
        Vector{Float64}[],
        Matrix{Float64}[],
        buffer_size,
        update_interval,
        nees_target,
        nees_tolerance,
        0,
        0.0
    )
end

"""
    add_sample!(monitor, residual, covariance)

Add a residual sample to the calibration buffer.
"""
function add_sample!(monitor::CalibrationMonitor, residual::AbstractVector, covariance::AbstractMatrix)
    push!(monitor.residual_buffer, Vector{Float64}(residual))
    push!(monitor.covariance_buffer, Matrix{Float64}(covariance))

    # Trim buffer if too large
    while length(monitor.residual_buffer) > monitor.buffer_size
        popfirst!(monitor.residual_buffer)
        popfirst!(monitor.covariance_buffer)
    end

    monitor.samples_since_update += 1
end

"""
    check_calibration_status(monitor) -> (needs_update::Bool, current_nees::Float64)

Check if calibration update is needed.
"""
function check_calibration_status(monitor::CalibrationMonitor)
    if length(monitor.residual_buffer) < 50
        return (false, 0.0)
    end

    d = length(monitor.residual_buffer[1])
    current_nees = nees_from_residuals(monitor.residual_buffer, monitor.covariance_buffer, d)
    monitor.last_nees = current_nees

    needs_update = monitor.samples_since_update >= monitor.update_interval &&
                   abs(current_nees - monitor.nees_target) > monitor.nees_tolerance

    return (needs_update, current_nees)
end

"""
    update_calibration!(monitor) -> CalibrationResult

Update model mismatch configuration based on residual statistics.

Uses a simple scaling approach:
- If NEES > 1: increase Q_model (filter overconfident)
- If NEES < 1: decrease Q_model (filter underconfident)
"""
function update_calibration!(monitor::CalibrationMonitor)
    if length(monitor.residual_buffer) < 50
        return CalibrationResult(
            monitor.state, monitor.config, 0.0, 0.0, 0,
            false, "Insufficient samples for calibration"
        )
    end

    d = length(monitor.residual_buffer[1])
    nees_before = nees_from_residuals(monitor.residual_buffer, monitor.covariance_buffer, d)

    # Compute scaling factor
    # If NEES = 1.5, we need to increase Q_model by factor of ~1.5
    # If NEES = 0.7, we need to decrease Q_model by factor of ~0.7
    scale_factor = sqrt(nees_before)  # sqrt because we scale σ, not σ²

    # Limit scaling to prevent instability
    scale_factor = clamp(scale_factor, 0.5, 2.0)

    # Only scale the unmodeled component (most tunable)
    new_config = ModelMismatchConfig(
        σ_map_B = monitor.config.σ_map_B,
        σ_map_G = monitor.config.σ_map_G,
        σ_unmodeled_B = monitor.config.σ_unmodeled_B * scale_factor,
        σ_unmodeled_G = monitor.config.σ_unmodeled_G * scale_factor,
        σ_temporal_B = monitor.config.σ_temporal_B,
        σ_temporal_G = monitor.config.σ_temporal_G,
        σ_calibration_B = monitor.config.σ_calibration_B,
        σ_calibration_G = monitor.config.σ_calibration_G
    )

    monitor.config = new_config
    monitor.samples_since_update = 0

    # Check convergence
    converged = abs(nees_before - monitor.nees_target) < monitor.nees_tolerance

    if converged
        monitor.state = CALIBRATION_CONVERGED
    else
        monitor.state = CALIBRATION_LEARNING
    end

    # Estimate new NEES (will be approximately 1.0 after scaling)
    nees_after = nees_before / scale_factor^2

    return CalibrationResult(
        monitor.state,
        new_config,
        nees_before,
        nees_after,
        length(monitor.residual_buffer),
        converged,
        converged ? "Calibration converged" : "Calibration updated, scale=$(round(scale_factor, digits=3))"
    )
end

"""
    calibrate_from_residuals(residuals, covariances, d; kwargs...) -> ModelMismatchConfig

One-shot calibration from residual statistics.

Computes optimal Q_model to achieve NEES ≈ 1.0.
"""
function calibrate_from_residuals(
    residuals::Vector{<:AbstractVector},
    covariances::Vector{<:AbstractMatrix},
    d::Int;
    base_config::ModelMismatchConfig = DEFAULT_MODEL_MISMATCH_CONFIG,
    nees_target::Float64 = 1.0
)
    if length(residuals) < 20
        return base_config
    end

    current_nees = nees_from_residuals(residuals, covariances, d)

    if current_nees < 0.01
        # NEES too low - something wrong with data
        return base_config
    end

    # Scale factor to achieve target NEES
    scale_factor = sqrt(current_nees / nees_target)
    scale_factor = clamp(scale_factor, 0.1, 10.0)

    return ModelMismatchConfig(
        σ_map_B = base_config.σ_map_B,
        σ_map_G = base_config.σ_map_G,
        σ_unmodeled_B = base_config.σ_unmodeled_B * scale_factor,
        σ_unmodeled_G = base_config.σ_unmodeled_G * scale_factor,
        σ_temporal_B = base_config.σ_temporal_B,
        σ_temporal_G = base_config.σ_temporal_G,
        σ_calibration_B = base_config.σ_calibration_B,
        σ_calibration_G = base_config.σ_calibration_G
    )
end

export add_sample!, CalibrationState
export CALIBRATION_INIT, CALIBRATION_LEARNING, CALIBRATION_CONVERGED, CALIBRATION_FROZEN
