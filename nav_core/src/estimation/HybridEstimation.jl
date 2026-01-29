# =============================================================================
# HybridEstimation.jl - Source-Aware Hybrid Magnetic Update
# =============================================================================
#
# Purpose: Provide switch logic for when to use source contributions vs
# background-only estimation. Ensures source-aware updates don't regress
# compared to background-only baseline.
#
# Key Design Principles:
# 1. Sources break linear background assumption (1/r³ vs polynomial)
# 2. Source contributions should help, not hurt
# 3. If source model is poor, fall back to background-only
# 4. Provide clear decision audit trail
#
# Physical Justification:
# - Dipole field: B ~ μ₀m/(4πr³) with 1/r³ decay
# - Harmonic basis: polynomial in position, cannot represent 1/r³
# - Solution: subtract predicted source field, update with residual
# - Risk: inaccurate source model corrupts residual
#
# =============================================================================

module HybridEstimation

using LinearAlgebra
using StaticArrays
using Statistics

# =============================================================================
# Types
# =============================================================================

"""
Decision mode for hybrid estimation.

- BACKGROUND_ONLY: Use tile/background field only (source blind)
- SOURCE_AWARE: Subtract source contributions, use residual
- SOURCE_DOMINANT: Near a strong source, use source model primarily
- DEGRADED: High uncertainty, use conservative update
"""
@enum HybridMode begin
    BACKGROUND_ONLY = 0
    SOURCE_AWARE = 1
    SOURCE_DOMINANT = 2
    DEGRADED = 3
end

"""
Configuration for hybrid estimation decision logic.

# Threshold Justifications

## source_contribution_threshold = 0.3 (30%)
If sources contribute >30% of predicted field magnitude, the linearization
error from source model uncertainty becomes significant. Derived from
typical source position uncertainty of ~2m and gradient at 5m range.

## source_confidence_threshold = 0.8
Require 80% confidence in source model before subtracting. Based on
typical promotion gate requiring spatial compactness and temporal persistence.

## proximity_switch_distance_m = 10.0
Below 10m from a source, field gradients exceed 1/r⁴ scaling (dipole gradient),
making the background model inadequate. Switch to source-dominant mode.

## max_source_subtraction_nT = 500e-9 T (500 nT)
Cap source subtraction to prevent overcorrection from model errors.
Derivation: A strong ferromagnetic object (m ≈ 1000 A·m²) at minimum
operational range r = 5m produces B ≈ μ₀m/(4πr³) = 1e-7 × 1000/125
= 800 nT. The cap at 500 nT accommodates most encounters while
protecting against model errors at close range where linearization
degrades (position uncertainty σ_pos ~ 2m is ~40% of range).

## residual_chi2_threshold = 11.345
χ²(3, p=0.01) threshold. If residual after source subtraction exceeds this,
the source model is inconsistent and we fall back to background-only.

## fallback_inflation_factor = 3.0
When falling back to background-only due to source model failure, inflate
uncertainty by 3× to account for unmodeled source contributions.
"""
struct HybridEstimationConfig
    source_contribution_threshold::Float64
    source_confidence_threshold::Float64
    proximity_switch_distance_m::Float64
    max_source_subtraction_nT::Float64
    residual_chi2_threshold::Float64
    fallback_inflation_factor::Float64

    function HybridEstimationConfig(;
        source_contribution_threshold::Float64 = 0.3,
        source_confidence_threshold::Float64 = 0.8,
        proximity_switch_distance_m::Float64 = 10.0,
        max_source_subtraction_nT::Float64 = 500e-9,  # 500 nT in Tesla
        residual_chi2_threshold::Float64 = 11.345,    # χ²(3, 0.01)
        fallback_inflation_factor::Float64 = 3.0
    )
        new(
            source_contribution_threshold,
            source_confidence_threshold,
            proximity_switch_distance_m,
            max_source_subtraction_nT,
            residual_chi2_threshold,
            fallback_inflation_factor
        )
    end
end

"""
Decision result from hybrid estimation mode selection.

Provides full audit trail for debugging and analysis.
"""
struct HybridDecision
    mode::HybridMode
    reason::String
    source_contribution_ratio::Float64
    min_source_distance_m::Float64
    source_confidence::Float64
    source_subtraction::SVector{3,Float64}
    uncertainty_inflation::Float64
end

"""
Result of hybrid magnetic update.

Contains the cleaned measurement and decision audit.
"""
struct HybridUpdateResult
    z_effective::SVector{3,Float64}      # Measurement to use for update
    R_effective::SMatrix{3,3,Float64,9}  # Measurement noise to use
    decision::HybridDecision
    B_background::SVector{3,Float64}     # Background prediction
    B_sources::SVector{3,Float64}        # Source prediction (may be zero)
end

"""
Source contribution for hybrid estimation.

Represents a single source's contribution to the magnetic field.
"""
struct SourceContribution
    source_id::Int
    position::SVector{3,Float64}
    moment::SVector{3,Float64}
    B_contribution::SVector{3,Float64}
    Sigma_contribution::SMatrix{3,3,Float64,9}
    confidence::Float64
end

# =============================================================================
# Mode Selection Logic
# =============================================================================

"""
    select_hybrid_mode(
        measurement_position::AbstractVector,
        B_measured::AbstractVector,
        B_background::AbstractVector,
        Sigma_background::AbstractMatrix,
        source_contributions::Vector{SourceContribution},
        config::HybridEstimationConfig
    ) -> HybridDecision

Select the appropriate hybrid estimation mode based on:
1. Source proximity
2. Source contribution magnitude
3. Source model confidence
4. Measurement-prediction consistency

Returns a HybridDecision with mode and full audit trail.
"""
function select_hybrid_mode(
    measurement_position::AbstractVector,
    B_measured::AbstractVector,
    B_background::AbstractVector,
    Sigma_background::AbstractMatrix,
    source_contributions::Vector{SourceContribution},
    config::HybridEstimationConfig
)
    pos = SVector{3,Float64}(measurement_position)
    z = SVector{3,Float64}(B_measured)
    B_bg = SVector{3,Float64}(B_background)

    # No sources: use background only
    if isempty(source_contributions)
        return HybridDecision(
            BACKGROUND_ONLY,
            "No tracked sources",
            0.0, Inf, 0.0,
            SVector{3,Float64}(0.0, 0.0, 0.0),
            1.0
        )
    end

    # Compute aggregate source contribution
    B_sources = SVector{3,Float64}(0.0, 0.0, 0.0)
    min_distance = Inf
    total_confidence = 0.0
    n_sources = length(source_contributions)

    for src in source_contributions
        B_sources = B_sources + src.B_contribution
        d = norm(pos - src.position)
        min_distance = min(min_distance, d)
        total_confidence += src.confidence
    end
    avg_confidence = total_confidence / n_sources

    # Compute contribution ratio
    B_total_pred = B_bg + B_sources
    source_ratio = norm(B_sources) / max(norm(B_total_pred), 1e-12)

    # Check for source-dominant mode (very close to source)
    if min_distance < config.proximity_switch_distance_m
        if avg_confidence >= config.source_confidence_threshold
            return HybridDecision(
                SOURCE_DOMINANT,
                "Close proximity to high-confidence source (d=$(round(min_distance, digits=1))m)",
                source_ratio,
                min_distance,
                avg_confidence,
                B_sources,
                1.0
            )
        else
            return HybridDecision(
                DEGRADED,
                "Close to low-confidence source - inflating uncertainty",
                source_ratio,
                min_distance,
                avg_confidence,
                SVector{3,Float64}(0.0, 0.0, 0.0),  # Don't subtract uncertain source
                config.fallback_inflation_factor
            )
        end
    end

    # Check if source contribution is significant
    if source_ratio > config.source_contribution_threshold
        if avg_confidence >= config.source_confidence_threshold
            # Limit subtraction magnitude
            subtraction = B_sources
            sub_magnitude = norm(subtraction)
            if sub_magnitude > config.max_source_subtraction_nT
                subtraction = subtraction * (config.max_source_subtraction_nT / sub_magnitude)
            end

            # Verify consistency after subtraction
            residual = z - B_bg - subtraction
            chi2 = residual' * (Sigma_background \ residual)

            if chi2 < config.residual_chi2_threshold
                return HybridDecision(
                    SOURCE_AWARE,
                    "Source-aware update (contribution=$(round(source_ratio*100, digits=0))%)",
                    source_ratio,
                    min_distance,
                    avg_confidence,
                    subtraction,
                    1.0
                )
            else
                return HybridDecision(
                    DEGRADED,
                    "Source subtraction inconsistent (χ²=$(round(chi2, digits=1)) > threshold)",
                    source_ratio,
                    min_distance,
                    avg_confidence,
                    SVector{3,Float64}(0.0, 0.0, 0.0),
                    config.fallback_inflation_factor
                )
            end
        else
            return HybridDecision(
                DEGRADED,
                "Large source contribution with low confidence - inflating uncertainty",
                source_ratio,
                min_distance,
                avg_confidence,
                SVector{3,Float64}(0.0, 0.0, 0.0),
                config.fallback_inflation_factor
            )
        end
    end

    # Low source contribution: background-only is fine
    return HybridDecision(
        BACKGROUND_ONLY,
        "Source contribution negligible ($(round(source_ratio*100, digits=0))%)",
        source_ratio,
        min_distance,
        avg_confidence,
        SVector{3,Float64}(0.0, 0.0, 0.0),
        1.0
    )
end

# =============================================================================
# Hybrid Update Computation
# =============================================================================

"""
    compute_hybrid_update(
        B_measured::AbstractVector,
        R_sensor::AbstractMatrix,
        B_background::AbstractVector,
        Sigma_background::AbstractMatrix,
        decision::HybridDecision
    ) -> HybridUpdateResult

Compute the effective measurement and noise for EKF update based on
the hybrid decision.

The effective measurement is:
- BACKGROUND_ONLY: z_eff = z - B_background
- SOURCE_AWARE: z_eff = z - B_background - B_sources
- SOURCE_DOMINANT: z_eff = z - B_sources (use source model primarily)
- DEGRADED: z_eff = z - B_background with inflated R

The effective noise includes the decision's uncertainty inflation.
"""
function compute_hybrid_update(
    B_measured::AbstractVector,
    R_sensor::AbstractMatrix,
    B_background::AbstractVector,
    Sigma_background::AbstractMatrix,
    decision::HybridDecision
)
    z = SVector{3,Float64}(B_measured)
    R = SMatrix{3,3,Float64,9}(R_sensor)
    B_bg = SVector{3,Float64}(B_background)
    Sigma_bg = SMatrix{3,3,Float64,9}(Sigma_background)

    # Compute effective measurement based on mode
    if decision.mode == SOURCE_DOMINANT
        # Use source model as primary prediction
        z_eff = z - decision.source_subtraction
        B_src = decision.source_subtraction
    elseif decision.mode == SOURCE_AWARE
        # Subtract both background and source
        z_eff = z - B_bg - decision.source_subtraction
        B_src = decision.source_subtraction
    else
        # BACKGROUND_ONLY or DEGRADED: only subtract background
        z_eff = z - B_bg
        B_src = SVector{3,Float64}(0.0, 0.0, 0.0)
    end

    # Compute effective noise with inflation
    inflation = decision.uncertainty_inflation
    R_eff = R * inflation^2 + Sigma_bg * inflation^2

    return HybridUpdateResult(
        z_eff,
        R_eff,
        decision,
        B_bg,
        B_src
    )
end

"""
    full_hybrid_update(
        measurement_position::AbstractVector,
        B_measured::AbstractVector,
        R_sensor::AbstractMatrix,
        B_background::AbstractVector,
        Sigma_background::AbstractMatrix,
        source_contributions::Vector{SourceContribution},
        config::HybridEstimationConfig
    ) -> HybridUpdateResult

Complete hybrid update: select mode and compute effective measurement/noise.

This is the main entry point for hybrid estimation.
"""
function full_hybrid_update(
    measurement_position::AbstractVector,
    B_measured::AbstractVector,
    R_sensor::AbstractMatrix,
    B_background::AbstractVector,
    Sigma_background::AbstractMatrix,
    source_contributions::Vector{SourceContribution},
    config::HybridEstimationConfig
)
    decision = select_hybrid_mode(
        measurement_position,
        B_measured,
        B_background,
        Sigma_background,
        source_contributions,
        config
    )

    return compute_hybrid_update(
        B_measured,
        R_sensor,
        B_background,
        Sigma_background,
        decision
    )
end

# =============================================================================
# Comparison Utilities
# =============================================================================

"""
    compare_modes(
        measurement_position::AbstractVector,
        B_measured::AbstractVector,
        R_sensor::AbstractMatrix,
        B_background::AbstractVector,
        Sigma_background::AbstractMatrix,
        source_contributions::Vector{SourceContribution},
        config::HybridEstimationConfig
    ) -> NamedTuple

Compare source-aware vs background-only results for analysis.

Returns residuals and innovation statistics for both modes.
"""
function compare_modes(
    measurement_position::AbstractVector,
    B_measured::AbstractVector,
    R_sensor::AbstractMatrix,
    B_background::AbstractVector,
    Sigma_background::AbstractMatrix,
    source_contributions::Vector{SourceContribution},
    config::HybridEstimationConfig
)
    z = SVector{3,Float64}(B_measured)
    B_bg = SVector{3,Float64}(B_background)
    R = SMatrix{3,3,Float64,9}(R_sensor)
    Sigma_bg = SMatrix{3,3,Float64,9}(Sigma_background)

    # Background-only residual
    residual_bg_only = z - B_bg
    S_bg_only = R + Sigma_bg
    nis_bg_only = residual_bg_only' * (S_bg_only \ residual_bg_only)

    # Source-aware residual
    B_sources = SVector{3,Float64}(0.0, 0.0, 0.0)
    for src in source_contributions
        B_sources = B_sources + src.B_contribution
    end
    residual_src_aware = z - B_bg - B_sources
    nis_src_aware = residual_src_aware' * (S_bg_only \ residual_src_aware)

    return (
        residual_bg_only = residual_bg_only,
        residual_src_aware = residual_src_aware,
        nis_bg_only = nis_bg_only,
        nis_src_aware = nis_src_aware,
        source_contribution = B_sources,
        source_improved = nis_src_aware < nis_bg_only
    )
end

"""
    validate_no_regression(
        results_src_aware::Vector{HybridUpdateResult},
        results_bg_only::Vector{HybridUpdateResult}
    ) -> Bool

Validate that source-aware estimation doesn't regress vs background-only.

# Criterion
Source-aware is acceptable if:
- Mean NIS is not significantly worse (within 10%)
- P90 NIS is not significantly worse (within 20%)
- No individual NIS exceeds 2× background-only value

These thresholds ensure source subtraction helps overall without
creating pathological outliers.
"""
function validate_no_regression(
    nis_src_aware::Vector{Float64},
    nis_bg_only::Vector{Float64};
    mean_tolerance::Float64 = 0.1,
    p90_tolerance::Float64 = 0.2,
    individual_tolerance::Float64 = 2.0
)
    if isempty(nis_src_aware) || isempty(nis_bg_only)
        return true
    end

    # Mean check
    mean_src = mean(nis_src_aware)
    mean_bg = mean(nis_bg_only)
    if mean_src > mean_bg * (1 + mean_tolerance)
        return false
    end

    # P90 check
    p90_src = quantile(nis_src_aware, 0.9)
    p90_bg = quantile(nis_bg_only, 0.9)
    if p90_src > p90_bg * (1 + p90_tolerance)
        return false
    end

    # Individual check
    for (nis_s, nis_b) in zip(nis_src_aware, nis_bg_only)
        if nis_s > nis_b * individual_tolerance
            return false
        end
    end

    return true
end

# =============================================================================
# Dipole Field Helper (Physics)
# =============================================================================

"""
    dipole_field(measurement_pos::AbstractVector, source_pos::AbstractVector, moment::AbstractVector) -> SVector{3,Float64}

Compute dipole magnetic field at measurement position.

# Physics
B(r) = (μ₀/4π) × [3(m·r̂)r̂ - m] / |r|³

where:
- μ₀ = 4π × 10⁻⁷ T·m/A (permeability of free space)
- m = magnetic moment vector [A·m²]
- r = measurement_pos - source_pos
- r̂ = r / |r|

# Units
- Input positions: meters [m]
- Input moment: A·m² (but typically pre-scaled to give Tesla output)
- Output: Tesla [T]
"""
function dipole_field(
    measurement_pos::AbstractVector,
    source_pos::AbstractVector,
    moment::AbstractVector
)
    r = SVector{3,Float64}(measurement_pos) - SVector{3,Float64}(source_pos)
    m = SVector{3,Float64}(moment)

    r_mag = norm(r)
    if r_mag < 0.1  # Avoid singularity at source location
        return SVector{3,Float64}(0.0, 0.0, 0.0)
    end

    r_hat = r / r_mag

    # μ₀/(4π) = 1e-7 T·m/A
    mu0_over_4pi = 1e-7

    m_dot_r = dot(m, r_hat)

    B = mu0_over_4pi * (3.0 * m_dot_r * r_hat - m) / r_mag^3

    return B
end

"""
    create_source_contribution(
        measurement_pos::AbstractVector,
        source_id::Int,
        source_pos::AbstractVector,
        source_moment::AbstractVector,
        source_pos_covariance::AbstractMatrix,
        source_moment_covariance::AbstractMatrix,
        confidence::Float64
    ) -> SourceContribution

Create a SourceContribution from source parameters.

Computes field contribution and propagates uncertainty from
position and moment covariances.
"""
function create_source_contribution(
    measurement_pos::AbstractVector,
    source_id::Int,
    source_pos::AbstractVector,
    source_moment::AbstractVector,
    source_pos_covariance::AbstractMatrix,
    source_moment_covariance::AbstractMatrix,
    confidence::Float64
)
    pos = SVector{3,Float64}(measurement_pos)
    src_pos = SVector{3,Float64}(source_pos)
    moment = SVector{3,Float64}(source_moment)

    # Compute field
    B = dipole_field(pos, src_pos, moment)

    # Compute Jacobians for uncertainty propagation
    # Use finite differences for simplicity
    eps = 1e-6

    # Jacobian w.r.t. source position (3×3)
    J_pos = SMatrix{3,3,Float64,9}(zeros(3,3))
    for i in 1:3
        delta = SVector{3,Float64}(i==1 ? eps : 0.0, i==2 ? eps : 0.0, i==3 ? eps : 0.0)
        B_plus = dipole_field(pos, src_pos + delta, moment)
        B_minus = dipole_field(pos, src_pos - delta, moment)
        J_pos = setindex(J_pos, (B_plus - B_minus) / (2*eps), :, i)
    end

    # Jacobian w.r.t. moment (3×3)
    J_mom = SMatrix{3,3,Float64,9}(zeros(3,3))
    for i in 1:3
        delta = SVector{3,Float64}(i==1 ? eps : 0.0, i==2 ? eps : 0.0, i==3 ? eps : 0.0)
        B_plus = dipole_field(pos, src_pos, moment + delta)
        B_minus = dipole_field(pos, src_pos, moment - delta)
        J_mom = setindex(J_mom, (B_plus - B_minus) / (2*eps), :, i)
    end

    # Propagate uncertainty
    P_pos = SMatrix{3,3,Float64,9}(source_pos_covariance)
    P_mom = SMatrix{3,3,Float64,9}(source_moment_covariance)

    Sigma = J_pos * P_pos * J_pos' + J_mom * P_mom * J_mom'

    return SourceContribution(
        source_id,
        src_pos,
        moment,
        B,
        Sigma,
        confidence
    )
end

# Helper to set matrix column (workaround for SMatrix immutability)
function setindex(M::SMatrix{3,3,Float64,9}, col::SVector{3,Float64}, ::Colon, j::Int)
    data = collect(M)
    data[:, j] = col
    return SMatrix{3,3,Float64,9}(data)
end

# =============================================================================
# Exports
# =============================================================================

export HybridMode, BACKGROUND_ONLY, SOURCE_AWARE, SOURCE_DOMINANT, DEGRADED
export HybridEstimationConfig
export HybridDecision, HybridUpdateResult, SourceContribution
export select_hybrid_mode, compute_hybrid_update, full_hybrid_update
export compare_modes, validate_no_regression
export dipole_field, create_source_contribution

end # module
