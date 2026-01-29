# ============================================================================
# Gradient Integration - d=8 Full Tensor Measurement Support
# ============================================================================
#
# Ported from AUV-Navigation/src/gradient_integration.jl
#
# Enable gradient measurements (B + ∇B) for faster observability.
#
# The gradient tensor is symmetric and traceless (Maxwell), so:
# - 9 components → 5 independent: [Gxx, Gyy, Gxy, Gxz, Gyz]
# - Gzz = -(Gxx + Gyy)
#
# Benefits:
# - Faster observability (~2 measurements vs ~50 for scalar)
# - Tighter drift envelope
# - Better feature localization
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Gradient Configuration
# ============================================================================

"""
    GradientConfig

Configuration for gradient-enabled measurements.
"""
struct GradientConfig
    enabled::Bool                   # Use gradients in residual
    noise_G::Float64                # Gradient noise std (T/m)
    noise_B::Float64                # Field noise std (T)
    weight_ratio::Float64           # Relative weight of gradient vs field
    d::Int                          # Measurement dimension (3 or 8)
end

function GradientConfig(;
    enabled::Bool = true,
    noise_G::Real = 5e-9,           # 5 nT/m gradient noise
    noise_B::Real = 5e-9,           # 5 nT field noise
    weight_ratio::Real = 1.0        # Equal weight by default
)
    d = enabled ? 8 : 3
    GradientConfig(enabled, Float64(noise_G), Float64(noise_B), Float64(weight_ratio), d)
end

const DEFAULT_GRADIENT_CONFIG = GradientConfig()
const GRADIENT_DISABLED_CONFIG = GradientConfig(enabled = false)

# ============================================================================
# χ² Thresholds for d=8
# ============================================================================

# Note: d=3 thresholds (CHI2_D3_MILD_THRESHOLD, CHI2_D3_STRONG_THRESHOLD)
# are defined in residual_manager.jl with values:
#   - CHI2_D3_MILD_THRESHOLD = 11.3449 (p=0.01, 99th percentile)
#   - CHI2_D3_STRONG_THRESHOLD = 16.2662 (p=0.001, 99.9th percentile)

# For d=8 (field + gradient)
"""χ²(8, 0.99) threshold for mild outlier (d=8). P(γ > 20.09) = 0.01"""
const CHI2_D8_MILD_THRESHOLD = 20.0902

"""χ²(8, 0.999) threshold for strong outlier (d=8). P(γ > 26.12) = 0.001"""
const CHI2_D8_STRONG_THRESHOLD = 26.1245

"""
    chi2_threshold_for_d(d, p)

Get appropriate χ² threshold for dimension d and tail probability p.

For d=3 and d=8, uses pre-computed values. For other dimensions,
uses Wilson-Hilferty approximation.
"""
function chi2_threshold_for_d(d::Int, p::Float64)
    if d == 8
        return p >= 0.999 ? CHI2_D8_STRONG_THRESHOLD : CHI2_D8_MILD_THRESHOLD
    else
        # General approximation for chi-squared quantile
        # Using Wilson-Hilferty transformation
        if p >= 0.999
            z = 3.09  # z_0.999
        elseif p >= 0.99
            z = 2.326  # z_0.99
        elseif p >= 0.95
            z = 1.645  # z_0.95
        else
            z = 1.0
        end
        return d * (1.0 - 2.0/(9*d) + z * sqrt(2.0/(9*d)))^3
    end
end

# ============================================================================
# Combined Measurement Vector
# ============================================================================

"""
    GradientMeasurement

Combined field + gradient measurement.
"""
struct GradientMeasurement
    B::SVector{3, Float64}         # Field [T]
    G::SVector{5, Float64}         # Gradient tensor (packed) [T/m]
    position::SVector{3, Float64}  # Position [m]
    timestamp::Float64
    gradient_valid::Bool           # Whether gradient is valid
end

function GradientMeasurement(B::AbstractVector, G::AbstractVector,
                              position::AbstractVector, timestamp::Float64;
                              gradient_valid::Bool = true)
    GradientMeasurement(
        SVector{3}(B...),
        SVector{5}(G...),
        SVector{3}(position...),
        timestamp,
        gradient_valid
    )
end

function GradientMeasurement(B::AbstractVector, G::AbstractMatrix,
                              position::AbstractVector, timestamp::Float64;
                              gradient_valid::Bool = true)
    G5 = pack_gradient_tensor(G)
    GradientMeasurement(
        SVector{3}(B...),
        G5,
        SVector{3}(position...),
        timestamp,
        gradient_valid
    )
end

"""Get full measurement vector (d=8 or d=3)."""
function gradient_measurement_vector(m::GradientMeasurement; use_gradient::Bool = true)
    if use_gradient && m.gradient_valid
        return vcat(m.B, m.G)
    else
        return m.B
    end
end

"""Get measurement covariance (d=8 or d=3)."""
function gradient_measurement_covariance(config::GradientConfig)
    if config.enabled
        return Diagonal(vcat(fill(config.noise_B^2, 3), fill(config.noise_G^2, 5)))
    else
        return Diagonal(fill(config.noise_B^2, 3))
    end
end

# ============================================================================
# Gradient-Enabled Residual Statistics
# ============================================================================

"""
    GradientResidualStatistics

Statistics for gradient-enabled measurements.
"""
struct GradientResidualStatistics
    position::SVector{3, Float64}
    residual_B::SVector{3, Float64}     # Field residual
    residual_G::SVector{5, Float64}     # Gradient residual
    chi2_B::Float64                      # Field-only χ²
    chi2_G::Float64                      # Gradient-only χ²
    chi2_combined::Float64               # Combined χ²
    timestamp::Float64
    decision::Symbol                     # :inlier, :mild_outlier, :strong_outlier
    dimension::Int                       # 3 or 8
end

"""
    compute_gradient_residual_statistics(B_meas, B_pred, G_meas, G_pred, position, timestamp; config)

Compute residual statistics for gradient-enabled measurements.
"""
function compute_gradient_residual_statistics(B_meas::AbstractVector, B_pred::AbstractVector,
                                               G_meas::AbstractVector, G_pred::AbstractVector,
                                               position::AbstractVector, timestamp::Float64;
                                               config::GradientConfig = DEFAULT_GRADIENT_CONFIG)
    # Field residual
    r_B = SVector{3}(B_meas...) - SVector{3}(B_pred...)
    χ²_B = dot(r_B, r_B) / config.noise_B^2

    if !config.enabled
        # Field-only mode
        decision = if χ²_B > CHI2_D3_STRONG_THRESHOLD
            :strong_outlier
        elseif χ²_B > CHI2_D3_MILD_THRESHOLD
            :mild_outlier
        else
            :inlier
        end

        return GradientResidualStatistics(
            SVector{3}(position...),
            r_B,
            zeros(SVector{5}),
            χ²_B, 0.0, χ²_B,
            timestamp,
            decision,
            3
        )
    end

    # Gradient residual
    r_G = SVector{5}(G_meas...) - SVector{5}(G_pred...)
    χ²_G = dot(r_G, r_G) / config.noise_G^2

    # Combined χ² (normalized)
    Σ_inv = Diagonal(vcat(fill(1/config.noise_B^2, 3), fill(1/config.noise_G^2, 5)))
    r_combined = vcat(r_B, r_G)
    χ²_combined = dot(r_combined, Σ_inv * r_combined)

    # Decision using d=8 thresholds
    decision = if χ²_combined > CHI2_D8_STRONG_THRESHOLD
        :strong_outlier
    elseif χ²_combined > CHI2_D8_MILD_THRESHOLD
        :mild_outlier
    else
        :inlier
    end

    return GradientResidualStatistics(
        SVector{3}(position...),
        r_B,
        r_G,
        χ²_B, χ²_G, χ²_combined,
        timestamp,
        decision,
        8
    )
end

# ============================================================================
# Gradient-Enabled Confidence Model
# ============================================================================

"""
    GradientConfidenceModel

Nav confidence model calibrated for gradient measurements.

Uses √γ form: k = a + b × √(max(0, γ - offset))
"""
struct GradientConfidenceModel
    a::Float64           # Base multiplier
    b::Float64           # Slope
    offset::Float64      # γ offset (typically d-1)
end

# Default model for d=8
const GRADIENT_CONFIDENCE_MODEL = GradientConfidenceModel(1.2, 0.25, 8.0)

# Field-only model for comparison (d=3)
const FIELD_CONFIDENCE_MODEL = GradientConfidenceModel(1.26, 0.31, 3.0)

"""
    compute_gradient_confidence(γ; model)

Compute confidence from χ² statistic.

Returns (confidence, multiplier) where:
- multiplier = a + b × √(max(0, γ - offset))
- confidence = 1 / multiplier
"""
function compute_gradient_confidence(γ::Float64;
                                       model::GradientConfidenceModel = GRADIENT_CONFIDENCE_MODEL)
    k = model.a + model.b * sqrt(max(0.0, γ - model.offset))
    confidence = 1.0 / k
    return (confidence = clamp(confidence, 0.0, 1.0), multiplier = k)
end

"""Get human-readable confidence label."""
function confidence_label(conf::Float64)
    if conf >= 0.8
        return "HIGH"
    elseif conf >= 0.5
        return "MEDIUM"
    elseif conf >= 0.2
        return "LOW"
    else
        return "UNRELIABLE"
    end
end

# ============================================================================
# Gradient Baseline Comparison
# ============================================================================

"""
    GradientBaseline

Compare performance with and without gradients.
"""
mutable struct GradientBaseline
    # Field-only statistics
    field_chi2_sum::Float64
    field_chi2_count::Int
    field_outlier_count::Int

    # Gradient-enabled statistics
    gradient_chi2_sum::Float64
    gradient_chi2_count::Int
    gradient_outlier_count::Int
end

GradientBaseline() = GradientBaseline(0.0, 0, 0, 0.0, 0, 0)

"""Update baseline with paired statistics."""
function update_gradient_baseline!(baseline::GradientBaseline,
                                    stats_field::GradientResidualStatistics,
                                    stats_gradient::GradientResidualStatistics)
    # Field-only
    baseline.field_chi2_sum += stats_field.chi2_combined
    baseline.field_chi2_count += 1
    if stats_field.decision != :inlier
        baseline.field_outlier_count += 1
    end

    # Gradient-enabled
    baseline.gradient_chi2_sum += stats_gradient.chi2_combined
    baseline.gradient_chi2_count += 1
    if stats_gradient.decision != :inlier
        baseline.gradient_outlier_count += 1
    end
end

"""Get comparison statistics."""
function get_baseline_comparison(baseline::GradientBaseline)
    if baseline.field_chi2_count == 0 || baseline.gradient_chi2_count == 0
        return (improvement = 0.0, field_mean = 0.0, gradient_mean = 0.0,
                field_outlier_rate = 0.0, gradient_outlier_rate = 0.0)
    end

    field_mean = baseline.field_chi2_sum / baseline.field_chi2_count
    gradient_mean = baseline.gradient_chi2_sum / baseline.gradient_chi2_count

    # Expected χ² is d, so improvement is measured relative to that
    field_normalized = field_mean / 3.0  # d=3
    gradient_normalized = gradient_mean / 8.0  # d=8

    improvement = field_normalized / max(gradient_normalized, 0.01)

    return (
        improvement = improvement,
        field_mean = field_mean,
        gradient_mean = gradient_mean,
        field_outlier_rate = baseline.field_outlier_count / baseline.field_chi2_count,
        gradient_outlier_rate = baseline.gradient_outlier_count / baseline.gradient_chi2_count
    )
end

# ============================================================================
# Exports
# ============================================================================

export GradientConfig, DEFAULT_GRADIENT_CONFIG, GRADIENT_DISABLED_CONFIG
export CHI2_D8_MILD_THRESHOLD, CHI2_D8_STRONG_THRESHOLD
export chi2_threshold_for_d
export GradientMeasurement, gradient_measurement_vector, gradient_measurement_covariance
export GradientResidualStatistics, compute_gradient_residual_statistics
export GradientConfidenceModel, GRADIENT_CONFIDENCE_MODEL, FIELD_CONFIDENCE_MODEL
export compute_gradient_confidence, confidence_label
export GradientBaseline, update_gradient_baseline!, get_baseline_comparison
