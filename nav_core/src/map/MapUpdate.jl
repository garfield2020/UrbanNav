# ============================================================================
# Map Update Pipeline (Phase B)
# ============================================================================
#
# Implements information-form (equivalently, weighted least squares) updates
# for map coefficient learning.
#
# Mathematical Framework:
#   Prior: α ~ N(α₀, P_α₀)
#   Measurement model: z = H·α + v, v ~ N(0, R)
#   Information form: Λ = P⁻¹, η = Λ·α
#   Update: Λ_new = Λ_old + H'R⁻¹H
#           η_new = η_old + H'R⁻¹z
#
# Key Properties:
# - Commutative: order of updates doesn't affect final result (within tolerance)
# - Incremental: can process measurements one at a time
# - Batch-efficient: multiple measurements can be accumulated
#
# Physics Contract:
# - Map updates modify ONLY map coefficients α, NEVER truth world
# - All uncertainties in SI units (Tesla² for field, (T/m)² for gradient)
# - Coordinate frames: NED world frame throughout
#
# ============================================================================

using LinearAlgebra
using StaticArrays
using Distributions: Chisq, quantile

# ============================================================================
# Statistical Thresholds (with p-value justification)
# ============================================================================

# Chi-square thresholds for outlier detection
# For d=3 (field only): χ²(3)
# For d=8 (field + gradient): χ²(8)
#
# Reference: Standard statistical tables
# p = 0.01: mild outlier (1% false positive rate)
# p = 0.001: strong outlier (0.1% false positive rate)

"""
    chi2_outlier_threshold(dim::Int, p_value::Float64)

Compute χ² threshold for outlier detection.

# Arguments
- `dim`: Degrees of freedom (measurement dimension)
- `p_value`: Desired false positive rate (e.g., 0.01 for 1%)

# Returns
Threshold τ such that P(χ²(dim) > τ) = p_value

# Example
```julia
# For 3D field measurement, 1% false positive rate:
τ = chi2_outlier_threshold(3, 0.01)  # ≈ 11.345
```
"""
function chi2_outlier_threshold(dim::Int, p_value::Float64)
    @assert dim > 0 "Dimension must be positive"
    @assert 0 < p_value < 1 "p_value must be in (0, 1)"
    return quantile(Chisq(dim), 1 - p_value)
end

# Pre-computed thresholds for common cases (verified against statistical tables)
const CHI2_D3_P001 = 11.345   # P(χ²(3) > 11.345) = 0.01
const CHI2_D3_P0001 = 16.266  # P(χ²(3) > 16.266) = 0.001
const CHI2_D8_P001 = 20.090   # P(χ²(8) > 20.090) = 0.01
const CHI2_D8_P0001 = 26.125  # P(χ²(8) > 26.125) = 0.001

"""
    default_outlier_threshold(dim::Int)

Get default outlier threshold for given measurement dimension.

Uses p = 0.01 (1% false positive rate) as default.
"""
function default_outlier_threshold(dim::Int)
    if dim == 3
        return CHI2_D3_P001
    elseif dim == 8
        return CHI2_D8_P001
    else
        return chi2_outlier_threshold(dim, 0.01)
    end
end

# ============================================================================
# Teachability Thresholds (with physical justification)
# ============================================================================

# Teachability gating prevents "bad pose states teaching the map"
#
# Physical reasoning:
# - If pose uncertainty σ_pos [m] is large, the measurement location is uncertain
# - This uncertainty propagates to gradient-based corrections
# - For gradient G [T/m], pose error δx [m] causes field error G·δx [T]
#
# Rule of thumb: pose uncertainty should contribute << measurement uncertainty
# σ²_pose_contribution = |G|² · σ²_pos
#
# For typical gradient |G| ≈ 100 nT/m = 100e-9 T/m and sensor noise σ_B ≈ 10 nT:
# σ_pos < σ_B / |G| = 10e-9 / 100e-9 = 0.1 m → σ²_pos < 0.01 m²
#
# More conservative threshold: σ²_pos < 1.0 m² (σ_pos < 1 m)

"""
    DEFAULT_MAX_POSE_UNCERTAINTY_M2

Maximum pose uncertainty [m²] for map learning to proceed.

# Physical Justification
For gradient strength |G| = 100 nT/m and sensor noise σ_B = 10 nT:
- Pose error δx causes field error δB = |G|·δx
- For δB < σ_B, need δx < σ_B/|G| = 0.1 m

Conservative threshold of 1.0 m² (1 m standard deviation) provides
10× margin for typical operating conditions.
"""
const DEFAULT_MAX_POSE_UNCERTAINTY_M2 = 1.0  # [m²]

"""
    compute_pose_field_coupling(pose_uncertainty_m2::Float64, gradient_norm::Float64)

Compute field uncertainty contribution from pose uncertainty.

# Arguments
- `pose_uncertainty_m2`: Position variance [m²]
- `gradient_norm`: Gradient magnitude [T/m]

# Returns
Field variance contribution [T²]: σ²_B_pose = |G|² · σ²_pos
"""
function compute_pose_field_coupling(pose_uncertainty_m2::Float64, gradient_norm::Float64)
    return gradient_norm^2 * pose_uncertainty_m2
end

# ============================================================================
# Information-Form Map Updater
# ============================================================================

"""
    MapUpdaterConfig

Configuration for map update algorithm.

# Fields
- `outlier_p_value::Float64`: p-value for outlier threshold (default: 0.01)
- `min_weight::Float64`: Minimum update weight ∈ [0, 1] (default: 0.1)
- `max_pose_uncertainty_m2::Float64`: Maximum pose variance [m²] (default: 1.0)
- `regularization::Float64`: Regularization for near-singular matrices (default: 1e-12)

# Threshold Justifications
- `outlier_p_value = 0.01`: 1% false positive rate (standard χ² test)
- `min_weight = 0.1`: Reject updates weighted below 10% effectiveness
- `max_pose_uncertainty = 1.0 m²`: Conservative teachability gate (see docstring)
- `regularization = 1e-12`: Numerical stability, well below measurement precision
"""
struct MapUpdaterConfig
    outlier_p_value::Float64
    min_weight::Float64
    max_pose_uncertainty_m2::Float64
    regularization::Float64

    function MapUpdaterConfig(;
        outlier_p_value::Float64 = 0.01,
        min_weight::Float64 = 0.1,
        max_pose_uncertainty_m2::Float64 = DEFAULT_MAX_POSE_UNCERTAINTY_M2,
        regularization::Float64 = 1e-12
    )
        @assert 0 < outlier_p_value < 1 "outlier_p_value must be in (0, 1)"
        @assert 0 <= min_weight <= 1 "min_weight must be in [0, 1]"
        @assert max_pose_uncertainty_m2 > 0 "max_pose_uncertainty must be positive"
        @assert regularization > 0 "regularization must be positive"
        new(outlier_p_value, min_weight, max_pose_uncertainty_m2, regularization)
    end
end

const DEFAULT_MAP_UPDATER_CONFIG = MapUpdaterConfig()

"""
    InformationFormUpdater <: AbstractMapUpdater

Map updater using information-form (canonical) Kalman filter update.

# Information-Form Update Equations
    Λ_prior = P_prior⁻¹          (prior information matrix)
    η_prior = Λ_prior · α_prior  (prior information vector)

    Λ_update = H' R⁻¹ H          (measurement information)
    η_update = H' R⁻¹ z          (measurement contribution)

    Λ_post = Λ_prior + Λ_update
    η_post = η_prior + η_update

    α_post = Λ_post⁻¹ η_post
    P_post = Λ_post⁻¹

# Properties
- **Commutative**: Same data in different order gives same result (within tolerance)
- **Numerically stable**: Information form avoids subtractive cancellation
- **Conservative**: Rejects outliers and low-quality updates based on χ² test

# Architecture Note
This updater modifies ONLY map coefficients α. It has NO access to:
- Truth world model
- Navigation state
- Sensor raw data

All information comes through MapUpdateMessage, enforcing clean separation.
"""
struct InformationFormUpdater <: AbstractMapUpdater
    config::MapUpdaterConfig
end

InformationFormUpdater(; kwargs...) = InformationFormUpdater(MapUpdaterConfig(; kwargs...))

"""
    can_update(updater::InformationFormUpdater, msg::MapUpdateMessage)

Check if an update message passes teachability gates.

# Gating Criteria
1. **Weight threshold**: msg.weight ≥ config.min_weight
2. **Pose uncertainty**: msg.pose_uncertainty ≤ config.max_pose_uncertainty_m2
3. **Valid data**: No NaN/Inf in innovation or Jacobian

Returns (allowed::Bool, reason::Symbol).
Reason is :ok if allowed, otherwise the rejection reason.
"""
function can_update(updater::InformationFormUpdater, msg::MapUpdateMessage)
    config = updater.config

    # Check weight threshold
    if msg.weight < config.min_weight
        return (false, :weight_threshold)
    end

    # Check pose uncertainty (teachability gate)
    if msg.pose_uncertainty > config.max_pose_uncertainty_m2
        return (false, :low_teachability)
    end

    # Check for valid innovation
    if any(isnan, msg.innovation) || any(isinf, msg.innovation)
        return (false, :invalid_innovation)
    end

    # Check for valid Jacobian
    if any(isnan, msg.H) || any(isinf, msg.H)
        return (false, :invalid_jacobian)
    end

    return (true, :ok)
end

"""
    compute_normalized_innovation_squared(msg::MapUpdateMessage)

Compute normalized innovation squared (NIS) for χ² outlier test.

    NIS = r' R⁻¹ r

where r is the innovation (measurement residual) and R is the measurement covariance.

Under H₀ (no outlier), NIS ~ χ²(d) where d = dim(r).
"""
function compute_normalized_innovation_squared(msg::MapUpdateMessage)
    R = msg.R
    r = msg.innovation
    d = length(r)

    # Add relative regularization for numerical stability
    # Use 1e-10 times the minimum diagonal element, or absolute floor of 1e-30
    min_diag = minimum(abs.(diag(R)))
    reg = max(min_diag * 1e-10, 1e-30)
    R_reg = R + reg * I(d)

    try
        # Use Cholesky for numerical stability
        L = cholesky(Symmetric(R_reg))
        r_white = L.L \ r
        return dot(r_white, r_white)
    catch e
        # Fallback to direct computation if Cholesky fails
        try
            R_inv = inv(R_reg)
            return r' * R_inv * r
        catch
            return Inf  # Reject if matrix is singular
        end
    end
end

"""
    is_outlier(updater::InformationFormUpdater, msg::MapUpdateMessage)

Test if innovation is an outlier using χ² test.

# Test
    H₀: measurement is consistent with map prediction
    H₁: measurement is an outlier

Test statistic: NIS = r' R⁻¹ r ~ χ²(d) under H₀
Reject H₀ if NIS > χ²_{1-p}(d) where p = config.outlier_p_value
"""
function is_outlier(updater::InformationFormUpdater, msg::MapUpdateMessage)
    nis = compute_normalized_innovation_squared(msg)
    d = length(msg.innovation)
    threshold = chi2_outlier_threshold(d, updater.config.outlier_p_value)
    return nis > threshold
end

"""
    apply_update(updater::InformationFormUpdater, tile::MapTileData, msg::MapUpdateMessage)

Apply a single update to tile coefficients using information form.

# Returns
MapUpdateResult containing:
- Updated coefficients α (or original if rejected)
- Updated covariance P (or original if rejected)
- Count of applied/rejected updates
- Rejection reasons (if any)
"""
function apply_update(updater::InformationFormUpdater, tile::MapTileData,
                      msg::MapUpdateMessage)
    config = updater.config

    # Check teachability gates
    allowed, reason = can_update(updater, msg)
    if !allowed
        return MapUpdateResult(
            false,
            tile.coefficients,
            tile.covariance,
            0, 1, [reason]
        )
    end

    # Check for outlier
    if is_outlier(updater, msg)
        return MapUpdateResult(
            false,
            tile.coefficients,
            tile.covariance,
            0, 1, [:outlier]
        )
    end

    # Get current state
    α = tile.coefficients
    P = tile.covariance
    n = length(α)

    # Extract update data
    H = msg.H
    R = msg.R
    z = msg.innovation + H * α  # Reconstruct measurement from innovation

    # Check dimension compatibility
    if size(H, 2) != n
        return MapUpdateResult(
            false,
            tile.coefficients,
            tile.covariance,
            0, 1, [:dimension_mismatch]
        )
    end

    # Apply weight to measurement covariance (weight < 1 means more uncertain)
    R_weighted = R / max(msg.weight, 1e-10)

    # Information form update
    try
        ε = config.regularization

        # Prior information
        P_reg = P + ε * I(n)
        Λ_prior = inv(Symmetric(P_reg))
        η_prior = Λ_prior * α

        # Measurement information
        R_reg = R_weighted + ε * I(size(R, 1))
        R_inv = inv(Symmetric(R_reg))
        Λ_meas = H' * R_inv * H
        η_meas = H' * R_inv * z

        # Posterior information
        Λ_post = Λ_prior + Λ_meas
        η_post = η_prior + η_meas

        # Convert back to mean/covariance form
        P_new = inv(Symmetric(Λ_post))
        P_new = Matrix((P_new + P_new') / 2)  # Ensure exact symmetry
        α_new = P_new * η_post

        return MapUpdateResult(
            true,
            Vector(α_new),
            P_new,
            1, 0, Symbol[]
        )
    catch e
        return MapUpdateResult(
            false,
            tile.coefficients,
            tile.covariance,
            0, 1, [:singular]
        )
    end
end

"""
    apply_batch(updater::InformationFormUpdater, tile::MapTileData,
                msgs::Vector{MapUpdateMessage})

Apply multiple updates efficiently using batched information form.

# Batch Update Property
Information form is additive in the information contributions:

    Λ_batch = Σᵢ Hᵢ' Rᵢ⁻¹ Hᵢ
    η_batch = Σᵢ Hᵢ' Rᵢ⁻¹ zᵢ

This guarantees **commutativity**: same messages in any order produce
identical results (within numerical tolerance).

# Returns
MapUpdateResult with aggregated statistics across all messages.
"""
function apply_batch(updater::InformationFormUpdater, tile::MapTileData,
                     msgs::Vector{MapUpdateMessage})
    if isempty(msgs)
        return MapUpdateResult(false, tile.coefficients, tile.covariance, 0, 0, Symbol[])
    end

    config = updater.config
    ε = config.regularization

    # Get current state
    α = tile.coefficients
    P = tile.covariance
    n = length(α)

    # Prior information
    P_reg = P + ε * I(n)
    Λ_prior = inv(Symmetric(P_reg))
    η_prior = Λ_prior * α

    # Accumulate batch information
    Λ_batch = zeros(n, n)
    η_batch = zeros(n)

    updates_applied = 0
    updates_rejected = 0
    rejection_reasons = Symbol[]

    for msg in msgs
        # Check teachability gates
        allowed, reason = can_update(updater, msg)
        if !allowed
            updates_rejected += 1
            reason ∉ rejection_reasons && push!(rejection_reasons, reason)
            continue
        end

        # Check for outlier
        if is_outlier(updater, msg)
            updates_rejected += 1
            :outlier ∉ rejection_reasons && push!(rejection_reasons, :outlier)
            continue
        end

        # Check dimensions
        H = msg.H
        R = msg.R
        if size(H, 2) != n
            updates_rejected += 1
            :dimension_mismatch ∉ rejection_reasons && push!(rejection_reasons, :dimension_mismatch)
            continue
        end

        # Reconstruct measurement from innovation
        z = msg.innovation + H * α

        # Apply weight
        R_weighted = R / max(msg.weight, 1e-10)

        try
            R_reg = R_weighted + ε * I(size(R, 1))
            R_inv = inv(Symmetric(R_reg))
            Λ_batch += H' * R_inv * H
            η_batch += H' * R_inv * z
            updates_applied += 1
        catch
            updates_rejected += 1
            :singular ∉ rejection_reasons && push!(rejection_reasons, :singular)
        end
    end

    if updates_applied == 0
        return MapUpdateResult(
            false,
            tile.coefficients,
            tile.covariance,
            0, updates_rejected, rejection_reasons
        )
    end

    # Apply batch update
    try
        Λ_post = Λ_prior + Λ_batch
        η_post = η_prior + η_batch

        P_new = inv(Symmetric(Λ_post))
        P_new = Matrix((P_new + P_new') / 2)  # Ensure exact symmetry
        α_new = P_new * η_post

        return MapUpdateResult(
            true,
            Vector(α_new),
            P_new,
            updates_applied, updates_rejected, rejection_reasons
        )
    catch
        return MapUpdateResult(
            false,
            tile.coefficients,
            tile.covariance,
            0, updates_rejected + updates_applied,
            vcat(rejection_reasons, [:batch_singular])
        )
    end
end

# ============================================================================
# Verification Utilities
# ============================================================================

"""
    verify_commutativity(updater::AbstractMapUpdater, tile::MapTileData,
                        msgs::Vector{MapUpdateMessage}; tol::Float64 = 1e-10)

Verify that batch update is commutative (order-independent).

This is a key property of the information-form update: applying the same
measurements in different orders should produce identical results.

# Returns
true if coefficient difference < tol after applying in forward vs reverse order.
"""
function verify_commutativity(updater::AbstractMapUpdater, tile::MapTileData,
                              msgs::Vector{MapUpdateMessage}; tol::Float64 = 1e-10)
    if length(msgs) < 2
        return true
    end

    # Apply in original order
    result1 = apply_batch(updater, tile, msgs)

    # Apply in reverse order
    result2 = apply_batch(updater, tile, reverse(msgs))

    if !result1.success || !result2.success
        return result1.success == result2.success
    end

    # Check coefficient equality
    coeff_diff = norm(result1.coefficients - result2.coefficients)
    cov_diff = norm(result1.covariance - result2.covariance)

    return coeff_diff < tol && cov_diff < tol
end

"""
    verify_covariance_reduction(tile_before::MapTileData, result::MapUpdateResult)

Verify that update reduced covariance (information gain).

For a valid measurement update, we expect:
    trace(P_after) ≤ trace(P_before)

Returns (reduced::Bool, before_trace::Float64, after_trace::Float64)
"""
function verify_covariance_reduction(tile_before::MapTileData, result::MapUpdateResult)
    if !result.success || result.updates_applied == 0
        return (true, NaN, NaN)  # No update, trivially satisfied
    end

    trace_before = tr(tile_before.covariance)
    trace_after = tr(result.covariance)

    # Allow small numerical tolerance
    reduced = trace_after <= trace_before * (1 + 1e-10)

    return (reduced, trace_before, trace_after)
end

# ============================================================================
# Exports
# ============================================================================

export MapUpdaterConfig, DEFAULT_MAP_UPDATER_CONFIG
export InformationFormUpdater
export can_update, apply_update, apply_batch
export chi2_outlier_threshold, default_outlier_threshold
export compute_normalized_innovation_squared, is_outlier
export compute_pose_field_coupling
export verify_commutativity, verify_covariance_reduction

# Pre-computed thresholds
export CHI2_D3_P001, CHI2_D3_P0001, CHI2_D8_P001, CHI2_D8_P0001
export DEFAULT_MAX_POSE_UNCERTAINTY_M2
