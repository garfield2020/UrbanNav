# ============================================================================
# OnlineTileUpdater.jl - Online Tile Coefficient Updates (Phase C Step 5)
# ============================================================================
#
# Updates harmonic tile coefficients from streaming measurements using
# information-form fusion for numerical stability.
#
# Key Constraints:
# 1. Tile basis is for SMOOTH fields - must not absorb 1/r³ dipole signatures
# 2. Updates must maintain coefficient conditioning as mission length increases
# 3. Map covariance shrinks only when justified by data (no artificial collapse)
#
# Architecture:
# - Information-form updates: I_new = I_old + H'R⁻¹H, numerically stable
# - Teachability gating: reject updates when pose uncertainty is high
# - Residual gating: reject outliers that would corrupt tile
# - Smoothness prior: optional tile-to-tile continuity regularization
#
# INV-07: This updater has NO access to truth world. All information comes
# through MapUpdateMessage structs containing measurements and statistics.
#
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Update Configuration
# ============================================================================

"""
    OnlineTileUpdaterConfig

Configuration for online tile coefficient updates.

# Fields
## Teachability Gating
- `max_pose_uncertainty_m2::Float64`: Maximum pose uncertainty for teachable update [m²]
- `min_gradient_energy::Float64`: Minimum gradient energy for informative update [T²/m²]

## Residual Gating
- `outlier_threshold_chi2::Float64`: Chi-square threshold for outlier rejection
- `max_consecutive_outliers::Int`: Max outliers before flagging tile issue

## Update Control
- `min_weight::Float64`: Minimum update weight to apply [0-1]
- `max_covariance_shrink_rate::Float64`: Max per-update covariance shrink [0-1]
- `regularization_strength::Float64`: Diagonal regularization for stability

## Smoothness Prior
- `enable_smoothness_prior::Bool`: Enable tile-to-tile smoothness
- `smoothness_weight::Float64`: Weight for smoothness prior

# Physics Justification
- max_pose_uncertainty_m2: 10 m² default; larger means measurement position too uncertain
- outlier_threshold_chi2: 16.266 = χ²(3, 0.001) for d=3 field measurements
- max_covariance_shrink_rate: 0.95 prevents artificial rapid collapse
"""
struct OnlineTileUpdaterConfig
    # Teachability gating
    max_pose_uncertainty_m2::Float64
    min_gradient_energy::Float64

    # Residual gating
    outlier_threshold_chi2::Float64
    max_consecutive_outliers::Int

    # Update control
    min_weight::Float64
    max_covariance_shrink_rate::Float64
    regularization_strength::Float64

    # Smoothness prior
    enable_smoothness_prior::Bool
    smoothness_weight::Float64

    function OnlineTileUpdaterConfig(;
        max_pose_uncertainty_m2::Float64 = 10.0,
        min_gradient_energy::Float64 = 1e-18,  # (1 nT/m)²
        outlier_threshold_chi2::Float64 = 16.266,  # χ²(3, 0.001)
        max_consecutive_outliers::Int = 3,
        min_weight::Float64 = 0.1,
        max_covariance_shrink_rate::Float64 = 0.95,
        regularization_strength::Float64 = 1e-12,
        enable_smoothness_prior::Bool = false,
        smoothness_weight::Float64 = 0.01
    )
        @assert max_pose_uncertainty_m2 > 0 "max_pose_uncertainty must be positive"
        @assert outlier_threshold_chi2 > 0 "outlier_threshold must be positive"
        @assert 0.0 <= min_weight <= 1.0 "min_weight must be in [0, 1]"
        @assert 0.0 < max_covariance_shrink_rate <= 1.0 "shrink_rate must be in (0, 1]"
        @assert regularization_strength >= 0 "regularization must be non-negative"

        new(max_pose_uncertainty_m2, min_gradient_energy, outlier_threshold_chi2,
            max_consecutive_outliers, min_weight, max_covariance_shrink_rate,
            regularization_strength, enable_smoothness_prior, smoothness_weight)
    end
end

"""Default online tile updater configuration."""
const DEFAULT_ONLINE_TILE_UPDATER_CONFIG = OnlineTileUpdaterConfig()

# ============================================================================
# Update Message (from measurement)
# ============================================================================

"""
    TileUpdateObservation

Single observation for tile update.

# Fields
- `position::Vec3Map`: Measurement position in world frame [m]
- `field_residual::Vec3Map`: z - h(α), innovation w.r.t. current tile [T]
- `R_meas::Mat3Map`: Measurement covariance (sensor only) [T²]
- `pose_uncertainty::Float64`: Position uncertainty at measurement [m²]
- `timestamp::Float64`: Observation timestamp [s]
- `weight::Float64`: Quality weight [0-1]

# Usage
Observations are collected during fast loop and batched for slow loop update.
"""
struct TileUpdateObservation
    position::Vec3Map
    field_residual::Vec3Map
    R_meas::Mat3Map
    pose_uncertainty::Float64
    timestamp::Float64
    weight::Float64
end

"""Create observation with default weight."""
function TileUpdateObservation(position::AbstractVector, residual::AbstractVector,
                               R::AbstractMatrix, pose_unc::Float64, timestamp::Float64)
    TileUpdateObservation(
        Vec3Map(position...),
        Vec3Map(residual...),
        Mat3Map(R...),
        pose_unc,
        timestamp,
        1.0  # Default full weight
    )
end

# ============================================================================
# Update Result
# ============================================================================

"""
    TileUpdateRejection

Reasons for rejecting a tile update.
"""
@enum TileUpdateRejection begin
    REJECT_NONE = 0
    REJECT_LOW_TEACHABILITY = 1   # Pose uncertainty too high
    REJECT_OUTLIER = 2            # Residual too large
    REJECT_LOW_WEIGHT = 3         # Weight below threshold
    REJECT_SINGULAR = 4           # Update would cause singularity
    REJECT_LOW_GRADIENT = 5       # Gradient energy too low
end

"""
    TileUpdateResult

Result of applying update(s) to a tile.

# Fields
- `success::Bool`: Whether any updates were applied
- `updates_applied::Int`: Number of updates successfully applied
- `updates_rejected::Int`: Number of updates rejected
- `rejection_reasons::Vector{TileUpdateRejection}`: Reasons for rejections
- `new_coefficients::Vector{Float64}`: Updated coefficients
- `new_covariance::Matrix{Float64}`: Updated covariance
- `covariance_shrink_ratio::Float64`: Ratio of new to old covariance trace
- `max_chi2::Float64`: Maximum chi-square of applied updates
"""
struct TileUpdateResult
    success::Bool
    updates_applied::Int
    updates_rejected::Int
    rejection_reasons::Vector{TileUpdateRejection}
    new_coefficients::Vector{Float64}
    new_covariance::Matrix{Float64}
    covariance_shrink_ratio::Float64
    max_chi2::Float64
end

"""Failed update result (no changes)."""
function failed_update_result(tile::SlamTileState, reasons::Vector{TileUpdateRejection})
    TileUpdateResult(
        false, 0, length(reasons), reasons,
        copy(tile.coefficients), copy(tile.covariance),
        1.0, 0.0
    )
end

# ============================================================================
# Online Tile Updater
# ============================================================================

"""
    OnlineTileUpdater

Stateless updater for tile coefficients using information-form fusion.

# Architecture
The updater is STATELESS - all state is in SlamTileState.
This ensures:
- Determinism: same inputs → same outputs
- Testability: can unit test with mock tiles
- INV-07 compliance: no access to truth world

# Algorithm
Uses information-form updates for numerical stability:
```
I_new = I_old + Σᵢ wᵢ Hᵢ' Rᵢ⁻¹ Hᵢ
i_new = i_old + Σᵢ wᵢ Hᵢ' Rᵢ⁻¹ zᵢ
α_new = I_new⁻¹ i_new
P_new = I_new⁻¹
```

# Critical Constraint
The tile basis is for SMOOTH fields. Dipole signatures (1/r³) must NOT
be absorbed into tile coefficients. This is enforced by:
1. Residual gating rejects large outliers (likely dipoles)
2. Source separation upstream removes known dipole contributions
3. Smoothness prior prevents local overfitting
"""
struct OnlineTileUpdater
    config::OnlineTileUpdaterConfig
end

"""Default tile updater."""
const DEFAULT_ONLINE_TILE_UPDATER = OnlineTileUpdater(DEFAULT_ONLINE_TILE_UPDATER_CONFIG)

# ============================================================================
# Single Update
# ============================================================================

"""
    can_apply_update(updater::OnlineTileUpdater, tile::SlamTileState,
                     obs::TileUpdateObservation) -> (Bool, TileUpdateRejection)

Check if an observation can be used for tile update.

# Gates (in order)
1. Weight gate: obs.weight >= min_weight
2. Teachability gate: obs.pose_uncertainty <= max_pose_uncertainty
3. Outlier gate: chi² <= threshold
"""
function can_apply_update(updater::OnlineTileUpdater, tile::SlamTileState,
                          obs::TileUpdateObservation)
    config = updater.config

    # Weight gate
    if obs.weight < config.min_weight
        return (false, REJECT_LOW_WEIGHT)
    end

    # Teachability gate
    if obs.pose_uncertainty > config.max_pose_uncertainty_m2
        return (false, REJECT_LOW_TEACHABILITY)
    end

    # Outlier gate (compute chi-square of residual)
    # Use current tile to predict, compare with observation
    R_total = Matrix(obs.R_meas) + config.regularization_strength * I
    chi2 = dot(obs.field_residual, R_total \ Vector(obs.field_residual))

    if chi2 > config.outlier_threshold_chi2
        return (false, REJECT_OUTLIER)
    end

    return (true, REJECT_NONE)
end

"""
Maximum active basis dimension until Tier2 Jacobians are implemented.
Tier2 (quadratic, dims 9:15) requires matching Jacobian columns AND
matching prediction in OnlineMapProvider. Until both exist, the active
subspace is capped at 8 (linear: B₀ + G).

To enable Tier2:
1. Fill H[:, 9:15] here with quadratic basis Jacobians
2. Implement quadratic prediction in OnlineMapProvider.evaluate_tile_prediction
3. Raise this constant to 15
"""
const TIER2_MAX_ACTIVE_DIM = 8

"""
Minimum XY baseline within tile for gradient learning (k > 3).
Physics: gradient columns scale as dx. With tile_size=50m,
30m span gives gradient entries ~60% of tile scale. Must be
synchronized with OnlineMapProvider prediction gate — if solve
uses k=8 but prediction uses B0-only, gradient coupling in the
information matrix causes B0 covariance to shrink too fast,
producing NEES overconfidence (observed as NEES spikes ~50-120).
"""
const GRADIENT_MIN_SPAN_M = 30.0

"""
Minimum observations before gradient learning.
Statistics: 8D solve needs at least 8 independent obs.
80 provides 10× margin, ensuring gradient information matrix
is well-conditioned before gradient dims influence the solve.
Must match OnlineMapProvider prediction gate threshold.
"""
const GRADIENT_MIN_OBS = 80

# ============================================================================
# Tier-2 (Quadratic) Gate Constants
# ============================================================================
# All gate math uses ΔI (data information only), never I_total (prior + data).
# This prevents the "prior dominates conditioning" bug class.

"""
Minimum XY baseline for Tier-2 unlock [m]. 90% of 50m tile.
Quadratic terms scale as O(r²); 45m baseline gives quadratic entries
~2025/r² significant. Stricter than Tier-1 (30m) because curvature
needs larger spatial excitation to be observable.
"""
const TIER2_MIN_SPAN_M = 45.0

"""
Minimum per-axis span for Tier-2 [m]. Both X AND Y must exceed this.
Curvature requires 2D excitation; 1D traversal is insufficient.
Set to 30m (60% of 50m tile) to reject straight-line trajectories
where lateral drift from process noise can reach ~22m over 450m.
"""
const TIER2_MIN_SPAN_BOTH = 30.0

"""
Minimum observations for Tier-2 unlock.
15D solve needs ≥15 independent obs. 120 gives 8× margin.
Stricter than Tier-1 (80) since 7 additional curvature DOF.
"""
const TIER2_MIN_OBS = 120

"""
Scale-invariant information gain ratio threshold for Tier-2 unlock.
ρ = tr(ΔI_qq) / tr(I_qq_prior) must exceed this value.
0.01 means data contributes ≥1% of prior information — sufficient for
the posterior to begin shrinking meaningfully. Scale-invariant: works
regardless of basis scaling (1e-9 factor, tile size, etc.).
"""
const TIER2_MIN_INFO_GAIN_RATIO = 0.01

"""
Min rank of ΔI_quad (7×7). Demanding full rank=7 is too strict;
≥5 protects against degenerate trajectories.
"""
const TIER2_MIN_RANK_QUAD = 5

"""
Cross-coupling ratio threshold: ||ΔI_aq||_F / ||ΔI_aa||_F < 20%.
Ensures quadratic block is weakly coupled to linear block in DATA FIM.
Uses Frobenius norm for testability.
"""
const TIER2_MAX_CROSS_COUPLING = 0.2

"""
Minimum relative conditioning of ΔI_qq: σ_min/σ_max ≥ threshold.
Scale-invariant alternative to absolute condition number check.
1e-6 means condition number ≤ 1e6, generous enough for quadratic
terms while still rejecting rank-deficient data.
"""
const TIER2_MIN_RELATIVE_COND = 1e-6

"""
Re-lock if relative conditioning σ_min/σ_max of ΔI_qq drops below this.
10× stricter than unlock (1e-7 vs 1e-6) for hysteresis.
"""
const TIER2_RELOCK_RELATIVE_COND = 1e-7

"""
Re-lock if spatial diversity drops below this [m].
20% margin above Tier-1 threshold (30m).
Accounts for vehicle exiting tile.
"""
const TIER2_RELOCK_SPAN_M = 36.0

"""
Column norm threshold for Jacobian informativeness detection.

A Jacobian column is considered "informed" (carrying real measurement information)
if its L2 norm exceeds this threshold. This must be:
- Well above floating-point noise (~1e-16): prevents false positives from FP artifacts
- Well below smallest physical Jacobian entry: gradient terms scale as Δx (meters),
  so the smallest real entry is O(1e-3) for mm-scale offsets in T/m units.

Used by both `informed_jacobian_rank` (Scheduler) and `_last_informed_column` (Updater)
to ensure consistent active_dim determination across the entire update pipeline.
"""
const JACOBIAN_COL_NORM_THRESHOLD = 1e-10

"""
    canonical_informed_dim(H::Matrix{Float64}) -> Int

Determine the canonical active dimension from a Jacobian matrix by finding the
last column with L2 norm above JACOBIAN_COL_NORM_THRESHOLD, then clamping to
the canonical basis dimensions {3, 8, 15}.

This is the single implementation used by both Scheduler and Updater to ensure
they always agree on active_dim for a given Jacobian.
"""
function canonical_informed_dim(H::Matrix{Float64})
    last_informed = 0
    for j in 1:size(H, 2)
        col_norm_sq = 0.0
        for i in 1:size(H, 1)
            col_norm_sq += H[i, j]^2
        end
        if sqrt(col_norm_sq) > JACOBIAN_COL_NORM_THRESHOLD
            last_informed = j
        end
    end
    # Clamp to canonical basis dimensions: {3, 8, 15}
    if last_informed <= 3
        return 3
    elseif last_informed <= 8
        return 8
    else
        return 15
    end
end

"""
    compute_tile_jacobian(tile::SlamTileState, position::Vec3Map) -> Matrix{Float64}

Compute Jacobian of predicted field w.r.t. tile coefficients.

For linear harmonic model with coefficients [B0x, B0y, B0z, G5...]:
- B(x) = B0 + G(G5) · (x - x_ref)

Returns 3 × n_coef matrix.
"""
function compute_tile_jacobian(tile::SlamTileState, position::Vec3Map)
    n_coef = tile_state_dim(tile)

    # Use normalized coordinates x̃ = (x - center) / L
    # This makes all Jacobian entries O(1) for in-tile positions,
    # eliminating the condition number pathology from ad-hoc 1e-9 scaling.
    # Coefficients are in normalized units: α_physical = α_normalized / L^k
    # where k is the order (0 for B0, 1 for gradient, 2 for quadratic).
    x̃ = (position - tile.center) / tile.scale

    return field_jacobian(x̃, tile.model_mode)
end

# ============================================================================
# Batch Update (Information Form)
# ============================================================================

"""
    apply_batch_update!(updater::OnlineTileUpdater, tile::SlamTileState,
                        observations::Vector{TileUpdateObservation}) -> TileUpdateResult

Apply batch of observations to tile using information-form fusion.

# Algorithm
Information form is numerically stable for sequential updates:
```
I_batch = Σᵢ wᵢ Hᵢ' Rᵢ⁻¹ Hᵢ
i_batch = Σᵢ wᵢ Hᵢ' Rᵢ⁻¹ zᵢ

I_new = I_old + I_batch
P_new = (I_new + λI)⁻¹
α_new = P_new (I_old α_old + i_batch)
```

# Covariance Shrink Limiting
To prevent artificial rapid convergence (which would violate NEES honesty),
the covariance shrink is limited to max_covariance_shrink_rate per batch.
"""
function apply_batch_update!(updater::OnlineTileUpdater, tile::SlamTileState,
                             observations::Vector{TileUpdateObservation})
    config = updater.config
    n_coef = tile_state_dim(tile)

    # Track rejections
    rejection_reasons = TileUpdateRejection[]
    updates_applied = 0
    max_chi2 = 0.0

    # Determine active subspace from Jacobian structure.
    # Build one Jacobian to discover active_k, then accumulate in k×k only.
    # This makes the updater structurally incapable of producing coupled
    # active↔inactive information — the Scheduler doesn't have to trust us.
    active_k = n_coef  # will be refined from first Jacobian
    active_k_set = false

    # Temporary storage for per-observation data (to process after active_k is known)
    obs_data = Vector{Tuple{Matrix{Float64}, Matrix{Float64}, Float64, Vector{Float64}, Float64}}()

    for obs in observations
        can_apply, reason = can_apply_update(updater, tile, obs)
        if !can_apply
            push!(rejection_reasons, reason)
            continue
        end

        H_full = compute_tile_jacobian(tile, obs.position)
        R_inv = inv(Matrix(obs.R_meas) + config.regularization_strength * LinearAlgebra.I)
        w = obs.weight
        chi2 = dot(obs.field_residual, R_inv * Vector(obs.field_residual))

        # Determine active_k from first valid Jacobian
        if !active_k_set
            # Use informed_jacobian_rank if available (from Scheduler module),
            # otherwise scan for last informative column
            k = _last_informed_column(H_full)
            active_k = k
            active_k_set = true
        end

        push!(obs_data, (H_full, R_inv, w, Vector(obs.field_residual), chi2))
        updates_applied += 1
    end

    if updates_applied == 0
        return failed_update_result(tile, rejection_reasons)
    end

    # Active subspace indices
    active_idx = 1:active_k

    # Accumulate information in active subspace only (k×k)
    I_batch_aa = zeros(active_k, active_k)
    i_batch_a = zeros(active_k)

    for (H_full, R_inv, w, residual, chi2) in obs_data
        H_a = H_full[:, active_idx]  # 3 × k
        I_batch_aa += w * H_a' * R_inv * H_a
        i_batch_a += w * H_a' * R_inv * residual
        max_chi2 = max(max_chi2, chi2)
    end

    # Current active-subspace information
    I_aa_old = tile.information[active_idx, active_idx]
    I_aa_new = I_aa_old + I_batch_aa

    # Regularize for inversion (active subspace only)
    I_aa_reg = I_aa_new + config.regularization_strength * LinearAlgebra.I(active_k)

    # Check conditioning of active subspace (data-informed, not prior-dominated)
    cond_num = cond(I_aa_reg)
    if cond_num > 1e12
        push!(rejection_reasons, REJECT_SINGULAR)
        return failed_update_result(tile, rejection_reasons)
    end

    # Solve in active subspace
    P_aa_new = inv(I_aa_reg)
    i_aa_old = I_aa_old * tile.coefficients[active_idx]
    α_a_new = P_aa_new * (i_aa_old + i_batch_a)

    # Build full result vectors (frozen dims unchanged)
    α_new = copy(tile.coefficients)
    α_new[active_idx] = α_a_new

    P_new = copy(tile.covariance)
    P_new[active_idx, active_idx] = P_aa_new

    # Enforce cross-block zeros in result
    if active_k < n_coef
        inactive_idx = (active_k+1):n_coef
        P_new[active_idx, inactive_idx] .= 0.0
        P_new[inactive_idx, active_idx] .= 0.0
    end

    # Covariance shrink limiting (INV-04 protection) — active subspace only
    old_trace = tr(tile.covariance[active_idx, active_idx])
    new_trace = tr(P_aa_new)
    shrink_ratio = new_trace / max(old_trace, 1e-30)

    if shrink_ratio < config.max_covariance_shrink_rate
        scale = config.max_covariance_shrink_rate / shrink_ratio
        P_new[active_idx, active_idx] *= scale
        shrink_ratio = config.max_covariance_shrink_rate
    end

    return TileUpdateResult(
        true,
        updates_applied,
        length(rejection_reasons),
        rejection_reasons,
        Vector(α_new),
        Matrix(P_new),
        shrink_ratio,
        max_chi2
    )
end

"""Internal: delegate to shared canonical_informed_dim for consistency."""
_last_informed_column(H::Matrix{Float64}) = canonical_informed_dim(H)

"""
    apply_update_result!(tile::SlamTileState, result::TileUpdateResult)

DEPRECATED: Direct tile mutation bypasses active-subspace protections.
Use OnlineSlamScheduler.process_slow_loop! which enforces:
- k×k active-subspace solve (not full n×n)
- Data-quality condition gating
- Cross-block zero invariant

This function is retained for test compatibility only. Production code
must go through the Scheduler.
"""
function apply_update_result!(tile::SlamTileState, result::TileUpdateResult)
    if !haskey(ENV, "NAVCORE_ALLOW_UNSAFE_TILE_MUTATION")
        error("apply_update_result! is deprecated. Use OnlineSlamScheduler for tile updates. " *
              "Set ENV[\"NAVCORE_ALLOW_UNSAFE_TILE_MUTATION\"]=\"1\" for test-only use.")
    end

    if !result.success
        return tile
    end

    tile.coefficients = result.new_coefficients
    tile.covariance = result.new_covariance
    tile.information = inv(result.new_covariance + 1e-12 * I)
    tile.observation_count += result.updates_applied
    tile.version += 1

    # Promote from probationary if enough observations
    if tile.is_probationary && tile.observation_count >= 20
        tile.is_probationary = false
    end

    return tile
end

# ============================================================================
# Smoothness Prior (Optional)
# ============================================================================

"""
    SmoothnessPrior

Prior enforcing smoothness between adjacent tiles.

# Physics
For smooth (source-free) magnetic fields, adjacent tiles should have
continuous field values at their boundaries. This prior penalizes
discontinuities.

# Mathematical Form
For tiles i, j sharing boundary:
J_smooth = λ ||α_i - α_j||² weighted by boundary conditions
"""
struct SmoothnessPrior
    weight::Float64
    boundary_positions::Vector{Vec3Map}  # Positions where continuity enforced
end

"""
    compute_smoothness_information(prior::SmoothnessPrior,
                                   tile_i::SlamTileState, tile_j::SlamTileState)
        -> (I_smooth, i_smooth)

Compute information contribution from smoothness prior between two tiles.

Returns information matrix and vector to be added to tile_i's update.
"""
function compute_smoothness_information(prior::SmoothnessPrior,
                                        tile_i::SlamTileState, tile_j::SlamTileState)
    n_i = tile_state_dim(tile_i)
    n_j = tile_state_dim(tile_j)
    λ = prior.weight

    I_smooth = zeros(n_i, n_i)
    i_smooth = zeros(n_i)

    for pos in prior.boundary_positions
        # Jacobians at boundary point
        H_i = compute_tile_jacobian(tile_i, pos)
        H_j = compute_tile_jacobian(tile_j, pos)

        # Predicted fields at boundary
        B_i = H_i * tile_i.coefficients
        B_j = H_j[1:3, 1:min(n_j, 8)] * tile_j.coefficients[1:min(n_j, 8)]

        # Continuity residual
        residual = B_i - B_j

        # Add to information (penalize discontinuity)
        I_smooth += λ * H_i' * H_i
        i_smooth += λ * H_i' * residual
    end

    return I_smooth, i_smooth
end

# ============================================================================
# Dipole Rejection (Critical Constraint)
# ============================================================================

"""
    is_likely_dipole_signature(residual::Vec3Map, position::Vec3Map,
                               tile::SlamTileState) -> Bool

Check if residual pattern is consistent with unmodeled dipole.

# Physics
Dipole fields have characteristic 1/r³ spatial structure and specific
angular dependence. If residual matches this pattern, it should NOT
be absorbed into tile coefficients.

# Heuristic
- Large residual magnitude compared to tile prediction
- Rapid spatial variation (if multiple nearby measurements available)
"""
function is_likely_dipole_signature(residual::Vec3Map, position::Vec3Map,
                                    tile::SlamTileState;
                                    dipole_threshold::Float64 = 1e-6)
    # Simple heuristic: large residuals are likely dipoles
    residual_mag = norm(residual)

    # Compare to expected tile prediction magnitude
    B_pred_mag = norm(tile.coefficients[1:3])

    # If residual is comparable to or larger than background, likely dipole
    if residual_mag > 0.5 * B_pred_mag
        return true
    end

    # If residual magnitude exceeds threshold, likely dipole
    if residual_mag > dipole_threshold
        return true
    end

    return false
end

# ============================================================================
# Statistics
# ============================================================================

"""
    TileUpdaterStatistics

Statistics for tile updater performance monitoring.
"""
mutable struct TileUpdaterStatistics
    total_observations::Int
    observations_applied::Int
    observations_rejected_teachability::Int
    observations_rejected_outlier::Int
    observations_rejected_weight::Int
    observations_rejected_singular::Int
    total_batches::Int
    successful_batches::Int
    max_chi2_seen::Float64
    min_shrink_ratio::Float64
end

function TileUpdaterStatistics()
    TileUpdaterStatistics(0, 0, 0, 0, 0, 0, 0, 0, 0.0, 1.0)
end

"""Update statistics from batch result."""
function update_statistics!(stats::TileUpdaterStatistics, result::TileUpdateResult)
    stats.total_batches += 1
    stats.total_observations += result.updates_applied + result.updates_rejected
    stats.observations_applied += result.updates_applied

    for reason in result.rejection_reasons
        if reason == REJECT_LOW_TEACHABILITY
            stats.observations_rejected_teachability += 1
        elseif reason == REJECT_OUTLIER
            stats.observations_rejected_outlier += 1
        elseif reason == REJECT_LOW_WEIGHT
            stats.observations_rejected_weight += 1
        elseif reason == REJECT_SINGULAR
            stats.observations_rejected_singular += 1
        end
    end

    if result.success
        stats.successful_batches += 1
        stats.max_chi2_seen = max(stats.max_chi2_seen, result.max_chi2)
        stats.min_shrink_ratio = min(stats.min_shrink_ratio, result.covariance_shrink_ratio)
    end

    return stats
end

"""Compute acceptance rate."""
function acceptance_rate(stats::TileUpdaterStatistics)
    stats.total_observations == 0 ? 0.0 :
        stats.observations_applied / stats.total_observations
end

"""Format statistics for display."""
function format_updater_statistics(stats::TileUpdaterStatistics)
    return """
    Tile Updater Statistics:
      Total observations: $(stats.total_observations)
      Applied: $(stats.observations_applied) ($(round(100*acceptance_rate(stats), digits=1))%)
      Rejected:
        - Teachability: $(stats.observations_rejected_teachability)
        - Outlier: $(stats.observations_rejected_outlier)
        - Low weight: $(stats.observations_rejected_weight)
        - Singular: $(stats.observations_rejected_singular)
      Batches: $(stats.successful_batches)/$(stats.total_batches) successful
      Max χ²: $(round(stats.max_chi2_seen, digits=2))
      Min shrink ratio: $(round(stats.min_shrink_ratio, digits=4))
    """
end

# ============================================================================
# Exports
# ============================================================================

export TIER2_MAX_ACTIVE_DIM, JACOBIAN_COL_NORM_THRESHOLD, canonical_informed_dim
export GRADIENT_MIN_SPAN_M, GRADIENT_MIN_OBS
export OnlineTileUpdaterConfig, DEFAULT_ONLINE_TILE_UPDATER_CONFIG
export TileUpdateObservation
export TileUpdateRejection, REJECT_NONE, REJECT_LOW_TEACHABILITY
export REJECT_OUTLIER, REJECT_LOW_WEIGHT, REJECT_SINGULAR, REJECT_LOW_GRADIENT
export TileUpdateResult
export OnlineTileUpdater, DEFAULT_ONLINE_TILE_UPDATER
export can_apply_update, compute_tile_jacobian
export apply_batch_update!, apply_update_result!
export SmoothnessPrior, compute_smoothness_information
export is_likely_dipole_signature
export TileUpdaterStatistics, update_statistics!, acceptance_rate
export format_updater_statistics
