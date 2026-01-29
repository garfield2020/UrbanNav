# ============================================================================
# Tier2EfficacyMetrics.jl - Quadratic Value Measurement (Phase F Step 4)
# ============================================================================
#
# Purpose: Quantify the benefit of Tier-2 (d=15 quadratic) map learning
# relative to Tier-1 (d=8 linear) baseline, enabling evidence-based
# decisions about when quadratic enrichment improves navigation.
#
# Metrics:
# 1. χ² reduction: How much does d=15 reduce measurement residuals
#    compared to d=8? Measured on held-out data to avoid overfitting.
#
# 2. RMSE improvement: Does d=15 improve position accuracy in navigation?
#
# 3. Unlock coverage: What fraction of tiles have tier2_active = true?
#
# Evaluation scopes:
# - SCOPE_UNLOCKED: Only tiles where tier2_active = true
# - SCOPE_BOUNDARY: Tiles at unlock/lock boundary
# - SCOPE_GLOBAL: All tiles (including locked)
#
# Physics:
# χ² = r' Σ⁻¹ r where r = z - B_pred is the field prediction residual.
# For d=8: r₈ = z - H₈α₈
# For d=15: r₁₅ = z - H₁₅α₁₅
# If quadratic terms capture real field curvature, r₁₅ < r₈ and χ² drops.
#
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Types
# ============================================================================

"""
    EvaluationScope

Scope for Tier-2 efficacy evaluation.

- `SCOPE_UNLOCKED`: Only tiles with tier2_active = true.
  Measures the benefit where quadratic is actually used.
- `SCOPE_BOUNDARY`: Tiles that have been unlocked/relocked at least once.
  Identifies borderline cases for gate calibration.
- `SCOPE_GLOBAL`: All tiles including locked ones.
  Measures system-wide impact (diluted by tiles where d=15 is inactive).
"""
@enum EvaluationScope begin
    SCOPE_UNLOCKED = 1
    SCOPE_BOUNDARY = 2
    SCOPE_GLOBAL = 3
end

"""
    Tier2EfficacyReport

Summary of d=15 vs d=8 comparison for a set of tiles.

# Fields
- `scope::EvaluationScope`: Which tiles were included
- `chi2_linear::Float64`: Mean χ² per observation using d=8 prediction
- `chi2_quadratic::Float64`: Mean χ² per observation using d=15 prediction
- `chi2_reduction_pct::Float64`: 100 × (1 - chi2_quadratic / chi2_linear) [%]
- `segment_rmse_linear::Float64`: Position RMSE from linear segments [m]
- `segment_rmse_quadratic::Float64`: Position RMSE from quadratic segments [m]
- `rmse_improvement_pct::Float64`: 100 × (1 - rmse_quad / rmse_linear) [%]
- `unlock_coverage_pct::Float64`: Fraction of tiles with tier2_active [%]
- `n_tiles::Int`: Number of tiles evaluated
- `n_obs::Int`: Number of observations evaluated

# Interpretation
- `chi2_reduction_pct > 0`: Quadratic model fits data better
- `chi2_reduction_pct ≥ 20`: Phase F exit criterion (strong local benefit)
- `rmse_improvement_pct ≥ 10`: Phase F exit criterion (mission-level benefit)
- Negative values indicate regression (quadratic is worse)
"""
struct Tier2EfficacyReport
    scope::EvaluationScope
    chi2_linear::Float64
    chi2_quadratic::Float64
    chi2_reduction_pct::Float64
    segment_rmse_linear::Float64
    segment_rmse_quadratic::Float64
    rmse_improvement_pct::Float64
    unlock_coverage_pct::Float64
    n_tiles::Int
    n_obs::Int
end

# ============================================================================
# Probe Data Structure
# ============================================================================

"""
    TileProbe

A single measurement probe for evaluating tile prediction quality.

# Fields
- `position::Vec3Map`: World position of measurement [m]
- `field_obs::Vec3Map`: Observed magnetic field [T]
- `R::Mat3Map`: Measurement noise covariance [T²]

Probes are typically held-out measurements not used for tile learning.
"""
struct TileProbe
    position::Vec3Map
    field_obs::Vec3Map
    R::Mat3Map
end

# ============================================================================
# Core Functions
# ============================================================================

"""
    evaluate_tier2_efficacy(tiles, probes_by_tile, scope) -> Tier2EfficacyReport

Evaluate the efficacy of Tier-2 (d=15) vs Tier-1 (d=8) across tiles.

# Arguments
- `tiles::Vector{SlamTileState}`: Tiles to evaluate
- `probes_by_tile::Dict{MapTileID, Vector{TileProbe}}`: Held-out probes per tile.
  Each probe contains an observed field measurement and its noise covariance.
- `scope::EvaluationScope`: Which tiles to include

# Returns
`Tier2EfficacyReport` with χ² and RMSE comparisons.

# Algorithm
For each tile matching the scope filter:
1. Predict field at each probe using d=8 coefficients (α[1:8]) → B₈
2. Predict field at each probe using d=15 coefficients (α[1:15]) → B₁₅
3. Compute χ² for each: χ²_k = (z - B_k)' R⁻¹ (z - B_k)
4. Aggregate mean χ² across all probes

RMSE fields are set to 0.0 (populated by integration test harness that
has access to truth positions; this module operates on field residuals only).

# Filtering
- SCOPE_UNLOCKED: tile.tier2_active == true
- SCOPE_BOUNDARY: tile.tier2_relock_count > 0
- SCOPE_GLOBAL: all tiles
"""
function evaluate_tier2_efficacy(tiles::Vector{SlamTileState},
                                  probes_by_tile::Dict{MapTileID, Vector{TileProbe}},
                                  scope::EvaluationScope)
    total_chi2_linear = 0.0
    total_chi2_quad = 0.0
    total_obs = 0
    n_tiles_eval = 0

    n_unlocked = count(t -> t.tier2_active, tiles)
    n_total = length(tiles)

    for tile in tiles
        # Scope filter
        if scope == SCOPE_UNLOCKED && !tile.tier2_active
            continue
        end
        if scope == SCOPE_BOUNDARY && tile.tier2_relock_count == 0
            continue
        end

        probes = get(probes_by_tile, tile.tile_id, TileProbe[])
        if isempty(probes)
            continue
        end

        n_tiles_eval += 1
        frame = TileLocalFrame(tile.center, tile.scale)
        n_coef = length(tile.coefficients)

        for probe in probes
            x̃ = normalize_position(frame, probe.position)
            R_inv = inv(Matrix{Float64}(probe.R))

            # d=8 prediction (linear only)
            B_linear = evaluate_field(tile.coefficients, x̃, MODE_LINEAR)
            r_linear = Vector{Float64}(probe.field_obs - B_linear)
            chi2_lin = dot(r_linear, R_inv * r_linear)

            # d=15 prediction (quadratic, if coefficients available)
            if n_coef >= 15
                B_quad = evaluate_field(tile.coefficients, x̃, MODE_QUADRATIC)
            else
                B_quad = B_linear  # Fall back to linear if no quadratic coefficients
            end
            r_quad = Vector{Float64}(probe.field_obs - B_quad)
            chi2_q = dot(r_quad, R_inv * r_quad)

            total_chi2_linear += chi2_lin
            total_chi2_quad += chi2_q
            total_obs += 1
        end
    end

    # Compute summary
    if total_obs > 0
        mean_chi2_lin = total_chi2_linear / total_obs
        mean_chi2_q = total_chi2_quad / total_obs
        chi2_reduction = mean_chi2_lin > 0 ? 100.0 * (1.0 - mean_chi2_q / mean_chi2_lin) : 0.0
    else
        mean_chi2_lin = 0.0
        mean_chi2_q = 0.0
        chi2_reduction = 0.0
    end

    coverage_pct = n_total > 0 ? 100.0 * n_unlocked / n_total : 0.0

    return Tier2EfficacyReport(
        scope,
        mean_chi2_lin,
        mean_chi2_q,
        chi2_reduction,
        0.0,   # segment_rmse_linear (populated by integration harness)
        0.0,   # segment_rmse_quadratic
        0.0,   # rmse_improvement_pct
        coverage_pct,
        n_tiles_eval,
        total_obs
    )
end

"""
    unlock_coverage(tiles) -> Float64

Fraction of tiles with tier2_active = true.

# Returns
Value in [0, 1]. Returns 0.0 if no tiles provided.
"""
function unlock_coverage(tiles::Vector{SlamTileState})
    n = length(tiles)
    n == 0 && return 0.0
    return count(t -> t.tier2_active, tiles) / n
end

# ============================================================================
# Exports
# ============================================================================

export EvaluationScope, SCOPE_UNLOCKED, SCOPE_BOUNDARY, SCOPE_GLOBAL
export Tier2EfficacyReport, TileProbe
export evaluate_tier2_efficacy, unlock_coverage
