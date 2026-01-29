# ============================================================================
# SourceCoupledUpdate.jl - Joint Inference (Phase G Step 6)
# ============================================================================
#
# Provides source-coupled residual computation and modified tile/nav updates
# that account for active dipole sources.
#
# Three coupling modes:
# - SHADOW: no modification (baseline)
# - COV_ONLY: inflate innovation covariance
# - SUBTRACT: subtract source field + inflate covariance
#
# Safety: bounded subtraction, no cross-covariance, rollback support.
#
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Types
# ============================================================================

"""
    SourceCoupledResidual

Residual with source coupling applied.

# Fields
- `raw_residual::Vec3Map`: Original residual z - B_bg [T]
- `source_subtracted_residual::Vec3Map`: After subtracting source predictions [T]
- `source_inflated_R::Mat3Map`: Inflated measurement covariance [T²]
- `active_subtractions::Vector{Int}`: Source IDs contributing subtraction
- `active_inflations::Vector{Int}`: Source IDs contributing inflation
"""
struct SourceCoupledResidual
    raw_residual::Vec3Map
    source_subtracted_residual::Vec3Map
    source_inflated_R::Mat3Map
    active_subtractions::Vector{Int}
    active_inflations::Vector{Int}
end

# ============================================================================
# Coupled Residual Computation
# ============================================================================

"""
    compute_source_coupled_residual(z::Vec3Map, B_bg::Vec3Map, R_sensor::Mat3Map,
                                     tracks::Dict{Int, SourceTrack},
                                     config::SourceCouplingConfig) -> SourceCoupledResidual

Compute the source-coupled residual for navigation/map update.

# Logic
For each active track:
- SHADOW: no contribution
- COV_ONLY: add H_src P_src H_src' to R
- SUBTRACT: subtract predicted field, add covariance
"""
function compute_source_coupled_residual(z::Vec3Map, B_bg::Vec3Map, R_sensor::Mat3Map,
                                          position::Vec3Map,
                                          tracks::Dict{Int, SourceTrack},
                                          config::SourceCouplingConfig)
    raw_residual = z - B_bg
    subtracted = Vector{Float64}(raw_residual)
    R_inflated = Matrix{Float64}(R_sensor)
    subtractions = Int[]
    inflations = Int[]

    max_T = config.max_subtraction_field_T

    for (id, track) in tracks
        if track.status in (TRACK_PROVISIONAL, TRACK_CONFIRMED, TRACK_LOCKED)
            mode = track.coupling_mode

            if mode == SOURCE_SHADOW
                continue
            end

            # Compute source field and covariance at measurement position
            B_src = dipole_field_at(position, source_position(track.source_state),
                                    source_moment(track.source_state))
            H_src = compute_source_jacobian(position, track.source_state)
            Σ_src = H_src * Matrix(track.source_state.covariance) * H_src'

            if mode == SOURCE_COV_ONLY || mode == SOURCE_SUBTRACT
                R_inflated += Σ_src
                push!(inflations, id)
            end

            if mode == SOURCE_SUBTRACT
                # Bounded subtraction
                for i in 1:3
                    subtracted[i] -= clamp(B_src[i], -max_T, max_T)
                end
                push!(subtractions, id)
            end
        end
    end

    # Add diagonal inflation for COV_ONLY
    if !isempty(inflations) && isempty(subtractions)
        R_inflated += config.nav_covariance_inflation_T2 * I(3)
    end

    return SourceCoupledResidual(
        raw_residual,
        Vec3Map(subtracted...),
        Mat3Map(R_inflated),
        subtractions,
        inflations
    )
end

# ============================================================================
# Coupled Tile Update
# ============================================================================

"""
    source_coupled_tile_update!(tile::SlamTileState, coupled::SourceCoupledResidual,
                                 position::Vec3Map, updater_config::OnlineTileUpdaterConfig)
        -> Union{TileUpdateResult, Nothing}

Apply tile update using source-coupled residual.
Sources are subtracted so tile sees only background field residual.
"""
function source_coupled_tile_update!(tile::SlamTileState, coupled::SourceCoupledResidual,
                                      position::Vec3Map, updater_config::OnlineTileUpdaterConfig)
    # Create tile update observation with source-subtracted residual
    obs = TileUpdateObservation(
        position,
        coupled.source_subtracted_residual,
        coupled.source_inflated_R,
        0.1,  # pose_uncertainty (assume good)
        0.0,  # timestamp
        1.0   # weight
    )

    # Delegate to existing tile updater
    updater = OnlineTileUpdater(updater_config)
    return apply_batch_update!(updater, tile, [obs])
end

# ============================================================================
# Coupled Nav Update
# ============================================================================

"""
    source_coupled_nav_update!(nav_state, nav_P::Matrix{Float64},
                                coupled::SourceCoupledResidual,
                                H_nav::Matrix{Float64},
                                config::SourceCouplingConfig)
        -> (updated_state, updated_P, chi2)

Apply navigation state update using source-coupled residual.

# Algorithm
Standard EKF with inflated covariance:
1. S = H_nav P H_nav' + R_inflated
2. K = P H_nav' S⁻¹
3. x += K * residual_coupled
4. P = (I - KH) P (I - KH)' + K R K' (Joseph form)
"""
function source_coupled_nav_update!(nav_P::Matrix{Float64},
                                     coupled::SourceCoupledResidual,
                                     H_nav::Matrix{Float64},
                                     config::SourceCouplingConfig)
    residual = coupled.source_subtracted_residual
    R = Matrix{Float64}(coupled.source_inflated_R)

    S = H_nav * nav_P * H_nav' + R
    S_reg = S + 1e-20 * I(3)

    chi2 = Vector(residual)' * (S_reg \ Vector(residual))

    K = nav_P * H_nav' * inv(S_reg)

    # State correction
    dx = K * Vector(residual)

    # Covariance update (Joseph form)
    n = size(nav_P, 1)
    I_KH = Matrix{Float64}(I(n)) - K * H_nav
    P_new = I_KH * nav_P * I_KH' + K * R * K'

    return (dx, P_new, chi2)
end

# ============================================================================
# Phase G+ Step 5: Adaptive Covariance Inflation
# ============================================================================

"""
    compute_adaptive_inflation(track::SourceTrack, position::Vec3Map,
                               R_sensor::Mat3Map) → Mat3Map

Compute adaptive inflation matrix H_src P_src H_src^T for a single source.

Instead of a fixed scalar inflation, this propagates the source's actual
parameter covariance through the field Jacobian to get the field-space
uncertainty contribution. As the source converges (P_src → 0), the
inflation naturally vanishes.

Physics: The inflation represents how much the measurement residual
could be affected by source parameter uncertainty. This is the correct
Bayesian treatment — marginalizing over source state uncertainty.

# Returns
3×3 PSD matrix representing source-induced field uncertainty [T²].
Bounded by max_inflation_factor × R_sensor to prevent numerical issues.
"""
function compute_adaptive_inflation(track::SourceTrack, position::Vec3Map,
                                     R_sensor::Mat3Map;
                                     max_inflation_factor::Float64 = 10.0)
    H_src = compute_source_jacobian(position, track.source_state)  # 3×6
    P_src = Matrix(track.source_state.covariance)  # 6×6
    Σ_inflation = H_src * P_src * H_src'  # 3×3

    # Bound inflation to prevent numerical issues
    R_diag = tr(Matrix(R_sensor)) / 3.0
    max_entry = max_inflation_factor * R_diag
    for i in 1:3
        Σ_inflation[i, i] = min(Σ_inflation[i, i], max_entry)
    end

    return Mat3Map(Σ_inflation)
end

"""
    apply_adaptive_inflation!(S::Matrix{Float64}, active_tracks::Dict{Int, SourceTrack},
                               position::Vec3Map, R_sensor::Mat3Map;
                               max_inflation_factor::Float64 = 10.0) → Mat3Map

Apply adaptive covariance inflation from all COV_ONLY and SUBTRACT tracks.

Accumulates H_src P_src H_src^T from each active track with coupling mode
≥ COV_ONLY. This replaces the fixed nav_covariance_inflation_T2 with a
physically motivated, state-dependent inflation.

Compatible with existing apply_source_coupling! interface — the returned
matrix can be added to the innovation covariance S.

# Returns
3×3 total inflation matrix [T²].
"""
function apply_adaptive_inflation!(S::Matrix{Float64},
                                    active_tracks::Dict{Int, SourceTrack},
                                    position::Vec3Map,
                                    R_sensor::Mat3Map;
                                    max_inflation_factor::Float64 = 10.0)
    total_inflation = zeros(3, 3)

    for (id, track) in active_tracks
        if track.status in (TRACK_PROVISIONAL, TRACK_CONFIRMED, TRACK_LOCKED)
            if track.coupling_mode in (SOURCE_COV_ONLY, SOURCE_SUBTRACT)
                Σ_src = compute_adaptive_inflation(track, position, R_sensor;
                                                    max_inflation_factor=max_inflation_factor)
                total_inflation .+= Matrix(Σ_src)
            end
        end
    end

    S .+= total_inflation
    return Mat3Map(total_inflation)
end

# ============================================================================
# Exports
# ============================================================================

export SourceCoupledResidual
export compute_source_coupled_residual
export source_coupled_tile_update!, source_coupled_nav_update!
export compute_adaptive_inflation, apply_adaptive_inflation!
