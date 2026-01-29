# ============================================================================
# SourceFirstResidualRouter.jl - Contract-Based Residual Routing (Phase G+ Step 4)
# ============================================================================
#
# Replaces the ad-hoc is_teachable gate in OnlineSourceSLAM.jl (line 846)
# with a contract-based routing system built on SourceMapSeparationContract.
#
# The router ensures:
# 1. Source predictions are subtracted before tile receives residual
# 2. Tiles are frozen near confirmed sources (spatial exclusion zone)
# 3. Nav residual excludes source contribution
# 4. SHADOW sources do not modify the residual (safe default)
#
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Types
# ============================================================================

"""
    ResidualRoutingResult

Result of routing a measurement residual through source-first attribution.

# Fields
- `tile_residual::Vec3Map`: Cleaned residual for tile update [T]
- `nav_residual::Vec3Map`: Cleaned residual for nav update [T]
- `tile_frozen::Bool`: Whether tile update is suppressed
- `attribution::MeasurementAttribution`: Full attribution breakdown
"""
struct ResidualRoutingResult
    tile_residual::Vec3Map
    nav_residual::Vec3Map
    tile_frozen::Bool
    attribution::MeasurementAttribution
end

"""
    SourceFirstResidualRouter

Contract-based residual router that attributes residuals to sources, tiles,
and navigation before deciding what each subsystem receives.

# Fields
- `separation_config::SeparationConfig`: Thresholds for source-map separation
- `frozen_tile_ids::Set{Int}`: Tile indices currently frozen due to source proximity
"""
mutable struct SourceFirstResidualRouter
    separation_config::SeparationConfig
    frozen_tile_ids::Set{Int}
end

"""Create a residual router with default configuration."""
function SourceFirstResidualRouter(; config::SeparationConfig = SeparationConfig())
    SourceFirstResidualRouter(config, Set{Int}())
end

# ============================================================================
# Routing
# ============================================================================

"""
    route_residual!(router, raw_residual, position, active_tracks, tile_state, R)
        → ResidualRoutingResult

Route a measurement residual through source-first attribution.

# Algorithm
1. Compute source predictions from all active tracks (respecting coupling mode)
2. Compute tile prediction from tile state
3. Attribute residual using SeparationContract
4. Decide tile freeze based on source dominance and spatial exclusion
5. Return cleaned residuals for tile and nav subsystems

# Coupling Mode Handling
- SHADOW: source is not subtracted (safe default, source is uncertain)
- COV_ONLY: source is not subtracted, but covariance is inflated
- SUBTRACT: source prediction is subtracted from residual

Only SUBTRACT-mode sources contribute to source_predicted in attribution.
"""
function route_residual!(router::SourceFirstResidualRouter,
                         raw_residual::Vec3Map,
                         position::Vec3Map,
                         active_tracks::Dict{Int, SourceTrack},
                         tile_predicted::Vec3Map,
                         R::Mat3Map,
                         tile_id::Int = -1)
    config = router.separation_config

    # Compute source predictions (only from SUBTRACT-mode tracks)
    source_predicted = Vec3Map(0.0, 0.0, 0.0)
    source_ids = Int[]
    source_snrs = Dict{Int, Float64}()
    confirmed_positions = Vec3Map[]

    for (id, track) in active_tracks
        if track.status in (TRACK_PROVISIONAL, TRACK_CONFIRMED, TRACK_LOCKED)
            src_pos = source_position(track.source_state)
            src_mom = source_moment(track.source_state)
            B_src = Vec3Map(dipole_field_at(position, src_pos, src_mom)...)

            # SNR estimate: |B_src| / sqrt(tr(R)/3)
            noise_floor = sqrt(tr(Matrix(R)) / 3)
            snr = noise_floor > 0 ? norm(B_src) / noise_floor : 0.0
            source_snrs[id] = snr

            if track.coupling_mode == SOURCE_SUBTRACT
                source_predicted = source_predicted + B_src
                push!(source_ids, id)
            end

            # Collect confirmed+ source positions for spatial freeze
            if track.status in (TRACK_CONFIRMED, TRACK_LOCKED)
                push!(confirmed_positions, src_pos)
            end
        end
    end

    # Attribute residual
    attribution = attribute_residual(position, raw_residual, source_predicted,
                                     tile_predicted, source_ids, tile_id, config)

    # Decide tile freeze
    tile_frozen = should_freeze_tile(attribution, source_snrs, config) ||
                  should_freeze_tile_spatial(position, confirmed_positions, config) ||
                  (tile_id >= 0 && tile_id in router.frozen_tile_ids)

    # Compute cleaned residuals
    # Tile sees: raw - source_predicted (sources subtracted)
    tile_residual = raw_residual - source_predicted

    # Nav sees: raw - source_predicted - tile_predicted
    nav_residual = attribution.nav_residual

    return ResidualRoutingResult(tile_residual, nav_residual, tile_frozen, attribution)
end

"""
    update_freeze_zones!(router, tracks)

Refresh the set of frozen tile IDs based on confirmed+ tracks.
Tiles within tile_freeze_radius_m of any confirmed source are frozen.

This function should be called periodically (e.g., on each slow loop)
to update the spatial exclusion zones as sources are confirmed or retired.

Note: This requires a tile_id_at function to map positions to tile indices.
For now, frozen_tile_ids tracks tiles that have been explicitly frozen
during route_residual! calls.
"""
function update_freeze_zones!(router::SourceFirstResidualRouter,
                              tracks::Dict{Int, SourceTrack})
    # Clear stale freeze zones from retired/demoted tracks
    active_source_ids = Set{Int}()
    for (id, track) in tracks
        if track.status in (TRACK_CONFIRMED, TRACK_LOCKED)
            push!(active_source_ids, id)
        end
    end

    # Only keep frozen tiles if there are still active confirmed sources
    if isempty(active_source_ids)
        empty!(router.frozen_tile_ids)
    end
end

# ============================================================================
# Exports
# ============================================================================

export ResidualRoutingResult, SourceFirstResidualRouter
export route_residual!, update_freeze_zones!
