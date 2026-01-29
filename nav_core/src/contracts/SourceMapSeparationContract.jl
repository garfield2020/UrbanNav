# Source-Map Separation Contract
# Formal contract for residual attribution between sources and tiles.
# Ensures nav-safe partitioning: source predictions are subtracted before tile learning,
# and tiles are frozen when source dominance exceeds threshold.

using LinearAlgebra
using StaticArrays

"""
    ResidualOwner

Who owns a measurement residual after attribution.
"""
@enum ResidualOwner begin
    OWNER_NAV = 1            # Navigation state error
    OWNER_TILE = 2           # Tile model error
    OWNER_SOURCE = 3         # Source model error
    OWNER_UNATTRIBUTED = 4   # Cannot be attributed
end

"""
    MeasurementAttribution

Full attribution of a single measurement residual to sources, tile, and navigation.
Invariant: raw_residual ≈ source_predicted + tile_predicted + nav_residual
"""
struct MeasurementAttribution
    position::Vec3Map
    raw_residual::Vec3Map
    source_predicted::Vec3Map        # sum of active source predictions
    tile_predicted::Vec3Map          # tile model prediction
    nav_residual::Vec3Map            # raw - source - tile
    owner::ResidualOwner
    source_ids::Vector{Int}          # contributing source track IDs
    tile_id::Int                     # tile index (-1 if none)
    source_fraction::Float64         # |source_predicted| / |raw_residual|
end

"""
    SeparationConfig

Thresholds for source-map separation decisions.
- source_dominance_threshold: fraction of residual explained by sources above which
  tile learning is frozen. 0.5 chosen as the point where source model explains
  more variance than background, making tile learning unreliable.
- tile_freeze_radius_m: spatial exclusion zone around confirmed sources. 15m is
  ~3× typical standoff distance, ensuring tile doesn't absorb source signal in
  the near-field where dipole gradient is steep.
- min_source_snr_for_freeze: SNR threshold (ratio of source signal to noise floor)
  to trigger freeze. 3.0 corresponds to ~99.7% confidence that signal is real
  (3σ detection threshold).
"""
struct SeparationConfig
    source_dominance_threshold::Float64   # 0.5
    tile_freeze_radius_m::Float64         # 15.0
    min_source_snr_for_freeze::Float64    # 3.0
end

SeparationConfig() = SeparationConfig(0.5, 15.0, 3.0)

"""
    attribute_residual(raw, source_pred, tile_pred, config) → MeasurementAttribution

Attribute a raw residual to source, tile, and nav components.
The nav_residual is the remainder after subtracting source and tile predictions.
Owner is determined by which component explains the largest fraction.
"""
function attribute_residual(position::Vec3Map, raw::Vec3Map, source_pred::Vec3Map,
                           tile_pred::Vec3Map, source_ids::Vector{Int},
                           tile_id::Int, config::SeparationConfig)
    nav_residual = raw - source_pred - tile_pred
    raw_norm = norm(raw)

    if raw_norm < 1e-15  # Effectively zero residual
        source_fraction = 0.0
        owner = OWNER_UNATTRIBUTED
    else
        source_fraction = norm(source_pred) / raw_norm
        tile_fraction = norm(tile_pred) / raw_norm
        nav_fraction = norm(nav_residual) / raw_norm

        if source_fraction >= tile_fraction && source_fraction >= nav_fraction
            owner = OWNER_SOURCE
        elseif tile_fraction >= nav_fraction
            owner = OWNER_TILE
        else
            owner = OWNER_NAV
        end
    end

    return MeasurementAttribution(
        position, raw, source_pred, tile_pred, nav_residual,
        owner, source_ids, tile_id, source_fraction
    )
end

"""
    should_freeze_tile(attribution, source_snrs, config) → Bool

Determine whether tile update should be suppressed for this measurement.
Tile is frozen when:
1. Source fraction exceeds dominance threshold (source explains most of the residual), AND
2. At least one contributing source has SNR above the freeze threshold.

This prevents tile learning from absorbing source signal, which would corrupt
the background model and reduce source localization accuracy.
"""
function should_freeze_tile(attribution::MeasurementAttribution,
                           source_snrs::Dict{Int,Float64},
                           config::SeparationConfig)::Bool
    # Check source dominance
    if attribution.source_fraction < config.source_dominance_threshold
        return false
    end

    # Check SNR threshold - at least one contributing source must be significant
    for sid in attribution.source_ids
        snr = get(source_snrs, sid, 0.0)
        if snr >= config.min_source_snr_for_freeze
            return true
        end
    end

    return false
end

"""
    should_freeze_tile_spatial(position, source_positions, config) → Bool

Check spatial exclusion zone. Tile is frozen if the measurement position
is within tile_freeze_radius_m of any confirmed source position.
"""
function should_freeze_tile_spatial(position::Vec3Map,
                                   source_positions::Vector{Vec3Map},
                                   config::SeparationConfig)::Bool
    for src_pos in source_positions
        if norm(position - src_pos) < config.tile_freeze_radius_m
            return true
        end
    end
    return false
end

"""
    partition_residual(raw, source_pred, tile_pred, config) → (source_residual, tile_residual, nav_residual)

Conservative partition: each component's residual is bounded by the raw residual in norm.
- source_residual: raw - tile_pred (what sources must explain)
- tile_residual: raw - source_pred (what tile must explain)
- nav_residual: raw - source_pred - tile_pred (leftover for nav)

Conservative guarantee: ‖nav_residual‖ ≤ ‖raw‖ when source and tile predictions
are not adversarial (i.e., they don't amplify the residual).
"""
function partition_residual(raw::Vec3Map, source_pred::Vec3Map, tile_pred::Vec3Map,
                           config::SeparationConfig)
    source_residual = raw - tile_pred
    tile_residual = raw - source_pred
    nav_residual = raw - source_pred - tile_pred
    return (source_residual, tile_residual, nav_residual)
end
