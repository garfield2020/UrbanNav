# ============================================================================
# OnlineMapProvider.jl - Online Map Query with Uncertainty (Phase C Step 4)
# ============================================================================
#
# Provides map queries from the online SLAM state, including:
# - Background field from tile coefficients
# - Source contributions from tracked dipoles
# - Full uncertainty (Σ_map) for honest filtering
#
# Key Invariant (INV-04):
# Σ_total = H·P·H' + R_sensor + Σ_map + Q_model
#
# The map is NO LONGER static - it has uncertainty that must propagate
# through the measurement update to maintain NEES honesty.
#
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Online Map Query Configuration
# ============================================================================

"""
    OnlineMapQueryConfig

Configuration for online map queries.

# Fields
- `include_sources::Bool`: Include source (dipole) contributions in prediction
- `source_cutoff_distance::Float64`: Distance beyond which sources are ignored [m]
- `extrapolation_penalty::Float64`: Uncertainty multiplier outside coverage
- `min_tile_observations::Int`: Minimum observations before tile is usable
- `probationary_penalty::Float64`: Uncertainty multiplier for probationary tiles

# Physics Justification
- source_cutoff_distance: 50m default; dipole 1/r³ fall-off means negligible beyond this
- extrapolation_penalty: 10× uncertainty when extrapolating
- min_tile_observations: 10 observations for stable coefficient estimate
"""
struct OnlineMapQueryConfig
    include_sources::Bool
    source_cutoff_distance::Float64
    extrapolation_penalty::Float64
    min_tile_observations::Int
    probationary_penalty::Float64

    function OnlineMapQueryConfig(;
        include_sources::Bool = true,
        source_cutoff_distance::Float64 = 50.0,
        extrapolation_penalty::Float64 = 10.0,
        min_tile_observations::Int = 10,
        probationary_penalty::Float64 = 3.0
    )
        @assert source_cutoff_distance > 0 "source_cutoff_distance must be positive"
        @assert extrapolation_penalty >= 1.0 "extrapolation_penalty must be >= 1"
        @assert min_tile_observations >= 1 "min_tile_observations must be >= 1"
        @assert probationary_penalty >= 1.0 "probationary_penalty must be >= 1"

        new(include_sources, source_cutoff_distance, extrapolation_penalty,
            min_tile_observations, probationary_penalty)
    end
end

"""Default online map query configuration."""
const DEFAULT_ONLINE_MAP_QUERY_CONFIG = OnlineMapQueryConfig()

"""
    is_tier2_eligible(tile::SlamTileState) -> Bool

Check whether a tile currently meets Tier-2 spatial diversity requirements.
Used by both Scheduler (unlock gate) and Provider (prediction gate).
Single source of truth prevents solve/prediction desynchronization.

# Gates
- XY span ≥ TIER2_MIN_SPAN_M (45m): quadratic terms need large spatial baseline
- Both X AND Y span ≥ TIER2_MIN_SPAN_BOTH (22.5m): 2D excitation required
- Observation count ≥ TIER2_MIN_OBS (120): 8× margin for 15D solve
"""
function is_tier2_eligible(tile::SlamTileState)
    span = tile.position_bbox_max - tile.position_bbox_min
    span_xy = max(span[1], span[2])
    span_both = min(span[1], span[2])
    return span_xy >= TIER2_MIN_SPAN_M &&
           span_both >= TIER2_MIN_SPAN_BOTH &&
           tile.observation_count >= TIER2_MIN_OBS
end

# ============================================================================
# Online Map Query Result (Extended)
# ============================================================================

"""
    OnlineMapQueryResult

Extended query result with full uncertainty decomposition.

# Fields
## Predictions
- `B_pred::Vec3Map`: Total predicted field [T]
- `G_pred::Mat3Map`: Predicted gradient tensor [T/m]
- `B_background::Vec3Map`: Background field from tile [T]
- `B_sources::Vec3Map`: Field from tracked sources [T]

## Uncertainties (for Σ_total composition)
- `Σ_B::Mat3Map`: Total field covariance [T²]
- `Σ_G::SMatrix{5,5}`: Gradient covariance (packed) [T²/m²]
- `Σ_B_tile::Mat3Map`: Tile contribution to field covariance [T²]
- `Σ_B_sources::Mat3Map`: Source contribution to field covariance [T²]

## Metadata
- `in_coverage::Bool`: Position is within map coverage
- `tile_id::MapTileID`: Tile providing background prediction
- `n_sources_included::Int`: Number of sources in prediction
- `tile_quality::Float64`: Quality metric for tile [0-1]

# Usage in EKF Update
```julia
# Measurement covariance (INV-04: Σ_total closure)
S = H * P * H' + R_sensor + result.Σ_B + Q_model
```
"""
struct OnlineMapQueryResult
    # Predictions
    B_pred::Vec3Map
    G_pred::Mat3Map
    B_background::Vec3Map
    B_sources::Vec3Map

    # Uncertainties
    Σ_B::Mat3Map
    Σ_G::SMatrix{5, 5, Float64, 25}
    Σ_B_tile::Mat3Map
    Σ_B_sources::Mat3Map

    # Metadata
    in_coverage::Bool
    tile_id::MapTileID
    n_sources_included::Int
    tile_quality::Float64
end

"""Convert to standard MapQueryResult (for interface compatibility)."""
function to_map_query_result(r::OnlineMapQueryResult)
    MapQueryResult(
        r.B_pred,
        r.G_pred,
        r.Σ_B,
        r.Σ_G,
        SMatrix{3, 5}(zeros(3, 5)),  # No cross-covariance in simplified form
        r.in_coverage,
        r.tile_id
    )
end

# ============================================================================
# Online Map Provider
# ============================================================================

"""
    OnlineMapProvider <: AbstractMapProvider

Map provider that queries the online SLAM state.

Unlike FrozenFileMapProvider, this provider:
1. Queries mutable tile states that change during mission
2. Includes source (dipole) contributions in predictions
3. Propagates full uncertainty including tile and source covariances

# Architecture
The provider is STATELESS - it queries the SlamAugmentedState passed to it.
This ensures:
- Determinism: same state + position → same result
- Thread safety: no internal mutation
- Testability: can mock state for unit tests

# Usage
```julia
provider = OnlineMapProvider()
result = query_online_map(provider, slam_state, position)

# Use in EKF
Σ_map = result.Σ_B  # Map contribution to measurement covariance
```
"""
struct OnlineMapProvider <: AbstractMapProvider
    config::OnlineMapQueryConfig
    tile_size::Float64  # Tile size for spatial indexing [m]
end

function OnlineMapProvider(;
    config::OnlineMapQueryConfig = DEFAULT_ONLINE_MAP_QUERY_CONFIG,
    tile_size::Float64 = 50.0
)
    OnlineMapProvider(config, tile_size)
end

"""Default online map provider."""
const DEFAULT_ONLINE_MAP_PROVIDER = OnlineMapProvider()

# ============================================================================
# Core Query Function
# ============================================================================

"""
    query_online_map(provider::OnlineMapProvider, state::SlamAugmentedState,
                     position::AbstractVector) -> OnlineMapQueryResult

Query the online map at a position.

# Returns
Full prediction with uncertainty decomposition suitable for NEES-honest filtering.

# Algorithm
1. Find relevant tile from SlamAugmentedState
2. Evaluate background field and gradient from tile coefficients
3. Propagate tile coefficient uncertainty to field uncertainty
4. If sources enabled, add contributions from nearby tracked sources
5. Propagate source state uncertainty to field uncertainty
6. Apply quality/coverage penalties as appropriate
"""
function query_online_map(provider::OnlineMapProvider, state::SlamAugmentedState,
                          position::AbstractVector)
    pos = Vec3Map(position...)
    config = provider.config

    # Default values for no-coverage case
    B_background = Vec3Map(0.0, 0.0, 0.0)
    B_sources = Vec3Map(0.0, 0.0, 0.0)
    G_pred = Mat3Map(zeros(3, 3))
    Σ_B_tile = Mat3Map(1e-6 * I)  # 1000 nT default uncertainty
    Σ_G = SMatrix{5, 5}(1e-12 * I)  # Default gradient uncertainty
    Σ_B_sources = Mat3Map(zeros(3, 3))
    in_coverage = false
    tile_id = MapTileID(0, 0)
    tile_quality = 0.0
    n_sources_included = 0

    # Query tile
    tile_id = tile_id_at(pos, provider.tile_size)
    tile = get_tile_state(state, tile_id)

    if tile !== nothing
        # Evaluate background field from tile
        B_background, G_pred, Σ_B_tile, Σ_G = evaluate_tile_prediction(tile, pos, config)
        in_coverage = true
        tile_quality = compute_tile_quality_metric(tile, config)
    end

    # Query sources
    if config.include_sources
        B_sources, Σ_B_sources, n_sources_included = evaluate_source_contributions(
            state.source_states, pos, config
        )
    end

    # Total prediction and uncertainty
    B_pred = B_background + B_sources
    Σ_B = Mat3Map(Matrix(Σ_B_tile) + Matrix(Σ_B_sources))

    return OnlineMapQueryResult(
        B_pred, G_pred, B_background, B_sources,
        Σ_B, Σ_G, Σ_B_tile, Σ_B_sources,
        in_coverage, tile_id, n_sources_included, tile_quality
    )
end

# ============================================================================
# Tile Prediction
# ============================================================================

"""
    evaluate_tile_prediction(tile::SlamTileState, position::Vec3Map,
                             config::OnlineMapQueryConfig)
        -> (B_pred, G_pred, Σ_B, Σ_G)

Evaluate tile prediction with uncertainty propagation.

# Physics
For linear harmonic model:
- B(x) = B₀ + G₀·(x - x_ref)
- Σ_B = Σ_B₀ + J_G·Σ_G₅·J_G'

where J_G is the Jacobian of field w.r.t. packed gradient coefficients.
"""
function evaluate_tile_prediction(tile::SlamTileState, position::Vec3Map,
                                  config::OnlineMapQueryConfig)
    n_coef = tile_state_dim(tile)
    coef = tile.coefficients
    Σ_coef = tile.covariance

    # Ensure we have at least 8 coefficients (3 field + 5 gradient)
    if n_coef < 8
        # Pad with zeros if needed
        coef = vcat(coef, zeros(8 - n_coef))
        Σ_coef_padded = zeros(8, 8)
        Σ_coef_padded[1:n_coef, 1:n_coef] = Σ_coef
        Σ_coef = Σ_coef_padded
    end

    # Normalized coordinates: x̃ = (x - center) / L
    # Makes all basis terms O(1), eliminating ad-hoc 1e-9 scaling.
    # Coefficients are in normalized units [T]; convert to physical via /L^k.
    x̃ = Vec3Map((position - tile.center) / tile.scale)

    # Determine active mode based on prediction gates
    span = tile.position_bbox_max - tile.position_bbox_min
    span_xy = max(span[1], span[2])
    use_gradient = span_xy >= GRADIENT_MIN_SPAN_M && tile.observation_count >= GRADIENT_MIN_OBS
    use_quadratic = n_coef >= 15 && tile.tier2_active && is_tier2_eligible(tile)

    # Select prediction mode based on gates
    if use_quadratic
        pred_mode = MODE_QUADRATIC
    elseif use_gradient
        pred_mode = MODE_LINEAR
    else
        pred_mode = MODE_B0
    end

    # Field prediction using MapBasisContract (single source of truth)
    B_pred = evaluate_field(coef, x̃, pred_mode)

    # Gradient prediction (from linear coefficients in normalized coords)
    if use_gradient
        G5 = SVector{5}(coef[4:8]...)
        G0 = unpack_gradient(G5)
    else
        G0 = Mat3Map(zeros(3, 3))
    end
    G_pred = G0

    # Uncertainty propagation: Σ_B = H · Σ_coef · H'
    # where H is the Jacobian for the active prediction mode
    n_active = n_coefficients(pred_mode)
    H_pred = field_jacobian(x̃, pred_mode)
    Σ_active = Σ_coef[1:n_active, 1:n_active]
    Σ_B = Mat3Map((H_pred * Σ_active * H_pred')...)

    # Gradient covariance (for downstream gradient observations)
    Σ_G5 = SMatrix{5, 5}(Σ_coef[4:8, 4:8]...)

    # Apply penalties
    penalty = 1.0
    if tile.is_probationary
        penalty *= config.probationary_penalty^2
    end
    if tile.observation_count < config.min_tile_observations
        penalty *= (config.min_tile_observations / max(tile.observation_count, 1))
    end

    if penalty > 1.0
        Σ_B = Mat3Map((penalty * Matrix(Σ_B))...)
        Σ_G5 = SMatrix{5, 5}(penalty * Matrix(Σ_G5))
    end

    return B_pred, G_pred, Σ_B, Σ_G5
end

"""
    compute_tile_quality_metric(tile::SlamTileState, config::OnlineMapQueryConfig) -> Float64

Compute quality metric for tile [0-1].

Based on:
- Observation count (more = higher quality)
- Covariance magnitude (lower = higher quality)
- Probationary status (probationary = lower quality)
"""
function compute_tile_quality_metric(tile::SlamTileState, config::OnlineMapQueryConfig)
    # Observation-based quality
    obs_quality = min(1.0, tile.observation_count / (2 * config.min_tile_observations))

    # Covariance-based quality (inverse of trace normalized)
    cov_trace = tr(tile.covariance)
    cov_quality = exp(-cov_trace / 1e-12)  # Normalized by typical good covariance

    # Probationary penalty
    prob_factor = tile.is_probationary ? 0.5 : 1.0

    return obs_quality * cov_quality * prob_factor
end

# ============================================================================
# Source Contributions
# ============================================================================

"""
    evaluate_source_contributions(sources::Vector{SlamSourceState}, position::Vec3Map,
                                  config::OnlineMapQueryConfig)
        -> (B_sources, Σ_B_sources, n_included)

Evaluate field contributions from tracked sources.

# Physics
Each source is a magnetic dipole:
B_dipole(r) = (μ₀/4π) [3(m·r̂)r̂ - m] / |r|³

Uncertainty propagated from source state [position, moment] to field.
"""
function evaluate_source_contributions(sources::Vector{SlamSourceState}, position::Vec3Map,
                                       config::OnlineMapQueryConfig)
    B_total = Vec3Map(0.0, 0.0, 0.0)
    Σ_total = zeros(3, 3)
    n_included = 0

    for src in sources
        # Skip non-active sources
        if src.lifecycle != :active
            continue
        end

        # Distance check
        src_pos = source_position(src)
        r_vec = position - src_pos
        r_mag = norm(r_vec)

        if r_mag > config.source_cutoff_distance || r_mag < 0.1  # Avoid singularity
            continue
        end

        # Compute dipole field
        dipole_state = to_dipole_state(src)
        B_dipole = Vec3Map(feature_field(position, dipole_state)...)
        B_total = B_total + B_dipole

        # Propagate uncertainty from source state to field
        J_source = compute_source_field_jacobian(position, src)
        Σ_source = Matrix(src.covariance)
        Σ_field_from_source = J_source * Σ_source * J_source'

        Σ_total += Σ_field_from_source
        n_included += 1
    end

    return B_total, Mat3Map(Σ_total...), n_included
end

"""
    compute_source_field_jacobian(position::Vec3Map, src::SlamSourceState) -> Matrix{Float64}

Compute Jacobian of dipole field w.r.t. source state [position, moment].

Returns 3×6 matrix: ∂B/∂[px, py, pz, mx, my, mz]
"""
function compute_source_field_jacobian(position::Vec3Map, src::SlamSourceState)
    # Use feature_field_jacobian from dipole.jl
    dipole_state = to_dipole_state(src)
    return Matrix(feature_field_jacobian(position, dipole_state))
end

# ============================================================================
# Σ_total Composition Helper
# ============================================================================

"""
    SigmaMapComponents

Components of map uncertainty for Σ_total composition.

# Usage
```julia
components = compute_sigma_map_components(result)
Σ_total = H * P * H' + R_sensor + components.Σ_map + Q_model
```
"""
struct SigmaMapComponents
    Σ_map::Matrix{Float64}        # Total map uncertainty (field)
    Σ_map_tile::Matrix{Float64}   # Tile contribution
    Σ_map_sources::Matrix{Float64} # Source contribution
    Σ_map_gradient::Matrix{Float64} # Gradient uncertainty (if using d=8)
end

"""
    compute_sigma_map_components(result::OnlineMapQueryResult;
                                 include_gradient::Bool=false) -> SigmaMapComponents

Extract map uncertainty components for Σ_total composition.

# INV-04: NEES Honesty
This function provides the Σ_map term that must be included in the
innovation covariance to maintain NEES calibration:

    S = H·P·H' + R_sensor + Σ_map + Q_model

Without Σ_map, the filter becomes overconfident when the map is uncertain.
"""
function compute_sigma_map_components(result::OnlineMapQueryResult;
                                      include_gradient::Bool = false)
    Σ_map = Matrix(result.Σ_B)
    Σ_map_tile = Matrix(result.Σ_B_tile)
    Σ_map_sources = Matrix(result.Σ_B_sources)

    if include_gradient
        # For d=8 mode: include gradient uncertainty
        Σ_map_gradient = Matrix(result.Σ_G)
    else
        Σ_map_gradient = zeros(5, 5)
    end

    return SigmaMapComponents(Σ_map, Σ_map_tile, Σ_map_sources, Σ_map_gradient)
end

"""
    build_sigma_total_with_map(H::AbstractMatrix, P::AbstractMatrix,
                               R_sensor::AbstractMatrix, Σ_map::AbstractMatrix,
                               Q_model::AbstractMatrix=zeros(size(R_sensor)))
        -> Matrix{Float64}

Build full innovation covariance including map uncertainty.

# INV-04 Implementation
    S = H·P·H' + R_sensor + Σ_map + Q_model

# Arguments
- `H`: Measurement Jacobian
- `P`: State covariance
- `R_sensor`: Sensor noise covariance
- `Σ_map`: Map uncertainty (from compute_sigma_map_components)
- `Q_model`: Model uncertainty (default: zero)

# Returns
Innovation covariance S suitable for Kalman gain computation.
"""
function build_sigma_total_with_map(H::AbstractMatrix, P::AbstractMatrix,
                                    R_sensor::AbstractMatrix, Σ_map::AbstractMatrix,
                                    Q_model::AbstractMatrix = zeros(size(R_sensor)))
    # Predicted measurement covariance
    HPHT = H * P * H'

    # Total innovation covariance
    S = HPHT + R_sensor + Σ_map + Q_model

    # Ensure positive definite (numerical safety)
    return ensure_spd(Matrix(S))
end

# ============================================================================
# AbstractMapProvider Interface Implementation
# ============================================================================

"""
    query_map(provider::OnlineMapProvider, slam_state::SlamAugmentedState,
              position::AbstractVector) -> MapQueryResult

Standard interface implementation for compatibility.
"""
function query_map(provider::OnlineMapProvider, slam_state::SlamAugmentedState,
                   position::AbstractVector)
    result = query_online_map(provider, slam_state, position)
    return to_map_query_result(result)
end

# ============================================================================
# Exports
# ============================================================================

export OnlineMapQueryConfig, DEFAULT_ONLINE_MAP_QUERY_CONFIG
export OnlineMapQueryResult, to_map_query_result
export OnlineMapProvider, DEFAULT_ONLINE_MAP_PROVIDER
export query_online_map
export compute_tile_quality_metric
export evaluate_tile_prediction, evaluate_source_contributions
export SigmaMapComponents, compute_sigma_map_components
export build_sigma_total_with_map
