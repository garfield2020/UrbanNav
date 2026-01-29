# ============================================================================
# Map Contract - Frozen Map Mode Data Model (Phase A)
# ============================================================================
#
# Defines the authoritative interfaces for map-backed magnetic field prediction.
#
# Physics Contract:
# - Maps represent magnetic scalar potential Φ expanded in harmonic basis
# - Field B = -∇Φ satisfies Maxwell (∇·B = 0, ∇×B = 0 in source-free region)
# - Gradient tensor G = ∂B/∂x is symmetric and traceless (5 independent components)
#
# Invariants:
# - Maps are IMMUTABLE during frozen mode (no online learning)
# - All positions in NED world frame [m]
# - All fields in Tesla [T], gradients in [T/m]
# - Uncertainties as covariances, not standard deviations
# ============================================================================

using LinearAlgebra
using StaticArrays
using Dates

# ============================================================================
# Type Aliases (consistent with StateContract.jl)
# ============================================================================

const Vec3Map = SVector{3, Float64}
const Mat3Map = SMatrix{3, 3, Float64, 9}

# ============================================================================
# Map Metadata
# ============================================================================

"""
    MapBasisType

Enumeration of supported harmonic basis types.

Canonical tile coefficient counts (Maxwell-consistent, ∇·B=0):
- `LINEAR`:    8 coefficients (3 field B₀ + 5 traceless symmetric gradient G)
- `QUADRATIC`: 15 coefficients (8 linear + 7 quadratic solid harmonics Q)
- `CUBIC`:     Not yet defined for tiles

NOTE: Earlier versions documented LINEAR=4, QUADRATIC=10, CUBIC=16 based on a
different (packed/potential) parameterization. The canonical tile counts are
defined in BasisEnrichment.COEFFICIENT_COUNTS and must match everywhere.
"""
@enum MapBasisType begin
    MAP_BASIS_LINEAR = 1      # B₀ + G₀·Δx (8 DOF: 3 field + 5 traceless gradient)
    MAP_BASIS_QUADRATIC = 2   # Up to order 2 (15 coefficients, Maxwell-consistent)
    MAP_BASIS_CUBIC = 3       # Up to order 3 (coefficient count TBD)
end

# ============================================================================
# Canonical Tile Coefficient Counts (Single Source of Truth)
# ============================================================================

"""
    N_TILE_COEF_CONSTANT = 3
    N_TILE_COEF_LINEAR   = 8
    N_TILE_COEF_QUADRATIC = 15

Canonical tile coefficient counts for Maxwell-consistent magnetic field models.
These are the ONLY valid tile dimensions for Phase E.

Layout:
  coef[1:3]  = B₀  (field intercept, 3 components)
  coef[4:8]  = G₅  (gradient, 5 DOF: traceless symmetric 3×3)
  coef[9:15] = Q₇  (quadratic, 7 DOF: divergence-free solid harmonics)

Physics: B(x) = -∇Φ where Φ = Σᵢ αᵢ φᵢ(x - center), ∇²φᵢ = 0.
The ∇·B=0 constraint reduces order-2 from 10 unconstrained Hessian DOF to 7.
"""
const N_TILE_COEF_CONSTANT  = 3
const N_TILE_COEF_LINEAR    = 8
const N_TILE_COEF_QUADRATIC = 15

"""
    n_tile_coefficients(basis::MapBasisType) -> Int

Return the canonical number of tile coefficients for a given basis type.
This is the single source of truth—all other coefficient count definitions
must agree with this function.
"""
function n_tile_coefficients(basis::MapBasisType)
    if basis == MAP_BASIS_LINEAR
        return N_TILE_COEF_LINEAR       # 8
    elseif basis == MAP_BASIS_QUADRATIC
        return N_TILE_COEF_QUADRATIC    # 15
    elseif basis == MAP_BASIS_CUBIC
        error("CUBIC tile coefficient count not yet canonicalized")
    else
        error("Unknown MapBasisType: $basis")
    end
end

"""
    MapFrame

Coordinate frame specification for map data.

# Fields
- `name::Symbol`: Frame identifier (`:NED_world`, `:NED_local`, `:ECEF`)
- `origin::Vec3Map`: Origin position in parent frame [m]
- `orientation::Mat3Map`: Rotation from map frame to parent frame
"""
struct MapFrame
    name::Symbol
    origin::Vec3Map
    orientation::Mat3Map
end

"""Default NED world frame centered at origin."""
function MapFrame()
    MapFrame(:NED_world, Vec3Map(0.0, 0.0, 0.0), Mat3Map(I))
end

"""
    MapMetadata

Immutable metadata describing a frozen map.

# Fields
- `version::String`: Semantic version (e.g., "1.0.0")
- `created::DateTime`: Creation timestamp (UTC)
- `mission_id::String`: Identifier of mission that built this map
- `frame::MapFrame`: Coordinate frame specification
- `basis_type::MapBasisType`: Harmonic basis used
- `tile_size::Float64`: Tile edge length [m] (0.0 for single-tile global)
- `coverage_bounds::Tuple{Vec3Map, Vec3Map}`: (min_corner, max_corner) [m]

# Invariants
- version follows semantic versioning
- tile_size ≥ 0 (0 indicates single global tile)
- coverage_bounds[1] ≤ coverage_bounds[2] componentwise
"""
struct MapMetadata
    version::String
    created::DateTime
    mission_id::String
    frame::MapFrame
    basis_type::MapBasisType
    tile_size::Float64
    coverage_bounds::Tuple{Vec3Map, Vec3Map}
end

function MapMetadata(;
    version::String = "1.0.0",
    mission_id::String = "unknown",
    frame::MapFrame = MapFrame(),
    basis_type::MapBasisType = MAP_BASIS_LINEAR,
    tile_size::Float64 = 0.0,
    coverage_min::AbstractVector = [-Inf, -Inf, -Inf],
    coverage_max::AbstractVector = [Inf, Inf, Inf]
)
    MapMetadata(
        version,
        now(UTC),
        mission_id,
        frame,
        basis_type,
        tile_size,
        (Vec3Map(coverage_min...), Vec3Map(coverage_max...))
    )
end

# ============================================================================
# Map Tile Identifier
# ============================================================================

"""
    MapTileID

Unique identifier for a tile within a tiled map.

# Fields
- `ix::Int`: X-index (East direction in NED)
- `iy::Int`: Y-index (North direction in NED)
- `iz::Int`: Z-index (Down direction in NED), typically 0 for 2D tiling

# Indexing Convention
Tile (ix, iy, iz) covers region:
- x ∈ [ix * tile_size, (ix+1) * tile_size)
- y ∈ [iy * tile_size, (iy+1) * tile_size)
- z ∈ [iz * tile_size, (iz+1) * tile_size) if 3D tiling

For 2D horizontal tiling (typical), iz = 0 and z is unbounded.
"""
struct MapTileID
    ix::Int
    iy::Int
    iz::Int
end

MapTileID(ix::Int, iy::Int) = MapTileID(ix, iy, 0)

"""Compute tile ID from world position and tile size."""
function tile_id_at(position::AbstractVector, tile_size::Float64)
    if tile_size <= 0.0
        return MapTileID(0, 0, 0)  # Global single-tile
    end
    ix = floor(Int, position[1] / tile_size)
    iy = floor(Int, position[2] / tile_size)
    iz = 0  # 2D horizontal tiling
    return MapTileID(ix, iy, iz)
end

"""Compute tile center position from tile ID."""
function tile_center(id::MapTileID, tile_size::Float64)
    if tile_size <= 0.0
        return Vec3Map(0.0, 0.0, 0.0)
    end
    x = (id.ix + 0.5) * tile_size
    y = (id.iy + 0.5) * tile_size
    z = 0.0  # Horizontal tiles have no vertical center
    return Vec3Map(x, y, z)
end

# ============================================================================
# Map Query Result
# ============================================================================

"""
    MapQueryResult

Result of querying the map at a position.

# Fields
- `B_pred::Vec3Map`: Predicted magnetic field [T] in world frame
- `G_pred::Mat3Map`: Predicted gradient tensor [T/m] in world frame (symmetric)
- `Σ_B::Mat3Map`: Field prediction covariance [T²]
- `Σ_G::SMatrix{5,5}`: Gradient prediction covariance [T²/m²] (packed 5-component)
- `Σ_BG::SMatrix{3,5}`: Cross-covariance between B and G [T²/m]
- `in_coverage::Bool`: True if position is within map coverage
- `tile_id::MapTileID`: Tile that provided this prediction

# Physics Notes
- G_pred is symmetric: G[i,j] = G[j,i]
- G_pred is traceless: G[1,1] + G[2,2] + G[3,3] = 0 (Maxwell: ∇·B = 0)
- Packed gradient: [Gxx, Gyy, Gxy, Gxz, Gyz], Gzz = -(Gxx + Gyy)

# Uncertainty Semantics
- Σ_B: 3×3 covariance of field prediction error
- Σ_G: 5×5 covariance of packed gradient prediction error
- These represent MAP UNCERTAINTY, not sensor noise
- For EKF update: R_total = R_sensor + Σ_map
"""
struct MapQueryResult
    B_pred::Vec3Map
    G_pred::Mat3Map
    Σ_B::Mat3Map
    Σ_G::SMatrix{5, 5, Float64, 25}
    Σ_BG::SMatrix{3, 5, Float64, 15}
    in_coverage::Bool
    tile_id::MapTileID
end

"""Create query result with diagonal covariances (typical case)."""
function MapQueryResult(;
    B_pred::AbstractVector,
    G_pred::AbstractMatrix,
    σ_B::Float64,           # Field uncertainty [T] (1-sigma)
    σ_G::Float64,           # Gradient uncertainty [T/m] (1-sigma)
    in_coverage::Bool = true,
    tile_id::MapTileID = MapTileID(0, 0)
)
    MapQueryResult(
        Vec3Map(B_pred...),
        Mat3Map(G_pred...),
        Mat3Map(σ_B^2 * I),
        SMatrix{5,5}(σ_G^2 * I),
        SMatrix{3,5}(zeros(3, 5)),
        in_coverage,
        tile_id
    )
end

"""Pack full 3×3 gradient tensor to 5 independent components."""
function pack_gradient(G::AbstractMatrix)
    # [Gxx, Gyy, Gxy, Gxz, Gyz]
    # Note: Gzz = -(Gxx + Gyy) from traceless constraint
    return SVector{5, Float64}(G[1,1], G[2,2], G[1,2], G[1,3], G[2,3])
end

"""Unpack 5 components to full 3×3 symmetric traceless tensor."""
function unpack_gradient(G5::AbstractVector)
    Gxx, Gyy, Gxy, Gxz, Gyz = G5[1], G5[2], G5[3], G5[4], G5[5]
    Gzz = -(Gxx + Gyy)  # Traceless: ∇·B = 0
    return @SMatrix [
        Gxx  Gxy  Gxz;
        Gxy  Gyy  Gyz;
        Gxz  Gyz  Gzz
    ]
end

# ============================================================================
# Map Model (Frozen Map Data Structure)
# ============================================================================

"""
    MapTileData

Coefficient data for a single tile.

# Fields
- `id::MapTileID`: Tile identifier
- `center::Vec3Map`: Tile center in world frame [m]
- `coefficients::Vector{Float64}`: Harmonic basis coefficients
- `covariance::Matrix{Float64}`: Coefficient covariance matrix
- `observation_count::Int`: Number of observations used in fit

# Physics Notes
- Field: B(x) = -∇Φ where Φ = Σᵢ αᵢ φᵢ(x - center)
- Each φᵢ satisfies Laplace equation: ∇²φᵢ = 0
- Number of tile coefficients depends on basis order (Maxwell-consistent):
  - LINEAR: 8 (3 field + 5 gradient)
  - QUADRATIC: 15 (8 linear + 7 quadratic solid harmonics)
  - CUBIC: TBD
"""
struct MapTileData
    id::MapTileID
    center::Vec3Map
    coefficients::Vector{Float64}
    covariance::Matrix{Float64}
    observation_count::Int
end

"""
    MapModel

Immutable frozen map for magnetic field prediction.

# Fields
- `metadata::MapMetadata`: Map metadata and configuration
- `tiles::Dict{MapTileID, MapTileData}`: Tile coefficient data
- `global_tile::Union{Nothing, MapTileData}`: Global fallback tile (if tiled)

# Invariants
- Map is IMMUTABLE after construction
- All queries return predictions with associated uncertainties
- Queries outside coverage use global_tile or return high uncertainty

# Usage
```julia
map = load_map(path)
result = query_map(map, position)
B_pred = result.B_pred
Σ_map = result.Σ_B  # For EKF: R_total = R_sensor + Σ_map
```
"""
struct MapModel
    metadata::MapMetadata
    tiles::Dict{MapTileID, MapTileData}
    global_tile::Union{Nothing, MapTileData}
end

"""Create single-tile (global) map model."""
function MapModel(;
    metadata::MapMetadata,
    coefficients::AbstractVector,
    covariance::AbstractMatrix,
    center::AbstractVector = [0.0, 0.0, 0.0],
    observation_count::Int = 0
)
    global_tile = MapTileData(
        MapTileID(0, 0),
        Vec3Map(center...),
        Vector{Float64}(coefficients),
        Matrix{Float64}(covariance),
        observation_count
    )
    MapModel(metadata, Dict{MapTileID, MapTileData}(), global_tile)
end

"""Check if map has coverage at position."""
function has_coverage(map::MapModel, position::AbstractVector)
    bounds = map.metadata.coverage_bounds
    return all(bounds[1] .<= position) && all(position .<= bounds[2])
end

"""Get tile for position (or global fallback)."""
function get_tile(map::MapModel, position::AbstractVector)
    tile_size = map.metadata.tile_size

    if tile_size > 0.0
        # Tiled map
        id = tile_id_at(position, tile_size)
        if haskey(map.tiles, id)
            return map.tiles[id]
        end
    end

    # Fall back to global tile
    return map.global_tile
end

# ============================================================================
# Abstract Map Provider Interface
# ============================================================================

"""
    AbstractMapProvider

Abstract interface for map data sources.

Implementations must provide:
- `load_map(provider, path)::MapModel` - Load map from source
- `query_map(provider, map, position)::MapQueryResult` - Query map at position

Two standard implementations:
1. `FrozenFileMapProvider` - Loads pre-computed map from disk (production)
2. `TruthMapProvider` - Samples truth world directly (simulation only)
"""
abstract type AbstractMapProvider end

"""
    load_map(provider::AbstractMapProvider, path::String)::MapModel

Load a map from the specified path/source.
"""
function load_map end

"""
    query_map(provider::AbstractMapProvider, map::MapModel, position::AbstractVector)::MapQueryResult

Query the map at the given position.

Returns predicted field B, gradient G, and associated uncertainties.
Position must be in the map's coordinate frame (typically NED world).
"""
function query_map end

# ============================================================================
# Phase B: Map Update Contract
# ============================================================================
#
# Phase B extends frozen maps with CONTROLLED updates.
#
# Key Invariants:
# - Updates modify ONLY map parameters (coefficients, covariance)
# - Updates NEVER modify truth world (enforced by architecture)
# - Updates are VERSIONED with full provenance
# - Query remains PURE: same (map, position) → same result
# - Updates are COMMUTATIVE in information form (within tolerance)
# ============================================================================

"""
    MapVersionInfo

Version and provenance tracking for map updates.

# Fields
- `version::Int`: Monotonic version counter (0 = initial, frozen)
- `parent_version::Int`: Version this update was based on
- `semantic_version::String`: Human-readable version string
- `created::DateTime`: Timestamp of this version
- `update_count::Int`: Cumulative number of updates applied
- `scenario_hash::UInt64`: Hash of source data for reproducibility
- `mission_ids::Vector{String}`: Missions that contributed to this version

# Invariants
- version > parent_version for all updates
- version = 0 means frozen (Phase A) map, no updates allowed to produce version 0
- update_count is cumulative across all versions in the chain
"""
struct MapVersionInfo
    version::Int
    parent_version::Int
    semantic_version::String
    created::DateTime
    update_count::Int
    scenario_hash::UInt64
    mission_ids::Vector{String}
end

"""Create initial frozen map version (Phase A)."""
function MapVersionInfo(; mission_id::String = "frozen")
    MapVersionInfo(0, -1, "1.0.0-frozen", now(UTC), 0, UInt64(0), [mission_id])
end

"""Create new version from parent."""
function MapVersionInfo(parent::MapVersionInfo;
                        mission_id::String,
                        update_count::Int,
                        scenario_hash::UInt64 = UInt64(0))
    new_version = parent.version + 1
    major = 1
    minor = new_version
    semantic = "$(major).$(minor).0"

    MapVersionInfo(
        new_version,
        parent.version,
        semantic,
        now(UTC),
        parent.update_count + update_count,
        scenario_hash,
        vcat(parent.mission_ids, [mission_id])
    )
end

"""Check if map is frozen (Phase A, no updates)."""
is_frozen(v::MapVersionInfo) = v.version == 0

"""
    MapUpdateSource

Identifies the source of map update data.

# Variants
- `UPDATE_SOURCE_NAVIGATION`: From navigation residuals (single vehicle)
- `UPDATE_SOURCE_SURVEY`: From dedicated survey pass
- `UPDATE_SOURCE_FLEET`: From fleet fusion (Phase B+)
- `UPDATE_SOURCE_MANUAL`: Manual/external correction
"""
@enum MapUpdateSource begin
    UPDATE_SOURCE_NAVIGATION = 1
    UPDATE_SOURCE_SURVEY = 2
    UPDATE_SOURCE_FLEET = 3
    UPDATE_SOURCE_MANUAL = 4
end

"""
    MapUpdateMessage

Bounded, typed message for map coefficient updates.

# Fields
- `tile_id::MapTileID`: Target tile for update
- `position::Vec3Map`: Position of measurement [m]
- `timestamp::Float64`: Measurement time [s]
- `innovation::Vector{Float64}`: z - h(α), measurement residual
- `H::Matrix{Float64}`: Jacobian ∂h/∂α at current estimate
- `R::Matrix{Float64}`: Measurement covariance (sensor only)
- `source::MapUpdateSource`: Where this update came from
- `pose_uncertainty::Float64`: Position uncertainty at measurement [m²]
- `weight::Float64`: Quality weight ∈ [0, 1] (0 = ignore, 1 = full weight)

# Size Bounds (enforced at construction)
- innovation: length ≤ 8 (3 field + 5 gradient components)
- H: rows ≤ 8, cols ≤ 16 (max basis order)
- R: 8×8 maximum

# Usage
Updates should be REJECTED if:
- pose_uncertainty > teachability threshold
- weight < minimum weight threshold
- innovation normalized ≫ 3 (outlier)
"""
struct MapUpdateMessage
    tile_id::MapTileID
    position::Vec3Map
    timestamp::Float64
    innovation::Vector{Float64}
    H::Matrix{Float64}
    R::Matrix{Float64}
    source::MapUpdateSource
    pose_uncertainty::Float64
    weight::Float64

    function MapUpdateMessage(tile_id, position, timestamp, innovation, H, R, source,
                              pose_uncertainty, weight)
        # Enforce size bounds
        @assert length(innovation) <= 8 "Innovation too large: $(length(innovation)) > 8"
        @assert size(H, 1) <= 8 "H rows too large: $(size(H, 1)) > 8"
        @assert size(H, 2) <= 16 "H cols too large: $(size(H, 2)) > 16"
        @assert size(R, 1) <= 8 && size(R, 2) <= 8 "R too large: $(size(R))"
        @assert 0.0 <= weight <= 1.0 "Weight out of bounds: $weight"
        @assert pose_uncertainty >= 0.0 "Negative pose uncertainty"

        new(tile_id, position, timestamp, innovation, H, R, source, pose_uncertainty, weight)
    end
end

"""Convenience constructor for typical navigation updates."""
function MapUpdateMessage(;
    tile_id::MapTileID,
    position::AbstractVector,
    timestamp::Float64,
    innovation::AbstractVector,
    H::AbstractMatrix,
    R::AbstractMatrix,
    source::MapUpdateSource = UPDATE_SOURCE_NAVIGATION,
    pose_uncertainty::Float64 = 0.0,
    weight::Float64 = 1.0
)
    MapUpdateMessage(
        tile_id,
        Vec3Map(position...),
        timestamp,
        Vector{Float64}(innovation),
        Matrix{Float64}(H),
        Matrix{Float64}(R),
        source,
        pose_uncertainty,
        weight
    )
end

"""
    MapUpdateResult

Result of applying map update(s).

# Fields
- `success::Bool`: Whether update was applied
- `coefficients::Vector{Float64}`: Updated coefficients (or original if not applied)
- `covariance::Matrix{Float64}`: Updated covariance
- `updates_applied::Int`: Number of updates actually applied
- `updates_rejected::Int`: Number rejected (outliers, low teachability)
- `rejection_reasons::Vector{Symbol}`: Reasons for rejections

# Rejection Reasons
- `:outlier`: Normalized innovation > threshold
- `:low_teachability`: Pose uncertainty too high
- `:low_observability`: Gradient energy below threshold
- `:singular`: Near-singular update matrix
- `:weight_threshold`: Weight below minimum
"""
struct MapUpdateResult
    success::Bool
    coefficients::Vector{Float64}
    covariance::Matrix{Float64}
    updates_applied::Int
    updates_rejected::Int
    rejection_reasons::Vector{Symbol}
end

"""
    AbstractMapUpdater

Abstract interface for map update implementations.

Implementations must provide:
- `apply_update(updater, tile, message)::MapUpdateResult` - Apply single update
- `apply_batch(updater, tile, messages)::MapUpdateResult` - Apply batch of updates
- `can_update(updater, message)::Tuple{Bool, Symbol}` - Check if update is acceptable

# Architecture Constraint
MapUpdater NEVER has access to truth world. All information comes through
MapUpdateMessage, which contains only measurements and their statistics.
"""
abstract type AbstractMapUpdater end

"""
    apply_update(updater::AbstractMapUpdater, tile::MapTileData, msg::MapUpdateMessage)

Apply a single update to tile coefficients.

Returns MapUpdateResult with updated coefficients and covariance.
"""
function apply_update end

"""
    apply_batch(updater::AbstractMapUpdater, tile::MapTileData, msgs::Vector{MapUpdateMessage})

Apply multiple updates to tile coefficients.

Information-form batch update for efficiency and commutativity.
"""
function apply_batch end

"""
    can_update(updater::AbstractMapUpdater, msg::MapUpdateMessage)

Check if an update message should be applied.

Returns (allowed::Bool, reason::Symbol).
Reason is :ok if allowed, otherwise the rejection reason.
"""
function can_update end

"""
    MutableMapModel

Map model that supports Phase B updates.

# Fields
- Same as MapModel, plus:
- `version_info::MapVersionInfo`: Version tracking
- `update_history::Vector{Tuple{DateTime, Int, Symbol}}`: (time, count, source)

# Invariants
- Version increases monotonically with updates
- History preserves all update metadata for rollback
- Query behavior identical to MapModel (pure, deterministic)
"""
mutable struct MutableMapModel
    metadata::MapMetadata
    tiles::Dict{MapTileID, MapTileData}
    global_tile::Union{Nothing, MapTileData}
    version_info::MapVersionInfo
    update_history::Vector{Tuple{DateTime, Int, Symbol}}
end

"""Create mutable map from frozen map (Phase A → Phase B transition)."""
function MutableMapModel(frozen::MapModel; mission_id::String = "phase_b_start")
    MutableMapModel(
        frozen.metadata,
        deepcopy(frozen.tiles),
        frozen.global_tile === nothing ? nothing : deepcopy(frozen.global_tile),
        MapVersionInfo(mission_id = mission_id),
        Tuple{DateTime, Int, Symbol}[]
    )
end

"""Convert mutable map back to frozen (snapshot)."""
function freeze(m::MutableMapModel)::MapModel
    MapModel(
        m.metadata,
        deepcopy(m.tiles),
        m.global_tile === nothing ? nothing : deepcopy(m.global_tile)
    )
end

"""Update tile in mutable map."""
function update_tile!(m::MutableMapModel, tile_id::MapTileID, result::MapUpdateResult;
                      source::MapUpdateSource = UPDATE_SOURCE_NAVIGATION)
    if !result.success || result.updates_applied == 0
        return false
    end

    old_tile = haskey(m.tiles, tile_id) ? m.tiles[tile_id] : m.global_tile

    if old_tile === nothing
        return false
    end

    new_tile = MapTileData(
        old_tile.id,
        old_tile.center,
        result.coefficients,
        result.covariance,
        old_tile.observation_count + result.updates_applied
    )

    if tile_id == MapTileID(0, 0) && m.global_tile !== nothing
        m.global_tile = new_tile
    else
        m.tiles[tile_id] = new_tile
    end

    # Update version
    m.version_info = MapVersionInfo(
        m.version_info,
        mission_id = "update",
        update_count = result.updates_applied
    )

    # Record in history
    push!(m.update_history, (now(UTC), result.updates_applied, Symbol(source)))

    return true
end

"""Check if map is frozen (no updates applied)."""
is_frozen(m::MutableMapModel) = is_frozen(m.version_info)

"""Get current map version."""
get_version(m::MutableMapModel) = m.version_info.version

# ============================================================================
# Exports
# ============================================================================

export Vec3Map, Mat3Map
export MapBasisType, MAP_BASIS_LINEAR, MAP_BASIS_QUADRATIC, MAP_BASIS_CUBIC
export N_TILE_COEF_CONSTANT, N_TILE_COEF_LINEAR, N_TILE_COEF_QUADRATIC
export n_tile_coefficients
export MapFrame, MapMetadata
export MapTileID, tile_id_at, tile_center
export MapQueryResult, pack_gradient, unpack_gradient
export MapTileData, MapModel, has_coverage, get_tile
export AbstractMapProvider, load_map, query_map

# Phase B exports
export MapVersionInfo, is_frozen
export MapUpdateSource, UPDATE_SOURCE_NAVIGATION, UPDATE_SOURCE_SURVEY
export UPDATE_SOURCE_FLEET, UPDATE_SOURCE_MANUAL
export MapUpdateMessage, MapUpdateResult
export AbstractMapUpdater, apply_update, apply_batch, can_update
export MutableMapModel, freeze, update_tile!, get_version
