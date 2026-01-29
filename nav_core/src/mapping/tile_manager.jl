# ============================================================================
# Tile Manager - Spatial indexing and tile lookup
# ============================================================================
#
# Ported from AUV-Navigation/src/map_estimation.jl
#
# Manages tile creation, lookup, and spatial indexing for the map.
# Tiles are indexed by (ix, iy) integer coordinates.
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Tile Manager Structure
# ============================================================================

"""
    TileManager

Manages spatial indexing and lookup of tiles.

# Fields
- `tiles::Dict{Tuple{Int,Int}, TileCoefficients}`: Tiles indexed by (ix, iy)
- `tile_size::Float64`: Size of each tile (meters)
- `max_order::Int`: Harmonic order for new tiles
- `origin::Vec3`: Origin of tile grid
- `next_id::Int`: Next tile ID to assign
"""
mutable struct TileManager
    tiles::Dict{Tuple{Int,Int}, TileCoefficients}
    tile_size::Float64
    max_order::Int
    origin::Vec3
    next_id::Int
end

"""
    TileManager(; tile_size, max_order, origin)

Create a new tile manager.
"""
function TileManager(;
    tile_size::Real = 50.0,
    max_order::Int = 3,
    origin::AbstractVector = [0.0, 0.0, 0.0]
)
    TileManager(
        Dict{Tuple{Int,Int}, TileCoefficients}(),
        Float64(tile_size),
        max_order,
        Vec3(origin...),
        1
    )
end

# ============================================================================
# Tile Indexing
# ============================================================================

"""
    tile_index(tm::TileManager, world_pos::AbstractVector)

Get (ix, iy) tile index for world position.
"""
function tile_index(tm::TileManager, world_pos::AbstractVector)
    rel_pos = Vec3(world_pos...) - tm.origin
    ix = floor(Int, rel_pos[1] / tm.tile_size)
    iy = floor(Int, rel_pos[2] / tm.tile_size)
    return (ix, iy)
end

"""
    tile_center(tm::TileManager, idx::Tuple{Int,Int})

Get world-frame center of tile at index.
"""
function tile_center(tm::TileManager, idx::Tuple{Int,Int})
    ix, iy = idx
    cx = tm.origin[1] + (ix + 0.5) * tm.tile_size
    cy = tm.origin[2] + (iy + 0.5) * tm.tile_size
    cz = tm.origin[3]  # Tiles are 2.5D: x-y indexed, z evaluated in 3D
    return Vec3(cx, cy, cz)
end

# ============================================================================
# Tile Access
# ============================================================================

"""
    get_or_create_tile!(tm::TileManager, world_pos::AbstractVector)

Get tile containing position, creating if needed.
"""
function get_or_create_tile!(tm::TileManager, world_pos::AbstractVector)
    idx = tile_index(tm, world_pos)

    if !haskey(tm.tiles, idx)
        center = tile_center(tm, idx)
        tile = TileCoefficients(
            center = center,
            size = tm.tile_size,
            max_order = tm.max_order,
            id = tm.next_id
        )
        tm.tiles[idx] = tile
        tm.next_id += 1
    end

    return tm.tiles[idx]
end

"""
    get_tile(tm::TileManager, world_pos::AbstractVector)

Get tile containing position, or nothing if not created.
"""
function get_tile(tm::TileManager, world_pos::AbstractVector)
    idx = tile_index(tm, world_pos)
    return get(tm.tiles, idx, nothing)
end

"""
    get_tile_by_index(tm::TileManager, idx::Tuple{Int,Int})

Get tile by index, or nothing if not created.
"""
function get_tile_by_index(tm::TileManager, idx::Tuple{Int,Int})
    return get(tm.tiles, idx, nothing)
end

"""
    has_tile(tm::TileManager, world_pos::AbstractVector)

Check if a tile exists at the given position.
"""
function has_tile(tm::TileManager, world_pos::AbstractVector)
    idx = tile_index(tm, world_pos)
    return haskey(tm.tiles, idx)
end

# ============================================================================
# Tile Neighbors
# ============================================================================

"""
    get_neighboring_tiles(tm::TileManager, idx::Tuple{Int,Int})

Get indices and tiles of neighboring tiles (4-connected).
"""
function get_neighboring_tiles(tm::TileManager, idx::Tuple{Int,Int})
    ix, iy = idx
    neighbors = [
        (ix+1, iy), (ix-1, iy),
        (ix, iy+1), (ix, iy-1)
    ]
    return [(n, tm.tiles[n]) for n in neighbors if haskey(tm.tiles, n)]
end

"""
    get_8_neighboring_tiles(tm::TileManager, idx::Tuple{Int,Int})

Get indices and tiles of neighboring tiles (8-connected).
"""
function get_8_neighboring_tiles(tm::TileManager, idx::Tuple{Int,Int})
    ix, iy = idx
    neighbors = [
        (ix+1, iy), (ix-1, iy), (ix, iy+1), (ix, iy-1),
        (ix+1, iy+1), (ix+1, iy-1), (ix-1, iy+1), (ix-1, iy-1)
    ]
    return [(n, tm.tiles[n]) for n in neighbors if haskey(tm.tiles, n)]
end

# ============================================================================
# Tile Iteration
# ============================================================================

"""
    all_tiles(tm::TileManager)

Get all tiles as vector.
"""
all_tiles(tm::TileManager) = collect(values(tm.tiles))

"""
    all_tile_indices(tm::TileManager)

Get all tile indices.
"""
all_tile_indices(tm::TileManager) = collect(keys(tm.tiles))

"""
    n_tiles(tm::TileManager)

Get number of tiles.
"""
n_tiles(tm::TileManager) = length(tm.tiles)

# ============================================================================
# Field Evaluation via TileManager
# ============================================================================

"""
    evaluate_field_at(tm::TileManager, world_pos::AbstractVector)

Evaluate field at position using appropriate tile.

Returns (B, found) where found is true if tile exists.
"""
function evaluate_field_at(tm::TileManager, world_pos::AbstractVector)
    tile = get_tile(tm, world_pos)
    if tile === nothing
        return (Vec3(0.0, 0.0, 0.0), false)
    end
    return (evaluate_tile_field(tile, world_pos), true)
end

"""
    evaluate_field_and_gradient_at(tm::TileManager, world_pos::AbstractVector)

Evaluate field and gradient at position using appropriate tile.

Returns (B, G5, found) where found is true if tile exists.
"""
function evaluate_field_and_gradient_at(tm::TileManager, world_pos::AbstractVector)
    tile = get_tile(tm, world_pos)
    if tile === nothing
        return (Vec3(0.0, 0.0, 0.0), zeros(SVector{5}), false)
    end
    B, G5 = evaluate_tile_field_and_gradient(tile, world_pos)
    return (B, G5, true)
end

# ============================================================================
# Tile Statistics
# ============================================================================

"""
    TileStatistics

Summary statistics for tile manager.
"""
struct TileStatistics
    n_tiles::Int
    total_coefficients::Int
    coefficient_mean::Float64
    coefficient_std::Float64
    covariance_trace_mean::Float64
    bounding_box::Tuple{Vec3, Vec3}
end

"""
    compute_tile_statistics(tm::TileManager)

Compute summary statistics for all tiles.
"""
function compute_tile_statistics(tm::TileManager)
    if isempty(tm.tiles)
        return TileStatistics(
            0, 0, 0.0, 0.0, 0.0,
            (Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
        )
    end

    all_coeffs = Float64[]
    all_cov_traces = Float64[]
    min_corner = Vec3(Inf, Inf, Inf)
    max_corner = Vec3(-Inf, -Inf, -Inf)

    for tile in values(tm.tiles)
        append!(all_coeffs, tile.coefficients)
        push!(all_cov_traces, tr(tile.covariance))

        half_size = tile.size / 2
        tile_min = tile.center - Vec3(half_size, half_size, half_size)
        tile_max = tile.center + Vec3(half_size, half_size, half_size)

        min_corner = Vec3(
            min(min_corner[1], tile_min[1]),
            min(min_corner[2], tile_min[2]),
            min(min_corner[3], tile_min[3])
        )
        max_corner = Vec3(
            max(max_corner[1], tile_max[1]),
            max(max_corner[2], tile_max[2]),
            max(max_corner[3], tile_max[3])
        )
    end

    coeff_mean = isempty(all_coeffs) ? 0.0 : sum(all_coeffs) / length(all_coeffs)
    coeff_var = isempty(all_coeffs) ? 0.0 : sum((c - coeff_mean)^2 for c in all_coeffs) / length(all_coeffs)
    coeff_std = sqrt(coeff_var)
    cov_trace_mean = isempty(all_cov_traces) ? 0.0 : sum(all_cov_traces) / length(all_cov_traces)

    return TileStatistics(
        length(tm.tiles),
        length(all_coeffs),
        coeff_mean,
        coeff_std,
        cov_trace_mean,
        (min_corner, max_corner)
    )
end

# ============================================================================
# Exports
# ============================================================================

export TileManager
export tile_index, tile_center
export get_or_create_tile!, get_tile, get_tile_by_index, has_tile
export get_neighboring_tiles, get_8_neighboring_tiles
export all_tiles, all_tile_indices, n_tiles
export evaluate_field_at, evaluate_field_and_gradient_at
export TileStatistics, compute_tile_statistics
