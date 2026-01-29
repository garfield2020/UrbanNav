# ============================================================================
# Map Store - Persistent Storage and Versioning
# ============================================================================
#
# Ported from AUV-Navigation/src/map_store.jl
#
# Implements persistent storage for learned tile coefficients across missions.
# Enables map persistence, versioning, and rollback capabilities.
#
# Design Principles:
# 1. Every map save creates a new version (immutable versions)
# 2. Full version history maintained for auditability
# 3. Efficient diff computation between versions
# 4. Rollback capability for recovery from bad updates
# ============================================================================

using LinearAlgebra
using Dates
using Serialization

# ============================================================================
# Tile Snapshot
# ============================================================================

"""
    TileSnapshot

Immutable snapshot of a single tile's state.

# Fields
- `id::Int`: Tile identifier
- `center::Vector{Float64}`: Tile center position [x, y, z]
- `size::Float64`: Tile size (meters)
- `coefficients::Vector{Float64}`: Basis function coefficients
- `covariance::Matrix{Float64}`: Coefficient covariance
- `max_order::Int`: Maximum harmonic order
- `observation_count::Int`: Number of observations incorporated
- `last_updated::DateTime`: When tile was last updated
"""
struct TileSnapshot
    id::Int
    center::Vector{Float64}
    size::Float64
    coefficients::Vector{Float64}
    covariance::Matrix{Float64}
    max_order::Int
    observation_count::Int
    last_updated::DateTime
end

"""Create snapshot from live TileCoefficients."""
function TileSnapshot(tile::TileCoefficients; observation_count::Int = 0)
    TileSnapshot(
        tile.id,
        Vector{Float64}([tile.center[1], tile.center[2], tile.center[3]]),
        tile.size,
        copy(tile.coefficients),
        copy(tile.covariance),
        tile.max_order,
        observation_count,
        now(UTC)
    )
end

"""Restore TileCoefficients from snapshot."""
function restore_tile(snapshot::TileSnapshot)
    tile = TileCoefficients(
        center = snapshot.center,
        size = snapshot.size,
        max_order = snapshot.max_order,
        id = snapshot.id
    )
    tile.coefficients .= snapshot.coefficients
    tile.covariance .= snapshot.covariance
    return tile
end

# ============================================================================
# Map Version
# ============================================================================

"""
    MapVersion

Snapshot of tile coefficients at a specific point in time.

# Fields
- `version_id::String`: Unique identifier for this version
- `timestamp::DateTime`: When this version was created
- `parent_version::Union{String, Nothing}`: Previous version (nothing for initial)
- `mission_id::String`: Identifier of mission that created this version
- `tile_data::Dict{Tuple{Int,Int}, TileSnapshot}`: Snapshot of all tiles
- `metadata::Dict{String, Any}`: Additional metadata
"""
struct MapVersion
    version_id::String
    timestamp::DateTime
    parent_version::Union{String, Nothing}
    mission_id::String
    tile_data::Dict{Tuple{Int,Int}, TileSnapshot}
    metadata::Dict{String, Any}
end

"""Generate unique version ID."""
function generate_version_id()
    # Simple UUID-like ID using timestamp and random component
    ts = Dates.format(now(UTC), "yyyymmddHHMMSSsss")
    rand_part = string(rand(UInt32), base=16, pad=8)
    return "$ts-$rand_part"
end

# ============================================================================
# Map Diff
# ============================================================================

"""
    TileDiff

Difference between tile states across versions.
"""
struct TileDiff
    tile_idx::Tuple{Int,Int}
    change_type::Symbol              # :added, :removed, :modified
    coefficient_delta::Union{Vector{Float64}, Nothing}
    covariance_delta::Union{Matrix{Float64}, Nothing}
    mahalanobis_distance::Float64
end

"""
    MapDiff

Summary of differences between two map versions.
"""
struct MapDiff
    from_version::String
    to_version::String
    tile_diffs::Vector{TileDiff}
    n_added::Int
    n_removed::Int
    n_modified::Int
    total_coefficient_drift::Float64
    max_mahalanobis::Float64
end

"""Compute difference between two tile states."""
function compute_tile_diff(idx::Tuple{Int,Int},
                           old_tile::Union{TileSnapshot, Nothing},
                           new_tile::Union{TileSnapshot, Nothing})
    if old_tile === nothing && new_tile !== nothing
        return TileDiff(idx, :added, nothing, nothing, 0.0)
    elseif old_tile !== nothing && new_tile === nothing
        return TileDiff(idx, :removed, nothing, nothing, 0.0)
    elseif old_tile !== nothing && new_tile !== nothing
        coeff_delta = new_tile.coefficients - old_tile.coefficients
        cov_delta = new_tile.covariance - old_tile.covariance

        # Compute Mahalanobis distance
        avg_cov = (old_tile.covariance + new_tile.covariance) / 2
        n = size(avg_cov, 1)
        avg_cov_reg = avg_cov + 1e-10 * I(n)

        maha_dist = try
            maha_sq = coeff_delta' * (avg_cov_reg \ coeff_delta)
            sqrt(max(maha_sq, 0.0))
        catch
            norm(coeff_delta) / sqrt(tr(avg_cov) / n)
        end

        return TileDiff(idx, :modified, coeff_delta, cov_delta, maha_dist)
    else
        error("Both old and new tiles are nothing for index $idx")
    end
end

"""Compute comprehensive diff between two map versions."""
function compute_map_diff(version1::MapVersion, version2::MapVersion)
    tile_diffs = TileDiff[]
    n_added = 0
    n_removed = 0
    n_modified = 0
    total_drift = 0.0
    max_maha = 0.0

    all_indices = union(keys(version1.tile_data), keys(version2.tile_data))

    for idx in all_indices
        old_tile = get(version1.tile_data, idx, nothing)
        new_tile = get(version2.tile_data, idx, nothing)

        diff = compute_tile_diff(idx, old_tile, new_tile)
        push!(tile_diffs, diff)

        if diff.change_type == :added
            n_added += 1
        elseif diff.change_type == :removed
            n_removed += 1
        elseif diff.change_type == :modified
            n_modified += 1
            if diff.coefficient_delta !== nothing
                total_drift += norm(diff.coefficient_delta)
            end
            max_maha = max(max_maha, diff.mahalanobis_distance)
        end
    end

    return MapDiff(
        version1.version_id,
        version2.version_id,
        tile_diffs,
        n_added, n_removed, n_modified,
        total_drift, max_maha
    )
end

# ============================================================================
# Map Store
# ============================================================================

"""
    MapStore

File-based storage for map versions with history.

# Fields
- `base_path::String`: Directory for storing map files
- `versions::Vector{MapVersion}`: In-memory version history
- `version_index::Dict{String, Int}`: Map from ID to index
- `current::Union{String, Nothing}`: Currently active version
- `tile_size::Float64`: Default tile size for new maps
- `max_order::Int`: Default harmonic order for new tiles
"""
mutable struct MapStore
    base_path::String
    versions::Vector{MapVersion}
    version_index::Dict{String, Int}
    current::Union{String, Nothing}
    tile_size::Float64
    max_order::Int
end

"""Create or load a map store from the given directory."""
function MapStore(base_path::String; tile_size::Float64 = 50.0, max_order::Int = 3)
    mkpath(base_path)

    store = MapStore(
        base_path,
        MapVersion[],
        Dict{String, Int}(),
        nothing,
        tile_size,
        max_order
    )

    # Try to load existing index
    index_path = joinpath(base_path, "index.jls")
    if isfile(index_path)
        load_store_index!(store)
    end

    return store
end

"""Get path to store index file."""
store_index_path(store::MapStore) = joinpath(store.base_path, "index.jls")

"""Get path to version file."""
function version_file_path(store::MapStore, version_id::String)
    joinpath(store.base_path, "versions", "$version_id.jls")
end

"""Save store index to disk."""
function save_store_index!(store::MapStore)
    index_data = Dict(
        "current" => store.current === nothing ? "" : store.current,
        "tile_size" => store.tile_size,
        "max_order" => store.max_order,
        "version_ids" => [v.version_id for v in store.versions],
        "version_missions" => [v.mission_id for v in store.versions],
        "version_timestamps" => [string(v.timestamp) for v in store.versions],
        "version_parents" => [v.parent_version === nothing ? "" : v.parent_version
                              for v in store.versions]
    )

    open(store_index_path(store), "w") do io
        serialize(io, index_data)
    end
end

"""Load store index from disk."""
function load_store_index!(store::MapStore)
    if !isfile(store_index_path(store))
        return
    end

    index_data = open(deserialize, store_index_path(store))

    store.tile_size = get(index_data, "tile_size", 50.0)
    store.max_order = get(index_data, "max_order", 3)

    current_str = get(index_data, "current", "")
    store.current = isempty(current_str) ? nothing : current_str

    version_ids = get(index_data, "version_ids", String[])
    version_missions = get(index_data, "version_missions", String[])
    version_timestamps = get(index_data, "version_timestamps", String[])
    version_parents = get(index_data, "version_parents", String[])

    empty!(store.versions)
    empty!(store.version_index)

    for i in eachindex(version_ids)
        parent = isempty(version_parents[i]) ? nothing : version_parents[i]
        ts = DateTime(version_timestamps[i])

        # Create placeholder version
        version = MapVersion(
            version_ids[i],
            ts,
            parent,
            version_missions[i],
            Dict{Tuple{Int,Int}, TileSnapshot}(),
            Dict{String, Any}()
        )

        push!(store.versions, version)
        store.version_index[version_ids[i]] = length(store.versions)
    end
end

"""Save a version to disk."""
function save_map_version!(store::MapStore, version::MapVersion)
    versions_dir = joinpath(store.base_path, "versions")
    mkpath(versions_dir)

    # Serialize tile data
    tile_data_serializable = Dict{String, Any}()
    for (idx, snapshot) in version.tile_data
        key = "$(idx[1])_$(idx[2])"
        tile_data_serializable[key] = Dict(
            "id" => snapshot.id,
            "center" => snapshot.center,
            "size" => snapshot.size,
            "coefficients" => snapshot.coefficients,
            "covariance" => snapshot.covariance,
            "max_order" => snapshot.max_order,
            "observation_count" => snapshot.observation_count,
            "last_updated" => string(snapshot.last_updated)
        )
    end

    version_data = Dict(
        "version_id" => version.version_id,
        "timestamp" => string(version.timestamp),
        "parent_version" => version.parent_version === nothing ? "" : version.parent_version,
        "mission_id" => version.mission_id,
        "tile_data" => tile_data_serializable,
        "metadata" => version.metadata
    )

    open(version_file_path(store, version.version_id), "w") do io
        serialize(io, version_data)
    end
end

"""Load full version data from disk."""
function load_map_version(store::MapStore, version_id::String)
    path = version_file_path(store, version_id)
    if !isfile(path)
        error("Version file not found: $path")
    end

    version_data = open(deserialize, path)

    # Reconstruct tile data
    tile_data = Dict{Tuple{Int,Int}, TileSnapshot}()
    for (key, tile_dict) in version_data["tile_data"]
        parts = split(key, "_")
        idx = (parse(Int, parts[1]), parse(Int, parts[2]))

        snapshot = TileSnapshot(
            tile_dict["id"],
            Vector{Float64}(tile_dict["center"]),
            tile_dict["size"],
            Vector{Float64}(tile_dict["coefficients"]),
            Matrix{Float64}(tile_dict["covariance"]),
            tile_dict["max_order"],
            tile_dict["observation_count"],
            DateTime(tile_dict["last_updated"])
        )

        tile_data[idx] = snapshot
    end

    parent = isempty(version_data["parent_version"]) ? nothing : version_data["parent_version"]

    return MapVersion(
        version_data["version_id"],
        DateTime(version_data["timestamp"]),
        parent,
        version_data["mission_id"],
        tile_data,
        version_data["metadata"]
    )
end

"""
    save_map!(store, tile_manager, mission_id; observation_counts, metadata)

Save current tile manager state as a new version.
Returns the version ID.
"""
function save_map!(store::MapStore, tile_manager::TileManager, mission_id::String;
                   observation_counts::Dict{Tuple{Int,Int}, Int} = Dict{Tuple{Int,Int}, Int}(),
                   metadata::Dict{String, Any} = Dict{String, Any}())
    # Create tile snapshots
    tile_data = Dict{Tuple{Int,Int}, TileSnapshot}()
    for (idx, tile) in tile_manager.tiles
        obs_count = get(observation_counts, idx, 0)
        tile_data[idx] = TileSnapshot(tile; observation_count=obs_count)
    end

    # Create new version
    version_id = generate_version_id()
    version = MapVersion(
        version_id,
        now(UTC),
        store.current,
        mission_id,
        tile_data,
        metadata
    )

    # Save to disk
    save_map_version!(store, version)

    # Update in-memory state
    push!(store.versions, version)
    store.version_index[version_id] = length(store.versions)
    store.current = version_id

    # Save index
    save_store_index!(store)

    return version_id
end

"""Load a specific version and return a TileManager."""
function load_map(store::MapStore, version_id::String)
    version = load_map_version(store, version_id)

    # Infer tile_size and max_order from first tile
    if !isempty(version.tile_data)
        first_tile = first(values(version.tile_data))
        tile_size = first_tile.size
        max_order = first_tile.max_order
    else
        tile_size = store.tile_size
        max_order = store.max_order
    end

    tm = TileManager(tile_size=tile_size, max_order=max_order)

    # Restore tiles
    for (idx, snapshot) in version.tile_data
        tile = restore_tile(snapshot)
        tm.tiles[idx] = tile
        tm.next_id = max(tm.next_id, tile.id + 1)
    end

    return tm
end

"""Load the most recent version."""
function load_latest_map(store::MapStore)
    if store.current === nothing
        error("No versions available in store")
    end
    return load_map(store, store.current)
end

"""Get summary of all versions."""
function list_map_versions(store::MapStore)
    summaries = NamedTuple[]

    for version in store.versions
        full_version = load_map_version(store, version.version_id)

        push!(summaries, (
            version_id = version.version_id,
            timestamp = version.timestamp,
            mission_id = version.mission_id,
            parent_version = version.parent_version,
            n_tiles = length(full_version.tile_data)
        ))
    end

    return summaries
end

"""Compute diff between two versions."""
function diff_map_versions(store::MapStore, v1_id::String, v2_id::String)
    v1 = load_map_version(store, v1_id)
    v2 = load_map_version(store, v2_id)
    return compute_map_diff(v1, v2)
end

"""Set the specified version as current (rollback)."""
function rollback_map!(store::MapStore, version_id::String)
    if !haskey(store.version_index, version_id)
        error("Version not found: $version_id")
    end

    store.current = version_id
    save_store_index!(store)

    return load_map(store, version_id)
end

"""Get chain of versions from initial to specified version."""
function get_version_chain(store::MapStore, version_id::String)
    chain = String[]
    current_id = version_id

    while current_id !== nothing
        pushfirst!(chain, current_id)

        idx = store.version_index[current_id]
        parent = store.versions[idx].parent_version
        current_id = parent
    end

    return chain
end

"""Compute summary statistics for a version."""
function compute_version_statistics(store::MapStore, version_id::String)
    version = load_map_version(store, version_id)

    n_tiles = length(version.tile_data)
    total_obs = sum(t.observation_count for t in values(version.tile_data); init=0)

    all_coeffs = Float64[]
    all_cov_traces = Float64[]

    for tile in values(version.tile_data)
        append!(all_coeffs, tile.coefficients)
        push!(all_cov_traces, tr(tile.covariance))
    end

    coeff_mean = isempty(all_coeffs) ? 0.0 : sum(all_coeffs) / length(all_coeffs)
    coeff_var = isempty(all_coeffs) ? 0.0 : sum((c - coeff_mean)^2 for c in all_coeffs) / length(all_coeffs)
    coeff_std = sqrt(coeff_var)
    cov_trace_mean = isempty(all_cov_traces) ? 0.0 : sum(all_cov_traces) / length(all_cov_traces)

    return (
        n_tiles = n_tiles,
        total_observations = total_obs,
        coefficient_mean = coeff_mean,
        coefficient_std = coeff_std,
        covariance_trace_mean = cov_trace_mean,
        timestamp = version.timestamp,
        mission_id = version.mission_id
    )
end

# ============================================================================
# Exports
# ============================================================================

export TileSnapshot, restore_tile
export MapVersion, generate_version_id
export TileDiff, MapDiff, compute_tile_diff, compute_map_diff
export MapStore
export save_store_index!, load_store_index!
export save_map_version!, load_map_version
export save_map!, load_map, load_latest_map
export list_map_versions, diff_map_versions, rollback_map!
export get_version_chain, compute_version_statistics
