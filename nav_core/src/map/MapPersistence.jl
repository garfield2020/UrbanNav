# ============================================================================
# MapPersistence.jl - Map Checkpoint File I/O (Phase C Multi-Mission Support)
# ============================================================================
#
# Provides serialization/deserialization of map checkpoints for persistence
# across missions. This enables the "manifold collapse" workflow:
#
#   Mission 1: Start with prior → Learn → Save checkpoint
#   Mission 2: Load checkpoint → Continue learning → Save checkpoint
#   ...
#   Mission N: Map converges to steady state
#
# Key Design Decisions:
# 1. JSON format for human-readable inspection and versioning
# 2. Checksum verification for integrity
# 3. Version field for forward compatibility
# 4. Provenance preservation for audit trail
#
# ============================================================================

using JSON3
using SHA
using Dates
using LinearAlgebra

# ============================================================================
# File Format Version
# ============================================================================

"""Current checkpoint file format version."""
const CHECKPOINT_FILE_VERSION = "1.0.0"

# ============================================================================
# Multi-Tile Checkpoint (for full SLAM state persistence)
# ============================================================================

"""
    MultiTileCheckpoint

Checkpoint containing multiple tiles for full mission persistence.

# Fields
- `version::String`: File format version
- `timestamp::DateTime`: When checkpoint was created
- `mission_id::String`: Mission that created this checkpoint
- `slam_version::Int`: SLAM state version number
- `tiles::Vector{MapCheckpoint}`: Individual tile checkpoints
- `global_metrics::Dict{String, Float64}`: Mission-level metrics
- `checksum::String`: SHA-256 of serialized tile data

# Usage
```julia
# End of Mission 1
checkpoint = create_multi_tile_checkpoint(slam_state, "mission_001")
save_multi_tile_checkpoint("maps/mission_001.json", checkpoint)

# Start of Mission 2
checkpoint = load_multi_tile_checkpoint("maps/mission_001.json")
enable_online_learning!(est; initial_checkpoint=checkpoint)
```
"""
struct MultiTileCheckpoint
    version::String
    timestamp::DateTime
    mission_id::String
    slam_version::Int
    tiles::Vector{MapCheckpoint}
    global_metrics::Dict{String, Float64}
    checksum::String
end

# ============================================================================
# Checkpoint Creation from SLAM State
# ============================================================================

"""
    create_multi_tile_checkpoint(slam_state::SlamAugmentedState, mission_id::String;
                                  metrics::Dict{String, Float64} = Dict()) -> MultiTileCheckpoint

Create a multi-tile checkpoint from current SLAM state.

# Arguments
- `slam_state`: Current SlamAugmentedState with learned tiles
- `mission_id`: Identifier for this mission
- `metrics`: Optional mission-level metrics (RMSE, NEES, etc.)

# Returns
MultiTileCheckpoint ready for persistence.
"""
function create_multi_tile_checkpoint(slam_state::SlamAugmentedState, mission_id::String;
                                       metrics::Dict{String, Float64} = Dict{String, Float64}())
    tiles = MapCheckpoint[]

    for (tile_id, tile_state) in slam_state.tile_states
        # Create provenance
        provenance = MapProvenance(
            slam_state.version - 1,  # Parent version
            [mission_id],
            tile_state.observation_count,
            UInt64(0),
            :learning,
            "End-of-mission checkpoint"
        )

        # Determine active_dim: the informed subspace dimension.
        # For tiles with n_coef > 8 where Tier2 is not yet active,
        # active_dim = 8 (linear basis only).
        n_coef = length(tile_state.coefficients)
        active_dim = min(n_coef, TIER2_MAX_ACTIVE_DIM)

        checkpoint = MapCheckpoint(
            tile_state.version,
            now(UTC),
            copy(tile_state.coefficients),
            copy(tile_state.covariance),
            tile_id,
            tile_state.observation_count,
            provenance,
            active_dim
        )

        push!(tiles, checkpoint)
    end

    # Compute checksum
    checksum = compute_checkpoint_checksum(tiles)

    return MultiTileCheckpoint(
        CHECKPOINT_FILE_VERSION,
        now(UTC),
        mission_id,
        slam_state.version,
        tiles,
        metrics,
        checksum
    )
end

"""
    compute_checkpoint_checksum(tiles::Vector{MapCheckpoint}) -> String

Compute SHA-256 checksum of tile data for integrity verification.
"""
function compute_checkpoint_checksum(tiles::Vector{MapCheckpoint})
    # Serialize tile data deterministically
    data_parts = String[]
    for tile in sort(tiles, by=t -> (t.tile_id.ix, t.tile_id.iy))
        push!(data_parts, "$(tile.tile_id.ix),$(tile.tile_id.iy)")
        push!(data_parts, join(map(string, tile.coefficients), ","))
        push!(data_parts, "$(tile.observation_count)")
    end

    data_string = join(data_parts, "|")
    hash_bytes = sha256(Vector{UInt8}(data_string))
    return bytes2hex(hash_bytes)
end

# ============================================================================
# File I/O
# ============================================================================

"""
    save_multi_tile_checkpoint(path::String, checkpoint::MultiTileCheckpoint)

Save multi-tile checkpoint to JSON file.

# File Structure
```json
{
    "version": "1.0.0",
    "timestamp": "2026-01-26T10:30:00",
    "mission_id": "mission_001",
    "slam_version": 5,
    "tiles": [...],
    "global_metrics": {...},
    "checksum": "abc123..."
}
```
"""
function save_multi_tile_checkpoint(path::String, checkpoint::MultiTileCheckpoint)
    # Convert to serializable format
    data = Dict{String, Any}(
        "version" => checkpoint.version,
        "timestamp" => string(checkpoint.timestamp),
        "mission_id" => checkpoint.mission_id,
        "slam_version" => checkpoint.slam_version,
        "global_metrics" => checkpoint.global_metrics,
        "checksum" => checkpoint.checksum,
        "tiles" => [serialize_tile_checkpoint(t) for t in checkpoint.tiles]
    )

    # Ensure directory exists
    mkpath(dirname(path))

    # Write JSON
    open(path, "w") do io
        JSON3.pretty(io, data)
    end

    @info "Saved checkpoint" path=path n_tiles=length(checkpoint.tiles) version=checkpoint.slam_version

    return path
end

"""
    load_multi_tile_checkpoint(path::String) -> MultiTileCheckpoint

Load multi-tile checkpoint from JSON file.

# Throws
- `ErrorException` if file doesn't exist
- `ErrorException` if checksum verification fails
- `ErrorException` if version is incompatible
"""
function load_multi_tile_checkpoint(path::String)
    if !isfile(path)
        error("Checkpoint file not found: $path")
    end

    # Read JSON
    data = JSON3.read(read(path, String))

    # Version check
    file_version = get(data, :version, "unknown")
    if !is_compatible_version(file_version, CHECKPOINT_FILE_VERSION)
        error("Incompatible checkpoint version: $file_version (expected: $CHECKPOINT_FILE_VERSION)")
    end

    # Deserialize tiles
    tiles = [deserialize_tile_checkpoint(t) for t in data[:tiles]]

    # Verify checksum
    computed_checksum = compute_checkpoint_checksum(tiles)
    stored_checksum = get(data, :checksum, "")
    if computed_checksum != stored_checksum
        error("Checkpoint checksum mismatch: file may be corrupted")
    end

    # Parse global metrics
    global_metrics = Dict{String, Float64}()
    if haskey(data, :global_metrics)
        for (k, v) in pairs(data[:global_metrics])
            global_metrics[string(k)] = Float64(v)
        end
    end

    return MultiTileCheckpoint(
        string(file_version),
        DateTime(get(data, :timestamp, "2000-01-01T00:00:00")),
        string(get(data, :mission_id, "unknown")),
        Int(get(data, :slam_version, 0)),
        tiles,
        global_metrics,
        stored_checksum
    )
end

"""Check if file version is compatible with current version."""
function is_compatible_version(file_version::String, current_version::String)
    # For now, exact match required. Future: semver comparison.
    file_major = parse(Int, split(file_version, ".")[1])
    current_major = parse(Int, split(current_version, ".")[1])
    return file_major == current_major
end

# ============================================================================
# Tile Serialization Helpers
# ============================================================================

"""Serialize a MapCheckpoint to a Dict for JSON."""
function serialize_tile_checkpoint(tile::MapCheckpoint)
    return Dict{String, Any}(
        "version" => tile.version,
        "timestamp" => string(tile.timestamp),
        "tile_id" => Dict("ix" => tile.tile_id.ix, "iy" => tile.tile_id.iy),
        "coefficients" => collect(tile.coefficients),
        "covariance" => [collect(row) for row in eachrow(tile.covariance)],
        "observation_count" => tile.observation_count,
        "provenance" => Dict(
            "parent_version" => tile.provenance.parent_version,
            "mission_ids" => tile.provenance.mission_ids,
            "update_count" => tile.provenance.update_count,
            "method" => string(tile.provenance.method),
            "notes" => tile.provenance.notes
        ),
        "active_dim" => tile.active_dim
    )
end

"""Deserialize a Dict to MapCheckpoint."""
function deserialize_tile_checkpoint(data)::MapCheckpoint
    tile_id = MapTileID(data[:tile_id][:ix], data[:tile_id][:iy])

    coefficients = Vector{Float64}(data[:coefficients])
    covariance = Matrix{Float64}(hcat([collect(row) for row in data[:covariance]]...)')

    prov_data = data[:provenance]
    provenance = MapProvenance(
        Int(prov_data[:parent_version]),
        Vector{String}(prov_data[:mission_ids]),
        Int(prov_data[:update_count]),
        UInt64(0),
        Symbol(prov_data[:method]),
        String(get(prov_data, :notes, ""))
    )

    # active_dim: default to n_coef for legacy checkpoints without the field
    n_coef = length(coefficients)
    active_dim = Int(get(data, :active_dim, min(n_coef, TIER2_MAX_ACTIVE_DIM)))

    return MapCheckpoint(
        Int(data[:version]),
        DateTime(get(data, :timestamp, "2000-01-01T00:00:00")),
        coefficients,
        covariance,
        tile_id,
        Int(data[:observation_count]),
        provenance,
        active_dim
    )
end

# ============================================================================
# Conversion to SLAM State
# ============================================================================

"""
    checkpoint_to_slam_tiles(checkpoint::MultiTileCheckpoint) -> Dict{MapTileID, SlamTileState}

Convert checkpoint tiles to SlamTileState for initializing OnlineMapProvider.

This is the key function for multi-mission persistence:
- Mission 1 ends: slam_state.tile_states → checkpoint → file
- Mission 2 starts: file → checkpoint → slam_state.tile_states
"""
function checkpoint_to_slam_tiles(checkpoint::MultiTileCheckpoint)
    tile_states = Dict{MapTileID, SlamTileState}()

    for tile_cp in checkpoint.tiles
        # Create SlamTileState from checkpoint
        n_coef = length(tile_cp.coefficients)
        cov = copy(tile_cp.covariance)

        # Enforce cross-block zero invariant on load.
        # Clamp active_dim to current runtime policy: even if a checkpoint was saved
        # with active_k=15 (Tier2 was once active), the current runtime may restrict
        # to TIER2_MAX_ACTIVE_DIM=8. Always use the more restrictive of the two.
        active_k = min(tile_cp.active_dim, TIER2_MAX_ACTIVE_DIM)
        if active_k < n_coef
            active_idx = 1:active_k
            inactive_idx = (active_k+1):n_coef
            cov[active_idx, inactive_idx] .= 0.0
            cov[inactive_idx, active_idx] .= 0.0
        end

        info = inv(cov + 1e-12 * I)  # Regularize

        # Re-enforce cross-block zeros on information matrix too
        if active_k < n_coef
            info[active_idx, inactive_idx] .= 0.0
            info[inactive_idx, active_idx] .= 0.0
        end

        # Estimate center from tile ID (assumes 50m tiles)
        tile_size = 50.0
        center = Vec3Map(
            (tile_cp.tile_id.ix + 0.5) * tile_size,
            (tile_cp.tile_id.iy + 0.5) * tile_size,
            0.0  # Z not stored in checkpoint
        )

        tile_state = SlamTileState(
            tile_cp.tile_id,
            center,
            DEFAULT_TILE_SCALE,  # scale (tile half-width)
            mode_from_dim(length(tile_cp.coefficients)),  # model_mode
            copy(tile_cp.coefficients),
            cov,
            info,
            tile_cp.observation_count,
            0.0,  # last_update_time
            tile_cp.version,
            false,  # Not probationary if loaded from checkpoint
            Vec3Map(0.0, 0.0, 0.0),  # position_bbox_min
            Vec3Map(0.0, 0.0, 0.0),  # position_bbox_max
            false,  # tier2_active (default locked on restore; runtime policy controls)
            0       # tier2_relock_count
        )

        tile_states[tile_cp.tile_id] = tile_state
    end

    return tile_states
end

"""
    restore_slam_state_from_checkpoint!(slam_state::SlamAugmentedState,
                                         checkpoint::MultiTileCheckpoint)

Restore SLAM state tiles from a checkpoint (in-place modification).

# Note
This replaces tile_states but preserves nav_state and source_states.
"""
function restore_slam_state_from_checkpoint!(slam_state::SlamAugmentedState,
                                              checkpoint::MultiTileCheckpoint)
    # Replace tile states
    empty!(slam_state.tile_states)
    for (id, tile) in checkpoint_to_slam_tiles(checkpoint)
        slam_state.tile_states[id] = tile
    end

    # Update version to continue from checkpoint
    slam_state.version = checkpoint.slam_version

    @info "Restored SLAM state from checkpoint" mission_id=checkpoint.mission_id n_tiles=length(slam_state.tile_states) version=slam_state.version

    return slam_state
end

# ============================================================================
# Convenience Functions
# ============================================================================

"""
    checkpoint_exists(path::String) -> Bool

Check if a checkpoint file exists.
"""
checkpoint_exists(path::String) = isfile(path)

"""
    checkpoint_path(base_dir::String, mission_id::String) -> String

Generate standard checkpoint path for a mission.
"""
function checkpoint_path(base_dir::String, mission_id::String)
    return joinpath(base_dir, "checkpoints", "$(mission_id)_checkpoint.json")
end

"""
    list_checkpoints(base_dir::String) -> Vector{String}

List all checkpoint files in a directory.
"""
function list_checkpoints(base_dir::String)
    cp_dir = joinpath(base_dir, "checkpoints")
    if !isdir(cp_dir)
        return String[]
    end
    return filter(f -> endswith(f, "_checkpoint.json"), readdir(cp_dir, join=true))
end

# ============================================================================
# Exports
# ============================================================================

export MultiTileCheckpoint
export create_multi_tile_checkpoint, compute_checkpoint_checksum
export save_multi_tile_checkpoint, load_multi_tile_checkpoint
export checkpoint_to_slam_tiles, restore_slam_state_from_checkpoint!
export checkpoint_exists, checkpoint_path, list_checkpoints
