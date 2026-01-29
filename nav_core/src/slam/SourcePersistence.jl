# ============================================================================
# SourcePersistence.jl - Source Persistence (Phase G Step 10)
# ============================================================================
#
# Provides checkpoint/restore and serialization for source tracks
# to support mission restart and multi-mission source persistence.
#
# Extends SlamCheckpoint with source track state.
#
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Extended Checkpoint
# ============================================================================

"""
    SourceCheckpoint

Extended checkpoint including full source track state.

# Fields
- `slam_checkpoint::SlamCheckpoint`: Base SLAM checkpoint
- `track_snapshot::Dict{Int, SourceTrack}`: Source track state snapshot
- `tracker_next_id::Int`: Next track ID
- `timestamp::Float64`: Checkpoint timestamp [s]
"""
struct SourceCheckpoint
    slam_checkpoint::SlamCheckpoint
    track_snapshot::Dict{Int, SourceTrack}
    tracker_next_id::Int
    timestamp::Float64
end

"""
    create_source_checkpoint(state::SlamAugmentedState, tracker::SourceTracker,
                              nees::Float64) -> SourceCheckpoint

Create a checkpoint of both SLAM state and source tracks.
"""
function create_source_checkpoint(state::SlamAugmentedState, tracker::SourceTracker,
                                   nees::Float64)
    slam_cp = create_checkpoint(state, nees)
    track_snap = checkpoint_sources(tracker)

    SourceCheckpoint(slam_cp, track_snap, tracker.next_id, state.timestamp)
end

"""
    restore_source_checkpoint!(state::SlamAugmentedState, tracker::SourceTracker,
                                cp::SourceCheckpoint)

Restore SLAM state and source tracks from checkpoint.
"""
function restore_source_checkpoint!(state::SlamAugmentedState, tracker::SourceTracker,
                                     cp::SourceCheckpoint)
    restore_from_checkpoint!(state, cp.slam_checkpoint)
    restore_sources!(tracker, cp.track_snapshot)
    tracker.next_id = cp.tracker_next_id
end

# ============================================================================
# File Persistence
# ============================================================================

"""
    save_source_tracks(filepath::String, tracker::SourceTracker)

Save source tracks to file.
"""
function save_source_tracks(filepath::String, tracker::SourceTracker)
    open(filepath, "w") do io
        write(io, Int32(tracker.next_id))
        serialize_source_tracks(tracker.tracks, io)
    end
end

"""
    load_source_tracks!(filepath::String, tracker::SourceTracker)

Load source tracks from file into tracker.
"""
function load_source_tracks!(filepath::String, tracker::SourceTracker)
    open(filepath, "r") do io
        tracker.next_id = Int(read(io, Int32))
        tracker.tracks = deserialize_source_tracks(io)
    end
end

# ============================================================================
# Exports
# ============================================================================

export SourceCheckpoint
export create_source_checkpoint, restore_source_checkpoint!
export save_source_tracks, load_source_tracks!
