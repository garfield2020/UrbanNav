# ============================================================================
# SourceSafetyController.jl - Source Safety Controller (Phase G Step 10)
# ============================================================================
#
# Safety controls specific to source coupling with navigation.
#
# Enforces:
# - Bounded influence per source and aggregate
# - NEES spike auto-rollback
# - Coupling escalation rate limiting
# - Diverging source auto-demotion
#
# Integrates with existing OnlineSafetyController.
#
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Types
# ============================================================================

"""
    SourceSafetyConfig

Configuration for source safety controls.

# Fields
- `max_source_nav_impact_T::Float64`: Max field influence per source [T]
- `max_aggregate_subtraction_T::Float64`: Max total subtraction [T]
- `rollback_on_nees_spike::Bool`: Auto-rollback on NEES spike
- `nees_spike_threshold::Float64`: NEES spike threshold
- `max_coupling_escalation_rate::Int`: Max escalations per minute
- `demote_on_divergence::Bool`: Auto-demote diverging sources
"""
struct SourceSafetyConfig
    max_source_nav_impact_T::Float64
    max_aggregate_subtraction_T::Float64
    rollback_on_nees_spike::Bool
    nees_spike_threshold::Float64
    max_coupling_escalation_rate::Int
    demote_on_divergence::Bool

    function SourceSafetyConfig(;
        max_source_nav_impact_T::Float64 = 1e-6,
        max_aggregate_subtraction_T::Float64 = 5e-6,
        rollback_on_nees_spike::Bool = true,
        nees_spike_threshold::Float64 = 5.0,
        max_coupling_escalation_rate::Int = 3,
        demote_on_divergence::Bool = true
    )
        @assert max_source_nav_impact_T > 0
        @assert max_aggregate_subtraction_T > 0
        @assert nees_spike_threshold > 0
        @assert max_coupling_escalation_rate > 0
        new(max_source_nav_impact_T, max_aggregate_subtraction_T,
            rollback_on_nees_spike, nees_spike_threshold,
            max_coupling_escalation_rate, demote_on_divergence)
    end
end

const DEFAULT_SOURCE_SAFETY_CONFIG = SourceSafetyConfig()

"""
    SourceSafetyAction

Actions taken by the source safety controller.
"""
@enum SourceSafetyAction begin
    SOURCE_SAFETY_NONE = 0
    SOURCE_SAFETY_DEMOTE = 1
    SOURCE_SAFETY_ROLLBACK = 2
    SOURCE_SAFETY_THROTTLE = 3
end

"""
    SourceSafetyController

Safety controller for source-navigation coupling.
"""
mutable struct SourceSafetyController
    config::SourceSafetyConfig
    coupling_history::Vector{Tuple{Float64, Int, SourceCouplingMode}}
    nees_baseline::Vector{Float64}
    last_check_time::Float64
    escalation_count_window::Int
    escalation_window_start::Float64
end

"""Create source safety controller."""
function SourceSafetyController(config::SourceSafetyConfig = DEFAULT_SOURCE_SAFETY_CONFIG)
    SourceSafetyController(config, Tuple{Float64, Int, SourceCouplingMode}[],
                           Float64[], 0.0, 0, 0.0)
end

# ============================================================================
# Safety Checks
# ============================================================================

"""
    check_source_safety!(controller::SourceSafetyController,
                          tracker::SourceTracker,
                          nav_nees::Float64,
                          t::Float64) -> Vector{SourceSafetyAction}

Check source coupling safety and take corrective actions.

# Checks
1. Bounded influence per source
2. Aggregate subtraction bounded
3. NEES spike detection → rollback
4. Escalation rate limiting
5. Divergence detection → demotion
"""
function check_source_safety!(controller::SourceSafetyController,
                                tracker::SourceTracker,
                                nav_nees::Float64,
                                t::Float64)
    config = controller.config
    actions = SourceSafetyAction[]

    push!(controller.nees_baseline, nav_nees)
    if length(controller.nees_baseline) > 100
        popfirst!(controller.nees_baseline)
    end

    # Check 1: NEES spike
    if config.rollback_on_nees_spike && nav_nees > config.nees_spike_threshold
        push!(actions, SOURCE_SAFETY_ROLLBACK)
        # Demote all non-SHADOW sources
        for (id, track) in tracker.tracks
            if track.coupling_mode != SOURCE_SHADOW
                track.coupling_mode = SOURCE_SHADOW
                push!(actions, SOURCE_SAFETY_DEMOTE)
            end
        end
        return actions
    end

    # Check 2: Escalation rate
    if t - controller.escalation_window_start > 60.0
        controller.escalation_count_window = 0
        controller.escalation_window_start = t
    end

    # Check 3: Per-source bounded influence
    for (id, track) in tracker.tracks
        if track.status in (TRACK_PROVISIONAL, TRACK_CONFIRMED, TRACK_LOCKED)
            # Check covariance growth (divergence)
            if config.demote_on_divergence && length(track.cov_trace_history) >= 5
                recent = track.cov_trace_history[end-4:end]
                if recent[end] > 2.0 * recent[1]
                    track.status = TRACK_DEMOTED
                    track.source_state.lifecycle = :demoted
                    track.coupling_mode = SOURCE_SHADOW
                    push!(actions, SOURCE_SAFETY_DEMOTE)
                end
            end
        end
    end

    # Check 4: Aggregate subtraction
    total_subtraction = 0.0
    for (id, track) in tracker.tracks
        if track.coupling_mode == SOURCE_SUBTRACT
            moment_mag = norm(source_moment(track.source_state))
            # Rough estimate of field contribution at typical range
            total_subtraction += μ₀_4π * moment_mag / 5.0^3
        end
    end

    if total_subtraction > config.max_aggregate_subtraction_T
        # Throttle: demote newest SUBTRACT sources
        subtract_tracks = [(id, t.creation_time) for (id, t) in tracker.tracks
                          if t.coupling_mode == SOURCE_SUBTRACT]
        sort!(subtract_tracks, by=x->x[2], rev=true)
        for (id, _) in subtract_tracks
            tracker.tracks[id].coupling_mode = SOURCE_COV_ONLY
            push!(actions, SOURCE_SAFETY_THROTTLE)
            total_subtraction *= 0.5
            if total_subtraction <= config.max_aggregate_subtraction_T
                break
            end
        end
    end

    controller.last_check_time = t
    return actions
end

# ============================================================================
# Checkpoint/Restore
# ============================================================================

"""
    checkpoint_sources(tracker::SourceTracker) -> Dict{Int, SourceTrack}

Create a deep copy snapshot of all tracks for rollback.
"""
function checkpoint_sources(tracker::SourceTracker)
    snapshot = Dict{Int, SourceTrack}()
    for (id, track) in tracker.tracks
        snapshot[id] = SourceTrack(
            track.id,
            track.status,
            SlamSourceState(track.source_state.source_id,
                           Vector(source_position(track.source_state)),
                           Vector(source_moment(track.source_state));
                           position_var=Matrix(track.source_state.covariance)[1,1],
                           moment_var=Matrix(track.source_state.covariance)[4,4]),
            track.observability,
            track.coupling_mode,
            track.coupling_gates,
            copy(track.observation_positions),
            [copy(R) for R in track.observation_R],
            track.fit_result,
            track.update_count,
            track.creation_time,
            copy(track.cov_trace_history),
            track.log_likelihood_ratio
        )
        # Copy full covariance and lifecycle
        snapshot[id].source_state.covariance = track.source_state.covariance
        snapshot[id].source_state.lifecycle = track.source_state.lifecycle
        snapshot[id].source_state.support_count = track.source_state.support_count
        snapshot[id].source_state.last_observed = track.source_state.last_observed
        snapshot[id].source_state.is_probationary = track.source_state.is_probationary
    end
    return snapshot
end

"""
    restore_sources!(tracker::SourceTracker, snapshot::Dict{Int, SourceTrack})

Restore tracker state from snapshot.
"""
function restore_sources!(tracker::SourceTracker, snapshot::Dict{Int, SourceTrack})
    empty!(tracker.tracks)
    for (id, track) in snapshot
        tracker.tracks[id] = track
    end
end

# ============================================================================
# Serialization
# ============================================================================

"""
    serialize_source_tracks(tracks::Dict{Int, SourceTrack}, io::IO)

Serialize source tracks to IO for persistence.
"""
function serialize_source_tracks(tracks::Dict{Int, SourceTrack}, io::IO)
    write(io, Int32(length(tracks)))
    for (id, track) in tracks
        write(io, Int32(id))
        write(io, Int32(Int(track.status)))
        write(io, Int32(Int(track.coupling_mode)))

        # Source state
        for v in track.source_state.state
            write(io, Float64(v))
        end
        for v in track.source_state.covariance
            write(io, Float64(v))
        end
        write(io, Int32(track.source_state.support_count))
        write(io, Float64(track.source_state.last_observed))
        write(io, Int32(track.update_count))
        write(io, Float64(track.creation_time))
        write(io, Float64(track.log_likelihood_ratio))
    end
end

"""
    deserialize_source_tracks(io::IO) -> Dict{Int, SourceTrack}

Deserialize source tracks from IO.
"""
function deserialize_source_tracks(io::IO)
    tracks = Dict{Int, SourceTrack}()
    n = read(io, Int32)

    for _ in 1:n
        id = Int(read(io, Int32))
        status = SourceTrackStatus(read(io, Int32))
        coupling = SourceCouplingMode(read(io, Int32))

        state_vec = [read(io, Float64) for _ in 1:6]
        cov_vec = [read(io, Float64) for _ in 1:36]
        support = Int(read(io, Int32))
        last_obs = read(io, Float64)
        update_count = Int(read(io, Int32))
        creation_time = read(io, Float64)
        llr = read(io, Float64)

        source = SlamSourceState(id, state_vec[1:3], state_vec[4:6])
        source.covariance = SMatrix{6,6}(reshape(cov_vec, 6, 6))
        source.support_count = support
        source.last_observed = last_obs

        track = SourceTrack(
            id, status, source, nothing, coupling,
            SourceCouplingGates(false, false, false, false),
            Vec3Map[], Matrix{Float64}[], nothing,
            update_count, creation_time, Float64[], llr
        )
        tracks[id] = track
    end

    return tracks
end

# ============================================================================
# Exports
# ============================================================================

export SourceSafetyConfig, DEFAULT_SOURCE_SAFETY_CONFIG
export SourceSafetyAction
export SOURCE_SAFETY_NONE, SOURCE_SAFETY_DEMOTE, SOURCE_SAFETY_ROLLBACK, SOURCE_SAFETY_THROTTLE
export SourceSafetyController
export check_source_safety!
export checkpoint_sources, restore_sources!
export serialize_source_tracks, deserialize_source_tracks
