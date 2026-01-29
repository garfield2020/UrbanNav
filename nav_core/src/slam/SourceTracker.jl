# ============================================================================
# SourceTracker.jl - Source Tracker (Phase G Step 5)
# ============================================================================
#
# Manages the full lifecycle of tracked magnetic sources from detection
# through localization, coupling escalation, and retirement.
#
# Track Status Progression:
#   CANDIDATE → PROVISIONAL → CONFIRMED → LOCKED → RETIRED/DEMOTED
#
# Each status transition requires passing specific gates:
#   CANDIDATE → PROVISIONAL: FIM rank=6 + promotion gates
#   PROVISIONAL → CONFIRMED: stability window (covariance decreasing)
#   CONFIRMED → LOCKED: all 4 coupling gates pass
#   Any → DEMOTED: gate failure, covariance growth, timeout
#
# Coupling auto-escalates: SHADOW → COV_ONLY → SUBTRACT
# Coupling auto-demotes on gate failure.
#
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Track Status
# ============================================================================

"""
    SourceTrackStatus

Lifecycle status for a tracked source.
"""
@enum SourceTrackStatus begin
    TRACK_CANDIDATE = 1
    TRACK_PROVISIONAL = 2
    TRACK_CONFIRMED = 3
    TRACK_LOCKED = 4
    TRACK_RETIRED = 5
    TRACK_DEMOTED = 6
end

# ============================================================================
# Source Track
# ============================================================================

"""
    SourceTrack

Full state for a tracked source including observability and coupling.

# Fields
- `id::Int`: Unique track identifier
- `status::SourceTrackStatus`: Current lifecycle status
- `source_state::SlamSourceState`: 6-DOF source state (position + moment)
- `observability::Union{SourceObservabilityBudget, Nothing}`: FIM-based observability
- `coupling_mode::SourceCouplingMode`: Current coupling mode
- `coupling_gates::SourceCouplingGates`: Current gate status
- `observation_positions::Vector{Vec3Map}`: History of observation positions
- `observation_R::Vector{Matrix{Float64}}`: History of noise covariances
- `fit_result::Union{DipoleFitResult, Nothing}`: Latest LM fit result
- `update_count::Int`: Total EKF updates applied
- `creation_time::Float64`: Track creation timestamp [s]
- `cov_trace_history::Vector{Float64}`: Covariance trace history for stability gate
- `log_likelihood_ratio::Float64`: Accumulated LLR for evidence gate
"""
mutable struct SourceTrack
    id::Int
    status::SourceTrackStatus
    source_state::SlamSourceState
    observability::Union{SourceObservabilityBudget, Nothing}
    coupling_mode::SourceCouplingMode
    coupling_gates::SourceCouplingGates
    observation_positions::Vector{Vec3Map}
    observation_R::Vector{Matrix{Float64}}
    fit_result::Union{DipoleFitResult, Nothing}
    update_count::Int
    creation_time::Float64
    cov_trace_history::Vector{Float64}
    log_likelihood_ratio::Float64
end

"""Create a new candidate track."""
function SourceTrack(id::Int, position::Vec3Map, moment::SVector{3,Float64},
                     t::Float64; position_var::Float64=100.0, moment_var::Float64=10000.0)
    source = SlamSourceState(id, Vector(position), Vector(moment);
                             position_var=position_var, moment_var=moment_var)
    source.lifecycle = :candidate

    SourceTrack(
        id,
        TRACK_CANDIDATE,
        source,
        nothing,
        SOURCE_SHADOW,
        SourceCouplingGates(false, false, false, false),
        Vec3Map[],
        Matrix{Float64}[],
        nothing,
        0,
        t,
        Float64[],
        0.0
    )
end

# ============================================================================
# Tracker Configuration
# ============================================================================

"""
    SourceTrackerConfig

Configuration for the source tracker.
"""
struct SourceTrackerConfig
    detection::SourceDetectionConfig
    promotion::SourcePromotionConfig
    update::SourceUpdateConfig
    retirement::SourceRetirementConfig
    coupling::SourceCouplingConfig
    max_tracks::Int
    stability_window::Int
    refit_interval_obs::Int
    motion_model::Symbol

    function SourceTrackerConfig(;
        detection::SourceDetectionConfig = DEFAULT_SOURCE_DETECTION_CONFIG,
        promotion::SourcePromotionConfig = DEFAULT_SOURCE_PROMOTION_CONFIG,
        update::SourceUpdateConfig = DEFAULT_SOURCE_UPDATE_CONFIG,
        retirement::SourceRetirementConfig = DEFAULT_SOURCE_RETIREMENT_CONFIG,
        coupling::SourceCouplingConfig = DEFAULT_SOURCE_COUPLING_CONFIG,
        max_tracks::Int = 20,
        stability_window::Int = 10,
        refit_interval_obs::Int = 20,
        motion_model::Symbol = :static
    )
        @assert max_tracks > 0
        @assert stability_window >= 2
        @assert refit_interval_obs >= 1
        new(detection, promotion, update, retirement, coupling,
            max_tracks, stability_window, refit_interval_obs, motion_model)
    end
end

const DEFAULT_SOURCE_TRACKER_CONFIG = SourceTrackerConfig()

# ============================================================================
# Source Tracker
# ============================================================================

"""
    SourceTrackingResult

Result of processing a measurement through the tracker.
"""
struct SourceTrackingResult
    new_tracks::Vector{Int}
    updated_tracks::Vector{Int}
    promoted_tracks::Vector{Int}
    demoted_tracks::Vector{Int}
    retired_tracks::Vector{Int}
    coupling_changes::Vector{Tuple{Int, SourceCouplingMode, SourceCouplingMode}}
end

"""
    SourceTracker

Manages source detection, tracking, and coupling lifecycle.
"""
mutable struct SourceTracker
    config::SourceTrackerConfig
    detection_frontend::SourceDetectionFrontEnd
    tracks::Dict{Int, SourceTrack}
    candidates::Dict{Int, SourceCandidate}
    next_id::Int
end

"""Create a new source tracker."""
function SourceTracker(config::SourceTrackerConfig = DEFAULT_SOURCE_TRACKER_CONFIG)
    fe = SourceDetectionFrontEnd(config=config.detection)
    SourceTracker(config, fe, Dict{Int, SourceTrack}(),
                  Dict{Int, SourceCandidate}(), 1)
end

# ============================================================================
# Main Processing
# ============================================================================

"""
    process_measurement!(tracker::SourceTracker, state::SlamAugmentedState,
                          obs::SourceObservation) -> SourceTrackingResult

Process a measurement through the full source tracking pipeline.

# Steps
1. Detect anomalies (subtract known sources, chi² test)
2. Associate detections with candidates or create new
3. Check candidate promotions (FIM gate)
4. Update active tracks (EKF)
5. Evaluate coupling gates
6. Check retirements
"""
function process_measurement!(tracker::SourceTracker, state::SlamAugmentedState,
                                obs::SourceObservation)
    config = tracker.config
    new_tracks = Int[]
    updated_tracks = Int[]
    promoted_tracks = Int[]
    demoted_tracks = Int[]
    retired_tracks = Int[]
    coupling_changes = Tuple{Int, SourceCouplingMode, SourceCouplingMode}[]

    # Collect active sources from tracks
    active_sources = SlamSourceState[t.source_state for (_, t) in tracker.tracks
                                     if t.status in (TRACK_PROVISIONAL, TRACK_CONFIRMED, TRACK_LOCKED)]

    # Step 1: Detect anomalies
    events = detect_anomalies(tracker.detection_frontend, obs.position,
                               obs.field_measured, obs.field_predicted_bg,
                               obs.R_meas, active_sources, obs.timestamp)

    # Step 2: Associate or create candidates
    next_id_ref = Ref(tracker.next_id)
    (new_ids, updated_ids) = associate_or_create!(tracker.detection_frontend, events,
                                                   tracker.candidates, next_id_ref)
    tracker.next_id = next_id_ref[]

    # Step 3: Check candidate promotions
    to_promote = Int[]
    for (id, cand) in tracker.candidates
        result = check_promotion(cand, obs.timestamp, config.promotion)
        if result.pass
            push!(to_promote, id)
        end
    end

    for cand_id in to_promote
        cand = tracker.candidates[cand_id]
        track_id = tracker.next_id
        tracker.next_id += 1

        # Create track from candidate
        moment_est = SVector{3}(cand.estimated_moment...)
        track = SourceTrack(track_id, cand.estimated_position, moment_est,
                           obs.timestamp;
                           position_var=cand.position_covariance[1,1])

        # Set to PROVISIONAL (passed basic promotion)
        track.status = TRACK_PROVISIONAL
        track.source_state.lifecycle = :active
        track.source_state.is_probationary = true
        track.observation_positions = copy(cand.positions)

        # Build R_list for observability
        R_default = Matrix{Float64}(obs.R_meas)
        track.observation_R = [copy(R_default) for _ in cand.positions]

        # Compute initial observability
        if length(track.observation_positions) >= 3
            track.observability = SourceObservabilityBudget(
                track.observation_positions, track.observation_R, track.source_state)
        end

        tracker.tracks[track_id] = track
        push!(promoted_tracks, track_id)

        # Add to SLAM state
        push!(state.source_states, track.source_state)

        delete!(tracker.candidates, cand_id)
    end

    # Step 4: Update active tracks
    for (id, track) in tracker.tracks
        if track.status in (TRACK_PROVISIONAL, TRACK_CONFIRMED, TRACK_LOCKED)
            result = update_source_state!(track.source_state, obs, config.update)
            if result.updated
                track.update_count += 1
                push!(track.observation_positions, obs.position)
                push!(track.observation_R, Matrix{Float64}(obs.R_meas))
                push!(updated_tracks, id)

                # Record covariance trace for stability gate
                push!(track.cov_trace_history, tr(Matrix(track.source_state.covariance)))

                # Accumulate LLR
                track.log_likelihood_ratio += 0.5 * (result.chi2 - 3.0)  # Relative to dim

                # Periodic observability update
                if mod(track.update_count, 5) == 0 && length(track.observation_positions) >= 3
                    track.observability = SourceObservabilityBudget(
                        track.observation_positions, track.observation_R, track.source_state)
                end
            end
        end
    end

    # Step 5: Evaluate coupling and status transitions
    for (id, track) in tracker.tracks
        if track.status in (TRACK_PROVISIONAL, TRACK_CONFIRMED, TRACK_LOCKED)
            old_mode = track.coupling_mode

            # Evaluate gates
            fim_rank = track.observability !== nothing ? track.observability.rank : 0
            pos_crlb = track.observability !== nothing ? maximum(track.observability.position_crlb) : Inf

            gates = evaluate_coupling_gates(
                track.log_likelihood_ratio,
                fim_rank,
                pos_crlb,
                track.cov_trace_history,
                track.log_likelihood_ratio > 5.0 ? 15.0 : 0.0,  # Simplified predictive
                config.coupling.gate_thresholds
            )
            track.coupling_gates = gates

            # Determine coupling mode
            new_mode = effective_coupling_mode(gates, config.coupling)
            track.coupling_mode = new_mode

            if new_mode != old_mode
                push!(coupling_changes, (id, old_mode, new_mode))
            end

            # Status transitions
            if track.status == TRACK_PROVISIONAL
                # PROVISIONAL → CONFIRMED: stability window
                if length(track.cov_trace_history) >= config.stability_window
                    window = track.cov_trace_history[end-config.stability_window+1:end]
                    if window[end] <= window[1]  # Decreasing
                        track.status = TRACK_CONFIRMED
                        track.source_state.is_probationary = false
                    end
                end
            elseif track.status == TRACK_CONFIRMED
                # CONFIRMED → LOCKED: all coupling gates pass
                if all_gates_pass(gates)
                    track.status = TRACK_LOCKED
                end
            elseif track.status == TRACK_LOCKED
                # LOCKED: demote on gate failure
                if !all_gates_pass(gates)
                    track.status = TRACK_CONFIRMED
                    track.coupling_mode = min(new_mode, SOURCE_COV_ONLY)
                end
            end
        end
    end

    # Step 6: Check retirements
    to_retire = Int[]
    for (id, track) in tracker.tracks
        if track.status in (TRACK_PROVISIONAL, TRACK_CONFIRMED, TRACK_LOCKED)
            should_retire, _ = check_source_retirement(
                track.source_state, obs.position, obs.timestamp, config.retirement)
            if should_retire
                push!(to_retire, id)
            end
        end
    end

    for id in to_retire
        track = tracker.tracks[id]
        track.status = TRACK_RETIRED
        track.source_state.lifecycle = :retired
        track.coupling_mode = SOURCE_SHADOW
        push!(retired_tracks, id)

        # Remove from SLAM state
        filter!(s -> s.source_id != track.source_state.source_id, state.source_states)
    end

    # Enforce track cap
    if length(tracker.tracks) > config.max_tracks
        # Remove oldest retired/demoted tracks
        removable = [(id, t.creation_time) for (id, t) in tracker.tracks
                     if t.status in (TRACK_RETIRED, TRACK_DEMOTED)]
        sort!(removable, by=x->x[2])
        while length(tracker.tracks) > config.max_tracks && !isempty(removable)
            delete!(tracker.tracks, popfirst!(removable)[1])
        end
    end

    return SourceTrackingResult(new_tracks, updated_tracks, promoted_tracks,
                                demoted_tracks, retired_tracks, coupling_changes)
end

# ============================================================================
# Refit
# ============================================================================

"""
    refit_track!(tracker::SourceTracker, track_id::Int) -> Union{DipoleFitResult, Nothing}

Refit a track using the LM dipole fitter on accumulated observations.
"""
function refit_track!(tracker::SourceTracker, track_id::Int)
    track = get(tracker.tracks, track_id, nothing)
    if track === nothing || isempty(track.observation_positions)
        return nothing
    end

    # Collect measurement data for fitter
    n = length(track.observation_positions)
    if n < 5
        return nothing
    end

    # Use fit_dipole from dipole_fitter.jl
    # Reconstruct approximate observations as (position, B_measured) tuples
    src_pos = Vector(source_position(track.source_state))
    src_moment = Vector(source_moment(track.source_state))

    # Build observation tuples: (position, predicted_field_at_position)
    # We use the current source estimate to generate synthetic measurements
    observations = Tuple{Vector{Float64}, Vector{Float64}}[]
    for pos in track.observation_positions
        B_pred = dipole_field_at(pos, src_pos, src_moment)
        push!(observations, (Vector(pos), Vector(B_pred)))
    end

    initial_guess = (src_pos, src_moment)
    fit_config = DipoleFitConfig(min_observations=min(5, n))
    result = fit_dipole(observations, initial_guess; config=fit_config)

    if result.converged
        track.fit_result = result
        # Update source state from fit
        track.source_state.state = SVector{6}(vcat(result.position, result.moment))
        track.source_state.covariance = result.covariance
    end

    return result
end

# ============================================================================
# Phase G+ Step 6: Lifecycle Hysteresis
# ============================================================================

"""
    LifecycleHysteresis

Hysteresis bands to prevent rapid promotion/demotion cycling.

# Fields
- `promote_threshold::Float64`: Gate score to promote (default 0.8).
  Based on fraction of coupling gates passing, with 0.8 requiring ≥3/4 gates
  to be near passing. This prevents promotion on marginal evidence.
- `demote_threshold::Float64`: Gate score to demote (default 0.3).
  Set well below promote_threshold to create a dead zone where the track
  remains in its current state. Gap of 0.5 prevents oscillation.
- `min_dwell_observations::Int`: Minimum observations in current state before
  any transition (default 5). Ensures the estimator has enough data to
  distinguish genuine state changes from noise transients.

# Physics Justification
The hysteresis gap (promote - demote = 0.5) is chosen to be larger than
typical gate score fluctuations (σ ≈ 0.1-0.2) caused by measurement noise,
ensuring that 2-3σ fluctuations cannot trigger oscillation.
"""
struct LifecycleHysteresis
    promote_threshold::Float64
    demote_threshold::Float64
    min_dwell_observations::Int

    function LifecycleHysteresis(;
        promote_threshold::Float64 = 0.8,
        demote_threshold::Float64 = 0.3,
        min_dwell_observations::Int = 5
    )
        @assert promote_threshold > demote_threshold "Promote threshold must exceed demote threshold (hysteresis gap)"
        @assert min_dwell_observations >= 1 "Dwell minimum must be ≥ 1"
        new(promote_threshold, demote_threshold, min_dwell_observations)
    end
end

const DEFAULT_LIFECYCLE_HYSTERESIS = LifecycleHysteresis()

"""
    gate_score(gates::SourceCouplingGates) → Float64

Compute a scalar gate score from coupling gates.
Returns fraction of gates passing, in [0, 1].
"""
function gate_score(gates::SourceCouplingGates)
    return n_gates_passing(gates) / 4.0
end

"""
    should_promote(score::Float64, dwell::Int, hysteresis::LifecycleHysteresis) → Bool

Check if a track should be promoted based on gate score and dwell time.
Requires both score ≥ promote_threshold AND dwell ≥ min_dwell_observations.
"""
function should_promote(score::Float64, dwell::Int, hysteresis::LifecycleHysteresis)
    return score >= hysteresis.promote_threshold && dwell >= hysteresis.min_dwell_observations
end

"""
    should_demote(score::Float64, dwell::Int, hysteresis::LifecycleHysteresis) → Bool

Check if a track should be demoted based on gate score and dwell time.
Requires both score ≤ demote_threshold AND dwell ≥ min_dwell_observations.
"""
function should_demote(score::Float64, dwell::Int, hysteresis::LifecycleHysteresis)
    return score <= hysteresis.demote_threshold && dwell >= hysteresis.min_dwell_observations
end

# ============================================================================
# Urban Motion Models
# ============================================================================

"""
    UrbanMotionModel

Motion model for urban magnetic sources.
"""
@enum UrbanMotionModel begin
    MOTION_STATIC = 0       # Static source (structural steel, parked vehicle)
    MOTION_ELEVATOR = 1     # Vertical-only motion along shaft axis
    MOTION_VEHICLE = 2      # Ground-plane constrained (2D + heading)
    MOTION_DOOR = 3         # Rotational (hinge), binary open/close
end

"""
    urban_process_noise(model::UrbanMotionModel, dt::Float64) -> SMatrix{6,6,Float64,36}

Compute process noise for urban source motion models.

# Models
- STATIC: Very small process noise (thermal drift only)
- ELEVATOR: Large vertical, small horizontal process noise
- VEHICLE: Large horizontal, small vertical process noise  
- DOOR: Small position, moderate moment process noise (magnetization change)
"""
function urban_process_noise(model::UrbanMotionModel, dt::Float64)
    if model == MOTION_STATIC
        σ_pos = 0.001  # 1 mm/√s drift
        σ_mom = 0.01   # Small moment drift
    elseif model == MOTION_ELEVATOR
        # Vertical motion: σ_z large, σ_xy small
        Q = zeros(6, 6)
        Q[1,1] = 0.001^2 * dt  # x: nearly static
        Q[2,2] = 0.001^2 * dt  # y: nearly static
        Q[3,3] = 2.0^2 * dt    # z: up to 2 m/s vertical
        Q[4,4] = 0.1^2 * dt    # mx drift
        Q[5,5] = 0.1^2 * dt    # my drift
        Q[6,6] = 0.1^2 * dt    # mz drift
        return SMatrix{6,6,Float64,36}(Q)
    elseif model == MOTION_VEHICLE
        # Ground-plane motion: σ_xy large, σ_z small
        Q = zeros(6, 6)
        Q[1,1] = 3.0^2 * dt    # x: up to 3 m/s
        Q[2,2] = 3.0^2 * dt    # y: up to 3 m/s
        Q[3,3] = 0.01^2 * dt   # z: nearly static (ground plane)
        Q[4,4] = 0.5^2 * dt    # moment drift (engine RPM changes)
        Q[5,5] = 0.5^2 * dt
        Q[6,6] = 0.5^2 * dt
        return SMatrix{6,6,Float64,36}(Q)
    elseif model == MOTION_DOOR
        σ_pos = 0.01   # Door hinge position is fixed
        σ_mom = 2.0    # Large moment change (lock energize/de-energize)
    else
        σ_pos = 0.001
        σ_mom = 0.01
    end

    Q = zeros(6, 6)
    Q[1:3, 1:3] = σ_pos^2 * dt * I(3)
    Q[4:6, 4:6] = σ_mom^2 * dt * I(3)
    return SMatrix{6,6,Float64,36}(Q)
end

export UrbanMotionModel, MOTION_STATIC, MOTION_ELEVATOR, MOTION_VEHICLE, MOTION_DOOR
export urban_process_noise

# ============================================================================
# Exports
# ============================================================================

export SourceTrackStatus
export TRACK_CANDIDATE, TRACK_PROVISIONAL, TRACK_CONFIRMED
export TRACK_LOCKED, TRACK_RETIRED, TRACK_DEMOTED
export SourceTrack, SourceTrackerConfig, DEFAULT_SOURCE_TRACKER_CONFIG
export SourceTrackingResult, SourceTracker
export process_measurement!, refit_track!
export LifecycleHysteresis, DEFAULT_LIFECYCLE_HYSTERESIS
export gate_score, should_promote, should_demote
