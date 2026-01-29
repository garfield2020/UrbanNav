# ============================================================================
# OnlineManifoldCollapse.jl - Real-Time Manifold Collapse Tracking (Phase C Step 7)
# ============================================================================
#
# Extends Phase B manifold collapse metrics for real-time online tracking.
#
# Key Additions:
# 1. Incremental statistics (no need to store full trajectory)
# 2. Real-time convergence detection during mission
# 3. Joint state dimension tracking (nav + map + sources)
# 4. Source-map correlation monitoring
# 5. Live collapse rate estimation
#
# Integration:
# - Called by OnlineSlamScheduler on slow loop
# - Provides convergence signals for adaptive behavior
# - Feeds into safety controller for anomaly detection
#
# ============================================================================

using LinearAlgebra
using Statistics: mean, var

# ============================================================================
# Real-Time Collapse Tracker Configuration
# ============================================================================

"""
    OnlineCollapseConfig

Configuration for real-time manifold collapse tracking.

# Fields
- `snapshot_interval_obs::Int`: Observations between snapshots
- `window_size::Int`: Sliding window size for rate estimation
- `min_reduction_rate::Float64`: Minimum required reduction rate
- `max_reduction_rate::Float64`: Maximum healthy reduction rate
- `stall_threshold::Float64`: Threshold for detecting learning stall
- `divergence_threshold::Float64`: Threshold for detecting divergence

# Physics Justification
- snapshot_interval_obs = 10: Captures dynamics without overhead
- window_size = 50: ~1 minute at 1 Hz keyframes
- min_reduction_rate: Below this, learning is stalled
- max_reduction_rate: Above this, learning may be overfitting
"""
struct OnlineCollapseConfig
    snapshot_interval_obs::Int
    window_size::Int
    min_reduction_rate::Float64
    max_reduction_rate::Float64
    stall_threshold::Float64
    divergence_threshold::Float64

    function OnlineCollapseConfig(;
        snapshot_interval_obs::Int = 10,
        window_size::Int = 50,
        min_reduction_rate::Float64 = 0.001,
        max_reduction_rate::Float64 = 0.5,
        stall_threshold::Float64 = 0.0001,
        divergence_threshold::Float64 = 1.5
    )
        @assert snapshot_interval_obs > 0
        @assert window_size > 0
        @assert min_reduction_rate >= 0
        @assert max_reduction_rate > min_reduction_rate
        @assert stall_threshold >= 0
        @assert divergence_threshold > 1

        new(snapshot_interval_obs, window_size, min_reduction_rate,
            max_reduction_rate, stall_threshold, divergence_threshold)
    end
end

const DEFAULT_ONLINE_COLLAPSE_CONFIG = OnlineCollapseConfig()

# ============================================================================
# Incremental Statistics
# ============================================================================

"""
    IncrementalStatistics

Running statistics for online computation.

Uses Welford's algorithm for numerical stability.
"""
mutable struct IncrementalStatistics
    n::Int
    mean::Float64
    M2::Float64  # Sum of squared differences from mean
    min_val::Float64
    max_val::Float64
    last_val::Float64
end

function IncrementalStatistics()
    IncrementalStatistics(0, 0.0, 0.0, Inf, -Inf, 0.0)
end

"""Update statistics with new value (Welford's algorithm)."""
function update_stats!(stats::IncrementalStatistics, x::Float64)
    stats.n += 1
    delta = x - stats.mean
    stats.mean += delta / stats.n
    delta2 = x - stats.mean
    stats.M2 += delta * delta2
    stats.min_val = min(stats.min_val, x)
    stats.max_val = max(stats.max_val, x)
    stats.last_val = x
    return stats
end

"""Get variance from incremental statistics."""
function get_variance(stats::IncrementalStatistics)
    stats.n < 2 ? 0.0 : stats.M2 / (stats.n - 1)
end

"""Get standard deviation."""
function get_std(stats::IncrementalStatistics)
    sqrt(get_variance(stats))
end

# ============================================================================
# Online Collapse State
# ============================================================================

"""
    CollapseState

Current state of manifold collapse.
"""
@enum CollapseState begin
    COLLAPSE_INACTIVE = 0      # No learning active
    COLLAPSE_LEARNING = 1      # Active learning, healthy
    COLLAPSE_CONVERGING = 2    # Near convergence
    COLLAPSE_CONVERGED = 3     # Converged (minimal further reduction)
    COLLAPSE_STALLED = 4       # Learning stalled
    COLLAPSE_DIVERGING = 5     # Covariance growing (bad)
end

# ============================================================================
# Online Collapse Snapshot
# ============================================================================

"""
    OnlineCollapseSnapshot

Lightweight snapshot for online tracking.

Smaller footprint than Phase B CollapseSnapshot - stores only
what's needed for real-time monitoring.
"""
struct OnlineCollapseSnapshot
    timestamp::Float64
    observation_count::Int

    # Covariance metrics (scalars, not full matrix)
    trace_nav::Float64      # Navigation covariance trace
    trace_map::Float64      # Map covariance trace (total)
    trace_sources::Float64  # Source covariance trace (total)

    # Uncertainty bounds
    position_std::Float64   # √(P_pos[1,1] + P_pos[2,2] + P_pos[3,3])
    heading_std::Float64    # Heading uncertainty [rad]

    # Performance metrics
    rmse_recent::Float64    # RMSE over recent window
    nees_recent::Float64    # NEES over recent window

    # Learning indicators
    map_update_rate::Float64    # Updates per observation
    source_count::Int           # Active sources
end

"""Create snapshot from SLAM state."""
function OnlineCollapseSnapshot(timestamp::Float64, obs_count::Int,
                                 nav_P::AbstractMatrix, map_trace::Float64,
                                 source_trace::Float64, rmse::Float64, nees::Float64,
                                 map_updates::Int, n_sources::Int)
    # Extract position uncertainty
    pos_var = nav_P[1,1] + nav_P[2,2] + nav_P[3,3]
    pos_std = sqrt(max(pos_var, 0.0))

    # Extract heading uncertainty (simplified - assumes attitude in indices 7:9)
    heading_var = size(nav_P, 1) >= 9 ? nav_P[9,9] : 0.0
    heading_std = sqrt(max(heading_var, 0.0))

    # Map update rate (updates per observation)
    update_rate = obs_count > 0 ? map_updates / obs_count : 0.0

    OnlineCollapseSnapshot(
        timestamp, obs_count,
        tr(nav_P), map_trace, source_trace,
        pos_std, heading_std,
        rmse, nees,
        update_rate, n_sources
    )
end

# ============================================================================
# Online Collapse Tracker
# ============================================================================

"""
    OnlineCollapseTracker

Real-time tracker for manifold collapse.

# Architecture
- Maintains sliding window of recent snapshots
- Computes incremental statistics without storing full history
- Detects convergence, stalls, and divergence in real-time
- Provides signals for adaptive SLAM behavior

# Usage
```julia
tracker = OnlineCollapseTracker()

# On each slow loop:
status = update_tracker!(tracker, slam_state, rmse, nees)

if status.state == COLLAPSE_CONVERGED
    # Could reduce update frequency or freeze map
end

if status.state == COLLAPSE_DIVERGING
    # Trigger rollback or alert
end
```
"""
mutable struct OnlineCollapseTracker
    config::OnlineCollapseConfig

    # Sliding window of recent snapshots
    recent_snapshots::Vector{OnlineCollapseSnapshot}

    # Incremental statistics (full history)
    trace_stats::IncrementalStatistics
    rmse_stats::IncrementalStatistics
    nees_stats::IncrementalStatistics

    # Baseline (first observation or reset point)
    baseline_trace::Float64
    baseline_timestamp::Float64

    # Current state
    state::CollapseState
    observations_since_last_snapshot::Int
    total_observations::Int
    total_map_updates::Int

    # Convergence tracking
    convergence_progress::Float64  # 0 to 1
    last_significant_reduction::Float64  # timestamp
end

"""Create online collapse tracker."""
function OnlineCollapseTracker(config::OnlineCollapseConfig = DEFAULT_ONLINE_COLLAPSE_CONFIG)
    OnlineCollapseTracker(
        config,
        OnlineCollapseSnapshot[],
        IncrementalStatistics(),
        IncrementalStatistics(),
        IncrementalStatistics(),
        0.0, 0.0,
        COLLAPSE_INACTIVE,
        0, 0, 0,
        0.0, 0.0
    )
end

"""Reset tracker to initial state."""
function reset_tracker!(tracker::OnlineCollapseTracker)
    empty!(tracker.recent_snapshots)
    tracker.trace_stats = IncrementalStatistics()
    tracker.rmse_stats = IncrementalStatistics()
    tracker.nees_stats = IncrementalStatistics()
    tracker.baseline_trace = 0.0
    tracker.baseline_timestamp = 0.0
    tracker.state = COLLAPSE_INACTIVE
    tracker.observations_since_last_snapshot = 0
    tracker.total_observations = 0
    tracker.total_map_updates = 0
    tracker.convergence_progress = 0.0
    tracker.last_significant_reduction = 0.0
    return tracker
end

# ============================================================================
# Tracker Update
# ============================================================================

"""
    OnlineCollapseStatus

Status returned from tracker update.
"""
struct OnlineCollapseStatus
    state::CollapseState
    trace_total::Float64
    trace_reduction_pct::Float64
    convergence_progress::Float64
    reduction_rate::Float64
    mean_nees::Float64
    is_healthy::Bool
    alert::Symbol  # :none, :stalled, :diverging, :converged
end

"""
    update_tracker!(tracker::OnlineCollapseTracker, slam_state::SlamAugmentedState,
                    nav_P::AbstractMatrix, rmse::Float64, nees::Float64,
                    timestamp::Float64, map_updates_this_cycle::Int)

Update tracker with new observation.

# Arguments
- `slam_state`: Current SLAM augmented state
- `nav_P`: Navigation covariance (15×15)
- `rmse`: Recent position RMSE [m]
- `nees`: Recent NEES value
- `timestamp`: Current time [s]
- `map_updates_this_cycle`: Number of map updates since last call
"""
function update_tracker!(tracker::OnlineCollapseTracker, slam_state::SlamAugmentedState,
                          nav_P::AbstractMatrix, rmse::Float64, nees::Float64,
                          timestamp::Float64, map_updates_this_cycle::Int)
    config = tracker.config

    tracker.total_observations += 1
    tracker.observations_since_last_snapshot += 1
    tracker.total_map_updates += map_updates_this_cycle

    # Compute current traces
    trace_nav = tr(nav_P)
    trace_map = sum(tr(t.covariance) for t in values(slam_state.tile_states); init=0.0)
    trace_sources = sum(tr(Matrix(s.covariance)) for s in slam_state.source_states; init=0.0)
    trace_total = trace_nav + trace_map + trace_sources

    # Initialize baseline if needed
    if tracker.baseline_trace == 0.0
        tracker.baseline_trace = trace_total
        tracker.baseline_timestamp = timestamp
        tracker.state = COLLAPSE_LEARNING
    end

    # Update incremental statistics
    update_stats!(tracker.trace_stats, trace_total)
    update_stats!(tracker.rmse_stats, rmse)
    update_stats!(tracker.nees_stats, nees)

    # Create snapshot if interval reached
    if tracker.observations_since_last_snapshot >= config.snapshot_interval_obs
        snap = OnlineCollapseSnapshot(
            timestamp, tracker.total_observations,
            nav_P, trace_map, trace_sources,
            rmse, nees,
            tracker.total_map_updates, length(slam_state.source_states)
        )

        push!(tracker.recent_snapshots, snap)
        tracker.observations_since_last_snapshot = 0

        # Maintain sliding window
        while length(tracker.recent_snapshots) > config.window_size
            popfirst!(tracker.recent_snapshots)
        end
    end

    # Compute trace reduction
    trace_reduction_pct = (tracker.baseline_trace - trace_total) / tracker.baseline_trace * 100

    # Compute reduction rate from sliding window
    reduction_rate = compute_reduction_rate(tracker)

    # Update convergence progress (0 to 1)
    # Based on trace reduction toward some target
    target_reduction = 90.0  # Target 90% reduction for "converged"
    tracker.convergence_progress = min(trace_reduction_pct / target_reduction, 1.0)

    # Track last significant reduction
    if reduction_rate > config.stall_threshold
        tracker.last_significant_reduction = timestamp
    end

    # Determine state and alerts
    state, alert = determine_collapse_state(tracker, trace_total, reduction_rate, timestamp)
    tracker.state = state

    # Check health
    is_healthy = (state == COLLAPSE_LEARNING || state == COLLAPSE_CONVERGING ||
                  state == COLLAPSE_CONVERGED)

    return OnlineCollapseStatus(
        state,
        trace_total,
        trace_reduction_pct,
        tracker.convergence_progress,
        reduction_rate,
        tracker.nees_stats.mean,
        is_healthy,
        alert
    )
end

"""Compute reduction rate from sliding window."""
function compute_reduction_rate(tracker::OnlineCollapseTracker)
    snaps = tracker.recent_snapshots
    n = length(snaps)

    if n < 2
        return 0.0
    end

    # Linear regression of trace over observation count
    first_snap = snaps[1]
    last_snap = snaps[end]

    Δ_obs = last_snap.observation_count - first_snap.observation_count
    if Δ_obs == 0
        return 0.0
    end

    trace_first = first_snap.trace_nav + first_snap.trace_map + first_snap.trace_sources
    trace_last = last_snap.trace_nav + last_snap.trace_map + last_snap.trace_sources

    # Rate: (trace_first - trace_last) / Δ_obs
    # Positive = reduction (good)
    rate = (trace_first - trace_last) / Δ_obs

    return rate
end

"""Determine collapse state and alert from current metrics."""
function determine_collapse_state(tracker::OnlineCollapseTracker, trace_total::Float64,
                                   reduction_rate::Float64, timestamp::Float64)
    config = tracker.config

    # Check for divergence (trace growing)
    if trace_total > tracker.baseline_trace * config.divergence_threshold
        return (COLLAPSE_DIVERGING, :diverging)
    end

    # Check for stall (no reduction for extended period)
    time_since_reduction = timestamp - tracker.last_significant_reduction
    if time_since_reduction > 30.0 && reduction_rate < config.stall_threshold
        return (COLLAPSE_STALLED, :stalled)
    end

    # Check for convergence (high progress, low rate)
    if tracker.convergence_progress > 0.9 && reduction_rate < config.min_reduction_rate
        return (COLLAPSE_CONVERGED, :converged)
    end

    # Check for converging (moderate progress)
    if tracker.convergence_progress > 0.5
        return (COLLAPSE_CONVERGING, :none)
    end

    # Normal learning
    return (COLLAPSE_LEARNING, :none)
end

# ============================================================================
# Real-Time Metrics
# ============================================================================

"""
    RealTimeCollapseMetrics

Metrics available in real-time (without storing full history).
"""
struct RealTimeCollapseMetrics
    # Total observations and updates
    total_observations::Int
    total_map_updates::Int
    update_acceptance_rate::Float64

    # Trace metrics
    baseline_trace::Float64
    current_trace::Float64
    trace_reduction_pct::Float64

    # Statistics
    mean_rmse::Float64
    std_rmse::Float64
    mean_nees::Float64
    std_nees::Float64

    # Rates
    reduction_rate::Float64
    convergence_progress::Float64

    # State
    state::CollapseState
    time_in_state::Float64
end

"""Get current real-time metrics from tracker."""
function get_realtime_metrics(tracker::OnlineCollapseTracker, timestamp::Float64)
    # Compute current trace from most recent snapshot
    current_trace = if !isempty(tracker.recent_snapshots)
        s = tracker.recent_snapshots[end]
        s.trace_nav + s.trace_map + s.trace_sources
    else
        tracker.baseline_trace
    end

    # Trace reduction
    trace_reduction = tracker.baseline_trace > 0 ?
        (tracker.baseline_trace - current_trace) / tracker.baseline_trace * 100 : 0.0

    # Update acceptance rate
    acceptance = tracker.total_observations > 0 ?
        tracker.total_map_updates / tracker.total_observations : 0.0

    # Time in state (simplified - would need state transition tracking for accuracy)
    time_in_state = timestamp - tracker.baseline_timestamp

    RealTimeCollapseMetrics(
        tracker.total_observations,
        tracker.total_map_updates,
        acceptance,
        tracker.baseline_trace,
        current_trace,
        trace_reduction,
        tracker.rmse_stats.mean,
        get_std(tracker.rmse_stats),
        tracker.nees_stats.mean,
        get_std(tracker.nees_stats),
        compute_reduction_rate(tracker),
        tracker.convergence_progress,
        tracker.state,
        time_in_state
    )
end

# ============================================================================
# Convergence Criterion (Online)
# ============================================================================

"""
    OnlineConvergenceCriteria

Criteria for online convergence detection.
"""
struct OnlineConvergenceCriteria
    min_trace_reduction_pct::Float64
    max_position_std::Float64
    nees_min::Float64
    nees_max::Float64
    min_observations::Int
    min_convergence_progress::Float64

    function OnlineConvergenceCriteria(;
        min_trace_reduction_pct::Float64 = 50.0,
        max_position_std::Float64 = 5.0,
        nees_min::Float64 = 0.5,
        nees_max::Float64 = 2.0,
        min_observations::Int = 100,
        min_convergence_progress::Float64 = 0.8
    )
        new(min_trace_reduction_pct, max_position_std, nees_min, nees_max,
            min_observations, min_convergence_progress)
    end
end

const DEFAULT_ONLINE_CONVERGENCE_CRITERIA = OnlineConvergenceCriteria()

"""
    OnlineConvergenceResult

Result of online convergence check.
"""
struct OnlineConvergenceResult
    converged::Bool
    progress::Float64
    position_std::Float64
    mean_nees::Float64
    trace_reduction_pct::Float64
    criteria_met::Dict{Symbol, Bool}
end

"""Check online convergence criteria."""
function check_online_convergence(tracker::OnlineCollapseTracker,
                                   criteria::OnlineConvergenceCriteria =
                                   DEFAULT_ONLINE_CONVERGENCE_CRITERIA)
    met = Dict{Symbol, Bool}()

    # Get current metrics
    snaps = tracker.recent_snapshots
    if isempty(snaps)
        return OnlineConvergenceResult(false, 0.0, Inf, 1.0, 0.0, met)
    end

    last_snap = snaps[end]
    current_trace = last_snap.trace_nav + last_snap.trace_map + last_snap.trace_sources

    # Check criteria
    trace_reduction = tracker.baseline_trace > 0 ?
        (tracker.baseline_trace - current_trace) / tracker.baseline_trace * 100 : 0.0

    met[:trace_reduction] = trace_reduction >= criteria.min_trace_reduction_pct
    met[:position_std] = last_snap.position_std <= criteria.max_position_std
    met[:nees_bounds] = criteria.nees_min <= tracker.nees_stats.mean <= criteria.nees_max
    met[:min_observations] = tracker.total_observations >= criteria.min_observations
    met[:convergence_progress] = tracker.convergence_progress >= criteria.min_convergence_progress

    converged = all(values(met))

    OnlineConvergenceResult(
        converged,
        tracker.convergence_progress,
        last_snap.position_std,
        tracker.nees_stats.mean,
        trace_reduction,
        met
    )
end

# ============================================================================
# Integration with Phase B
# ============================================================================

"""
    export_to_trajectory(tracker::OnlineCollapseTracker, mission_id::String)

Export online tracker data to Phase B CollapseTrajectory for offline analysis.
"""
function export_to_trajectory(tracker::OnlineCollapseTracker, mission_id::String)
    traj = CollapseTrajectory()
    traj.baseline_trace = tracker.baseline_trace
    traj.baseline_det = 0.0  # Not tracked in online mode

    for snap in tracker.recent_snapshots
        # Create Phase B snapshot from online snapshot
        total_trace = snap.trace_nav + snap.trace_map + snap.trace_sources

        phase_b_snap = CollapseSnapshot(
            mission_id,
            snap.observation_count,
            snap.timestamp,
            total_trace,
            0.0,  # determinant not tracked
            Float64[],  # eigenvalues not tracked
            snap.rmse_recent,
            snap.nees_recent,
            [snap.position_std, 0.0, 0.0]  # position error approximation
        )

        push!(traj.snapshots, phase_b_snap)
    end

    if !(mission_id in traj.mission_sequence)
        push!(traj.mission_sequence, mission_id)
    end

    return traj
end

# ============================================================================
# Formatting
# ============================================================================

"""Format collapse state as string."""
function format_collapse_state(state::CollapseState)
    if state == COLLAPSE_INACTIVE
        return "INACTIVE"
    elseif state == COLLAPSE_LEARNING
        return "LEARNING"
    elseif state == COLLAPSE_CONVERGING
        return "CONVERGING"
    elseif state == COLLAPSE_CONVERGED
        return "CONVERGED"
    elseif state == COLLAPSE_STALLED
        return "STALLED"
    elseif state == COLLAPSE_DIVERGING
        return "DIVERGING"
    else
        return "UNKNOWN"
    end
end

"""Format online collapse status."""
function format_online_collapse_status(status::OnlineCollapseStatus)
    state_str = format_collapse_state(status.state)
    health_str = status.is_healthy ? "✓" : "✗"

    return """
    Collapse Status: $state_str $health_str
      Trace: $(round(status.trace_total, sigdigits=3)) ($(round(status.trace_reduction_pct, digits=1))% reduction)
      Progress: $(round(status.convergence_progress * 100, digits=1))%
      Rate: $(round(status.reduction_rate, sigdigits=3))/obs
      NEES: $(round(status.mean_nees, digits=2))
      Alert: $(status.alert)
    """
end

"""Format real-time metrics."""
function format_realtime_metrics(metrics::RealTimeCollapseMetrics)
    return """
    Real-Time Collapse Metrics
    ==========================
    Observations: $(metrics.total_observations)
    Map updates: $(metrics.total_map_updates) ($(round(metrics.update_acceptance_rate * 100, digits=1))% acceptance)

    Trace: $(round(metrics.current_trace, sigdigits=3)) / $(round(metrics.baseline_trace, sigdigits=3))
           ($(round(metrics.trace_reduction_pct, digits=1))% reduction)

    RMSE: $(round(metrics.mean_rmse, digits=2)) ± $(round(metrics.std_rmse, digits=2)) m
    NEES: $(round(metrics.mean_nees, digits=2)) ± $(round(metrics.std_nees, digits=2))

    Reduction rate: $(round(metrics.reduction_rate, sigdigits=3))/obs
    Convergence: $(round(metrics.convergence_progress * 100, digits=1))%
    State: $(format_collapse_state(metrics.state))
    """
end

# ============================================================================
# Exports
# ============================================================================

export OnlineCollapseConfig, DEFAULT_ONLINE_COLLAPSE_CONFIG
export IncrementalStatistics, update_stats!, get_variance, get_std
export CollapseState, COLLAPSE_INACTIVE, COLLAPSE_LEARNING, COLLAPSE_CONVERGING
export COLLAPSE_CONVERGED, COLLAPSE_STALLED, COLLAPSE_DIVERGING
export OnlineCollapseSnapshot
export OnlineCollapseTracker, reset_tracker!
export OnlineCollapseStatus, update_tracker!
export compute_reduction_rate
export RealTimeCollapseMetrics, get_realtime_metrics
export OnlineConvergenceCriteria, DEFAULT_ONLINE_CONVERGENCE_CRITERIA
export OnlineConvergenceResult, check_online_convergence
export export_to_trajectory
export format_collapse_state, format_online_collapse_status, format_realtime_metrics
