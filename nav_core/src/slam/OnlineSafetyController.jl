# ============================================================================
# OnlineSafetyController.jl - Online SLAM Safety Controls (Phase C Step 8)
# ============================================================================
#
# Implements safety controls to prevent online learning from degrading
# navigation performance.
#
# Safety Invariants Enforced:
# - INV-04: NEES honesty (automatic rollback on violation)
# - INV-06: Timing budget compliance (throttling)
# - INV-08: Rollback capability (checkpoint management)
#
# Key Controls:
# 1. NEES Monitor - detects filter overconfidence
# 2. Covariance Bounds - prevents artificial collapse
# 3. Divergence Detector - catches learning failures
# 4. Update Throttle - maintains real-time performance
# 5. Automatic Rollback - reverts bad updates
#
# Architecture:
# - Controller monitors SLAM state and metrics
# - Produces SafetyAction recommendations
# - Integrates with scheduler to throttle/pause learning
# - Triggers checkpoint rollback when needed
#
# ============================================================================

using LinearAlgebra
using Statistics: mean

# ============================================================================
# Safety Configuration
# ============================================================================

"""
    SafetyControllerConfig

Configuration for online SLAM safety controls.

# Fields
## NEES Monitoring
- `nees_warn_threshold::Float64`: NEES above this triggers warning
- `nees_rollback_threshold::Float64`: NEES above this triggers rollback
- `nees_window_size::Int`: Window for NEES averaging
- `nees_consistency_min::Float64`: Minimum fraction of NEES in bounds

## Covariance Bounds
- `min_position_variance::Float64`: Minimum allowed position variance [m²]
- `max_position_variance::Float64`: Maximum allowed position variance [m²]
- `max_covariance_shrink_rate::Float64`: Max shrink per observation (INV-04)

## Divergence Detection
- `max_consecutive_outliers::Int`: Outliers before flagging
- `divergence_trace_ratio::Float64`: Trace growth ratio for divergence
- `stall_duration_s::Float64`: Duration before stall detection [s]

## Timing
- `max_slow_loop_time_ms::Float64`: Max time for slow loop [ms]
- `throttle_after_overruns::Int`: Overruns before throttling
- `emergency_pause_threshold_ms::Float64`: Pause learning if exceeded

## Checkpoints
- `checkpoint_interval_obs::Int`: Observations between checkpoints
- `max_checkpoints::Int`: Maximum stored checkpoints
- `validation_nees_threshold::Float64`: NEES must be below this for promotion

# Physics Justification
- nees_rollback_threshold = 3.0: Mean NEES > 3 indicates severe overconfidence
- min_position_variance = 0.01 m²: σ = 10 cm minimum (sensor limits)
- max_slow_loop_time_ms = 45: Leaves 5ms headroom in 50ms budget
"""
struct SafetyControllerConfig
    # NEES monitoring
    nees_warn_threshold::Float64
    nees_rollback_threshold::Float64
    nees_window_size::Int
    nees_consistency_min::Float64

    # Covariance bounds
    min_position_variance::Float64
    max_position_variance::Float64
    max_covariance_shrink_rate::Float64

    # Divergence detection
    max_consecutive_outliers::Int
    divergence_trace_ratio::Float64
    stall_duration_s::Float64

    # Timing
    max_slow_loop_time_ms::Float64
    throttle_after_overruns::Int
    emergency_pause_threshold_ms::Float64

    # Checkpoints
    checkpoint_interval_obs::Int
    max_checkpoints::Int
    validation_nees_threshold::Float64

    function SafetyControllerConfig(;
        nees_warn_threshold::Float64 = 2.0,
        nees_rollback_threshold::Float64 = 3.0,
        nees_window_size::Int = 50,
        nees_consistency_min::Float64 = 0.7,
        min_position_variance::Float64 = 0.01,
        max_position_variance::Float64 = 1000.0,
        max_covariance_shrink_rate::Float64 = 0.95,
        max_consecutive_outliers::Int = 5,
        divergence_trace_ratio::Float64 = 2.0,
        stall_duration_s::Float64 = 30.0,
        max_slow_loop_time_ms::Float64 = 45.0,
        throttle_after_overruns::Int = 3,
        emergency_pause_threshold_ms::Float64 = 80.0,
        checkpoint_interval_obs::Int = 100,
        max_checkpoints::Int = 10,
        validation_nees_threshold::Float64 = 1.5
    )
        @assert nees_warn_threshold > 0
        @assert nees_rollback_threshold > nees_warn_threshold
        @assert nees_window_size > 0
        @assert 0 < nees_consistency_min <= 1
        @assert min_position_variance > 0
        @assert max_position_variance > min_position_variance
        @assert 0 < max_covariance_shrink_rate <= 1
        @assert max_consecutive_outliers > 0
        @assert divergence_trace_ratio > 1
        @assert stall_duration_s > 0
        @assert max_slow_loop_time_ms > 0
        @assert throttle_after_overruns > 0
        @assert emergency_pause_threshold_ms > max_slow_loop_time_ms
        @assert checkpoint_interval_obs > 0
        @assert max_checkpoints > 0
        @assert validation_nees_threshold > 0

        new(nees_warn_threshold, nees_rollback_threshold, nees_window_size,
            nees_consistency_min, min_position_variance, max_position_variance,
            max_covariance_shrink_rate, max_consecutive_outliers, divergence_trace_ratio,
            stall_duration_s, max_slow_loop_time_ms, throttle_after_overruns,
            emergency_pause_threshold_ms, checkpoint_interval_obs, max_checkpoints,
            validation_nees_threshold)
    end
end

const DEFAULT_SAFETY_CONFIG = SafetyControllerConfig()

# ============================================================================
# Safety Actions
# ============================================================================

"""
    SafetyAction

Actions the safety controller can recommend.
"""
@enum SafetyAction begin
    ACTION_NONE = 0           # No action needed
    ACTION_WARN = 1           # Warning, continue with caution
    ACTION_THROTTLE = 2       # Reduce update frequency
    ACTION_PAUSE_LEARNING = 3 # Pause map learning, continue navigation
    ACTION_ROLLBACK = 4       # Rollback to last good checkpoint
    ACTION_EMERGENCY_STOP = 5 # Stop learning immediately
end

"""
    SafetyAlert

Alert levels for safety monitoring.
"""
@enum SafetyAlert begin
    ALERT_NONE = 0
    ALERT_NEES_HIGH = 1
    ALERT_COVARIANCE_COLLAPSE = 2
    ALERT_COVARIANCE_EXPLOSION = 3
    ALERT_DIVERGENCE = 4
    ALERT_STALL = 5
    ALERT_TIMING_OVERRUN = 6
    ALERT_OUTLIER_BURST = 7
end

# ============================================================================
# Safety State
# ============================================================================

"""
    SafetyMonitorState

Internal state of the safety monitor.
"""
mutable struct SafetyMonitorState
    # NEES history (sliding window)
    nees_window::Vector{Float64}
    nees_mean::Float64
    nees_consistency::Float64

    # Covariance tracking
    last_position_trace::Float64
    baseline_position_trace::Float64

    # Outlier tracking
    consecutive_outliers::Int
    total_outliers::Int

    # Timing tracking
    consecutive_overruns::Int
    total_overruns::Int
    max_loop_time_seen::Float64

    # Stall detection
    last_significant_update::Float64

    # State
    current_action::SafetyAction
    current_alerts::Vector{SafetyAlert}
end

function SafetyMonitorState()
    SafetyMonitorState(
        Float64[], 1.0, 1.0,
        0.0, 0.0,
        0, 0,
        0, 0, 0.0,
        0.0,
        ACTION_NONE, SafetyAlert[]
    )
end

# ============================================================================
# Checkpoint Management
# ============================================================================

"""
    SafetyCheckpoint

Checkpoint for potential rollback.
"""
struct SafetyCheckpoint
    timestamp::Float64
    observation_count::Int
    slam_checkpoint::SlamCheckpoint
    nees_at_checkpoint::Float64
    is_validated::Bool
    validation_timestamp::Float64
end

"""Create safety checkpoint from SLAM checkpoint."""
function SafetyCheckpoint(slam_cp::SlamCheckpoint, obs_count::Int;
                          validated::Bool = false, val_time::Float64 = 0.0)
    SafetyCheckpoint(
        slam_cp.timestamp,
        obs_count,
        slam_cp,
        slam_cp.nees_at_checkpoint,
        validated,
        val_time
    )
end

# ============================================================================
# Safety Result
# ============================================================================

"""
    SafetyCheckResult

Result of a safety check cycle.
"""
struct SafetyCheckResult
    action::SafetyAction
    alerts::Vector{SafetyAlert}
    nees_mean::Float64
    position_trace::Float64
    is_healthy::Bool
    should_checkpoint::Bool
    should_rollback::Bool
    rollback_target::Union{Nothing, SafetyCheckpoint}
    message::String
end

# ============================================================================
# Online Safety Controller
# ============================================================================

"""
    OnlineSafetyController

Safety controller for online SLAM.

# Architecture
- Monitors NEES, covariance, timing, and learning health
- Maintains checkpoint history for rollback
- Produces safety actions for scheduler integration
- Reports alerts for external monitoring

# Usage
```julia
controller = OnlineSafetyController()

# On each observation:
result = check_safety!(controller, slam_state, nees, loop_time)

if result.action == ACTION_ROLLBACK
    restore_from_checkpoint!(slam_state, result.rollback_target.slam_checkpoint)
end

if result.action == ACTION_PAUSE_LEARNING
    scheduler.learning_paused = true
end
```
"""
mutable struct OnlineSafetyController
    config::SafetyControllerConfig
    monitor::SafetyMonitorState
    checkpoints::Vector{SafetyCheckpoint}
    last_validated_checkpoint::Union{Nothing, SafetyCheckpoint}

    # Counters
    observations_since_checkpoint::Int
    total_observations::Int
    total_rollbacks::Int
    total_pauses::Int
end

"""Create safety controller."""
function OnlineSafetyController(config::SafetyControllerConfig = DEFAULT_SAFETY_CONFIG)
    OnlineSafetyController(
        config,
        SafetyMonitorState(),
        SafetyCheckpoint[],
        nothing,
        0, 0, 0, 0
    )
end

"""Reset safety controller."""
function reset_controller!(controller::OnlineSafetyController)
    controller.monitor = SafetyMonitorState()
    empty!(controller.checkpoints)
    controller.last_validated_checkpoint = nothing
    controller.observations_since_checkpoint = 0
    controller.total_observations = 0
    controller.total_rollbacks = 0
    controller.total_pauses = 0
    return controller
end

# ============================================================================
# Safety Check
# ============================================================================

"""
    check_safety!(controller::OnlineSafetyController, slam_state::SlamAugmentedState,
                  nees::Float64, loop_time_ms::Float64, timestamp::Float64;
                  is_outlier::Bool = false) -> SafetyCheckResult

Run safety checks and return recommended action.

# Arguments
- `slam_state`: Current SLAM state
- `nees`: Current NEES value
- `loop_time_ms`: Time for last slow loop [ms]
- `timestamp`: Current timestamp [s]
- `is_outlier`: Whether last observation was rejected as outlier
"""
function check_safety!(controller::OnlineSafetyController, slam_state::SlamAugmentedState,
                        nees::Float64, loop_time_ms::Float64, timestamp::Float64;
                        is_outlier::Bool = false)
    config = controller.config
    monitor = controller.monitor
    alerts = SafetyAlert[]

    controller.total_observations += 1
    controller.observations_since_checkpoint += 1

    # Update NEES window
    update_nees_window!(monitor, nees, config)

    # Update position trace
    nav_P = slam_state.nav_state.covariance
    pos_trace = nav_P[1,1] + nav_P[2,2] + nav_P[3,3]

    if monitor.baseline_position_trace == 0.0
        monitor.baseline_position_trace = pos_trace
    end
    monitor.last_position_trace = pos_trace

    # Update outlier tracking
    if is_outlier
        monitor.consecutive_outliers += 1
        monitor.total_outliers += 1
    else
        monitor.consecutive_outliers = 0
        monitor.last_significant_update = timestamp
    end

    # Update timing tracking
    if loop_time_ms > config.max_slow_loop_time_ms
        monitor.consecutive_overruns += 1
        monitor.total_overruns += 1
    else
        monitor.consecutive_overruns = 0
    end
    monitor.max_loop_time_seen = max(monitor.max_loop_time_seen, loop_time_ms)

    # === Run Safety Checks ===

    # 1. NEES check
    if monitor.nees_mean > config.nees_rollback_threshold
        push!(alerts, ALERT_NEES_HIGH)
    elseif monitor.nees_mean > config.nees_warn_threshold
        push!(alerts, ALERT_NEES_HIGH)
    end

    # 2. Covariance collapse check
    if pos_trace < config.min_position_variance
        push!(alerts, ALERT_COVARIANCE_COLLAPSE)
    end

    # 3. Covariance explosion check
    if pos_trace > config.max_position_variance
        push!(alerts, ALERT_COVARIANCE_EXPLOSION)
    end

    # 4. Divergence check (trace growing relative to baseline)
    if pos_trace > monitor.baseline_position_trace * config.divergence_trace_ratio
        push!(alerts, ALERT_DIVERGENCE)
    end

    # 5. Stall check
    time_since_update = timestamp - monitor.last_significant_update
    if time_since_update > config.stall_duration_s && monitor.last_significant_update > 0
        push!(alerts, ALERT_STALL)
    end

    # 6. Timing check
    if monitor.consecutive_overruns >= config.throttle_after_overruns
        push!(alerts, ALERT_TIMING_OVERRUN)
    end

    # 7. Outlier burst check
    if monitor.consecutive_outliers >= config.max_consecutive_outliers
        push!(alerts, ALERT_OUTLIER_BURST)
    end

    # === Determine Action ===
    action, rollback_needed, rollback_target, message = determine_safety_action(
        controller, alerts, loop_time_ms, timestamp
    )

    monitor.current_action = action
    monitor.current_alerts = alerts

    # === Checkpoint Management ===
    should_checkpoint = (controller.observations_since_checkpoint >=
                        config.checkpoint_interval_obs) &&
                       isempty(alerts) &&
                       monitor.nees_mean < config.validation_nees_threshold

    if should_checkpoint
        create_safety_checkpoint!(controller, slam_state, timestamp)
    end

    # Validate pending checkpoints if NEES stable
    if monitor.nees_consistency >= config.nees_consistency_min
        validate_checkpoints!(controller, timestamp)
    end

    SafetyCheckResult(
        action,
        alerts,
        monitor.nees_mean,
        pos_trace,
        isempty(alerts),
        should_checkpoint,
        rollback_needed,
        rollback_target,
        message
    )
end

"""Update NEES sliding window."""
function update_nees_window!(monitor::SafetyMonitorState, nees::Float64,
                              config::SafetyControllerConfig)
    push!(monitor.nees_window, nees)

    # Maintain window size
    while length(monitor.nees_window) > config.nees_window_size
        popfirst!(monitor.nees_window)
    end

    # Update statistics
    if !isempty(monitor.nees_window)
        monitor.nees_mean = mean(monitor.nees_window)
        # Consistency: fraction of NEES in [0.1, 5.0]
        in_bounds = count(n -> 0.1 < n < 5.0, monitor.nees_window)
        monitor.nees_consistency = in_bounds / length(monitor.nees_window)
    end
end

"""Determine safety action from alerts."""
function determine_safety_action(controller::OnlineSafetyController,
                                  alerts::Vector{SafetyAlert},
                                  loop_time_ms::Float64,
                                  timestamp::Float64)
    config = controller.config
    monitor = controller.monitor

    # Default: no action
    action = ACTION_NONE
    rollback_needed = false
    rollback_target = nothing
    message = "OK"

    # Emergency timing violation
    if loop_time_ms > config.emergency_pause_threshold_ms
        action = ACTION_EMERGENCY_STOP
        message = "Emergency: timing budget exceeded ($(round(loop_time_ms, digits=1))ms)"
        controller.total_pauses += 1
        return (action, rollback_needed, rollback_target, message)
    end

    # Check for rollback triggers
    if ALERT_NEES_HIGH in alerts && monitor.nees_mean > config.nees_rollback_threshold
        rollback_needed = true
        rollback_target = controller.last_validated_checkpoint
        if rollback_target !== nothing
            action = ACTION_ROLLBACK
            message = "Rollback: NEES too high ($(round(monitor.nees_mean, digits=2)))"
            controller.total_rollbacks += 1
        else
            action = ACTION_PAUSE_LEARNING
            message = "Pause: NEES high, no validated checkpoint for rollback"
            controller.total_pauses += 1
        end
        return (action, rollback_needed, rollback_target, message)
    end

    # Check for divergence
    if ALERT_DIVERGENCE in alerts
        rollback_needed = true
        rollback_target = controller.last_validated_checkpoint
        if rollback_target !== nothing
            action = ACTION_ROLLBACK
            message = "Rollback: covariance divergence detected"
            controller.total_rollbacks += 1
        else
            action = ACTION_PAUSE_LEARNING
            message = "Pause: divergence detected, no checkpoint"
            controller.total_pauses += 1
        end
        return (action, rollback_needed, rollback_target, message)
    end

    # Check for covariance collapse (artificial)
    if ALERT_COVARIANCE_COLLAPSE in alerts
        action = ACTION_PAUSE_LEARNING
        message = "Pause: covariance collapsed below minimum"
        controller.total_pauses += 1
        return (action, rollback_needed, rollback_target, message)
    end

    # Check for timing overruns
    if ALERT_TIMING_OVERRUN in alerts
        action = ACTION_THROTTLE
        message = "Throttle: timing overruns ($(monitor.consecutive_overruns) consecutive)"
        return (action, rollback_needed, rollback_target, message)
    end

    # Check for outlier burst
    if ALERT_OUTLIER_BURST in alerts
        action = ACTION_WARN
        message = "Warning: consecutive outliers ($(monitor.consecutive_outliers))"
        return (action, rollback_needed, rollback_target, message)
    end

    # Check for stall
    if ALERT_STALL in alerts
        action = ACTION_WARN
        message = "Warning: learning stalled"
        return (action, rollback_needed, rollback_target, message)
    end

    # Mild NEES warning
    if ALERT_NEES_HIGH in alerts
        action = ACTION_WARN
        message = "Warning: NEES elevated ($(round(monitor.nees_mean, digits=2)))"
        return (action, rollback_needed, rollback_target, message)
    end

    return (action, rollback_needed, rollback_target, message)
end

# ============================================================================
# Checkpoint Management
# ============================================================================

"""Create a safety checkpoint."""
function create_safety_checkpoint!(controller::OnlineSafetyController,
                                    slam_state::SlamAugmentedState,
                                    timestamp::Float64)
    # Create SLAM checkpoint
    slam_cp = create_checkpoint(slam_state, controller.monitor.nees_mean)

    # Wrap in safety checkpoint
    safety_cp = SafetyCheckpoint(slam_cp, controller.total_observations)

    push!(controller.checkpoints, safety_cp)
    controller.observations_since_checkpoint = 0

    # Prune old checkpoints
    while length(controller.checkpoints) > controller.config.max_checkpoints
        popfirst!(controller.checkpoints)
    end
end

"""Validate pending checkpoints if NEES has been stable."""
function validate_checkpoints!(controller::OnlineSafetyController, timestamp::Float64)
    config = controller.config

    for (i, cp) in enumerate(controller.checkpoints)
        if !cp.is_validated
            # Validate if enough time has passed with good NEES
            time_since_cp = timestamp - cp.timestamp
            if time_since_cp > 10.0 && controller.monitor.nees_mean < config.validation_nees_threshold
                # Promote to validated
                validated_cp = SafetyCheckpoint(
                    cp.timestamp,
                    cp.observation_count,
                    cp.slam_checkpoint,
                    cp.nees_at_checkpoint,
                    true,
                    timestamp
                )
                controller.checkpoints[i] = validated_cp
                controller.last_validated_checkpoint = validated_cp
            end
        end
    end
end

"""Get last validated checkpoint (for rollback)."""
function get_rollback_checkpoint(controller::OnlineSafetyController)
    return controller.last_validated_checkpoint
end

"""Perform rollback to checkpoint."""
function execute_rollback!(controller::OnlineSafetyController,
                            slam_state::SlamAugmentedState,
                            checkpoint::SafetyCheckpoint)
    # Restore SLAM state
    restore_from_checkpoint!(slam_state, checkpoint.slam_checkpoint)

    # Reset safety monitor state
    controller.monitor.baseline_position_trace = 0.0  # Will be reset on next check
    controller.monitor.consecutive_outliers = 0
    controller.monitor.consecutive_overruns = 0
    empty!(controller.monitor.nees_window)

    # Don't clear checkpoints - keep history before rollback point
    # But remove checkpoints after the rollback point
    filter!(cp -> cp.timestamp <= checkpoint.timestamp, controller.checkpoints)

    return slam_state
end

# ============================================================================
# Statistics
# ============================================================================

"""
    SafetyStatistics

Statistics for safety monitoring.
"""
struct SafetyStatistics
    total_observations::Int
    total_rollbacks::Int
    total_pauses::Int
    total_outliers::Int
    total_timing_overruns::Int
    checkpoints_available::Int
    has_validated_checkpoint::Bool
    current_nees_mean::Float64
    max_loop_time_ms::Float64
    current_action::SafetyAction
end

"""Get current safety statistics."""
function get_safety_statistics(controller::OnlineSafetyController)
    SafetyStatistics(
        controller.total_observations,
        controller.total_rollbacks,
        controller.total_pauses,
        controller.monitor.total_outliers,
        controller.monitor.total_overruns,
        length(controller.checkpoints),
        controller.last_validated_checkpoint !== nothing,
        controller.monitor.nees_mean,
        controller.monitor.max_loop_time_seen,
        controller.monitor.current_action
    )
end

"""Format safety statistics."""
function format_safety_statistics(stats::SafetyStatistics)
    action_str = if stats.current_action == ACTION_NONE
        "NORMAL"
    elseif stats.current_action == ACTION_WARN
        "WARNING"
    elseif stats.current_action == ACTION_THROTTLE
        "THROTTLED"
    elseif stats.current_action == ACTION_PAUSE_LEARNING
        "PAUSED"
    elseif stats.current_action == ACTION_ROLLBACK
        "ROLLBACK"
    else
        "EMERGENCY"
    end

    validated_str = stats.has_validated_checkpoint ? "✓" : "✗"

    return """
    Safety Controller Statistics
    ============================
    Observations: $(stats.total_observations)
    Rollbacks: $(stats.total_rollbacks)
    Pauses: $(stats.total_pauses)
    Outliers: $(stats.total_outliers)
    Timing overruns: $(stats.total_timing_overruns)

    Checkpoints: $(stats.checkpoints_available) available
    Validated checkpoint: $validated_str

    Current NEES: $(round(stats.current_nees_mean, digits=2))
    Max loop time: $(round(stats.max_loop_time_ms, digits=1))ms
    Status: $action_str
    """
end

"""Format safety check result."""
function format_safety_result(result::SafetyCheckResult)
    action_str = if result.action == ACTION_NONE
        "✓ OK"
    elseif result.action == ACTION_WARN
        "⚠ WARNING"
    elseif result.action == ACTION_THROTTLE
        "⚡ THROTTLE"
    elseif result.action == ACTION_PAUSE_LEARNING
        "⏸ PAUSED"
    elseif result.action == ACTION_ROLLBACK
        "↩ ROLLBACK"
    else
        "🛑 EMERGENCY"
    end

    alerts_str = isempty(result.alerts) ? "none" : join(string.(result.alerts), ", ")

    return """
    Safety Check: $action_str
      Alerts: $alerts_str
      NEES: $(round(result.nees_mean, digits=2))
      Position trace: $(round(result.position_trace, sigdigits=3))
      Message: $(result.message)
    """
end

# ============================================================================
# Exports
# ============================================================================

export SafetyControllerConfig, DEFAULT_SAFETY_CONFIG
export SafetyAction, ACTION_NONE, ACTION_WARN, ACTION_THROTTLE
export ACTION_PAUSE_LEARNING, ACTION_ROLLBACK, ACTION_EMERGENCY_STOP
export SafetyAlert, ALERT_NONE, ALERT_NEES_HIGH, ALERT_COVARIANCE_COLLAPSE
export ALERT_COVARIANCE_EXPLOSION, ALERT_DIVERGENCE, ALERT_STALL
export ALERT_TIMING_OVERRUN, ALERT_OUTLIER_BURST
export SafetyMonitorState, SafetyCheckpoint
export SafetyCheckResult, OnlineSafetyController
export reset_controller!, check_safety!
export get_rollback_checkpoint, execute_rollback!
export SafetyStatistics, get_safety_statistics
export format_safety_statistics, format_safety_result
