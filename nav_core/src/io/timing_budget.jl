# ============================================================================
# timing_budget.jl - Real-Time Scheduling and Timing Budget
# ============================================================================
#
# Ported from AUV-Navigation/src/timing_budget.jl
#
# Embedded systems have hard timing constraints. When processing takes
# too long, we must skip expensive work (full graph optimize, feature
# updates) to maintain real-time guarantees.
#
# Budget hierarchy:
# - Keyframe budget: Total time for one keyframe cycle (~50-100 ms)
# - Factor budget: Time per factor type (IMU, Odometry, magnetic)
# - Optimization budget: Time for graph optimization
#
# When over-budget:
# 1. Skip non-critical work (feature lifecycle, absorption)
# 2. Use approximate solutions (fewer LM iterations)
# 3. Defer full optimization to next cycle
# 4. Log warning for operator awareness
# ============================================================================

export TimingBudget, DEFAULT_TIMING_BUDGET, HIGH_PERF_TIMING_BUDGET, EMBEDDED_TIMING_BUDGET
export TaskTiming, TimingTracker
export start_keyframe!, end_keyframe!, time_task!
export remaining_budget, can_afford
export ThrottleDecision, NO_THROTTLE, SKIP_FEATURE_WORK, SKIP_MAP_OPTIMIZE
export SKIP_FULL_OPTIMIZE, EMERGENCY_THROTTLE
export throttle_decision, should_skip_feature_work, should_skip_optimization
export record_skipped_work!
export AdaptiveBudgetConfig, adapt_budget!
export TimingStats, get_timing_stats, print_timing_report
export with_timing_budget
export LoadSpikeDetector, update_load!

# ============================================================================
# Timing Configuration
# ============================================================================

"""
    TimingBudget

Time budgets for real-time operation.

All times in seconds.

# Fields
- `keyframe_total::Float64` - Total time for keyframe processing
- `imu_integration::Float64` - IMU preintegration budget
- `odometry_factor::Float64` - Odometry factor creation
- `barometer_factor::Float64` - Depth factor creation
- `magnetic_factor::Float64` - Magnetic factor + residual computation
- `graph_optimize::Float64` - Full graph optimization
- `map_optimize::Float64` - Map tile optimization
- `joint_optimize::Float64` - Joint pose+map optimization
- `feature_lifecycle::Float64` - Lifecycle check and retirement
- `feature_absorption::Float64` - Absorption processing
- `feature_disambiguation::Float64` - Association and disambiguation
- `safety_margin::Float64` - Reserved for unexpected work
"""
struct TimingBudget
    # Per-keyframe budgets
    keyframe_total::Float64          # Total time for keyframe processing
    imu_integration::Float64         # IMU preintegration budget
    odometry_factor::Float64              # Odometry factor creation
    barometer_factor::Float64            # Depth factor creation
    magnetic_factor::Float64         # Magnetic factor + residual computation

    # Optimization budgets
    graph_optimize::Float64          # Full graph optimization
    map_optimize::Float64            # Map tile optimization
    joint_optimize::Float64          # Joint pose+map optimization

    # Feature budgets
    feature_lifecycle::Float64       # Lifecycle check and retirement
    feature_absorption::Float64      # Absorption processing
    feature_disambiguation::Float64  # Association and disambiguation

    # Safety margins
    safety_margin::Float64           # Reserved for unexpected work
end

function TimingBudget(;
    keyframe_total::Real = 0.100,      # 100 ms per keyframe (10 Hz)
    imu_integration::Real = 0.005,     # 5 ms for IMU
    odometry_factor::Real = 0.002,          # 2 ms for Odometry
    barometer_factor::Real = 0.001,        # 1 ms for depth
    magnetic_factor::Real = 0.010,     # 10 ms for magnetic
    graph_optimize::Real = 0.030,      # 30 ms for optimization
    map_optimize::Real = 0.015,        # 15 ms for map
    joint_optimize::Real = 0.040,      # 40 ms for joint
    feature_lifecycle::Real = 0.005,   # 5 ms for lifecycle
    feature_absorption::Real = 0.005,  # 5 ms for absorption
    feature_disambiguation::Real = 0.005, # 5 ms for disambiguation
    safety_margin::Real = 0.010        # 10 ms safety
)
    TimingBudget(
        Float64(keyframe_total),
        Float64(imu_integration),
        Float64(odometry_factor),
        Float64(barometer_factor),
        Float64(magnetic_factor),
        Float64(graph_optimize),
        Float64(map_optimize),
        Float64(joint_optimize),
        Float64(feature_lifecycle),
        Float64(feature_absorption),
        Float64(feature_disambiguation),
        Float64(safety_margin)
    )
end

const DEFAULT_TIMING_BUDGET = TimingBudget()

# High-performance budget for powerful hardware
const HIGH_PERF_TIMING_BUDGET = TimingBudget(
    keyframe_total = 0.050,      # 50 ms (20 Hz)
    graph_optimize = 0.015,
    joint_optimize = 0.020
)

# Embedded budget for constrained hardware
const EMBEDDED_TIMING_BUDGET = TimingBudget(
    keyframe_total = 0.200,      # 200 ms (5 Hz)
    graph_optimize = 0.080,
    joint_optimize = 0.100,
    feature_lifecycle = 0.010,
    feature_absorption = 0.010,
    safety_margin = 0.020
)

# ============================================================================
# Timing Tracker
# ============================================================================

"""
    TaskTiming

Timing record for a single task execution.
"""
struct TaskTiming
    task_name::String
    budget::Float64
    actual::Float64
    over_budget::Bool
    timestamp::Float64
end

"""
    TimingTracker

Tracks execution timing and budget compliance.

# Fields
- `budget::TimingBudget` - Budget configuration
- `keyframe_start::Float64` - Start time of current keyframe
- `keyframe_elapsed::Float64` - Elapsed time in current keyframe
- `task_timings::Vector{TaskTiming}` - Task timings for current keyframe
- `total_keyframes::Int` - Total keyframes processed
- `over_budget_count::Int` - Number of over-budget keyframes
- `skipped_work_count::Int` - Number of times work was skipped
- `avg_keyframe_time::Float64` - Running average keyframe time
- `avg_optimize_time::Float64` - Running average optimization time
- `alpha::Float64` - EMA smoothing factor
- `estimated_load::Float64` - Current load factor (1.0 = normal)
- `peak_load::Float64` - Peak observed load
- `warning_issued::Bool` - Whether warning was issued this keyframe
- `last_warning_time::Float64` - Time of last warning
- `warning_cooldown::Float64` - Seconds between warnings
"""
mutable struct TimingTracker
    budget::TimingBudget

    # Current keyframe timing
    keyframe_start::Float64
    keyframe_elapsed::Float64

    # Task timings for current keyframe
    task_timings::Vector{TaskTiming}

    # Historical statistics
    total_keyframes::Int
    over_budget_count::Int
    skipped_work_count::Int

    # Running averages (exponential moving average)
    avg_keyframe_time::Float64
    avg_optimize_time::Float64
    alpha::Float64  # EMA smoothing factor

    # Load estimation
    estimated_load::Float64  # Current load factor (1.0 = normal)
    peak_load::Float64       # Peak observed load

    # Warning state
    warning_issued::Bool
    last_warning_time::Float64
    warning_cooldown::Float64  # Seconds between warnings
end

function TimingTracker(; budget::TimingBudget = DEFAULT_TIMING_BUDGET)
    TimingTracker(
        budget,
        0.0, 0.0,           # keyframe timing
        TaskTiming[],        # task timings
        0, 0, 0,             # counters
        0.0, 0.0, 0.1,       # averages
        1.0, 1.0,            # load
        false, 0.0, 5.0      # warnings
    )
end

# ============================================================================
# Timing Operations
# ============================================================================

"""
    start_keyframe!(tracker)

Start timing a new keyframe. Returns start time.
"""
function start_keyframe!(tracker::TimingTracker)
    tracker.keyframe_start = time()
    tracker.keyframe_elapsed = 0.0
    empty!(tracker.task_timings)
    tracker.warning_issued = false
    return tracker.keyframe_start
end

"""
    end_keyframe!(tracker)

End keyframe timing and update statistics.

Returns named tuple with elapsed time, budget, over_budget flag, and load.
"""
function end_keyframe!(tracker::TimingTracker)
    tracker.keyframe_elapsed = time() - tracker.keyframe_start
    tracker.total_keyframes += 1

    # Update running average
    tracker.avg_keyframe_time = tracker.alpha * tracker.keyframe_elapsed +
                                 (1 - tracker.alpha) * tracker.avg_keyframe_time

    # Check if over budget
    over_budget = tracker.keyframe_elapsed > tracker.budget.keyframe_total
    if over_budget
        tracker.over_budget_count += 1
    end

    # Update load estimate
    tracker.estimated_load = tracker.keyframe_elapsed / tracker.budget.keyframe_total
    tracker.peak_load = max(tracker.peak_load, tracker.estimated_load)

    return (
        elapsed = tracker.keyframe_elapsed,
        budget = tracker.budget.keyframe_total,
        over_budget = over_budget,
        load = tracker.estimated_load
    )
end

"""
    time_task!(f, tracker, task_name, budget)

Execute function `f` while tracking timing against budget.

Returns (result, timing) where timing includes over_budget flag.
"""
function time_task!(f::Function, tracker::TimingTracker, task_name::String, budget::Float64)
    start_time = time()
    result = f()
    elapsed = time() - start_time

    over_budget = elapsed > budget
    timing = TaskTiming(task_name, budget, elapsed, over_budget, start_time)
    push!(tracker.task_timings, timing)

    tracker.keyframe_elapsed += elapsed

    return (result = result, timing = timing)
end

"""
    remaining_budget(tracker)

Get remaining budget for current keyframe.
"""
function remaining_budget(tracker::TimingTracker)
    used = time() - tracker.keyframe_start
    remaining = tracker.budget.keyframe_total - used - tracker.budget.safety_margin
    return max(0.0, remaining)
end

"""
    can_afford(tracker, task_budget)

Check if we can afford to run a task within remaining budget.
"""
function can_afford(tracker::TimingTracker, task_budget::Float64)
    return remaining_budget(tracker) >= task_budget
end

# ============================================================================
# Load-Based Throttling
# ============================================================================

"""
    ThrottleDecision

Decision about what work to skip when over-budget.
"""
@enum ThrottleDecision begin
    NO_THROTTLE           # Run everything
    SKIP_FEATURE_WORK     # Skip lifecycle, absorption, disambiguation
    SKIP_MAP_OPTIMIZE     # Skip map optimization
    SKIP_FULL_OPTIMIZE    # Skip full graph optimization (use prediction)
    EMERGENCY_THROTTLE    # Skip all non-essential work
end

"""
    throttle_decision(tracker; load_threshold=2.0)

Decide what work to skip based on current load.

- load < 1.0: No throttling
- load 1.0-2.0: Skip feature work
- load 2.0-5.0: Skip map optimization
- load 5.0-10.0: Skip full optimization
- load > 10.0: Emergency throttle
"""
function throttle_decision(tracker::TimingTracker; load_threshold::Float64 = 2.0)
    load = tracker.estimated_load

    if load < 1.0
        return NO_THROTTLE
    elseif load < load_threshold
        return SKIP_FEATURE_WORK
    elseif load < 5.0
        return SKIP_MAP_OPTIMIZE
    elseif load < 10.0
        return SKIP_FULL_OPTIMIZE
    else
        return EMERGENCY_THROTTLE
    end
end

"""
    should_skip_feature_work(tracker)

Check if feature work should be skipped.
"""
function should_skip_feature_work(tracker::TimingTracker)
    decision = throttle_decision(tracker)
    return decision != NO_THROTTLE
end

"""
    should_skip_optimization(tracker)

Check if optimization should be skipped.
"""
function should_skip_optimization(tracker::TimingTracker)
    decision = throttle_decision(tracker)
    return decision in (SKIP_FULL_OPTIMIZE, EMERGENCY_THROTTLE)
end

"""
    record_skipped_work!(tracker, reason)

Record that work was skipped due to timing constraints.
"""
function record_skipped_work!(tracker::TimingTracker, reason::String)
    tracker.skipped_work_count += 1

    # Issue warning if needed (with cooldown)
    current_time = time()
    if !tracker.warning_issued &&
       (current_time - tracker.last_warning_time) > tracker.warning_cooldown

        @warn "Timing budget exceeded" load=round(tracker.estimated_load, digits=2) reason=reason
        tracker.warning_issued = true
        tracker.last_warning_time = current_time
    end
end

# ============================================================================
# Adaptive Budget Tuning
# ============================================================================

"""
    AdaptiveBudgetConfig

Configuration for adaptive budget adjustment.

# Fields
- `enable_adaptation::Bool` - Enable adaptive tuning
- `min_scale::Float64` - Minimum budget scale (e.g., 0.5)
- `max_scale::Float64` - Maximum budget scale (e.g., 2.0)
- `adaptation_rate::Float64` - How fast to adapt (0-1)
- `target_utilization::Float64` - Target budget utilization (e.g., 0.8)
"""
struct AdaptiveBudgetConfig
    enable_adaptation::Bool
    min_scale::Float64      # Minimum budget scale (e.g., 0.5)
    max_scale::Float64      # Maximum budget scale (e.g., 2.0)
    adaptation_rate::Float64 # How fast to adapt (0-1)
    target_utilization::Float64 # Target budget utilization (e.g., 0.8)
end

function AdaptiveBudgetConfig(;
    enable_adaptation::Bool = true,
    min_scale::Real = 0.5,
    max_scale::Real = 2.0,
    adaptation_rate::Real = 0.1,
    target_utilization::Real = 0.8
)
    AdaptiveBudgetConfig(
        enable_adaptation,
        Float64(min_scale),
        Float64(max_scale),
        Float64(adaptation_rate),
        Float64(target_utilization)
    )
end

"""
    adapt_budget!(tracker; config)

Adapt budget based on actual performance.

If consistently under-utilizing, reduce budgets.
If consistently over-budget, increase budgets (within limits).

Returns `nothing` if no adaptation needed, or (scale, utilization) tuple.
"""
function adapt_budget!(tracker::TimingTracker;
                        config::AdaptiveBudgetConfig = AdaptiveBudgetConfig())
    if !config.enable_adaptation || tracker.total_keyframes < 10
        return nothing
    end

    utilization = tracker.avg_keyframe_time / tracker.budget.keyframe_total

    # Compute scaling factor
    if utilization < config.target_utilization * 0.8
        # Under-utilizing: could tighten budget
        scale = 1.0 - config.adaptation_rate
    elseif utilization > config.target_utilization * 1.2
        # Over-utilizing: need to relax budget
        scale = 1.0 + config.adaptation_rate
    else
        # In target range
        return nothing
    end

    scale = clamp(scale, config.min_scale, config.max_scale)

    return (scale = scale, utilization = utilization)
end

# ============================================================================
# Timing Statistics
# ============================================================================

"""
    TimingStats

Summary statistics for timing performance.

# Fields
- `total_keyframes::Int` - Total keyframes processed
- `over_budget_count::Int` - Number of over-budget keyframes
- `over_budget_rate::Float64` - Fraction of keyframes over budget
- `skipped_work_count::Int` - Number of times work was skipped
- `avg_keyframe_time::Float64` - Average keyframe processing time
- `avg_optimize_time::Float64` - Average optimization time
- `peak_load::Float64` - Peak load observed
- `current_load::Float64` - Current load factor
"""
struct TimingStats
    total_keyframes::Int
    over_budget_count::Int
    over_budget_rate::Float64
    skipped_work_count::Int
    avg_keyframe_time::Float64
    avg_optimize_time::Float64
    peak_load::Float64
    current_load::Float64
end

"""
    get_timing_stats(tracker)

Get summary timing statistics.
"""
function get_timing_stats(tracker::TimingTracker)
    over_budget_rate = tracker.total_keyframes > 0 ?
        tracker.over_budget_count / tracker.total_keyframes : 0.0

    TimingStats(
        tracker.total_keyframes,
        tracker.over_budget_count,
        over_budget_rate,
        tracker.skipped_work_count,
        tracker.avg_keyframe_time,
        tracker.avg_optimize_time,
        tracker.peak_load,
        tracker.estimated_load
    )
end

"""
    print_timing_report(tracker)

Print detailed timing report.
"""
function print_timing_report(tracker::TimingTracker)
    stats = get_timing_stats(tracker)

    println("="^50)
    println("Timing Budget Report")
    println("="^50)
    println("Total keyframes: ", stats.total_keyframes)
    println("Over-budget count: ", stats.over_budget_count,
            " (", round(stats.over_budget_rate * 100, digits=1), "%)")
    println("Skipped work count: ", stats.skipped_work_count)
    println()
    println("Timing:")
    println("  Budget per keyframe: ", round(tracker.budget.keyframe_total * 1000, digits=1), " ms")
    println("  Avg keyframe time: ", round(stats.avg_keyframe_time * 1000, digits=1), " ms")
    println("  Avg utilization: ", round(stats.avg_keyframe_time / tracker.budget.keyframe_total * 100, digits=1), "%")
    println()
    println("Load:")
    println("  Current: ", round(stats.current_load, digits=2), "x")
    println("  Peak: ", round(stats.peak_load, digits=2), "x")
    println()

    if !isempty(tracker.task_timings)
        println("Last keyframe tasks:")
        for t in tracker.task_timings
            status = t.over_budget ? "OVER" : "OK"
            println("  ", t.task_name, ": ",
                   round(t.actual * 1000, digits=2), " ms / ",
                   round(t.budget * 1000, digits=2), " ms [", status, "]")
        end
    end
    println("="^50)
end

# ============================================================================
# Throttled Work Wrappers
# ============================================================================

"""
    with_timing_budget(f, tracker, task_name, budget; skip_if_over=false)

Execute function with timing tracking.

If skip_if_over=true and we're over budget, skip the work entirely.

Returns named tuple (result, skipped, timing).
"""
function with_timing_budget(f::Function, tracker::TimingTracker,
                             task_name::String, budget::Float64;
                             skip_if_over::Bool = false)
    if skip_if_over && !can_afford(tracker, budget)
        record_skipped_work!(tracker, task_name)
        return (result = nothing, skipped = true, timing = nothing)
    end

    result, timing = time_task!(f, tracker, task_name, budget)
    return (result = result, skipped = false, timing = timing)
end

# ============================================================================
# Load Spike Detection
# ============================================================================

"""
    LoadSpikeDetector

Detects sudden load increases that might indicate problems.

# Fields
- `history::Vector{Float64}` - Recent load history
- `window_size::Int` - Size of history window
- `spike_threshold::Float64` - Multiple of baseline to trigger spike
- `baseline::Float64` - Current baseline load
"""
mutable struct LoadSpikeDetector
    history::Vector{Float64}
    window_size::Int
    spike_threshold::Float64  # Multiple of baseline
    baseline::Float64
end

function LoadSpikeDetector(; window_size::Int = 20, spike_threshold::Real = 3.0)
    LoadSpikeDetector(Float64[], window_size, Float64(spike_threshold), 1.0)
end

"""
    update_load!(detector, load)

Update load history and check for spikes.

Returns (is_spike, load, baseline) named tuple.
"""
function update_load!(detector::LoadSpikeDetector, load::Float64)
    push!(detector.history, load)

    # Maintain window size
    while length(detector.history) > detector.window_size
        popfirst!(detector.history)
    end

    # Compute baseline (median of recent history)
    if length(detector.history) >= 3
        detector.baseline = _median(detector.history[1:end-1])
    end

    # Check for spike
    is_spike = load > detector.baseline * detector.spike_threshold

    return (is_spike = is_spike, load = load, baseline = detector.baseline)
end

"""
    _median(v)

Simple median for load calculation. Internal function.
"""
function _median(v::Vector{Float64})
    sorted = sort(v)
    n = length(sorted)
    if n == 0
        return 0.0
    elseif n % 2 == 1
        return sorted[(n + 1) ÷ 2]
    else
        return (sorted[n ÷ 2] + sorted[n ÷ 2 + 1]) / 2
    end
end
