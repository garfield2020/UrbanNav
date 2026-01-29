# ============================================================================
# ttd_metrics.jl - Time-To-Detect (TTD) Metrics
# ============================================================================
#
# V1.0 Qualification Closure Plan Step 5:
# Convert silent-divergence gate into quantitative TTD metric.
#
# TTD = Time from fault injection to health system detection
#
# This module provides:
# 1. TTDSample - Individual detection event measurement
# 2. TTDStatistics - Aggregate statistics across samples
# 3. TTDTracker - Real-time TTD tracking during simulation
# 4. TTDGate - Quantitative gate based on TTD thresholds
# 5. TTD analysis and reporting
#
# Key insight: Silent divergence is TTD > max_allowed_ttd
# This converts binary pass/fail into continuous metric.
# ============================================================================

using Statistics
using LinearAlgebra

export TTDSample, TTDStatistics, TTDTracker
export TTDConfig, DEFAULT_TTD_CONFIG
export TTDGateResult, TTDGate, evaluate_ttd_gate
export start_fault!, record_detection!, get_ttd_samples
export compute_ttd_statistics, reset_tracker!, update_tracker!
export TTDScenarioResult, TTDSuiteResult
export run_ttd_scenario, analyze_ttd_results
export format_ttd_report, format_ttd_summary
export is_silent_divergence, finalize_fault!

# ============================================================================
# TTD Sample
# ============================================================================

"""
    TTDSample

A single Time-To-Detect measurement.

# Fields
- `fault_type`: Type of fault that was injected
- `fault_start_time`: When the fault was injected
- `detection_time`: When the health system detected it (Inf if never)
- `ttd`: Time-To-Detect = detection_time - fault_start_time
- `detected`: Whether the fault was ever detected
- `detection_health_state`: Health state at detection (CAUTION/WARNING/CRITICAL)
- `peak_error`: Maximum position error during fault
- `scenario_name`: Name of the scenario
"""
struct TTDSample
    fault_type::FaultType
    fault_start_time::Float64
    detection_time::Float64
    ttd::Float64
    detected::Bool
    detection_health_state::Int
    peak_error::Float64
    scenario_name::String
end

function TTDSample(;
    fault_type::FaultType = FAULT_NONE,
    fault_start_time::Float64 = 0.0,
    detection_time::Float64 = Inf,
    detected::Bool = false,
    detection_health_state::Int = 0,
    peak_error::Float64 = 0.0,
    scenario_name::String = ""
)
    ttd = detected ? detection_time - fault_start_time : Inf
    TTDSample(fault_type, fault_start_time, detection_time, ttd,
              detected, detection_health_state, peak_error, scenario_name)
end

"""
    is_silent_divergence(sample::TTDSample, max_ttd::Float64) -> Bool

Check if this sample represents silent divergence (TTD exceeds threshold).
"""
function is_silent_divergence(sample::TTDSample, max_ttd::Float64)
    !sample.detected || sample.ttd > max_ttd
end

# ============================================================================
# TTD Statistics
# ============================================================================

"""
    TTDStatistics

Aggregate statistics for TTD measurements.

# Fields
- `n_samples`: Total number of samples
- `n_detected`: Number where fault was detected
- `n_undetected`: Number where fault was never detected
- `detection_rate`: Fraction detected (n_detected / n_samples)
- `mean_ttd`: Mean TTD for detected faults
- `std_ttd`: Standard deviation of TTD
- `min_ttd`: Minimum TTD
- `max_ttd`: Maximum TTD
- `p50_ttd`: Median TTD
- `p95_ttd`: 95th percentile TTD
- `p99_ttd`: 99th percentile TTD
- `silent_divergence_count`: Count exceeding threshold
- `silent_divergence_rate`: Fraction with silent divergence
"""
struct TTDStatistics
    n_samples::Int
    n_detected::Int
    n_undetected::Int
    detection_rate::Float64

    mean_ttd::Float64
    std_ttd::Float64
    min_ttd::Float64
    max_ttd::Float64
    p50_ttd::Float64
    p95_ttd::Float64
    p99_ttd::Float64

    silent_divergence_count::Int
    silent_divergence_rate::Float64

    # Per-fault-type breakdown
    ttd_by_fault_type::Dict{FaultType, Float64}
end

"""
    compute_ttd_statistics(samples::Vector{TTDSample};
                           max_ttd_threshold::Float64=5.0) -> TTDStatistics

Compute aggregate TTD statistics from samples.
"""
function compute_ttd_statistics(samples::Vector{TTDSample};
                                 max_ttd_threshold::Float64 = 5.0)
    n = length(samples)

    if n == 0
        return TTDStatistics(
            0, 0, 0, 0.0,
            NaN, NaN, NaN, NaN, NaN, NaN, NaN,
            0, 0.0,
            Dict{FaultType, Float64}()
        )
    end

    detected = filter(s -> s.detected, samples)
    n_detected = length(detected)
    n_undetected = n - n_detected
    detection_rate = n_detected / n

    # TTD statistics (only for detected faults)
    if n_detected > 0
        ttds = [s.ttd for s in detected]
        mean_ttd = mean(ttds)
        std_ttd = n_detected > 1 ? std(ttds) : 0.0
        min_ttd = minimum(ttds)
        max_ttd = maximum(ttds)

        sorted_ttds = sort(ttds)
        p50_ttd = sorted_ttds[max(1, ceil(Int, 0.5 * n_detected))]
        p95_ttd = sorted_ttds[max(1, ceil(Int, 0.95 * n_detected))]
        p99_ttd = sorted_ttds[max(1, ceil(Int, 0.99 * n_detected))]
    else
        mean_ttd = Inf
        std_ttd = NaN
        min_ttd = Inf
        max_ttd = Inf
        p50_ttd = Inf
        p95_ttd = Inf
        p99_ttd = Inf
    end

    # Silent divergence count
    silent_count = count(s -> is_silent_divergence(s, max_ttd_threshold), samples)
    silent_rate = silent_count / n

    # Per-fault-type mean TTD
    ttd_by_type = Dict{FaultType, Float64}()
    for ft in unique([s.fault_type for s in samples])
        ft_samples = filter(s -> s.fault_type == ft && s.detected, samples)
        if !isempty(ft_samples)
            ttd_by_type[ft] = mean([s.ttd for s in ft_samples])
        end
    end

    TTDStatistics(
        n, n_detected, n_undetected, detection_rate,
        mean_ttd, std_ttd, min_ttd, max_ttd, p50_ttd, p95_ttd, p99_ttd,
        silent_count, silent_rate,
        ttd_by_type
    )
end

# ============================================================================
# TTD Configuration
# ============================================================================

"""
    TTDConfig

Configuration for TTD tracking and gates.

# Fields
- `max_ttd_external`: Max TTD for external (customer-facing) gate (s)
- `max_ttd_internal`: Target TTD for internal engineering gate (s)
- `position_error_threshold`: Position error that triggers "fault active" (m)
- `min_health_state_for_detection`: Minimum health state for detection (1=CAUTION)
- `detection_persistence`: Samples health must stay elevated for detection
"""
Base.@kwdef struct TTDConfig
    max_ttd_external::Float64 = 5.0    # V1.0 external gate: 5s
    max_ttd_internal::Float64 = 3.0    # Internal target: 3s
    position_error_threshold::Float64 = 5.0  # 5m position error
    min_health_state_for_detection::Int = 1  # CAUTION
    detection_persistence::Int = 3           # 3 consecutive samples
end

const DEFAULT_TTD_CONFIG = TTDConfig()

# ============================================================================
# TTD Tracker
# ============================================================================

"""
    TTDTrackerState

Internal state for tracking a single fault injection.
"""
mutable struct TTDTrackerState
    fault_type::FaultType
    fault_start_time::Float64
    is_active::Bool

    detection_time::Float64
    detected::Bool
    detection_health_state::Int

    # Detection requires persistence
    elevated_health_count::Int

    # Error tracking
    peak_error::Float64

    # Scenario info
    scenario_name::String
end

"""
    TTDTracker

Real-time tracker for Time-To-Detect measurements.
"""
mutable struct TTDTracker
    config::TTDConfig
    samples::Vector{TTDSample}

    # Currently active fault tracking
    active_fault::Union{Nothing, TTDTrackerState}

    # Time series for analysis
    error_history::Vector{Float64}
    health_history::Vector{Int}
    timestamps::Vector{Float64}

    current_scenario::String
end

function TTDTracker(config::TTDConfig = DEFAULT_TTD_CONFIG)
    TTDTracker(
        config,
        TTDSample[],
        nothing,
        Float64[],
        Int[],
        Float64[],
        ""
    )
end

"""
    reset_tracker!(tracker::TTDTracker)

Reset tracker state for a new scenario.
"""
function reset_tracker!(tracker::TTDTracker)
    tracker.active_fault = nothing
    empty!(tracker.error_history)
    empty!(tracker.health_history)
    empty!(tracker.timestamps)
end

"""
    start_fault!(tracker, fault_type, fault_start_time; scenario_name="")

Notify tracker that a fault has been injected.
"""
function start_fault!(tracker::TTDTracker, fault_type::FaultType,
                      fault_start_time::Float64;
                      scenario_name::String = tracker.current_scenario)
    # If there's an active fault, finalize it first
    if tracker.active_fault !== nothing
        finalize_fault!(tracker)
    end

    tracker.active_fault = TTDTrackerState(
        fault_type,
        fault_start_time,
        true,
        Inf,
        false,
        0,
        0,
        0.0,
        scenario_name
    )
end

"""
    update_tracker!(tracker, position_error, health_state, timestamp)

Update tracker with current state.
"""
function update_tracker!(tracker::TTDTracker,
                         position_error::Float64,
                         health_state::Int,
                         timestamp::Float64)
    # Update history
    push!(tracker.error_history, position_error)
    push!(tracker.health_history, health_state)
    push!(tracker.timestamps, timestamp)

    # Trim history
    max_history = 1000
    if length(tracker.error_history) > max_history
        deleteat!(tracker.error_history, 1:length(tracker.error_history) - max_history)
        deleteat!(tracker.health_history, 1:length(tracker.health_history) - max_history)
        deleteat!(tracker.timestamps, 1:length(tracker.timestamps) - max_history)
    end

    # Update active fault tracking
    if tracker.active_fault !== nothing
        state = tracker.active_fault

        # Update peak error
        state.peak_error = max(state.peak_error, position_error)

        # Check for detection (if not already detected)
        if !state.detected
            if health_state >= tracker.config.min_health_state_for_detection
                state.elevated_health_count += 1

                if state.elevated_health_count >= tracker.config.detection_persistence
                    # Detected!
                    state.detected = true
                    state.detection_time = timestamp
                    state.detection_health_state = health_state
                end
            else
                state.elevated_health_count = 0
            end
        end
    end
end

"""
    record_detection!(tracker, health_state, timestamp)

Manually record a detection event.
"""
function record_detection!(tracker::TTDTracker, health_state::Int, timestamp::Float64)
    if tracker.active_fault !== nothing && !tracker.active_fault.detected
        tracker.active_fault.detected = true
        tracker.active_fault.detection_time = timestamp
        tracker.active_fault.detection_health_state = health_state
    end
end

"""
    finalize_fault!(tracker)

Finalize current fault tracking and create a TTDSample.
"""
function finalize_fault!(tracker::TTDTracker)
    if tracker.active_fault === nothing
        return nothing
    end

    state = tracker.active_fault

    sample = TTDSample(
        fault_type = state.fault_type,
        fault_start_time = state.fault_start_time,
        detection_time = state.detection_time,
        detected = state.detected,
        detection_health_state = state.detection_health_state,
        peak_error = state.peak_error,
        scenario_name = state.scenario_name
    )

    push!(tracker.samples, sample)
    tracker.active_fault = nothing

    return sample
end

"""
    get_ttd_samples(tracker) -> Vector{TTDSample}

Get all recorded TTD samples.
"""
function get_ttd_samples(tracker::TTDTracker)
    # Finalize any active fault first
    if tracker.active_fault !== nothing
        finalize_fault!(tracker)
    end
    return tracker.samples
end

# ============================================================================
# TTD Gate
# ============================================================================

"""
    TTDGateResult

Result of TTD gate evaluation.
"""
struct TTDGateResult
    passed::Bool
    max_ttd_observed::Float64
    max_ttd_threshold::Float64

    # Statistics
    n_samples::Int
    n_passed::Int
    n_failed::Int
    pass_rate::Float64

    # Detailed statistics
    statistics::TTDStatistics

    # Failed samples (for debugging)
    failed_samples::Vector{TTDSample}

    # Summary
    summary::String
end

"""
    TTDGate

Gate for TTD-based qualification.
"""
struct TTDGate
    config::TTDConfig
    tier::GateTier
    name::String
    description::String
end

function TTDGate(;
    config::TTDConfig = DEFAULT_TTD_CONFIG,
    tier::GateTier = TIER_EXTERNAL,
    name::String = "EXT_006",
    description::String = "Time-To-Detect < max threshold"
)
    TTDGate(config, tier, name, description)
end

"""
    evaluate_ttd_gate(gate::TTDGate, samples::Vector{TTDSample}) -> TTDGateResult

Evaluate TTD gate against samples.
"""
function evaluate_ttd_gate(gate::TTDGate, samples::Vector{TTDSample})
    threshold = gate.tier == TIER_EXTERNAL ?
                gate.config.max_ttd_external :
                gate.config.max_ttd_internal

    n = length(samples)

    if n == 0
        return TTDGateResult(
            true, 0.0, threshold,
            0, 0, 0, 1.0,
            TTDStatistics(0, 0, 0, 0.0, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0, 0.0, Dict()),
            TTDSample[],
            "No samples to evaluate"
        )
    end

    # Compute statistics
    stats = compute_ttd_statistics(samples; max_ttd_threshold = threshold)

    # Find failed samples
    failed = filter(s -> is_silent_divergence(s, threshold), samples)
    n_failed = length(failed)
    n_passed = n - n_failed
    pass_rate = n_passed / n

    # Max observed TTD
    detected = filter(s -> s.detected, samples)
    max_ttd_observed = isempty(detected) ? Inf : maximum(s.ttd for s in detected)

    # Gate passes if no silent divergence
    passed = n_failed == 0

    # Summary
    summary = if passed
        "PASS: All $(n) faults detected within $(threshold)s (max TTD: $(round(max_ttd_observed, digits=2))s)"
    else
        "FAIL: $(n_failed)/$(n) faults had TTD > $(threshold)s (silent divergence)"
    end

    TTDGateResult(
        passed, max_ttd_observed, threshold,
        n, n_passed, n_failed, pass_rate,
        stats, failed, summary
    )
end

# ============================================================================
# TTD Scenario Execution
# ============================================================================

"""
    TTDScenarioResult

Result of running TTD measurement for a single scenario.
"""
struct TTDScenarioResult
    scenario_name::String
    samples::Vector{TTDSample}
    statistics::TTDStatistics

    # Per-fault results
    fault_results::Vector{TTDSample}

    # Gate results
    external_gate_passed::Bool
    internal_gate_passed::Bool
    max_ttd::Float64
end

"""
    run_ttd_scenario(scenario, tracker, step_func, dt; scenario_duration=nothing)

Run a scenario and collect TTD measurements.

# Arguments
- `scenario`: FaultScenario to run
- `tracker`: TTDTracker instance
- `step_func`: Function(time, dt) -> (position_error, health_state)
- `dt`: Time step

Returns TTDScenarioResult with TTD measurements.
"""
function run_ttd_scenario(scenario::FaultScenario,
                          tracker::TTDTracker,
                          step_func::Function,
                          dt::Float64;
                          scenario_duration::Union{Nothing, Float64} = nothing)
    duration = something(scenario_duration, scenario.duration)

    # Reset tracker
    reset_tracker!(tracker)
    tracker.current_scenario = scenario.name

    # Start faults as specified in scenario
    for fault in scenario.faults
        # Schedule fault start (will be triggered when time reaches start_time)
    end

    # Run simulation
    time = 0.0
    fault_started = Dict{Int, Bool}()

    while time < duration
        # Check if any faults should start
        for (i, fault) in enumerate(scenario.faults)
            if !get(fault_started, i, false) && time >= fault.start_time
                start_fault!(tracker, fault.fault_type, fault.start_time;
                            scenario_name = scenario.name)
                fault_started[i] = true
            end
        end

        # Get current state from simulation
        position_error, health_state = step_func(time, dt)

        # Update tracker
        update_tracker!(tracker, position_error, health_state, time)

        time += dt
    end

    # Finalize any remaining active fault
    finalize_fault!(tracker)

    # Get samples for this scenario
    scenario_samples = filter(s -> s.scenario_name == scenario.name, tracker.samples)

    # Compute statistics
    stats = compute_ttd_statistics(scenario_samples;
                                   max_ttd_threshold = tracker.config.max_ttd_external)

    # Evaluate gates
    external_gate = TTDGate(config = tracker.config, tier = TIER_EXTERNAL)
    internal_gate = TTDGate(config = tracker.config, tier = TIER_INTERNAL)

    ext_result = evaluate_ttd_gate(external_gate, scenario_samples)
    int_result = evaluate_ttd_gate(internal_gate, scenario_samples)

    max_ttd = stats.max_ttd

    TTDScenarioResult(
        scenario.name,
        scenario_samples,
        stats,
        scenario_samples,
        ext_result.passed,
        int_result.passed,
        max_ttd
    )
end

"""
    TTDSuiteResult

Result of running TTD measurements across multiple scenarios.
"""
struct TTDSuiteResult
    scenarios::Vector{TTDScenarioResult}
    overall_statistics::TTDStatistics

    # Aggregate results
    n_scenarios::Int
    n_scenarios_external_pass::Int
    n_scenarios_internal_pass::Int

    # Overall gate status
    external_gate_passed::Bool
    internal_gate_passed::Bool

    # Summary
    max_ttd_observed::Float64
    mean_ttd::Float64
end

"""
    analyze_ttd_results(results::Vector{TTDScenarioResult}) -> TTDSuiteResult

Aggregate TTD results from multiple scenarios.
"""
function analyze_ttd_results(results::Vector{TTDScenarioResult})
    n = length(results)

    if n == 0
        return TTDSuiteResult(
            TTDScenarioResult[],
            TTDStatistics(0, 0, 0, 0.0, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0, 0.0, Dict()),
            0, 0, 0, true, true, 0.0, NaN
        )
    end

    # Collect all samples
    all_samples = TTDSample[]
    for r in results
        append!(all_samples, r.samples)
    end

    # Overall statistics
    overall_stats = compute_ttd_statistics(all_samples)

    # Aggregate gate results
    n_ext_pass = count(r -> r.external_gate_passed, results)
    n_int_pass = count(r -> r.internal_gate_passed, results)

    ext_passed = n_ext_pass == n
    int_passed = n_int_pass == n

    max_ttd = maximum(r.max_ttd for r in results; init=0.0)

    TTDSuiteResult(
        results,
        overall_stats,
        n, n_ext_pass, n_int_pass,
        ext_passed, int_passed,
        max_ttd,
        overall_stats.mean_ttd
    )
end

# ============================================================================
# Reporting
# ============================================================================

"""
    format_ttd_report(result::TTDGateResult) -> String

Format a TTD gate result as a report.
"""
function format_ttd_report(result::TTDGateResult)
    lines = String[]

    push!(lines, "=" ^ 60)
    push!(lines, "TIME-TO-DETECT (TTD) GATE REPORT")
    push!(lines, "=" ^ 60)
    push!(lines, "")

    status = result.passed ? "PASS" : "FAIL"
    push!(lines, "Status: $status")
    push!(lines, result.summary)
    push!(lines, "")

    push!(lines, "-" ^ 60)
    push!(lines, "STATISTICS")
    push!(lines, "-" ^ 60)

    stats = result.statistics
    push!(lines, "Total samples: $(stats.n_samples)")
    push!(lines, "Detection rate: $(round(stats.detection_rate * 100, digits=1))%")
    push!(lines, "")

    if stats.n_detected > 0
        push!(lines, "TTD (detected faults):")
        push!(lines, "  Mean:   $(round(stats.mean_ttd, digits=3))s")
        push!(lines, "  Std:    $(round(stats.std_ttd, digits=3))s")
        push!(lines, "  Min:    $(round(stats.min_ttd, digits=3))s")
        push!(lines, "  Max:    $(round(stats.max_ttd, digits=3))s")
        push!(lines, "  p50:    $(round(stats.p50_ttd, digits=3))s")
        push!(lines, "  p95:    $(round(stats.p95_ttd, digits=3))s")
        push!(lines, "  p99:    $(round(stats.p99_ttd, digits=3))s")
    else
        push!(lines, "No faults detected")
    end
    push!(lines, "")

    push!(lines, "Silent divergence (TTD > $(result.max_ttd_threshold)s):")
    push!(lines, "  Count: $(stats.silent_divergence_count)")
    push!(lines, "  Rate:  $(round(stats.silent_divergence_rate * 100, digits=1))%")
    push!(lines, "")

    if !isempty(result.failed_samples)
        push!(lines, "-" ^ 60)
        push!(lines, "FAILED SAMPLES")
        push!(lines, "-" ^ 60)
        for (i, s) in enumerate(result.failed_samples[1:min(5, length(result.failed_samples))])
            push!(lines, "[$i] $(s.scenario_name): $(s.fault_type)")
            push!(lines, "    TTD: $(isinf(s.ttd) ? "NOT DETECTED" : "$(round(s.ttd, digits=2))s")")
            push!(lines, "    Peak error: $(round(s.peak_error, digits=2))m")
        end
        if length(result.failed_samples) > 5
            push!(lines, "... and $(length(result.failed_samples) - 5) more")
        end
    end

    push!(lines, "=" ^ 60)

    join(lines, "\n")
end

"""
    format_ttd_summary(suite::TTDSuiteResult) -> String

Format a TTD suite summary.
"""
function format_ttd_summary(suite::TTDSuiteResult)
    lines = String[]

    push!(lines, "=" ^ 60)
    push!(lines, "TTD QUALIFICATION SUMMARY")
    push!(lines, "=" ^ 60)
    push!(lines, "")

    ext_status = suite.external_gate_passed ? "PASS" : "FAIL"
    int_status = suite.internal_gate_passed ? "PASS" : "FAIL"

    push!(lines, "External Gate (max 5s): $ext_status ($(suite.n_scenarios_external_pass)/$(suite.n_scenarios))")
    push!(lines, "Internal Gate (max 3s): $int_status ($(suite.n_scenarios_internal_pass)/$(suite.n_scenarios))")
    push!(lines, "")

    push!(lines, "Overall TTD:")
    push!(lines, "  Max observed: $(round(suite.max_ttd_observed, digits=2))s")
    if !isnan(suite.mean_ttd)
        push!(lines, "  Mean:         $(round(suite.mean_ttd, digits=2))s")
    end
    push!(lines, "")

    push!(lines, "-" ^ 60)
    push!(lines, "PER-SCENARIO RESULTS")
    push!(lines, "-" ^ 60)

    for r in suite.scenarios
        ext_mark = r.external_gate_passed ? "+" : "X"
        int_mark = r.internal_gate_passed ? "+" : "X"
        max_str = isinf(r.max_ttd) ? "N/A" : "$(round(r.max_ttd, digits=2))s"
        push!(lines, "[$ext_mark|$int_mark] $(r.scenario_name): max TTD = $max_str")
    end

    push!(lines, "=" ^ 60)

    join(lines, "\n")
end

# ============================================================================
# Integration with Tiered Gates
# ============================================================================

"""
    create_ttd_external_gate(config::TTDConfig) -> TieredGate

Create external TTD gate for V1.0 qualification.
"""
function create_ttd_external_gate(config::TTDConfig = DEFAULT_TTD_CONFIG)
    threshold = config.max_ttd_external
    TieredGate(
        "EXT_006",
        "Time-To-Detect",
        "All faults detected within $(threshold)s",
        TIER_EXTERNAL,
        threshold,
        "s",
        function(results)
            # Check max TTD from results
            max_ttd = get(results, :max_ttd, 0.0)
            passed = max_ttd <= threshold
            message = passed ?
                "Max TTD $(round(max_ttd, digits=2))s <= $(threshold)s" :
                "Max TTD $(round(max_ttd, digits=2))s > $(threshold)s (SILENT DIVERGENCE)"
            (passed, max_ttd, message)
        end
    )
end

"""
    create_ttd_internal_gate(config::TTDConfig) -> TieredGate

Create internal TTD gate for engineering diagnostics.
"""
function create_ttd_internal_gate(config::TTDConfig = DEFAULT_TTD_CONFIG)
    threshold = config.max_ttd_internal
    TieredGate(
        "INT_005",
        "Time-To-Detect (Engineering)",
        "Engineering target: faults detected within $(threshold)s",
        TIER_INTERNAL,
        threshold,
        "s",
        function(results)
            max_ttd = get(results, :max_ttd, 0.0)
            passed = max_ttd <= threshold
            message = passed ?
                "Max TTD $(round(max_ttd, digits=2))s <= $(threshold)s" :
                "Max TTD $(round(max_ttd, digits=2))s > $(threshold)s"
            (passed, max_ttd, message)
        end
    )
end

export create_ttd_external_gate, create_ttd_internal_gate
