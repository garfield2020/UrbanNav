# ============================================================================
# qualification.jl - Two-Mode Qualification Framework
# ============================================================================
#
# Step 10 of NEES Recovery Plan:
# Re-run qualification with two explicit modes.
#
# The key insight is that observability and performance are separate concerns:
#
# Mode 1: OBSERVABILITY QUALIFICATION
#   - Can we estimate the state at all?
#   - Are unobservable conditions properly detected and flagged?
#   - Does the health system respond appropriately?
#   Pass criteria: No silent divergence, proper health flags
#
# Mode 2: PERFORMANCE QUALIFICATION
#   - How accurately do we estimate observable states?
#   - Is NEES properly calibrated (≈ state_dim)?
#   - Is consistency above threshold (>85%)?
#   Pass criteria: NEES in bounds, RMSE acceptable, consistency >85%
#
# This separation allows:
# - Clear root cause analysis when things fail
# - Different acceptance criteria for each mode
# - Performance testing only in observable regions
# ============================================================================

export QualificationMode, MODE_OBSERVABILITY, MODE_PERFORMANCE
export QualificationConfig, QualificationResult, QualificationReport
export QualificationRunner, QualificationGate
export run_qualification!, run_observability_qualification!, run_performance_qualification!
export create_default_qualification, qualification_summary
export QUAL_PASS, QUAL_FAIL, QUAL_CONDITIONAL, QUAL_NOT_RUN

using LinearAlgebra
using Statistics

# ============================================================================
# Qualification Mode
# ============================================================================

"""
    QualificationMode

The two modes of qualification testing.
"""
@enum QualificationMode begin
    MODE_OBSERVABILITY = 1   # Test observability detection
    MODE_PERFORMANCE = 2     # Test estimation accuracy
end

"""
    QualificationStatus

Overall qualification status.
"""
@enum QualificationStatus begin
    QUAL_NOT_RUN = 0      # Not yet executed
    QUAL_PASS = 1         # All criteria met
    QUAL_CONDITIONAL = 2  # Some criteria met with known limitations
    QUAL_FAIL = 3         # Critical criteria not met
end

# ============================================================================
# Qualification Configuration
# ============================================================================

"""
    QualificationConfig

Configuration for qualification testing.

# Observability Criteria
- `max_silent_divergence_time::Float64`: Max time error can grow undetected (s)
- `min_health_response_rate::Float64`: Fraction of faults that must be flagged
- `observability_condition_threshold::Float64`: Max condition number for "observable"

# Performance Criteria
- `nees_target::Float64`: Expected NEES (≈ state_dim for well-calibrated filter)
- `nees_tolerance::Float64`: Acceptable deviation from target
- `min_consistency::Float64`: Minimum fraction of NEES in chi-squared bounds
- `max_rmse_position::Float64`: Maximum position RMSE (m)
- `max_rmse_velocity::Float64`: Maximum velocity RMSE (m/s)

# Test Configuration
- `n_monte_carlo::Int`: Number of Monte Carlo runs per scenario
- `scenario_duration::Float64`: Duration of each scenario (s)
- `dt::Float64`: Time step (s)
"""
Base.@kwdef struct QualificationConfig
    # Observability criteria
    max_silent_divergence_time::Float64 = 5.0
    min_health_response_rate::Float64 = 0.95
    observability_condition_threshold::Float64 = 1e6

    # Performance criteria
    nees_target::Float64 = 3.0          # For 3-DOF position
    nees_tolerance::Float64 = 0.5       # ±0.5 around target
    min_consistency::Float64 = 0.85     # 85% in bounds
    max_rmse_position::Float64 = 5.0    # 5m
    max_rmse_velocity::Float64 = 0.5    # 0.5 m/s

    # Test configuration
    n_monte_carlo::Int = 10
    scenario_duration::Float64 = 60.0
    dt::Float64 = 0.1
end

const DEFAULT_QUALIFICATION_CONFIG = QualificationConfig()

# ============================================================================
# Qualification Results
# ============================================================================

"""
    ObservabilityQualResult

Result of observability qualification for one scenario.
"""
struct ObservabilityQualResult
    scenario_name::String
    passed::Bool
    silent_divergence_detected::Bool
    max_undetected_time::Float64
    health_response_rate::Float64
    condition_numbers::Vector{Float64}
    failure_reasons::Vector{String}
end

"""
    PerformanceQualResult

Result of performance qualification for one scenario.
"""
struct PerformanceQualResult
    scenario_name::String
    passed::Bool
    nees_mean::Float64
    nees_std::Float64
    consistency::Float64
    rmse_position::Float64
    rmse_velocity::Float64
    failure_reasons::Vector{String}
end

"""
    QualificationResult

Complete result of a qualification run.
"""
struct QualificationResult
    mode::QualificationMode
    status::QualificationStatus
    config::QualificationConfig
    observability_results::Vector{ObservabilityQualResult}
    performance_results::Vector{PerformanceQualResult}
    start_time::Float64
    end_time::Float64
    summary::String
end

# ============================================================================
# Qualification Report
# ============================================================================

"""
    QualificationReport

Human-readable qualification report.
"""
struct QualificationReport
    title::String
    timestamp::Float64
    observability_status::QualificationStatus
    performance_status::QualificationStatus
    overall_status::QualificationStatus

    # Summary statistics
    n_obs_scenarios::Int
    n_obs_passed::Int
    n_perf_scenarios::Int
    n_perf_passed::Int

    # Key metrics
    worst_silent_time::Float64
    best_consistency::Float64
    worst_consistency::Float64
    mean_nees::Float64

    # Detailed findings
    critical_issues::Vector{String}
    warnings::Vector{String}
    recommendations::Vector{String}
end

# ============================================================================
# Qualification Gates
# ============================================================================

"""
    QualificationGate

A single pass/fail gate in the qualification process.
"""
struct QualificationGate
    name::String
    description::String
    check::Function  # (results) -> (passed::Bool, message::String)
    critical::Bool   # If true, failure blocks qualification
end

"""
    create_observability_gates(config) -> Vector{QualificationGate}

Create standard observability qualification gates.
"""
function create_observability_gates(config::QualificationConfig)
    gates = QualificationGate[]

    # Gate 1: No silent divergence
    push!(gates, QualificationGate(
        "no_silent_divergence",
        "No undetected position divergence beyond threshold",
        function(results)
            for r in results
                if r.silent_divergence_detected
                    return (false, "Silent divergence in $(r.scenario_name)")
                end
                if r.max_undetected_time > config.max_silent_divergence_time
                    return (false, "Undetected time $(r.max_undetected_time)s > $(config.max_silent_divergence_time)s in $(r.scenario_name)")
                end
            end
            return (true, "No silent divergence detected")
        end,
        true  # Critical
    ))

    # Gate 2: Health system response
    push!(gates, QualificationGate(
        "health_response",
        "Health system flags faults at required rate",
        function(results)
            response_rates = [r.health_response_rate for r in results]
            min_rate = isempty(response_rates) ? 1.0 : minimum(response_rates)
            if min_rate < config.min_health_response_rate
                return (false, "Health response rate $(min_rate) < $(config.min_health_response_rate)")
            end
            return (true, "Health response rate $(min_rate) >= $(config.min_health_response_rate)")
        end,
        true  # Critical
    ))

    # Gate 3: Observability condition numbers
    push!(gates, QualificationGate(
        "condition_numbers",
        "Observability condition numbers within bounds",
        function(results)
            for r in results
                if !isempty(r.condition_numbers)
                    max_cond = maximum(r.condition_numbers)
                    if max_cond > config.observability_condition_threshold
                        return (false, "Condition number $max_cond > threshold in $(r.scenario_name)")
                    end
                end
            end
            return (true, "All condition numbers within bounds")
        end,
        false  # Warning only
    ))

    return gates
end

"""
    create_performance_gates(config) -> Vector{QualificationGate}

Create standard performance qualification gates.
"""
function create_performance_gates(config::QualificationConfig)
    gates = QualificationGate[]

    # Gate 1: NEES calibration
    push!(gates, QualificationGate(
        "nees_calibration",
        "NEES properly calibrated (mean ≈ state_dim)",
        function(results)
            nees_means = [r.nees_mean for r in results if !isinf(r.nees_mean)]
            if isempty(nees_means)
                return (false, "No valid NEES data")
            end
            overall_mean = mean(nees_means)
            if abs(overall_mean - config.nees_target) > config.nees_tolerance
                return (false, "NEES mean $(overall_mean) outside $(config.nees_target) ± $(config.nees_tolerance)")
            end
            return (true, "NEES mean $(overall_mean) within bounds")
        end,
        true  # Critical
    ))

    # Gate 2: Consistency
    push!(gates, QualificationGate(
        "consistency",
        "NEES consistency above threshold",
        function(results)
            consistencies = [r.consistency for r in results]
            min_cons = isempty(consistencies) ? 0.0 : minimum(consistencies)
            if min_cons < config.min_consistency
                return (false, "Consistency $(min_cons) < $(config.min_consistency)")
            end
            return (true, "Consistency $(min_cons) >= $(config.min_consistency)")
        end,
        true  # Critical
    ))

    # Gate 3: Position RMSE
    push!(gates, QualificationGate(
        "rmse_position",
        "Position RMSE within bounds",
        function(results)
            rmses = [r.rmse_position for r in results if !isinf(r.rmse_position)]
            if isempty(rmses)
                return (false, "No valid RMSE data")
            end
            max_rmse = maximum(rmses)
            if max_rmse > config.max_rmse_position
                return (false, "Position RMSE $(max_rmse) > $(config.max_rmse_position)")
            end
            return (true, "Position RMSE $(max_rmse) <= $(config.max_rmse_position)")
        end,
        true  # Critical
    ))

    # Gate 4: Velocity RMSE
    push!(gates, QualificationGate(
        "rmse_velocity",
        "Velocity RMSE within bounds",
        function(results)
            rmses = [r.rmse_velocity for r in results if !isinf(r.rmse_velocity)]
            if isempty(rmses)
                return (true, "No velocity RMSE data (OK)")
            end
            max_rmse = maximum(rmses)
            if max_rmse > config.max_rmse_velocity
                return (false, "Velocity RMSE $(max_rmse) > $(config.max_rmse_velocity)")
            end
            return (true, "Velocity RMSE $(max_rmse) <= $(config.max_rmse_velocity)")
        end,
        false  # Warning only
    ))

    return gates
end

export create_observability_gates, create_performance_gates

# ============================================================================
# Qualification Runner
# ============================================================================

"""
    QualificationRunner

Main qualification test runner.

# Fields
- `config::QualificationConfig`: Configuration
- `observability_gates::Vector{QualificationGate}`: Observability gates
- `performance_gates::Vector{QualificationGate}`: Performance gates
- `scenarios::Vector{FaultScenario}`: Test scenarios
- `results::Vector{QualificationResult}`: Accumulated results
"""
mutable struct QualificationRunner
    config::QualificationConfig
    observability_gates::Vector{QualificationGate}
    performance_gates::Vector{QualificationGate}
    scenarios::Vector{Any}  # FaultScenario or DOE
    results::Vector{QualificationResult}
end

function QualificationRunner(config::QualificationConfig = DEFAULT_QUALIFICATION_CONFIG)
    QualificationRunner(
        config,
        create_observability_gates(config),
        create_performance_gates(config),
        Any[],
        QualificationResult[]
    )
end

"""
    add_scenario!(runner, scenario)

Add a test scenario to the runner.
"""
function add_scenario!(runner::QualificationRunner, scenario)
    push!(runner.scenarios, scenario)
    return runner
end

"""
    run_observability_qualification!(runner, run_func) -> QualificationResult

Run observability qualification tests.

# Arguments
- `runner`: QualificationRunner
- `run_func`: Function(scenario) -> ObservabilityQualResult
"""
function run_observability_qualification!(runner::QualificationRunner, run_func::Function)
    start_time = time()
    results = ObservabilityQualResult[]

    for scenario in runner.scenarios
        try
            result = run_func(scenario)
            push!(results, result)
        catch e
            # Record failure
            push!(results, ObservabilityQualResult(
                string(scenario),
                false,
                true,  # Assume silent divergence on error
                Inf,
                0.0,
                Float64[],
                [string(e)]
            ))
        end
    end

    # Evaluate gates
    all_passed = true
    critical_failed = false
    failure_messages = String[]

    for gate in runner.observability_gates
        passed, message = gate.check(results)
        if !passed
            push!(failure_messages, "$(gate.name): $message")
            if gate.critical
                critical_failed = true
            end
            all_passed = false
        end
    end

    # Determine status
    status = if critical_failed
        QUAL_FAIL
    elseif all_passed
        QUAL_PASS
    else
        QUAL_CONDITIONAL
    end

    summary = if status == QUAL_PASS
        "Observability qualification PASSED: $(length(results)) scenarios, all gates passed"
    elseif status == QUAL_CONDITIONAL
        "Observability qualification CONDITIONAL: $(join(failure_messages, "; "))"
    else
        "Observability qualification FAILED: $(join(failure_messages, "; "))"
    end

    result = QualificationResult(
        MODE_OBSERVABILITY,
        status,
        runner.config,
        results,
        PerformanceQualResult[],
        start_time,
        time(),
        summary
    )

    push!(runner.results, result)
    return result
end

"""
    run_performance_qualification!(runner, run_func) -> QualificationResult

Run performance qualification tests.

# Arguments
- `runner`: QualificationRunner
- `run_func`: Function(scenario) -> PerformanceQualResult
"""
function run_performance_qualification!(runner::QualificationRunner, run_func::Function)
    start_time = time()
    results = PerformanceQualResult[]

    for scenario in runner.scenarios
        try
            result = run_func(scenario)
            push!(results, result)
        catch e
            # Record failure
            push!(results, PerformanceQualResult(
                string(scenario),
                false,
                Inf, Inf, 0.0, Inf, Inf,
                [string(e)]
            ))
        end
    end

    # Evaluate gates
    all_passed = true
    critical_failed = false
    failure_messages = String[]

    for gate in runner.performance_gates
        passed, message = gate.check(results)
        if !passed
            push!(failure_messages, "$(gate.name): $message")
            if gate.critical
                critical_failed = true
            end
            all_passed = false
        end
    end

    # Determine status
    status = if critical_failed
        QUAL_FAIL
    elseif all_passed
        QUAL_PASS
    else
        QUAL_CONDITIONAL
    end

    summary = if status == QUAL_PASS
        "Performance qualification PASSED: $(length(results)) scenarios, all gates passed"
    elseif status == QUAL_CONDITIONAL
        "Performance qualification CONDITIONAL: $(join(failure_messages, "; "))"
    else
        "Performance qualification FAILED: $(join(failure_messages, "; "))"
    end

    result = QualificationResult(
        MODE_PERFORMANCE,
        status,
        runner.config,
        ObservabilityQualResult[],
        results,
        start_time,
        time(),
        summary
    )

    push!(runner.results, result)
    return result
end

"""
    run_qualification!(runner, obs_func, perf_func) -> QualificationReport

Run full two-mode qualification.

# Arguments
- `runner`: QualificationRunner
- `obs_func`: Function for observability tests
- `perf_func`: Function for performance tests

# Returns
Complete QualificationReport with both modes.
"""
function run_qualification!(runner::QualificationRunner,
                            obs_func::Function,
                            perf_func::Function)
    # Run observability qualification first
    obs_result = run_observability_qualification!(runner, obs_func)

    # Run performance qualification
    perf_result = run_performance_qualification!(runner, perf_func)

    # Generate report
    return generate_report(runner, obs_result, perf_result)
end

"""
    generate_report(runner, obs_result, perf_result) -> QualificationReport

Generate a complete qualification report.
"""
function generate_report(runner::QualificationRunner,
                         obs_result::QualificationResult,
                         perf_result::QualificationResult)

    # Overall status: worst of the two
    overall_status = max(obs_result.status, perf_result.status)

    # Count passed scenarios
    n_obs_passed = count(r -> r.passed, obs_result.observability_results)
    n_perf_passed = count(r -> r.passed, perf_result.performance_results)

    # Key metrics
    silent_times = [r.max_undetected_time for r in obs_result.observability_results]
    worst_silent = isempty(silent_times) ? 0.0 : maximum(filter(!isinf, silent_times))

    consistencies = [r.consistency for r in perf_result.performance_results]
    best_cons = isempty(consistencies) ? 0.0 : maximum(consistencies)
    worst_cons = isempty(consistencies) ? 0.0 : minimum(consistencies)

    nees_means = [r.nees_mean for r in perf_result.performance_results if !isinf(r.nees_mean)]
    mean_nees = isempty(nees_means) ? 0.0 : mean(nees_means)

    # Collect issues
    critical_issues = String[]
    warnings = String[]
    recommendations = String[]

    if obs_result.status == QUAL_FAIL
        push!(critical_issues, "Observability qualification failed")
        for r in obs_result.observability_results
            append!(critical_issues, r.failure_reasons)
        end
    end

    if perf_result.status == QUAL_FAIL
        push!(critical_issues, "Performance qualification failed")
        for r in perf_result.performance_results
            append!(critical_issues, r.failure_reasons)
        end
    end

    if obs_result.status == QUAL_CONDITIONAL
        push!(warnings, "Observability qualification passed with conditions")
    end

    if perf_result.status == QUAL_CONDITIONAL
        push!(warnings, "Performance qualification passed with conditions")
    end

    # Generate recommendations
    if worst_silent > 0
        push!(recommendations, "Review health checker thresholds for faster fault detection")
    end

    if worst_cons < runner.config.min_consistency
        push!(recommendations, "Tune process noise or measurement noise for better NEES calibration")
    end

    if mean_nees > runner.config.nees_target + runner.config.nees_tolerance
        push!(recommendations, "Filter may be overconfident - increase process noise or Q_model")
    elseif mean_nees < runner.config.nees_target - runner.config.nees_tolerance
        push!(recommendations, "Filter may be underconfident - decrease process noise")
    end

    return QualificationReport(
        "D8 Navigation Qualification",
        time(),
        obs_result.status,
        perf_result.status,
        overall_status,
        length(obs_result.observability_results),
        n_obs_passed,
        length(perf_result.performance_results),
        n_perf_passed,
        worst_silent,
        best_cons,
        worst_cons,
        mean_nees,
        unique(critical_issues),
        unique(warnings),
        unique(recommendations)
    )
end

export QualificationRunner, add_scenario!
export run_observability_qualification!, run_performance_qualification!, run_qualification!

# ============================================================================
# Qualification Summary
# ============================================================================

"""
    qualification_summary(report::QualificationReport) -> String

Generate a human-readable summary of the qualification report.
"""
function qualification_summary(report::QualificationReport)
    status_str = if report.overall_status == QUAL_PASS
        "PASS"
    elseif report.overall_status == QUAL_CONDITIONAL
        "CONDITIONAL PASS"
    elseif report.overall_status == QUAL_FAIL
        "FAIL"
    else
        "NOT RUN"
    end

    lines = String[]
    push!(lines, "=" ^ 60)
    push!(lines, "$(report.title)")
    push!(lines, "=" ^ 60)
    push!(lines, "")
    push!(lines, "OVERALL STATUS: $status_str")
    push!(lines, "")
    push!(lines, "Observability Qualification: $(report.observability_status)")
    push!(lines, "  Scenarios: $(report.n_obs_passed)/$(report.n_obs_scenarios) passed")
    push!(lines, "  Worst silent divergence time: $(round(report.worst_silent_time, digits=2))s")
    push!(lines, "")
    push!(lines, "Performance Qualification: $(report.performance_status)")
    push!(lines, "  Scenarios: $(report.n_perf_passed)/$(report.n_perf_scenarios) passed")
    push!(lines, "  NEES mean: $(round(report.mean_nees, digits=2))")
    push!(lines, "  Consistency range: $(round(report.worst_consistency, digits=2)) - $(round(report.best_consistency, digits=2))")
    push!(lines, "")

    if !isempty(report.critical_issues)
        push!(lines, "CRITICAL ISSUES:")
        for issue in report.critical_issues
            push!(lines, "  - $issue")
        end
        push!(lines, "")
    end

    if !isempty(report.warnings)
        push!(lines, "WARNINGS:")
        for warning in report.warnings
            push!(lines, "  - $warning")
        end
        push!(lines, "")
    end

    if !isempty(report.recommendations)
        push!(lines, "RECOMMENDATIONS:")
        for rec in report.recommendations
            push!(lines, "  - $rec")
        end
        push!(lines, "")
    end

    push!(lines, "=" ^ 60)

    return join(lines, "\n")
end

export qualification_summary

# ============================================================================
# Default Qualification Setup
# ============================================================================

"""
    create_default_qualification() -> QualificationRunner

Create a default qualification runner with standard scenarios.
"""
function create_default_qualification()
    config = QualificationConfig()
    runner = QualificationRunner(config)

    # Add standard fault scenarios (from Step 8)
    add_scenario!(runner, (
        name = "lawnmower_nominal",
        trajectory = TRAJ_LAWNMOWER,
        field_strength = FIELD_NOMINAL,
        sensors = SENSORS_FULL
    ))

    add_scenario!(runner, (
        name = "straight_line",
        trajectory = TRAJ_STRAIGHT,
        field_strength = FIELD_NOMINAL,
        sensors = SENSORS_FULL
    ))

    add_scenario!(runner, (
        name = "dvl_dropout",
        trajectory = TRAJ_LAWNMOWER,
        field_strength = FIELD_NOMINAL,
        sensors = SENSORS_NO_ODOMETRY
    ))

    add_scenario!(runner, (
        name = "weak_gradient",
        trajectory = TRAJ_LAWNMOWER,
        field_strength = FIELD_WEAK,
        sensors = SENSORS_FULL
    ))

    add_scenario!(runner, (
        name = "no_compass_turn",
        trajectory = TRAJ_LAWNMOWER,
        field_strength = FIELD_NOMINAL,
        sensors = SENSORS_NO_COMPASS
    ))

    return runner
end

export create_default_qualification

# ============================================================================
# Quick Qualification Check
# ============================================================================

"""
    quick_qualification_check(obs_results, perf_results, config) -> (Bool, String)

Quick pass/fail check without full report generation.

Returns (passed, reason).
"""
function quick_qualification_check(
    obs_results::Vector{ObservabilityQualResult},
    perf_results::Vector{PerformanceQualResult},
    config::QualificationConfig = DEFAULT_QUALIFICATION_CONFIG
)
    # Check observability
    for r in obs_results
        if r.silent_divergence_detected
            return (false, "Silent divergence in $(r.scenario_name)")
        end
        if r.max_undetected_time > config.max_silent_divergence_time
            return (false, "Undetected time exceeded in $(r.scenario_name)")
        end
    end

    # Check performance
    for r in perf_results
        if r.consistency < config.min_consistency
            return (false, "Consistency $(r.consistency) below threshold in $(r.scenario_name)")
        end
        if r.rmse_position > config.max_rmse_position
            return (false, "RMSE $(r.rmse_position) above threshold in $(r.scenario_name)")
        end
    end

    # Check NEES calibration
    nees_means = [r.nees_mean for r in perf_results if !isinf(r.nees_mean)]
    if !isempty(nees_means)
        overall_nees = mean(nees_means)
        if abs(overall_nees - config.nees_target) > config.nees_tolerance
            return (false, "NEES mean $(overall_nees) outside target range")
        end
    end

    return (true, "All qualification checks passed")
end

export quick_qualification_check

# ============================================================================
# Integration with Previous Steps
# ============================================================================

"""
    validate_nees_calibration(nees_history, state_dim; min_samples=100) -> (Bool, Float64, Float64)

Validate NEES calibration from Step 5-7.

Returns (passed, mean_nees, consistency).
"""
function validate_nees_calibration(nees_history::Vector{Float64}, state_dim::Int;
                                    min_samples::Int = 100)
    if length(nees_history) < min_samples
        return (false, 0.0, 0.0)
    end

    mean_nees = mean(nees_history)

    # Chi-squared bounds for state_dim DOF
    lower = max(0, state_dim - 2 * sqrt(2 * state_dim))
    upper = state_dim + 2 * sqrt(2 * state_dim)

    in_bounds = count(n -> lower <= n <= upper, nees_history)
    consistency = in_bounds / length(nees_history)

    # NEES should be approximately state_dim
    nees_ok = abs(mean_nees - state_dim) < 0.5 * state_dim
    consistency_ok = consistency >= 0.85

    return (nees_ok && consistency_ok, mean_nees, consistency)
end

"""
    validate_innovation_covariance(innovations, S_matrices; threshold=0.1) -> Bool

Validate innovation covariance formula S = HPH' + R + Q_model from Step 5.

Checks that normalized innovations are approximately unit normal.
"""
function validate_innovation_covariance(
    innovations::Vector{<:AbstractVector},
    S_matrices::Vector{<:AbstractMatrix};
    threshold::Float64 = 0.1
)
    if isempty(innovations)
        return false
    end

    # Compute normalized innovations
    nis_values = Float64[]
    for (innov, S) in zip(innovations, S_matrices)
        try
            nis = innov' * inv(S) * innov
            push!(nis_values, nis)
        catch
            # Skip if S is singular
        end
    end

    if isempty(nis_values)
        return false
    end

    # NIS should be chi-squared with measurement dimension DOF
    # Mean should be approximately measurement dimension
    mean_nis = mean(nis_values)
    expected_nis = length(innovations[1])  # Measurement dimension

    return abs(mean_nis - expected_nis) / expected_nis < threshold
end

"""
    validate_process_noise(Q_history, innovations, dt; threshold=0.2) -> Bool

Validate process noise calibration from Step 7.

Checks that process noise is consistent with observed innovation growth.
"""
function validate_process_noise(
    Q_history::Vector{<:AbstractMatrix},
    innovation_norms::Vector{Float64},
    dt::Float64;
    threshold::Float64 = 0.2
)
    if length(innovation_norms) < 10
        return true  # Not enough data to validate
    end

    # Expected innovation growth from process noise
    avg_Q_trace = mean([tr(Q) for Q in Q_history])
    expected_growth_var = avg_Q_trace * dt

    # Observed innovation variance
    observed_var = var(innovation_norms)

    # They should be in the same ballpark
    ratio = observed_var / max(expected_growth_var, 1e-12)
    return 0.5 < ratio < 2.0
end

export validate_nees_calibration, validate_innovation_covariance, validate_process_noise
