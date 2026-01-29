# ============================================================================
# seed_grid.jl - Seed-Grid Execution Harness
# ============================================================================
#
# V1.0 Qualification Closure Plan Step 3:
# Run qualification scenarios across multiple random seeds for statistical
# validity. Monte Carlo execution provides confidence in results.
#
# Execution Modes:
# - QUICK: 10 seeds per scenario (fast feedback, CI-friendly)
# - FULL: 100 seeds per scenario (production qualification)
# - CUSTOM: User-specified seed count
#
# Features:
# - Deterministic seeding for reproducibility
# - Progress tracking and ETA estimation
# - Result aggregation with statistics
# - Failure analysis across seeds
# ============================================================================

using Random
using Statistics

export SeedGridMode, MODE_QUICK, MODE_FULL, MODE_CUSTOM
export SeedGridConfig, QUICK_SEED_CONFIG, FULL_SEED_CONFIG
export SeedRunResult, SeedGridResult, ScenarioSeedResults
export run_seed_grid, aggregate_seed_results
export SeedGridRunner, run!, get_progress, get_results
export format_seed_grid_summary, analyze_seed_failures

# ============================================================================
# Seed Grid Mode
# ============================================================================

"""
    SeedGridMode

Execution mode for seed grid.
"""
@enum SeedGridMode begin
    MODE_QUICK = 1    # 10 seeds - CI, quick feedback
    MODE_FULL = 2     # 100 seeds - production qualification
    MODE_CUSTOM = 3   # User-specified
end

# ============================================================================
# Seed Grid Configuration
# ============================================================================

"""
    SeedGridConfig

Configuration for seed-grid execution.

# Fields
- `mode::SeedGridMode`: Execution mode
- `n_seeds::Int`: Number of seeds per scenario
- `base_seed::UInt64`: Base seed for reproducibility
- `parallel::Bool`: Use parallel execution (when available)
- `save_individual::Bool`: Save individual seed results
- `stop_on_failure::Bool`: Stop scenario on first failure
- `verbose::Bool`: Print progress
- `progress_interval::Int`: Print progress every N seeds
"""
Base.@kwdef struct SeedGridConfig
    mode::SeedGridMode = MODE_QUICK
    n_seeds::Int = 10
    base_seed::UInt64 = 0x12345678
    parallel::Bool = false
    save_individual::Bool = true
    stop_on_failure::Bool = false
    verbose::Bool = true
    progress_interval::Int = 10
end

"""Quick qualification: 10 seeds per scenario"""
const QUICK_SEED_CONFIG = SeedGridConfig(
    mode = MODE_QUICK,
    n_seeds = 10,
    verbose = true,
    progress_interval = 5
)

"""Full qualification: 100 seeds per scenario"""
const FULL_SEED_CONFIG = SeedGridConfig(
    mode = MODE_FULL,
    n_seeds = 100,
    verbose = true,
    progress_interval = 25
)

# ============================================================================
# Seed Generation
# ============================================================================

"""
    generate_seeds(base_seed::UInt64, n::Int) -> Vector{UInt64}

Generate N deterministic seeds from a base seed.
Uses a simple hash-based approach for reproducibility.
"""
function generate_seeds(base_seed::UInt64, n::Int)
    seeds = Vector{UInt64}(undef, n)
    rng = Random.MersenneTwister(base_seed)
    for i in 1:n
        seeds[i] = rand(rng, UInt64)
    end
    seeds
end

"""
    generate_scenario_seeds(base_seed::UInt64, scenario_idx::Int, n::Int) -> Vector{UInt64}

Generate seeds for a specific scenario (different from other scenarios).
"""
function generate_scenario_seeds(base_seed::UInt64, scenario_idx::Int, n::Int)
    # Combine base seed with scenario index for unique seed stream
    combined = base_seed ⊻ (UInt64(scenario_idx) * 0x9E3779B97F4A7C15)
    generate_seeds(combined, n)
end

# ============================================================================
# Individual Seed Result
# ============================================================================

"""
    SeedRunResult

Result of running a single scenario with one seed.
"""
struct SeedRunResult
    seed::UInt64
    seed_idx::Int
    passed::Bool

    # Observability metrics
    silent_divergence::Bool
    max_undetected_time::Float64
    health_response_rate::Float64

    # Performance metrics
    nees_mean::Float64
    nees_consistency::Float64
    rmse_position::Float64
    rmse_velocity::Float64

    # Execution
    execution_time_s::Float64
    error_message::String
end

function SeedRunResult(seed::UInt64, seed_idx::Int, obs::ObservabilityQualResult,
                       perf::PerformanceQualResult, exec_time::Float64)
    SeedRunResult(
        seed, seed_idx,
        obs.passed && perf.passed,
        obs.silent_divergence_detected,
        obs.max_undetected_time,
        obs.health_response_rate,
        perf.nees_mean,
        perf.consistency,
        perf.rmse_position,
        perf.rmse_velocity,
        exec_time,
        ""
    )
end

function SeedRunResult(seed::UInt64, seed_idx::Int, error_msg::String, exec_time::Float64)
    SeedRunResult(
        seed, seed_idx,
        false,
        false, Inf, 0.0,
        Inf, 0.0, Inf, Inf,
        exec_time,
        error_msg
    )
end

# ============================================================================
# Scenario Seed Results (aggregated)
# ============================================================================

"""
    ScenarioSeedResults

Aggregated results for one scenario across all seeds.
"""
struct ScenarioSeedResults
    scenario_name::String
    n_seeds::Int
    n_passed::Int
    n_failed::Int
    pass_rate::Float64

    # Observability statistics
    silent_divergence_count::Int
    max_undetected_time_stats::NamedTuple{(:min, :max, :mean, :std), NTuple{4, Float64}}
    health_response_stats::NamedTuple{(:min, :max, :mean, :std), NTuple{4, Float64}}

    # Performance statistics
    nees_mean_stats::NamedTuple{(:min, :max, :mean, :std), NTuple{4, Float64}}
    consistency_stats::NamedTuple{(:min, :max, :mean, :std), NTuple{4, Float64}}
    rmse_position_stats::NamedTuple{(:min, :max, :mean, :std), NTuple{4, Float64}}
    rmse_velocity_stats::NamedTuple{(:min, :max, :mean, :std), NTuple{4, Float64}}

    # Execution
    total_time_s::Float64
    mean_time_per_seed_s::Float64

    # Individual results (if saved)
    individual_results::Vector{SeedRunResult}

    # Failure analysis
    failure_seeds::Vector{UInt64}
    failure_reasons::Vector{String}
end

"""
    compute_stats(values::Vector{Float64}) -> NamedTuple

Compute min, max, mean, std for a vector of values.
"""
function compute_stats(values::Vector{Float64})
    valid = filter(x -> !isinf(x) && !isnan(x), values)
    if isempty(valid)
        return (min=Inf, max=-Inf, mean=NaN, std=NaN)
    end
    (
        min = minimum(valid),
        max = maximum(valid),
        mean = mean(valid),
        std = length(valid) > 1 ? std(valid) : 0.0
    )
end

"""
    aggregate_seed_results(scenario_name::String, results::Vector{SeedRunResult}) -> ScenarioSeedResults

Aggregate individual seed results into statistics.
"""
function aggregate_seed_results(scenario_name::String, results::Vector{SeedRunResult})
    n_seeds = length(results)
    n_passed = count(r -> r.passed, results)
    n_failed = n_seeds - n_passed
    pass_rate = n_seeds > 0 ? n_passed / n_seeds : 0.0

    # Extract metric vectors
    undetected_times = [r.max_undetected_time for r in results]
    health_rates = [r.health_response_rate for r in results]
    nees_means = [r.nees_mean for r in results]
    consistencies = [r.nees_consistency for r in results]
    rmse_pos = [r.rmse_position for r in results]
    rmse_vel = [r.rmse_velocity for r in results]
    exec_times = [r.execution_time_s for r in results]

    # Failure analysis
    failure_seeds = [r.seed for r in results if !r.passed]
    failure_reasons = unique([r.error_message for r in results if !r.passed && !isempty(r.error_message)])

    ScenarioSeedResults(
        scenario_name,
        n_seeds,
        n_passed,
        n_failed,
        pass_rate,
        count(r -> r.silent_divergence, results),
        compute_stats(undetected_times),
        compute_stats(health_rates),
        compute_stats(nees_means),
        compute_stats(consistencies),
        compute_stats(rmse_pos),
        compute_stats(rmse_vel),
        sum(exec_times),
        n_seeds > 0 ? sum(exec_times) / n_seeds : 0.0,
        results,
        failure_seeds,
        failure_reasons
    )
end

# ============================================================================
# Full Seed Grid Result
# ============================================================================

"""
    SeedGridResult

Complete result of seed-grid qualification.
"""
struct SeedGridResult
    config::SeedGridConfig
    scenarios::Vector{ScenarioSeedResults}

    # Overall statistics
    total_scenarios::Int
    total_seeds::Int
    total_passed::Int
    total_failed::Int
    overall_pass_rate::Float64

    # Worst performers
    worst_scenario::String
    worst_pass_rate::Float64

    # Execution
    total_time_s::Float64
    start_time::Float64
    end_time::Float64

    # Overall status
    status::QualificationStatus
end

function SeedGridResult(config::SeedGridConfig, scenarios::Vector{ScenarioSeedResults},
                        start_time::Float64, end_time::Float64)
    total_scenarios = length(scenarios)

    # Handle empty scenarios
    if isempty(scenarios)
        return SeedGridResult(
            config, scenarios,
            0, 0, 0, 0, 1.0,
            "", 1.0,
            end_time - start_time, start_time, end_time,
            QUAL_PASS  # No failures = pass
        )
    end

    total_seeds = sum(s.n_seeds for s in scenarios)
    total_passed = sum(s.n_passed for s in scenarios)
    total_failed = sum(s.n_failed for s in scenarios)
    overall_pass_rate = total_seeds > 0 ? total_passed / total_seeds : 1.0

    # Find worst performer
    worst_idx = argmin([s.pass_rate for s in scenarios])
    worst_scenario = scenarios[worst_idx].scenario_name
    worst_pass_rate = scenarios[worst_idx].pass_rate

    # Determine status
    # All scenarios must have ≥95% pass rate for PASS
    # ≥85% for CONDITIONAL
    status = if all(s.pass_rate >= 0.95 for s in scenarios)
        QUAL_PASS
    elseif all(s.pass_rate >= 0.85 for s in scenarios)
        QUAL_CONDITIONAL
    else
        QUAL_FAIL
    end

    SeedGridResult(
        config, scenarios,
        total_scenarios, total_seeds, total_passed, total_failed, overall_pass_rate,
        worst_scenario, worst_pass_rate,
        end_time - start_time, start_time, end_time,
        status
    )
end

# ============================================================================
# Seed Grid Runner
# ============================================================================

"""
    SeedGridRunner

Stateful runner for seed-grid execution with progress tracking.
"""
mutable struct SeedGridRunner
    config::SeedGridConfig
    scenarios::Vector{ScenarioDefinition}

    # Progress tracking
    current_scenario_idx::Int
    current_seed_idx::Int
    total_seeds_completed::Int
    total_seeds::Int

    # Results
    scenario_results::Vector{ScenarioSeedResults}
    current_seed_results::Vector{SeedRunResult}

    # Timing
    start_time::Float64
    scenario_start_time::Float64

    # Status
    running::Bool
    completed::Bool
end

function SeedGridRunner(config::SeedGridConfig, scenarios::Vector{ScenarioDefinition})
    SeedGridRunner(
        config, scenarios,
        0, 0, 0, length(scenarios) * config.n_seeds,
        ScenarioSeedResults[],
        SeedRunResult[],
        0.0, 0.0,
        false, false
    )
end

"""
    get_progress(runner::SeedGridRunner) -> NamedTuple

Get current progress information.
"""
function get_progress(runner::SeedGridRunner)
    elapsed = runner.start_time > 0 ? time() - runner.start_time : 0.0
    rate = runner.total_seeds_completed > 0 ? elapsed / runner.total_seeds_completed : 0.0
    remaining = runner.total_seeds - runner.total_seeds_completed
    eta = rate * remaining

    (
        scenario_idx = runner.current_scenario_idx,
        seed_idx = runner.current_seed_idx,
        total_completed = runner.total_seeds_completed,
        total_seeds = runner.total_seeds,
        percent = 100.0 * runner.total_seeds_completed / max(runner.total_seeds, 1),
        elapsed_s = elapsed,
        eta_s = eta,
        running = runner.running,
        completed = runner.completed
    )
end

"""
    run!(runner::SeedGridRunner; simulator=nothing) -> SeedGridResult

Execute the seed grid.
"""
function run!(runner::SeedGridRunner; simulator::Union{Nothing, Function} = nothing)
    runner.running = true
    runner.start_time = time()
    runner.scenario_results = ScenarioSeedResults[]

    config = runner.config

    for (scenario_idx, scenario) in enumerate(runner.scenarios)
        runner.current_scenario_idx = scenario_idx
        runner.scenario_start_time = time()
        runner.current_seed_results = SeedRunResult[]

        # Generate seeds for this scenario
        seeds = generate_scenario_seeds(config.base_seed, scenario_idx, config.n_seeds)

        if config.verbose
            println("Scenario $(scenario_idx)/$(length(runner.scenarios)): $(scenario.metadata.name)")
        end

        for (seed_idx, seed) in enumerate(seeds)
            runner.current_seed_idx = seed_idx

            seed_start = time()

            try
                # Set RNG seed for reproducibility
                Random.seed!(seed)

                # Run scenario with this seed
                result = run_scenario_qualification(scenario;
                    n_seeds=1, verbose=false, simulator=simulator)

                # Create seed result
                if result.observability_result !== nothing && result.performance_result !== nothing
                    seed_result = SeedRunResult(
                        seed, seed_idx,
                        result.observability_result,
                        result.performance_result,
                        time() - seed_start
                    )
                else
                    seed_result = SeedRunResult(seed, seed_idx, "No results", time() - seed_start)
                end

                push!(runner.current_seed_results, seed_result)

            catch e
                # Record error
                seed_result = SeedRunResult(seed, seed_idx, string(e), time() - seed_start)
                push!(runner.current_seed_results, seed_result)

                if config.stop_on_failure
                    break
                end
            end

            runner.total_seeds_completed += 1

            # Progress update
            if config.verbose && seed_idx % config.progress_interval == 0
                progress = get_progress(runner)
                println("  Seed $(seed_idx)/$(config.n_seeds) ($(round(progress.percent, digits=1))% total, ETA: $(round(progress.eta_s, digits=1))s)")
            end
        end

        # Aggregate scenario results
        scenario_agg = aggregate_seed_results(scenario.metadata.name, runner.current_seed_results)
        push!(runner.scenario_results, scenario_agg)

        if config.verbose
            println("  Complete: $(scenario_agg.n_passed)/$(scenario_agg.n_seeds) passed ($(round(scenario_agg.pass_rate * 100, digits=1))%)")
        end
    end

    runner.running = false
    runner.completed = true

    SeedGridResult(config, runner.scenario_results, runner.start_time, time())
end

"""
    get_results(runner::SeedGridRunner) -> Union{Nothing, SeedGridResult}

Get final results if completed.
"""
function get_results(runner::SeedGridRunner)
    if !runner.completed
        return nothing
    end
    SeedGridResult(runner.config, runner.scenario_results, runner.start_time, time())
end

# ============================================================================
# Convenience Function
# ============================================================================

"""
    run_seed_grid(scenarios::Vector{ScenarioDefinition};
                  config::SeedGridConfig=QUICK_SEED_CONFIG,
                  simulator=nothing) -> SeedGridResult

Run seed-grid qualification on scenarios.

# Example
```julia
scenarios = load_all_scenarios("scenarios/qualification/v1_0")
result = run_seed_grid(scenarios; config=QUICK_SEED_CONFIG)
println(format_seed_grid_summary(result))
```
"""
function run_seed_grid(scenarios::Vector{ScenarioDefinition};
                       config::SeedGridConfig = QUICK_SEED_CONFIG,
                       simulator::Union{Nothing, Function} = nothing)
    runner = SeedGridRunner(config, scenarios)
    run!(runner; simulator=simulator)
end

# ============================================================================
# Reporting
# ============================================================================

"""
    format_seed_grid_summary(result::SeedGridResult) -> String

Generate a formatted summary of seed-grid results.
"""
function format_seed_grid_summary(result::SeedGridResult)
    lines = String[]

    push!(lines, "=" ^ 70)
    push!(lines, "SEED-GRID QUALIFICATION SUMMARY")
    push!(lines, "=" ^ 70)
    push!(lines, "")

    # Mode and configuration
    mode_str = result.config.mode == MODE_QUICK ? "QUICK (10 seeds)" :
               result.config.mode == MODE_FULL ? "FULL (100 seeds)" :
               "CUSTOM ($(result.config.n_seeds) seeds)"
    push!(lines, "Mode: $mode_str")
    push!(lines, "Base seed: 0x$(string(result.config.base_seed, base=16))")
    push!(lines, "")

    # Overall status
    status_str = if result.status == QUAL_PASS
        "✓ QUALIFIED"
    elseif result.status == QUAL_CONDITIONAL
        "◐ CONDITIONAL"
    else
        "✗ NOT QUALIFIED"
    end
    push!(lines, "OVERALL STATUS: $status_str")
    push!(lines, "")

    # Summary statistics
    push!(lines, "-" ^ 70)
    push!(lines, "SUMMARY")
    push!(lines, "-" ^ 70)
    push!(lines, "Total scenarios: $(result.total_scenarios)")
    push!(lines, "Total seeds: $(result.total_seeds)")
    push!(lines, "Passed: $(result.total_passed) ($(round(result.overall_pass_rate * 100, digits=1))%)")
    push!(lines, "Failed: $(result.total_failed)")
    push!(lines, "Execution time: $(round(result.total_time_s, digits=1))s")
    push!(lines, "")

    # Per-scenario results
    push!(lines, "-" ^ 70)
    push!(lines, "SCENARIO RESULTS")
    push!(lines, "-" ^ 70)

    for s in result.scenarios
        status = s.pass_rate >= 0.95 ? "✓" : s.pass_rate >= 0.85 ? "◐" : "✗"
        push!(lines, "$status $(s.scenario_name): $(s.n_passed)/$(s.n_seeds) ($(round(s.pass_rate * 100, digits=1))%)")
        push!(lines, "    RMSE position: $(round(s.rmse_position_stats.mean, digits=2))m ± $(round(s.rmse_position_stats.std, digits=2))m")
        push!(lines, "    NEES mean: $(round(s.nees_mean_stats.mean, digits=2)) ± $(round(s.nees_mean_stats.std, digits=2))")
        push!(lines, "    Consistency: $(round(s.consistency_stats.mean * 100, digits=1))% ± $(round(s.consistency_stats.std * 100, digits=1))%")
        if s.silent_divergence_count > 0
            push!(lines, "    ⚠ Silent divergence: $(s.silent_divergence_count) seeds")
        end
    end
    push!(lines, "")

    # Worst performer
    if result.worst_pass_rate < 1.0
        push!(lines, "-" ^ 70)
        push!(lines, "ATTENTION")
        push!(lines, "-" ^ 70)
        push!(lines, "Worst scenario: $(result.worst_scenario)")
        push!(lines, "Pass rate: $(round(result.worst_pass_rate * 100, digits=1))%")
    end
    push!(lines, "")

    push!(lines, "=" ^ 70)

    join(lines, "\n")
end

"""
    analyze_seed_failures(result::SeedGridResult) -> String

Analyze patterns in seed failures.
"""
function analyze_seed_failures(result::SeedGridResult)
    lines = String[]

    push!(lines, "=" ^ 70)
    push!(lines, "FAILURE ANALYSIS")
    push!(lines, "=" ^ 70)
    push!(lines, "")

    total_failures = sum(s.n_failed for s in result.scenarios)

    if total_failures == 0
        push!(lines, "No failures to analyze.")
        return join(lines, "\n")
    end

    push!(lines, "Total failures: $total_failures")
    push!(lines, "")

    for s in result.scenarios
        if s.n_failed > 0
            push!(lines, "-" ^ 70)
            push!(lines, "$(s.scenario_name): $(s.n_failed) failures")
            push!(lines, "-" ^ 70)

            # Show failure seeds (first 5)
            n_show = min(5, length(s.failure_seeds))
            if n_show > 0
                push!(lines, "Failing seeds (first $n_show):")
                for seed in s.failure_seeds[1:n_show]
                    push!(lines, "  - 0x$(string(seed, base=16))")
                end
            end

            # Show failure reasons
            if !isempty(s.failure_reasons)
                push!(lines, "Failure reasons:")
                for reason in s.failure_reasons
                    push!(lines, "  - $reason")
                end
            end

            # Check for patterns
            if s.silent_divergence_count > 0
                div_rate = s.silent_divergence_count / s.n_seeds
                push!(lines, "Silent divergence rate: $(round(div_rate * 100, digits=1))%")
            end

            push!(lines, "")
        end
    end

    push!(lines, "=" ^ 70)

    join(lines, "\n")
end
