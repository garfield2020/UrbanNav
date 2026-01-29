# ============================================================================
# doe.jl - Design of Experiments for Navigation Testing
# ============================================================================
#
# Step 9 of NEES Recovery Plan:
# Add "observability DOE" separate from "performance DOE"
#
# Key insight: Observability and performance are different concerns:
# - Observability DOE: Can we estimate the state at all? (rank, conditioning)
# - Performance DOE: How accurately do we estimate it? (RMSE, NEES)
#
# Separating these allows:
# 1. Clear pass/fail criteria for each
# 2. Identification of observability boundaries
# 3. Performance testing only in observable regions
# 4. Root cause analysis when things go wrong
# ============================================================================

export DOEFactor, DOELevel, DOEDesign, DOERun, DOEResult
export TrajectoryType, TRAJ_STRAIGHT, TRAJ_LAWNMOWER, TRAJ_SPIRAL, TRAJ_HOVER
export FieldStrength, FIELD_WEAK, FIELD_NOMINAL, FIELD_STRONG
export GradientStrength, GRAD_WEAK, GRAD_NOMINAL, GRAD_STRONG
export SensorConfig, SENSORS_FULL, SENSORS_NO_ODOMETRY, SENSORS_NO_COMPASS, SENSORS_MINIMAL

export ObservabilityDOE, PerformanceDOE
export ObservabilityMetrics, PerformanceMetrics
export create_full_factorial, create_fractional_factorial
export run_doe!, analyze_observability, analyze_performance
export DOEAnalysis, find_observability_boundary, compute_main_effects

using LinearAlgebra
using Statistics

# ============================================================================
# DOE Factor Definitions
# ============================================================================

"""
    TrajectoryType

Types of vehicle trajectories for DOE.
"""
@enum TrajectoryType begin
    TRAJ_STRAIGHT = 1    # Straight line (yaw unobservable)
    TRAJ_LAWNMOWER = 2   # Lawnmower pattern (Y-axis stress during turns)
    TRAJ_SPIRAL = 3      # Spiral (good observability)
    TRAJ_HOVER = 4       # Stationary hover (position unobservable from velocity)
    TRAJ_CIRCLE = 5      # Circular path
    TRAJ_FIGURE_EIGHT = 6 # Figure-8 pattern
end

"""
    FieldStrength

Magnetic field strength levels.
"""
@enum FieldStrength begin
    FIELD_WEAK = 1       # < 10 nT anomaly
    FIELD_NOMINAL = 2    # 10-100 nT anomaly
    FIELD_STRONG = 3     # > 100 nT anomaly
end

"""
    GradientStrength

Magnetic gradient strength levels.
"""
@enum GradientStrength begin
    GRAD_WEAK = 1        # < 1 nT/m
    GRAD_NOMINAL = 2     # 1-10 nT/m
    GRAD_STRONG = 3      # > 10 nT/m
end

"""
    SensorConfig

Available sensor configurations.
"""
@enum SensorConfig begin
    SENSORS_FULL = 1         # All sensors available
    SENSORS_NO_ODOMETRY = 2       # Odometry dropout
    SENSORS_NO_COMPASS = 3   # Compass unavailable
    SENSORS_NO_DEPTH = 4     # Depth sensor unavailable
    SENSORS_MINIMAL = 5      # Only IMU + magnetometer
end

# ============================================================================
# DOE Factor and Level Types
# ============================================================================

"""
    DOELevel

A single level for a DOE factor.

# Fields
- `name::String`: Level name
- `value::Any`: Level value (enum, number, etc.)
- `description::String`: Human-readable description
"""
struct DOELevel
    name::String
    value::Any
    description::String
end

DOELevel(name::String, value::Any) = DOELevel(name, value, "")

"""
    DOEFactor

A factor in the DOE with multiple levels.

# Fields
- `name::Symbol`: Factor name
- `levels::Vector{DOELevel}`: Available levels
- `description::String`: Factor description
"""
struct DOEFactor
    name::Symbol
    levels::Vector{DOELevel}
    description::String
end

DOEFactor(name::Symbol, levels::Vector{DOELevel}) = DOEFactor(name, levels, "")

# ============================================================================
# Standard Factors
# ============================================================================

"""
    trajectory_factor() -> DOEFactor

Standard trajectory factor with common patterns.
"""
function trajectory_factor()
    DOEFactor(:trajectory, [
        DOELevel("straight", TRAJ_STRAIGHT, "Straight line motion"),
        DOELevel("lawnmower", TRAJ_LAWNMOWER, "Lawnmower survey pattern"),
        DOELevel("spiral", TRAJ_SPIRAL, "Spiral/helix path"),
        DOELevel("hover", TRAJ_HOVER, "Stationary hover"),
    ], "Vehicle trajectory type")
end

"""
    field_strength_factor() -> DOEFactor

Standard field strength factor.
"""
function field_strength_factor()
    DOEFactor(:field_strength, [
        DOELevel("weak", FIELD_WEAK, "Weak magnetic anomaly"),
        DOELevel("nominal", FIELD_NOMINAL, "Typical magnetic anomaly"),
        DOELevel("strong", FIELD_STRONG, "Strong magnetic anomaly"),
    ], "Magnetic field anomaly strength")
end

"""
    gradient_strength_factor() -> DOEFactor

Standard gradient strength factor.
"""
function gradient_strength_factor()
    DOEFactor(:gradient_strength, [
        DOELevel("weak", GRAD_WEAK, "Weak gradients"),
        DOELevel("nominal", GRAD_NOMINAL, "Typical gradients"),
        DOELevel("strong", GRAD_STRONG, "Strong gradients"),
    ], "Magnetic gradient strength")
end

"""
    sensor_config_factor() -> DOEFactor

Standard sensor configuration factor.
"""
function sensor_config_factor()
    DOEFactor(:sensors, [
        DOELevel("full", SENSORS_FULL, "All sensors available"),
        DOELevel("no_odometry", SENSORS_NO_ODOMETRY, "Odometry dropout"),
        DOELevel("no_compass", SENSORS_NO_COMPASS, "Compass unavailable"),
        DOELevel("minimal", SENSORS_MINIMAL, "Only IMU + magnetometer"),
    ], "Sensor availability")
end

export trajectory_factor, field_strength_factor, gradient_strength_factor, sensor_config_factor

# ============================================================================
# DOE Design
# ============================================================================

"""
    DOEDesign

A complete DOE design specifying factors and runs.

# Fields
- `name::String`: Design name
- `factors::Vector{DOEFactor}`: Factors in the design
- `runs::Vector{Dict{Symbol, DOELevel}}`: Combinations to run
- `design_type::Symbol`: :full_factorial, :fractional, :custom
"""
struct DOEDesign
    name::String
    factors::Vector{DOEFactor}
    runs::Vector{Dict{Symbol, DOELevel}}
    design_type::Symbol
end

"""
    create_full_factorial(name, factors) -> DOEDesign

Create a full factorial design (all combinations).
"""
function create_full_factorial(name::String, factors::Vector{DOEFactor})
    runs = Dict{Symbol, DOELevel}[]

    # Generate all combinations
    level_counts = [length(f.levels) for f in factors]
    n_runs = prod(level_counts)

    for run_idx in 0:(n_runs-1)
        run = Dict{Symbol, DOELevel}()
        remaining = run_idx

        for (i, factor) in enumerate(factors)
            n_levels = length(factor.levels)
            level_idx = (remaining % n_levels) + 1
            remaining = remaining ÷ n_levels
            run[factor.name] = factor.levels[level_idx]
        end

        push!(runs, run)
    end

    DOEDesign(name, factors, runs, :full_factorial)
end

"""
    create_fractional_factorial(name, factors; fraction=0.5) -> DOEDesign

Create a fractional factorial design (subset of combinations).

Uses Latin hypercube-style selection to cover the design space.
"""
function create_fractional_factorial(name::String, factors::Vector{DOEFactor};
                                     fraction::Float64 = 0.5)
    full = create_full_factorial(name, factors)
    n_select = max(1, round(Int, length(full.runs) * fraction))

    # Select runs to maximize coverage
    # Simple approach: evenly spaced selection
    step = length(full.runs) / n_select
    indices = [round(Int, 1 + i * step) for i in 0:(n_select-1)]
    indices = clamp.(indices, 1, length(full.runs))

    selected_runs = [full.runs[i] for i in unique(indices)]

    DOEDesign(name, factors, selected_runs, :fractional)
end

# ============================================================================
# DOE Run and Results
# ============================================================================

"""
    DOERun

A single run in the DOE.

# Fields
- `run_id::Int`: Unique run identifier
- `levels::Dict{Symbol, DOELevel}`: Factor levels for this run
- `status::Symbol`: :pending, :running, :completed, :failed
- `start_time::Float64`: When run started
- `end_time::Float64`: When run ended
"""
mutable struct DOERun
    run_id::Int
    levels::Dict{Symbol, DOELevel}
    status::Symbol
    start_time::Float64
    end_time::Float64
end

DOERun(run_id::Int, levels::Dict{Symbol, DOELevel}) =
    DOERun(run_id, levels, :pending, 0.0, 0.0)

"""
    DOEResult

Result of a DOE run.

# Fields
- `run::DOERun`: The run that produced this result
- `observability::ObservabilityMetrics`: Observability analysis
- `performance::PerformanceMetrics`: Performance metrics
- `passed::Bool`: Whether the run passed acceptance criteria
- `failure_reason::String`: If failed, why
"""
struct DOEResult
    run::DOERun
    observability::Any  # ObservabilityMetrics
    performance::Any    # PerformanceMetrics
    passed::Bool
    failure_reason::String
end

# ============================================================================
# Observability Metrics
# ============================================================================

"""
    ObservabilityMetrics

Metrics for observability DOE.

# Fields
- `fisher_info::Matrix{Float64}`: Fisher information matrix
- `condition_number::Float64`: Condition number of Fisher info
- `rank::Int`: Effective rank
- `singular_values::Vector{Float64}`: Singular values
- `observable_dims::Vector{Bool}`: Which dimensions are observable
- `weakest_direction::Vector{Float64}`: Least observable direction
- `observability_gramian::Matrix{Float64}`: Observability Gramian
"""
struct ObservabilityMetrics
    fisher_info::Matrix{Float64}
    condition_number::Float64
    rank::Int
    singular_values::Vector{Float64}
    observable_dims::Vector{Bool}
    weakest_direction::Vector{Float64}
    observability_gramian::Matrix{Float64}
end

"""
    compute_observability_metrics(H, R, n_states) -> ObservabilityMetrics

Compute observability metrics from measurement Jacobian.

# Arguments
- `H`: Measurement Jacobian (m × n)
- `R`: Measurement noise covariance (m × m)
- `n_states`: Number of state dimensions
"""
function compute_observability_metrics(H::AbstractMatrix, R::AbstractMatrix, n_states::Int)
    m, n = size(H)

    if m == 0 || n == 0
        return ObservabilityMetrics(
            zeros(n_states, n_states),
            Inf, 0, Float64[],
            fill(false, n_states),
            zeros(n_states),
            zeros(n_states, n_states)
        )
    end

    # Fisher information: I = H' R^-1 H
    try
        R_inv = inv(R)
        fisher = H' * R_inv * H

        # Extend to full state dimension if needed
        if n < n_states
            fisher_full = zeros(n_states, n_states)
            fisher_full[1:n, 1:n] = fisher
            fisher = fisher_full
        end

        # SVD for rank and condition number
        F = svd(fisher)
        singular_values = F.S

        # Effective rank (singular values above threshold)
        threshold = 1e-10 * maximum(singular_values)
        eff_rank = count(s -> s > threshold, singular_values)

        # Condition number (use smallest singular value, including zeros)
        # If rank-deficient, condition number is Inf
        cond_num = if eff_rank < n_states || minimum(singular_values) < threshold
            Inf  # Rank deficient
        elseif eff_rank > 0
            singular_values[1] / max(singular_values[end], 1e-15)
        else
            Inf
        end

        # Observable dimensions (diagonal elements above threshold)
        observable_dims = [fisher[i,i] > threshold for i in 1:n_states]

        # Weakest direction (smallest singular vector)
        weakest = if length(F.V) > 0
            F.V[:, end]
        else
            zeros(n_states)
        end

        # Observability Gramian (accumulated over measurements)
        gramian = fisher  # Simplified: use Fisher info as proxy

        return ObservabilityMetrics(
            fisher, cond_num, eff_rank, singular_values,
            observable_dims, weakest, gramian
        )
    catch e
        return ObservabilityMetrics(
            zeros(n_states, n_states),
            Inf, 0, Float64[],
            fill(false, n_states),
            zeros(n_states),
            zeros(n_states, n_states)
        )
    end
end

"""
    is_observable(metrics::ObservabilityMetrics; min_rank=3, max_cond=1e6) -> Bool

Check if system is observable based on metrics.
"""
function is_observable(metrics::ObservabilityMetrics;
                       min_rank::Int = 3,
                       max_cond::Float64 = 1e6)
    return metrics.rank >= min_rank && metrics.condition_number < max_cond
end

export ObservabilityMetrics, compute_observability_metrics, is_observable

# ============================================================================
# Performance Metrics
# ============================================================================

"""
    PerformanceMetrics

Metrics for performance DOE.

# Fields
- `rmse_position::Float64`: Position RMSE (m)
- `rmse_velocity::Float64`: Velocity RMSE (m/s)
- `rmse_attitude::Float64`: Attitude RMSE (rad)
- `nees_mean::Float64`: Mean NEES (should be ≈ state_dim)
- `nees_std::Float64`: NEES standard deviation
- `nis_mean::Float64`: Mean NIS (should be ≈ meas_dim)
- `max_error::Float64`: Maximum position error (m)
- `convergence_time::Float64`: Time to converge (s)
- `consistency::Float64`: Fraction of samples with NEES in bounds
"""
struct PerformanceMetrics
    rmse_position::Float64
    rmse_velocity::Float64
    rmse_attitude::Float64
    nees_mean::Float64
    nees_std::Float64
    nis_mean::Float64
    max_error::Float64
    convergence_time::Float64
    consistency::Float64
end

"""
    compute_performance_metrics(errors, covariances, state_dim) -> PerformanceMetrics

Compute performance metrics from estimation errors.

# Arguments
- `errors`: Vector of error vectors [pos, vel, att, ...]
- `covariances`: Vector of covariance matrices
- `state_dim`: State dimension for NEES bounds
"""
function compute_performance_metrics(
    errors::Vector{<:AbstractVector},
    covariances::Vector{<:AbstractMatrix},
    state_dim::Int
)
    n = length(errors)
    if n == 0
        return PerformanceMetrics(
            Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, 0.0
        )
    end

    # Position RMSE (first 3 components)
    pos_errors = [norm(e[1:min(3, length(e))]) for e in errors]
    rmse_pos = sqrt(mean(pos_errors .^ 2))

    # Velocity RMSE (components 4-6)
    vel_errors = [length(e) >= 6 ? norm(e[4:6]) : 0.0 for e in errors]
    rmse_vel = sqrt(mean(vel_errors .^ 2))

    # Attitude RMSE (components 7-9)
    att_errors = [length(e) >= 9 ? norm(e[7:9]) : 0.0 for e in errors]
    rmse_att = sqrt(mean(att_errors .^ 2))

    # NEES computation
    nees_values = Float64[]
    for (e, P) in zip(errors, covariances)
        n_e = length(e)
        n_P = size(P, 1)
        n_use = min(n_e, n_P)
        if n_use > 0
            try
                nees = e[1:n_use]' * inv(P[1:n_use, 1:n_use]) * e[1:n_use]
                push!(nees_values, nees)
            catch
                # Skip if inversion fails
            end
        end
    end

    nees_mean = isempty(nees_values) ? Inf : mean(nees_values)
    nees_std = isempty(nees_values) ? Inf : std(nees_values)

    # NIS would require innovations, placeholder for now
    nis_mean = 0.0

    # Max error
    max_error = maximum(pos_errors)

    # Convergence time (when error drops below threshold)
    threshold = rmse_pos * 2
    convergence_idx = findfirst(e -> e < threshold, pos_errors)
    convergence_time = convergence_idx !== nothing ? Float64(convergence_idx) : Inf

    # Consistency (fraction with NEES in chi-squared bounds)
    # For state_dim DOF, 95% bounds are approximately [state_dim - 2*sqrt(2*state_dim), state_dim + 2*sqrt(2*state_dim)]
    lower = max(0, state_dim - 2 * sqrt(2 * state_dim))
    upper = state_dim + 2 * sqrt(2 * state_dim)
    in_bounds = count(n -> lower <= n <= upper, nees_values)
    consistency = isempty(nees_values) ? 0.0 : in_bounds / length(nees_values)

    return PerformanceMetrics(
        rmse_pos, rmse_vel, rmse_att,
        nees_mean, nees_std, nis_mean,
        max_error, convergence_time, consistency
    )
end

"""
    passes_performance(metrics::PerformanceMetrics; max_rmse=5.0, min_consistency=0.85) -> Bool

Check if performance metrics pass acceptance criteria.
"""
function passes_performance(metrics::PerformanceMetrics;
                            max_rmse::Float64 = 5.0,
                            min_consistency::Float64 = 0.85)
    return metrics.rmse_position < max_rmse && metrics.consistency >= min_consistency
end

export PerformanceMetrics, compute_performance_metrics, passes_performance

# ============================================================================
# Observability DOE
# ============================================================================

"""
    ObservabilityDOE

DOE specifically for testing observability conditions.

# Key Questions
1. Under what conditions is each state observable?
2. What are the observability boundaries?
3. Which factor combinations cause unobservability?
"""
struct ObservabilityDOE
    design::DOEDesign
    results::Vector{DOEResult}
    observability_threshold::Float64  # Min condition number for "observable"
    rank_threshold::Int                # Min rank for "observable"
end

function ObservabilityDOE(design::DOEDesign;
                          observability_threshold::Float64 = 1e6,
                          rank_threshold::Int = 3)
    ObservabilityDOE(design, DOEResult[], observability_threshold, rank_threshold)
end

"""
    create_observability_doe(; include_trajectory=true, include_sensors=true) -> ObservabilityDOE

Create a standard observability DOE.
"""
function create_observability_doe(;
    include_trajectory::Bool = true,
    include_sensors::Bool = true,
    include_field::Bool = true,
    include_gradient::Bool = true
)
    factors = DOEFactor[]

    if include_trajectory
        push!(factors, trajectory_factor())
    end
    if include_sensors
        push!(factors, sensor_config_factor())
    end
    if include_field
        push!(factors, field_strength_factor())
    end
    if include_gradient
        push!(factors, gradient_strength_factor())
    end

    design = create_full_factorial("observability_doe", factors)
    ObservabilityDOE(design)
end

export ObservabilityDOE, create_observability_doe

# ============================================================================
# Performance DOE
# ============================================================================

"""
    PerformanceDOE

DOE specifically for testing estimation performance.

Only runs in conditions where observability is confirmed.
"""
struct PerformanceDOE
    design::DOEDesign
    results::Vector{DOEResult}
    rmse_threshold::Float64       # Max acceptable RMSE
    consistency_threshold::Float64 # Min acceptable NEES consistency
    observable_conditions::Vector{Dict{Symbol, DOELevel}}  # Pre-filtered conditions
end

function PerformanceDOE(design::DOEDesign;
                        rmse_threshold::Float64 = 5.0,
                        consistency_threshold::Float64 = 0.85)
    PerformanceDOE(design, DOEResult[], rmse_threshold, consistency_threshold,
                   Dict{Symbol, DOELevel}[])
end

"""
    create_performance_doe(observability_results; ...) -> PerformanceDOE

Create a performance DOE filtered to observable conditions.

# Arguments
- `observability_results`: Results from ObservabilityDOE
"""
function create_performance_doe(observability_results::Vector{DOEResult};
                                 rmse_threshold::Float64 = 5.0,
                                 consistency_threshold::Float64 = 0.85)
    # Filter to only observable conditions
    observable_runs = Dict{Symbol, DOELevel}[]
    for result in observability_results
        if result.observability !== nothing &&
           is_observable(result.observability)
            push!(observable_runs, result.run.levels)
        end
    end

    # Create design from observable runs only
    if isempty(observable_runs)
        # Fallback: use original design but mark as filtered
        factors = DOEFactor[]
        design = DOEDesign("performance_doe_empty", factors, observable_runs, :custom)
    else
        # Extract factors from first run
        factors = DOEFactor[]
        first_run = observable_runs[1]
        for (name, level) in first_run
            push!(factors, DOEFactor(name, [level]))
        end
        design = DOEDesign("performance_doe", factors, observable_runs, :custom)
    end

    PerformanceDOE(design, DOEResult[], rmse_threshold, consistency_threshold, observable_runs)
end

export PerformanceDOE, create_performance_doe

# ============================================================================
# DOE Analysis
# ============================================================================

"""
    DOEAnalysis

Analysis results from a DOE.

# Fields
- `main_effects::Dict{Symbol, Vector{Float64}}`: Effect of each factor level
- `interactions::Dict{Tuple{Symbol,Symbol}, Matrix{Float64}}`: Two-factor interactions
- `optimal_levels::Dict{Symbol, DOELevel}`: Best level for each factor
- `observability_boundary::Vector{Dict{Symbol, DOELevel}}`: Boundary conditions
"""
struct DOEAnalysis
    main_effects::Dict{Symbol, Vector{Float64}}
    interactions::Dict{Tuple{Symbol,Symbol}, Matrix{Float64}}
    optimal_levels::Dict{Symbol, DOELevel}
    observability_boundary::Vector{Dict{Symbol, DOELevel}}
end

"""
    analyze_observability(doe::ObservabilityDOE) -> DOEAnalysis

Analyze observability DOE results.
"""
function analyze_observability(doe::ObservabilityDOE)
    if isempty(doe.results)
        return DOEAnalysis(
            Dict{Symbol, Vector{Float64}}(),
            Dict{Tuple{Symbol,Symbol}, Matrix{Float64}}(),
            Dict{Symbol, DOELevel}(),
            Dict{Symbol, DOELevel}[]
        )
    end

    # Compute main effects (average observability metric by factor level)
    main_effects = Dict{Symbol, Vector{Float64}}()

    for factor in doe.design.factors
        effects = Float64[]
        for level in factor.levels
            # Find runs with this level
            matching = filter(r -> r.run.levels[factor.name] == level, doe.results)
            if !isempty(matching)
                # Use condition number as metric (lower = more observable)
                cond_nums = [r.observability.condition_number for r in matching
                            if r.observability !== nothing]
                if !isempty(cond_nums)
                    push!(effects, mean(cond_nums))
                else
                    push!(effects, Inf)
                end
            else
                push!(effects, Inf)
            end
        end
        main_effects[factor.name] = effects
    end

    # Find optimal levels (lowest condition number)
    optimal_levels = Dict{Symbol, DOELevel}()
    for factor in doe.design.factors
        effects = main_effects[factor.name]
        best_idx = argmin(effects)
        optimal_levels[factor.name] = factor.levels[best_idx]
    end

    # Find observability boundary (conditions where observability changes)
    boundary = Dict{Symbol, DOELevel}[]
    observable_runs = filter(r -> r.observability !== nothing &&
                                  is_observable(r.observability), doe.results)
    unobservable_runs = filter(r -> r.observability !== nothing &&
                                    !is_observable(r.observability), doe.results)

    # Boundary = runs adjacent to both observable and unobservable
    for obs_run in observable_runs
        for unobs_run in unobservable_runs
            # Check if they differ by exactly one factor
            diff_count = 0
            for factor in doe.design.factors
                if obs_run.run.levels[factor.name] != unobs_run.run.levels[factor.name]
                    diff_count += 1
                end
            end
            if diff_count == 1
                push!(boundary, obs_run.run.levels)
                break
            end
        end
    end

    # Placeholder for interactions
    interactions = Dict{Tuple{Symbol,Symbol}, Matrix{Float64}}()

    return DOEAnalysis(main_effects, interactions, optimal_levels, unique(boundary))
end

"""
    analyze_performance(doe::PerformanceDOE) -> DOEAnalysis

Analyze performance DOE results.
"""
function analyze_performance(doe::PerformanceDOE)
    if isempty(doe.results)
        return DOEAnalysis(
            Dict{Symbol, Vector{Float64}}(),
            Dict{Tuple{Symbol,Symbol}, Matrix{Float64}}(),
            Dict{Symbol, DOELevel}(),
            Dict{Symbol, DOELevel}[]
        )
    end

    # Compute main effects using RMSE
    main_effects = Dict{Symbol, Vector{Float64}}()

    for factor in doe.design.factors
        effects = Float64[]
        for level in factor.levels
            matching = filter(r -> r.run.levels[factor.name] == level, doe.results)
            if !isempty(matching)
                rmses = [r.performance.rmse_position for r in matching
                        if r.performance !== nothing]
                if !isempty(rmses)
                    push!(effects, mean(rmses))
                else
                    push!(effects, Inf)
                end
            else
                push!(effects, Inf)
            end
        end
        main_effects[factor.name] = effects
    end

    # Find optimal levels (lowest RMSE)
    optimal_levels = Dict{Symbol, DOELevel}()
    for factor in doe.design.factors
        effects = main_effects[factor.name]
        best_idx = argmin(effects)
        optimal_levels[factor.name] = factor.levels[best_idx]
    end

    return DOEAnalysis(main_effects, Dict(), optimal_levels, Dict{Symbol, DOELevel}[])
end

"""
    find_observability_boundary(analysis::DOEAnalysis) -> Vector{Dict{Symbol, DOELevel}}

Extract the observability boundary conditions from analysis.
"""
function find_observability_boundary(analysis::DOEAnalysis)
    return analysis.observability_boundary
end

"""
    compute_main_effects(analysis::DOEAnalysis, factor::Symbol) -> Vector{Float64}

Get main effects for a specific factor.
"""
function compute_main_effects(analysis::DOEAnalysis, factor::Symbol)
    return get(analysis.main_effects, factor, Float64[])
end

export DOEAnalysis, analyze_observability, analyze_performance
export find_observability_boundary, compute_main_effects

# ============================================================================
# DOE Runner
# ============================================================================

"""
    run_doe!(doe, run_func) -> Vector{DOEResult}

Run all experiments in the DOE.

# Arguments
- `doe`: ObservabilityDOE or PerformanceDOE
- `run_func`: Function(levels::Dict{Symbol, DOELevel}) -> (obs_metrics, perf_metrics)

# Returns
Vector of DOEResult for each run.
"""
function run_doe!(doe::Union{ObservabilityDOE, PerformanceDOE}, run_func::Function)
    results = DOEResult[]

    for (i, levels) in enumerate(doe.design.runs)
        run = DOERun(i, levels)
        run.status = :running
        run.start_time = time()

        try
            obs_metrics, perf_metrics = run_func(levels)
            run.end_time = time()
            run.status = :completed

            # Determine pass/fail
            passed = true
            failure_reason = ""

            if doe isa ObservabilityDOE
                if obs_metrics !== nothing && !is_observable(obs_metrics)
                    passed = false
                    failure_reason = "Unobservable (cond=$(obs_metrics.condition_number), rank=$(obs_metrics.rank))"
                end
            else
                if perf_metrics !== nothing && !passes_performance(perf_metrics)
                    passed = false
                    failure_reason = "Performance (RMSE=$(perf_metrics.rmse_position), consistency=$(perf_metrics.consistency))"
                end
            end

            result = DOEResult(run, obs_metrics, perf_metrics, passed, failure_reason)
            push!(results, result)
            push!(doe.results, result)

        catch e
            run.end_time = time()
            run.status = :failed
            result = DOEResult(run, nothing, nothing, false, string(e))
            push!(results, result)
            push!(doe.results, result)
        end
    end

    return results
end

export run_doe!

# ============================================================================
# Predefined DOE Configurations
# ============================================================================

"""
    quick_observability_doe() -> ObservabilityDOE

Quick observability check with minimal factor combinations.
"""
function quick_observability_doe()
    factors = [trajectory_factor(), sensor_config_factor()]
    design = create_fractional_factorial("quick_obs", factors; fraction=0.5)
    ObservabilityDOE(design)
end

"""
    comprehensive_observability_doe() -> ObservabilityDOE

Comprehensive observability analysis with all factors.
"""
function comprehensive_observability_doe()
    factors = [
        trajectory_factor(),
        sensor_config_factor(),
        field_strength_factor(),
        gradient_strength_factor()
    ]
    design = create_full_factorial("comprehensive_obs", factors)
    ObservabilityDOE(design)
end

"""
    lawnmower_observability_doe() -> ObservabilityDOE

Focused DOE for lawnmower pattern observability issues.
"""
function lawnmower_observability_doe()
    # Custom trajectory factor focused on lawnmower variations
    traj = DOEFactor(:trajectory, [
        DOELevel("straight", TRAJ_STRAIGHT, "Straight legs"),
        DOELevel("lawnmower_tight", TRAJ_LAWNMOWER, "Tight turns"),
        DOELevel("lawnmower_wide", TRAJ_LAWNMOWER, "Wide turns"),
    ])

    # Y-axis gradient focus
    grad = DOEFactor(:y_gradient, [
        DOELevel("weak", 0.5, "0.5 nT/m"),
        DOELevel("nominal", 5.0, "5 nT/m"),
        DOELevel("strong", 20.0, "20 nT/m"),
    ])

    design = create_full_factorial("lawnmower_obs", [traj, grad])
    ObservabilityDOE(design)
end

export quick_observability_doe, comprehensive_observability_doe, lawnmower_observability_doe
