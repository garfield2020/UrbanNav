# ============================================================================
# HealthMonitor.jl - System health monitoring
# ============================================================================
#
# Ported from AUV-Navigation/src/health_monitor.jl
#
# Health monitoring is READ-ONLY with respect to navigation state.
# It observes but does not modify.
#
# Architecture:
#   StateEstimator → HealthMonitor → HealthReport → HealthStateMachine
#                         ↓
#                   Telemetry
# ============================================================================

using LinearAlgebra

export HealthMonitor, check_health, get_report, register_checker!
export AbstractHealthChecker, run_check

# ============================================================================
# Abstract Checker Interface
# ============================================================================

"""
    AbstractHealthChecker

Base type for health check implementations.

Implementations must define:
    run_check(checker, context) -> (MonitoredHealthState, Dict{Symbol,Float64})
"""
abstract type AbstractHealthChecker end

"""
    run_check(checker, context) -> (MonitoredHealthState, Dict{Symbol,Float64})

Run a health check. Must be implemented by checker types.

# Returns
- `state`: Resulting health state for this subsystem
- `metrics`: Dictionary of computed metrics
"""
function run_check end

# ============================================================================
# Health Context
# ============================================================================

"""
    HealthContext

Context passed to health checkers containing observable state.

# Fields
- `covariance`: Current state covariance matrix (or nothing)
- `innovations`: Recent innovation vectors
- `residuals`: Recent residual statistics
- `sensor_status`: Current sensor availability
- `timing_load`: Real-time timing load factor
- `timestamp`: Current time
"""
struct HealthContext
    covariance::Union{Matrix{Float64}, Nothing}
    innovations::Vector{Vector{Float64}}
    residuals::Vector{Float64}
    sensor_status::Dict{Symbol, Bool}
    timing_load::Float64
    timestamp::Float64
end

function HealthContext(;
    covariance = nothing,
    innovations = Vector{Float64}[],
    residuals = Float64[],
    sensor_status = Dict{Symbol, Bool}(),
    timing_load::Real = 0.0,
    timestamp::Real = 0.0
)
    HealthContext(
        covariance,
        innovations,
        residuals,
        sensor_status,
        Float64(timing_load),
        Float64(timestamp)
    )
end

# ============================================================================
# Health Monitor
# ============================================================================

"""
    HealthMonitor

Central health monitoring system.

# Fields
- `state`: Current health states per subsystem
- `history`: Transition history
- `checkers`: Registered health checkers
- `metrics`: Current metric values
- `timestamp`: Last update time
"""
mutable struct HealthMonitor
    state::Dict{Symbol, MonitoredHealthState}
    history::Vector{HealthTransition}
    checkers::Dict{Symbol, AbstractHealthChecker}
    metrics::Dict{Symbol, Float64}
    timestamp::Float64
    max_history::Int
end

"""
    HealthMonitor(; max_history=100)

Create a new health monitor with all subsystems in NOMINAL state.
"""
function HealthMonitor(; max_history::Int = 100)
    initial_state = Dict{Symbol, MonitoredHealthState}()
    for subsystem in HEALTH_SUBSYSTEMS
        initial_state[subsystem] = NOMINAL
    end

    return HealthMonitor(
        initial_state,
        HealthTransition[],
        Dict{Symbol, AbstractHealthChecker}(),
        Dict{Symbol, Float64}(),
        0.0,
        max_history
    )
end

"""
    register_checker!(monitor, subsystem, checker)

Register a health checker for a subsystem.
"""
function register_checker!(monitor::HealthMonitor, subsystem::Symbol,
                           checker::AbstractHealthChecker)
    if !(subsystem in HEALTH_SUBSYSTEMS)
        @warn "Registering checker for unknown subsystem" subsystem
    end
    monitor.checkers[subsystem] = checker
    return monitor
end

"""
    check_health(monitor, context) -> HealthReport

Run all health checks and return a report.
"""
function check_health(monitor::HealthMonitor, context::HealthContext)
    monitor.timestamp = context.timestamp

    # Run registered checkers
    for (subsystem, checker) in monitor.checkers
        try
            new_state, metrics = run_check(checker, context)

            # Record metrics
            for (k, v) in metrics
                monitor.metrics[k] = v
            end

            # Handle state transition
            old_state = get(monitor.state, subsystem, NOMINAL)
            if new_state != old_state
                if is_valid_health_transition(old_state, new_state)
                    monitor.state[subsystem] = new_state
                    transition = HealthTransition(
                        old_state, new_state, context.timestamp,
                        "Automatic transition from $(typeof(checker))",
                        subsystem
                    )
                    push!(monitor.history, transition)

                    # Trim history if needed
                    if length(monitor.history) > monitor.max_history
                        deleteat!(monitor.history, 1:length(monitor.history) - monitor.max_history)
                    end
                else
                    @warn "Invalid health transition" subsystem old_state new_state
                end
            end
        catch e
            @warn "Health checker failed" subsystem exception=e
            # Mark subsystem as WARNING if checker fails
            monitor.state[subsystem] = WARNING
        end
    end

    return get_report(monitor)
end

"""
    get_report(monitor) -> HealthReport

Get current health report.
"""
function get_report(monitor::HealthMonitor)
    # Overall health is worst of all subsystems
    overall = NOMINAL
    for state in values(monitor.state)
        if Int(state) > Int(overall)
            overall = state
        end
    end

    # Recent transitions (last 10)
    n = length(monitor.history)
    recent = n > 0 ? monitor.history[max(1, n-9):n] : HealthTransition[]

    return HealthReport(
        overall,
        copy(monitor.state),
        copy(monitor.metrics),
        monitor.timestamp,
        recent
    )
end

"""
    reset!(monitor)

Reset monitor to initial state.
"""
function reset!(monitor::HealthMonitor)
    for subsystem in HEALTH_SUBSYSTEMS
        monitor.state[subsystem] = NOMINAL
    end
    empty!(monitor.history)
    empty!(monitor.metrics)
    monitor.timestamp = 0.0
    return monitor
end

"""
    get_subsystem_state(monitor, subsystem) -> MonitoredHealthState

Get current state of a specific subsystem.
"""
function get_subsystem_state(monitor::HealthMonitor, subsystem::Symbol)
    return get(monitor.state, subsystem, NOMINAL)
end

# ============================================================================
# Standard Health Checkers
# ============================================================================

"""
    CovarianceChecker <: AbstractHealthChecker

Checks covariance consistency.

# Parameters
- `position_threshold`: Position std threshold for CAUTION [m]
- `position_critical`: Position std threshold for CRITICAL [m]
- `velocity_threshold`: Velocity std threshold [m/s]
- `condition_threshold`: Condition number threshold for CAUTION
- `condition_critical`: Condition number threshold for CRITICAL
"""
struct CovarianceChecker <: AbstractHealthChecker
    position_threshold::Float64
    position_critical::Float64
    velocity_threshold::Float64
    condition_threshold::Float64
    condition_critical::Float64
end

function CovarianceChecker(;
    position_threshold::Real = 5.0,
    position_critical::Real = 20.0,
    velocity_threshold::Real = 1.0,
    condition_threshold::Real = 1e6,
    condition_critical::Real = 1e9
)
    CovarianceChecker(
        Float64(position_threshold),
        Float64(position_critical),
        Float64(velocity_threshold),
        Float64(condition_threshold),
        Float64(condition_critical)
    )
end

function run_check(checker::CovarianceChecker, context::HealthContext)
    metrics = Dict{Symbol, Float64}()

    if context.covariance === nothing
        # No covariance available
        return NOMINAL, metrics
    end

    P = context.covariance
    n = size(P, 1)

    # Extract uncertainties (assuming standard state layout: pos, vel, ...)
    pos_end = min(3, n)
    vel_end = min(6, n)

    pos_var = n >= 3 ? tr(P[1:pos_end, 1:pos_end]) : 0.0
    vel_var = n >= 6 ? tr(P[4:vel_end, 4:vel_end]) : 0.0

    # Use LinearAlgebra.cond for condition number
    cond_num = try
        cond(P)
    catch
        Inf
    end

    metrics[:position_uncertainty] = sqrt(pos_var)
    metrics[:velocity_uncertainty] = sqrt(vel_var)
    metrics[:covariance_condition] = cond_num

    # Determine health state
    state = NOMINAL
    pos_std = sqrt(pos_var)

    if pos_std > checker.position_critical || cond_num > checker.condition_critical
        state = CRITICAL
    elseif pos_std > checker.position_threshold || cond_num > checker.condition_threshold
        state = WARNING
    elseif pos_std > checker.position_threshold * 0.5
        state = CAUTION
    end

    return state, metrics
end

export CovarianceChecker

"""
    InnovationChecker <: AbstractHealthChecker

Checks measurement innovations for consistency.

# Parameters
- `threshold_caution`: χ² threshold for CAUTION
- `threshold_critical`: χ² threshold for CRITICAL
- `window_size`: Number of innovations to consider
"""
struct InnovationChecker <: AbstractHealthChecker
    threshold_caution::Float64
    threshold_critical::Float64
    window_size::Int
end

function InnovationChecker(;
    threshold_caution::Real = 9.0,    # ~99% for 3 DOF
    threshold_critical::Real = 16.27, # ~99.9% for 3 DOF
    window_size::Int = 20
)
    InnovationChecker(Float64(threshold_caution), Float64(threshold_critical), window_size)
end

function run_check(checker::InnovationChecker, context::HealthContext)
    metrics = Dict{Symbol, Float64}()

    if isempty(context.innovations)
        metrics[:innovation_mean] = 0.0
        metrics[:innovation_count] = 0.0
        return NOMINAL, metrics
    end

    # Compute chi-squared values for each innovation
    chi2_values = Float64[]
    for innov in context.innovations[max(1, end-checker.window_size+1):end]
        push!(chi2_values, dot(innov, innov))
    end

    mean_chi2 = mean(chi2_values)
    max_chi2 = maximum(chi2_values)

    metrics[:innovation_mean] = mean_chi2
    metrics[:innovation_max] = max_chi2
    metrics[:innovation_count] = Float64(length(chi2_values))

    # Determine health state
    state = NOMINAL
    if max_chi2 > checker.threshold_critical
        state = CRITICAL
    elseif mean_chi2 > checker.threshold_caution
        state = WARNING
    elseif mean_chi2 > checker.threshold_caution * 0.5
        state = CAUTION
    end

    return state, metrics
end

export InnovationChecker

"""
    ResidualChecker <: AbstractHealthChecker

Checks residual sequence for anomalies.

# Parameters
- `outlier_threshold`: Fraction of outliers for CAUTION
- `outlier_critical`: Fraction of outliers for CRITICAL
- `chi2_threshold`: Per-measurement χ² threshold
"""
struct ResidualChecker <: AbstractHealthChecker
    outlier_threshold::Float64
    outlier_critical::Float64
    chi2_threshold::Float64
end

function ResidualChecker(;
    outlier_threshold::Real = 0.1,   # 10% outliers → CAUTION
    outlier_critical::Real = 0.3,    # 30% outliers → CRITICAL
    chi2_threshold::Real = 11.345    # χ²(3, 0.99)
)
    ResidualChecker(Float64(outlier_threshold), Float64(outlier_critical), Float64(chi2_threshold))
end

function run_check(checker::ResidualChecker, context::HealthContext)
    metrics = Dict{Symbol, Float64}()

    if isempty(context.residuals)
        metrics[:residual_mean] = 0.0
        metrics[:outlier_rate] = 0.0
        return NOMINAL, metrics
    end

    residuals = context.residuals
    mean_resid = mean(residuals)
    outlier_count = count(r -> r > checker.chi2_threshold, residuals)
    outlier_rate = outlier_count / length(residuals)

    metrics[:residual_mean] = mean_resid
    metrics[:outlier_rate] = outlier_rate
    metrics[:residual_count] = Float64(length(residuals))

    # Determine health state
    state = NOMINAL
    if outlier_rate > checker.outlier_critical
        state = CRITICAL
    elseif outlier_rate > checker.outlier_threshold
        state = WARNING
    elseif outlier_rate > checker.outlier_threshold * 0.5
        state = CAUTION
    end

    return state, metrics
end

export ResidualChecker

"""
    SensorChecker <: AbstractHealthChecker

Checks sensor availability.

# Parameters
- `required_sensors`: Sensors that must be available
- `min_sensors`: Minimum sensor count for NOMINAL
"""
struct SensorChecker <: AbstractHealthChecker
    required_sensors::Vector{Symbol}
    min_sensors::Int
end

function SensorChecker(;
    required_sensors::Vector{Symbol} = [:imu],
    min_sensors::Int = 2
)
    SensorChecker(required_sensors, min_sensors)
end

function run_check(checker::SensorChecker, context::HealthContext)
    metrics = Dict{Symbol, Float64}()

    sensor_status = context.sensor_status
    active_count = count(values(sensor_status))
    required_available = all(get(sensor_status, s, false) for s in checker.required_sensors)

    metrics[:active_sensors] = Float64(active_count)
    metrics[:required_available] = required_available ? 1.0 : 0.0

    # Determine health state
    state = NOMINAL
    if !required_available
        state = CRITICAL
    elseif active_count < checker.min_sensors
        state = WARNING
    elseif active_count < checker.min_sensors + 1
        state = CAUTION
    end

    return state, metrics
end

export SensorChecker

"""
    TimingChecker <: AbstractHealthChecker

Checks real-time timing budget.

# Parameters
- `caution_load`: Load threshold for CAUTION
- `critical_load`: Load threshold for CRITICAL
"""
struct TimingChecker <: AbstractHealthChecker
    caution_load::Float64
    critical_load::Float64
end

function TimingChecker(;
    caution_load::Real = 0.7,   # 70% CPU load → CAUTION
    critical_load::Real = 0.95  # 95% CPU load → CRITICAL
)
    TimingChecker(Float64(caution_load), Float64(critical_load))
end

function run_check(checker::TimingChecker, context::HealthContext)
    metrics = Dict{Symbol, Float64}()

    load = context.timing_load
    metrics[:timing_load] = load
    metrics[:timing_margin] = 1.0 - load

    # Determine health state
    state = NOMINAL
    if load > checker.critical_load
        state = CRITICAL
    elseif load > checker.caution_load
        state = WARNING
    elseif load > checker.caution_load * 0.8
        state = CAUTION
    end

    return state, metrics
end

export TimingChecker

# ============================================================================
# Observability Checker (Y-axis detection)
# ============================================================================

"""
    ObservabilityChecker <: AbstractHealthChecker

Checks position observability from magnetic measurements.

CRITICAL: Detects when Y-axis becomes unobservable due to:
- Collinear dipole sources along X-axis
- Weak magnetic gradients in Y direction
- Insufficient geometric diversity

# Parameters
- `min_y_info::Float64`: Minimum information gain in Y for NOMINAL
- `critical_y_info::Float64`: Information threshold for CRITICAL
- `gradient_ratio_threshold::Float64`: Min ratio of Gy/Gx for observability

# Key Metrics
- `y_information`: Fisher information in Y direction (from H'R⁻¹H)
- `x_information`: Fisher information in X direction
- `info_ratio`: y_information / x_information (should be > 0.1 for balance)
- `gradient_y_magnitude`: Magnetic gradient strength in Y direction
"""
struct ObservabilityChecker <: AbstractHealthChecker
    min_y_info::Float64           # Minimum Y information for NOMINAL
    critical_y_info::Float64      # Y information below this → CRITICAL
    gradient_ratio_threshold::Float64  # Min |Gy|/|Gx| ratio
    condition_number_threshold::Float64  # Max condition number for NOMINAL
end

function ObservabilityChecker(;
    min_y_info::Real = 1e-6,      # Reasonable Y information
    critical_y_info::Real = 1e-9, # Nearly unobservable
    gradient_ratio_threshold::Real = 0.1,  # Y gradient at least 10% of X
    condition_number_threshold::Real = 1e6  # Well-conditioned
)
    ObservabilityChecker(
        Float64(min_y_info),
        Float64(critical_y_info),
        Float64(gradient_ratio_threshold),
        Float64(condition_number_threshold)
    )
end

"""
    compute_position_observability(H_mag::AbstractMatrix, R::AbstractMatrix)

Compute observability metrics for X and Y axes from magnetic Jacobian.

The Fisher information matrix for position is: I = H' R⁻¹ H
The diagonal elements give per-axis information.

Returns: (info_x, info_y, info_z, condition_number)
"""
function compute_position_observability(H_mag::AbstractMatrix, R::AbstractMatrix)
    m, n = size(H_mag)

    if m == 0 || n < 2
        return (0.0, 0.0, 0.0, Inf)
    end

    # Fisher information: I = H' R⁻¹ H
    # For simplicity with diagonal R, this is sum of squared Jacobian elements / R
    try
        R_inv = inv(R)
        I_fisher = H_mag' * R_inv * H_mag

        # Extract position information (assuming first 3 columns are position)
        n_pos = min(3, n)
        I_pos = I_fisher[1:n_pos, 1:n_pos]

        # Condition number of position block
        cond_num = cond(I_pos)

        # Per-axis information (diagonal elements)
        info_x = n_pos >= 1 ? I_pos[1, 1] : 0.0
        info_y = n_pos >= 2 ? I_pos[2, 2] : 0.0
        info_z = n_pos >= 3 ? I_pos[3, 3] : 0.0

        return (info_x, info_y, info_z, cond_num)
    catch e
        # Matrix inversion failed
        return (0.0, 0.0, 0.0, Inf)
    end
end

"""
    compute_gradient_observability(gradients::Vector{<:AbstractMatrix})

Compute observability from magnetic gradient measurements.
Checks if gradients span both X and Y directions.

Returns: (grad_x_magnitude, grad_y_magnitude, span_ratio)
"""
function compute_gradient_observability(gradients::Vector{<:AbstractMatrix})
    if isempty(gradients)
        return (0.0, 0.0, 0.0)
    end

    # Accumulate gradient magnitudes in each direction
    sum_gx = 0.0
    sum_gy = 0.0

    for G in gradients
        # G is 3x3 gradient tensor
        # dB/dx affects X observability, dB/dy affects Y observability
        if size(G, 2) >= 2
            sum_gx += norm(G[:, 1])  # dB/dx column
            sum_gy += norm(G[:, 2])  # dB/dy column
        end
    end

    n = length(gradients)
    grad_x = sum_gx / n
    grad_y = sum_gy / n

    # Ratio of Y to X gradient (should be close to 1 for balanced observability)
    span_ratio = grad_x > 0 ? grad_y / grad_x : 0.0

    return (grad_x, grad_y, span_ratio)
end

function run_check(checker::ObservabilityChecker, context::HealthContext)
    metrics = Dict{Symbol, Float64}()

    # Default values if no observability data available
    metrics[:y_information] = 0.0
    metrics[:x_information] = 0.0
    metrics[:info_ratio] = 0.0
    metrics[:observability_cond] = Inf
    metrics[:y_observable] = 0.0  # 0 = unobservable, 1 = observable

    # Check if we have a covariance to analyze
    if context.covariance === nothing
        return NOMINAL, metrics  # Can't check without data
    end

    P = context.covariance
    n = size(P, 1)

    if n < 3
        return NOMINAL, metrics  # Need at least 3 states
    end

    # Use inverse covariance (information matrix) as proxy for observability
    try
        I_info = inv(P)

        # Position block is typically first 3 states
        I_pos = I_info[1:3, 1:3]

        # Per-axis information from diagonal
        info_x = abs(I_pos[1, 1])
        info_y = abs(I_pos[2, 2])
        info_z = abs(I_pos[3, 3])

        # Condition number of position information
        cond_num = cond(I_pos)

        # Information ratio
        info_ratio = info_x > 0 ? info_y / info_x : 0.0

        metrics[:y_information] = info_y
        metrics[:x_information] = info_x
        metrics[:info_ratio] = info_ratio
        metrics[:observability_cond] = cond_num

        # Determine observability status
        if info_y > checker.min_y_info
            metrics[:y_observable] = 1.0
        else
            metrics[:y_observable] = 0.0
        end

        # Determine health state
        state = NOMINAL

        if info_y < checker.critical_y_info
            state = CRITICAL
        elseif info_y < checker.min_y_info || info_ratio < checker.gradient_ratio_threshold
            state = WARNING
        elseif cond_num > checker.condition_number_threshold
            state = CAUTION
        end

        return state, metrics

    catch e
        # Singular covariance - observability is compromised
        metrics[:y_observable] = 0.0
        return WARNING, metrics
    end
end

export ObservabilityChecker, compute_position_observability, compute_gradient_observability

# ============================================================================
# Convenience Functions
# ============================================================================

"""
    create_default_monitor() -> HealthMonitor

Create a health monitor with standard checkers registered.
"""
function create_default_monitor()
    monitor = HealthMonitor()

    register_checker!(monitor, :covariance, CovarianceChecker())
    register_checker!(monitor, :innovation, InnovationChecker())
    register_checker!(monitor, :residual, ResidualChecker())
    register_checker!(monitor, :timing, TimingChecker())

    return monitor
end

export create_default_monitor, HealthContext, reset!
export get_subsystem_state
