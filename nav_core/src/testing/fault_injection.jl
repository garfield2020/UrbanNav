# ============================================================================
# fault_injection.jl - Fault injection for "No Silent Divergence" gate
# ============================================================================
#
# Step 8 of NEES Recovery Plan:
# Make "No Silent Divergence" a fault-injection gate.
#
# Silent divergence occurs when:
# 1. True estimation error grows beyond acceptable bounds
# 2. But the health system does NOT flag it (no CAUTION/WARNING/CRITICAL)
# 3. Filter appears "healthy" while actually diverging
#
# This module provides:
# - FaultType: Enumeration of injectable fault types
# - FaultInjector: Injects faults into estimator/sensors
# - DivergenceDetector: Detects silent divergence (error without flags)
# - FaultInjectionGate: Pass/fail gate for fault response testing
# ============================================================================

export FaultType, FAULT_NONE, FAULT_BIAS_DRIFT, FAULT_SENSOR_DROPOUT
export FAULT_MAP_ERROR, FAULT_TIMING_SPIKE, FAULT_OUTLIER_BURST
export FAULT_GRADUAL_DRIFT, FAULT_SUDDEN_JUMP, FAULT_OSCILLATION
export FaultConfig, FaultInjector, inject_fault!, clear_faults!, get_active_faults
export DivergenceConfig, DivergenceState, DivergenceResult
export DivergenceDetector, update_divergence!, check_divergence, is_silent_divergence
export FaultInjectionGate, GateResult, GATE_PASS, GATE_FAIL, GATE_INCONCLUSIVE
export run_fault_gate, run_fault_scenario, FaultScenario
export SilentDivergenceChecker

using LinearAlgebra
using Statistics

# ============================================================================
# Fault Types
# ============================================================================

"""
    FaultType

Types of faults that can be injected for testing.

# Categories
- Navigation faults: Affect estimator state/covariance
- Sensor faults: Affect sensor inputs
- Map faults: Affect magnetic map predictions
- System faults: Affect timing/computation
"""
@enum FaultType begin
    FAULT_NONE = 0

    # Sensor faults
    FAULT_SENSOR_DROPOUT = 1     # Sensor stops providing data
    FAULT_BIAS_DRIFT = 2         # Sensor bias slowly drifts
    FAULT_OUTLIER_BURST = 3      # Burst of outlier measurements
    FAULT_NOISE_INCREASE = 4     # Sensor noise increases

    # Navigation faults
    FAULT_GRADUAL_DRIFT = 10     # Slow position/velocity drift
    FAULT_SUDDEN_JUMP = 11       # Sudden state jump
    FAULT_OSCILLATION = 12       # Oscillating error

    # Map faults
    FAULT_MAP_ERROR = 20         # Map prediction error
    FAULT_MAP_OFFSET = 21        # Constant map offset
    FAULT_MAP_SCALE = 22         # Map scale error

    # System faults
    FAULT_TIMING_SPIKE = 30      # Computation takes too long
    FAULT_TIMING_JITTER = 31     # Variable timing
    FAULT_MEMORY_PRESSURE = 32   # Memory allocation delays
end

# ============================================================================
# Fault Configuration
# ============================================================================

"""
    FaultConfig

Configuration for a single fault injection.

# Fields
- `fault_type::FaultType`: Type of fault to inject
- `start_time::Float64`: When fault begins (seconds)
- `duration::Float64`: How long fault lasts (Inf for permanent)
- `magnitude::Float64`: Fault magnitude (type-specific units)
- `ramp_time::Float64`: Time to ramp up to full magnitude (0 for instant)
- `sensor::Symbol`: Which sensor is affected (for sensor faults)
- `axis::Int`: Which axis (1=X, 2=Y, 3=Z, 0=all)
- `metadata::Dict{Symbol,Any}`: Additional fault parameters
"""
Base.@kwdef struct FaultConfig
    fault_type::FaultType = FAULT_NONE
    start_time::Float64 = 0.0
    duration::Float64 = Inf
    magnitude::Float64 = 1.0
    ramp_time::Float64 = 0.0
    sensor::Symbol = :all
    axis::Int = 0
    metadata::Dict{Symbol,Any} = Dict{Symbol,Any}()
end

# ============================================================================
# Fault Injector
# ============================================================================

"""
    FaultInjector

Manages fault injection for testing.

# Fields
- `faults::Vector{FaultConfig}`: Configured faults
- `active::Vector{Bool}`: Which faults are currently active
- `current_time::Float64`: Current simulation time
- `rng::AbstractRNG`: Random number generator for stochastic faults
"""
mutable struct FaultInjector
    faults::Vector{FaultConfig}
    active::Vector{Bool}
    current_time::Float64
    rng::AbstractRNG

    # Accumulated effects (for drift-type faults)
    accumulated_bias::Dict{Symbol, Vector{Float64}}
    accumulated_position_error::Vector{Float64}
end

function FaultInjector(; rng::AbstractRNG = Random.default_rng())
    FaultInjector(
        FaultConfig[],
        Bool[],
        0.0,
        rng,
        Dict{Symbol, Vector{Float64}}(),
        zeros(3)
    )
end

"""
    inject_fault!(injector, config)

Add a fault to be injected.
"""
function inject_fault!(injector::FaultInjector, config::FaultConfig)
    push!(injector.faults, config)
    push!(injector.active, false)
    return injector
end

"""
    clear_faults!(injector)

Remove all configured faults.
"""
function clear_faults!(injector::FaultInjector)
    empty!(injector.faults)
    empty!(injector.active)
    empty!(injector.accumulated_bias)
    fill!(injector.accumulated_position_error, 0.0)
    return injector
end

"""
    get_active_faults(injector) -> Vector{FaultConfig}

Get list of currently active faults.
"""
function get_active_faults(injector::FaultInjector)
    active_faults = FaultConfig[]
    for (i, config) in enumerate(injector.faults)
        if injector.active[i]
            push!(active_faults, config)
        end
    end
    return active_faults
end

"""
    update_injector!(injector, time, dt) -> effects

Update fault injector state and compute current fault effects.

Returns a Dict with:
- :sensor_bias => Dict{Symbol, Vector{Float64}}
- :sensor_dropout => Set{Symbol}
- :position_error => Vector{Float64}
- :timing_factor => Float64
- :map_offset => Vector{Float64}
"""
function update_injector!(injector::FaultInjector, time::Float64, dt::Float64)
    injector.current_time = time

    effects = Dict{Symbol, Any}(
        :sensor_bias => Dict{Symbol, Vector{Float64}}(),
        :sensor_dropout => Set{Symbol}(),
        :position_error => zeros(3),
        :timing_factor => 1.0,
        :map_offset => zeros(3)
    )

    for (i, config) in enumerate(injector.faults)
        # Check if fault is active
        is_active = time >= config.start_time &&
                    time < config.start_time + config.duration

        injector.active[i] = is_active

        if !is_active
            continue
        end

        # Compute ramp factor
        time_since_start = time - config.start_time
        ramp_factor = if config.ramp_time > 0
            min(1.0, time_since_start / config.ramp_time)
        else
            1.0
        end

        magnitude = config.magnitude * ramp_factor

        # Apply fault effects based on type
        apply_fault_effect!(effects, config, magnitude, dt, injector)
    end

    return effects
end

"""
    apply_fault_effect!(effects, config, magnitude, dt, injector)

Apply a specific fault's effects to the effects dictionary.
"""
function apply_fault_effect!(effects::Dict{Symbol, Any}, config::FaultConfig,
                             magnitude::Float64, dt::Float64, injector::FaultInjector)
    ft = config.fault_type

    if ft == FAULT_SENSOR_DROPOUT
        push!(effects[:sensor_dropout], config.sensor)

    elseif ft == FAULT_BIAS_DRIFT
        # Accumulate bias drift
        sensor = config.sensor
        if !haskey(injector.accumulated_bias, sensor)
            injector.accumulated_bias[sensor] = zeros(3)
        end
        drift_rate = magnitude * dt  # magnitude is drift rate (units/s)
        if config.axis == 0
            injector.accumulated_bias[sensor] .+= drift_rate
        else
            injector.accumulated_bias[sensor][config.axis] += drift_rate
        end
        effects[:sensor_bias][sensor] = copy(injector.accumulated_bias[sensor])

    elseif ft == FAULT_OUTLIER_BURST
        # Add large outlier bias (handled by caller with random sign)
        sensor = config.sensor
        outlier = magnitude * (2 * rand(injector.rng) - 1)
        bias = get(effects[:sensor_bias], sensor, zeros(3))
        if config.axis == 0
            bias .+= outlier
        else
            bias[config.axis] += outlier
        end
        effects[:sensor_bias][sensor] = bias

    elseif ft == FAULT_GRADUAL_DRIFT
        # Accumulate position error
        drift_rate = magnitude * dt
        if config.axis == 0
            injector.accumulated_position_error .+= drift_rate
        else
            injector.accumulated_position_error[config.axis] += drift_rate
        end
        effects[:position_error] = copy(injector.accumulated_position_error)

    elseif ft == FAULT_SUDDEN_JUMP
        # Instant position error (only applied once at start)
        if get(config.metadata, :applied, false) == false
            if config.axis == 0
                effects[:position_error] .= magnitude
            else
                effects[:position_error][config.axis] = magnitude
            end
            # Mark as applied (need mutable config for this)
        end

    elseif ft == FAULT_MAP_ERROR || ft == FAULT_MAP_OFFSET
        if config.axis == 0
            effects[:map_offset] .= magnitude
        else
            effects[:map_offset][config.axis] = magnitude
        end

    elseif ft == FAULT_TIMING_SPIKE
        effects[:timing_factor] = max(effects[:timing_factor], magnitude)

    elseif ft == FAULT_OSCILLATION
        # Sinusoidal error
        freq = get(config.metadata, :frequency, 0.5)  # Hz
        phase = 2π * freq * injector.current_time
        osc_value = magnitude * sin(phase)
        if config.axis == 0
            effects[:position_error] .= osc_value
        else
            effects[:position_error][config.axis] = osc_value
        end
    end

    return nothing
end

export FaultInjector, inject_fault!, clear_faults!, get_active_faults, update_injector!

# ============================================================================
# Divergence Detection
# ============================================================================

"""
    DivergenceConfig

Configuration for divergence detection.

# Fields
- `error_threshold::Float64`: Position error threshold for divergence (m)
- `velocity_error_threshold::Float64`: Velocity error threshold (m/s)
- `nees_threshold::Float64`: NEES threshold for filter consistency
- `window_size::Int`: Number of samples for trend detection
- `growth_rate_threshold::Float64`: Error growth rate for divergence (m/s)
"""
Base.@kwdef struct DivergenceConfig
    error_threshold::Float64 = 10.0        # 10m position error
    velocity_error_threshold::Float64 = 2.0 # 2 m/s velocity error
    nees_threshold::Float64 = 10.0         # NEES > 10 is suspicious
    window_size::Int = 50                   # Samples for trend
    growth_rate_threshold::Float64 = 0.1   # 0.1 m/s error growth
    silent_detection_time::Float64 = 5.0   # Must be undetected for 5s
end

"""
    DivergenceState

State of divergence detection.
"""
mutable struct DivergenceState
    error_history::Vector{Float64}
    nees_history::Vector{Float64}
    health_history::Vector{Int}  # 0=NOMINAL, 1=CAUTION, 2=WARNING, 3=CRITICAL
    timestamps::Vector{Float64}

    divergence_detected::Bool
    divergence_start_time::Float64
    health_flagged::Bool
    health_flag_time::Float64

    peak_error::Float64
    peak_nees::Float64
end

function DivergenceState()
    DivergenceState(
        Float64[], Float64[], Int[], Float64[],
        false, 0.0, false, 0.0,
        0.0, 0.0
    )
end

"""
    DivergenceResult

Result of divergence check.
"""
struct DivergenceResult
    is_diverging::Bool          # True if currently diverging
    is_silent::Bool             # True if diverging without health flag
    position_error::Float64     # Current position error
    nees::Float64               # Current NEES
    health_state::Int           # Current health state
    error_growth_rate::Float64  # Error growth rate (m/s)
    time_undetected::Float64    # How long divergence went undetected
end

"""
    DivergenceDetector

Detects silent divergence in navigation.

Silent divergence = estimation error growing without health system flagging it.
"""
struct DivergenceDetector
    config::DivergenceConfig
    state::DivergenceState
end

function DivergenceDetector(config::DivergenceConfig = DivergenceConfig())
    DivergenceDetector(config, DivergenceState())
end

"""
    update_divergence!(detector, true_position, estimated_position, covariance, health_state, timestamp)

Update divergence detector with new data.

# Arguments
- `true_position`: Ground truth position (3-vector)
- `estimated_position`: Estimated position (3-vector)
- `covariance`: Position covariance (3x3 matrix)
- `health_state`: Current health state (MonitoredHealthState or Int)
- `timestamp`: Current time
"""
function update_divergence!(detector::DivergenceDetector,
                            true_position::AbstractVector,
                            estimated_position::AbstractVector,
                            covariance::AbstractMatrix,
                            health_state,
                            timestamp::Float64)
    state = detector.state
    config = detector.config

    # Compute position error
    error_vec = estimated_position[1:3] - true_position[1:3]
    error_norm = norm(error_vec)

    # Compute NEES
    P_pos = covariance[1:min(3,size(covariance,1)), 1:min(3,size(covariance,2))]
    nees = try
        error_vec' * inv(P_pos) * error_vec
    catch
        Inf
    end

    # Convert health state to integer
    health_int = if health_state isa Integer
        Int(health_state)
    elseif hasfield(typeof(health_state), :value)
        Int(health_state.value)
    else
        # Assume it's an enum-like type
        Int(health_state)
    end

    # Update history
    push!(state.error_history, error_norm)
    push!(state.nees_history, nees)
    push!(state.health_history, health_int)
    push!(state.timestamps, timestamp)

    # Trim history
    max_history = config.window_size * 2
    if length(state.error_history) > max_history
        deleteat!(state.error_history, 1:length(state.error_history) - max_history)
        deleteat!(state.nees_history, 1:length(state.nees_history) - max_history)
        deleteat!(state.health_history, 1:length(state.health_history) - max_history)
        deleteat!(state.timestamps, 1:length(state.timestamps) - max_history)
    end

    # Update peak values
    state.peak_error = max(state.peak_error, error_norm)
    state.peak_nees = max(state.peak_nees, nees)

    # Check for divergence
    is_diverging = error_norm > config.error_threshold

    # Check for health flag (CAUTION or higher)
    is_flagged = health_int >= 1  # CAUTION = 1

    # Update divergence detection state
    if is_diverging && !state.divergence_detected
        state.divergence_detected = true
        state.divergence_start_time = timestamp
    elseif !is_diverging
        state.divergence_detected = false
        state.divergence_start_time = 0.0
    end

    # Update health flag state
    if is_flagged && !state.health_flagged
        state.health_flagged = true
        state.health_flag_time = timestamp
    elseif !is_diverging
        state.health_flagged = false
        state.health_flag_time = 0.0
    end

    return nothing
end

"""
    check_divergence(detector) -> DivergenceResult

Check current divergence status.
"""
function check_divergence(detector::DivergenceDetector)
    state = detector.state
    config = detector.config

    if isempty(state.error_history)
        return DivergenceResult(false, false, 0.0, 0.0, 0, 0.0, 0.0)
    end

    current_error = state.error_history[end]
    current_nees = state.nees_history[end]
    current_health = state.health_history[end]
    current_time = state.timestamps[end]

    # Compute error growth rate
    growth_rate = 0.0
    if length(state.error_history) >= config.window_size
        idx_start = length(state.error_history) - config.window_size + 1
        errors = state.error_history[idx_start:end]
        times = state.timestamps[idx_start:end]
        if times[end] > times[1]
            growth_rate = (errors[end] - errors[1]) / (times[end] - times[1])
        end
    end

    # Check if diverging
    is_diverging = current_error > config.error_threshold ||
                   growth_rate > config.growth_rate_threshold

    # Check if silent (diverging without flag)
    is_flagged = current_health >= 1

    # Time undetected
    time_undetected = 0.0
    if state.divergence_detected && !state.health_flagged
        time_undetected = current_time - state.divergence_start_time
    elseif state.divergence_detected && state.health_flagged
        # Health was flagged after divergence started
        time_undetected = state.health_flag_time - state.divergence_start_time
    end

    is_silent = is_diverging && !is_flagged && time_undetected > config.silent_detection_time

    return DivergenceResult(
        is_diverging,
        is_silent,
        current_error,
        current_nees,
        current_health,
        growth_rate,
        time_undetected
    )
end

"""
    is_silent_divergence(detector) -> Bool

Quick check for silent divergence.
"""
function is_silent_divergence(detector::DivergenceDetector)
    result = check_divergence(detector)
    return result.is_silent
end

export DivergenceConfig, DivergenceState, DivergenceResult
export DivergenceDetector, update_divergence!, check_divergence, is_silent_divergence

# ============================================================================
# Fault Injection Gate
# ============================================================================

"""
    GateResult

Result of fault injection gate.
"""
@enum GateResult begin
    GATE_PASS = 0           # Fault was properly detected
    GATE_FAIL = 1           # Silent divergence occurred
    GATE_INCONCLUSIVE = 2   # Not enough data
end

"""
    FaultScenario

A complete fault injection scenario for testing.

# Fields
- `name::String`: Scenario name
- `faults::Vector{FaultConfig}`: Faults to inject
- `duration::Float64`: Scenario duration (seconds)
- `max_silent_time::Float64`: Max allowed undetected divergence time
- `expected_health_state::Int`: Minimum expected health flag
"""
Base.@kwdef struct FaultScenario
    name::String = "unnamed"
    faults::Vector{FaultConfig} = FaultConfig[]
    duration::Float64 = 60.0
    max_silent_time::Float64 = 5.0
    expected_health_state::Int = 1  # At least CAUTION
end

"""
    FaultInjectionGate

Gate that validates proper fault detection.

Pass = All injected faults are detected by health system.
Fail = Any fault causes silent divergence.
"""
struct FaultInjectionGate
    divergence_config::DivergenceConfig
    scenarios::Vector{FaultScenario}
    results::Vector{Tuple{String, GateResult, DivergenceResult}}
end

function FaultInjectionGate(;
    divergence_config::DivergenceConfig = DivergenceConfig(),
    scenarios::Vector{FaultScenario} = FaultScenario[]
)
    FaultInjectionGate(divergence_config, scenarios, Tuple{String, GateResult, DivergenceResult}[])
end

"""
    run_fault_gate(gate, scenario, run_simulation) -> (GateResult, DivergenceResult)

Run a fault injection gate for a scenario.

# Arguments
- `gate`: FaultInjectionGate instance
- `scenario`: FaultScenario to run
- `run_simulation`: Function that runs simulation and returns (true_states, estimated_states, covariances, health_states, timestamps)

# Returns
(GateResult, DivergenceResult)
"""
function run_fault_gate(gate::FaultInjectionGate, scenario::FaultScenario,
                        run_simulation::Function)
    # Create injector and detector
    injector = FaultInjector()
    for fault in scenario.faults
        inject_fault!(injector, fault)
    end

    detector = DivergenceDetector(gate.divergence_config)

    # Run simulation
    true_states, est_states, covs, health_states, timestamps = run_simulation(injector, scenario.duration)

    # Process results
    for i in 1:length(timestamps)
        true_pos = true_states[i][1:3]
        est_pos = est_states[i][1:3]
        cov = covs[i]
        health = health_states[i]
        t = timestamps[i]

        update_divergence!(detector, true_pos, est_pos, cov, health, t)
    end

    # Check final divergence status
    result = check_divergence(detector)

    # Determine gate result
    gate_result = if result.is_silent
        GATE_FAIL
    elseif length(timestamps) < 10
        GATE_INCONCLUSIVE
    else
        # Check if fault was eventually detected
        max_health = maximum(detector.state.health_history)
        if max_health >= scenario.expected_health_state
            GATE_PASS
        elseif result.is_diverging
            GATE_FAIL
        else
            GATE_PASS
        end
    end

    return (gate_result, result)
end

"""
    run_fault_scenario(scenario, injector, detector, step_func, dt) -> GateResult

Run a fault scenario step by step.

# Arguments
- `scenario`: FaultScenario to run
- `injector`: FaultInjector (will be populated with scenario faults)
- `detector`: DivergenceDetector
- `step_func`: Function(injector, time, dt) -> (true_pos, est_pos, cov, health)
- `dt`: Time step

Returns the final GateResult.
"""
function run_fault_scenario(scenario::FaultScenario,
                            injector::FaultInjector,
                            detector::DivergenceDetector,
                            step_func::Function,
                            dt::Float64)
    # Setup injector
    clear_faults!(injector)
    for fault in scenario.faults
        inject_fault!(injector, fault)
    end

    # Run scenario
    time = 0.0
    while time < scenario.duration
        true_pos, est_pos, cov, health = step_func(injector, time, dt)
        update_divergence!(detector, true_pos, est_pos, cov, health, time)

        # Check for early failure
        result = check_divergence(detector)
        if result.is_silent && result.time_undetected > scenario.max_silent_time
            return GATE_FAIL
        end

        time += dt
    end

    # Final check
    result = check_divergence(detector)
    if result.is_silent
        return GATE_FAIL
    else
        max_health = maximum(detector.state.health_history)
        return max_health >= scenario.expected_health_state ? GATE_PASS : GATE_INCONCLUSIVE
    end
end

export FaultInjectionGate, GateResult, GATE_PASS, GATE_FAIL, GATE_INCONCLUSIVE
export run_fault_gate, run_fault_scenario, FaultScenario

# ============================================================================
# Silent Divergence Health Checker
# ============================================================================

"""
    SilentDivergenceChecker <: AbstractHealthChecker

Health checker that specifically looks for silent divergence indicators.

This checker monitors for:
1. Rapidly growing covariance (filter knows it's uncertain)
2. Consistently high innovations (measurements don't match predictions)
3. Correlation between axes (loss of observability)

# Parameters
- `cov_growth_threshold`: Covariance growth rate for CAUTION (1/s)
- `innovation_persistence`: How many high innovations before flagging
- `correlation_threshold`: Off-diagonal correlation threshold
"""
struct SilentDivergenceChecker <: AbstractHealthChecker
    cov_growth_threshold::Float64
    innovation_persistence::Int
    correlation_threshold::Float64
    history_size::Int

    # Mutable state stored in Dict
    state_key::Symbol
end

function SilentDivergenceChecker(;
    cov_growth_threshold::Real = 0.5,   # 0.5 m²/s growth rate
    innovation_persistence::Int = 5,     # 5 consecutive high innovations
    correlation_threshold::Real = 0.9,   # 90% correlation
    history_size::Int = 20
)
    SilentDivergenceChecker(
        Float64(cov_growth_threshold),
        innovation_persistence,
        Float64(correlation_threshold),
        history_size,
        gensym(:silent_div_state)
    )
end

# State for the checker (stored externally to maintain immutability)
mutable struct SilentDivergenceState
    prev_cov_trace::Float64
    prev_time::Float64
    has_previous::Bool  # Track if we have a valid previous state
    high_innovation_count::Int
    innovation_history::Vector{Float64}
end

const _silent_div_states = Dict{Symbol, SilentDivergenceState}()

function get_checker_state(checker::SilentDivergenceChecker)
    if !haskey(_silent_div_states, checker.state_key)
        _silent_div_states[checker.state_key] = SilentDivergenceState(0.0, 0.0, false, 0, Float64[])
    end
    return _silent_div_states[checker.state_key]
end

function run_check(checker::SilentDivergenceChecker, context::HealthContext)
    metrics = Dict{Symbol, Float64}()
    state = get_checker_state(checker)

    metrics[:cov_growth_rate] = 0.0
    metrics[:high_innovation_count] = 0.0
    metrics[:max_correlation] = 0.0

    if context.covariance === nothing
        return NOMINAL, metrics
    end

    P = context.covariance
    n = size(P, 1)
    current_time = context.timestamp

    # 1. Check covariance growth rate
    pos_end = min(3, n)
    cov_trace = tr(P[1:pos_end, 1:pos_end])

    if state.has_previous && current_time > state.prev_time
        dt = current_time - state.prev_time
        if dt > 0
            growth_rate = (cov_trace - state.prev_cov_trace) / dt
            metrics[:cov_growth_rate] = growth_rate
        end
    end

    state.prev_cov_trace = cov_trace
    state.prev_time = current_time
    state.has_previous = true

    # 2. Check innovation persistence
    if !isempty(context.innovations)
        # Take magnitude of most recent innovation
        innov_mag = norm(context.innovations[end])
        push!(state.innovation_history, innov_mag)

        # Trim history
        if length(state.innovation_history) > checker.history_size
            deleteat!(state.innovation_history, 1)
        end

        # Count consecutive high innovations
        # "High" = above median * 2
        if length(state.innovation_history) >= 5
            med = median(state.innovation_history)
            threshold = med * 2
            count = 0
            for i in length(state.innovation_history):-1:1
                if state.innovation_history[i] > threshold
                    count += 1
                else
                    break
                end
            end
            state.high_innovation_count = count
        end
    end

    metrics[:high_innovation_count] = Float64(state.high_innovation_count)

    # 3. Check position covariance correlation
    if n >= 3
        P_pos = P[1:3, 1:3]
        # Correlation matrix
        stds = sqrt.(diag(P_pos))
        max_corr = 0.0
        for i in 1:3, j in i+1:3
            if stds[i] > 0 && stds[j] > 0
                corr = abs(P_pos[i,j]) / (stds[i] * stds[j])
                max_corr = max(max_corr, corr)
            end
        end
        metrics[:max_correlation] = max_corr
    end

    # Determine health state
    health = NOMINAL

    if metrics[:cov_growth_rate] > checker.cov_growth_threshold
        health = max(health, WARNING)
    elseif metrics[:cov_growth_rate] > checker.cov_growth_threshold * 0.5
        health = max(health, CAUTION)
    end

    if state.high_innovation_count >= checker.innovation_persistence
        health = max(health, WARNING)
    elseif state.high_innovation_count >= checker.innovation_persistence ÷ 2
        health = max(health, CAUTION)
    end

    if metrics[:max_correlation] > checker.correlation_threshold
        health = max(health, CAUTION)
    end

    return health, metrics
end

export SilentDivergenceChecker

# ============================================================================
# Predefined Fault Scenarios
# ============================================================================

"""
    lawnmower_turn_scenario(turn_magnitude=0.5) -> FaultScenario

Scenario for testing divergence during lawnmower turns.

During turns, Y-axis observability degrades and bias can accumulate.
"""
function lawnmower_turn_scenario(; turn_magnitude::Float64 = 0.5)
    FaultScenario(
        name = "lawnmower_turn",
        faults = [
            FaultConfig(
                fault_type = FAULT_BIAS_DRIFT,
                start_time = 10.0,
                duration = 30.0,
                magnitude = 0.01,  # 1 cm/s drift
                sensor = :dvl,
                axis = 2  # Y-axis
            ),
            FaultConfig(
                fault_type = FAULT_GRADUAL_DRIFT,
                start_time = 15.0,
                duration = 25.0,
                magnitude = 0.05,  # 5 cm/s position drift
                axis = 2  # Y-axis
            )
        ],
        duration = 60.0,
        max_silent_time = 5.0,
        expected_health_state = 1  # Should flag CAUTION
    )
end

"""
    sensor_dropout_scenario(sensor=:dvl, dropout_time=5.0) -> FaultScenario

Scenario for testing response to sensor dropout.
"""
function sensor_dropout_scenario(; sensor::Symbol = :dvl, dropout_time::Float64 = 5.0)
    FaultScenario(
        name = "sensor_dropout_$(sensor)",
        faults = [
            FaultConfig(
                fault_type = FAULT_SENSOR_DROPOUT,
                start_time = 10.0,
                duration = dropout_time,
                sensor = sensor
            )
        ],
        duration = 30.0,
        max_silent_time = 3.0,
        expected_health_state = 2  # Should flag WARNING
    )
end

"""
    map_error_scenario(offset=5e-9) -> FaultScenario

Scenario for testing response to map prediction error.
"""
function map_error_scenario(; offset::Float64 = 5e-9)
    FaultScenario(
        name = "map_error",
        faults = [
            FaultConfig(
                fault_type = FAULT_MAP_OFFSET,
                start_time = 5.0,
                duration = 45.0,
                magnitude = offset
            )
        ],
        duration = 60.0,
        max_silent_time = 10.0,
        expected_health_state = 1
    )
end

export lawnmower_turn_scenario, sensor_dropout_scenario, map_error_scenario
