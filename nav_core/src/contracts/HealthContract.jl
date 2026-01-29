# ============================================================================
# HealthContract.jl - Authoritative definition of system health states
# ============================================================================
#
# Ported from AUV-Navigation/src/health_state_machine.jl
#
# This module implements a formal state machine with:
# - Explicit state definitions: NAV_HEALTHY, NAV_DEGRADED, NAV_UNRELIABLE
# - Requirements-driven transitions (not ad-hoc if-statements)
# - Transition logging with rule identification
# - Integration with fragility and confidence metrics
#
# Health monitoring is read-only with respect to navigation state.
# ============================================================================

export HealthState, HEALTH_HEALTHY, HEALTH_DEGRADED, HEALTH_UNRELIABLE
export SensorStatus, SENSOR_OK, SENSOR_DEGRADED, SENSOR_FAILED
export SafeStateAction, ACTION_NONE, ACTION_REDUCE_SPEED, ACTION_HOLD_POSITION, ACTION_SURFACE
export TransitionRule, TransitionInputs, TransitionRecord
export HealthStateMachineConfig, DEFAULT_HSM_CONFIG
export HealthStateMachine, evaluate_transition!, get_current_state, get_nav_state
export StateAction, get_state_actions
export print_transition_log, export_transition_log_json
export build_transition_inputs
export valid_health_transitions, is_valid_transition

# ============================================================================
# State Definitions
# ============================================================================

"""
    HealthState

Navigation health states.

States follow a severity hierarchy:
- HEALTH_HEALTHY: All systems functioning within spec
- HEALTH_DEGRADED: Some systems outside spec but mission can continue
- HEALTH_UNRELIABLE: Nav solution cannot be trusted
"""
@enum HealthState begin
    HEALTH_HEALTHY = 1      # All sensors nominal, nav trustworthy
    HEALTH_DEGRADED = 2     # Some issues, nav uncertain but usable
    HEALTH_UNRELIABLE = 3   # Nav solution cannot be trusted
end

"""
    SensorStatus

Individual sensor health status.
"""
@enum SensorStatus begin
    SENSOR_OK = 1
    SENSOR_DEGRADED = 2
    SENSOR_FAILED = 3
end

"""
    SafeStateAction

Recommended safe-state actions.
"""
@enum SafeStateAction begin
    ACTION_NONE = 1
    ACTION_REDUCE_SPEED = 2
    ACTION_HOLD_POSITION = 3
    ACTION_SURFACE = 4
end

# ============================================================================
# Transition Rules
# ============================================================================

"""
    TransitionRule

A formal transition rule with ID, description, and evaluation function.
"""
struct TransitionRule
    id::String                      # Rule ID (e.g., "RULE-HD-1")
    description::String             # Human-readable description
    from_state::HealthState         # Source state
    to_state::HealthState           # Target state
    evaluate::Function              # (inputs) -> Bool
end

"""
    TransitionInputs

All inputs needed to evaluate transition rules.
"""
struct TransitionInputs
    # From fragility
    fragility::Float64
    lambda_min::Float64
    fragility_rate::Float64
    is_anticipating::Bool

    # From confidence
    confidence::Float64

    # From residual statistics
    gamma_tail_rate::Float64       # Fraction of γ > threshold

    # From sensor status
    dvl_status::SensorStatus
    ftm_status::SensorStatus
    depth_status::SensorStatus
    imu_status::SensorStatus

    # From covariance
    position_std::Float64

    # Timing
    time_in_state::Float64
    current_time::Float64
end

"""
    TransitionRecord

Record of a state transition for audit logging.
"""
struct TransitionRecord
    timestamp::Float64
    from_state::HealthState
    to_state::HealthState
    rule_id::String
    rule_description::String
    inputs_snapshot::Dict{Symbol, Any}
end

# ============================================================================
# State Machine Configuration
# ============================================================================

"""
    HealthStateMachineConfig

Configuration for the health state machine.
"""
struct HealthStateMachineConfig
    # Fragility thresholds
    fragility_degraded::Float64      # HEALTHY → DEGRADED
    fragility_critical::Float64      # → UNRELIABLE
    fragility_recovery::Float64      # DEGRADED → HEALTHY

    # Confidence thresholds
    confidence_degraded::Float64     # HEALTHY → DEGRADED
    confidence_recovery::Float64     # DEGRADED → HEALTHY

    # Tail rate thresholds
    tail_rate_degraded::Float64      # γ tail rate for DEGRADED

    # Position std thresholds (m)
    position_std_lost::Float64       # → UNRELIABLE
    position_std_recovery::Float64   # UNRELIABLE → DEGRADED

    # Lambda thresholds
    lambda_critical::Float64         # Information eigenvalue for UNRELIABLE

    # Timing (s)
    healthy_hold_time::Float64       # Time to stay good before HEALTHY
    degraded_timeout::Float64        # Max time in DEGRADED before UNRELIABLE
    recovery_time::Float64           # Time to stay good before recovery
end

function HealthStateMachineConfig(;
    fragility_degraded::Float64 = 0.7,
    fragility_critical::Float64 = 0.95,
    fragility_recovery::Float64 = 0.35,
    confidence_degraded::Float64 = 0.5,
    confidence_recovery::Float64 = 0.65,
    tail_rate_degraded::Float64 = 0.03,
    position_std_lost::Float64 = 20.0,
    position_std_recovery::Float64 = 10.0,
    lambda_critical::Float64 = 1e-9,
    healthy_hold_time::Float64 = 3.0,
    degraded_timeout::Float64 = 10.0,
    recovery_time::Float64 = 5.0
)
    HealthStateMachineConfig(
        fragility_degraded, fragility_critical, fragility_recovery,
        confidence_degraded, confidence_recovery, tail_rate_degraded,
        position_std_lost, position_std_recovery, lambda_critical,
        healthy_hold_time, degraded_timeout, recovery_time
    )
end

const DEFAULT_HSM_CONFIG = HealthStateMachineConfig()

# ============================================================================
# Valid State Transitions
# ============================================================================

"""
    valid_health_transitions() -> Dict{HealthState, Vector{HealthState}}

Return the valid health state transitions.
Transitions not in this map are invalid.
"""
function valid_health_transitions()
    return Dict(
        HEALTH_HEALTHY => [HEALTH_DEGRADED, HEALTH_UNRELIABLE],
        HEALTH_DEGRADED => [HEALTH_HEALTHY, HEALTH_UNRELIABLE],
        HEALTH_UNRELIABLE => [HEALTH_DEGRADED]
    )
end

"""
    is_valid_transition(from::HealthState, to::HealthState) -> Bool

Check if a state transition is valid.
"""
function is_valid_transition(from::HealthState, to::HealthState)
    from == to && return true
    return to in get(valid_health_transitions(), from, HealthState[])
end

# ============================================================================
# Transition Rule Definitions
# ============================================================================

"""
    create_transition_rules(config::HealthStateMachineConfig)

Create all transition rules based on configuration.
"""
function create_transition_rules(config::HealthStateMachineConfig)
    rules = TransitionRule[]

    # HEALTHY → DEGRADED rules
    push!(rules, TransitionRule(
        "RULE-HD-1", "Fragility exceeded threshold",
        HEALTH_HEALTHY, HEALTH_DEGRADED,
        inputs -> inputs.fragility > config.fragility_degraded
    ))

    push!(rules, TransitionRule(
        "RULE-HD-2", "Confidence below threshold",
        HEALTH_HEALTHY, HEALTH_DEGRADED,
        inputs -> inputs.confidence < config.confidence_degraded
    ))

    push!(rules, TransitionRule(
        "RULE-HD-3", "Sensor failed",
        HEALTH_HEALTHY, HEALTH_DEGRADED,
        inputs -> any(s -> s == SENSOR_FAILED,
                     [inputs.dvl_status, inputs.ftm_status, inputs.depth_status, inputs.imu_status])
    ))

    push!(rules, TransitionRule(
        "RULE-HD-4", "High γ tail rate",
        HEALTH_HEALTHY, HEALTH_DEGRADED,
        inputs -> inputs.gamma_tail_rate > config.tail_rate_degraded
    ))

    push!(rules, TransitionRule(
        "RULE-HD-5", "Fragility rising rapidly",
        HEALTH_HEALTHY, HEALTH_DEGRADED,
        inputs -> inputs.is_anticipating && inputs.fragility_rate > 0.15
    ))

    # DEGRADED → HEALTHY rules
    push!(rules, TransitionRule(
        "RULE-DH-1", "All metrics recovered and stable",
        HEALTH_DEGRADED, HEALTH_HEALTHY,
        inputs -> begin
            sensors_ok = all(s -> s == SENSOR_OK,
                            [inputs.dvl_status, inputs.ftm_status, inputs.depth_status, inputs.imu_status])
            fragility_ok = inputs.fragility < config.fragility_recovery
            confidence_ok = inputs.confidence > config.confidence_recovery
            stable = inputs.time_in_state > config.healthy_hold_time
            sensors_ok && fragility_ok && confidence_ok && stable
        end
    ))

    # DEGRADED → UNRELIABLE rules
    push!(rules, TransitionRule(
        "RULE-DU-1", "Position uncertainty exceeded threshold",
        HEALTH_DEGRADED, HEALTH_UNRELIABLE,
        inputs -> inputs.position_std > config.position_std_lost
    ))

    push!(rules, TransitionRule(
        "RULE-DU-2", "Fragility critical",
        HEALTH_DEGRADED, HEALTH_UNRELIABLE,
        inputs -> inputs.fragility > config.fragility_critical
    ))

    push!(rules, TransitionRule(
        "RULE-DU-3", "Degraded timeout with multiple sensor failures",
        HEALTH_DEGRADED, HEALTH_UNRELIABLE,
        inputs -> begin
            n_failed = count(s -> s == SENSOR_FAILED,
                            [inputs.dvl_status, inputs.ftm_status, inputs.depth_status, inputs.imu_status])
            inputs.time_in_state > config.degraded_timeout && n_failed >= 2
        end
    ))

    push!(rules, TransitionRule(
        "RULE-DU-4", "Information eigenvalue critical",
        HEALTH_DEGRADED, HEALTH_UNRELIABLE,
        inputs -> inputs.lambda_min < config.lambda_critical
    ))

    # UNRELIABLE → DEGRADED rules
    push!(rules, TransitionRule(
        "RULE-UD-1", "Position and fragility recovered",
        HEALTH_UNRELIABLE, HEALTH_DEGRADED,
        inputs -> begin
            pos_ok = inputs.position_std < config.position_std_recovery
            frag_ok = inputs.fragility < config.fragility_degraded
            stable = inputs.time_in_state > config.recovery_time
            pos_ok && frag_ok && stable
        end
    ))

    return rules
end

# ============================================================================
# State Machine
# ============================================================================

"""
    HealthStateMachine

The verified health state machine.
"""
mutable struct HealthStateMachine
    config::HealthStateMachineConfig
    rules::Vector{TransitionRule}
    current_state::HealthState
    state_entry_time::Float64
    transition_log::Vector{TransitionRecord}
    max_log_size::Int
end

function HealthStateMachine(; config::HealthStateMachineConfig = DEFAULT_HSM_CONFIG,
                             max_log_size::Int = 1000)
    HealthStateMachine(
        config,
        create_transition_rules(config),
        HEALTH_HEALTHY,
        0.0,
        TransitionRecord[],
        max_log_size
    )
end

"""
    evaluate_transition!(hsm::HealthStateMachine, inputs::TransitionInputs)

Evaluate all applicable transition rules and execute first matching one.

Returns (new_state, transition_record_or_nothing).
"""
function evaluate_transition!(hsm::HealthStateMachine, inputs::TransitionInputs)
    current = hsm.current_state

    applicable = filter(r -> r.from_state == current, hsm.rules)

    for rule in applicable
        if rule.evaluate(inputs)
            old_state = hsm.current_state
            hsm.current_state = rule.to_state
            hsm.state_entry_time = inputs.current_time

            record = TransitionRecord(
                inputs.current_time, old_state, rule.to_state,
                rule.id, rule.description,
                Dict(
                    :fragility => inputs.fragility,
                    :confidence => inputs.confidence,
                    :position_std => inputs.position_std,
                    :gamma_tail_rate => inputs.gamma_tail_rate
                )
            )

            push!(hsm.transition_log, record)

            while length(hsm.transition_log) > hsm.max_log_size
                popfirst!(hsm.transition_log)
            end

            return (hsm.current_state, record)
        end
    end

    return (hsm.current_state, nothing)
end

"""
    get_current_state(hsm::HealthStateMachine, current_time::Float64)

Get current state and time in state.
"""
function get_current_state(hsm::HealthStateMachine, current_time::Float64)
    time_in_state = current_time - hsm.state_entry_time
    return (state = hsm.current_state, time_in_state = time_in_state)
end

"""
    get_nav_state(hsm::HealthStateMachine)

Get current health state.
"""
function get_nav_state(hsm::HealthStateMachine)
    return hsm.current_state
end

# ============================================================================
# Output Actions
# ============================================================================

"""
    StateAction

Actions to take in each state.
"""
struct StateAction
    reduce_speed::Bool
    inhibit_feature_promotion::Bool
    increase_logging::Bool
    emergency_mode::Bool
    recommended_action::SafeStateAction
end

"""
    get_state_actions(state::HealthState)

Get recommended actions for current state.
"""
function get_state_actions(state::HealthState)
    if state == HEALTH_HEALTHY
        return StateAction(false, false, false, false, ACTION_NONE)
    elseif state == HEALTH_DEGRADED
        return StateAction(true, true, true, false, ACTION_REDUCE_SPEED)
    else  # HEALTH_UNRELIABLE
        return StateAction(true, true, true, true, ACTION_HOLD_POSITION)
    end
end

# ============================================================================
# Transition Logging
# ============================================================================

"""
    print_transition_log(hsm::HealthStateMachine; io=stdout, last_n=10)

Print recent transition log.
"""
function print_transition_log(hsm::HealthStateMachine; io::IO = stdout, last_n::Int = 10)
    records = hsm.transition_log[max(1, end-last_n+1):end]

    println(io, "="^70)
    println(io, "Health State Machine Transition Log")
    println(io, "="^70)

    if isempty(records)
        println(io, "No transitions recorded.")
        return
    end

    for (i, r) in enumerate(records)
        println(io, "\n[$i] t=$(round(r.timestamp, digits=2))s")
        println(io, "    $(r.from_state) → $(r.to_state)")
        println(io, "    Rule: $(r.rule_id) - $(r.rule_description)")
    end

    println(io, "\n" * "="^70)
    println(io, "Current state: $(hsm.current_state)")
    println(io, "="^70)
end

"""
    export_transition_log_json(hsm::HealthStateMachine)

Export transition log as JSON-serializable Dict.
"""
function export_transition_log_json(hsm::HealthStateMachine)
    return Dict(
        "current_state" => string(hsm.current_state),
        "state_entry_time" => hsm.state_entry_time,
        "transitions" => [
            Dict(
                "timestamp" => r.timestamp,
                "from_state" => string(r.from_state),
                "to_state" => string(r.to_state),
                "rule_id" => r.rule_id,
                "rule_description" => r.rule_description,
                "inputs" => r.inputs_snapshot
            ) for r in hsm.transition_log
        ]
    )
end

# ============================================================================
# Integration Helper
# ============================================================================

"""
    build_transition_inputs(;kwargs...)

Build TransitionInputs from available data.
"""
function build_transition_inputs(;
    fragility::Float64 = 0.0,
    lambda_min::Float64 = 1.0,
    fragility_rate::Float64 = 0.0,
    is_anticipating::Bool = false,
    confidence::Float64 = 1.0,
    gamma_tail_rate::Float64 = 0.0,
    dvl_status::SensorStatus = SENSOR_OK,
    ftm_status::SensorStatus = SENSOR_OK,
    depth_status::SensorStatus = SENSOR_OK,
    imu_status::SensorStatus = SENSOR_OK,
    position_std::Float64 = 0.0,
    time_in_state::Float64 = 0.0,
    current_time::Float64 = 0.0
)
    TransitionInputs(
        fragility, lambda_min, fragility_rate, is_anticipating,
        confidence, gamma_tail_rate,
        dvl_status, ftm_status, depth_status, imu_status,
        position_std, time_in_state, current_time
    )
end
