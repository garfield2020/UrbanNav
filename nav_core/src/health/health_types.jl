# ============================================================================
# Health Monitoring Types
# ============================================================================
#
# Ported from AUV-Navigation/src/health_types.jl
#
# Defines types for per-subsystem health monitoring that integrates with
# the HealthContract state machine.
#
# HealthContract defines navigational health states (HEALTHY, DEGRADED, UNRELIABLE)
# This file defines per-subsystem monitoring states (NOMINAL, CAUTION, WARNING, CRITICAL)
# ============================================================================

using LinearAlgebra

# ============================================================================
# Monitored Subsystems
# ============================================================================

"""
    HEALTH_SUBSYSTEMS

List of monitored subsystems in the health monitor.
"""
const HEALTH_SUBSYSTEMS = [
    :covariance,      # State covariance consistency
    :innovation,      # Measurement innovation statistics
    :residual,        # Residual sequence health
    :odometry,             # Odometry sensor status
    :imu,             # IMU sensor status
    :barometer,       # Barometer sensor status
    :ftm,             # FTM sensor status
    :timing,          # Real-time timing budget
    :feasibility,     # Solution feasibility
    :observability    # Position observability (especially Y-axis)
]

# ============================================================================
# Per-Subsystem Health States
# ============================================================================

"""
    MonitoredHealthState

Per-subsystem health state (distinct from navigational HealthState).

- `NOMINAL`: All metrics within normal bounds
- `CAUTION`: Some metrics elevated, monitor closely
- `WARNING`: Significant degradation, may affect nav accuracy
- `CRITICAL`: Severe degradation, intervention needed
"""
@enum MonitoredHealthState begin
    NOMINAL = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3
end

# Aliases for backwards compatibility with existing HealthMonitor code
const DEGRADED = WARNING

# ============================================================================
# Degradation Reasons (specific flags for known limitations)
# ============================================================================

"""
    DegradationReason

Specific reasons for navigation degradation.
These flags identify known limitations that cause degraded performance.
"""
@enum DegradationReason begin
    DEGRADED_NONE = 0           # No degradation
    DEGRADED_Y_UNOBS = 1        # Y-axis position unobservable (collinear dipoles)
    DEGRADED_X_UNOBS = 2        # X-axis position unobservable
    DEGRADED_YAW_UNOBS = 3      # Yaw unobservable (no compass updates)
    DEGRADED_WEAK_GRADIENT = 4  # Magnetic gradient too weak for updates
    DEGRADED_SENSOR_GAP = 5     # Gap in sensor measurements
    DEGRADED_MAP_UNINFORMATIVE = 6  # Frozen map gradient energy too low (Phase A)
    DEGRADED_MAP_EXTRAPOLATION = 7  # Position outside frozen map coverage (Phase A)
    DEGRADED_MAP_RESIDUALS = 8      # Repeated large magnetic residuals (Phase A)
end

"""
    DegradationFlags

Bitfield-style container for multiple simultaneous degradation reasons.
"""
struct DegradationFlags
    flags::Set{DegradationReason}
end

DegradationFlags() = DegradationFlags(Set{DegradationReason}())

"""Check if a specific degradation flag is set."""
function has_degradation(df::DegradationFlags, reason::DegradationReason)
    return reason in df.flags
end

"""Add a degradation reason."""
function add_degradation(df::DegradationFlags, reason::DegradationReason)
    new_flags = copy(df.flags)
    push!(new_flags, reason)
    return DegradationFlags(new_flags)
end

"""Remove a degradation reason."""
function remove_degradation(df::DegradationFlags, reason::DegradationReason)
    new_flags = copy(df.flags)
    delete!(new_flags, reason)
    return DegradationFlags(new_flags)
end

"""Check if any degradation flags are set."""
function is_degraded(df::DegradationFlags)
    return !isempty(df.flags)
end

"""Get list of all active degradation reasons."""
function get_degradations(df::DegradationFlags)
    return collect(df.flags)
end

"""Check if transition between monitored states is valid."""
function is_valid_health_transition(from::MonitoredHealthState, to::MonitoredHealthState)
    # Allow any transition (monitored states can change quickly based on metrics)
    return true
end

# ============================================================================
# Health Transition Record
# ============================================================================

"""
    HealthTransition

Record of a health state transition.
"""
struct HealthTransition
    from_state::MonitoredHealthState
    to_state::MonitoredHealthState
    timestamp::Float64
    reason::String
    subsystem::Symbol
end

# ============================================================================
# Health Report
# ============================================================================

"""
    HealthReport

Summary of current health status across all subsystems.

# Fields
- `overall_state`: Worst state across all subsystems
- `subsystem_states`: State of each subsystem
- `metrics`: Current metric values
- `timestamp`: Report generation time
- `recent_transitions`: Recent state changes
"""
struct HealthReport
    overall_state::MonitoredHealthState
    subsystem_states::Dict{Symbol, MonitoredHealthState}
    metrics::Dict{Symbol, Float64}
    timestamp::Float64
    recent_transitions::Vector{HealthTransition}
end

"""Check if overall health is acceptable for normal operation."""
function is_healthy(report::HealthReport)
    return report.overall_state <= CAUTION
end

"""Get subsystems in degraded or worse state."""
function get_degraded_subsystems(report::HealthReport)
    return [k for (k, v) in report.subsystem_states if v >= WARNING]
end

# ============================================================================
# Health Metrics
# ============================================================================

"""
    HealthMetrics

Collection of health metrics for aggregation.
"""
struct HealthMetrics
    position_uncertainty::Float64    # Position std [m]
    velocity_uncertainty::Float64    # Velocity std [m/s]
    condition_number::Float64        # Covariance condition number
    fragility::Float64               # Navigation fragility
    chi2_mean::Float64               # Mean χ² statistic
    outlier_rate::Float64            # Fraction of outliers
    timing_margin::Float64           # Real-time timing margin
    sensor_count::Int                # Number of active sensors
end

function HealthMetrics(;
    position_uncertainty::Real = 0.0,
    velocity_uncertainty::Real = 0.0,
    condition_number::Real = 1.0,
    fragility::Real = 0.0,
    chi2_mean::Real = 1.0,
    outlier_rate::Real = 0.0,
    timing_margin::Real = 1.0,
    sensor_count::Int = 4
)
    HealthMetrics(
        Float64(position_uncertainty),
        Float64(velocity_uncertainty),
        Float64(condition_number),
        Float64(fragility),
        Float64(chi2_mean),
        Float64(outlier_rate),
        Float64(timing_margin),
        sensor_count
    )
end

"""Convert HealthMetrics to Dict for storage."""
function metrics_to_dict(m::HealthMetrics)
    return Dict{Symbol, Float64}(
        :position_uncertainty => m.position_uncertainty,
        :velocity_uncertainty => m.velocity_uncertainty,
        :condition_number => m.condition_number,
        :fragility => m.fragility,
        :chi2_mean => m.chi2_mean,
        :outlier_rate => m.outlier_rate,
        :timing_margin => m.timing_margin,
        :sensor_count => Float64(m.sensor_count)
    )
end

# ============================================================================
# Mapping to Navigational Health
# ============================================================================

"""
    map_to_nav_health(report::HealthReport) -> HealthState

Map per-subsystem health report to navigational HealthState.

This bridges the monitoring layer to the HealthContract state machine.
"""
function map_to_nav_health(report::HealthReport)
    overall = report.overall_state

    if overall == NOMINAL
        return HEALTH_HEALTHY
    elseif overall == CAUTION
        return HEALTH_HEALTHY  # Caution doesn't trigger nav degradation
    elseif overall == WARNING
        return HEALTH_DEGRADED
    else  # CRITICAL
        return HEALTH_UNRELIABLE
    end
end

"""
    build_transition_inputs_from_report(report::HealthReport, config...) -> TransitionInputs

Build TransitionInputs for HealthStateMachine from a HealthReport.
"""
function build_transition_inputs_from_report(report::HealthReport;
                                               lambda_min::Float64 = 1e-6,
                                               gamma_tail_rate::Float64 = 0.0,
                                               is_anticipating::Bool = false)
    m = report.metrics

    # Extract metrics with defaults
    position_std = get(m, :position_uncertainty, 0.0)
    fragility = get(m, :fragility, 0.0)
    fragility_rate = 0.0  # Would need history to compute

    # Map sensor states to SensorStatus
    function subsystem_to_sensor(s::MonitoredHealthState)
        if s == NOMINAL
            return SENSOR_OK
        elseif s <= WARNING
            return SENSOR_DEGRADED
        else
            return SENSOR_FAILED
        end
    end

    odometry_status = subsystem_to_sensor(get(report.subsystem_states, :odometry, NOMINAL))
    ftm_status = subsystem_to_sensor(get(report.subsystem_states, :ftm, NOMINAL))
    depth_status = subsystem_to_sensor(get(report.subsystem_states, :barometer, NOMINAL))
    imu_status = subsystem_to_sensor(get(report.subsystem_states, :imu, NOMINAL))

    # Confidence from overall state
    confidence = if report.overall_state == NOMINAL
        1.0
    elseif report.overall_state == CAUTION
        0.8
    elseif report.overall_state == WARNING
        0.5
    else
        0.2
    end

    return build_transition_inputs(
        fragility = fragility,
        lambda_min = lambda_min,
        fragility_rate = fragility_rate,
        is_anticipating = is_anticipating,
        confidence = confidence,
        gamma_tail_rate = gamma_tail_rate,
        odometry_status = odometry_status,
        ftm_status = ftm_status,
        depth_status = depth_status,
        imu_status = imu_status,
        position_std = position_std,
        current_time = report.timestamp
    )
end

# ============================================================================
# Exports
# ============================================================================

export MonitoredHealthState, NOMINAL, CAUTION, WARNING, CRITICAL, DEGRADED
export HEALTH_SUBSYSTEMS
export is_valid_health_transition
export HealthTransition
export HealthReport, is_healthy, get_degraded_subsystems
export HealthMetrics, metrics_to_dict
export map_to_nav_health, build_transition_inputs_from_report
export DegradationReason, DEGRADED_NONE, DEGRADED_Y_UNOBS, DEGRADED_X_UNOBS
export DEGRADED_YAW_UNOBS, DEGRADED_WEAK_GRADIENT, DEGRADED_SENSOR_GAP
export DEGRADED_MAP_UNINFORMATIVE, DEGRADED_MAP_EXTRAPOLATION, DEGRADED_MAP_RESIDUALS
export DegradationFlags, has_degradation, add_degradation, remove_degradation
export is_degraded, get_degradations
