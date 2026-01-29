# ============================================================================
# observability_classification.jl - Observability Failure Classification
# ============================================================================
#
# V1.0 Qualification Closure Plan Step 6:
# Classify different types of observability failures.
#
# Key insight: Not all observability losses are equal:
# 1. EXPECTED: Known physics limitations (Y-axis during straight line)
# 2. UNEXPECTED: System failures that should trigger health alerts
# 3. RECOVERABLE: Temporary loss during maneuvers
# 4. PERSISTENT: Fundamental sensor/trajectory limitations
#
# This classification enables:
# - Proper handling of "expected" failures (not counted against qualification)
# - Clear root cause identification for unexpected failures
# - Guidance for trajectory planning to maintain observability
# ============================================================================

using LinearAlgebra
using Statistics

export ObservabilityFailureCategory
export OBS_EXPECTED_TRAJECTORY, OBS_EXPECTED_SENSOR
export OBS_EXPECTED_ENVIRONMENT, OBS_UNEXPECTED_SYSTEM
export OBS_UNEXPECTED_UNKNOWN, OBS_RECOVERABLE, OBS_PERSISTENT

export ObservabilityFailureClass
export OBSFAIL_Y_AXIS_STRAIGHT, OBSFAIL_HEADING_STATIC
export OBSFAIL_Odometry_DROPOUT, OBSFAIL_COMPASS_FAILURE
export OBSFAIL_WEAK_GRADIENT, OBSFAIL_MAGNETIC_INTERFERENCE
export OBSFAIL_FILTER_DIVERGENCE, OBSFAIL_UNKNOWN

export ObservabilityFailure, ObservabilityClassifier
export ObservabilityClassificationConfig, DEFAULT_OBS_CLASS_CONFIG
export classify_observability_failure, is_expected_failure
export get_failure_mitigation, get_failure_explanation
export ObservabilityFailureReport, generate_failure_report
export format_observability_classification
export KnownLimitation, V1_0_KNOWN_LIMITATIONS
export is_known_limitation, get_known_limitation

# ============================================================================
# Failure Categories
# ============================================================================

"""
    ObservabilityFailureCategory

High-level category of observability failure.
"""
@enum ObservabilityFailureCategory begin
    OBS_EXPECTED_TRAJECTORY = 1   # Known physics (Y-axis in straight line)
    OBS_EXPECTED_SENSOR = 2       # Expected sensor limitation
    OBS_EXPECTED_ENVIRONMENT = 3  # Known environmental limitation
    OBS_UNEXPECTED_SYSTEM = 4     # System failure (should trigger alert)
    OBS_UNEXPECTED_UNKNOWN = 5    # Unknown cause (investigate)
    OBS_RECOVERABLE = 6           # Temporary, will recover
    OBS_PERSISTENT = 7            # Fundamental limitation
end

"""
    ObservabilityFailureClass

Specific type of observability failure.
"""
@enum ObservabilityFailureClass begin
    # Trajectory-induced (expected)
    OBSFAIL_Y_AXIS_STRAIGHT = 1      # Y-axis unobservable during straight line
    OBSFAIL_HEADING_STATIC = 2       # Heading unobservable when stationary
    OBSFAIL_HOVER_POSITION = 3       # Position drift during hover

    # Sensor-induced
    OBSFAIL_Odometry_DROPOUT = 10         # Odometry sensor dropout
    OBSFAIL_COMPASS_FAILURE = 11     # Compass/heading sensor failure
    OBSFAIL_BAROMETER_FAILURE = 12       # Depth sensor failure
    OBSFAIL_FTM_DROPOUT = 13         # Magnetometer dropout

    # Environment-induced
    OBSFAIL_WEAK_GRADIENT = 20       # Weak magnetic gradient
    OBSFAIL_MAGNETIC_INTERFERENCE = 21  # External interference
    OBSFAIL_UNIFORM_FIELD = 22       # Uniform field (no information)

    # System failures (unexpected)
    OBSFAIL_FILTER_DIVERGENCE = 30   # Filter state diverged
    OBSFAIL_COVARIANCE_SINGULAR = 31 # Covariance became singular
    OBSFAIL_NUMERICAL_INSTABILITY = 32  # Numerical issues

    # Unknown
    OBSFAIL_UNKNOWN = 99
end

# ============================================================================
# Failure Definition
# ============================================================================

"""
    ObservabilityFailure

A detected observability failure with classification.

# Fields
- `class_type`: Specific failure class
- `category`: High-level category
- `affected_states`: Which states lost observability [1=x, 2=y, 3=z, ...]
- `severity`: 0.0-1.0 (0=minor, 1=complete loss)
- `is_expected`: Whether this was a known/expected limitation
- `is_recoverable`: Whether system can recover
- `trigger_time`: When the failure was detected
- `trigger_condition`: What caused the failure
- `confidence`: Confidence in classification (0-1)
"""
struct ObservabilityFailure
    class_type::ObservabilityFailureClass
    category::ObservabilityFailureCategory
    affected_states::Vector{Int}
    severity::Float64
    is_expected::Bool
    is_recoverable::Bool
    trigger_time::Float64
    trigger_condition::String
    confidence::Float64
end

function ObservabilityFailure(;
    class_type::ObservabilityFailureClass = OBSFAIL_UNKNOWN,
    category::ObservabilityFailureCategory = OBS_UNEXPECTED_UNKNOWN,
    affected_states::Vector{Int} = Int[],
    severity::Float64 = 1.0,
    is_expected::Bool = false,
    is_recoverable::Bool = true,
    trigger_time::Float64 = 0.0,
    trigger_condition::String = "Unknown",
    confidence::Float64 = 0.5
)
    ObservabilityFailure(
        class_type, category, affected_states, severity,
        is_expected, is_recoverable, trigger_time, trigger_condition, confidence
    )
end

# ============================================================================
# Known Limitations (V1.0)
# ============================================================================

"""
    KnownLimitation

A documented, expected limitation of the navigation system.
"""
struct KnownLimitation
    id::String
    class_type::ObservabilityFailureClass
    description::String
    affected_states::Vector{Int}
    trigger_conditions::Vector{String}
    mitigation::String
    documentation_ref::String
end

"""
    V1_0_KNOWN_LIMITATIONS

Documented known limitations for V1.0 qualification.

These are expected behaviors that don't count as failures.
"""
const V1_0_KNOWN_LIMITATIONS = [
    KnownLimitation(
        "KL-001",
        OBSFAIL_Y_AXIS_STRAIGHT,
        "Y-axis position becomes unobservable during straight-line motion",
        [2],  # Y-axis (state 2)
        ["trajectory_type == STRAIGHT", "heading_rate < 0.01 rad/s"],
        "Execute periodic heading changes (>5°) every 60 seconds",
        "V1.0 Qualification Report Section 4.2"
    ),
    KnownLimitation(
        "KL-002",
        OBSFAIL_HEADING_STATIC,
        "Heading becomes unobservable during hover/static conditions",
        [7, 8, 9],  # Attitude states
        ["velocity < 0.1 m/s", "angular_rate < 0.01 rad/s"],
        "Maintain minimum velocity of 0.2 m/s or use compass",
        "V1.0 Qualification Report Section 4.3"
    ),
    KnownLimitation(
        "KL-003",
        OBSFAIL_Odometry_DROPOUT,
        "Odometry dropout during tracking-loss or high speed",
        [4, 5, 6],  # Velocity states
        ["altitude > 200m", "bottom_lock == false"],
        "Reduce altitude or switch to inertial-only mode",
        "V1.0 Qualification Report Section 4.4"
    ),
    KnownLimitation(
        "KL-004",
        OBSFAIL_WEAK_GRADIENT,
        "Weak magnetic gradient reduces position observability",
        [1, 2, 3],  # Position states
        ["gradient_strength < 0.5 nT/m"],
        "Increase measurement integration time or proceed with caution",
        "V1.0 Qualification Report Section 4.5"
    ),
]

"""
    is_known_limitation(failure::ObservabilityFailure) -> Bool

Check if a failure matches a known limitation.
"""
function is_known_limitation(failure::ObservabilityFailure)
    for kl in V1_0_KNOWN_LIMITATIONS
        if failure.class_type == kl.class_type
            return true
        end
    end
    return false
end

"""
    get_known_limitation(failure::ObservabilityFailure) -> Union{KnownLimitation, Nothing}

Get the matching known limitation if any.
"""
function get_known_limitation(failure::ObservabilityFailure)
    for kl in V1_0_KNOWN_LIMITATIONS
        if failure.class_type == kl.class_type
            return kl
        end
    end
    return nothing
end

# ============================================================================
# Classification Configuration
# ============================================================================

"""
    ObservabilityClassificationConfig

Configuration for the observability classifier.

# Fields
- `straight_line_heading_threshold`: Max heading rate for "straight" (rad/s)
- `static_velocity_threshold`: Max velocity for "static" (m/s)
- `weak_gradient_threshold`: Gradient below this is "weak" (nT/m)
- `condition_number_threshold`: Condition number above this is "poorly conditioned"
- `singular_value_threshold`: Singular values below this are "zero"
- `recovery_timeout`: Max time for recoverable failures (s)
"""
Base.@kwdef struct ObservabilityClassificationConfig
    straight_line_heading_threshold::Float64 = 0.01   # rad/s
    static_velocity_threshold::Float64 = 0.1          # m/s
    weak_gradient_threshold::Float64 = 0.5            # nT/m
    condition_number_threshold::Float64 = 1e6
    singular_value_threshold::Float64 = 1e-10
    recovery_timeout::Float64 = 30.0                  # seconds
end

const DEFAULT_OBS_CLASS_CONFIG = ObservabilityClassificationConfig()

# ============================================================================
# Observability Classifier
# ============================================================================

"""
    ObservabilityClassifierState

Internal state for the classifier.
"""
mutable struct ObservabilityClassifierState
    # History of observable dimensions
    observable_history::Vector{Vector{Bool}}
    timestamps::Vector{Float64}

    # Current failure tracking
    active_failures::Vector{ObservabilityFailure}
    failure_start_times::Dict{ObservabilityFailureClass, Float64}

    # Trajectory state
    current_heading_rate::Float64
    current_velocity::Float64
    current_gradient_strength::Float64

    # Sensor state
    odometry_available::Bool
    compass_available::Bool
    depth_available::Bool
end

function ObservabilityClassifierState()
    ObservabilityClassifierState(
        Vector{Bool}[],
        Float64[],
        ObservabilityFailure[],
        Dict{ObservabilityFailureClass, Float64}(),
        0.0, 0.0, 10.0,
        true, true, true
    )
end

"""
    ObservabilityClassifier

Classifier for observability failures.
"""
struct ObservabilityClassifier
    config::ObservabilityClassificationConfig
    state::ObservabilityClassifierState
end

function ObservabilityClassifier(config::ObservabilityClassificationConfig = DEFAULT_OBS_CLASS_CONFIG)
    ObservabilityClassifier(config, ObservabilityClassifierState())
end

"""
    update_classifier!(classifier, obs_metrics, trajectory_state, sensor_state, timestamp)

Update classifier with current system state.

# Arguments
- `obs_metrics`: ObservabilityMetrics from DOE module
- `trajectory_state`: (heading_rate, velocity_magnitude)
- `sensor_state`: (odometry_available, compass_available, depth_available)
- `timestamp`: Current time
"""
function update_classifier!(classifier::ObservabilityClassifier,
                            obs_metrics::ObservabilityMetrics,
                            trajectory_state::Tuple{Float64, Float64},
                            sensor_state::Tuple{Bool, Bool, Bool},
                            timestamp::Float64)
    state = classifier.state
    config = classifier.config

    # Update trajectory state
    state.current_heading_rate = trajectory_state[1]
    state.current_velocity = trajectory_state[2]

    # Update sensor state
    state.odometry_available = sensor_state[1]
    state.compass_available = sensor_state[2]
    state.depth_available = sensor_state[3]

    # Update observable history
    push!(state.observable_history, obs_metrics.observable_dims)
    push!(state.timestamps, timestamp)

    # Trim history
    max_history = 100
    if length(state.observable_history) > max_history
        deleteat!(state.observable_history, 1)
        deleteat!(state.timestamps, 1)
    end
end

"""
    classify_observability_failure(classifier, obs_metrics, gradient_strength) -> ObservabilityFailure

Classify the current observability state.

Returns OBSFAIL_UNKNOWN with severity=0 if no failure detected.
"""
function classify_observability_failure(classifier::ObservabilityClassifier,
                                        obs_metrics::ObservabilityMetrics,
                                        gradient_strength::Float64 = 10.0)
    state = classifier.state
    config = classifier.config

    state.current_gradient_strength = gradient_strength

    # Find unobservable dimensions
    unobservable = findall(.!obs_metrics.observable_dims)

    if isempty(unobservable)
        # No failure
        return ObservabilityFailure(
            class_type = OBSFAIL_UNKNOWN,
            category = OBS_EXPECTED_TRAJECTORY,
            affected_states = Int[],
            severity = 0.0,
            is_expected = true,
            is_recoverable = true,
            trigger_time = isempty(state.timestamps) ? 0.0 : state.timestamps[end],
            trigger_condition = "Fully observable",
            confidence = 1.0
        )
    end

    # Classify based on affected states and conditions
    timestamp = isempty(state.timestamps) ? 0.0 : state.timestamps[end]

    # Check for Y-axis unobservability during straight line
    if 2 in unobservable && state.current_heading_rate < config.straight_line_heading_threshold
        return ObservabilityFailure(
            class_type = OBSFAIL_Y_AXIS_STRAIGHT,
            category = OBS_EXPECTED_TRAJECTORY,
            affected_states = [2],
            severity = 0.8,
            is_expected = true,
            is_recoverable = true,
            trigger_time = timestamp,
            trigger_condition = "Straight-line motion (heading_rate=$(round(state.current_heading_rate, digits=4)) rad/s)",
            confidence = 0.95
        )
    end

    # Check for heading unobservability during static
    attitude_states = [7, 8, 9]
    if any(s in unobservable for s in attitude_states) && state.current_velocity < config.static_velocity_threshold
        return ObservabilityFailure(
            class_type = OBSFAIL_HEADING_STATIC,
            category = OBS_EXPECTED_TRAJECTORY,
            affected_states = intersect(unobservable, attitude_states),
            severity = 0.6,
            is_expected = true,
            is_recoverable = true,
            trigger_time = timestamp,
            trigger_condition = "Static/hover (velocity=$(round(state.current_velocity, digits=2)) m/s)",
            confidence = 0.90
        )
    end

    # Check for Odometry dropout
    velocity_states = [4, 5, 6]
    if !state.odometry_available && any(s in unobservable for s in velocity_states)
        return ObservabilityFailure(
            class_type = OBSFAIL_Odometry_DROPOUT,
            category = OBS_EXPECTED_SENSOR,
            affected_states = intersect(unobservable, velocity_states),
            severity = 0.9,
            is_expected = true,  # Odometry dropout is expected in some conditions
            is_recoverable = true,
            trigger_time = timestamp,
            trigger_condition = "Odometry unavailable",
            confidence = 0.95
        )
    end

    # Check for compass failure
    if !state.compass_available && any(s in unobservable for s in attitude_states)
        return ObservabilityFailure(
            class_type = OBSFAIL_COMPASS_FAILURE,
            category = OBS_EXPECTED_SENSOR,
            affected_states = intersect(unobservable, attitude_states),
            severity = 0.7,
            is_expected = true,
            is_recoverable = true,
            trigger_time = timestamp,
            trigger_condition = "Compass unavailable",
            confidence = 0.90
        )
    end

    # Check for weak gradient
    position_states = [1, 2, 3]
    if gradient_strength < config.weak_gradient_threshold && any(s in unobservable for s in position_states)
        return ObservabilityFailure(
            class_type = OBSFAIL_WEAK_GRADIENT,
            category = OBS_EXPECTED_ENVIRONMENT,
            affected_states = intersect(unobservable, position_states),
            severity = 0.7,
            is_expected = true,
            is_recoverable = true,
            trigger_time = timestamp,
            trigger_condition = "Weak gradient ($(round(gradient_strength, digits=2)) nT/m)",
            confidence = 0.85
        )
    end

    # Check for numerical issues (condition number)
    if obs_metrics.condition_number > config.condition_number_threshold
        return ObservabilityFailure(
            class_type = OBSFAIL_COVARIANCE_SINGULAR,
            category = OBS_UNEXPECTED_SYSTEM,
            affected_states = unobservable,
            severity = 0.95,
            is_expected = false,
            is_recoverable = false,
            trigger_time = timestamp,
            trigger_condition = "Condition number=$(round(obs_metrics.condition_number, sigdigits=2))",
            confidence = 0.85
        )
    end

    # Check for filter divergence (all position unobservable unexpectedly)
    if all(s in unobservable for s in position_states) &&
       state.current_heading_rate > config.straight_line_heading_threshold
        return ObservabilityFailure(
            class_type = OBSFAIL_FILTER_DIVERGENCE,
            category = OBS_UNEXPECTED_SYSTEM,
            affected_states = position_states,
            severity = 1.0,
            is_expected = false,
            is_recoverable = false,
            trigger_time = timestamp,
            trigger_condition = "Full position unobservability during maneuvering",
            confidence = 0.80
        )
    end

    # Unknown failure
    return ObservabilityFailure(
        class_type = OBSFAIL_UNKNOWN,
        category = OBS_UNEXPECTED_UNKNOWN,
        affected_states = unobservable,
        severity = 0.8,
        is_expected = false,
        is_recoverable = true,
        trigger_time = timestamp,
        trigger_condition = "Unclassified observability loss",
        confidence = 0.50
    )
end

"""
    is_expected_failure(failure::ObservabilityFailure) -> Bool

Check if a failure is expected/documented.
"""
function is_expected_failure(failure::ObservabilityFailure)
    failure.is_expected || is_known_limitation(failure)
end

# ============================================================================
# Mitigation and Explanation
# ============================================================================

"""
    get_failure_mitigation(failure::ObservabilityFailure) -> String

Get recommended mitigation for a failure.
"""
function get_failure_mitigation(failure::ObservabilityFailure)
    # Check known limitations first
    kl = get_known_limitation(failure)
    if kl !== nothing
        return kl.mitigation
    end

    # Default mitigations by class
    mitigations = Dict(
        OBSFAIL_Y_AXIS_STRAIGHT => "Execute heading change (>5°) to restore Y-axis observability",
        OBSFAIL_HEADING_STATIC => "Increase velocity to >0.2 m/s or rely on compass",
        OBSFAIL_HOVER_POSITION => "Begin forward motion to restore velocity observability",
        OBSFAIL_Odometry_DROPOUT => "Reduce altitude or switch to Odometry-less mode",
        OBSFAIL_COMPASS_FAILURE => "Maintain maneuvering to preserve heading observability",
        OBSFAIL_BAROMETER_FAILURE => "Use magnetic vertical gradient for altitude",
        OBSFAIL_FTM_DROPOUT => "Switch to dead reckoning mode",
        OBSFAIL_WEAK_GRADIENT => "Navigate to stronger gradient region or extend integration",
        OBSFAIL_MAGNETIC_INTERFERENCE => "Distance from interference source",
        OBSFAIL_UNIFORM_FIELD => "Navigate to non-uniform region",
        OBSFAIL_FILTER_DIVERGENCE => "EMERGENCY: Surface and recalibrate",
        OBSFAIL_COVARIANCE_SINGULAR => "Reset filter covariance to initial values",
        OBSFAIL_NUMERICAL_INSTABILITY => "Reduce measurement rate or check sensor data",
        OBSFAIL_UNKNOWN => "Investigate root cause before continuing"
    )

    get(mitigations, failure.class_type, "No mitigation available")
end

"""
    get_failure_explanation(failure::ObservabilityFailure) -> String

Get human-readable explanation for a failure.
"""
function get_failure_explanation(failure::ObservabilityFailure)
    explanations = Dict(
        OBSFAIL_Y_AXIS_STRAIGHT => """
            During straight-line motion, the vehicle's Y-axis (cross-track) position
            becomes unobservable from magnetic measurements. This is a fundamental
            physics limitation: magnetic gradients only provide information along
            the direction of motion. This is a KNOWN LIMITATION of the system.
            """,
        OBSFAIL_HEADING_STATIC => """
            When the vehicle is stationary or moving very slowly, heading becomes
            unobservable from magnetic gradients. The magnetic field provides
            position information, but heading must be derived from motion or
            external sensors (compass).
            """,
        OBSFAIL_Odometry_DROPOUT => """
            Odometry (Wheel/Visual Odometry) has lost tracking lock, typically due to
            altitude exceeding range or rough bottom conditions. Velocity
            measurements are unavailable, increasing position uncertainty.
            """,
        OBSFAIL_WEAK_GRADIENT => """
            The magnetic field gradient is too weak to provide reliable position
            information. The vehicle is in a magnetically "flat" region where
            position changes produce minimal field changes.
            """,
        OBSFAIL_FILTER_DIVERGENCE => """
            WARNING: The navigation filter has diverged. Estimated state no longer
            tracks true state. Immediate intervention required.
            """,
        OBSFAIL_UNKNOWN => """
            An unclassified observability loss has occurred. Further investigation
            is needed to determine the root cause.
            """
    )

    explanation = get(explanations, failure.class_type, "No explanation available")
    return strip(explanation)
end

# ============================================================================
# Failure Report
# ============================================================================

"""
    ObservabilityFailureReport

Report summarizing observability failures for a scenario.
"""
struct ObservabilityFailureReport
    scenario_name::String
    total_failures::Int
    expected_failures::Int
    unexpected_failures::Int

    # By category
    by_category::Dict{ObservabilityFailureCategory, Int}

    # By class
    by_class::Dict{ObservabilityFailureClass, Int}

    # Individual failures
    failures::Vector{ObservabilityFailure}

    # Statistics
    total_unobservable_time::Float64
    max_severity::Float64
    primary_failure_class::ObservabilityFailureClass

    # Verdict
    qualifies::Bool  # True if only expected failures
    issues::Vector{String}
end

"""
    generate_failure_report(failures::Vector{ObservabilityFailure}, scenario_name::String) -> ObservabilityFailureReport

Generate a failure report from a list of failures.
"""
function generate_failure_report(failures::Vector{ObservabilityFailure},
                                  scenario_name::String)
    n = length(failures)

    if n == 0
        return ObservabilityFailureReport(
            scenario_name, 0, 0, 0,
            Dict{ObservabilityFailureCategory, Int}(),
            Dict{ObservabilityFailureClass, Int}(),
            ObservabilityFailure[],
            0.0, 0.0, OBSFAIL_UNKNOWN,
            true, String[]
        )
    end

    expected = count(f -> is_expected_failure(f), failures)
    unexpected = n - expected

    # Count by category
    by_category = Dict{ObservabilityFailureCategory, Int}()
    for f in failures
        by_category[f.category] = get(by_category, f.category, 0) + 1
    end

    # Count by class
    by_class = Dict{ObservabilityFailureClass, Int}()
    for f in failures
        by_class[f.class_type] = get(by_class, f.class_type, 0) + 1
    end

    # Statistics
    max_severity = maximum(f.severity for f in failures)

    # Primary failure class (most common)
    primary_class = if isempty(by_class)
        OBSFAIL_UNKNOWN
    else
        argmax(by_class)
    end

    # Estimate unobservable time (simplified)
    total_unobs_time = sum(f.severity for f in failures)  # Proxy

    # Issues
    issues = String[]
    if unexpected > 0
        push!(issues, "$unexpected unexpected observability failures detected")
    end
    for f in failures
        if !is_expected_failure(f) && f.severity > 0.8
            push!(issues, "High-severity unexpected failure: $(f.class_type)")
        end
    end

    qualifies = unexpected == 0

    ObservabilityFailureReport(
        scenario_name, n, expected, unexpected,
        by_category, by_class, failures,
        total_unobs_time, max_severity, primary_class,
        qualifies, issues
    )
end

"""
    format_observability_classification(report::ObservabilityFailureReport) -> String

Format an observability failure report.
"""
function format_observability_classification(report::ObservabilityFailureReport)
    lines = String[]

    push!(lines, "=" ^ 70)
    push!(lines, "OBSERVABILITY FAILURE CLASSIFICATION REPORT")
    push!(lines, "=" ^ 70)
    push!(lines, "")

    push!(lines, "Scenario: $(report.scenario_name)")
    push!(lines, "Status: $(report.qualifies ? "QUALIFIES" : "DOES NOT QUALIFY")")
    push!(lines, "")

    push!(lines, "-" ^ 70)
    push!(lines, "SUMMARY")
    push!(lines, "-" ^ 70)
    push!(lines, "Total failures: $(report.total_failures)")
    push!(lines, "  Expected: $(report.expected_failures)")
    push!(lines, "  Unexpected: $(report.unexpected_failures)")
    push!(lines, "Max severity: $(round(report.max_severity, digits=2))")
    push!(lines, "Primary failure type: $(report.primary_failure_class)")
    push!(lines, "")

    if !isempty(report.by_category)
        push!(lines, "-" ^ 70)
        push!(lines, "BY CATEGORY")
        push!(lines, "-" ^ 70)
        for (cat, count) in sort(collect(report.by_category), by=x->x[2], rev=true)
            push!(lines, "  $(cat): $count")
        end
        push!(lines, "")
    end

    if !isempty(report.by_class)
        push!(lines, "-" ^ 70)
        push!(lines, "BY FAILURE CLASS")
        push!(lines, "-" ^ 70)
        for (cls, count) in sort(collect(report.by_class), by=x->x[2], rev=true)
            is_exp = cls in [f.class_type for f in V1_0_KNOWN_LIMITATIONS]
            exp_mark = is_exp ? " [KNOWN]" : ""
            push!(lines, "  $(cls): $count$exp_mark")
        end
        push!(lines, "")
    end

    if !isempty(report.issues)
        push!(lines, "-" ^ 70)
        push!(lines, "ISSUES")
        push!(lines, "-" ^ 70)
        for issue in report.issues
            push!(lines, "  - $issue")
        end
        push!(lines, "")
    end

    # Show first few failures with details
    if !isempty(report.failures)
        push!(lines, "-" ^ 70)
        push!(lines, "FAILURE DETAILS (first 5)")
        push!(lines, "-" ^ 70)
        for (i, f) in enumerate(report.failures[1:min(5, length(report.failures))])
            exp_mark = is_expected_failure(f) ? "[EXPECTED]" : "[UNEXPECTED]"
            push!(lines, "[$i] $(f.class_type) $exp_mark")
            push!(lines, "    Category: $(f.category)")
            push!(lines, "    Severity: $(round(f.severity, digits=2))")
            push!(lines, "    Affected states: $(f.affected_states)")
            push!(lines, "    Trigger: $(f.trigger_condition)")
            push!(lines, "    Mitigation: $(get_failure_mitigation(f))")
            push!(lines, "")
        end
    end

    push!(lines, "=" ^ 70)

    join(lines, "\n")
end

# ============================================================================
# Integration with Qualification
# ============================================================================

export ObservabilityQualificationResult
export evaluate_observability_qualification

"""
    ObservabilityQualificationResult

Result of observability qualification for a scenario.
"""
struct ObservabilityQualificationResult
    scenario_name::String
    passed::Bool
    failure_report::ObservabilityFailureReport

    # Key metrics
    total_time::Float64
    observable_time::Float64
    observable_fraction::Float64

    # Known limitations encountered
    known_limitations_triggered::Vector{String}

    # Summary
    summary::String
end

"""
    evaluate_observability_qualification(failures, scenario_name, total_time) -> ObservabilityQualificationResult

Evaluate whether a scenario passes observability qualification.

A scenario passes if:
1. All failures are expected/known limitations, OR
2. Unexpected failures are recoverable and don't exceed thresholds
"""
function evaluate_observability_qualification(failures::Vector{ObservabilityFailure},
                                               scenario_name::String,
                                               total_time::Float64)
    report = generate_failure_report(failures, scenario_name)

    # Estimate observable time
    unobs_time = sum(f.severity * 1.0 for f in failures; init=0.0)  # Simplified
    obs_time = max(0.0, total_time - unobs_time)
    obs_fraction = total_time > 0 ? obs_time / total_time : 1.0

    # Find known limitations triggered
    kl_triggered = String[]
    for f in failures
        kl = get_known_limitation(f)
        if kl !== nothing && !(kl.id in kl_triggered)
            push!(kl_triggered, kl.id)
        end
    end

    # Determine pass/fail
    passed = report.qualifies

    # Generate summary
    summary = if passed
        if report.total_failures == 0
            "PASS: Fully observable throughout scenario"
        else
            "PASS: $(report.expected_failures) expected observability limitations encountered ($(join(kl_triggered, ", ")))"
        end
    else
        "FAIL: $(report.unexpected_failures) unexpected observability failures"
    end

    ObservabilityQualificationResult(
        scenario_name, passed, report,
        total_time, obs_time, obs_fraction,
        kl_triggered, summary
    )
end
