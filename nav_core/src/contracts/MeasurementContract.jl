# ============================================================================
# MeasurementContract.jl - Authoritative definition of measurement interfaces
# ============================================================================
#
# Ported from AUV-Navigation/src/interfaces.jl and src/sensors.jl
#
# This contract defines WHAT each sensor must provide:
# - Measurement value and type
# - Units and frame
# - Covariance structure
# - Timestamp and validity
#
# All sensor implementations must produce measurements conforming to this.
# ============================================================================

using LinearAlgebra
using StaticArrays

export AbstractMeasurement, AbstractMagneticField, AbstractSensor, AbstractEstimator
export AbstractResidualManager, ScenarioConfig, ScenarioResult
export field_at, gradient_at, field_at_body, gradient_at_body
export measure, measurement_covariance, measurement_dimension
export transform_field_to_body, transform_field_to_world
export transform_gradient_to_body, transform_gradient_to_world

# ============================================================================
# Abstract Types
# ============================================================================

"""
    AbstractMeasurement

Base type for all measurements in the system.
Every sensor produces measurements that extend this type.
"""
abstract type AbstractMeasurement end

"""
    AbstractMagneticField

Abstract type for any magnetic field model.

Required methods:
- `field_at(field, position, [time])` → Vec3 (Tesla)
- `gradient_at(field, position, [time])` → 3×3 Matrix (T/m)

Optional methods:
- `field_at_body(field, position, R, [time])` → Vec3 in body frame
- `gradient_at_body(field, position, R, [time])` → 3×3 in body frame
"""
abstract type AbstractMagneticField end

"""
    AbstractSensor

Abstract type for any sensor model.

Required methods:
- `measure(sensor, state, field, [time])` → measurement struct
- `measurement_covariance(sensor)` → covariance matrix
- `measurement_dimension(sensor)` → Int
"""
abstract type AbstractSensor end

"""
    AbstractEstimator

Abstract type for state estimation backends.

Required methods:
- `initialize!(estimator, initial_state)` → nothing
- `predict!(estimator, control_input, dt)` → predicted_state
- `update!(estimator, measurement, sensor)` → updated_state
- `get_state(estimator)` → current_state
- `get_covariance(estimator)` → covariance_matrix
"""
abstract type AbstractEstimator end

"""
    AbstractResidualManager

Abstract type for residual processing and anomaly detection.

Required methods:
- `process_residual!(manager, residual, covariance, position, time)` → stats
- `get_gating_decision(manager, stats)` → GatingDecision
- `get_candidates(manager)` → Vector of candidates
- `get_confirmed(manager)` → Vector of confirmed features
"""
abstract type AbstractResidualManager end

# ============================================================================
# Magnetic Field Interface
# ============================================================================

"""
    field_at(field::AbstractMagneticField, position, [time]) → Vec3

Compute magnetic field B at position (and optional time for AC sources).
Returns field in Tesla, world frame.
"""
function field_at end

"""
    gradient_at(field::AbstractMagneticField, position, [time]) → Matrix

Compute gradient tensor ∂Bᵢ/∂xⱼ at position.
Returns 3×3 symmetric tensor in T/m, world frame.
"""
function gradient_at end

"""
    field_at_body(field::AbstractMagneticField, position, R, [time]) → Vec3

Compute field in body frame: B_body = Rᵀ B_world.
Default implementation uses field_at + transform.
"""
function field_at_body(field::AbstractMagneticField, position, R, t = 0.0)
    B_world = field_at(field, position, t)
    return transform_field_to_body(B_world, R)
end

"""
    gradient_at_body(field::AbstractMagneticField, position, R, [time]) → Matrix

Compute gradient in body frame: G_body = Rᵀ G_world R.
Default implementation uses gradient_at + transform.
"""
function gradient_at_body(field::AbstractMagneticField, position, R, t = 0.0)
    G_world = gradient_at(field, position, t)
    return transform_gradient_to_body(G_world, R)
end

# ============================================================================
# Sensor Interface
# ============================================================================

"""
    measure(sensor::AbstractSensor, state, field, [time]) → measurement

Generate a measurement given vehicle state and field model.
Returns sensor-specific measurement struct.
"""
function measure end

"""
    measurement_covariance(sensor::AbstractSensor) → Matrix

Get the measurement noise covariance matrix.
"""
function measurement_covariance end

"""
    measurement_dimension(sensor::AbstractSensor) → Int

Get the dimension of the measurement vector.
"""
function measurement_dimension end

# ============================================================================
# Frame Transformations
# ============================================================================

"""
    transform_field_to_body(B_world::AbstractVector, R::AbstractMatrix)

Transform magnetic field from world frame to body frame.
B_body = Rᵀ · B_world
"""
function transform_field_to_body(B_world::AbstractVector, R::AbstractMatrix)
    return R' * B_world
end

"""
    transform_field_to_world(B_body::AbstractVector, R::AbstractMatrix)

Transform magnetic field from body frame to world frame.
B_world = R · B_body
"""
function transform_field_to_world(B_body::AbstractVector, R::AbstractMatrix)
    return R * B_body
end

"""
    transform_gradient_to_body(G_world::AbstractMatrix, R::AbstractMatrix)

Transform gradient tensor from world frame to body frame.
G_body = Rᵀ · G_world · R
"""
function transform_gradient_to_body(G_world::AbstractMatrix, R::AbstractMatrix)
    return R' * G_world * R
end

"""
    transform_gradient_to_world(G_body::AbstractMatrix, R::AbstractMatrix)

Transform gradient tensor from body frame to world frame.
G_world = R · G_body · Rᵀ
"""
function transform_gradient_to_world(G_body::AbstractMatrix, R::AbstractMatrix)
    return R * G_body * R'
end

# ============================================================================
# Scenario Configuration
# ============================================================================

"""
    ScenarioConfig

Configuration for a simulation scenario.

# Fields
- `name::String`: Scenario identifier
- `seed::Int`: Random seed for reproducibility
- `duration::Float64`: Simulation duration (s)
- `field_config::Dict`: Field model configuration
- `sensor_configs::Dict`: Sensor configurations by type
- `trajectory_config::Dict`: Trajectory configuration
- `estimator_config::Dict`: Estimator configuration
"""
struct ScenarioConfig
    name::String
    seed::Int
    duration::Float64
    field_config::Dict{Symbol, Any}
    sensor_configs::Dict{Symbol, Any}
    trajectory_config::Dict{Symbol, Any}
    estimator_config::Dict{Symbol, Any}
end

function ScenarioConfig(;
    name::String = "default",
    seed::Int = 42,
    duration::Real = 60.0,
    field_config::Dict = Dict{Symbol, Any}(),
    sensor_configs::Dict = Dict{Symbol, Any}(),
    trajectory_config::Dict = Dict{Symbol, Any}(),
    estimator_config::Dict = Dict{Symbol, Any}()
)
    ScenarioConfig(
        name, seed, Float64(duration),
        Dict{Symbol, Any}(field_config),
        Dict{Symbol, Any}(sensor_configs),
        Dict{Symbol, Any}(trajectory_config),
        Dict{Symbol, Any}(estimator_config)
    )
end

"""
    ScenarioResult

Results from running a scenario.

# Fields
- `config::ScenarioConfig`: Configuration used
- `timestamps::Vector{Float64}`: Time stamps
- `true_states::Vector`: True vehicle states
- `estimated_states::Vector`: Estimated states
- `residual_stats::Vector`: Residual statistics at each step
- `confirmed_features::Vector`: Confirmed anomaly features
- `metrics::Dict`: Summary metrics
"""
struct ScenarioResult
    config::ScenarioConfig
    timestamps::Vector{Float64}
    true_states::Vector
    estimated_states::Vector
    residual_stats::Vector
    confirmed_features::Vector
    metrics::Dict{Symbol, Any}
end

# ============================================================================
# Type Registration (for extensibility)
# ============================================================================

"""Registry of available field model types."""
const FIELD_REGISTRY = Dict{Symbol, Type}()

"""Registry of available sensor types."""
const SENSOR_REGISTRY = Dict{Symbol, Type}()

"""
    register_field!(name::Symbol, type::Type)

Register a field model type for factory construction.
"""
function register_field!(name::Symbol, type::Type)
    FIELD_REGISTRY[name] = type
end

"""
    register_sensor!(name::Symbol, type::Type)

Register a sensor type for factory construction.
"""
function register_sensor!(name::Symbol, type::Type)
    SENSOR_REGISTRY[name] = type
end

"""
    create_field(name::Symbol, config::Dict) → AbstractMagneticField

Factory function to create a field model from config.
"""
function create_field(name::Symbol, config::Dict)
    if !haskey(FIELD_REGISTRY, name)
        error("Unknown field type: $name. Available: $(keys(FIELD_REGISTRY))")
    end
    return _create_field(FIELD_REGISTRY[name], config)
end

"""
    create_sensor(name::Symbol, config::Dict) → AbstractSensor

Factory function to create a sensor from config.
"""
function create_sensor(name::Symbol, config::Dict)
    if !haskey(SENSOR_REGISTRY, name)
        error("Unknown sensor type: $name. Available: $(keys(SENSOR_REGISTRY))")
    end
    return _create_sensor(SENSOR_REGISTRY[name], config)
end

# Default implementations (to be overridden)
_create_field(::Type, config::Dict) = error("Not implemented")
_create_sensor(::Type, config::Dict) = error("Not implemented")
