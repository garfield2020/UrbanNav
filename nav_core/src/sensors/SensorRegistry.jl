# ============================================================================
# SensorRegistry.jl - Registry pattern for sensor models
# ============================================================================
#
# Sensors are pluggable. The registry maps sensor types to implementations.
# No more `if enable_odometry` scattered throughout the codebase.
#
# Usage:
#   register_sensor!(:odometry, OdometrySensorModel())
#   model = get_sensor(:odometry)
#   process!(model, measurement)
# ============================================================================

export AbstractSensorModel, SensorRegistry
export register_sensor!, get_sensor, list_sensors, has_sensor

"""
    AbstractSensorModel

Base type for sensor model implementations.
Each sensor type must implement the sensor model interface.
"""
abstract type AbstractSensorModel end

"""
Required interface for sensor models:
- `initialize!(model, config)` - Initialize with configuration
- `process(model, raw_data)` - Process raw data to measurement
- `predict(model, state)` - Predict measurement from state
- `validate(model, measurement)` - Validate measurement
"""
function initialize! end
function process end
function predict end
function validate end

"""
    SensorRegistry

Global registry for sensor model implementations.
Thread-safe via lock.
"""
struct SensorRegistry
    models::Dict{Symbol, AbstractSensorModel}
    lock::ReentrantLock
end

# Global singleton registry
const _SENSOR_REGISTRY = SensorRegistry(Dict{Symbol, AbstractSensorModel}(), ReentrantLock())

"""
    register_sensor!(sensor_type::Symbol, model::AbstractSensorModel)

Register a sensor model implementation.
Overwrites existing registration for the same type.
"""
function register_sensor!(sensor_type::Symbol, model::AbstractSensorModel)
    lock(_SENSOR_REGISTRY.lock) do
        _SENSOR_REGISTRY.models[sensor_type] = model
    end
    return nothing
end

"""
    get_sensor(sensor_type::Symbol) -> AbstractSensorModel

Retrieve a registered sensor model.
Throws KeyError if not registered.
"""
function get_sensor(sensor_type::Symbol)
    lock(_SENSOR_REGISTRY.lock) do
        return _SENSOR_REGISTRY.models[sensor_type]
    end
end

"""
    has_sensor(sensor_type::Symbol) -> Bool

Check if a sensor type is registered.
"""
function has_sensor(sensor_type::Symbol)
    lock(_SENSOR_REGISTRY.lock) do
        return haskey(_SENSOR_REGISTRY.models, sensor_type)
    end
end

"""
    list_sensors() -> Vector{Symbol}

List all registered sensor types.
"""
function list_sensors()
    lock(_SENSOR_REGISTRY.lock) do
        return collect(keys(_SENSOR_REGISTRY.models))
    end
end

"""
    clear_sensors!()

Clear all registered sensors. Primarily for testing.
"""
function clear_sensors!()
    lock(_SENSOR_REGISTRY.lock) do
        empty!(_SENSOR_REGISTRY.models)
    end
    return nothing
end

# ============================================================================
# Standard sensor model stubs (implementations in separate files)
# ============================================================================

# IMU model - to be implemented
# Odometry model - to be implemented
# Depth model - to be implemented
# Magnetometer model - to be implemented
