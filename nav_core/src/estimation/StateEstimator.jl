# ============================================================================
# StateEstimator.jl - Main state estimation interface
# ============================================================================

export StateEstimator, EstimatorConfig, StepResult
export create_estimator, step!, get_state, get_covariance

"""
    EstimatorConfig

Configuration for the state estimator.

# Fields
- `initial_state::Vector{Float64}` - Initial state estimate
- `initial_covariance::Matrix{Float64}` - Initial covariance
- `process_noise::Matrix{Float64}` - Process noise Q
- `window_size::Int` - Sliding window size (0 for full batch)
- `marginalize::Bool` - Enable marginalization
"""
Base.@kwdef struct EstimatorConfig
    initial_state::Vector{Float64}
    initial_covariance::Matrix{Float64}
    process_noise::Matrix{Float64}
    window_size::Int = 0
    marginalize::Bool = true
end

"""
    StepResult

Result of a single estimation step.

# Fields
- `state::Vector{Float64}` - Updated state estimate
- `covariance::Matrix{Float64}` - Updated covariance
- `innovations::Dict{Symbol, Float64}` - Per-sensor innovations
- `accepted::Dict{Symbol, Bool}` - Per-sensor acceptance flags
"""
struct StepResult
    state::Vector{Float64}
    covariance::Matrix{Float64}
    innovations::Dict{Symbol, Float64}
    accepted::Dict{Symbol, Bool}
end

"""
    StateEstimator{T}

Main state estimator wrapping the factor graph.

# Fields
- `graph::FactorGraph{T}` - Underlying factor graph
- `config::EstimatorConfig` - Estimator configuration
- `current_var_id::Int` - Current state variable ID
- `timestamp::Float64` - Current timestamp
"""
mutable struct StateEstimator{T<:Real}
    graph::FactorGraph{T}
    config::EstimatorConfig
    current_var_id::Int
    timestamp::Float64
end

"""
    create_estimator(config::EstimatorConfig) -> StateEstimator

Create a new state estimator with the given configuration.
"""
function create_estimator(config::EstimatorConfig)
    T = eltype(config.initial_state)
    graph = FactorGraph{T}()

    # Add initial state as first variable
    var_id = add_variable!(graph, length(config.initial_state),
                           config.initial_state, 0.0)

    # Add prior factor
    prior = PriorFactor(
        var_id,
        config.initial_state,
        config.initial_covariance,
        FactorMetadata(1, 0.0, :prior)
    )
    add_factor!(graph, prior)

    return StateEstimator{T}(graph, config, var_id, 0.0)
end

"""
    step!(estimator, measurements, dt) -> StepResult

Perform one estimation step with new measurements.
"""
function step!(estimator::StateEstimator{T},
               measurements::Vector{<:AbstractMeasurement},
               dt::Float64) where T

    # Update timestamp
    new_timestamp = estimator.timestamp + dt

    # Get current state
    current_var = get_variable(estimator.graph, estimator.current_var_id)
    current_state = current_var.value

    # Predict new state (simple constant velocity for now)
    predicted_state = predict_state(current_state, dt)

    # Add new state variable
    new_var_id = add_variable!(estimator.graph, length(predicted_state),
                               predicted_state, new_timestamp)

    # Add motion factor between states
    add_motion_factor!(estimator, estimator.current_var_id, new_var_id, dt)

    # Add measurement factors
    innovations = Dict{Symbol, Float64}()
    accepted = Dict{Symbol, Bool}()

    for meas in measurements
        if is_valid(meas)
            innov = add_measurement_factor!(estimator, new_var_id, meas)
            innovations[meas.meta.sensor_id] = innov
            accepted[meas.meta.sensor_id] = true
        else
            accepted[meas.meta.sensor_id] = false
        end
    end

    # Optimize graph
    result = optimize!(estimator.graph)

    # Update current state pointer
    estimator.current_var_id = new_var_id
    estimator.timestamp = new_timestamp

    # Get updated state and covariance
    updated_var = get_variable(estimator.graph, new_var_id)
    cov = marginal_covariance(estimator.graph, new_var_id)

    return StepResult(updated_var.value, cov, innovations, accepted)
end

"""
    get_state(estimator) -> Vector

Get current state estimate.
"""
function get_state(estimator::StateEstimator)
    var = get_variable(estimator.graph, estimator.current_var_id)
    return var.value
end

"""
    get_covariance(estimator) -> Matrix

Get current state covariance.
"""
function get_covariance(estimator::StateEstimator)
    return marginal_covariance(estimator.graph, estimator.current_var_id)
end

# ============================================================================
# Internal helpers
# ============================================================================

function predict_state(state::AbstractVector, dt::Float64)
    # Simple prediction - override for specific dynamics
    return copy(state)
end

function add_motion_factor!(estimator::StateEstimator, from_id::Int, to_id::Int, dt::Float64)
    # Add between factor for motion model
    dim = get_variable(estimator.graph, from_id).dim
    measurement = zeros(dim)  # Zero-mean motion
    Σ = estimator.config.process_noise * dt

    factor = BetweenFactor(
        from_id, to_id, measurement, Σ,
        FactorMetadata(length(estimator.graph.factors) + 1, estimator.timestamp, :motion)
    )
    add_factor!(estimator.graph, factor)
end

function add_measurement_factor!(estimator::StateEstimator, var_id::Int,
                                  meas::AbstractMeasurement)
    factor = MeasurementFactor(
        var_id, meas,
        FactorMetadata(length(estimator.graph.factors) + 1, meas.meta.timestamp, :measurement)
    )
    add_factor!(estimator.graph, factor)

    # Return innovation (placeholder)
    return 0.0
end
