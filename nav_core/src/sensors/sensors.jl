# ============================================================================
# Sensor Models for Urban Navigation
# ============================================================================
#
# Adapted for urban pedestrian and vehicle navigation
#
# Sensor models for odometry, barometer, FTM, and IMU.
# Each sensor provides measurement simulation and noise characterization.
# ============================================================================

using LinearAlgebra
using StaticArrays
using Random
using Rotations

export OdometryParams, OdometryMeasurement, OdometryModel
export BarometerParams, BarometerMeasurement, BarometerModel
export FTMParams, FTMMeasurement, FTMModel
export IMUParams, IMUMeasurement, IMUModel
export CompassParams, CompassMeasurement, CompassModel
export simulate_measurement, measurement_covariance, measurement_variance, wrap_angle
export measurement_covariance_B, measurement_covariance_G
export gradient_to_vector, vector_to_gradient

# ============================================================================
# Odometry (Wheel/Visual Odometry) Model
# ============================================================================

"""
    OdometryParams

Odometry sensor parameters for wheel or visual odometry.

# Fields
- `noise_std::Float64`: Velocity noise standard deviation (m/s)
- `scale_error::Float64`: Scale factor error
- `max_speed::Float64`: Maximum measurable speed (m/s)
- `dropout_prob::Float64`: Probability of measurement dropout (e.g., visual tracking loss)
"""
struct OdometryParams
    noise_std::Float64
    scale_error::Float64
    max_speed::Float64
    dropout_prob::Float64
end

function OdometryParams(;
    noise_std::Real = 0.01,
    scale_error::Real = 0.0,
    max_speed::Real = 50.0,
    dropout_prob::Real = 0.0
)
    OdometryParams(Float64(noise_std), Float64(scale_error), Float64(max_speed), Float64(dropout_prob))
end

"""
    OdometryMeasurement

Odometry measurement result.

# Fields
- `velocity::Vec3`: Measured velocity in body frame (m/s)
- `valid::Bool`: Whether measurement is valid
- `timestamp::Float64`: Measurement timestamp
"""
struct OdometryMeasurement
    velocity::Vec3
    valid::Bool
    timestamp::Float64
end

"""
    OdometryModel

Odometry sensor model for body-frame velocity measurement.
"""
struct OdometryModel
    params::OdometryParams
    rng::AbstractRNG
end

OdometryModel(params::OdometryParams = OdometryParams()) = OdometryModel(params, Random.default_rng())
OdometryModel(rng::AbstractRNG) = OdometryModel(OdometryParams(), rng)

"""
    simulate_measurement(model::OdometryModel, state::UrbanNavState)

Simulate odometry measurement from current state.
"""
function simulate_measurement(model::OdometryModel, state::UrbanNavState)
    if rand(model.rng) < model.params.dropout_prob
        return OdometryMeasurement(zeros(Vec3), false, state.timestamp)
    end

    vel_body = transform_to_body(state, state.velocity)
    noise = Vec3(randn(model.rng, 3)...) .* model.params.noise_std
    vel_measured = vel_body + noise

    OdometryMeasurement(vel_measured, true, state.timestamp)
end

"""
    measurement_covariance(model::OdometryModel)

Get odometry measurement covariance matrix (3×3).
"""
function measurement_covariance(model::OdometryModel)
    model.params.noise_std^2 * Mat3(I)
end

# ============================================================================
# Barometer (Altitude/Floor Height) Sensor Model
# ============================================================================

"""
    BarometerParams

Barometer sensor parameters for altitude/floor height estimation.

# Fields
- `noise_std::Float64`: Altitude noise standard deviation (m)
- `bias::Float64`: Constant altitude bias (m)
- `scale_factor::Float64`: Scale factor error
"""
struct BarometerParams
    noise_std::Float64
    bias::Float64
    scale_factor::Float64
end

function BarometerParams(;
    noise_std::Real = 0.1,
    bias::Real = 0.0,
    scale_factor::Real = 1.0
)
    BarometerParams(Float64(noise_std), Float64(bias), Float64(scale_factor))
end

"""
    BarometerMeasurement

Barometer measurement result.

# Fields
- `altitude::Float64`: Measured barometric altitude or floor height (m)
- `timestamp::Float64`: Measurement timestamp
"""
struct BarometerMeasurement
    altitude::Float64
    timestamp::Float64
end

"""
    BarometerModel

Barometer sensor model for altitude/floor height measurement.
"""
struct BarometerModel
    params::BarometerParams
    rng::AbstractRNG
end

BarometerModel(params::BarometerParams = BarometerParams()) = BarometerModel(params, Random.default_rng())
BarometerModel(rng::AbstractRNG) = BarometerModel(BarometerParams(), rng)

"""
    simulate_measurement(model::BarometerModel, state::UrbanNavState)

Simulate barometer measurement from current state.
"""
function simulate_measurement(model::BarometerModel, state::UrbanNavState)
    true_altitude = altitude(state)

    measured_altitude = model.params.scale_factor * true_altitude +
                     model.params.bias +
                     randn(model.rng) * model.params.noise_std

    BarometerMeasurement(measured_altitude, state.timestamp)
end

"""
    measurement_variance(model::BarometerModel)

Get barometer measurement variance.
"""
measurement_variance(model::BarometerModel) = model.params.noise_std^2

# ============================================================================
# FTM (Full Tensor Magnetometer) Model
# ============================================================================

"""
    FTMParams

FTM sensor parameters.

# Fields
- `noise_B::Float64`: Field noise standard deviation (T)
- `noise_G::Float64`: Gradient noise standard deviation (T/m)
- `measure_gradient::Bool`: Whether gradient is measured
"""
struct FTMParams
    noise_B::Float64
    noise_G::Float64
    measure_gradient::Bool
end

function FTMParams(;
    noise_B::Real = 5e-9,
    noise_G::Real = 5e-9,
    measure_gradient::Bool = false
)
    FTMParams(Float64(noise_B), Float64(noise_G), measure_gradient)
end

"""
    FTMMeasurement

FTM measurement result.

# Fields
- `B::Vec3`: Measured magnetic field (T)
- `G::Union{Nothing, SMatrix{3,3,Float64,9}}`: Gradient tensor (T/m)
- `position::Vec3`: Position where measurement was taken
- `timestamp::Float64`: Measurement timestamp
"""
struct FTMMeasurement
    B::Vec3
    G::Union{Nothing, SMatrix{3,3,Float64,9}}
    position::Vec3
    timestamp::Float64
end

"""
    FTMModel

FTM sensor model.
"""
struct FTMModel
    params::FTMParams
    rng::AbstractRNG
end

FTMModel(params::FTMParams = FTMParams()) = FTMModel(params, Random.default_rng())
FTMModel(rng::AbstractRNG) = FTMModel(FTMParams(), rng)

"""
    simulate_measurement(model::FTMModel, state::UrbanNavState, true_B, true_G=nothing)

Simulate FTM measurement given true field values.
"""
function simulate_measurement(
    model::FTMModel,
    state::UrbanNavState,
    true_B::AbstractVector,
    true_G::Union{Nothing, AbstractMatrix} = nothing
)
    noise_B = Vec3(randn(model.rng, 3)...) .* model.params.noise_B
    B_measured = Vec3(true_B...) + noise_B

    G_measured = nothing
    if model.params.measure_gradient && true_G !== nothing
        noise_G = SMatrix{3,3,Float64,9}(randn(model.rng, 3, 3) .* model.params.noise_G)
        G_noisy = SMatrix{3,3,Float64,9}(true_G) + noise_G

        G_sym = (G_noisy + G_noisy') / 2
        trace_third = tr(G_sym) / 3
        G_measured = G_sym - trace_third * SMatrix{3,3,Float64,9}(I)
    end

    FTMMeasurement(B_measured, G_measured, state.position, state.timestamp)
end

"""
    measurement_covariance_B(model::FTMModel)

Get field measurement covariance matrix (3×3).
"""
measurement_covariance_B(model::FTMModel) = model.params.noise_B^2 * Mat3(I)

"""
    measurement_covariance_G(model::FTMModel)

Get gradient measurement covariance (5×5).
"""
function measurement_covariance_G(model::FTMModel)
    model.params.noise_G^2 * SMatrix{5,5,Float64,25}(I)
end

"""
    gradient_to_vector(G::AbstractMatrix)

Convert 3×3 gradient tensor to 5-component vector [Gxx, Gyy, Gxy, Gxz, Gyz].
"""
function gradient_to_vector(G::AbstractMatrix)
    SVector{5, Float64}(G[1,1], G[2,2], G[1,2], G[1,3], G[2,3])
end

"""
    vector_to_gradient(v::AbstractVector)

Convert 5-component vector to 3×3 symmetric traceless gradient tensor.
"""
function vector_to_gradient(v::AbstractVector)
    Gxx, Gyy, Gxy, Gxz, Gyz = v[1:5]
    Gzz = -(Gxx + Gyy)
    SMatrix{3,3,Float64,9}(
        Gxx, Gxy, Gxz,
        Gxy, Gyy, Gyz,
        Gxz, Gyz, Gzz
    )
end

# ============================================================================
# IMU Model
# ============================================================================

"""
    IMUParams

IMU sensor parameters.

# Fields
- `gyro_noise::Float64`: Gyroscope noise density (rad/s/√Hz)
- `accel_noise::Float64`: Accelerometer noise density (m/s²/√Hz)
- `gyro_bias_stability::Float64`: Gyro bias instability (rad/s)
- `accel_bias_stability::Float64`: Accel bias instability (m/s²)
"""
struct IMUParams
    gyro_noise::Float64
    accel_noise::Float64
    gyro_bias_stability::Float64
    accel_bias_stability::Float64
end

function IMUParams(;
    gyro_noise::Real = 0.001,
    accel_noise::Real = 0.01,
    gyro_bias_stability::Real = 1e-5,
    accel_bias_stability::Real = 1e-4
)
    IMUParams(Float64(gyro_noise), Float64(accel_noise),
              Float64(gyro_bias_stability), Float64(accel_bias_stability))
end

"""
    IMUMeasurement

IMU measurement result.

# Fields
- `gyro::Vec3`: Measured angular velocity (rad/s)
- `accel::Vec3`: Measured specific force (m/s²)
- `timestamp::Float64`: Measurement timestamp
"""
struct IMUMeasurement
    gyro::Vec3
    accel::Vec3
    timestamp::Float64
end

"""
    IMUModel

IMU sensor model.
"""
struct IMUModel
    params::IMUParams
    rng::AbstractRNG
end

IMUModel(params::IMUParams = IMUParams()) = IMUModel(params, Random.default_rng())
IMUModel(rng::AbstractRNG) = IMUModel(IMUParams(), rng)

"""
    simulate_measurement(model::IMUModel, state::UrbanNavState, true_accel_world, true_gyro_body, dt)

Simulate IMU measurement.
"""
function simulate_measurement(
    model::IMUModel,
    state::UrbanNavState,
    true_accel_world::AbstractVector,
    true_gyro_body::AbstractVector,
    dt::Real
)
    g = Vec3(0.0, 0.0, -9.81)
    accel_specific = transform_to_body(state, Vec3(true_accel_world...) - g)

    noise_scale = sqrt(dt)
    gyro_noise = Vec3(randn(model.rng, 3)...) .* (model.params.gyro_noise * noise_scale)
    accel_noise = Vec3(randn(model.rng, 3)...) .* (model.params.accel_noise * noise_scale)

    gyro_measured = Vec3(true_gyro_body...) + state.bias_gyro + gyro_noise
    accel_measured = accel_specific + state.bias_accel + accel_noise

    IMUMeasurement(gyro_measured, accel_measured, state.timestamp)
end

# ============================================================================
# Compass / Heading Sensor Model
# ============================================================================
#
# CRITICAL for navigation: Without compass, yaw is unobservable
# and odometry velocity transformations will be incorrect during turns.
#
# Convention:
#   - Heading is yaw angle in NED frame
#   - 0 = North, π/2 = East, π = South, -π/2 = West
#   - Output is in radians, range [-π, π]
# ============================================================================

"""
    CompassParams

Compass/heading sensor parameters.

# Fields
- `noise_std::Float64`: Heading noise standard deviation (radians)
- `hard_iron::Vec3`: Hard iron bias in body frame (for magnetometer-based compass)
- `soft_iron::Mat3`: Soft iron distortion matrix
- `declination::Float64`: Local magnetic declination (radians, East positive)
"""
struct CompassParams
    noise_std::Float64
    hard_iron::Vec3
    soft_iron::SMatrix{3,3,Float64,9}
    declination::Float64
end

function CompassParams(;
    noise_std::Real = 0.02,  # ~1 degree
    hard_iron::AbstractVector = zeros(3),
    soft_iron::AbstractMatrix = SMatrix{3,3,Float64,9}(I),
    declination::Real = 0.0
)
    CompassParams(
        Float64(noise_std),
        Vec3(hard_iron...),
        SMatrix{3,3,Float64,9}(soft_iron),
        Float64(declination)
    )
end

"""
    CompassMeasurement

Compass measurement result.

# Fields
- `heading::Float64`: Measured heading in NED frame (radians)
- `valid::Bool`: Whether measurement is valid
- `timestamp::Float64`: Measurement timestamp
"""
struct CompassMeasurement
    heading::Float64
    valid::Bool
    timestamp::Float64
end

"""
    CompassModel

Compass sensor model for heading measurement.

The compass measures magnetic heading, which is then corrected for declination
to produce true heading (yaw in NED frame).

# Critical for navigation
Without compass updates, the estimator cannot track yaw during turns.
This causes odometry body-frame velocities to be incorrectly transformed,
leading to position divergence during navigation.
"""
struct CompassModel
    params::CompassParams
    rng::AbstractRNG
end

CompassModel(params::CompassParams = CompassParams()) = CompassModel(params, Random.default_rng())
CompassModel(rng::AbstractRNG) = CompassModel(CompassParams(), rng)

"""
    simulate_measurement(model::CompassModel, state::UrbanNavState)

Simulate compass measurement from current state.

Returns heading in NED frame (0 = North, π/2 = East).
"""
function simulate_measurement(model::CompassModel, state::UrbanNavState)
    # Get true heading from state (yaw angle)
    true_heading = yaw(state)

    # Add noise
    noise = randn(model.rng) * model.params.noise_std
    measured_heading = true_heading + noise

    # Wrap to [-π, π]
    measured_heading = wrap_angle(measured_heading)

    CompassMeasurement(measured_heading, true, state.timestamp)
end

"""
    simulate_measurement(model::CompassModel, true_heading::Real, timestamp::Real)

Simulate compass measurement from explicit heading value.
"""
function simulate_measurement(model::CompassModel, true_heading::Real, timestamp::Real)
    noise = randn(model.rng) * model.params.noise_std
    measured_heading = wrap_angle(true_heading + noise)
    CompassMeasurement(Float64(measured_heading), true, Float64(timestamp))
end

"""
    measurement_variance(model::CompassModel)

Get compass measurement variance.
"""
measurement_variance(model::CompassModel) = model.params.noise_std^2

"""
    wrap_angle(angle::Real) -> Float64

Wrap angle to [-π, π].
"""
function wrap_angle(angle::Real)
    a = Float64(angle)
    while a > π
        a -= 2π
    end
    while a < -π
        a += 2π
    end
    return a
end
