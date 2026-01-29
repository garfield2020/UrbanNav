"""
    MissionRunner

Main simulation runner for UrbanNav. Ties together a world (ElevatorWorld,
ParkingGarageWorld, or LobbyWorld), a trajectory, and sensor models to produce
a complete simulated mission with deterministic, reproducible results.
"""
module MissionRunner

using StaticArrays
using Random
using LinearAlgebra

# ---------------------------------------------------------------------------
# World and trajectory includes (expected to define their own modules)
# ---------------------------------------------------------------------------
include(joinpath(@__DIR__, "worlds", "elevator_world.jl"))
include(joinpath(@__DIR__, "worlds", "parking_garage_world.jl"))
include(joinpath(@__DIR__, "worlds", "lobby_world.jl"))
include(joinpath(@__DIR__, "trajectory.jl"))

using .ElevatorWorld
using .ParkingGarageWorld
using .LobbyWorld
using .Trajectory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

"""Default Earth magnetic field vector in the local NED frame (µT).
Approximately 50 µT magnitude, pointing north and downward at ~60° inclination."""
const EARTH_FIELD_NED = SVector(22.0, 0.0, 43.3)  # µT

# ---------------------------------------------------------------------------
# Sensor noise parameters
# ---------------------------------------------------------------------------

"""
    SensorNoiseParams

Standard deviations for each sensor channel.

# Fields
- `magnetometer_noise::Float64`: 1-σ magnetometer noise per axis (µT). Default 0.1 µT.
- `imu_accel_noise::Float64`: 1-σ accelerometer noise per axis (m/s²). Default 0.01 m/s².
- `imu_gyro_noise::Float64`: 1-σ gyroscope noise per axis (rad/s). Default 1e-4 rad/s.
- `odometry_noise::Float64`: 1-σ odometry speed noise (m/s). Default 0.02 m/s.
- `barometer_noise::Float64`: 1-σ barometric altitude noise (m). Default 0.5 m.
"""
Base.@kwdef struct SensorNoiseParams
    magnetometer_noise::Float64 = 0.1
    imu_accel_noise::Float64    = 0.01
    imu_gyro_noise::Float64     = 1e-4
    odometry_noise::Float64     = 0.02
    barometer_noise::Float64    = 0.5
end

# ---------------------------------------------------------------------------
# Mission configuration
# ---------------------------------------------------------------------------

"""
    MissionConfig

Full specification of a simulated mission.

# Fields
- `world`: a world instance (ElevatorWorld, ParkingGarageWorld, or LobbyWorld).
- `trajectory`: trajectory specification used to generate true pose over time.
- `sensor_noise::SensorNoiseParams`: noise parameters for all sensor channels.
- `duration::Float64`: total mission duration in seconds.
- `dt::Float64`: time step in seconds (default 0.1 s = 10 Hz).
- `seed::UInt64`: random seed for deterministic noise generation.
"""
Base.@kwdef struct MissionConfig
    world::Any
    trajectory::Any
    sensor_noise::SensorNoiseParams = SensorNoiseParams()
    duration::Float64 = 60.0
    dt::Float64       = 0.1
    seed::UInt64      = UInt64(42)
end

# ---------------------------------------------------------------------------
# Per-step measurement bundle
# ---------------------------------------------------------------------------

"""
    StepMeasurements

Sensor measurements produced at a single time step.

# Fields
- `magnetometer::SVector{3,Float64}`: measured magnetic field (µT) in body frame.
- `imu_accel::SVector{3,Float64}`: measured specific force (m/s²) in body frame.
- `imu_gyro::SVector{3,Float64}`: measured angular rate (rad/s) in body frame.
- `odometry::Float64`: measured forward speed (m/s).
- `barometer::Float64`: measured altitude (m).
"""
struct StepMeasurements
    magnetometer::SVector{3,Float64}
    imu_accel::SVector{3,Float64}
    imu_gyro::SVector{3,Float64}
    odometry::Float64
    barometer::Float64
end

# ---------------------------------------------------------------------------
# Source ground-truth snapshot
# ---------------------------------------------------------------------------

"""
    SourceSnapshot

Ground-truth state of all magnetic sources at a single time step.

# Fields
- `positions::Vector{SVector{3,Float64}}`: position of each source (m).
- `moments::Vector{SVector{3,Float64}}`: magnetic dipole moment of each source (A·m²).
"""
struct SourceSnapshot
    positions::Vector{SVector{3,Float64}}
    moments::Vector{SVector{3,Float64}}
end

# ---------------------------------------------------------------------------
# Mission result
# ---------------------------------------------------------------------------

"""
    MissionResult

Complete output of a simulated mission.

# Fields
- `timestamps::Vector{Float64}`: time at each step (s).
- `true_positions::Vector{SVector{3,Float64}}`: true position at each step (m).
- `true_velocities::Vector{SVector{3,Float64}}`: true velocity at each step (m/s).
- `true_attitudes::Vector{SMatrix{3,3,Float64,9}}`: true attitude (DCM, body-to-NED) at each step.
- `measurements::Vector{StepMeasurements}`: noisy sensor measurements at each step.
- `source_ground_truth::Vector{SourceSnapshot}`: magnetic source states at each step.
"""
struct MissionResult
    timestamps::Vector{Float64}
    true_positions::Vector{SVector{3,Float64}}
    true_velocities::Vector{SVector{3,Float64}}
    true_attitudes::Vector{SMatrix{3,3,Float64,9}}
    measurements::Vector{StepMeasurements}
    source_ground_truth::Vector{SourceSnapshot}
end

# ---------------------------------------------------------------------------
# World interface helpers (dispatch on concrete world types)
# ---------------------------------------------------------------------------

"""
    step_world!(world, dt)

Advance the world's dynamic sources by one time step `dt` seconds.
"""
step_world!(world, dt) = step!(world, dt)

"""
    world_magnetic_field(world, position)

Return the anomaly magnetic field (µT) produced by the world at `position`.
"""
world_magnetic_field(world, position) = magnetic_field(world, position)

"""
    world_background_field(world, position)

Return any static background magnetic anomaly (µT) at `position`.
Defaults to zero if the world does not define one.
"""
world_background_field(world, position) = try
    background_field(world, position)
catch
    SVector(0.0, 0.0, 0.0)
end

"""
    world_source_snapshot(world)

Return a `SourceSnapshot` with ground-truth source positions and moments.
"""
function world_source_snapshot(world)
    pos = source_positions(world)
    mom = source_moments(world)
    SourceSnapshot(pos, mom)
end

# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

"""
    run_mission(config::MissionConfig) -> MissionResult

Execute a full simulated mission.

The function steps through time from `0` to `config.duration` at intervals of
`config.dt`.  At each step it:

1. Queries the trajectory for the true position, velocity, and attitude.
2. Advances the world's dynamic sources by `dt`.
3. Computes the true magnetic field as the sum of the Earth field, a static
   background anomaly, and the world's source field, then rotates into the body
   frame.
4. Generates noisy sensor measurements using a deterministic RNG seeded from
   `config.seed`.
5. Records ground-truth source states.

Returns a [`MissionResult`](@ref) containing all recorded data.
"""
function run_mission(config::MissionConfig)::MissionResult
    rng = MersenneTwister(config.seed)

    n_steps = floor(Int, config.duration / config.dt) + 1

    # Pre-allocate output vectors
    timestamps       = Vector{Float64}(undef, n_steps)
    true_positions   = Vector{SVector{3,Float64}}(undef, n_steps)
    true_velocities  = Vector{SVector{3,Float64}}(undef, n_steps)
    true_attitudes   = Vector{SMatrix{3,3,Float64,9}}(undef, n_steps)
    measurements     = Vector{StepMeasurements}(undef, n_steps)
    source_gt        = Vector{SourceSnapshot}(undef, n_steps)

    noise = config.sensor_noise
    world = config.world
    traj  = config.trajectory

    for i in 1:n_steps
        t = (i - 1) * config.dt

        # --- 1. True pose from trajectory ---
        pos = position(traj, t)
        vel = velocity(traj, t)
        att = attitude(traj, t)  # 3×3 DCM: body-to-NED

        # --- 2. Step the world ---
        if i > 1
            step_world!(world, config.dt)
        end

        # --- 3. True magnetic field in NED, then rotate to body frame ---
        B_earth      = EARTH_FIELD_NED
        B_background = world_background_field(world, pos)
        B_sources    = world_magnetic_field(world, pos)
        B_ned        = B_earth + B_background + B_sources

        # att is body-to-NED, so NED-to-body = att'
        R_ned2body   = att'
        B_body       = R_ned2body * B_ned

        # --- 4. Noisy measurements ---
        mag_meas = B_body + SVector(
            noise.magnetometer_noise * randn(rng),
            noise.magnetometer_noise * randn(rng),
            noise.magnetometer_noise * randn(rng),
        )

        # IMU: true specific force in body frame (gravity + acceleration)
        # Simplified: assume gravity in NED is [0, 0, 9.81] m/s²
        gravity_ned   = SVector(0.0, 0.0, 9.81)
        true_accel    = R_ned2body * gravity_ned  # specific force at rest
        accel_meas    = true_accel + SVector(
            noise.imu_accel_noise * randn(rng),
            noise.imu_accel_noise * randn(rng),
            noise.imu_accel_noise * randn(rng),
        )

        # Gyroscope: true angular rate (query from trajectory if available)
        true_gyro = try
            angular_rate(traj, t)
        catch
            SVector(0.0, 0.0, 0.0)
        end
        gyro_meas = true_gyro + SVector(
            noise.imu_gyro_noise * randn(rng),
            noise.imu_gyro_noise * randn(rng),
            noise.imu_gyro_noise * randn(rng),
        )

        # Odometry: forward speed (norm of velocity projected onto body x-axis)
        body_vel      = R_ned2body * vel
        true_speed    = body_vel[1]
        speed_meas    = true_speed + noise.odometry_noise * randn(rng)

        # Barometer: altitude (negative of NED down component)
        true_alt      = -pos[3]
        baro_meas     = true_alt + noise.barometer_noise * randn(rng)

        step_meas = StepMeasurements(mag_meas, accel_meas, gyro_meas, speed_meas, baro_meas)

        # --- 5. Record everything ---
        timestamps[i]     = t
        true_positions[i] = pos
        true_velocities[i] = vel
        true_attitudes[i] = att
        measurements[i]   = step_meas
        source_gt[i]      = world_source_snapshot(world)
    end

    return MissionResult(
        timestamps,
        true_positions,
        true_velocities,
        true_attitudes,
        measurements,
        source_gt,
    )
end

export MissionConfig, SensorNoiseParams, StepMeasurements, SourceSnapshot, MissionResult
export run_mission, EARTH_FIELD_NED

end # module MissionRunner
