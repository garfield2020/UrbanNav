# ============================================================================
# StateContract.jl - Authoritative definition of state vector structure
# ============================================================================
#
# Ported from AUV-Navigation/src/state.jl
#
# This contract defines WHAT is in the state vector. All modules must conform.
# Changes here require review and acceptance test updates.
#
# State Partitioning (from SE specification):
# A. Vehicle state (time-indexed per keyframe k):
#    - Pose T_k in SE(3): position (3) + orientation quaternion (4)
#    - Velocity v_k (3)
#    - IMU biases b^g_k (3), b^a_k (3)
#
# Total vehicle state dimension: 3 + 4 + 3 + 3 + 3 = 16 (full)
# Error-state dimension: 3 + 3 + 3 + 3 + 3 = 15 (uses rotation vector)
#
# State ownership rules:
# - Only the estimation module writes to navigation state
# - Only the feature module writes to feature state
# - Health module reads but never writes state
# ============================================================================

using LinearAlgebra
using StaticArrays
using Rotations

# Type aliases for clarity
const Vec3 = SVector{3, Float64}
const Mat3 = SMatrix{3, 3, Float64, 9}
const Mat15 = SMatrix{15, 15, Float64, 225}

export Vec3, Mat3, Mat15
export UrbanNavState, Keyframe
export position, velocity, orientation, bias_gyro, bias_accel, altitude
export rotation_matrix, euler_angles, state_dim, error_state_vector
export apply_error_state!, transform_to_body, transform_to_world
export position_uncertainty, position_std
export create_initial_state, add_measurement!, get_measurement

# ============================================================================
# Core Navigation State
# ============================================================================

"""
    UrbanNavState

AUV vehicle state at a single instant.

# Fields
- `position::Vec3`: Position [x, y, z] in meters (NED, z positive down)
- `velocity::Vec3`: Velocity [vx, vy, vz] in m/s (world frame)
- `orientation::QuatRotation{Float64}`: Orientation (world to body)
- `bias_gyro::Vec3`: Gyroscope bias [bwx, bwy, bwz] in rad/s
- `bias_accel::Vec3`: Accelerometer bias [bax, bay, baz] in m/s²
- `timestamp::Float64`: State timestamp in seconds
- `covariance::Matrix{Float64}`: Error-state covariance (15×15)

# Frame Conventions
- World frame: NED (North-East-Down)
- Body frame: Forward-Right-Down
- Quaternion: Hamilton convention, world to body rotation
"""
mutable struct UrbanNavState
    position::Vec3
    velocity::Vec3
    orientation::QuatRotation{Float64}
    bias_gyro::Vec3
    bias_accel::Vec3
    timestamp::Float64
    covariance::Matrix{Float64}
end

# Constructors
function UrbanNavState(;
    position::AbstractVector = zeros(3),
    velocity::AbstractVector = zeros(3),
    orientation::Union{QuatRotation, AbstractVector} = QuatRotation(1.0, 0.0, 0.0, 0.0),
    bias_gyro::AbstractVector = zeros(3),
    bias_accel::AbstractVector = zeros(3),
    timestamp::Real = 0.0,
    covariance::AbstractMatrix = Matrix{Float64}(I, 15, 15) * 0.01
)
    pos = Vec3(position...)
    vel = Vec3(velocity...)
    bg = Vec3(bias_gyro...)
    ba = Vec3(bias_accel...)

    if orientation isa QuatRotation
        q = orientation
    elseif length(orientation) == 4
        q = QuatRotation(orientation...)
    else
        q = QuatRotation(RotationVec(orientation...))
    end

    UrbanNavState(pos, vel, q, bg, ba, Float64(timestamp), Matrix{Float64}(covariance))
end

# State dimension constants
const NAV_STATE_DIM = 15  # Error-state dimension

# ============================================================================
# Accessors
# ============================================================================

position(s::UrbanNavState) = s.position
velocity(s::UrbanNavState) = s.velocity
orientation(s::UrbanNavState) = s.orientation
bias_gyro(s::UrbanNavState) = s.bias_gyro
bias_accel(s::UrbanNavState) = s.bias_accel

"""
    altitude(s::UrbanNavState)

Depth below surface (positive value, assumes NED convention).
"""
altitude(s::UrbanNavState) = -s.position[3]

"""
    rotation_matrix(s::UrbanNavState)

Get 3×3 rotation matrix (world to body).
"""
rotation_matrix(s::UrbanNavState) = Mat3(s.orientation)

"""
    euler_angles(s::UrbanNavState)

Get Euler angles [roll, pitch, yaw] in radians.
"""
euler_angles(s::UrbanNavState) = SVector{3}(Rotations.params(RotXYZ(s.orientation)))

"""
    state_dim(::UrbanNavState)

Error-state dimension (15).
"""
state_dim(::UrbanNavState) = NAV_STATE_DIM
state_dim(::Type{UrbanNavState}) = NAV_STATE_DIM

"""
    state_indices(::Type{UrbanNavState}) -> NamedTuple

Return named indices into the error-state vector.
"""
function state_indices(::Type{UrbanNavState})
    return (
        position = 1:3,
        velocity = 4:6,
        orientation = 7:9,  # Rotation vector (error-state)
        bias_gyro = 10:12,
        bias_accel = 13:15
    )
end

"""
    error_state_vector(s::UrbanNavState)

Get error-state vector representation.
Returns [pos(3), vel(3), rot_vec(3), bias_g(3), bias_a(3)]
"""
function error_state_vector(s::UrbanNavState)
    rv = RotationVec(s.orientation)
    vcat(s.position, s.velocity, SVector{3}(rv.sx, rv.sy, rv.sz), s.bias_gyro, s.bias_accel)
end

"""
    apply_error_state!(s::UrbanNavState, dx::AbstractVector)

Apply error-state correction in-place.

# Arguments
- `s`: State to modify
- `dx`: Error state [dp(3), dv(3), dθ(3), dbg(3), dba(3)] (15,)
"""
function apply_error_state!(s::UrbanNavState, dx::AbstractVector)
    @assert length(dx) == 15 "Expected dx length 15, got $(length(dx))"

    s.position = s.position + Vec3(dx[1:3]...)
    s.velocity = s.velocity + Vec3(dx[4:6]...)

    # Multiplicative rotation update
    dθ = Vec3(dx[7:9]...)
    dR = QuatRotation(RotationVec(dθ...))
    s.orientation = dR * s.orientation

    s.bias_gyro = s.bias_gyro + Vec3(dx[10:12]...)
    s.bias_accel = s.bias_accel + Vec3(dx[13:15]...)

    return s
end

"""
    transform_to_body(s::UrbanNavState, world_vec::AbstractVector)

Transform vector from world frame to body frame.
"""
function transform_to_body(s::UrbanNavState, world_vec::AbstractVector)
    Vec3((s.orientation * Vec3(world_vec...))...)
end

"""
    transform_to_world(s::UrbanNavState, body_vec::AbstractVector)

Transform vector from body frame to world frame.
"""
function transform_to_world(s::UrbanNavState, body_vec::AbstractVector)
    Vec3((inv(s.orientation) * Vec3(body_vec...))...)
end

"""
    position_uncertainty(s::UrbanNavState)

Scalar position uncertainty (sqrt of position covariance trace).
"""
function position_uncertainty(s::UrbanNavState)
    sqrt(s.covariance[1,1] + s.covariance[2,2] + s.covariance[3,3])
end

"""
    position_std(s::UrbanNavState)

Position standard deviations [σx, σy, σz].
"""
function position_std(s::UrbanNavState)
    Vec3(sqrt(s.covariance[1,1]), sqrt(s.covariance[2,2]), sqrt(s.covariance[3,3]))
end

"""
    Base.copy(s::UrbanNavState)

Create a deep copy of the state.
"""
function Base.copy(s::UrbanNavState)
    UrbanNavState(
        s.position,
        s.velocity,
        s.orientation,
        s.bias_gyro,
        s.bias_accel,
        s.timestamp,
        copy(s.covariance)
    )
end

# ============================================================================
# Keyframe
# ============================================================================

"""
    Keyframe

A keyframe in the factor graph.

Keyframes are created at regular intervals (0.5-2.0 s per interface contract)
and store the vehicle state plus associated measurements.
"""
mutable struct Keyframe
    index::Int
    timestamp::Float64
    state::UrbanNavState
    measurements::Dict{Symbol, Any}
    factors::Vector{Tuple{Symbol, Int}}
end

function Keyframe(index::Int, timestamp::Float64, state::UrbanNavState)
    Keyframe(index, timestamp, state, Dict{Symbol, Any}(), Tuple{Symbol, Int}[])
end

"""
    add_measurement!(kf::Keyframe, sensor_type::Symbol, measurement)

Add a measurement to this keyframe.
"""
function add_measurement!(kf::Keyframe, sensor_type::Symbol, measurement)
    kf.measurements[sensor_type] = measurement
end

"""
    get_measurement(kf::Keyframe, sensor_type::Symbol)

Get measurement by sensor type.
"""
function get_measurement(kf::Keyframe, sensor_type::Symbol)
    get(kf.measurements, sensor_type, nothing)
end

# ============================================================================
# Factory Functions
# ============================================================================

"""
    create_initial_state(; kwargs...)

Create initial AUV state with specified uncertainties.

# Keyword Arguments
- `position`: Initial position [x, y, z]
- `velocity`: Initial velocity [vx, vy, vz]
- `orientation`: Initial orientation (QuatRotation or vector)
- `position_std`: Initial position uncertainty (meters)
- `velocity_std`: Initial velocity uncertainty (m/s)
- `orientation_std`: Initial orientation uncertainty (radians)
- `bias_gyro_std`: Initial gyro bias uncertainty (rad/s)
- `bias_accel_std`: Initial accel bias uncertainty (m/s²)
"""
function create_initial_state(;
    position = zeros(3),
    velocity = zeros(3),
    orientation = QuatRotation(1.0, 0.0, 0.0, 0.0),
    position_std::Real = 1.0,
    velocity_std::Real = 0.1,
    orientation_std::Real = 0.05,
    bias_gyro_std::Real = 0.001,
    bias_accel_std::Real = 0.01
)
    variances = [
        fill(position_std^2, 3);
        fill(velocity_std^2, 3);
        fill(orientation_std^2, 3);
        fill(bias_gyro_std^2, 3);
        fill(bias_accel_std^2, 3)
    ]
    cov = diagm(variances)

    UrbanNavState(
        position = position,
        velocity = velocity,
        orientation = orientation,
        covariance = cov
    )
end
