"""
    SourceDynamics

Motion models for urban magnetic sources used in the UrbanNav simulation harness.
Provides elevator, vehicle, and door motion models with process noise covariances.
"""
module SourceDynamics

using StaticArrays

export AbstractMotionModel, ElevatorMotion, VehicleMotion, DoorMotion
export step!, process_noise

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

"""
    AbstractMotionModel

Base type for all source motion models. Subtypes must implement:
- `step!(motion, state, dt)` — advance the source state by `dt` seconds
- `process_noise(motion)` — return the process noise covariance matrix
"""
abstract type AbstractMotionModel end

# ---------------------------------------------------------------------------
# Elevator Motion
# ---------------------------------------------------------------------------

"""
    ElevatorPhase

State machine phases for elevator motion.
"""
@enum ElevatorPhase begin
    STOPPED
    ACCELERATING
    CRUISING
    DECELERATING
end

"""
    ElevatorState

Mutable state for an elevator source.

# Fields
- `z::Float64` — current altitude (m)
- `vz::Float64` — current vertical velocity (m/s)
- `phase::ElevatorPhase` — current state-machine phase
- `current_floor::Int` — floor index the elevator is at or heading toward
- `target_floor::Int` — destination floor index
- `phase_timer::Float64` — time spent in current phase (s)
"""
mutable struct ElevatorState
    z::Float64
    vz::Float64
    phase::ElevatorPhase
    current_floor::Int
    target_floor::Int
    phase_timer::Float64
end

"""
    ElevatorMotion <: AbstractMotionModel

Vertical-only motion model for an elevator cabin. The elevator moves between
discrete floor stops using a four-phase state machine:
STOPPED → ACCELERATING → CRUISING → DECELERATING → STOPPED.

# Fields
- `floor_heights::Vector{Float64}` — altitude of each floor (m)
- `max_speed::Float64` — maximum vertical speed (m/s), default 2.0
- `acceleration::Float64` — acceleration/deceleration magnitude (m/s²), default 1.0
- `stop_dwell::Float64` — dwell time at each floor stop (s), default 5.0
- `floor_queue::Vector{Int}` — upcoming floor stops
- `σ_z::Float64` — process noise std dev in z (m), default 0.01
"""
struct ElevatorMotion <: AbstractMotionModel
    floor_heights::Vector{Float64}
    max_speed::Float64
    acceleration::Float64
    stop_dwell::Float64
    floor_queue::Vector{Int}
    σ_z::Float64
end

function ElevatorMotion(floor_heights::Vector{Float64};
                        max_speed::Float64 = 2.0,
                        acceleration::Float64 = 1.0,
                        stop_dwell::Float64 = 5.0,
                        floor_queue::Vector{Int} = Int[],
                        σ_z::Float64 = 0.01)
    ElevatorMotion(floor_heights, max_speed, acceleration, stop_dwell, floor_queue, σ_z)
end

"""
    step!(motion::ElevatorMotion, state::ElevatorState, dt::Float64)

Advance the elevator state by `dt` seconds through the state machine.
"""
function step!(motion::ElevatorMotion, state::ElevatorState, dt::Float64)
    state.phase_timer += dt

    if state.phase == STOPPED
        # Dwell, then pick next target
        if state.phase_timer >= motion.stop_dwell && !isempty(motion.floor_queue)
            state.target_floor = popfirst!(motion.floor_queue)
            state.phase = ACCELERATING
            state.phase_timer = 0.0
        end

    elseif state.phase == ACCELERATING
        dir = sign(motion.floor_heights[state.target_floor] - state.z)
        state.vz += dir * motion.acceleration * dt
        # Clamp to max speed
        if abs(state.vz) >= motion.max_speed
            state.vz = dir * motion.max_speed
            state.phase = CRUISING
            state.phase_timer = 0.0
        end
        state.z += state.vz * dt

    elseif state.phase == CRUISING
        target_z = motion.floor_heights[state.target_floor]
        dist_remaining = abs(target_z - state.z)
        decel_dist = motion.max_speed^2 / (2.0 * motion.acceleration)
        if dist_remaining <= decel_dist
            state.phase = DECELERATING
            state.phase_timer = 0.0
        end
        state.z += state.vz * dt

    elseif state.phase == DECELERATING
        dir = sign(motion.floor_heights[state.target_floor] - state.z)
        state.vz -= dir * motion.acceleration * dt
        state.z += state.vz * dt
        # Check arrival
        target_z = motion.floor_heights[state.target_floor]
        if abs(state.z - target_z) < 0.05 || sign(state.vz) != dir
            state.z = target_z
            state.vz = 0.0
            state.current_floor = state.target_floor
            state.phase = STOPPED
            state.phase_timer = 0.0
        end
    end

    return state
end

"""
    process_noise(motion::ElevatorMotion)

Return the 1×1 process noise covariance for elevator vertical position.
"""
function process_noise(motion::ElevatorMotion)
    return @SMatrix [motion.σ_z^2]
end

# ---------------------------------------------------------------------------
# Vehicle Motion
# ---------------------------------------------------------------------------

"""
    VehicleState

Mutable state for a ground-plane vehicle source.

# Fields
- `pos::SVector{2,Float64}` — (x, y) position (m)
- `heading::Float64` — heading angle (rad)
- `speed::Float64` — forward speed (m/s)
- `waypoint_idx::Int` — index of current target waypoint
"""
mutable struct VehicleState
    pos::SVector{2,Float64}
    heading::Float64
    speed::Float64
    waypoint_idx::Int
end

"""
    VehicleMotion <: AbstractMotionModel

Ground-plane (2D + heading) motion model with waypoint following and speed limits.

# Fields
- `waypoints::Vector{SVector{2,Float64}}` — ordered waypoint positions
- `speed_limit::Float64` — maximum speed (m/s), default 13.4 (~30 mph)
- `cruise_speed::Float64` — nominal cruise speed (m/s), default 8.9 (~20 mph)
- `max_steer_rate::Float64` — maximum heading rate (rad/s), default 0.5
- `arrival_radius::Float64` — waypoint arrival threshold (m), default 2.0
- `σ_xy::Float64` — process noise std dev in x/y (m), default 0.1
- `σ_θ::Float64` — process noise std dev in heading (rad), default 0.01
"""
struct VehicleMotion <: AbstractMotionModel
    waypoints::Vector{SVector{2,Float64}}
    speed_limit::Float64
    cruise_speed::Float64
    max_steer_rate::Float64
    arrival_radius::Float64
    σ_xy::Float64
    σ_θ::Float64
end

function VehicleMotion(waypoints::Vector{SVector{2,Float64}};
                       speed_limit::Float64 = 13.4,
                       cruise_speed::Float64 = 8.9,
                       max_steer_rate::Float64 = 0.5,
                       arrival_radius::Float64 = 2.0,
                       σ_xy::Float64 = 0.1,
                       σ_θ::Float64 = 0.01)
    VehicleMotion(waypoints, speed_limit, cruise_speed, max_steer_rate, arrival_radius, σ_xy, σ_θ)
end

"""
    step!(motion::VehicleMotion, state::VehicleState, dt::Float64)

Advance the vehicle state by `dt` seconds. Steers toward the current waypoint
and advances the waypoint index upon arrival.
"""
function step!(motion::VehicleMotion, state::VehicleState, dt::Float64)
    if state.waypoint_idx > length(motion.waypoints)
        state.speed = 0.0
        return state
    end

    target = motion.waypoints[state.waypoint_idx]
    diff = target - state.pos
    dist = norm(diff)

    # Check waypoint arrival
    if dist < motion.arrival_radius
        state.waypoint_idx += 1
        if state.waypoint_idx > length(motion.waypoints)
            state.speed = 0.0
            return state
        end
        target = motion.waypoints[state.waypoint_idx]
        diff = target - state.pos
        dist = norm(diff)
    end

    # Steer toward target
    desired_heading = atan(diff[2], diff[1])
    heading_error = mod(desired_heading - state.heading + π, 2π) - π
    steer = clamp(heading_error / dt, -motion.max_steer_rate, motion.max_steer_rate)
    state.heading += steer * dt

    # Speed control
    state.speed = min(motion.cruise_speed, motion.speed_limit)

    # Update position
    state.pos = state.pos + SVector(cos(state.heading), sin(state.heading)) * state.speed * dt

    return state
end

"""
    process_noise(motion::VehicleMotion)

Return the 3×3 process noise covariance for vehicle (x, y, heading).
"""
function process_noise(motion::VehicleMotion)
    return @SMatrix [motion.σ_xy^2  0.0           0.0;
                     0.0            motion.σ_xy^2 0.0;
                     0.0            0.0           motion.σ_θ^2]
end

# ---------------------------------------------------------------------------
# Door Motion
# ---------------------------------------------------------------------------

"""
    DoorState

Mutable state for a door source.

# Fields
- `fraction::Float64` — open fraction in [0, 1] (0 = closed, 1 = open)
- `target::Float64` — target fraction (0.0 or 1.0)
- `transitioning::Bool` — whether the door is currently moving
"""
mutable struct DoorState
    fraction::Float64
    target::Float64
    transitioning::Bool
end

"""
    DoorMotion <: AbstractMotionModel

Binary open/close door model with a linear transition time.

# Fields
- `transition_time::Float64` — time to fully open or close (s), default 2.0
- `σ_door::Float64` — process noise std dev on open fraction, default 0.005
"""
struct DoorMotion <: AbstractMotionModel
    transition_time::Float64
    σ_door::Float64
end

function DoorMotion(; transition_time::Float64 = 2.0, σ_door::Float64 = 0.005)
    DoorMotion(transition_time, σ_door)
end

"""
    step!(motion::DoorMotion, state::DoorState, dt::Float64)

Advance the door state by `dt` seconds. Linearly interpolates the open
fraction toward the target.
"""
function step!(motion::DoorMotion, state::DoorState, dt::Float64)
    if state.fraction ≈ state.target
        state.transitioning = false
        return state
    end

    state.transitioning = true
    rate = 1.0 / motion.transition_time
    dir = sign(state.target - state.fraction)
    state.fraction += dir * rate * dt
    state.fraction = clamp(state.fraction, 0.0, 1.0)

    # Snap to target if close enough
    if abs(state.fraction - state.target) < rate * dt
        state.fraction = state.target
        state.transitioning = false
    end

    return state
end

"""
    process_noise(motion::DoorMotion)

Return the 1×1 process noise covariance for door open fraction.
"""
function process_noise(motion::DoorMotion)
    return @SMatrix [motion.σ_door^2]
end

# Bring norm into scope for VehicleMotion
using LinearAlgebra: norm

end # module SourceDynamics
