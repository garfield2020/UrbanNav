"""
    UrbanTrajectories

Urban navigation trajectories for testing the UrbanNav simulation harness.
Provides parameterized trajectory generators for common indoor/urban movement
patterns including hallway patrols, elevator rides, spiral ramps, and lobby walks.
"""
module UrbanTrajectories

using StaticArrays
using LinearAlgebra: norm

export AbstractTrajectory, HallwayPatrol, ElevatorRide, SpiralRamp, LobbyPatrol
export position, velocity, duration

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

"""
    AbstractTrajectory

Base type for all test trajectories. Subtypes must implement:
- `position(traj, t)::SVector{3,Float64}` — 3D position at time `t`
- `velocity(traj, t)::SVector{3,Float64}` — 3D velocity at time `t`
- `duration(traj)::Float64` — total trajectory duration in seconds
"""
abstract type AbstractTrajectory end

# ---------------------------------------------------------------------------
# Hallway Patrol
# ---------------------------------------------------------------------------

"""
    HallwayPatrol <: AbstractTrajectory

Walk back and forth along a straight corridor at pedestrian speed.

# Fields
- `start_pos::SVector{3,Float64}` — one end of the corridor
- `end_pos::SVector{3,Float64}` — other end of the corridor
- `speed::Float64` — walking speed (m/s), default 1.3
- `num_laps::Int` — number of round trips, default 3
"""
struct HallwayPatrol <: AbstractTrajectory
    start_pos::SVector{3,Float64}
    end_pos::SVector{3,Float64}
    speed::Float64
    num_laps::Int
end

function HallwayPatrol(start_pos::SVector{3,Float64}, end_pos::SVector{3,Float64};
                       speed::Float64 = 1.3, num_laps::Int = 3)
    HallwayPatrol(start_pos, end_pos, speed, num_laps)
end

function duration(traj::HallwayPatrol)::Float64
    leg = norm(traj.end_pos - traj.start_pos)
    return 2.0 * leg * traj.num_laps / traj.speed
end

function position(traj::HallwayPatrol, t::Float64)::SVector{3,Float64}
    t = clamp(t, 0.0, duration(traj))
    leg = norm(traj.end_pos - traj.start_pos)
    leg_time = leg / traj.speed
    dir = (traj.end_pos - traj.start_pos) / leg

    # Determine position within a back-and-forth cycle
    cycle_time = 2.0 * leg_time
    phase = mod(t, cycle_time)
    if phase <= leg_time
        frac = phase / leg_time
        return traj.start_pos + dir * leg * frac
    else
        frac = (phase - leg_time) / leg_time
        return traj.end_pos - dir * leg * frac
    end
end

function velocity(traj::HallwayPatrol, t::Float64)::SVector{3,Float64}
    t = clamp(t, 0.0, duration(traj))
    leg = norm(traj.end_pos - traj.start_pos)
    leg_time = leg / traj.speed
    dir = (traj.end_pos - traj.start_pos) / leg

    cycle_time = 2.0 * leg_time
    phase = mod(t, cycle_time)
    if phase <= leg_time
        return dir * traj.speed
    else
        return -dir * traj.speed
    end
end

# ---------------------------------------------------------------------------
# Elevator Ride
# ---------------------------------------------------------------------------

"""
    ElevatorRide <: AbstractTrajectory

Walk to an elevator, ride up or down to another floor, then walk out.
The trajectory has three segments:
1. Walk from `start_pos` to `elevator_pos` on the origin floor.
2. Vertical ride from `start_floor_z` to `end_floor_z`.
3. Walk from elevator to `exit_pos` on the destination floor.

# Fields
- `start_pos::SVector{3,Float64}` — starting position (z = start floor altitude)
- `elevator_pos::SVector{2,Float64}` — (x, y) of elevator on both floors
- `exit_pos::SVector{3,Float64}` — destination position (z = end floor altitude)
- `start_floor_z::Float64` — altitude of origin floor (m)
- `end_floor_z::Float64` — altitude of destination floor (m)
- `walk_speed::Float64` — horizontal walking speed (m/s), default 1.3
- `elevator_speed::Float64` — vertical elevator speed (m/s), default 2.0
- `door_wait::Float64` — wait time at elevator doors (s), default 5.0
"""
struct ElevatorRide <: AbstractTrajectory
    start_pos::SVector{3,Float64}
    elevator_pos::SVector{2,Float64}
    exit_pos::SVector{3,Float64}
    start_floor_z::Float64
    end_floor_z::Float64
    walk_speed::Float64
    elevator_speed::Float64
    door_wait::Float64
end

function ElevatorRide(start_pos::SVector{3,Float64},
                      elevator_pos::SVector{2,Float64},
                      exit_pos::SVector{3,Float64};
                      walk_speed::Float64 = 1.3,
                      elevator_speed::Float64 = 2.0,
                      door_wait::Float64 = 5.0)
    ElevatorRide(start_pos, elevator_pos, exit_pos,
                 start_pos[3], exit_pos[3],
                 walk_speed, elevator_speed, door_wait)
end

# Segment durations
function _segment_times(traj::ElevatorRide)
    walk_in_dist = norm(SVector(traj.start_pos[1], traj.start_pos[2]) - traj.elevator_pos)
    walk_out_dist = norm(SVector(traj.exit_pos[1], traj.exit_pos[2]) - traj.elevator_pos)
    ride_dist = abs(traj.end_floor_z - traj.start_floor_z)

    t_walk_in  = walk_in_dist / traj.walk_speed
    t_wait1    = traj.door_wait
    t_ride     = ride_dist / traj.elevator_speed
    t_wait2    = traj.door_wait
    t_walk_out = walk_out_dist / traj.walk_speed

    return (t_walk_in, t_wait1, t_ride, t_wait2, t_walk_out)
end

function duration(traj::ElevatorRide)::Float64
    segs = _segment_times(traj)
    return sum(segs)
end

function position(traj::ElevatorRide, t::Float64)::SVector{3,Float64}
    t = clamp(t, 0.0, duration(traj))
    segs = _segment_times(traj)
    t_walk_in, t_wait1, t_ride, t_wait2, t_walk_out = segs

    elev3_start = SVector(traj.elevator_pos[1], traj.elevator_pos[2], traj.start_floor_z)
    elev3_end   = SVector(traj.elevator_pos[1], traj.elevator_pos[2], traj.end_floor_z)

    # Segment 1: walk to elevator
    if t <= t_walk_in
        frac = t_walk_in > 0 ? t / t_walk_in : 1.0
        return traj.start_pos + (elev3_start - traj.start_pos) * frac
    end
    t -= t_walk_in

    # Segment 2: wait at elevator door
    if t <= t_wait1
        return elev3_start
    end
    t -= t_wait1

    # Segment 3: ride
    if t <= t_ride
        frac = t_ride > 0 ? t / t_ride : 1.0
        return elev3_start + (elev3_end - elev3_start) * frac
    end
    t -= t_ride

    # Segment 4: wait at destination door
    if t <= t_wait2
        return elev3_end
    end
    t -= t_wait2

    # Segment 5: walk out
    frac = t_walk_out > 0 ? min(t / t_walk_out, 1.0) : 1.0
    return elev3_end + (traj.exit_pos - elev3_end) * frac
end

function velocity(traj::ElevatorRide, t::Float64)::SVector{3,Float64}
    t = clamp(t, 0.0, duration(traj))
    segs = _segment_times(traj)
    t_walk_in, t_wait1, t_ride, t_wait2, t_walk_out = segs

    elev3_start = SVector(traj.elevator_pos[1], traj.elevator_pos[2], traj.start_floor_z)
    elev3_end   = SVector(traj.elevator_pos[1], traj.elevator_pos[2], traj.end_floor_z)

    zero3 = SVector(0.0, 0.0, 0.0)

    if t <= t_walk_in
        d = elev3_start - traj.start_pos
        dist = norm(d)
        return dist > 0 ? d / dist * traj.walk_speed : zero3
    end
    t -= t_walk_in

    if t <= t_wait1
        return zero3
    end
    t -= t_wait1

    if t <= t_ride
        dz = traj.end_floor_z - traj.start_floor_z
        return SVector(0.0, 0.0, sign(dz) * traj.elevator_speed)
    end
    t -= t_ride

    if t <= t_wait2
        return zero3
    end

    d = traj.exit_pos - elev3_end
    dist = norm(d)
    return dist > 0 ? d / dist * traj.walk_speed : zero3
end

# ---------------------------------------------------------------------------
# Spiral Ramp
# ---------------------------------------------------------------------------

"""
    SpiralRamp <: AbstractTrajectory

Walk up or down a parking-garage spiral ramp with continuous altitude change.

# Fields
- `center::SVector{2,Float64}` — (x, y) center of the spiral
- `radius::Float64` — ramp radius (m), default 10.0
- `start_z::Float64` — starting altitude (m)
- `end_z::Float64` — ending altitude (m)
- `num_turns::Float64` — number of full turns, default 3.0
- `speed::Float64` — walking speed along the ramp (m/s), default 1.1
"""
struct SpiralRamp <: AbstractTrajectory
    center::SVector{2,Float64}
    radius::Float64
    start_z::Float64
    end_z::Float64
    num_turns::Float64
    speed::Float64
end

function SpiralRamp(center::SVector{2,Float64};
                    radius::Float64 = 10.0,
                    start_z::Float64 = 0.0,
                    end_z::Float64 = 9.0,
                    num_turns::Float64 = 3.0,
                    speed::Float64 = 1.1)
    SpiralRamp(center, radius, start_z, end_z, num_turns, speed)
end

function _ramp_arc_length(traj::SpiralRamp)
    dz = traj.end_z - traj.start_z
    circ = 2π * traj.radius * traj.num_turns
    return sqrt(circ^2 + dz^2)
end

function duration(traj::SpiralRamp)::Float64
    return _ramp_arc_length(traj) / traj.speed
end

function position(traj::SpiralRamp, t::Float64)::SVector{3,Float64}
    t = clamp(t, 0.0, duration(traj))
    dur = duration(traj)
    frac = dur > 0 ? t / dur : 0.0
    θ = 2π * traj.num_turns * frac
    z = traj.start_z + (traj.end_z - traj.start_z) * frac
    x = traj.center[1] + traj.radius * cos(θ)
    y = traj.center[2] + traj.radius * sin(θ)
    return SVector(x, y, z)
end

function velocity(traj::SpiralRamp, t::Float64)::SVector{3,Float64}
    t = clamp(t, 0.0, duration(traj))
    dur = duration(traj)
    frac = dur > 0 ? t / dur : 0.0
    θ = 2π * traj.num_turns * frac
    dθ_dt = dur > 0 ? 2π * traj.num_turns / dur : 0.0
    dz_dt = dur > 0 ? (traj.end_z - traj.start_z) / dur : 0.0

    vx = -traj.radius * sin(θ) * dθ_dt
    vy =  traj.radius * cos(θ) * dθ_dt
    return SVector(vx, vy, dz_dt)
end

# ---------------------------------------------------------------------------
# Lobby Patrol
# ---------------------------------------------------------------------------

"""
    LobbyPatrol <: AbstractTrajectory

Random walk within an open rectangular lobby. Uses a deterministic sequence of
pre-generated waypoints for reproducibility.

# Fields
- `origin::SVector{3,Float64}` — corner of the lobby (minimum x, y, z)
- `extents::SVector{2,Float64}` — lobby width and depth (m)
- `speed::Float64` — walking speed (m/s), default 1.0
- `waypoints::Vector{SVector{3,Float64}}` — pre-generated waypoint sequence
- `pause_time::Float64` — pause at each waypoint (s), default 2.0
"""
struct LobbyPatrol <: AbstractTrajectory
    origin::SVector{3,Float64}
    extents::SVector{2,Float64}
    speed::Float64
    waypoints::Vector{SVector{3,Float64}}
    pause_time::Float64
end

"""
    LobbyPatrol(origin, extents; speed, num_waypoints, pause_time, seed)

Construct a `LobbyPatrol` with deterministically generated waypoints.
Uses a simple linear congruential generator seeded by `seed` for reproducibility.
"""
function LobbyPatrol(origin::SVector{3,Float64},
                     extents::SVector{2,Float64};
                     speed::Float64 = 1.0,
                     num_waypoints::Int = 10,
                     pause_time::Float64 = 2.0,
                     seed::UInt64 = UInt64(42))
    # Simple LCG for deterministic waypoints (no Random dependency)
    s = seed
    wps = Vector{SVector{3,Float64}}(undef, num_waypoints)
    for i in 1:num_waypoints
        s = s * UInt64(6364136223846793005) + UInt64(1)
        rx = (Float64((s >> 33) & 0x7FFFFFFF) / Float64(0x7FFFFFFF))
        s = s * UInt64(6364136223846793005) + UInt64(1)
        ry = (Float64((s >> 33) & 0x7FFFFFFF) / Float64(0x7FFFFFFF))
        wps[i] = SVector(origin[1] + extents[1] * rx,
                         origin[2] + extents[2] * ry,
                         origin[3])
    end
    LobbyPatrol(origin, extents, speed, wps, pause_time)
end

function _segment_data(traj::LobbyPatrol)
    # Returns cumulative time at the start of each segment and segment info
    times = Float64[0.0]
    n = length(traj.waypoints)
    prev = traj.waypoints[1]
    # Start with a pause at the first waypoint
    push!(times, traj.pause_time)
    for i in 2:n
        wp = traj.waypoints[i]
        d = norm(wp - prev)
        walk_t = d / traj.speed
        push!(times, times[end] + walk_t)          # arrive at wp
        push!(times, times[end] + traj.pause_time)  # pause at wp
        prev = wp
    end
    return times
end

function duration(traj::LobbyPatrol)::Float64
    times = _segment_data(traj)
    return times[end]
end

function position(traj::LobbyPatrol, t::Float64)::SVector{3,Float64}
    t = clamp(t, 0.0, duration(traj))
    times = _segment_data(traj)
    n = length(traj.waypoints)

    # First pause
    if t <= times[2]
        return traj.waypoints[1]
    end

    idx = 2  # times index
    for i in 2:n
        walk_start = times[idx + 1]  # end of previous pause = start of walk
        # Actually: times layout is [0, pause_end, arrive_wp2, pause_wp2, arrive_wp3, ...]
        # idx=2 is end of first pause. idx=3 is arrive wp2. idx=4 is pause wp2 end, etc.
        t_walk_end = times[idx + 1]
        t_pause_end = times[idx + 2]

        if t <= t_walk_end
            # Walking from waypoint i-1 to waypoint i
            prev_pause_end = times[idx]
            walk_dur = t_walk_end - prev_pause_end
            frac = walk_dur > 0 ? (t - prev_pause_end) / walk_dur : 1.0
            return traj.waypoints[i-1] + (traj.waypoints[i] - traj.waypoints[i-1]) * frac
        end

        if t <= t_pause_end
            return traj.waypoints[i]
        end

        idx += 2
    end

    return traj.waypoints[end]
end

function velocity(traj::LobbyPatrol, t::Float64)::SVector{3,Float64}
    t = clamp(t, 0.0, duration(traj))
    times = _segment_data(traj)
    n = length(traj.waypoints)
    zero3 = SVector(0.0, 0.0, 0.0)

    if t <= times[2]
        return zero3
    end

    idx = 2
    for i in 2:n
        t_walk_end = times[idx + 1]
        t_pause_end = times[idx + 2]

        if t <= t_walk_end
            d = traj.waypoints[i] - traj.waypoints[i-1]
            dist = norm(d)
            return dist > 0 ? d / dist * traj.speed : zero3
        end

        if t <= t_pause_end
            return zero3
        end

        idx += 2
    end

    return zero3
end

end # module UrbanTrajectories
