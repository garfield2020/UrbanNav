"""
    ElevatorDOETrajectories

Six trajectory archetypes for elevator DOE testing. Each represents a different
pedestrian movement pattern near elevator shafts, designed to exercise different
observability and interference regimes.

All types implement the `AbstractTrajectory` interface:
- `position(traj, t)::SVector{3,Float64}`
- `velocity(traj, t)::SVector{3,Float64}`
- `duration(traj)::Float64`
"""
module ElevatorDOETrajectories

using StaticArrays
using LinearAlgebra: norm

export AbstractTrajectory
export CorridorParallel, PerpendicularCrossing, ShaftLoop
export StopAndGo, MultiFloorWalk, DualShaftPath
export position, velocity, duration

abstract type AbstractTrajectory end

# ============================================================================
# 1. CorridorParallel — walk parallel to shaft at configurable offset
# ============================================================================

"""
    CorridorParallel <: AbstractTrajectory

Walk parallel to an elevator shaft along the y-axis at a fixed x-offset.
Tests sustained exposure to the elevator magnetic field.

# Fields
- `shaft_xy::SVector{2,Float64}`: Shaft center position.
- `offset::Float64`: Perpendicular distance from shaft (m).
- `walk_length::Float64`: Length of corridor walk (m).
- `speed::Float64`: Walking speed (m/s).
- `num_laps::Int`: Number of back-and-forth laps.
- `z::Float64`: Walking altitude (m).
"""
struct CorridorParallel <: AbstractTrajectory
    shaft_xy::SVector{2,Float64}
    offset::Float64
    walk_length::Float64
    speed::Float64
    num_laps::Int
    z::Float64
end

function CorridorParallel(shaft_xy::SVector{2,Float64};
                          offset::Float64 = 3.0,
                          walk_length::Float64 = 40.0,
                          speed::Float64 = 1.3,
                          num_laps::Int = 3,
                          z::Float64 = 0.0)
    CorridorParallel(shaft_xy, offset, walk_length, speed, num_laps, z)
end

function duration(traj::CorridorParallel)::Float64
    return 2.0 * traj.walk_length * traj.num_laps / traj.speed
end

function position(traj::CorridorParallel, t::Float64)::SVector{3,Float64}
    t = clamp(t, 0.0, duration(traj))
    leg_time = traj.walk_length / traj.speed
    cycle = 2.0 * leg_time
    phase = mod(t, cycle)
    x0 = traj.shaft_xy[1] + traj.offset
    y_start = traj.shaft_xy[2] - traj.walk_length / 2.0
    if phase <= leg_time
        y = y_start + traj.walk_length * (phase / leg_time)
    else
        y = y_start + traj.walk_length * (1.0 - (phase - leg_time) / leg_time)
    end
    return SVector(x0, y, traj.z)
end

function velocity(traj::CorridorParallel, t::Float64)::SVector{3,Float64}
    t = clamp(t, 0.0, duration(traj))
    leg_time = traj.walk_length / traj.speed
    cycle = 2.0 * leg_time
    phase = mod(t, cycle)
    if phase <= leg_time
        return SVector(0.0, traj.speed, 0.0)
    else
        return SVector(0.0, -traj.speed, 0.0)
    end
end

# ============================================================================
# 2. PerpendicularCrossing — cross near shaft perpendicularly
# ============================================================================

"""
    PerpendicularCrossing <: AbstractTrajectory

Walk perpendicular to the shaft, crossing near it repeatedly.
Tests transient exposure as pedestrian passes through the field.
"""
struct PerpendicularCrossing <: AbstractTrajectory
    shaft_xy::SVector{2,Float64}
    crossing_length::Float64
    speed::Float64
    num_crossings::Int
    spacing::Float64  # y-spacing between crossings
    z::Float64
end

function PerpendicularCrossing(shaft_xy::SVector{2,Float64};
                                crossing_length::Float64 = 30.0,
                                speed::Float64 = 1.3,
                                num_crossings::Int = 5,
                                spacing::Float64 = 3.0,
                                z::Float64 = 0.0)
    PerpendicularCrossing(shaft_xy, crossing_length, speed, num_crossings, spacing, z)
end

function duration(traj::PerpendicularCrossing)::Float64
    cross_time = traj.crossing_length / traj.speed
    # Each crossing + turn to next lane
    turn_time = traj.spacing / traj.speed
    return traj.num_crossings * cross_time + (traj.num_crossings - 1) * turn_time
end

function _crossing_segment(traj::PerpendicularCrossing, t::Float64)
    cross_time = traj.crossing_length / traj.speed
    turn_time = traj.spacing / traj.speed
    segment_time = cross_time + turn_time

    crossing_idx = 0
    remaining = t
    for i in 1:traj.num_crossings
        if remaining <= cross_time
            return (i, :crossing, remaining)
        end
        remaining -= cross_time
        if i < traj.num_crossings
            if remaining <= turn_time
                return (i, :turn, remaining)
            end
            remaining -= turn_time
        end
    end
    return (traj.num_crossings, :crossing, cross_time)
end

function position(traj::PerpendicularCrossing, t::Float64)::SVector{3,Float64}
    t = clamp(t, 0.0, duration(traj))
    idx, phase, rem = _crossing_segment(traj, t)

    cross_time = traj.crossing_length / traj.speed
    turn_time = traj.spacing / traj.speed
    x_start = traj.shaft_xy[1] - traj.crossing_length / 2.0
    y_base = traj.shaft_xy[2] + (idx - 1) * traj.spacing

    direction = iseven(idx) ? -1.0 : 1.0

    if phase == :crossing
        frac = rem / cross_time
        x = direction > 0 ? x_start + traj.crossing_length * frac :
                            x_start + traj.crossing_length * (1.0 - frac)
        return SVector(x, y_base, traj.z)
    else
        # Turn: move in y
        x_end = direction > 0 ? x_start + traj.crossing_length : x_start
        frac = rem / turn_time
        y = y_base + traj.spacing * frac
        return SVector(x_end, y, traj.z)
    end
end

function velocity(traj::PerpendicularCrossing, t::Float64)::SVector{3,Float64}
    t = clamp(t, 0.0, duration(traj))
    idx, phase, _ = _crossing_segment(traj, t)
    direction = iseven(idx) ? -1.0 : 1.0

    if phase == :crossing
        return SVector(direction * traj.speed, 0.0, 0.0)
    else
        return SVector(0.0, traj.speed, 0.0)
    end
end

# ============================================================================
# 3. ShaftLoop — circular loop around shaft (best observability)
# ============================================================================

"""
    ShaftLoop <: AbstractTrajectory

Walk in a circular loop around the elevator shaft.
Provides best observability of the dipole field from all angles.
"""
struct ShaftLoop <: AbstractTrajectory
    shaft_xy::SVector{2,Float64}
    radius::Float64
    speed::Float64
    num_loops::Int
    z::Float64
end

function ShaftLoop(shaft_xy::SVector{2,Float64};
                   radius::Float64 = 5.0,
                   speed::Float64 = 1.2,
                   num_loops::Int = 3,
                   z::Float64 = 0.0)
    ShaftLoop(shaft_xy, radius, speed, num_loops, z)
end

function duration(traj::ShaftLoop)::Float64
    circumference = 2π * traj.radius
    return circumference * traj.num_loops / traj.speed
end

function position(traj::ShaftLoop, t::Float64)::SVector{3,Float64}
    t = clamp(t, 0.0, duration(traj))
    dur = duration(traj)
    θ = 2π * traj.num_loops * (t / dur)
    x = traj.shaft_xy[1] + traj.radius * cos(θ)
    y = traj.shaft_xy[2] + traj.radius * sin(θ)
    return SVector(x, y, traj.z)
end

function velocity(traj::ShaftLoop, t::Float64)::SVector{3,Float64}
    t = clamp(t, 0.0, duration(traj))
    dur = duration(traj)
    θ = 2π * traj.num_loops * (t / dur)
    ω = 2π * traj.num_loops / dur
    vx = -traj.radius * sin(θ) * ω
    vy = traj.radius * cos(θ) * ω
    return SVector(vx, vy, 0.0)
end

# ============================================================================
# 4. StopAndGo — waypoint patrol with pauses near shaft
# ============================================================================

"""
    StopAndGo <: AbstractTrajectory

Walk between waypoints with pauses, spending time stationary near the shaft.
Tests map poisoning risk from prolonged static exposure.
"""
struct StopAndGo <: AbstractTrajectory
    waypoints::Vector{SVector{3,Float64}}
    speed::Float64
    pause_time::Float64
    _cumulative_times::Vector{Float64}
end

function StopAndGo(shaft_xy::SVector{2,Float64};
                   speed::Float64 = 1.3,
                   pause_time::Float64 = 10.0,
                   offset::Float64 = 3.0,
                   z::Float64 = 0.0)
    # Generate waypoints: approach shaft, pause near it, move away, return
    wps = SVector{3,Float64}[
        SVector(shaft_xy[1] + 15.0, shaft_xy[2], z),
        SVector(shaft_xy[1] + offset, shaft_xy[2], z),
        SVector(shaft_xy[1] + offset, shaft_xy[2] + 10.0, z),
        SVector(shaft_xy[1] + 15.0, shaft_xy[2] + 10.0, z),
        SVector(shaft_xy[1] + offset, shaft_xy[2] + 10.0, z),
        SVector(shaft_xy[1] + offset, shaft_xy[2], z),
    ]

    # Compute cumulative times
    times = Float64[0.0]
    for i in 2:length(wps)
        walk_t = norm(wps[i] - wps[i-1]) / speed
        push!(times, times[end] + walk_t)
        push!(times, times[end] + pause_time)
    end

    StopAndGo(wps, speed, pause_time, times)
end

function duration(traj::StopAndGo)::Float64
    return traj._cumulative_times[end]
end

function position(traj::StopAndGo, t::Float64)::SVector{3,Float64}
    t = clamp(t, 0.0, duration(traj))
    times = traj._cumulative_times

    for i in 2:length(traj.waypoints)
        walk_start = times[2*(i-1) - 1]
        walk_end = times[2*(i-1)]
        pause_end = times[2*(i-1) + 1]

        if t <= walk_end
            frac = (walk_end - walk_start) > 0 ? (t - walk_start) / (walk_end - walk_start) : 1.0
            return traj.waypoints[i-1] + (traj.waypoints[i] - traj.waypoints[i-1]) * frac
        end
        if t <= pause_end
            return traj.waypoints[i]
        end
    end
    return traj.waypoints[end]
end

function velocity(traj::StopAndGo, t::Float64)::SVector{3,Float64}
    t = clamp(t, 0.0, duration(traj))
    times = traj._cumulative_times
    zero3 = SVector(0.0, 0.0, 0.0)

    for i in 2:length(traj.waypoints)
        walk_start = times[2*(i-1) - 1]
        walk_end = times[2*(i-1)]
        pause_end = times[2*(i-1) + 1]

        if t <= walk_end
            d = traj.waypoints[i] - traj.waypoints[i-1]
            dist = norm(d)
            return dist > 0 ? d / dist * traj.speed : zero3
        end
        if t <= pause_end
            return zero3
        end
    end
    return zero3
end

# ============================================================================
# 5. MultiFloorWalk — walk on two floors connected by stairs/ramp
# ============================================================================

"""
    MultiFloorWalk <: AbstractTrajectory

Walk on two floors, transitioning via stairs/ramp. Tests the elevator
field at multiple altitudes as the person changes floors.
"""
struct MultiFloorWalk <: AbstractTrajectory
    shaft_xy::SVector{2,Float64}
    floor_height::Float64
    walk_length::Float64
    speed::Float64
    offset::Float64
    stair_x_offset::Float64
end

function MultiFloorWalk(shaft_xy::SVector{2,Float64};
                        floor_height::Float64 = 3.5,
                        walk_length::Float64 = 30.0,
                        speed::Float64 = 1.2,
                        offset::Float64 = 3.0,
                        stair_x_offset::Float64 = 20.0)
    MultiFloorWalk(shaft_xy, floor_height, walk_length, speed, offset, stair_x_offset)
end

function duration(traj::MultiFloorWalk)::Float64
    # Walk on floor 1 → stairs → walk on floor 2 → stairs → walk on floor 1
    horiz = traj.walk_length / traj.speed
    stair_dist = sqrt(traj.stair_x_offset^2 + traj.floor_height^2)
    stair_time = stair_dist / (traj.speed * 0.7)  # slower on stairs
    return 3.0 * horiz + 2.0 * stair_time
end

function position(traj::MultiFloorWalk, t::Float64)::SVector{3,Float64}
    t = clamp(t, 0.0, duration(traj))
    horiz_time = traj.walk_length / traj.speed
    stair_dist = sqrt(traj.stair_x_offset^2 + traj.floor_height^2)
    stair_time = stair_dist / (traj.speed * 0.7)

    x_base = traj.shaft_xy[1] + traj.offset
    y_base = traj.shaft_xy[2]

    # Segment 1: Walk on floor 1
    if t <= horiz_time
        frac = t / horiz_time
        return SVector(x_base, y_base + traj.walk_length * frac, 0.0)
    end
    t -= horiz_time

    # Segment 2: Stairs up
    if t <= stair_time
        frac = t / stair_time
        return SVector(x_base + traj.stair_x_offset * frac,
                       y_base + traj.walk_length,
                       traj.floor_height * frac)
    end
    t -= stair_time

    # Segment 3: Walk on floor 2
    if t <= horiz_time
        frac = t / horiz_time
        return SVector(x_base + traj.stair_x_offset,
                       y_base + traj.walk_length * (1.0 - frac),
                       traj.floor_height)
    end
    t -= horiz_time

    # Segment 4: Stairs down
    if t <= stair_time
        frac = t / stair_time
        return SVector(x_base + traj.stair_x_offset * (1.0 - frac),
                       y_base,
                       traj.floor_height * (1.0 - frac))
    end
    t -= stair_time

    # Segment 5: Walk on floor 1 again
    frac = min(t / horiz_time, 1.0)
    return SVector(x_base, y_base + traj.walk_length * frac, 0.0)
end

function velocity(traj::MultiFloorWalk, t::Float64)::SVector{3,Float64}
    # Numerical derivative
    dt = 1e-4
    t = clamp(t, 0.0, duration(traj))
    t1 = max(t - dt/2, 0.0)
    t2 = min(t + dt/2, duration(traj))
    actual_dt = t2 - t1
    if actual_dt < 1e-10
        return SVector(0.0, 0.0, 0.0)
    end
    return (position(traj, t2) - position(traj, t1)) / actual_dt
end

# ============================================================================
# 6. DualShaftPath — path encountering two elevator shafts
# ============================================================================

"""
    DualShaftPath <: AbstractTrajectory

Walk a path that passes near two elevator shafts in sequence.
Tests multi-source interference scenarios.
"""
struct DualShaftPath <: AbstractTrajectory
    shaft1_xy::SVector{2,Float64}
    shaft2_xy::SVector{2,Float64}
    offset::Float64
    speed::Float64
    pause_time::Float64
    z::Float64
    _waypoints::Vector{SVector{3,Float64}}
    _cumulative_times::Vector{Float64}
end

function DualShaftPath(shaft1_xy::SVector{2,Float64},
                       shaft2_xy::SVector{2,Float64};
                       offset::Float64 = 3.0,
                       speed::Float64 = 1.3,
                       pause_time::Float64 = 5.0,
                       z::Float64 = 0.0)
    wps = SVector{3,Float64}[
        SVector(shaft1_xy[1] - 10.0, shaft1_xy[2], z),
        SVector(shaft1_xy[1] + offset, shaft1_xy[2], z),
        SVector(shaft2_xy[1] + offset, shaft2_xy[2], z),
        SVector(shaft2_xy[1] - 10.0, shaft2_xy[2], z),
    ]

    times = Float64[0.0]
    for i in 2:length(wps)
        walk_t = norm(wps[i] - wps[i-1]) / speed
        push!(times, times[end] + walk_t)
        push!(times, times[end] + pause_time)
    end

    DualShaftPath(shaft1_xy, shaft2_xy, offset, speed, pause_time, z, wps, times)
end

function duration(traj::DualShaftPath)::Float64
    return traj._cumulative_times[end]
end

function position(traj::DualShaftPath, t::Float64)::SVector{3,Float64}
    t = clamp(t, 0.0, duration(traj))
    times = traj._cumulative_times
    wps = traj._waypoints

    for i in 2:length(wps)
        walk_start = times[2*(i-1) - 1]
        walk_end = times[2*(i-1)]
        pause_end = times[2*(i-1) + 1]

        if t <= walk_end
            frac = (walk_end - walk_start) > 0 ? (t - walk_start) / (walk_end - walk_start) : 1.0
            return wps[i-1] + (wps[i] - wps[i-1]) * frac
        end
        if t <= pause_end
            return wps[i]
        end
    end
    return wps[end]
end

function velocity(traj::DualShaftPath, t::Float64)::SVector{3,Float64}
    t = clamp(t, 0.0, duration(traj))
    times = traj._cumulative_times
    wps = traj._waypoints
    zero3 = SVector(0.0, 0.0, 0.0)

    for i in 2:length(wps)
        walk_start = times[2*(i-1) - 1]
        walk_end = times[2*(i-1)]
        pause_end = times[2*(i-1) + 1]

        if t <= walk_end
            d = wps[i] - wps[i-1]
            dist = norm(d)
            return dist > 0 ? d / dist * traj.speed : zero3
        end
        if t <= pause_end
            return zero3
        end
    end
    return zero3
end

end # module ElevatorDOETrajectories
