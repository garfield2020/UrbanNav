module ElevatorWorldModule

using StaticArrays
using Random
using LinearAlgebra: dot

export ElevatorWorld, ElevatorState, ElevatorPhase,
       step!, magnetic_field, create_default_elevator_world

# Physical constants
const μ₀_over_4π = 1e-7  # μ₀/(4π) in T·m/A

"""
    ElevatorPhase

State machine phases for elevator vertical motion.
"""
@enum ElevatorPhase begin
    STOPPED
    ACCELERATING
    CRUISING
    DECELERATING
end

"""
    ElevatorState

State of a single elevator car within its shaft.

# Fields
- `position::SVector{3,Float64}`: 3D position of the elevator car center [m].
- `velocity::Float64`: Vertical velocity (positive = up) [m/s].
- `current_floor::Int`: Floor the elevator is at or heading toward.
- `target_floor::Int`: Destination floor.
- `phase::ElevatorPhase`: Current motion phase.
- `dipole_moment::SVector{3,Float64}`: Magnetic dipole moment vector [A·m²].
- `max_velocity::Float64`: Maximum cruise speed [m/s].
- `acceleration::Float64`: Acceleration/deceleration magnitude [m/s²].
- `dwell_time::Float64`: Time to remain stopped at a floor [s].
- `dwell_remaining::Float64`: Remaining dwell time [s].
- `shaft_xy::SVector{2,Float64}`: (x, y) position of the shaft center [m].
"""
mutable struct ElevatorState
    position::SVector{3,Float64}
    velocity::Float64
    current_floor::Int
    target_floor::Int
    phase::ElevatorPhase
    dipole_moment::SVector{3,Float64}
    max_velocity::Float64
    acceleration::Float64
    dwell_time::Float64
    dwell_remaining::Float64
    shaft_xy::SVector{2,Float64}
end

"""
    ElevatorWorld

Simulation world containing a building with multiple elevator shafts.

# Fields
- `elevators::Vector{ElevatorState}`: All elevator cars.
- `n_floors::Int`: Number of floors in the building.
- `floor_height::Float64`: Vertical distance between floors [m].
- `detection_range::Float64`: Maximum range for magnetic field contribution [m].
- `rng::AbstractRNG`: Deterministic random number generator.
- `time::Float64`: Elapsed simulation time [s].
"""
mutable struct ElevatorWorld
    elevators::Vector{ElevatorState}
    n_floors::Int
    floor_height::Float64
    detection_range::Float64
    rng::AbstractRNG
    time::Float64
end

"""
    floor_z(world::ElevatorWorld, floor::Int) -> Float64

Return the z-coordinate for the given floor number (1-indexed).
"""
floor_z(world::ElevatorWorld, floor::Int) = (floor - 1) * world.floor_height

"""
    _pick_new_target!(elev::ElevatorState, world::ElevatorWorld)

Choose a new random target floor different from the current floor and begin dwelling.
"""
function _pick_new_target!(elev::ElevatorState, world::ElevatorWorld)
    while true
        f = rand(world.rng, 1:world.n_floors)
        if f != elev.current_floor
            elev.target_floor = f
            return
        end
    end
end

"""
    _stopping_distance(speed::Float64, accel::Float64) -> Float64

Compute the distance required to decelerate from `speed` to zero at constant `accel`.
"""
_stopping_distance(speed::Float64, accel::Float64) = speed^2 / (2.0 * accel)

"""
    step!(world::ElevatorWorld, dt::Float64)

Advance every elevator in `world` by time step `dt` seconds.

Each elevator follows the state machine:
`STOPPED` (dwell) -> `ACCELERATING` -> `CRUISING` -> `DECELERATING` -> `STOPPED`.
"""
function step!(world::ElevatorWorld, dt::Float64)
    world.time += dt

    for elev in world.elevators
        _step_elevator!(elev, world, dt)
    end
    return nothing
end

function _step_elevator!(elev::ElevatorState, world::ElevatorWorld, dt::Float64)
    if elev.phase == STOPPED
        # Dwell at current floor
        elev.dwell_remaining -= dt
        if elev.dwell_remaining <= 0.0
            _pick_new_target!(elev, world)
            elev.phase = ACCELERATING
            elev.dwell_remaining = 0.0
        end
        return
    end

    target_z = floor_z(world, elev.target_floor)
    current_z = elev.position[3]
    direction = sign(target_z - current_z)  # +1.0 up, -1.0 down
    dist_remaining = abs(target_z - current_z)

    if elev.phase == ACCELERATING
        # Increase speed toward max_velocity
        new_speed = abs(elev.velocity) + elev.acceleration * dt
        if new_speed >= elev.max_velocity
            new_speed = elev.max_velocity
            elev.phase = CRUISING
        end
        # Check if we need to start decelerating
        if _stopping_distance(new_speed, elev.acceleration) >= dist_remaining
            elev.phase = DECELERATING
            new_speed = abs(elev.velocity)  # revert; decel branch handles motion
        end
        if elev.phase == ACCELERATING || elev.phase == CRUISING
            elev.velocity = direction * new_speed
        end
    end

    if elev.phase == CRUISING
        if _stopping_distance(abs(elev.velocity), elev.acceleration) >= dist_remaining
            elev.phase = DECELERATING
        end
    end

    if elev.phase == DECELERATING
        speed = abs(elev.velocity)
        new_speed = speed - elev.acceleration * dt
        if new_speed <= 0.0 || dist_remaining < 1e-6
            # Arrived
            elev.velocity = 0.0
            elev.position = SVector(elev.shaft_xy[1], elev.shaft_xy[2], target_z)
            elev.current_floor = elev.target_floor
            elev.phase = STOPPED
            elev.dwell_remaining = elev.dwell_time
            return
        end
        elev.velocity = direction * new_speed
    end

    # Integrate position
    new_z = elev.position[3] + elev.velocity * dt
    # Clamp to not overshoot target
    if direction > 0.0
        new_z = min(new_z, target_z)
    else
        new_z = max(new_z, target_z)
    end
    elev.position = SVector(elev.shaft_xy[1], elev.shaft_xy[2], new_z)

    # Check arrival after integration
    if abs(new_z - target_z) < 1e-6
        elev.velocity = 0.0
        elev.position = SVector(elev.shaft_xy[1], elev.shaft_xy[2], target_z)
        elev.current_floor = elev.target_floor
        elev.phase = STOPPED
        elev.dwell_remaining = elev.dwell_time
    end
end

"""
    magnetic_field(world::ElevatorWorld, position::SVector{3,Float64}) -> SVector{3,Float64}

Compute the total magnetic field [T] at `position` due to all elevator dipoles.

Uses the magnetic dipole model:

    B = (μ₀ / 4π) * [3(m · r̂)r̂ - m] / r³

where `m` is the dipole moment and `r` is the vector from the dipole to the observation
point. Contributions beyond `world.detection_range` are ignored.
"""
function magnetic_field(world::ElevatorWorld, position::SVector{3,Float64})
    B = @SVector zeros(3)

    for elev in world.elevators
        r_vec = position - elev.position
        r2 = dot(r_vec, r_vec)
        r = sqrt(r2)

        if r < 1e-10 || r > world.detection_range
            continue
        end

        r3 = r2 * r
        r_hat = r_vec / r
        m = elev.dipole_moment
        m_dot_rhat = dot(m, r_hat)

        B += μ₀_over_4π * (3.0 * m_dot_rhat * r_hat - m) / r3
    end

    return B
end

"""
    create_default_elevator_world(; n_elevators=2, n_floors=10, floor_height=3.5,
                                    detection_range=20.0, seed=42) -> ElevatorWorld

Create an `ElevatorWorld` with sensible defaults.

Elevator shafts are spaced 4 m apart along the x-axis at y = 0.
Each elevator receives a random vertical dipole moment in [50, 500] A·m²,
max velocity in [1, 4] m/s, acceleration of 1.5 m/s², and dwell time in [5, 30] s.
All elevators start at floor 1 in the `STOPPED` phase.
"""
function create_default_elevator_world(;
    n_elevators::Int = 2,
    n_floors::Int = 10,
    floor_height::Float64 = 3.5,
    detection_range::Float64 = 20.0,
    seed::Int = 42,
)
    rng = MersenneTwister(seed)
    elevators = ElevatorState[]

    for i in 1:n_elevators
        shaft_x = (i - 1) * 4.0
        shaft_xy = SVector(shaft_x, 0.0)

        dipole_mag = 50.0 + rand(rng) * 450.0  # 50–500 A·m²
        dipole = SVector(0.0, 0.0, dipole_mag)  # vertical dipole

        max_v = 1.0 + rand(rng) * 3.0           # 1–4 m/s
        accel = 1.5                               # m/s²
        dwell = 5.0 + rand(rng) * 25.0           # 5–30 s

        pos = SVector(shaft_x, 0.0, 0.0)         # floor 1, z = 0

        elev = ElevatorState(
            pos, 0.0, 1, 1, STOPPED, dipole,
            max_v, accel, dwell, dwell, shaft_xy,
        )
        push!(elevators, elev)
    end

    return ElevatorWorld(elevators, n_floors, floor_height, detection_range, rng, 0.0)
end

"""
    create_doe_elevator_world(; speed, dwell_time, shaft_positions, dipole_moment,
                                n_floors, floor_height, n_elevators, seed) -> ElevatorWorld

DOE-parameterized constructor allowing explicit control of all DOE factors.

# Keyword Arguments
- `speed::Float64`: Max elevator velocity (m/s). Default 1.5.
- `dwell_time::Float64`: Dwell time at each floor (s). Default 15.0.
- `shaft_positions::Vector{SVector{2,Float64}}`: XY positions of shafts.
- `dipole_moment::Float64`: Dipole moment magnitude (A·m²). Default 200.0.
- `n_floors::Int`: Number of floors. Default 10.
- `floor_height::Float64`: Floor-to-floor height (m). Default 3.5.
- `detection_range::Float64`: Magnetic field cutoff (m). Default 20.0.
- `seed::Int`: RNG seed. Default 42.
- `frozen::Bool`: If true, elevators never move (control condition). Default false.
"""
function create_doe_elevator_world(;
    speed::Float64 = 1.5,
    dwell_time::Float64 = 15.0,
    shaft_positions::Vector{SVector{2,Float64}} = [SVector(0.0, 0.0), SVector(4.0, 0.0)],
    dipole_moment::Float64 = 200.0,
    n_floors::Int = 10,
    floor_height::Float64 = 3.5,
    detection_range::Float64 = 20.0,
    seed::Int = 42,
    frozen::Bool = false,
)
    rng = MersenneTwister(seed)
    elevators = ElevatorState[]

    for shaft_xy in shaft_positions
        dipole = SVector(0.0, 0.0, dipole_moment)
        accel = 1.5
        pos = SVector(shaft_xy[1], shaft_xy[2], 0.0)

        # For frozen worlds, use effectively infinite dwell so elevator never moves
        effective_dwell = frozen ? 1e12 : dwell_time

        elev = ElevatorState(
            pos, 0.0, 1, 1, STOPPED, dipole,
            speed, accel, effective_dwell, effective_dwell, shaft_xy,
        )
        push!(elevators, elev)
    end

    return ElevatorWorld(elevators, n_floors, floor_height, detection_range, rng, 0.0)
end

export create_doe_elevator_world

end # module ElevatorWorldModule
