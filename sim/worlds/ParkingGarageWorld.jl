module ParkingGarageWorldModule

using StaticArrays
using Random
using LinearAlgebra: dot

export ParkingGarageWorld, VehicleState, step!, magnetic_field, create_default_garage

# Physical constants
const MU0_OVER_4PI = 1e-7  # μ₀/(4π) in T·m/A

"""
Vehicle state: position, heading, velocity, dipole moment, parked flag, and waypoint path.
"""
mutable struct VehicleState
    position::SVector{3,Float64}      # metres
    heading::Float64                   # radians
    velocity::Float64                  # m/s
    dipole_moment::Float64             # A·m²
    is_parked::Bool
    waypoints::Vector{SVector{3,Float64}}
    waypoint_index::Int
end

"""
Garage geometry descriptor.
"""
struct GarageGeometry
    n_levels::Int
    level_height::Float64             # metres
    length::Float64                   # x-extent (metres)
    width::Float64                    # y-extent (metres)
    ramp_positions::Vector{SVector{2,Float64}}  # (x,y) centres of ramps
end

"""
Parking garage simulation world.
"""
struct ParkingGarageWorld
    geometry::GarageGeometry
    vehicles::Vector{VehicleState}
    rng::MersenneTwister
    min_turn_radius::Float64          # metres
    detection_range::Float64          # metres
end

# ── Dipole field model ──────────────────────────────────────────────────────

"""
    dipole_field(moment_vec, r_vec) -> SVector{3,Float64}

Magnetic field (Tesla) of a dipole with moment vector `moment_vec` at
displacement `r_vec` from the dipole.
"""
function dipole_field(moment_vec::SVector{3,Float64}, r_vec::SVector{3,Float64})
    r2 = dot(r_vec, r_vec)
    r2 < 1e-12 && return SVector(0.0, 0.0, 0.0)
    r  = sqrt(r2)
    r3 = r2 * r
    r_hat = r_vec / r
    m_dot_rhat = dot(moment_vec, r_hat)
    return MU0_OVER_4PI * (3.0 * m_dot_rhat * r_hat - moment_vec) / r3
end

"""
    moment_vector(state::VehicleState) -> SVector{3,Float64}

Dipole moment vector oriented along vehicle heading in the ground plane.
"""
function moment_vector(state::VehicleState)
    m = state.dipole_moment
    SVector(m * cos(state.heading), m * sin(state.heading), 0.0)
end

# ── Total magnetic field ────────────────────────────────────────────────────

"""
    magnetic_field(world, position) -> SVector{3,Float64}

Total magnetic field (Tesla) at `position` from every vehicle in the world.
"""
function magnetic_field(world::ParkingGarageWorld, position::SVector{3,Float64})
    B = SVector(0.0, 0.0, 0.0)
    for v in world.vehicles
        r_vec = position - v.position
        if dot(r_vec, r_vec) < (world.detection_range)^2
            B = B + dipole_field(moment_vector(v), r_vec)
        end
    end
    return B
end

# ── Vehicle dynamics (waypoint following, ground-plane constrained) ─────────

"""
    step_vehicle!(v, dt, min_turn_radius)

Advance a single moving vehicle along its waypoint path for `dt` seconds.
Heading is adjusted toward the next waypoint, respecting the minimum turn
radius at the current speed.
"""
function step_vehicle!(v::VehicleState, dt::Float64, min_turn_radius::Float64)
    v.is_parked && return

    length(v.waypoints) == 0 && return

    target = v.waypoints[v.waypoint_index]
    dx = target[1] - v.position[1]
    dy = target[2] - v.position[2]
    dist = sqrt(dx^2 + dy^2)

    # Advance waypoint when close enough
    if dist < 0.5
        v.waypoint_index = mod1(v.waypoint_index + 1, length(v.waypoints))
        target = v.waypoints[v.waypoint_index]
        dx = target[1] - v.position[1]
        dy = target[2] - v.position[2]
        dist = sqrt(dx^2 + dy^2)
    end

    desired_heading = atan(dy, dx)
    delta = mod(desired_heading - v.heading + pi, 2pi) - pi  # shortest angle

    # Limit turn rate by minimum turn radius: ω_max = v / r_min
    if v.velocity > 0.0
        max_dheading = (v.velocity / min_turn_radius) * dt
        delta = clamp(delta, -max_dheading, max_dheading)
    end

    v.heading += delta

    # Move in heading direction, keep z on same level
    speed = min(v.velocity, dist / dt)  # slow near waypoint
    v.position = SVector(
        v.position[1] + speed * cos(v.heading) * dt,
        v.position[2] + speed * sin(v.heading) * dt,
        v.position[3],  # z stays on level
    )
end

"""
    step!(world, dt)

Advance all vehicles in the world by `dt` seconds.
"""
function step!(world::ParkingGarageWorld, dt::Float64)
    for v in world.vehicles
        step_vehicle!(v, dt, world.min_turn_radius)
    end
end

# ── Default garage constructor ──────────────────────────────────────────────

"""
    create_default_garage(; n_levels=3, n_vehicles=5, level_height=3.0,
                           seed=42) -> ParkingGarageWorld

Build a rectangular garage with `n_levels`, placing `n_vehicles` per level
(mix of parked and moving). Deterministic via `seed`.
"""
function create_default_garage(;
    n_levels::Int = 3,
    n_vehicles::Int = 5,
    level_height::Float64 = 3.0,
    seed::Int = 42,
)
    rng = MersenneTwister(seed)

    garage_length = 60.0  # metres
    garage_width  = 30.0  # metres

    # Ramp at centre-right of each level
    ramp_positions = [SVector(garage_length * 0.75, garage_width * 0.5)]

    geometry = GarageGeometry(n_levels, level_height, garage_length, garage_width,
                              ramp_positions)

    vehicles = VehicleState[]

    for level in 0:(n_levels - 1)
        z = level * level_height + 0.5  # 0.5 m above floor (axle height)

        for i in 1:n_vehicles
            is_parked = i <= div(n_vehicles * 3, 4)  # ~75 % parked
            dipole = 5.0 + rand(rng) * 95.0           # 5–100 A·m²

            if is_parked
                # Park in a grid pattern
                col = mod1(i, 6)
                row = div(i - 1, 6)
                px = 5.0 + col * 8.0
                py = 4.0 + row * 5.0
                heading = rand(rng) < 0.5 ? 0.0 : pi
                push!(vehicles, VehicleState(
                    SVector(px, py, z), heading, 0.0, dipole, true,
                    SVector{3,Float64}[], 1,
                ))
            else
                # Moving vehicle with a simple loop of waypoints on its level
                speed_kmh = 5.0 + rand(rng) * 10.0     # 5–15 km/h
                speed = speed_kmh / 3.6                  # m/s
                heading = rand(rng) * 2pi

                # Rectangular patrol route
                margin = 5.0
                wpts = SVector{3,Float64}[
                    SVector(margin, margin, z),
                    SVector(garage_length - margin, margin, z),
                    SVector(garage_length - margin, garage_width - margin, z),
                    SVector(margin, garage_width - margin, z),
                ]
                start_idx = rand(rng, 1:length(wpts))
                sx, sy = wpts[start_idx][1], wpts[start_idx][2]

                push!(vehicles, VehicleState(
                    SVector(sx, sy, z), heading, speed, dipole, false,
                    wpts, start_idx,
                ))
            end
        end
    end

    return ParkingGarageWorld(geometry, vehicles, rng, 3.0, 15.0)
end

end # module ParkingGarageWorldModule
