module LobbyWorldModule

using StaticArrays
using Random

export LobbyWorld, DoorSource, HVACSource, step!, magnetic_field, create_default_lobby

# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #

"""Electromagnetic door source (swing or sliding door with optional mag-lock)."""
mutable struct DoorSource
    position::SVector{3,Float64}       # hinge/center position [m]
    hinge_axis::SVector{3,Float64}     # unit axis of rotation
    moment_open::Float64               # dipole moment when fully open  [A·m²]
    moment_closed::Float64             # dipole moment when fully closed [A·m²]
    lock_moment_energized::Float64     # mag-lock moment when locked [A·m²]
    lock_moment_deenergized::Float64   # mag-lock moment when unlocked [A·m²]
    is_open::Bool
    transition_time::Float64           # time to fully open/close [s]
    transition_timer::Float64          # 0 = closed, transition_time = open
    detection_range::Float64           # effective range [m]
    open_probability::Float64          # probability of toggling per second
end

"""HVAC motor / compressor source with periodic cycling."""
mutable struct HVACSource
    position::SVector{3,Float64}       # location [m]
    moment_max::Float64                # peak dipole moment [A·m²]
    period::Float64                    # on/off cycle period [s]
    duty_cycle::Float64                # fraction of period that motor is on
    phase::Float64                     # phase offset [s]
    detection_range::Float64           # effective range [m]
end

"""Single-floor lobby world containing door and HVAC magnetic sources."""
mutable struct LobbyWorld
    # Geometry -------------------------------------------------------------- #
    floor_center::SVector{3,Float64}   # center of the lobby floor [m]
    floor_size::SVector{2,Float64}     # (length_x, length_y) [m]
    floor_height::Float64              # ceiling height [m]

    # Sources --------------------------------------------------------------- #
    doors::Vector{DoorSource}
    hvacs::Vector{HVACSource}

    # Simulation state ------------------------------------------------------ #
    time::Float64
    rng::MersenneTwister
end

# --------------------------------------------------------------------------- #
# Constructors
# --------------------------------------------------------------------------- #

"""
    create_default_lobby(; n_doors=4, n_hvac=2, seed=42)

Build a rectangular lobby with `n_doors` door sources placed along the
perimeter walls and `n_hvac` ceiling-mounted HVAC units.
"""
function create_default_lobby(; n_doors::Int=4, n_hvac::Int=2, seed::Int=42)
    rng = MersenneTwister(seed)

    floor_center = SVector(0.0, 0.0, 0.0)
    floor_size   = SVector(20.0, 12.0)   # 20 m x 12 m lobby
    floor_height = 4.0                    # 4 m ceiling

    half_x = floor_size[1] / 2.0
    half_y = floor_size[2] / 2.0

    # -- doors along perimeter ------------------------------------------------
    doors = DoorSource[]
    for i in 1:n_doors
        # distribute doors evenly: alternate between +y and -y walls
        frac = (i - 0.5) / n_doors
        if isodd(i)
            pos = SVector(-half_x + frac * floor_size[1], half_y, 0.0)
            axis = SVector(0.0, 0.0, 1.0)
        else
            pos = SVector(-half_x + frac * floor_size[1], -half_y, 0.0)
            axis = SVector(0.0, 0.0, 1.0)
        end

        moment_open   = 1.0 + rand(rng) * 19.0        # 1-20 A·m²
        moment_closed = 1.0 + rand(rng) * 5.0          # smaller when closed
        lock_energized   = 5.0 + rand(rng) * 15.0      # 5-20 A·m²
        lock_deenergized = 0.1 + rand(rng) * 0.9       # <1 A·m²
        transition_time  = 1.0 + rand(rng) * 4.0       # 1-5 s
        detection_range  = 2.0 + rand(rng) * 3.0       # 2-5 m
        open_prob        = 0.05 + rand(rng) * 0.10      # per-second toggle rate

        push!(doors, DoorSource(
            pos, axis,
            moment_open, moment_closed,
            lock_energized, lock_deenergized,
            false, transition_time, 0.0,
            detection_range, open_prob,
        ))
    end

    # -- HVAC units on ceiling ------------------------------------------------
    hvacs = HVACSource[]
    for j in 1:n_hvac
        frac = (j - 0.5) / n_hvac
        pos = SVector(-half_x + frac * floor_size[1], 0.0, floor_height)

        moment_max     = 1.0 + rand(rng) * 49.0       # 1-50 A·m²
        period         = 10.0 + rand(rng) * 50.0       # 10-60 s
        duty_cycle     = 0.3 + rand(rng) * 0.5         # 30-80 %
        phase          = rand(rng) * period
        detection_range = 3.0 + rand(rng) * 7.0        # 3-10 m

        push!(hvacs, HVACSource(
            pos, moment_max, period, duty_cycle, phase, detection_range,
        ))
    end

    return LobbyWorld(
        floor_center, floor_size, floor_height,
        doors, hvacs,
        0.0, rng,
    )
end

# --------------------------------------------------------------------------- #
# Stepping
# --------------------------------------------------------------------------- #

"""
    step!(world::LobbyWorld, dt::Float64)

Advance the world by `dt` seconds.  Doors randomly toggle open/closed and
their transition timers evolve.  HVAC phases advance with time.
"""
function step!(world::LobbyWorld, dt::Float64)
    world.time += dt

    # -- doors ----------------------------------------------------------------
    for door in world.doors
        # random toggle decision (Poisson-like per timestep)
        if rand(world.rng) < door.open_probability * dt
            door.is_open = !door.is_open
        end

        # advance transition timer toward target state
        target = door.is_open ? door.transition_time : 0.0
        if door.transition_timer < target
            door.transition_timer = min(door.transition_timer + dt, target)
        elseif door.transition_timer > target
            door.transition_timer = max(door.transition_timer - dt, target)
        end
    end

    return nothing
end

# --------------------------------------------------------------------------- #
# Magnetic field computation
# --------------------------------------------------------------------------- #

"""
    dipole_field(moment_vec, source_pos, eval_pos)

Compute the magnetic field [T] of a magnetic dipole at `eval_pos`.
    B(r) = (mu0 / 4pi) * (3(m·r̂)r̂ - m) / |r|³
"""
function dipole_field(
    moment_vec::SVector{3,Float64},
    source_pos::SVector{3,Float64},
    eval_pos::SVector{3,Float64},
)
    mu0_over_4pi = 1e-7  # T·m/A

    r_vec = eval_pos - source_pos
    r2 = dot(r_vec, r_vec)
    r2 == 0.0 && return SVector(0.0, 0.0, 0.0)

    r_mag = sqrt(r2)
    r_hat = r_vec / r_mag
    m_dot_r = dot(moment_vec, r_hat)

    return mu0_over_4pi * (3.0 * m_dot_r * r_hat - moment_vec) / (r_mag^3)
end

"""
    _door_moment(door::DoorSource)

Effective dipole moment vector of a door source at its current state.
Interpolates between closed and open moments based on transition progress,
and adds the electromagnetic lock contribution.
"""
function _door_moment(door::DoorSource)
    alpha = door.transition_timer / door.transition_time  # 0=closed, 1=open
    # interpolated structural moment
    m_struct = (1.0 - alpha) * door.moment_closed + alpha * door.moment_open
    # lock is energized when door is fully closed (alpha ≈ 0)
    m_lock = (1.0 - alpha) * door.lock_moment_energized +
             alpha * door.lock_moment_deenergized
    m_total = m_struct + m_lock
    # orient along hinge axis (simplified dipole orientation)
    return m_total * door.hinge_axis
end

"""
    _hvac_moment(hvac::HVACSource, t::Float64)

Time-varying dipole moment vector of an HVAC source (motor on/off cycling).
"""
function _hvac_moment(hvac::HVACSource, t::Float64)
    phase_t = mod(t + hvac.phase, hvac.period) / hvac.period
    is_on = phase_t < hvac.duty_cycle
    m_mag = is_on ? hvac.moment_max : 0.0
    # HVAC dipole oriented vertically (motor axis along z)
    return SVector(0.0, 0.0, m_mag)
end

"""
    magnetic_field(world::LobbyWorld, position::SVector{3,Float64})

Return the total magnetic field vector [T] at `position` due to all active
sources in the lobby.
"""
function magnetic_field(world::LobbyWorld, position::SVector{3,Float64})
    B = SVector(0.0, 0.0, 0.0)

    for door in world.doors
        r = norm(position - door.position)
        r > door.detection_range && continue
        m_vec = _door_moment(door)
        B = B + dipole_field(m_vec, door.position, position)
    end

    for hvac in world.hvacs
        r = norm(position - hvac.position)
        r > hvac.detection_range && continue
        m_vec = _hvac_moment(hvac, world.time)
        B = B + dipole_field(m_vec, hvac.position, position)
    end

    return B
end

# convenience: accept a plain tuple/vector
function magnetic_field(world::LobbyWorld, pos)
    return magnetic_field(world, SVector{3,Float64}(pos[1], pos[2], pos[3]))
end

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

function dot(a::SVector{3,Float64}, b::SVector{3,Float64})
    return a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
end

function norm(v::SVector{3,Float64})
    return sqrt(dot(v, v))
end

end # module LobbyWorldModule
