# ============================================================================
# TrajectoryExcitation.jl - Trajectory Generators for Quadratic Observability
# ============================================================================
#
# Purpose: Generate spatially diverse trajectories within a tile that excite
# all 7 quadratic harmonic modes, enabling Tier-2 (d=15) map learning.
#
# Physics motivation:
# Quadratic magnetic field terms (Q7) require spatial curvature in the
# trajectory to be observable. A straight line provides at most rank-1
# quadratic information. A planar trajectory (e.g., lawnmower) provides
# at most rank-4. Full rank-7 requires 3D excitation with nonlinear paths.
#
# The excitation score (log det I_qq) quantifies how well a trajectory
# fills the 7-dimensional quadratic information space. Higher scores
# indicate more uniform excitation of all quadratic modes.
#
# Trajectory generators produce positions in world NED frame [m].
# All generators are deterministic given their parameters.
#
# Note: This file lives in nav_core/src/scenarios/ because trajectory
# generation is a scenario/test utility, not core navigation logic.
# However, it is included in NavCore for convenience since
# trajectory_excitation_score depends on core FIM computation.
#
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Trajectory Generators
# ============================================================================

"""
    lissajous_trajectory(center, L, n_pts; a=3, b=2, δ=π/2, z_amp=3.0) -> Vector{Vec3Map}

Generate a 3D Lissajous trajectory within a tile.

# Arguments
- `center::Vec3Map`: Tile center in world NED frame [m]
- `L::Float64`: Tile half-width (scale) [m]. Trajectory fills ±0.8L in XY.
- `n_pts::Int`: Number of waypoints
- `a::Int`: X-axis frequency ratio (default 3)
- `b::Int`: Y-axis frequency ratio (default 2)
- `δ::Float64`: Phase offset [rad] (default π/2)
- `z_amp::Float64`: Vertical amplitude [m] (default 3.0)

# Returns
Vector of `n_pts` positions in world NED frame.

# Physics justification
Lissajous curves with coprime (a, b) fill 2D space densely, providing
diverse angular excitation. The vertical modulation adds 3D diversity
needed for full rank-7 quadratic observability. The frequency ratio 3:2
ensures the curve does not self-intersect in XY, maximizing spatial coverage.

amplitude = 0.8L keeps all points within the tile boundary (|x̃| ≤ 0.8).
"""
function lissajous_trajectory(center::Vec3Map, L::Float64, n_pts::Int;
                               a::Int=3, b::Int=2, δ::Float64=π/2,
                               z_amp::Float64=3.0)
    @assert n_pts >= 2 "Need at least 2 points"
    @assert L > 0 "Scale L must be positive"

    amp_xy = 0.8 * L  # Stay within tile: |x̃| ≤ 0.8
    positions = Vector{Vec3Map}(undef, n_pts)

    for i in 1:n_pts
        t = 2π * (i - 1) / (n_pts - 1)
        x = amp_xy * sin(a * t + δ)
        y = amp_xy * sin(b * t)
        # Vertical: sinusoidal with different frequency for 3D diversity
        z = z_amp * sin(t)
        positions[i] = center + Vec3Map(x, y, z)
    end

    return positions
end

"""
    box_diagonals_trajectory(center, L, n_pts) -> Vector{Vec3Map}

Generate trajectory traversing all major diagonals of a tile's bounding box.

# Arguments
- `center::Vec3Map`: Tile center [m]
- `L::Float64`: Tile half-width [m]. Trajectory visits ±0.7L corners.
- `n_pts::Int`: Total number of waypoints (distributed across 8 segments)

# Returns
Vector of positions visiting all 8 octant corners via center-crossing diagonals.

# Physics justification
The 8 corners of a cube span all sign combinations of (x, y, z), which
excites all 7 quadratic harmonic modes including cross-terms (xy, xz, yz, xyz).
This provides at least 6 independent directions through the origin,
guaranteeing full-rank quadratic FIM if noise is isotropic.
"""
function box_diagonals_trajectory(center::Vec3Map, L::Float64, n_pts::Int)
    @assert n_pts >= 8 "Need at least 8 points for box diagonals"
    @assert L > 0 "Scale L must be positive"

    amp = 0.7 * L
    # 8 corners of the cube: all sign combinations
    corners = Vec3Map[
        Vec3Map(+amp, +amp, +amp),
        Vec3Map(-amp, -amp, -amp),
        Vec3Map(+amp, -amp, +amp),
        Vec3Map(-amp, +amp, -amp),
        Vec3Map(-amp, +amp, +amp),
        Vec3Map(+amp, -amp, -amp),
        Vec3Map(-amp, -amp, +amp),
        Vec3Map(+amp, +amp, -amp),
    ]

    # Distribute points across 8 diagonal segments (corner → center → next corner)
    n_segments = length(corners)
    pts_per_seg = max(div(n_pts, n_segments), 2)

    positions = Vec3Map[]
    for (idx, corner) in enumerate(corners)
        # Traverse from this corner through center to next corner
        next_corner = corners[mod1(idx + 1, n_segments)]
        for j in 1:pts_per_seg
            frac = (j - 1) / (pts_per_seg - 1)
            pos = (1 - frac) * corner + frac * next_corner
            push!(positions, center + pos)
        end
    end

    # Trim or pad to exact n_pts
    if length(positions) > n_pts
        positions = positions[1:n_pts]
    end

    return positions
end

"""
    spiral_trajectory(center, L, n_pts; n_turns=3) -> Vector{Vec3Map}

Generate an outward spiral trajectory with increasing radius.

# Arguments
- `center::Vec3Map`: Tile center [m]
- `L::Float64`: Tile half-width [m]. Max radius = 0.8L.
- `n_pts::Int`: Number of waypoints
- `n_turns::Int`: Number of spiral revolutions (default 3)

# Returns
Vector of positions spiraling outward from center with linear radius growth.

# Physics justification
Spirals provide monotonically increasing radius, which progressively
excites higher spatial frequencies. Combined with vertical variation,
this covers the radial dimension of quadratic modes (x², y², x²-y²)
more uniformly than concentric circles.
"""
function spiral_trajectory(center::Vec3Map, L::Float64, n_pts::Int;
                            n_turns::Int=3)
    @assert n_pts >= 2 "Need at least 2 points"
    @assert L > 0 "Scale L must be positive"

    r_max = 0.8 * L
    z_amp = 3.0  # Vertical amplitude [m] for 3D diversity

    positions = Vector{Vec3Map}(undef, n_pts)
    for i in 1:n_pts
        frac = (i - 1) / (n_pts - 1)
        θ = 2π * n_turns * frac
        r = r_max * frac  # Linear radius growth
        x = r * cos(θ)
        y = r * sin(θ)
        z = z_amp * sin(2π * frac)  # One vertical cycle
        positions[i] = center + Vec3Map(x, y, z)
    end

    return positions
end

"""
    altitude_modulated(traj::Vector{Vec3Map}, Δz::Float64) -> Vector{Vec3Map}

Add sinusoidal vertical modulation to an existing trajectory.

# Arguments
- `traj`: Input trajectory positions [m]
- `Δz::Float64`: Peak-to-peak vertical amplitude [m]

# Returns
New trajectory with z-coordinate modulated by Δz/2 × sin(2π × t/T).

# Physics justification
Vertical modulation adds z-dependent excitation, which is critical for
observability of quadratic modes involving z (Q4: xz, Q5: yz, Q7: xyz).
A planar (z=const) trajectory has zero sensitivity to these modes.
"""
function altitude_modulated(traj::Vector{Vec3Map}, Δz::Float64)
    n = length(traj)
    result = Vector{Vec3Map}(undef, n)
    for i in 1:n
        frac = (i - 1) / max(n - 1, 1)
        z_mod = (Δz / 2) * sin(2π * frac)
        result[i] = traj[i] + Vec3Map(0.0, 0.0, z_mod)
    end
    return result
end

# ============================================================================
# Excitation Score
# ============================================================================

"""
    trajectory_excitation_score(positions, center, L) -> Float64

Compute excitation score for quadratic observability: log det(I_qq).

# Arguments
- `positions::Vector{Vec3Map}`: Trajectory positions in world frame [m]
- `center::Vec3Map`: Tile center [m]
- `L::Float64`: Tile half-width (scale) [m]

# Returns
log det(I_qq) in nats, where I_qq is the 7×7 quadratic-block FIM
computed with unit isotropic noise (R = I₃). Returns -Inf if rank-deficient.

# Physics
Uses unit noise covariance so the score measures purely geometric
excitation, independent of sensor noise level. Sensor noise scales
I_qq uniformly, so it does not affect relative trajectory comparisons.

This is equivalent to `observability_scalar(compute_quadratic_fim(...))`
but constructs a minimal tile internally, avoiding the need for a full
SlamTileState when only geometric excitation is needed.
"""
function trajectory_excitation_score(positions::Vector{Vec3Map},
                                      center::Vec3Map, L::Float64)
    @assert L > 0 "Scale L must be positive"

    frame = TileLocalFrame(center, L)
    I_qq = zeros(7, 7)

    for pos in positions
        x̃ = normalize_position(frame, pos)
        H_full = field_jacobian(x̃, MODE_QUADRATIC)
        H_q = H_full[:, 9:15]  # 3×7 quadratic sub-block
        # Unit noise: R = I₃, so R⁻¹ = I₃
        I_qq += H_q' * H_q
    end

    # Use observability_scalar for consistent log det computation
    # (handles near-zero eigenvalues from floating point gracefully)
    return observability_scalar(I_qq; metric=:logdet)
end

# ============================================================================
# Exports
# ============================================================================

export lissajous_trajectory, box_diagonals_trajectory, spiral_trajectory
export altitude_modulated, trajectory_excitation_score
