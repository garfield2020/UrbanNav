# ============================================================================
# Factor Graph Components for Urban Navigation
# ============================================================================
#
# Ported from UrbanNav/src/factors.jl
#
# Factor graph implementation with:
# - IMU preintegration factor
# - Odometry velocity factor
# - Barometer factor
# - Magnetic field factor
#
# Uses ForwardDiff for automatic Jacobian computation.
# ============================================================================

using LinearAlgebra
using StaticArrays
using Rotations
using ForwardDiff

export Factor, residual, jacobians, information_matrix
export IMUPreintegration, integrate!, correct_bias
export IMUPreintegrationFactor, OdometryFactor, BarometerFactor, MagneticFactor
export UrbanNavFactorGraph, add_node!, compute_total_error
export skew

# ============================================================================
# Abstract Factor Interface
# ============================================================================

"""
    Factor

Abstract base type for all factors.
"""
abstract type Factor end

"""
    residual(f::Factor, states...)

Compute residual vector for the factor.
"""
function residual end

"""
    jacobians(f::Factor, states...)

Compute Jacobians with respect to states.
"""
function jacobians end

"""
    information_matrix(f::Factor)

Get the information matrix (inverse covariance) for the factor.
"""
function information_matrix end

# ============================================================================
# IMU Preintegration
# ============================================================================

"""
    IMUPreintegration

Preintegrated IMU measurements between keyframes.

Stores Δp, Δv, Δq accumulated over the integration period,
along with Jacobians for bias correction.
"""
mutable struct IMUPreintegration
    dt_total::Float64
    delta_p::Vec3
    delta_v::Vec3
    delta_q::QuatRotation{Float64}
    covariance::Matrix{Float64}

    J_p_bg::Mat3
    J_p_ba::Mat3
    J_v_bg::Mat3
    J_v_ba::Mat3
    J_q_bg::Mat3

    bias_gyro_ref::Vec3
    bias_accel_ref::Vec3

    gyro_noise::Float64
    accel_noise::Float64
end

function IMUPreintegration(;
    bias_gyro::AbstractVector = zeros(3),
    bias_accel::AbstractVector = zeros(3),
    gyro_noise::Real = 0.001,
    accel_noise::Real = 0.01
)
    IMUPreintegration(
        0.0,
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 0.0),
        QuatRotation(1.0, 0.0, 0.0, 0.0),
        zeros(9, 9),
        Mat3(zeros(3,3)), Mat3(zeros(3,3)),
        Mat3(zeros(3,3)), Mat3(zeros(3,3)),
        Mat3(zeros(3,3)),
        Vec3(bias_gyro...), Vec3(bias_accel...),
        Float64(gyro_noise), Float64(accel_noise)
    )
end

"""
    integrate!(preint::IMUPreintegration, gyro, accel, dt)

Integrate a single IMU measurement.
"""
function integrate!(preint::IMUPreintegration, gyro::AbstractVector, accel::AbstractVector, dt::Real)
    dt = Float64(dt)

    ω = Vec3(gyro...) - preint.bias_gyro_ref
    a = Vec3(accel...) - preint.bias_accel_ref

    R = Mat3(preint.delta_q)

    θ = ω * dt
    θ_norm = norm(θ)
    if θ_norm > 1e-10
        axis = θ / θ_norm
        dq = QuatRotation(AngleAxis(θ_norm, axis[1], axis[2], axis[3]))
    else
        dq = QuatRotation(1.0, 0.0, 0.0, 0.0)
    end

    preint.delta_p = preint.delta_p + preint.delta_v * dt + 0.5 * R * a * dt^2
    preint.delta_v = preint.delta_v + R * a * dt
    preint.delta_q = preint.delta_q * dq

    dt2_half = 0.5 * dt^2
    preint.J_p_ba = preint.J_p_ba - R * dt2_half
    preint.J_v_ba = preint.J_v_ba - R * dt
    preint.J_q_bg = preint.J_q_bg - Mat3(I) * dt

    # Covariance propagation with PSD-preserving policy
    F = zeros(9, 9)
    F[1:3, 1:3] = I(3)
    F[1:3, 4:6] = I(3) * dt
    F[4:6, 4:6] = I(3)
    F[4:6, 7:9] = -R * skew(a) * dt
    F[7:9, 7:9] = Mat3(dq)'

    G = zeros(9, 6)
    G[4:6, 1:3] = R * dt
    G[7:9, 4:6] = I(3) * dt

    # Process noise with minimum floor to prevent numerical underflow
    q_min = 1e-12  # Minimum process noise variance
    Q = zeros(6, 6)
    Q[1:3, 1:3] = max(preint.accel_noise^2, q_min) * I(3)
    Q[4:6, 4:6] = max(preint.gyro_noise^2, q_min) * I(3)

    # Propagate covariance
    P_new = F * preint.covariance * F' + G * Q * G'

    # Symmetrize to prevent numerical asymmetry accumulation
    P_new = 0.5 * (P_new + P_new')

    # Add jitter to ensure strict positive definiteness
    ε_jitter = 1e-15
    P_new += ε_jitter * I(9)

    preint.covariance = P_new
    preint.dt_total += dt

    return nothing
end

"""
    skew(v::AbstractVector)

Create skew-symmetric matrix from vector.
"""
function skew(v::AbstractVector)
    SMatrix{3,3,Float64,9}(
        0, v[3], -v[2],
        -v[3], 0, v[1],
        v[2], -v[1], 0
    )
end

"""
    correct_bias(preint::IMUPreintegration, bg, ba)

Get bias-corrected preintegration values.
"""
function correct_bias(preint::IMUPreintegration, bg::AbstractVector, ba::AbstractVector)
    δbg = Vec3(bg...) - preint.bias_gyro_ref
    δba = Vec3(ba...) - preint.bias_accel_ref

    Δp_corr = preint.delta_p + preint.J_p_bg * δbg + preint.J_p_ba * δba
    Δv_corr = preint.delta_v + preint.J_v_bg * δbg + preint.J_v_ba * δba

    δθ = preint.J_q_bg * δbg
    dq_corr = QuatRotation(RotationVec(δθ...))
    Δq_corr = preint.delta_q * dq_corr

    return Δp_corr, Δv_corr, Δq_corr
end

# ============================================================================
# IMU Preintegration Factor
# ============================================================================

"""
    IMUPreintegrationFactor <: Factor

Factor connecting two keyframes via IMU preintegration.
"""
struct IMUPreintegrationFactor <: Factor
    preint::IMUPreintegration
    gravity::Vec3
end

function IMUPreintegrationFactor(preint::IMUPreintegration; gravity::Real = 9.81)
    IMUPreintegrationFactor(preint, Vec3(0.0, 0.0, -gravity))
end

"""
    residual(f::IMUPreintegrationFactor, state_i::UrbanNavState, state_j::UrbanNavState)

Compute IMU factor residual: [r_p, r_v, r_θ] (9,)
"""
function residual(f::IMUPreintegrationFactor, state_i::UrbanNavState, state_j::UrbanNavState)
    dt = f.preint.dt_total
    g = f.gravity

    Δp, Δv, Δq = correct_bias(f.preint, state_i.bias_gyro, state_i.bias_accel)

    Ri = Mat3(state_i.orientation)
    Rj = Mat3(state_j.orientation)

    p_pred = state_i.position + state_i.velocity * dt + 0.5 * g * dt^2 + Ri * Δp
    r_p = Rj' * (state_j.position - p_pred)

    v_pred = state_i.velocity + g * dt + Ri * Δv
    r_v = Rj' * (state_j.velocity - v_pred)

    R_pred = Ri * Mat3(Δq)
    R_err = R_pred' * Rj
    rv_err = RotationVec(QuatRotation(R_err))
    r_θ = Vec3(rv_err.sx, rv_err.sy, rv_err.sz)

    return vcat(r_p, r_v, r_θ)
end

function information_matrix(f::IMUPreintegrationFactor)
    # Ensure covariance is symmetric and well-conditioned before inversion
    cov = f.preint.covariance
    cov_sym = 0.5 * (cov + cov')

    # Add regularization scaled to covariance magnitude
    ε = max(1e-8, 1e-6 * tr(cov_sym) / 9)
    cov_reg = cov_sym + ε * I(9)

    # Use Cholesky for stable inversion
    try
        C = cholesky(Symmetric(cov_reg))
        return inv(C)
    catch
        # Fallback: eigendecomposition with clipping
        λ, V = eigen(Symmetric(cov_reg))
        λ_safe = max.(λ, 1e-10)
        return V * Diagonal(1.0 ./ λ_safe) * V'
    end
end

# ============================================================================
# Odometry Factor
# ============================================================================

"""
    OdometryFactor <: Factor

Odometry velocity measurement factor.
"""
struct OdometryFactor <: Factor
    velocity::Vec3
    noise_std::Float64
end

function OdometryFactor(velocity::AbstractVector; noise_std::Real = 0.01)
    OdometryFactor(Vec3(velocity...), Float64(noise_std))
end

"""
    residual(f::OdometryFactor, state::UrbanNavState)

Compute Odometry factor residual (3,).
"""
function residual(f::OdometryFactor, state::UrbanNavState)
    R_WB = Mat3(state.orientation)
    v_body_pred = R_WB' * state.velocity
    return f.velocity - v_body_pred
end

function information_matrix(f::OdometryFactor)
    σ² = f.noise_std^2
    return (1/σ²) * Mat3(I)
end

# ============================================================================
# Barometer Factor
# ============================================================================

"""
    BarometerFactor <: Factor

Barometer measurement factor.
"""
struct BarometerFactor <: Factor
    altitude::Float64
    noise_std::Float64
end

function BarometerFactor(altitude::Real; noise_std::Real = 0.1)
    BarometerFactor(Float64(altitude), Float64(noise_std))
end

"""
    residual(f::BarometerFactor, state::UrbanNavState)

Compute barometer factor residual (1,).

NED convention: position[3] is down (NED convention).
"""
function residual(f::BarometerFactor, state::UrbanNavState)
    predicted_altitude = -state.position[3]  # NED: z positive down, altitude positive up
    return SVector{1, Float64}(f.altitude - predicted_altitude)
end

function information_matrix(f::BarometerFactor)
    σ² = f.noise_std^2
    return SMatrix{1,1,Float64,1}(1/σ²)
end

# ============================================================================
# Magnetic Factor
# ============================================================================

"""
    MagneticFactor <: Factor

Magnetic field measurement factor.
"""
struct MagneticFactor <: Factor
    measured_B::Vec3
    predicted_B::Vec3
    noise_std::Float64
end

function MagneticFactor(measured_B::AbstractVector, predicted_B::AbstractVector; noise_std::Real = 5e-9)
    MagneticFactor(Vec3(measured_B...), Vec3(predicted_B...), Float64(noise_std))
end

"""
    residual(f::MagneticFactor, state::UrbanNavState)

Compute magnetic factor residual (3,).
"""
function residual(f::MagneticFactor, state::UrbanNavState)
    return f.measured_B - f.predicted_B
end

function information_matrix(f::MagneticFactor)
    σ² = f.noise_std^2
    return (1/σ²) * Mat3(I)
end

# ============================================================================
# Factor Graph
# ============================================================================

"""
    UrbanNavFactorGraph

Collection of factors and states for optimization.
"""
mutable struct UrbanNavFactorGraph
    states::Dict{Int, UrbanNavState}
    factors::Vector{Tuple{Factor, Vector{Int}}}
end

UrbanNavFactorGraph() = UrbanNavFactorGraph(Dict{Int, UrbanNavState}(), Tuple{Factor, Vector{Int}}[])

"""
    add_node!(graph::UrbanNavFactorGraph, index::Int, state::UrbanNavState)

Add a state node to the graph.
"""
function add_node!(graph::UrbanNavFactorGraph, index::Int, state::UrbanNavState)
    graph.states[index] = state
end

"""
    add_factor!(graph::UrbanNavFactorGraph, factor::Factor, state_indices::Vector{Int})

Add a factor connecting specified states.
"""
function add_factor!(graph::UrbanNavFactorGraph, factor::Factor, state_indices::Vector{Int})
    push!(graph.factors, (factor, state_indices))
end

"""
    compute_total_error(graph::UrbanNavFactorGraph)

Compute total weighted squared error.
"""
function compute_total_error(graph::UrbanNavFactorGraph)
    total = 0.0

    for (factor, indices) in graph.factors
        states = [graph.states[i] for i in indices]
        r = residual(factor, states...)
        Ω = information_matrix(factor)
        total += r' * Ω * r
    end

    return total
end

# ============================================================================
# Jacobians
# ============================================================================

function jacobians(f::OdometryFactor, state::UrbanNavState)
    # Analytic Jacobian for Odometry factor
    # residual = v_meas - R' * v_world
    # ∂r/∂v = -R'  (3x3 block at columns 4:6)
    # ∂r/∂θ = -R' * skew(v_world) (3x3 block at columns 7:9, using right perturbation)

    R = Mat3(state.orientation)
    v = state.velocity

    J = zeros(3, 15)

    # ∂r/∂v = -R'
    J[1:3, 4:6] = -Matrix(R')

    # ∂r/∂θ: derivative of R'*v w.r.t. rotation perturbation
    # Using d(R'*v)/dθ ≈ R' * skew(v) for small angle perturbation
    J[1:3, 7:9] = Matrix(R' * skew(v))

    return (SMatrix{3, 15, Float64, 45}(J),)
end

function jacobians(f::BarometerFactor, state::UrbanNavState)
    # residual = depth_meas - position[3]
    # ∂r/∂position[3] = -1
    J = zeros(1, 15)
    J[1, 3] = -1.0
    return (SMatrix{1, 15, Float64, 15}(J),)
end

function jacobians(f::MagneticFactor, state::UrbanNavState)
    J = zeros(3, 15)
    return (SMatrix{3, 15, Float64, 45}(J),)
end

function jacobians(f::IMUPreintegrationFactor, state_i::UrbanNavState, state_j::UrbanNavState)
    Ji = zeros(9, 15)
    Ji[1:3, 1:3] = -I(3)
    Ji[1:3, 4:6] = -I(3) * f.preint.dt_total
    Ji[4:6, 4:6] = -I(3)
    Ji[7:9, 7:9] = -I(3)

    Jj = zeros(9, 15)
    Jj[1:3, 1:3] = I(3)
    Jj[4:6, 4:6] = I(3)
    Jj[7:9, 7:9] = I(3)

    return (Ji, Jj)
end

# ============================================================================
# Optimization
# ============================================================================

"""
    optimize!(graph::UrbanNavFactorGraph; max_iterations=10, tolerance=1e-6)

Optimize the factor graph using Gauss-Newton.
Returns (converged, final_error, iterations)
"""
function optimize!(graph::UrbanNavFactorGraph; max_iterations::Int = 10, tolerance::Real = 1e-6)
    prev_error = compute_total_error(graph)

    for iter in 1:max_iterations
        state_indices = sort(collect(keys(graph.states)))
        n_states = length(state_indices)
        total_dim = n_states * 15

        H = zeros(total_dim, total_dim)
        g = zeros(total_dim)

        for (factor, factor_indices) in graph.factors
            states = [graph.states[i] for i in factor_indices]
            r = residual(factor, states...)
            Ω = information_matrix(factor)
            Js = jacobians(factor, states...)

            for (k, idx_k) in enumerate(factor_indices)
                block_k = findfirst(==(idx_k), state_indices)
                range_k = (block_k-1)*15+1 : block_k*15

                g[range_k] += Js[k]' * Ω * r

                for (l, idx_l) in enumerate(factor_indices)
                    block_l = findfirst(==(idx_l), state_indices)
                    range_l = (block_l-1)*15+1 : block_l*15

                    H[range_k, range_l] += Js[k]' * Ω * Js[l]
                end
            end
        end

        H += 1e-6 * I(total_dim)

        local dx
        try
            dx = -H \ g
        catch
            return (false, prev_error, iter)
        end

        for (k, idx) in enumerate(state_indices)
            range_k = (k-1)*15+1 : k*15
            dx_k = dx[range_k]
            apply_error_state!(graph.states[idx], dx_k)
        end

        new_error = compute_total_error(graph)

        if abs(new_error - prev_error) < tolerance
            return (true, new_error, iter)
        end

        prev_error = new_error
    end

    return (false, prev_error, max_iterations)
end
