# ============================================================================
# DipoleFitter.jl - Nonlinear estimation of magnetic dipole parameters
# ============================================================================
#
# Estimates dipole position p ∈ ℝ³ and moment m ∈ ℝ³ from magnetic field
# observations using Levenberg-Marquardt optimization.
#
# The measurement model is:
#   B(x) = (μ₀/4π) · [3(m·r̂)r̂ - m] / r³,   r = x - p
#
# State vector: θ = [p; m] ∈ ℝ⁶
# ============================================================================

using LinearAlgebra
using StaticArrays

export DipoleFitConfig, DipoleFitResult, ConfidenceLevel
export fit_dipole, fit_dipole_with_gradient
export CONFIDENCE_LOW, CONFIDENCE_MEDIUM, CONFIDENCE_HIGH
export compute_fit_covariance, assess_observability

# ============================================================================
# Confidence levels for promotion gating
# ============================================================================

@enum ConfidenceLevel begin
    CONFIDENCE_LOW
    CONFIDENCE_MEDIUM
    CONFIDENCE_HIGH
end

# ============================================================================
# Configuration
# ============================================================================

"""
    DipoleFitConfig

Configuration for dipole parameter estimation.
"""
struct DipoleFitConfig
    # Optimization parameters
    max_iterations::Int
    lambda_init::Float64          # Initial damping factor
    lambda_up::Float64            # Factor to increase lambda on bad step
    lambda_down::Float64          # Factor to decrease lambda on good step
    convergence_tol::Float64      # Relative cost reduction threshold
    gradient_tol::Float64         # Gradient norm threshold

    # Noise parameters
    σ_B::Float64                  # Field noise std (T)
    σ_G::Float64                  # Gradient noise std (T/m)

    # Regularization
    regularization_eps::Float64   # Regularization for Hessian

    # Confidence thresholds
    min_observations::Int         # Minimum observations for fit
    max_condition_number::Float64 # Max condition number for observability
    min_residual_reduction::Float64 # Min cost reduction ratio for acceptance

    # Moment bounds for physical plausibility
    moment_min::Float64           # Minimum plausible moment magnitude (A·m²)
    moment_max::Float64           # Maximum plausible moment magnitude (A·m²)
end

function DipoleFitConfig(;
    max_iterations::Int = 50,
    lambda_init::Float64 = 1e-3,
    lambda_up::Float64 = 10.0,
    lambda_down::Float64 = 0.1,
    convergence_tol::Float64 = 1e-8,
    gradient_tol::Float64 = 1e-10,
    σ_B::Float64 = 5e-9,
    σ_G::Float64 = 50e-9,
    regularization_eps::Float64 = 1e-10,
    min_observations::Int = 5,
    max_condition_number::Float64 = 1e8,
    min_residual_reduction::Float64 = 0.5,
    moment_min::Float64 = 1e-3,      # 1 mA·m²
    moment_max::Float64 = 1e6        # 1 MA·m² (very large anchor)
)
    DipoleFitConfig(
        max_iterations, lambda_init, lambda_up, lambda_down,
        convergence_tol, gradient_tol, σ_B, σ_G, regularization_eps,
        min_observations, max_condition_number, min_residual_reduction,
        moment_min, moment_max
    )
end

const DEFAULT_DIPOLE_FIT_CONFIG = DipoleFitConfig()

# ============================================================================
# Fit Result
# ============================================================================

"""
    DipoleFitResult

Result of dipole parameter estimation.
"""
struct DipoleFitResult
    # Estimated parameters
    position::Vec3               # Estimated dipole position
    moment::Vec3                 # Estimated magnetic moment

    # Convergence
    converged::Bool
    iterations::Int
    final_cost::Float64
    initial_cost::Float64
    cost_reduction::Float64      # (initial - final) / initial

    # Uncertainty
    covariance::SMatrix{6,6,Float64,36}  # Full 6x6 covariance
    position_std::Vec3           # Position uncertainty (1-σ)
    moment_std::Vec3             # Moment uncertainty (1-σ)

    # Observability
    condition_number::Float64
    rank_deficient::Bool

    # Confidence assessment
    confidence::ConfidenceLevel

    # Diagnostics
    residual_rms::Float64        # RMS of whitened residuals
    n_observations::Int
end

# ============================================================================
# Core Forward Model (inlined for performance)
# ============================================================================

"""
Compute dipole field at observation point.
"""
function _dipole_field(p_dipole::Vec3, m::Vec3, x_obs::Vec3)
    r_vec = x_obs - p_dipole
    r = norm(r_vec)

    if r < 1e-10
        return Vec3(0.0, 0.0, 0.0)
    end

    r̂ = r_vec / r
    m_dot_r̂ = dot(m, r̂)
    B = μ₀_4π * (3 * m_dot_r̂ * r̂ - m) / r^3

    return B
end

"""
Compute Jacobian of RESIDUAL w.r.t. parameters [p; m].
Returns 3×6 matrix.

Convention: residual r = B_meas - B_pred, so J_r = -∂B_pred/∂θ
This ensures Gauss-Newton steps descend the cost.
"""
function _residual_jacobian(p_dipole::Vec3, m::Vec3, x_obs::Vec3)
    r_vec = x_obs - p_dipole
    r = norm(r_vec)

    if r < 1e-10
        return zeros(3, 6)
    end

    r̂ = r_vec / r
    m_dot_r̂ = dot(m, r̂)
    r3 = r^3
    r4 = r^4

    # Model Jacobian ∂B_pred/∂p (note: ∂B/∂p = -∂B/∂x since B depends on x-p)
    # ∂B/∂x uses the corrected formula from physics.jl
    # ∂Bᵢ/∂xⱼ = μ₀_4π * (3/r⁴) * [-5(m·r̂)r̂ᵢr̂ⱼ + (m·r̂)δᵢⱼ + mᵢr̂ⱼ + r̂ᵢmⱼ]
    # Therefore ∂B/∂p = -∂B/∂x has opposite signs
    Jh_p = zeros(3, 3)
    for i in 1:3
        for j in 1:3
            δij = i == j ? 1.0 : 0.0
            # ∂Bᵢ/∂pⱼ = -∂Bᵢ/∂xⱼ
            Jh_p[i, j] = μ₀_4π * (3/r4) * (
                5 * m_dot_r̂ * r̂[i] * r̂[j] -
                m_dot_r̂ * δij -
                m[i] * r̂[j] -
                r̂[i] * m[j]
            )
        end
    end

    # Model Jacobian ∂B_pred/∂m
    # B = μ₀_4π * (3·(m·r̂)·r̂ - m) / r³
    # ∂Bᵢ/∂mⱼ = μ₀_4π * (3·r̂ⱼ·r̂ᵢ - δᵢⱼ) / r³
    Jh_m = zeros(3, 3)
    for i in 1:3
        for j in 1:3
            δij = i == j ? 1.0 : 0.0
            Jh_m[i, j] = μ₀_4π * (3 * r̂[j] * r̂[i] - δij) / r3
        end
    end

    # Residual Jacobian: J_r = -∂h/∂θ (since r = y - h)
    Jh = hcat(Jh_p, Jh_m)
    return -Jh
end

# ============================================================================
# Main Fitting Functions
# ============================================================================

"""
    fit_dipole(observations, initial_guess; config=DEFAULT_DIPOLE_FIT_CONFIG)

Fit dipole parameters (position, moment) from field observations.

# Arguments
- `observations`: Vector of (position, B_measured) tuples
- `initial_guess`: (p_init, m_init) initial parameter estimate

# Returns
- `DipoleFitResult` with estimated parameters and uncertainty
"""
function fit_dipole(
    observations::Vector{<:Tuple{<:AbstractVector, <:AbstractVector}},
    initial_guess::Tuple{<:AbstractVector, <:AbstractVector};
    config::DipoleFitConfig = DEFAULT_DIPOLE_FIT_CONFIG
)
    n_obs = length(observations)

    # Check minimum observations
    if n_obs < config.min_observations
        return _make_failed_result(initial_guess, n_obs,
            "Insufficient observations: $n_obs < $(config.min_observations)")
    end

    # Initialize state
    p = Vec3(initial_guess[1]...)
    m = Vec3(initial_guess[2]...)
    θ = vcat(Vector(p), Vector(m))

    # Whitening factor: L where Σ = L*L' (here Σ = σ²I, so L = σI)
    σ = config.σ_B

    # Compute initial whitened cost
    initial_cost = _compute_whitened_cost(θ, observations, σ)

    # Levenberg-Marquardt optimization
    λ = config.lambda_init
    cost = initial_cost

    converged = false
    iterations = 0

    for iter in 1:config.max_iterations
        iterations = iter

        # Compute whitened Jacobian and residual (consistent with cost)
        J_w, r_w = _compute_whitened_jacobian_residual(θ, observations, σ)

        # Build normal equations: (J'J + λI)Δθ = -J'r
        H = J_w' * J_w
        g = J_w' * r_w

        # Check gradient convergence
        if norm(g) < config.gradient_tol
            converged = true
            break
        end

        # Levenberg-Marquardt step with diagonal scaling
        H_reg = H + λ * (Diagonal(diag(H)) + config.regularization_eps * I(6))

        # Solve for step
        Δθ = -(H_reg \ g)

        # Backtracking line search
        α = 1.0
        step_accepted = false
        for _ in 1:10  # Max 10 backtracks
            θ_new = θ + α * Δθ
            cost_new = _compute_whitened_cost(θ_new, observations, σ)

            if cost_new < cost * (1 + 1e-12)  # Accept with tiny tolerance
                θ = θ_new
                reduction = (cost - cost_new) / max(cost, 1e-15)
                cost = cost_new
                λ = max(λ * config.lambda_down, 1e-15)
                step_accepted = true

                # Check convergence
                if reduction < config.convergence_tol && reduction >= 0
                    converged = true
                end
                break
            end
            α *= 0.5  # Backtrack
        end

        if !step_accepted
            # All backtracks failed - increase damping
            λ = min(λ * config.lambda_up, 1e10)
        end

        if converged
            break
        end
    end

    # Extract final parameters
    p_final = Vec3(θ[1], θ[2], θ[3])
    m_final = Vec3(θ[4], θ[5], θ[6])

    # Compute covariance from final Jacobian
    J_w_final, _ = _compute_whitened_jacobian_residual(θ, observations, σ)
    cov, cond_num, rank_def = _compute_covariance(J_w_final, config)

    # Compute residual RMS
    residual_rms = sqrt(cost / n_obs)

    # Assess confidence
    cost_reduction = (initial_cost - cost) / max(initial_cost, 1e-15)
    confidence = _assess_confidence(
        converged, cost_reduction, cond_num, rank_def,
        norm(m_final), n_obs, config
    )

    return DipoleFitResult(
        p_final,
        m_final,
        converged,
        iterations,
        cost,
        initial_cost,
        cost_reduction,
        cov,
        Vec3(sqrt(cov[1,1]), sqrt(cov[2,2]), sqrt(cov[3,3])),
        Vec3(sqrt(cov[4,4]), sqrt(cov[5,5]), sqrt(cov[6,6])),
        cond_num,
        rank_def,
        confidence,
        residual_rms,
        n_obs
    )
end

"""
    fit_dipole_with_gradient(observations, initial_guess; config)

Fit dipole using both field and gradient observations.

# Arguments
- `observations`: Vector of (position, B_measured, G_measured) tuples
- `initial_guess`: (p_init, m_init) initial parameter estimate
"""
function fit_dipole_with_gradient(
    observations::Vector{<:Tuple{<:AbstractVector, <:AbstractVector, <:AbstractMatrix}},
    initial_guess::Tuple{<:AbstractVector, <:AbstractVector};
    config::DipoleFitConfig = DEFAULT_DIPOLE_FIT_CONFIG
)
    n_obs = length(observations)

    if n_obs < config.min_observations
        return _make_failed_result(initial_guess, n_obs,
            "Insufficient observations: $n_obs < $(config.min_observations)")
    end

    # Initialize state
    p = Vec3(initial_guess[1]...)
    m = Vec3(initial_guess[2]...)
    θ = vcat(Vector(p), Vector(m))

    # Build weight matrix (8x8 for combined B+G)
    W_B = I(3) / config.σ_B^2
    W_G = I(5) / config.σ_G^2
    W = cat(W_B, W_G, dims=(1,2))

    # Compute initial cost
    initial_cost = _compute_cost_with_gradient(θ, observations, W_B, W_G)

    # Levenberg-Marquardt
    λ = config.lambda_init
    cost = initial_cost
    converged = false
    iterations = 0

    for iter in 1:config.max_iterations
        iterations = iter

        J, r = _compute_jacobian_residual_with_gradient(θ, observations)

        # Weight the Jacobian and residual
        n_total = 8 * n_obs
        W_full = kron(I(n_obs), W)
        JtW = J' * W_full
        H = JtW * J
        g = JtW * r

        if norm(g) < config.gradient_tol
            converged = true
            break
        end

        H_reg = H + λ * Diagonal(diag(H) .+ config.regularization_eps)
        Δθ = -(H_reg \ g)

        θ_new = θ + Δθ
        cost_new = _compute_cost_with_gradient(θ_new, observations, W_B, W_G)

        if cost_new < cost
            θ = θ_new
            reduction = (cost - cost_new) / cost
            cost = cost_new
            λ = max(λ * config.lambda_down, 1e-15)

            if reduction < config.convergence_tol
                converged = true
                break
            end
        else
            λ = min(λ * config.lambda_up, 1e15)
        end
    end

    p_final = Vec3(θ[1], θ[2], θ[3])
    m_final = Vec3(θ[4], θ[5], θ[6])

    J_final, r_final = _compute_jacobian_residual_with_gradient(θ, observations)
    W_full = kron(I(n_obs), W)
    JtW = J_final' * W_full
    H = JtW * J_final

    cov, cond_num, rank_def = _compute_covariance_from_hessian(H, config)

    residual_rms = sqrt(cost / n_obs)
    cost_reduction = (initial_cost - cost) / max(initial_cost, 1e-15)
    confidence = _assess_confidence(
        converged, cost_reduction, cond_num, rank_def,
        norm(m_final), n_obs, config
    )

    return DipoleFitResult(
        p_final,
        m_final,
        converged,
        iterations,
        cost,
        initial_cost,
        cost_reduction,
        cov,
        Vec3(sqrt(cov[1,1]), sqrt(cov[2,2]), sqrt(cov[3,3])),
        Vec3(sqrt(cov[4,4]), sqrt(cov[5,5]), sqrt(cov[6,6])),
        cond_num,
        rank_def,
        confidence,
        residual_rms,
        n_obs
    )
end

# ============================================================================
# Helper Functions
# ============================================================================

"""
Compute whitened cost: 0.5 * ||r_w||² where r_w = r/σ
This is the objective function for LM optimization.
"""
function _compute_whitened_cost(θ::Vector{Float64}, observations, σ::Float64)
    p = Vec3(θ[1], θ[2], θ[3])
    m = Vec3(θ[4], θ[5], θ[6])

    cost = 0.0
    for (x_obs, B_meas) in observations
        B_pred = _dipole_field(p, m, Vec3(x_obs...))
        r = Vector(B_meas) - Vector(B_pred)
        r_w = r / σ  # Whitened residual
        cost += dot(r_w, r_w)
    end

    return cost / 2
end

"""
Compute whitened Jacobian and residual for LM.
J_w = J/σ, r_w = r/σ so that cost = 0.5 * ||r_w||²
"""
function _compute_whitened_jacobian_residual(θ::Vector{Float64}, observations, σ::Float64)
    p = Vec3(θ[1], θ[2], θ[3])
    m = Vec3(θ[4], θ[5], θ[6])

    n_obs = length(observations)
    J_w = zeros(3 * n_obs, 6)
    r_w = zeros(3 * n_obs)

    for (i, (x_obs, B_meas)) in enumerate(observations)
        x = Vec3(x_obs...)
        B_pred = _dipole_field(p, m, x)

        # Residual Jacobian (note: uses _residual_jacobian which returns -∂h/∂θ)
        J_i = _residual_jacobian(p, m, x)

        # Residual: r = B_meas - B_pred
        r_i = Vector(B_meas) - Vector(B_pred)

        idx = 3*(i-1)+1 : 3*i
        J_w[idx, :] = J_i / σ   # Whitened Jacobian
        r_w[idx] = r_i / σ      # Whitened residual
    end

    return J_w, r_w
end

# Legacy cost function (for compatibility)
function _compute_cost(θ::Vector{Float64}, observations, W)
    p = Vec3(θ[1], θ[2], θ[3])
    m = Vec3(θ[4], θ[5], θ[6])

    cost = 0.0
    for (x_obs, B_meas) in observations
        B_pred = _dipole_field(p, m, Vec3(x_obs...))
        r = Vector(B_meas) - Vector(B_pred)
        cost += r' * W * r
    end

    return cost / 2
end

function _compute_cost_with_gradient(θ::Vector{Float64}, observations, W_B, W_G)
    p = Vec3(θ[1], θ[2], θ[3])
    m = Vec3(θ[4], θ[5], θ[6])
    dipole = MagneticDipole(p, m)

    cost = 0.0
    for (x_obs, B_meas, G_meas) in observations
        B_pred = field(dipole, x_obs)
        G_pred = gradient(dipole, x_obs)

        r_B = Vector(B_meas) - Vector(B_pred)
        r_G = gradient_to_vector(G_meas) - gradient_to_vector(G_pred)

        cost += r_B' * W_B * r_B + r_G' * W_G * r_G
    end

    return cost / 2
end

function _compute_jacobian_residual(θ::Vector{Float64}, observations)
    p = Vec3(θ[1], θ[2], θ[3])
    m = Vec3(θ[4], θ[5], θ[6])

    n_obs = length(observations)
    J = zeros(3 * n_obs, 6)
    r = zeros(3 * n_obs)

    for (i, (x_obs, B_meas)) in enumerate(observations)
        x = Vec3(x_obs...)
        B_pred = _dipole_field(p, m, x)
        J_i = _residual_jacobian(p, m, x)  # Use residual Jacobian (correct sign)

        idx = 3*(i-1)+1 : 3*i
        J[idx, :] = J_i
        r[idx] = Vector(B_meas) - Vector(B_pred)
    end

    return J, r
end

function _compute_jacobian_residual_with_gradient(θ::Vector{Float64}, observations)
    p = Vec3(θ[1], θ[2], θ[3])
    m = Vec3(θ[4], θ[5], θ[6])
    dipole = MagneticDipole(p, m)

    n_obs = length(observations)
    J = zeros(8 * n_obs, 6)
    r = zeros(8 * n_obs)

    for (i, (x_obs, B_meas, G_meas)) in enumerate(observations)
        x = Vec3(x_obs...)

        # Field - use residual Jacobian (negative of model Jacobian)
        B_pred = field(dipole, x)
        J_B = _residual_jacobian(p, m, x)

        # Gradient - use negative of model Jacobian for residual
        J_G = -_gradient_jacobian_fd(p, m, x)
        G_pred = gradient(dipole, x)

        idx_B = 8*(i-1)+1 : 8*(i-1)+3
        idx_G = 8*(i-1)+4 : 8*i

        J[idx_B, :] = J_B
        J[idx_G, :] = J_G

        r[idx_B] = Vector(B_meas) - Vector(B_pred)
        r[idx_G] = Vector(gradient_to_vector(G_meas) - gradient_to_vector(G_pred))
    end

    return J, r
end

function _gradient_jacobian_fd(p::Vec3, m::Vec3, x::Vec3; δ=1e-6)
    """Finite difference Jacobian of gradient tensor (5-component) w.r.t. θ"""
    J = zeros(5, 6)
    θ0 = vcat(Vector(p), Vector(m))

    for j in 1:6
        θ_plus = copy(θ0)
        θ_minus = copy(θ0)
        θ_plus[j] += δ
        θ_minus[j] -= δ

        p_plus = Vec3(θ_plus[1:3]...)
        m_plus = Vec3(θ_plus[4:6]...)
        p_minus = Vec3(θ_minus[1:3]...)
        m_minus = Vec3(θ_minus[4:6]...)

        G_plus = gradient(MagneticDipole(p_plus, m_plus), x)
        G_minus = gradient(MagneticDipole(p_minus, m_minus), x)

        J[:, j] = Vector(gradient_to_vector(G_plus) - gradient_to_vector(G_minus)) / (2δ)
    end

    return J
end

function _compute_covariance(J_w::Matrix{Float64}, config::DipoleFitConfig)
    """
    Compute parameter covariance from whitened Jacobian.

    For whitened J_w = J/σ, we have:
    Cov(θ) = (J'Σ⁻¹J)⁻¹ = σ²(J'J)⁻¹ = (J_w'J_w)⁻¹

    So we just invert J_w'J_w directly (no σ² factor needed).
    """
    JtJ = J_w' * J_w
    cond_num = cond(JtJ)
    rank_def = cond_num > config.max_condition_number

    # Regularize and invert
    ε = config.regularization_eps
    JtJ_reg = JtJ + ε * I(6)

    try
        cov = inv(JtJ_reg)  # No σ² factor - J is already whitened
        return SMatrix{6,6,Float64,36}(cov), cond_num, rank_def
    catch
        # Return large covariance on failure
        return SMatrix{6,6,Float64,36}(1e6 * I(6)), cond_num, true
    end
end

function _compute_covariance_from_hessian(H::Matrix{Float64}, config::DipoleFitConfig)
    """Compute covariance from Hessian (already weighted)."""
    cond_num = cond(H)
    rank_def = cond_num > config.max_condition_number

    ε = config.regularization_eps
    H_reg = H + ε * I(6)

    try
        cov = inv(H_reg)
        return SMatrix{6,6,Float64,36}(cov), cond_num, rank_def
    catch
        return SMatrix{6,6,Float64,36}(1e6 * I(6)), cond_num, true
    end
end

function _assess_confidence(
    converged::Bool,
    cost_reduction::Float64,
    cond_num::Float64,
    rank_def::Bool,
    moment_mag::Float64,
    n_obs::Int,
    config::DipoleFitConfig
)
    # Low confidence conditions
    if !converged || rank_def
        return CONFIDENCE_LOW
    end

    if cost_reduction < config.min_residual_reduction
        return CONFIDENCE_LOW
    end

    if moment_mag < config.moment_min || moment_mag > config.moment_max
        return CONFIDENCE_LOW
    end

    # High confidence conditions
    if cost_reduction > 0.9 && cond_num < 1e4 && n_obs >= 10
        return CONFIDENCE_HIGH
    end

    # Medium otherwise
    return CONFIDENCE_MEDIUM
end

function _make_failed_result(initial_guess, n_obs, reason)
    p = Vec3(initial_guess[1]...)
    m = Vec3(initial_guess[2]...)

    return DipoleFitResult(
        p, m,
        false,    # converged
        0,        # iterations
        Inf,      # final_cost
        Inf,      # initial_cost
        0.0,      # cost_reduction
        SMatrix{6,6,Float64,36}(1e6 * I(6)),
        Vec3(1e3, 1e3, 1e3),
        Vec3(1e3, 1e3, 1e3),
        Inf,      # condition_number
        true,     # rank_deficient
        CONFIDENCE_LOW,
        Inf,      # residual_rms
        n_obs
    )
end

# ============================================================================
# Observability Assessment
# ============================================================================

"""
    assess_observability(observation_positions; standoff_estimate=10.0)

Assess whether observation geometry provides sufficient observability
for dipole parameter estimation.

Returns (is_observable, condition_estimate, rank).
"""
function assess_observability(
    observation_positions::Vector{<:AbstractVector};
    standoff_estimate::Float64 = 10.0
)
    n = length(observation_positions)
    if n < 6
        return (false, Inf, n)
    end

    # Check geometric diversity
    # Compute spread in each dimension
    positions = hcat([Vector(p) for p in observation_positions]...)

    # Singular value decomposition of centered positions
    center = mean(positions, dims=2)
    centered = positions .- center

    if size(centered, 2) < 3
        return (false, Inf, size(centered, 2))
    end

    svd_result = svd(centered)
    singular_values = svd_result.S

    # Check for rank deficiency
    tol = 1e-6 * maximum(singular_values)
    rank = sum(singular_values .> tol)

    if rank < 3
        return (false, Inf, rank)
    end

    cond_est = singular_values[1] / singular_values[end]
    is_observable = cond_est < 1e6 && rank >= 3

    return (is_observable, cond_est, rank)
end

"""
    compute_fit_covariance(result::DipoleFitResult)

Extract position-only covariance (3×3) from full fit result.
"""
function compute_fit_covariance(result::DipoleFitResult)
    return SMatrix{3,3,Float64,9}(result.covariance[1:3, 1:3])
end
