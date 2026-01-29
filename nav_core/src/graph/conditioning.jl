# ============================================================================
# conditioning.jl - Numerical Conditioning Controls
# ============================================================================
#
# Ported from AUV-Navigation/src/conditioning.jl
#
# Key Components:
# 1. State/parameter scaling for balanced Hessians
# 2. Adaptive Levenberg-Marquardt damping
# 3. Optional Jacobi preconditioning
# 4. Eigenvalue bounds monitoring
# ============================================================================

using LinearAlgebra
using Statistics

export ScalingConfig, DEFAULT_SCALING_CONFIG
export build_scaling_matrix, compute_auto_scales
export LMDampingPolicy, LM_FIXED, LM_MULTIPLICATIVE, LM_NIELSEN, LM_TRUST_REGION
export LMDampingState, update_damping!, apply_lm_damping, apply_lm_damping_identity
export JacobiPreconditioner, build_jacobi_preconditioner
export apply_preconditioner, unapply_preconditioner
export ConditioningStats, compute_conditioning_stats
export check_spd, regularize_hessian
export ConditioningPipeline, SolveResult, solve_conditioned

# ============================================================================
# Scaling Configuration
# ============================================================================

"""
    ScalingConfig

Configuration for state and parameter scaling.

# Fields
- `pose_scale::Vector{Float64}`: Scale factors for pose (position, orientation)
- `velocity_scale::Float64`: Scale for velocity states
- `bias_gyro_scale::Float64`: Scale for gyro bias
- `bias_accel_scale::Float64`: Scale for accel bias
- `coeff_scale::Float64`: Scale for harmonic coefficients
- `feature_pos_scale::Float64`: Scale for feature positions
- `feature_moment_scale::Float64`: Scale for feature moments
- `auto_scale::Bool`: Automatically compute scales from data
"""
struct ScalingConfig
    pose_scale::Vector{Float64}      # [x, y, z, roll, pitch, yaw] or [x, y, z] for position
    velocity_scale::Float64
    bias_gyro_scale::Float64
    bias_accel_scale::Float64
    coeff_scale::Float64
    feature_pos_scale::Float64
    feature_moment_scale::Float64
    auto_scale::Bool
end

function ScalingConfig(;
    pose_scale::Vector{Float64} = [10.0, 10.0, 10.0, 0.1, 0.1, 0.1],
    velocity_scale::Float64 = 1.0,
    bias_gyro_scale::Float64 = 0.01,
    bias_accel_scale::Float64 = 0.1,
    coeff_scale::Float64 = 1e-9,
    feature_pos_scale::Float64 = 10.0,
    feature_moment_scale::Float64 = 100.0,
    auto_scale::Bool = false
)
    ScalingConfig(
        pose_scale, velocity_scale, bias_gyro_scale, bias_accel_scale,
        coeff_scale, feature_pos_scale, feature_moment_scale, auto_scale
    )
end

const DEFAULT_SCALING_CONFIG = ScalingConfig()

# ============================================================================
# Scaling Matrix Construction
# ============================================================================

"""
    build_scaling_matrix(n_poses, n_coeffs, n_features, config) -> Diagonal

Build diagonal scaling matrix for the full state vector.

The scaling matrix D is used to transform the problem:
    Original: min ||r(x)||²
    Scaled: min ||r(D*z)||² where x = D*z

This ensures all state components have similar magnitude, improving conditioning.
"""
function build_scaling_matrix(
    n_poses::Int,
    n_coeffs::Int,
    n_features::Int,
    config::ScalingConfig = DEFAULT_SCALING_CONFIG
)
    # Pose scaling: 6 DOF per pose (position + orientation error)
    pose_dim = 6
    pose_scales = repeat(config.pose_scale[1:min(pose_dim, length(config.pose_scale))], n_poses)
    if length(config.pose_scale) < pose_dim
        # Pad with ones if pose_scale is shorter
        pose_scales = vcat(pose_scales, ones(n_poses * pose_dim - length(pose_scales)))
    end

    # Coefficient scaling
    coeff_scales = fill(config.coeff_scale, n_coeffs)

    # Feature scaling: 6 DOF per feature (position + moment)
    feature_scales = Float64[]
    for _ in 1:n_features
        append!(feature_scales, fill(config.feature_pos_scale, 3))
        append!(feature_scales, fill(config.feature_moment_scale, 3))
    end

    # Combine
    all_scales = vcat(pose_scales, coeff_scales, feature_scales)

    return Diagonal(all_scales)
end

"""
    compute_auto_scales(H::AbstractMatrix, block_sizes::Vector{Int}) -> Vector{Float64}

Automatically compute scaling factors from Hessian diagonal.

Strategy: scale each block so that mean diagonal element ≈ 1.
"""
function compute_auto_scales(H::AbstractMatrix, block_sizes::Vector{Int})
    n = size(H, 1)
    scales = ones(n)

    idx = 1
    for block_size in block_sizes
        block_end = idx + block_size - 1
        if block_end > n
            break
        end

        # Compute mean diagonal for this block
        block_diag = [H[i, i] for i in idx:block_end]
        mean_diag = mean(abs.(block_diag))

        if mean_diag > 1e-15
            scale = sqrt(mean_diag)
            scales[idx:block_end] .= scale
        end

        idx = block_end + 1
    end

    return scales
end

# ============================================================================
# Levenberg-Marquardt Damping
# ============================================================================

"""
    LMDampingPolicy

Policy for adaptive Levenberg-Marquardt damping.

- `LM_FIXED`: Fixed damping factor
- `LM_MULTIPLICATIVE`: Multiply/divide by factor on success/failure
- `LM_NIELSEN`: Nielsen's strategy (smooth adaptation)
- `LM_TRUST_REGION`: Trust region style (adjust based on gain ratio)
"""
@enum LMDampingPolicy begin
    LM_FIXED        # Fixed damping factor
    LM_MULTIPLICATIVE  # Multiply/divide by factor on success/failure
    LM_NIELSEN      # Nielsen's strategy (smooth adaptation)
    LM_TRUST_REGION  # Trust region style (adjust based on gain ratio)
end

"""
    LMDampingState

State for adaptive LM damping.

# Fields
- `λ::Float64` - Current damping factor
- `λ_min::Float64` - Minimum allowed λ
- `λ_max::Float64` - Maximum allowed λ
- `policy::LMDampingPolicy` - Damping policy
- `ν::Float64` - Nielsen parameter (for LM_NIELSEN)
- `factor::Float64` - Multiplication factor (for LM_MULTIPLICATIVE)
- `trust_radius::Float64` - Trust region radius (for LM_TRUST_REGION)
- `iteration::Int` - Current iteration count
- `last_cost::Float64` - Last accepted cost
- `last_gain_ratio::Float64` - Last gain ratio
"""
mutable struct LMDampingState
    λ::Float64              # Current damping factor
    λ_min::Float64          # Minimum allowed λ
    λ_max::Float64          # Maximum allowed λ
    policy::LMDampingPolicy
    ν::Float64              # Nielsen parameter (for LM_NIELSEN)
    factor::Float64         # Multiplication factor (for LM_MULTIPLICATIVE)
    trust_radius::Float64   # Trust region radius (for LM_TRUST_REGION)
    iteration::Int
    last_cost::Float64
    last_gain_ratio::Float64
end

function LMDampingState(;
    λ_init::Float64 = 1e-3,
    λ_min::Float64 = 1e-10,
    λ_max::Float64 = 1e10,
    policy::LMDampingPolicy = LM_NIELSEN,
    ν::Float64 = 2.0,
    factor::Float64 = 10.0,
    trust_radius::Float64 = 1.0
)
    LMDampingState(λ_init, λ_min, λ_max, policy, ν, factor, trust_radius, 0, Inf, 0.0)
end

"""
    update_damping!(state, cost_old, cost_new, model_decrease)

Update LM damping based on step outcome.

# Arguments
- `state`: LMDampingState to update
- `cost_old`: Cost before step
- `cost_new`: Cost after step
- `model_decrease`: Predicted decrease from linear model

# Returns
- `accept::Bool`: Whether to accept the step
"""
function update_damping!(
    state::LMDampingState,
    cost_old::Float64,
    cost_new::Float64,
    model_decrease::Float64
)
    state.iteration += 1
    actual_decrease = cost_old - cost_new

    # Gain ratio: actual decrease / predicted decrease
    ρ = model_decrease > 1e-15 ? actual_decrease / model_decrease : 0.0
    state.last_gain_ratio = ρ

    if state.policy == LM_FIXED
        # Fixed damping: always accept if cost decreased
        accept = cost_new < cost_old
        state.last_cost = accept ? cost_new : cost_old
        return accept

    elseif state.policy == LM_MULTIPLICATIVE
        # Simple multiplicative update
        if cost_new < cost_old
            # Success: decrease damping
            state.λ = max(state.λ / state.factor, state.λ_min)
            state.last_cost = cost_new
            return true
        else
            # Failure: increase damping
            state.λ = min(state.λ * state.factor, state.λ_max)
            return false
        end

    elseif state.policy == LM_NIELSEN
        # Nielsen's strategy (smooth)
        if ρ > 0
            # Good step: reduce damping
            state.λ *= max(1/3, 1 - (2*ρ - 1)^3)
            state.λ = max(state.λ, state.λ_min)
            state.ν = 2.0
            state.last_cost = cost_new
            return true
        else
            # Bad step: increase damping
            state.λ *= state.ν
            state.λ = min(state.λ, state.λ_max)
            state.ν *= 2
            return false
        end

    elseif state.policy == LM_TRUST_REGION
        # Trust region style
        if ρ > 0.75
            # Excellent step: expand trust region, decrease λ
            state.trust_radius *= 2
            state.λ = max(state.λ / 2, state.λ_min)
        elseif ρ < 0.25
            # Poor step: shrink trust region, increase λ
            state.trust_radius /= 2
            state.λ = min(state.λ * 2, state.λ_max)
        end

        accept = ρ > 0
        if accept
            state.last_cost = cost_new
        end
        return accept
    end

    return false
end

"""
    apply_lm_damping(H::AbstractMatrix, λ::Float64) -> Matrix

Apply LM damping to Hessian: H_damped = H + λ * diag(H)
"""
function apply_lm_damping(H::AbstractMatrix, λ::Float64)
    n = size(H, 1)
    H_damped = Matrix(H)
    for i in 1:n
        H_damped[i, i] += λ * max(H[i, i], 1e-10)
    end
    return H_damped
end

"""
    apply_lm_damping_identity(H::AbstractMatrix, λ::Float64) -> Matrix

Apply LM damping with identity: H_damped = H + λ * I
"""
function apply_lm_damping_identity(H::AbstractMatrix, λ::Float64)
    return Matrix(H) + λ * I
end

# ============================================================================
# Jacobi Preconditioning
# ============================================================================

"""
    JacobiPreconditioner

Diagonal Jacobi preconditioner for improving conditioning.

# Fields
- `diag_inv_sqrt::Vector{Float64}` - Inverse square root of diagonal
- `enabled::Bool` - Whether preconditioner is enabled
"""
struct JacobiPreconditioner
    diag_inv_sqrt::Vector{Float64}
    enabled::Bool
end

"""
    build_jacobi_preconditioner(H::AbstractMatrix; ε=1e-10) -> JacobiPreconditioner

Build Jacobi preconditioner from Hessian diagonal.

P = diag(1/√H_ii)

Preconditioned system: P*H*P * (P⁻¹*Δx) = P*g
"""
function build_jacobi_preconditioner(H::AbstractMatrix; ε::Float64 = 1e-10)
    n = size(H, 1)
    diag_inv_sqrt = zeros(n)

    for i in 1:n
        h_ii = abs(H[i, i])
        if h_ii > ε
            diag_inv_sqrt[i] = 1.0 / sqrt(h_ii)
        else
            diag_inv_sqrt[i] = 1.0  # Don't scale near-zero diagonal
        end
    end

    return JacobiPreconditioner(diag_inv_sqrt, true)
end

"""
    apply_preconditioner(P::JacobiPreconditioner, H::AbstractMatrix) -> Matrix

Apply preconditioner: P * H * P
"""
function apply_preconditioner(P::JacobiPreconditioner, H::AbstractMatrix)
    if !P.enabled
        return Matrix(H)
    end

    n = size(H, 1)
    H_prec = Matrix{Float64}(undef, n, n)

    for i in 1:n
        for j in 1:n
            H_prec[i, j] = P.diag_inv_sqrt[i] * H[i, j] * P.diag_inv_sqrt[j]
        end
    end

    return H_prec
end

"""
    apply_preconditioner(P::JacobiPreconditioner, g::AbstractVector) -> Vector

Apply preconditioner to gradient: P * g
"""
function apply_preconditioner(P::JacobiPreconditioner, g::AbstractVector)
    if !P.enabled
        return Vector(g)
    end
    return P.diag_inv_sqrt .* g
end

"""
    unapply_preconditioner(P::JacobiPreconditioner, Δz::AbstractVector) -> Vector

Recover original step from preconditioned: Δx = P * Δz
"""
function unapply_preconditioner(P::JacobiPreconditioner, Δz::AbstractVector)
    if !P.enabled
        return Vector(Δz)
    end
    return P.diag_inv_sqrt .* Δz
end

# ============================================================================
# Condition Number Monitoring
# ============================================================================

"""
    ConditioningStats

Statistics for monitoring numerical conditioning.

# Fields
- `condition_number::Float64` - Condition number (κ = λ_max/λ_min)
- `λ_min::Float64` - Minimum eigenvalue
- `λ_max::Float64` - Maximum eigenvalue
- `rank_deficient::Bool` - Whether matrix is rank deficient
- `n_small_eigenvalues::Int` - Number of eigenvalues below threshold
- `trace::Float64` - Matrix trace
- `log_det::Float64` - Log determinant
- `is_well_conditioned::Bool` - Whether matrix passes conditioning threshold
"""
struct ConditioningStats
    condition_number::Float64
    λ_min::Float64
    λ_max::Float64
    rank_deficient::Bool
    n_small_eigenvalues::Int
    trace::Float64
    log_det::Float64
    is_well_conditioned::Bool
end

"""
    compute_conditioning_stats(H::AbstractMatrix; ε=1e-10) -> ConditioningStats

Compute conditioning statistics for a matrix.
"""
function compute_conditioning_stats(H::AbstractMatrix; ε::Float64 = 1e-10)
    n = size(H, 1)

    # Eigenvalue decomposition
    try
        λs = eigvals(Symmetric(H))
        λs_real = real.(λs)
        λs_sorted = sort(λs_real)

        λ_min = λs_sorted[1]
        λ_max = λs_sorted[end]

        # Condition number
        if abs(λ_min) < ε
            cond = Inf
            rank_deficient = true
        else
            cond = abs(λ_max) / abs(λ_min)
            rank_deficient = false
        end

        # Count small eigenvalues
        n_small = count(λ -> abs(λ) < ε, λs_real)

        # Trace and log-det
        tr_H = sum(λs_real)
        log_det = sum(log.(max.(λs_real, ε)))

        # Well-conditioned threshold
        is_well_cond = cond < 1e12 && λ_min > 0 && !rank_deficient

        return ConditioningStats(
            cond, λ_min, λ_max, rank_deficient, n_small, tr_H, log_det, is_well_cond
        )
    catch e
        # Eigendecomposition failed
        return ConditioningStats(
            Inf, 0.0, 0.0, true, n, 0.0, -Inf, false
        )
    end
end

"""
    check_spd(H::AbstractMatrix; ε=1e-10) -> (is_spd::Bool, λ_min::Float64)

Check if matrix is symmetric positive definite.
"""
function check_spd(H::AbstractMatrix; ε::Float64 = 1e-10)
    # Check symmetry
    if !issymmetric(H)
        H_sym = (H + H') / 2
        max_asym = maximum(abs.(H - H_sym))
        if max_asym > 1e-8
            return (false, 0.0)
        end
    end

    # Check positive definiteness
    try
        λs = eigvals(Symmetric(H))
        λ_min = minimum(real.(λs))
        is_spd = λ_min > ε
        return (is_spd, λ_min)
    catch
        return (false, 0.0)
    end
end

"""
    regularize_hessian(H::AbstractMatrix, λ_target::Float64; ε=1e-10) -> Matrix

Regularize Hessian to ensure minimum eigenvalue ≥ λ_target.
"""
function regularize_hessian(H::AbstractMatrix, λ_target::Float64; ε::Float64 = 1e-10)
    is_spd, λ_min = check_spd(H; ε = ε)

    if is_spd && λ_min >= λ_target
        return Matrix(H)
    end

    # Add regularization
    shift = λ_target - λ_min + ε
    return Matrix(H) + shift * I
end

# ============================================================================
# Combined Conditioning Pipeline
# ============================================================================

"""
    ConditioningPipeline

Full conditioning pipeline combining scaling, damping, and preconditioning.

# Fields
- `scaling_config::ScalingConfig` - State scaling configuration
- `lm_state::LMDampingState` - LM damping state
- `use_jacobi::Bool` - Whether to use Jacobi preconditioning
- `monitor_condition::Bool` - Whether to monitor condition number
- `condition_threshold::Float64` - Condition number threshold for warning
- `regularization_min::Float64` - Minimum regularization to add
"""
struct ConditioningPipeline
    scaling_config::ScalingConfig
    lm_state::LMDampingState
    use_jacobi::Bool
    monitor_condition::Bool
    condition_threshold::Float64
    regularization_min::Float64
end

function ConditioningPipeline(;
    scaling_config::ScalingConfig = DEFAULT_SCALING_CONFIG,
    lm_policy::LMDampingPolicy = LM_NIELSEN,
    λ_init::Float64 = 1e-3,
    use_jacobi::Bool = true,
    monitor_condition::Bool = true,
    condition_threshold::Float64 = 1e10,
    regularization_min::Float64 = 1e-6
)
    lm_state = LMDampingState(; λ_init = λ_init, policy = lm_policy)
    ConditioningPipeline(
        scaling_config, lm_state, use_jacobi, monitor_condition,
        condition_threshold, regularization_min
    )
end

"""
    SolveResult

Result from conditioned solve.

# Fields
- `Δx::Vector{Float64}` - Solution step
- `success::Bool` - Whether solve succeeded
- `iterations::Int` - Number of inner iterations
- `final_λ::Float64` - Final damping value
- `condition_stats::ConditioningStats` - Conditioning statistics
- `regularization_added::Float64` - Amount of regularization added
"""
struct SolveResult
    Δx::Vector{Float64}
    success::Bool
    iterations::Int
    final_λ::Float64
    condition_stats::ConditioningStats
    regularization_added::Float64
end

"""
    solve_conditioned(H, g, pipeline; max_iterations=10) -> SolveResult

Solve H * Δx = -g with conditioning pipeline.

Applies:
1. Jacobi preconditioning (if enabled)
2. LM damping
3. Condition number monitoring
4. Regularization if needed
"""
function solve_conditioned(
    H::AbstractMatrix,
    g::AbstractVector,
    pipeline::ConditioningPipeline;
    max_iterations::Int = 10
)
    n = length(g)
    @assert size(H) == (n, n)

    # Step 1: Build Jacobi preconditioner
    if pipeline.use_jacobi
        P = build_jacobi_preconditioner(H)
        H_work = apply_preconditioner(P, H)
        g_work = apply_preconditioner(P, g)
    else
        P = JacobiPreconditioner(ones(n), false)
        H_work = Matrix(H)
        g_work = Vector(g)
    end

    # Step 2: Check initial conditioning
    initial_stats = compute_conditioning_stats(H_work)
    regularization_added = 0.0

    # Step 3: Apply regularization if poorly conditioned
    if !initial_stats.is_well_conditioned || initial_stats.condition_number > pipeline.condition_threshold
        reg = max(pipeline.regularization_min, abs(initial_stats.λ_min) + pipeline.regularization_min)
        H_work = H_work + reg * I
        regularization_added = reg
    end

    # Step 4: Apply LM damping and solve
    λ = pipeline.lm_state.λ
    H_damped = apply_lm_damping(H_work, λ)

    # Step 5: Solve the system
    Δz = zeros(n)
    success = false
    iterations = 0

    for iter in 1:max_iterations
        iterations = iter

        try
            # Cholesky factorization (exploits SPD structure)
            C = cholesky(Symmetric(H_damped))
            Δz = -(C \ g_work)
            success = true
            break
        catch e
            if e isa PosDefException
                # Increase damping and try again
                λ *= 10
                if λ > pipeline.lm_state.λ_max
                    break
                end
                H_damped = apply_lm_damping(H_work, λ)
            else
                # Other error, try LU fallback
                try
                    Δz = -(H_damped \ g_work)
                    success = true
                    break
                catch
                    break
                end
            end
        end
    end

    # Step 6: Undo preconditioning
    Δx = unapply_preconditioner(P, Δz)

    # Step 7: Final conditioning stats
    final_stats = compute_conditioning_stats(H_damped)

    return SolveResult(Δx, success, iterations, λ, final_stats, regularization_added)
end
