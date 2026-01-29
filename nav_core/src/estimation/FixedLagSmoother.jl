# =============================================================================
# FixedLagSmoother.jl - Fixed-Lag Smoothing for Urban Navigation
# =============================================================================
#
# Purpose: Refine past pose estimates using future measurements within a
# bounded time window. Improves accuracy over pure filtering by allowing
# measurements to inform earlier states.
#
# Theory:
# A fixed-lag smoother maintains a window of L recent keyframes and
# jointly optimizes them using all measurements in the window. When
# keyframes exit the window, their information is preserved via Schur
# complement marginalization into a prior on the remaining states.
#
# This is equivalent to solving the MAP estimate:
#   x* = argmax p(x_{t-L:t} | z_{1:t})
# rather than the filter estimate p(x_t | z_{1:t}).
#
# Key property: smoothed estimates have lower covariance than filtered
# estimates because they use both past and future measurements.
#
# Complexity:
# - Per-step cost: O(L × d²) where d = state dimension per keyframe
# - Memory: O(L × d) for states + O(L × d²) for covariances
# - The Schur complement costs O(d³) per marginalized keyframe
#
# References:
#   - Barfoot, "State Estimation for Robotics", Ch. 8 (batch estimation)
#   - Kaess et al., "iSAM2: Incremental Smoothing and Mapping Using the
#     Bayes Tree", IJRR 2012 (incremental smoothing)
#   - Dong et al., "Motion Planning as Probabilistic Inference using
#     Gaussian Processes", RSS 2016 (fixed-lag GP smoothing)
#
# =============================================================================

module FixedLagSmoother

using LinearAlgebra
using StaticArrays
using Statistics

# =============================================================================
# Types
# =============================================================================

"""
Configuration for fixed-lag smoother.

# Parameter Justifications

## lag_keyframes = 10
At 1 Hz keyframe rate and 2 m/s AUV speed, 10 keyframes spans 20m.
Magnetic correlation length is typically 10-50m (Blakely, 1996),
so a 10-keyframe window captures the relevant spatial correlation.

## max_iterations = 5
Gauss-Newton typically converges in 3-5 iterations for well-conditioned
problems. 5 iterations provides margin without excessive computation.

## convergence_tolerance = 1e-6
Relative cost reduction threshold. Below this, further iterations
provide negligible accuracy improvement (sub-nanoTesla level changes
in typical AUV scenarios).

## lambda_init = 1e-4
Initial Levenberg-Marquardt damping. Small value starts near
Gauss-Newton (fast convergence) with LM fallback if needed.

## marginalization_min_eigenvalue = 1e-10
Minimum eigenvalue for marginalized Hessian block. Below this,
the state is unobservable and marginalization would inject
ill-conditioned information. Based on typical condition numbers
of ~1e8 for AUV navigation Hessians.
"""
struct SmootherConfig
    lag_keyframes::Int
    max_iterations::Int
    convergence_tolerance::Float64
    lambda_init::Float64
    lambda_factor::Float64
    marginalization_min_eigenvalue::Float64

    function SmootherConfig(;
        lag_keyframes::Int = 10,
        max_iterations::Int = 5,
        convergence_tolerance::Float64 = 1e-6,
        lambda_init::Float64 = 1e-4,
        lambda_factor::Float64 = 10.0,
        marginalization_min_eigenvalue::Float64 = 1e-10
    )
        @assert lag_keyframes >= 2 "Lag must be ≥ 2 keyframes"
        @assert max_iterations >= 1
        @assert convergence_tolerance > 0
        new(lag_keyframes, max_iterations, convergence_tolerance,
            lambda_init, lambda_factor, marginalization_min_eigenvalue)
    end
end

"""
A keyframe in the smoother window.

Contains the state estimate, associated measurements, and
linearization point for the Gauss-Newton solver.
"""
mutable struct SmootherKeyframe
    index::Int
    timestamp::Float64
    state::Vector{Float64}          # State vector (dim depends on problem)
    covariance::Matrix{Float64}     # Marginal covariance
    measurements::Vector{Any}       # Associated measurements
    is_marginalized::Bool           # True if already marginalized out
end

"""
A factor (constraint) between states in the smoother.

Represents a measurement or motion model constraint linking one or
more keyframes. Factors are linearized at each Gauss-Newton iteration.

# Fields
- `type`: Factor type (:motion, :magnetic, :odometry, :barometer, etc.)
- `keyframe_indices`: Which keyframes this factor connects
- `residual_fn`: Function(states...) → residual vector
- `jacobian_fn`: Function(states...) → tuple of Jacobian matrices
- `noise_covariance`: Measurement noise R (or process noise Q)
- `dimension`: Residual dimension
"""
struct SmootherFactor
    type::Symbol
    keyframe_indices::Vector{Int}
    residual_fn::Function
    jacobian_fn::Function
    noise_covariance::Matrix{Float64}
    dimension::Int
end

"""
Prior from marginalized keyframes.

When a keyframe exits the window, its information is preserved as
a quadratic prior on the remaining states via Schur complement:

  J_prior = -½ (x - x₀)' Λ_prior (x - x₀)

where Λ_prior = H_rr - H_rm H_mm⁻¹ H_mr (Schur complement of
the marginalized block in the full Hessian).

This is the "marginalization prior" in SLAM literature.
"""
struct MarginalizationPrior
    linearization_point::Vector{Float64}  # x₀ at which prior was computed
    information_matrix::Matrix{Float64}   # Λ_prior (Schur complement)
    residual::Vector{Float64}             # b_prior (gradient at linearization)
    connected_keyframe_indices::Vector{Int}
end

"""
Result of a smoother optimization step.
"""
struct SmootherResult
    converged::Bool
    iterations::Int
    initial_cost::Float64
    final_cost::Float64
    states::Vector{Vector{Float64}}       # Smoothed states
    covariances::Vector{Matrix{Float64}}  # Marginal covariances
end

"""
The fixed-lag smoother.

Maintains a sliding window of keyframes, jointly optimizes them,
and marginalizes old keyframes as they exit the window.
"""
mutable struct FixedLagSmootherState
    config::SmootherConfig
    keyframes::Vector{SmootherKeyframe}
    factors::Vector{SmootherFactor}
    priors::Vector{MarginalizationPrior}
    state_dimension::Int                  # Dimension per keyframe
    next_index::Int
end

function FixedLagSmootherState(config::SmootherConfig; state_dim::Int = 15)
    FixedLagSmootherState(
        config,
        SmootherKeyframe[],
        SmootherFactor[],
        MarginalizationPrior[],
        state_dim,
        1
    )
end

# =============================================================================
# Keyframe Management
# =============================================================================

"""
    add_keyframe!(smoother, timestamp, state, covariance) -> Int

Add a new keyframe to the smoother window. Returns the keyframe index.

If the window exceeds the lag, the oldest keyframe is marginalized.
"""
function add_keyframe!(
    smoother::FixedLagSmootherState,
    timestamp::Float64,
    state::Vector{Float64},
    covariance::Matrix{Float64}
)
    idx = smoother.next_index
    smoother.next_index += 1

    kf = SmootherKeyframe(
        idx, timestamp, copy(state), copy(covariance),
        Any[], false
    )
    push!(smoother.keyframes, kf)

    # Enforce window size
    while length(smoother.keyframes) > smoother.config.lag_keyframes
        marginalize_oldest!(smoother)
    end

    return idx
end

"""
    add_factor!(smoother, factor)

Add a measurement or motion factor to the smoother.
"""
function add_factor!(smoother::FixedLagSmootherState, factor::SmootherFactor)
    push!(smoother.factors, factor)
end

# =============================================================================
# Marginalization via Schur Complement
# =============================================================================

"""
    marginalize_oldest!(smoother)

Marginalize the oldest keyframe out of the window.

Uses the Schur complement to preserve the oldest keyframe's
information as a prior on the remaining states:

    H = [H_mm  H_mr]     →  Λ_prior = H_rr - H_mr' H_mm⁻¹ H_mr
        [H_mr' H_rr]

where m = marginalized, r = remaining.

# Numerical Safety
- Checks minimum eigenvalue of H_mm before inverting
- Falls back to pseudoinverse if H_mm is near-singular
- This prevents ill-conditioned priors from poisoning the window
"""
function marginalize_oldest!(smoother::FixedLagSmootherState)
    if isempty(smoother.keyframes)
        return
    end

    oldest = smoother.keyframes[1]
    d = smoother.state_dimension

    # Collect factors involving the oldest keyframe
    involved_factor_indices = Int[]
    connected_kf_indices = Int[]

    for (fi, factor) in enumerate(smoother.factors)
        if oldest.index in factor.keyframe_indices
            push!(involved_factor_indices, fi)
            for ki in factor.keyframe_indices
                if ki != oldest.index && !(ki in connected_kf_indices)
                    push!(connected_kf_indices, ki)
                end
            end
        end
    end

    if !isempty(connected_kf_indices)
        # Build local Hessian from involved factors
        n_connected = length(connected_kf_indices)
        total_dim = d + n_connected * d  # marginalized + remaining

        H_local = zeros(total_dim, total_dim)
        b_local = zeros(total_dim)

        # Map keyframe indices to local block positions
        # Block 0: oldest (to be marginalized)
        # Block 1..n: connected keyframes
        kf_to_block = Dict{Int, Int}()
        kf_to_block[oldest.index] = 0
        for (bi, ki) in enumerate(connected_kf_indices)
            kf_to_block[ki] = bi
        end

        for fi in involved_factor_indices
            factor = smoother.factors[fi]
            # Get states for this factor
            states = Vector{Float64}[]
            for ki in factor.keyframe_indices
                kf = _find_keyframe(smoother, ki)
                if kf !== nothing
                    push!(states, kf.state)
                end
            end

            if length(states) != length(factor.keyframe_indices)
                continue  # Skip if keyframe not found
            end

            # Compute residual and Jacobians
            r = factor.residual_fn(states...)
            Js = factor.jacobian_fn(states...)

            # Information-weight the residual
            R_inv = inv(factor.noise_covariance)

            # Accumulate into local Hessian
            for (i, ki) in enumerate(factor.keyframe_indices)
                bi = kf_to_block[ki]
                row_range = (bi * d + 1):((bi + 1) * d)
                Ji = Js[i]

                # Gradient
                b_local[row_range] .+= Ji' * R_inv * r

                for (j, kj) in enumerate(factor.keyframe_indices)
                    bj = kf_to_block[kj]
                    col_range = (bj * d + 1):((bj + 1) * d)
                    Jj = Js[j]

                    # Hessian block
                    H_local[row_range, col_range] .+= Ji' * R_inv * Jj
                end
            end
        end

        # Add prior from oldest keyframe's covariance
        if !all(oldest.covariance .== 0)
            P_inv = inv(oldest.covariance + 1e-12 * I)
            H_local[1:d, 1:d] .+= P_inv
        end

        # Schur complement
        H_mm = H_local[1:d, 1:d]
        H_mr = H_local[1:d, (d+1):end]
        H_rr = H_local[(d+1):end, (d+1):end]
        b_m = b_local[1:d]
        b_r = b_local[(d+1):end]

        # Check conditioning of H_mm
        min_eig = minimum(eigvals(Symmetric(H_mm)))

        if min_eig > smoother.config.marginalization_min_eigenvalue
            H_mm_inv = inv(Symmetric(H_mm))
        else
            # Use pseudoinverse for near-singular blocks
            H_mm_inv = pinv(H_mm)
        end

        # Schur complement: Λ_prior = H_rr - H_mr' H_mm⁻¹ H_mr
        Lambda_prior = H_rr - H_mr' * H_mm_inv * H_mr
        b_prior = b_r - H_mr' * H_mm_inv * b_m

        # Symmetrize
        Lambda_prior = 0.5 * (Lambda_prior + Lambda_prior')

        # Build linearization point from connected keyframes
        lin_point = Float64[]
        for ki in connected_kf_indices
            kf = _find_keyframe(smoother, ki)
            if kf !== nothing
                append!(lin_point, kf.state)
            end
        end

        prior = MarginalizationPrior(
            lin_point,
            Lambda_prior,
            b_prior,
            connected_kf_indices
        )
        push!(smoother.priors, prior)
    end

    # Remove oldest keyframe and its factors
    popfirst!(smoother.keyframes)
    filter!(f -> !(oldest.index in f.keyframe_indices), smoother.factors)
end

"""Find a keyframe by index."""
function _find_keyframe(smoother::FixedLagSmootherState, index::Int)
    for kf in smoother.keyframes
        if kf.index == index
            return kf
        end
    end
    return nothing
end

# =============================================================================
# Gauss-Newton Optimization
# =============================================================================

"""
    optimize!(smoother) -> SmootherResult

Run Gauss-Newton optimization on the current window.

Assembles the normal equations from all factors and marginalization
priors, then solves for the optimal state update:

    (H + λI) Δx = -b

where H = Σ Jᵢ' Rᵢ⁻¹ Jᵢ and b = Σ Jᵢ' Rᵢ⁻¹ rᵢ

Uses Levenberg-Marquardt damping for robustness.
"""
function optimize!(smoother::FixedLagSmootherState)
    if isempty(smoother.keyframes)
        return SmootherResult(true, 0, 0.0, 0.0, Vector{Float64}[], Matrix{Float64}[])
    end

    n_kf = length(smoother.keyframes)
    d = smoother.state_dimension
    total_dim = n_kf * d

    # Map keyframe indices to local block positions
    kf_to_block = Dict{Int, Int}()
    for (bi, kf) in enumerate(smoother.keyframes)
        kf_to_block[kf.index] = bi - 1  # 0-indexed blocks
    end

    lambda = smoother.config.lambda_init
    initial_cost = _compute_cost(smoother, kf_to_block)
    prev_cost = initial_cost

    converged = false
    iters = 0

    for iter in 1:smoother.config.max_iterations
        iters = iter

        # Assemble normal equations
        H = zeros(total_dim, total_dim)
        b = zeros(total_dim)

        # Accumulate from factors
        for factor in smoother.factors
            states = Vector{Float64}[]
            block_ids = Int[]
            all_found = true

            for ki in factor.keyframe_indices
                if haskey(kf_to_block, ki)
                    kf = _find_keyframe(smoother, ki)
                    if kf !== nothing
                        push!(states, kf.state)
                        push!(block_ids, kf_to_block[ki])
                    else
                        all_found = false
                    end
                else
                    all_found = false
                end
            end

            if !all_found || isempty(states)
                continue
            end

            r = factor.residual_fn(states...)
            Js = factor.jacobian_fn(states...)
            R_inv = inv(factor.noise_covariance)

            for (i, bi) in enumerate(block_ids)
                row_range = (bi * d + 1):((bi + 1) * d)
                Ji = Js[i]

                b[row_range] .+= Ji' * R_inv * r

                for (j, bj) in enumerate(block_ids)
                    col_range = (bj * d + 1):((bj + 1) * d)
                    Jj = Js[j]
                    H[row_range, col_range] .+= Ji' * R_inv * Jj
                end
            end
        end

        # Accumulate from marginalization priors
        for prior in smoother.priors
            _accumulate_prior!(H, b, prior, kf_to_block, d)
        end

        # LM damping
        H_damped = H + lambda * Diagonal(max.(diag(H), 1e-6))

        # Solve
        try
            delta = -(H_damped \ b)

            # Update states
            saved_states = [copy(kf.state) for kf in smoother.keyframes]

            for (bi, kf) in enumerate(smoother.keyframes)
                idx_range = ((bi - 1) * d + 1):(bi * d)
                kf.state .+= delta[idx_range]
            end

            # Check cost
            new_cost = _compute_cost(smoother, kf_to_block)

            if new_cost < prev_cost
                # Accept step, reduce damping
                lambda = max(lambda / smoother.config.lambda_factor, 1e-10)

                relative_improvement = (prev_cost - new_cost) / max(abs(prev_cost), 1e-12)
                prev_cost = new_cost

                if relative_improvement < smoother.config.convergence_tolerance
                    converged = true
                    break
                end
            else
                # Reject step, increase damping, restore states
                for (bi, kf) in enumerate(smoother.keyframes)
                    kf.state .= saved_states[bi]
                end
                lambda = min(lambda * smoother.config.lambda_factor, 1e10)
            end
        catch e
            if e isa SingularException || e isa LAPACKException
                lambda *= smoother.config.lambda_factor
            else
                rethrow(e)
            end
        end
    end

    # Extract marginal covariances
    covariances = _extract_covariances(smoother, kf_to_block)
    states = [copy(kf.state) for kf in smoother.keyframes]

    return SmootherResult(converged, iters, initial_cost, prev_cost, states, covariances)
end

"""Compute total cost from all factors."""
function _compute_cost(smoother::FixedLagSmootherState, kf_to_block::Dict{Int,Int})
    cost = 0.0

    for factor in smoother.factors
        states = Vector{Float64}[]
        all_found = true

        for ki in factor.keyframe_indices
            if haskey(kf_to_block, ki)
                kf = _find_keyframe(smoother, ki)
                if kf !== nothing
                    push!(states, kf.state)
                else
                    all_found = false
                end
            else
                all_found = false
            end
        end

        if !all_found || isempty(states)
            continue
        end

        r = factor.residual_fn(states...)
        R_inv = inv(factor.noise_covariance)
        cost += 0.5 * r' * R_inv * r
    end

    return cost
end

"""Accumulate marginalization prior into Hessian and gradient."""
function _accumulate_prior!(
    H::Matrix{Float64},
    b::Vector{Float64},
    prior::MarginalizationPrior,
    kf_to_block::Dict{Int,Int},
    d::Int
)
    n_connected = length(prior.connected_keyframe_indices)

    for (i, ki) in enumerate(prior.connected_keyframe_indices)
        if !haskey(kf_to_block, ki)
            continue
        end
        bi = kf_to_block[ki]
        row_range = (bi * d + 1):((bi + 1) * d)
        prior_row = ((i - 1) * d + 1):(i * d)

        # Gradient contribution
        b[row_range] .+= prior.residual[prior_row]

        for (j, kj) in enumerate(prior.connected_keyframe_indices)
            if !haskey(kf_to_block, kj)
                continue
            end
            bj = kf_to_block[kj]
            col_range = (bj * d + 1):((bj + 1) * d)
            prior_col = ((j - 1) * d + 1):(j * d)

            # Hessian contribution
            H[row_range, col_range] .+= prior.information_matrix[prior_row, prior_col]
        end
    end
end

"""Extract marginal covariances via Hessian inversion."""
function _extract_covariances(smoother::FixedLagSmootherState, kf_to_block::Dict{Int,Int})
    n_kf = length(smoother.keyframes)
    d = smoother.state_dimension
    total_dim = n_kf * d

    # Rebuild Hessian at current linearization point
    H = zeros(total_dim, total_dim)
    b = zeros(total_dim)

    for factor in smoother.factors
        states = Vector{Float64}[]
        block_ids = Int[]
        all_found = true

        for ki in factor.keyframe_indices
            if haskey(kf_to_block, ki)
                kf = _find_keyframe(smoother, ki)
                if kf !== nothing
                    push!(states, kf.state)
                    push!(block_ids, kf_to_block[ki])
                else
                    all_found = false
                end
            else
                all_found = false
            end
        end

        if !all_found || isempty(states)
            continue
        end

        Js = factor.jacobian_fn(states...)
        R_inv = inv(factor.noise_covariance)

        for (i, bi) in enumerate(block_ids)
            row_range = (bi * d + 1):((bi + 1) * d)
            Ji = Js[i]
            for (j, bj) in enumerate(block_ids)
                col_range = (bj * d + 1):((bj + 1) * d)
                Jj = Js[j]
                H[row_range, col_range] .+= Ji' * R_inv * Jj
            end
        end
    end

    # Add priors
    for prior in smoother.priors
        _accumulate_prior!(H, b, prior, kf_to_block, d)
    end

    # Regularize and invert
    H_sym = Symmetric(H + 1e-10 * I)
    try
        Sigma_full = inv(H_sym)
        return [Sigma_full[(bi*d+1):((bi+1)*d), (bi*d+1):((bi+1)*d)]
                for bi in 0:(n_kf-1)]
    catch
        # Return filter covariances as fallback
        return [copy(kf.covariance) for kf in smoother.keyframes]
    end
end

# =============================================================================
# Convenience: Smoothed State Query
# =============================================================================

"""
    get_smoothed_state(smoother, index) -> (state, covariance) or nothing

Get the smoothed state and covariance for a keyframe by index.
"""
function get_smoothed_state(smoother::FixedLagSmootherState, index::Int)
    kf = _find_keyframe(smoother, index)
    if kf === nothing
        return nothing
    end
    return (copy(kf.state), copy(kf.covariance))
end

"""
    get_latest_smoothed(smoother) -> (state, covariance) or nothing

Get the most recent smoothed state.
"""
function get_latest_smoothed(smoother::FixedLagSmootherState)
    if isempty(smoother.keyframes)
        return nothing
    end
    kf = smoother.keyframes[end]
    return (copy(kf.state), copy(kf.covariance))
end

"""
    window_size(smoother) -> Int

Current number of keyframes in the window.
"""
window_size(smoother::FixedLagSmootherState) = length(smoother.keyframes)

"""
    window_timestamps(smoother) -> Vector{Float64}

Timestamps of all keyframes in the current window.
"""
window_timestamps(smoother::FixedLagSmootherState) =
    [kf.timestamp for kf in smoother.keyframes]

# =============================================================================
# Exports
# =============================================================================

export SmootherConfig
export SmootherKeyframe, SmootherFactor, MarginalizationPrior
export SmootherResult, FixedLagSmootherState
export add_keyframe!, add_factor!, optimize!
export marginalize_oldest!
export get_smoothed_state, get_latest_smoothed
export window_size, window_timestamps

end # module
