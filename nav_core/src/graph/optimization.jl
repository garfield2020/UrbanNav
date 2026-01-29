# ============================================================================
# optimization.jl - Factor graph optimization (Gauss-Newton / LM)
# ============================================================================

export optimize!, OptimizationParams, OptimizationResult

"""
    OptimizationParams

Parameters for factor graph optimization.

# Fields
- `max_iterations::Int` - Maximum iterations (default: 100)
- `tolerance::Float64` - Convergence tolerance (default: 1e-6)
- `lambda_init::Float64` - Initial LM damping (default: 1e-4)
- `lambda_factor::Float64` - LM damping adjustment (default: 10)
- `use_lm::Bool` - Use Levenberg-Marquardt (default: true)
"""
Base.@kwdef struct OptimizationParams
    max_iterations::Int = 100
    tolerance::Float64 = 1e-6
    lambda_init::Float64 = 1e-4
    lambda_factor::Float64 = 10.0
    use_lm::Bool = true
end

"""
    OptimizationResult

Result of factor graph optimization.

# Fields
- `converged::Bool` - Whether optimization converged
- `iterations::Int` - Number of iterations performed
- `initial_error::Float64` - Initial total error
- `final_error::Float64` - Final total error
- `lambda::Float64` - Final LM damping parameter
"""
struct OptimizationResult
    converged::Bool
    iterations::Int
    initial_error::Float64
    final_error::Float64
    lambda::Float64
end

"""
    optimize!(graph, params=OptimizationParams()) -> OptimizationResult

Optimize the factor graph using Gauss-Newton or Levenberg-Marquardt.
"""
function optimize!(graph::FactorGraph{T}, params::OptimizationParams=OptimizationParams()) where T
    n = total_state_dim(graph)
    λ = params.lambda_init

    # Compute initial error
    initial_error = total_error(graph)
    prev_error = initial_error

    converged = false
    iter = 0

    for iter in 1:params.max_iterations
        # Build linear system: H * δ = b
        H, b = build_linear_system(graph)

        if params.use_lm
            # Add LM damping
            H += λ * Diagonal(diag(H))
        end

        # Solve for delta
        local δ
        try
            δ = H \ b
        catch e
            # System is singular, increase damping
            λ *= params.lambda_factor
            continue
        end

        # Check convergence
        if norm(δ) < params.tolerance
            converged = true
            break
        end

        # Update variables
        update_variables!(graph, δ)

        # Compute new error
        new_error = total_error(graph)

        if params.use_lm
            if new_error < prev_error
                # Accept update, decrease damping
                λ /= params.lambda_factor
                prev_error = new_error
            else
                # Reject update, increase damping
                update_variables!(graph, -δ)
                λ *= params.lambda_factor
            end
        else
            prev_error = new_error
        end
    end

    return OptimizationResult(converged, iter, initial_error, prev_error, λ)
end

"""
    total_error(graph) -> Float64

Compute total squared error of all factors.
"""
function total_error(graph::FactorGraph)
    error = 0.0
    for factor in graph.factors
        states = get_factor_states(graph, factor)
        r = residual(factor, states...)
        r_w = whiten(factor, r)
        error += dot(r_w, r_w)
    end
    return error / 2
end

"""
    build_linear_system(graph) -> (H, b)

Build the Hessian approximation and gradient vector.
"""
function build_linear_system(graph::FactorGraph{T}) where T
    n = total_state_dim(graph)
    H = zeros(T, n, n)
    b = zeros(T, n)

    # Build index map
    var_indices = build_variable_index_map(graph)

    for factor in graph.factors
        states = get_factor_states(graph, factor)
        connected = connected_states(factor)

        # Get residual and Jacobians
        r = residual(factor, states...)
        Js = jacobian(factor, states...)
        Σ = noise_model(factor)
        Σ_inv = inv(Symmetric(Σ))

        # Accumulate into H and b
        for (i, var_i) in enumerate(connected)
            Ji = Js[i]
            idx_i = var_indices[var_i]

            # Gradient contribution
            b[idx_i] .+= Ji' * Σ_inv * r

            for (j, var_j) in enumerate(connected)
                Jj = Js[j]
                idx_j = var_indices[var_j]

                # Hessian contribution
                H[idx_i, idx_j] .+= Ji' * Σ_inv * Jj
            end
        end
    end

    return Symmetric(H), b
end

"""
    build_variable_index_map(graph) -> Dict{Int, UnitRange}

Map variable IDs to their indices in the state vector.
"""
function build_variable_index_map(graph::FactorGraph)
    indices = Dict{Int, UnitRange{Int}}()
    idx = 1
    for var_id in graph.variable_order
        var = graph.variables[var_id]
        indices[var_id] = idx:(idx + var.dim - 1)
        idx += var.dim
    end
    return indices
end

"""
    get_factor_states(graph, factor) -> Tuple

Get state values for all variables connected to a factor.
"""
function get_factor_states(graph::FactorGraph, factor::AbstractFactor)
    var_ids = connected_states(factor)
    return Tuple(graph.variables[id].value for id in var_ids)
end

"""
    marginal_covariance(graph, var_id) -> Matrix

Compute marginal covariance for a variable.
"""
function marginal_covariance(graph::FactorGraph, var_id::Int)
    H, _ = build_linear_system(graph)
    var_indices = build_variable_index_map(graph)
    idx = var_indices[var_id]

    # Full covariance (expensive)
    P_full = inv(Symmetric(H))

    return P_full[idx, idx]
end
