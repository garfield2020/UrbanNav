# ============================================================================
# covariance.jl - Covariance management and utilities
# ============================================================================

export CovarianceManager, update_covariance!, check_consistency
export extract_subcovariance, reset_covariance!

"""
    CovarianceManager{T}

Manages covariance matrix updates and consistency checks.

# Fields
- `P::Matrix{T}` - Current covariance matrix
- `P_pred::Matrix{T}` - Predicted covariance (before update)
- `condition_threshold::T` - Max condition number before reset
- `min_eigenvalue::T` - Minimum allowed eigenvalue
"""
mutable struct CovarianceManager{T<:Real}
    P::Matrix{T}
    P_pred::Matrix{T}
    condition_threshold::T
    min_eigenvalue::T
end

"""
    CovarianceManager(P::Matrix; condition_threshold=1e10, min_eigenvalue=1e-12)

Create a covariance manager with initial covariance P.
"""
function CovarianceManager(P::Matrix{T};
                           condition_threshold::T=T(1e10),
                           min_eigenvalue::T=T(1e-12)) where T
    return CovarianceManager{T}(
        copy(P), copy(P), condition_threshold, min_eigenvalue
    )
end

"""
    update_covariance!(manager, P_new)

Update the covariance matrix with consistency checks.
"""
function update_covariance!(manager::CovarianceManager{T}, P_new::Matrix{T}) where T
    # Store prediction
    manager.P_pred .= manager.P

    # Symmetrize
    P_sym = (P_new + P_new') / 2

    # Check positive definiteness
    λ = eigvals(Symmetric(P_sym))
    if minimum(λ) < manager.min_eigenvalue
        # Regularize
        P_sym = P_sym + (manager.min_eigenvalue - minimum(λ)) * I
    end

    # Check condition number
    if condition_number(P_sym) > manager.condition_threshold
        @warn "Covariance condition number exceeded threshold, consider reset"
    end

    manager.P .= P_sym
    return manager.P
end

"""
    check_consistency(manager) -> NamedTuple

Check covariance matrix consistency.
"""
function check_consistency(manager::CovarianceManager{T}) where T
    P = manager.P
    λ = eigvals(Symmetric(P))

    return (
        is_symmetric = norm(P - P') < 1e-10,
        is_positive_definite = minimum(λ) > 0,
        min_eigenvalue = minimum(λ),
        max_eigenvalue = maximum(λ),
        condition_number = maximum(λ) / max(minimum(λ), eps(T)),
        trace = tr(P)
    )
end

"""
    extract_subcovariance(manager, indices) -> Matrix

Extract a subblock of the covariance matrix.
"""
function extract_subcovariance(manager::CovarianceManager, indices::AbstractVector{Int})
    return manager.P[indices, indices]
end

"""
    reset_covariance!(manager, P_new)

Force reset of covariance matrix (bypasses consistency checks).
"""
function reset_covariance!(manager::CovarianceManager{T}, P_new::Matrix{T}) where T
    manager.P .= (P_new + P_new') / 2
    manager.P_pred .= manager.P
    return manager.P
end

"""
    joseph_update(P, K, H, R) -> Matrix

Joseph form covariance update for numerical stability.
P_new = (I - KH)P(I - KH)' + KRK'
"""
function joseph_update(P::AbstractMatrix{T}, K::AbstractMatrix{T},
                       H::AbstractMatrix{T}, R::AbstractMatrix{T}) where T
    n = size(P, 1)
    IKH = I - K * H
    return IKH * P * IKH' + K * R * K'
end

"""
    schmidt_kalman_update(P, K, H, R, consider_indices) -> Matrix

Schmidt-Kalman update that considers but doesn't update certain states.
Used for consider covariance analysis.
"""
function schmidt_kalman_update(P::AbstractMatrix{T}, K::AbstractMatrix{T},
                               H::AbstractMatrix{T}, R::AbstractMatrix{T},
                               consider_indices::AbstractVector{Int}) where T
    n = size(P, 1)

    # Build consider matrix
    C = zeros(T, n, n)
    for i in consider_indices
        C[i, i] = one(T)
    end

    # Modified gain
    K_mod = K * (I - C)

    return joseph_update(P, K_mod, H, R)
end
