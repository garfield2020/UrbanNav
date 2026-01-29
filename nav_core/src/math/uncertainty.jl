# ============================================================================
# uncertainty.jl - Covariance propagation and uncertainty utilities
# ============================================================================

export propagate_covariance, mahalanobis_distance, sigma_points
export covariance_intersection, covariance_union

"""
    propagate_covariance(P, F, Q) -> Matrix

Propagate covariance through linear dynamics.
P_new = F * P * F' + Q
"""
function propagate_covariance(P::AbstractMatrix, F::AbstractMatrix, Q::AbstractMatrix)
    return F * P * F' + Q
end

"""
    mahalanobis_distance(r, S) -> Float64

Compute Mahalanobis distance for residual r with covariance S.
d = sqrt(r' * S^(-1) * r)
"""
function mahalanobis_distance(r::AbstractVector, S::AbstractMatrix)
    return sqrt(r' * (S \ r))
end

"""
    sigma_points(x, P, α=1e-3, β=2, κ=0) -> Matrix

Generate sigma points for unscented transform.
Returns matrix where each column is a sigma point.
"""
function sigma_points(x::AbstractVector{T}, P::AbstractMatrix{T};
                      α::T=T(1e-3), β::T=T(2), κ::T=T(0)) where T
    n = length(x)
    λ = α^2 * (n + κ) - n

    # Compute matrix square root
    L = cholesky(Symmetric((n + λ) * P)).L

    # Generate sigma points
    X = zeros(T, n, 2n + 1)
    X[:, 1] = x
    for i in 1:n
        X[:, i+1] = x + L[:, i]
        X[:, i+n+1] = x - L[:, i]
    end

    return X
end

"""
    covariance_intersection(P1, P2, ω=0.5) -> Matrix

Covariance intersection for conservative fusion of two estimates.
Useful when correlation is unknown.
"""
function covariance_intersection(P1::AbstractMatrix{T}, P2::AbstractMatrix{T};
                                  ω::T=T(0.5)) where T
    P1_inv = inv(Symmetric(P1))
    P2_inv = inv(Symmetric(P2))
    P_fused_inv = ω * P1_inv + (1 - ω) * P2_inv
    return inv(Symmetric(P_fused_inv))
end

"""
    covariance_union(P1, P2) -> Matrix

Conservative covariance union (outer bound).
"""
function covariance_union(P1::AbstractMatrix{T}, P2::AbstractMatrix{T}) where T
    # Simple outer ellipsoid approximation
    return P1 + P2
end

"""
    ensure_positive_definite(P, ε=1e-10) -> Matrix

Ensure matrix is positive definite by adding small regularization.
"""
function ensure_positive_definite(P::AbstractMatrix{T}; ε::T=T(1e-10)) where T
    P_sym = Symmetric((P + P') / 2)
    λ_min = minimum(eigvals(P_sym))
    if λ_min < ε
        return P_sym + (ε - λ_min) * I
    end
    return P_sym
end

"""
    condition_number(P) -> Float64

Compute condition number of covariance matrix.
"""
function condition_number(P::AbstractMatrix)
    λ = eigvals(Symmetric(P))
    return maximum(λ) / max(minimum(λ), eps())
end
