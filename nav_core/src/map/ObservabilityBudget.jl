# ============================================================================
# ObservabilityBudget.jl - Quadratic-Block Fisher Information for Tier-2 Gating
# ============================================================================
#
# Purpose: Compute the data-only Fisher Information Matrix (FIM) for the
# quadratic coefficient block (indices 9:15) to assess whether a tile has
# sufficient trajectory excitation to support d=15 (Tier-2) estimation.
#
# Physics:
# The measurement model is B(x) = H(x̃) · α, where H is the field Jacobian
# w.r.t. tile coefficients α in normalized coordinates x̃ = (x - center)/L.
# The data FIM for the quadratic block is:
#
#   ΔI_qq = Σ_k H_q(x̃_k)' R_k⁻¹ H_q(x̃_k)
#
# where H_q = H[:, 9:15] is the Jacobian sub-block for quadratic coefficients.
# This is a 7×7 PSD matrix whose eigenstructure reveals which quadratic
# modes are identifiable from the trajectory.
#
# Observability scalar: log det(ΔI_qq) measures total quadratic information
# in nats. This is invariant to coefficient rescaling and provides a
# single number for unlock gating.
#
# Normalization invariance: Because all computations use normalized
# coordinates x̃ = (x - center)/L, the FIM is independent of tile scale L
# and tile center translation (up to numerical precision).
#
# References:
#   - Bar-Shalom et al., "Estimation with Applications to Tracking and
#     Navigation", Ch. 2 (Fisher Information)
#   - Canciani & Raquet, "Absolute Position Determination Using Magnetic
#     Field Measurements", Navigation 2016
#
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Types
# ============================================================================

"""
    ObservabilityBudget

Summary of quadratic-block observability for a tile.

# Fields
- `I_qq::Matrix{Float64}`: 7×7 data-only FIM for quadratic coefficients (PSD)
- `scalar::Float64`: Observability scalar (log det I_qq) in nats. -Inf if singular.
- `rank::Int`: Numerical rank of I_qq (eigenvalues > 1e-10 × λ_max)
- `relative_cond::Float64`: σ_min / σ_max of I_qq. 0.0 if rank-deficient.

# Physics
I_qq encodes how well the trajectory excites each of the 7 Maxwell-consistent
quadratic harmonic modes. Full rank (7) with good conditioning indicates the
trajectory provides sufficient spatial diversity for Tier-2 learning.
"""
struct ObservabilityBudget
    I_qq::Matrix{Float64}
    scalar::Float64
    rank::Int
    relative_cond::Float64
end

# ============================================================================
# Core Computation
# ============================================================================

"""
    compute_quadratic_fim(positions, R_list, tile) -> Matrix{Float64}

Compute the 7×7 data-only Fisher Information Matrix for the quadratic
coefficient block (indices 9:15).

# Arguments
- `positions::Vector{Vec3Map}`: Observation positions in world frame [m]
- `R_list::Vector{<:AbstractMatrix}`: Per-observation 3×3 measurement noise
  covariance matrices [T²]. Each R_k represents sensor noise only.
- `tile::SlamTileState`: Tile providing center, scale, and model mode

# Returns
7×7 PSD matrix ΔI_qq = Σ_k H_q(x̃_k)' R_k⁻¹ H_q(x̃_k)

# Physics
Uses normalized coordinates x̃ = (x - center) / L so that all Jacobian
entries are O(1) for in-tile positions. The Jacobian is evaluated at
MODE_QUADRATIC regardless of tile's current mode, since we are assessing
*potential* quadratic observability.

# Complexity
O(N × 7² × 3) where N = number of observations. The 3×15 Jacobian is
computed per observation but only the 3×7 quadratic sub-block is used.
"""
function compute_quadratic_fim(positions::Vector{Vec3Map},
                                R_list::Vector{<:AbstractMatrix},
                                tile::SlamTileState)
    @assert length(positions) == length(R_list) "positions and R_list must have same length"

    I_qq = zeros(7, 7)
    frame = TileLocalFrame(tile.center, tile.scale)

    for k in eachindex(positions)
        x̃ = normalize_position(frame, positions[k])

        # Full 3×15 Jacobian at MODE_QUADRATIC (assess potential, not current mode)
        H_full = field_jacobian(x̃, MODE_QUADRATIC)
        H_q = H_full[:, 9:15]  # 3×7 quadratic sub-block

        R_inv = inv(Matrix{Float64}(R_list[k]))

        # Accumulate: I_qq += H_q' R⁻¹ H_q
        I_qq += H_q' * R_inv * H_q
    end

    return I_qq
end

"""
    observability_scalar(I_qq::AbstractMatrix; metric::Symbol=:logdet) -> Float64

Compute a scalar observability metric from the quadratic FIM.

# Arguments
- `I_qq`: 7×7 data-only FIM for quadratic block
- `metric`: Scalar reduction method
  - `:logdet` (default): log det(I_qq) in nats. Measures total information.
    Invariant to orthogonal rotations of the coefficient basis.
  - `:trace`: tr(I_qq). Simpler but not rotation-invariant.
  - `:min_eigenvalue`: λ_min(I_qq). Most conservative; dominated by weakest mode.

# Returns
Scalar observability value. For `:logdet`, returns -Inf if I_qq is singular.

# Physics justification
log det is the natural metric for Gaussian information: it equals the
differential entropy reduction when the quadratic parameters are estimated.
For a multivariate Gaussian, det(I) = 1/det(P), so log det(I) measures
how much the posterior concentrates relative to a flat prior.
"""
function observability_scalar(I_qq::AbstractMatrix; metric::Symbol=:logdet)
    if metric == :logdet
        # Use eigenvalues directly to avoid logdet DomainError on near-singular matrices.
        # Eigenvalues below ε are treated as zero (rank-deficient).
        eigs = eigvals(Symmetric(Matrix{Float64}(I_qq)))
        ε = 1e-30  # Well below any physical information contribution
        pos_eigs = filter(λ -> λ > ε, eigs)
        if isempty(pos_eigs)
            return -Inf
        end
        return sum(log.(pos_eigs))
    elseif metric == :trace
        return tr(I_qq)
    elseif metric == :min_eigenvalue
        return minimum(eigvals(Symmetric(Matrix{Float64}(I_qq))))
    else
        throw(ArgumentError("Unknown metric: $metric. Use :logdet, :trace, or :min_eigenvalue"))
    end
end

"""
    ObservabilityBudget(positions, R_list, tile) -> ObservabilityBudget

Convenience constructor: compute FIM, scalar, rank, and conditioning in one call.

# Arguments
Same as `compute_quadratic_fim`.

# Rank determination
Uses relative eigenvalue threshold of 1e-10 (same as ObservabilityMetrics.jl).
An eigenvalue λ is considered nonzero if λ > 1e-10 × λ_max. This threshold
is well above Float64 noise (~1e-16) and well below any physically meaningful
information contribution.
"""
function ObservabilityBudget(positions::Vector{Vec3Map},
                              R_list::Vector{<:AbstractMatrix},
                              tile::SlamTileState)
    I_qq = compute_quadratic_fim(positions, R_list, tile)
    scalar = observability_scalar(I_qq; metric=:logdet)

    eigs = eigvals(Symmetric(I_qq))
    λ_max = maximum(eigs)

    # Rank: count eigenvalues > 1e-10 × λ_max (relative threshold)
    threshold = 1e-10 * max(λ_max, 1e-30)
    rank = count(λ -> λ > threshold, eigs)

    # Relative conditioning: σ_min / σ_max
    svs = svdvals(I_qq)
    if length(svs) >= 2 && svs[1] > 0.0
        relative_cond = svs[end] / svs[1]
    else
        relative_cond = 0.0
    end

    return ObservabilityBudget(I_qq, scalar, rank, relative_cond)
end

# ============================================================================
# Exports
# ============================================================================

export compute_quadratic_fim, observability_scalar, ObservabilityBudget
