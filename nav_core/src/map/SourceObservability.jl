# ============================================================================
# SourceObservability.jl - Source FIM and Observability Gating (Phase G Step 4)
# ============================================================================
#
# Computes the Fisher Information Matrix for source parameters (6-DOF:
# position + moment) and provides observability gating for source promotion.
#
# Mirrors the pattern from ObservabilityBudget.jl (quadratic tile FIM)
# adapted for dipole source parameters.
#
# Physics: The dipole Jacobian ∂B/∂[pos,moment] maps source parameter
# uncertainty to field-space. The FIM I = Σ H' R⁻¹ H quantifies how
# well the trajectory constrains each parameter.
#
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Types
# ============================================================================

"""
    SourceObservabilityBudget

Summary of source parameter observability.

# Fields
- `I_source::Matrix{Float64}`: 6×6 data-only FIM [position + moment]
- `scalar::Float64`: log det(I_source) [nats]. -Inf if singular.
- `rank::Int`: Numerical rank of I_source
- `relative_cond::Float64`: σ_min / σ_max of I_source
- `position_crlb::SVector{3,Float64}`: sqrt(diag(I⁻¹)[1:3]) [m]
- `moment_crlb::SVector{3,Float64}`: sqrt(diag(I⁻¹)[4:6]) [A·m²]

# Physics
I_source encodes how well observation positions constrain the 6 dipole
parameters (3 position + 3 moment). Full rank (6) with good conditioning
requires spatially diverse observations around the source.
"""
struct SourceObservabilityBudget
    I_source::Matrix{Float64}
    scalar::Float64
    rank::Int
    relative_cond::Float64
    position_crlb::SVector{3, Float64}
    moment_crlb::SVector{3, Float64}
end

# ============================================================================
# FIM Computation
# ============================================================================

"""
    compute_source_fim(positions::Vector{Vec3Map},
                       R_list::Vector{<:AbstractMatrix},
                       source::SlamSourceState) -> Matrix{Float64}

Compute the 6×6 data-only Fisher Information Matrix for a dipole source.

# Arguments
- `positions`: Observation positions in world frame [m]
- `R_list`: Per-observation 3×3 measurement noise covariance [T²]
- `source`: Source state (position + moment)

# Returns
6×6 PSD matrix I = Σ_k H_k' R_k⁻¹ H_k

# Physics
Uses the dipole Jacobian H = ∂B/∂[pos, moment] from OnlineSourceSLAM.jl.
The position Jacobian scales as 1/r⁴ and the moment Jacobian as 1/r³,
so closer observations contribute disproportionately more information.
"""
function compute_source_fim(positions::Vector{Vec3Map},
                             R_list::Vector{<:AbstractMatrix},
                             source::SlamSourceState)
    @assert length(positions) == length(R_list) "positions and R_list must have same length"

    I_source = zeros(6, 6)

    for k in eachindex(positions)
        H = compute_source_jacobian(positions[k], source)  # 3×6
        R_inv = inv(Matrix{Float64}(R_list[k]))
        I_source += H' * R_inv * H
    end

    return I_source
end

"""
    source_observability_scalar(I_source::AbstractMatrix; metric::Symbol=:logdet) -> Float64

Compute scalar observability metric from source FIM.

Same interface as observability_scalar() in ObservabilityBudget.jl.
"""
function source_observability_scalar(I_source::AbstractMatrix; metric::Symbol=:logdet)
    if metric == :logdet
        eigs = eigvals(Symmetric(Matrix{Float64}(I_source)))
        ε = 1e-30
        pos_eigs = filter(λ -> λ > ε, eigs)
        return isempty(pos_eigs) ? -Inf : sum(log.(pos_eigs))
    elseif metric == :trace
        return tr(I_source)
    elseif metric == :min_eigenvalue
        return minimum(eigvals(Symmetric(Matrix{Float64}(I_source))))
    else
        throw(ArgumentError("Unknown metric: $metric"))
    end
end

# ============================================================================
# Budget Constructor
# ============================================================================

"""
    SourceObservabilityBudget(positions, R_list, source) -> SourceObservabilityBudget

Convenience constructor: compute FIM, scalar, rank, conditioning, and CRLBs.
"""
function SourceObservabilityBudget(positions::Vector{Vec3Map},
                                    R_list::Vector{<:AbstractMatrix},
                                    source::SlamSourceState)
    I_source = compute_source_fim(positions, R_list, source)
    scalar = source_observability_scalar(I_source; metric=:logdet)

    # Rank
    eigs = eigvals(Symmetric(I_source))
    λ_max = maximum(eigs)
    threshold = 1e-10 * max(λ_max, 1e-30)
    rank = count(λ -> λ > threshold, eigs)

    # Conditioning
    svs = svdvals(I_source)
    relative_cond = (length(svs) >= 2 && svs[1] > 0.0) ? svs[end] / svs[1] : 0.0

    # CRLBs from inverse FIM (if full rank)
    if rank == 6
        I_inv = inv(I_source + 1e-15 * I(6))  # Regularize
        position_crlb = SVector{3}(sqrt.(max.(diag(I_inv)[1:3], 0.0)))
        moment_crlb = SVector{3}(sqrt.(max.(diag(I_inv)[4:6], 0.0)))
    else
        position_crlb = SVector{3}(Inf, Inf, Inf)
        moment_crlb = SVector{3}(Inf, Inf, Inf)
    end

    return SourceObservabilityBudget(I_source, scalar, rank, relative_cond,
                                     position_crlb, moment_crlb)
end

# ============================================================================
# Gating
# ============================================================================

"""
    meets_observability_gate(budget::SourceObservabilityBudget;
                              min_rank::Int=6,
                              max_position_crlb_m::Float64=1.0) -> Bool

Check if source observability meets promotion requirements.

# Requirements
- FIM rank >= min_rank (default: 6 = full)
- Maximum position CRLB component <= threshold

# Returns
true if source is sufficiently observed for promotion.
"""
function meets_observability_gate(budget::SourceObservabilityBudget;
                                   min_rank::Int = 6,
                                   max_position_crlb_m::Float64 = 1.0)
    return budget.rank >= min_rank &&
           maximum(budget.position_crlb) <= max_position_crlb_m
end

# ============================================================================
# Phase G+ Step 3: Enhanced Source Observability
# ============================================================================

"""
    spatial_excitation_score(positions::Vector{Vec3Map}) → Float64

Measure angular diversity of observation geometry.
Returns a score in [0, 1] where:
- 0 = all observations are collinear (rank-deficient geometry)
- 1 = observations span full angular diversity (ideal Lissajous)

The score is computed as the ratio of the smallest to largest singular value
of the centered position matrix. This quantifies how well the observation
geometry fills 3D space around the source.

Physics: Dipole position observability requires observations from multiple
angular directions. Collinear passes provide moment information but poor
position constraints orthogonal to the line of approach.
"""
function spatial_excitation_score(positions::Vector{Vec3Map})
    n = length(positions)
    if n < 3
        return 0.0
    end

    # Center positions
    centroid = sum(positions) / n
    P = zeros(n, 3)
    for i in 1:n
        P[i, :] = Vector(positions[i] - centroid)
    end

    svs = svdvals(P)
    if svs[1] < 1e-10
        return 0.0
    end

    # Ratio of smallest to largest singular value of position spread
    # For 3D spread: use min(sv[2], sv[3]) / sv[1] since sv[3] may be small for 2D trajectories
    # Use geometric mean of ratios for balanced measure
    ratio_2 = svs[2] / svs[1]
    ratio_3 = length(svs) >= 3 ? svs[3] / svs[1] : 0.0

    return sqrt(ratio_2 * max(ratio_3, 0.0))
end

"""
    meets_excitation_gate(positions::Vector{Vec3Map}, source_pos::Vec3Map;
                          min_angular_span_rad::Float64=π/3) → Bool

Check if observations span sufficient angular diversity relative to the source.

The angular span is the maximum angle subtended between any two observation
directions as seen from the source. π/3 (60°) is the minimum for reasonable
position observability — below this, the position-moment ambiguity is severe.

Physics: For a dipole at the origin, the field pattern is symmetric about the
moment axis. Observations from a narrow angular range cannot distinguish
position changes along the line of sight from moment magnitude changes.
"""
function meets_excitation_gate(positions::Vector{Vec3Map}, source_pos::Vec3Map;
                               min_angular_span_rad::Float64 = π/3)
    n = length(positions)
    if n < 2
        return false
    end

    # Compute directions from source to each observation
    directions = Vec3Map[]
    for pos in positions
        d = pos - source_pos
        d_norm = norm(d)
        if d_norm > 0.1  # Skip if too close (< 10cm)
            push!(directions, d / d_norm)
        end
    end

    if length(directions) < 2
        return false
    end

    # Find maximum angular span
    max_angle = 0.0
    for i in 1:length(directions)
        for j in (i+1):length(directions)
            cos_angle = clamp(dot(directions[i], directions[j]), -1.0, 1.0)
            angle = acos(cos_angle)
            max_angle = max(max_angle, angle)
        end
    end

    return max_angle >= min_angular_span_rad
end

"""
    incremental_fim_update!(I_source::Matrix{Float64}, new_position::Vec3Map,
                            new_R::AbstractMatrix, source::SlamSourceState) → Matrix{Float64}

Online FIM accumulation: add one observation's information to existing FIM.
Avoids full recompute over all observations.

I_new = I_old + H' R⁻¹ H

This is exact because the FIM is a sum of per-observation contributions.
"""
function incremental_fim_update!(I_source::Matrix{Float64}, new_position::Vec3Map,
                                 new_R::AbstractMatrix, source::SlamSourceState)
    H = compute_source_jacobian(new_position, source)  # 3×6
    R_inv = inv(Matrix{Float64}(new_R))
    I_source .+= H' * R_inv * H
    return I_source
end

# ============================================================================
# Exports
# ============================================================================

export SourceObservabilityBudget
export compute_source_fim, source_observability_scalar
export meets_observability_gate
export spatial_excitation_score, meets_excitation_gate, incremental_fim_update!
