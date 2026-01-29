# ============================================================================
# Centralized Whitening Module
# ============================================================================
#
# Ported from AUV-Navigation/src/whitening.jl
#
# SINGLE SOURCE OF TRUTH for:
# - Residual whitening: e = L⁻¹r
# - Chi-square computation: γ = r'Σ⁻¹r = ||e||²
# - Covariance validation: SPD checks
#
# Requirements:
# - REQ-STAT-004: Whitening consistency (three equivalent methods)
# - REQ-STAT-008: Dimension-agnostic whitening
# - REQ-STAT-010: Covariance positive definiteness
# ============================================================================

using LinearAlgebra

export WhiteningResult
export whiten, whiten_full
export compute_chi2_direct, compute_chi2_cholesky, compute_chi2_whitened_sum
export verify_chi2_consistency
export whiten_field_only, whiten_field_gradient, build_combined_covariance
export TotalCovarianceComponents, build_total_covariance, whiten_with_total_covariance

# ============================================================================
# Whitening Result
# ============================================================================

"""
    WhiteningResult

Result of whitening operation with all intermediate values for audit.
"""
struct WhiteningResult
    r::Vector{Float64}          # Original residual
    Σ::Matrix{Float64}          # Covariance matrix
    L::LowerTriangular{Float64, Matrix{Float64}}  # Cholesky factor
    e::Vector{Float64}          # Whitened residual (L⁻¹r)
    γ::Float64                  # Chi-square statistic
    d::Int                      # Dimension
    regularization_used::Float64  # Regularization added (0 if none needed)
end

# ============================================================================
# Core Whitening Function
# ============================================================================

"""
    whiten(r, Σ; check_shapes=true, check_spd=true)

Whiten a residual vector using covariance matrix.

Returns (e, γ) where:
- e = L⁻¹r is the whitened residual
- γ = r'Σ⁻¹r = ||e||² is the chi-square statistic

# Arguments
- `r`: Residual vector (d-dimensional)
- `Σ`: Covariance matrix (d×d, must be SPD)
- `check_shapes`: Enable strict shape checking
- `check_spd`: Check positive definiteness
"""
function whiten(r::AbstractVector, Σ::AbstractMatrix;
                check_shapes::Bool = true,
                check_spd::Bool = true)
    result = whiten_full(r, Σ; check_shapes=check_shapes, check_spd=check_spd)
    return result.e, result.γ
end

"""
    whiten_full(r, Σ; kwargs...)

Full whitening with all intermediate values for audit.
"""
function whiten_full(r::AbstractVector, Σ::AbstractMatrix;
                     check_shapes::Bool = true,
                     check_spd::Bool = true)
    r_vec = Vector{Float64}(r)
    Σ_mat = Matrix{Float64}(Σ)
    d = length(r_vec)

    if check_shapes
        _check_shapes(r_vec, Σ_mat)
    end

    L, reg_used = _safe_cholesky(Σ_mat, check_spd)
    e = L \ r_vec
    γ = dot(e, e)

    return WhiteningResult(r_vec, Σ_mat, L, e, γ, d, reg_used)
end

# ============================================================================
# Shape Checking
# ============================================================================

function _check_shapes(r::Vector, Σ::Matrix)
    d = length(r)
    n, m = size(Σ)

    if n != m
        throw(DimensionMismatch(
            "Covariance matrix must be square: got ($n, $m)"
        ))
    end

    if d != n
        throw(DimensionMismatch(
            "Residual dimension ($d) must match covariance dimension ($n)"
        ))
    end

    if !issymmetric(Σ)
        max_asym = maximum(abs.(Σ - Σ'))
        if max_asym > 1e-12 * norm(Σ)
            throw(ArgumentError(
                "Covariance matrix is not symmetric: max asymmetry = $max_asym"
            ))
        end
    end
end

# ============================================================================
# Safe Cholesky Decomposition
# ============================================================================

function _safe_cholesky(Σ::Matrix{Float64}, check_spd::Bool)
    d = size(Σ, 1)

    try
        C = cholesky(Hermitian(Σ))
        return C.L, 0.0
    catch e
        if isa(e, PosDefException)
            if check_spd
                λ_min = minimum(eigvals(Hermitian(Σ)))
                throw(ArgumentError(
                    "Covariance matrix is not positive definite: λ_min = $λ_min"
                ))
            end
        else
            rethrow(e)
        end
    end

    # Add minimal regularization
    trace_Σ = tr(Σ)
    ε = 1e-10 * max(trace_Σ / d, 1e-20)

    for _ in 1:10
        try
            Σ_reg = Σ + ε * I(d)
            C = cholesky(Hermitian(Σ_reg))
            return C.L, ε
        catch
            ε *= 10
        end
    end

    @warn "Cholesky failed even with regularization. Using eigendecomposition."
    F = eigen(Hermitian(Σ))
    λ_clamped = max.(F.values, 1e-20)
    Σ_approx = F.vectors * Diagonal(λ_clamped) * F.vectors'
    C = cholesky(Hermitian(Σ_approx))
    return C.L, NaN
end

# ============================================================================
# Chi-Square Computation (Multiple Methods)
# ============================================================================

"""
    compute_chi2_direct(r, Σ)

Compute χ² via direct matrix solve: γ = r'Σ⁻¹r
"""
function compute_chi2_direct(r::AbstractVector, Σ::AbstractMatrix)
    return dot(r, Σ \ r)
end

"""
    compute_chi2_cholesky(r, Σ)

Compute χ² via Cholesky: γ = ||L⁻¹r||²
"""
function compute_chi2_cholesky(r::AbstractVector, Σ::AbstractMatrix)
    C = cholesky(Hermitian(Matrix(Σ)))
    e = C.L \ Vector(r)
    return dot(e, e)
end

"""
    compute_chi2_whitened_sum(r, Σ)

Compute χ² via sum of squared whitened components: γ = Σᵢ eᵢ²
"""
function compute_chi2_whitened_sum(r::AbstractVector, Σ::AbstractMatrix)
    C = cholesky(Hermitian(Matrix(Σ)))
    e = C.L \ Vector(r)
    return sum(e .^ 2)
end

"""
    verify_chi2_consistency(r, Σ; rtol=1e-10)

Verify that all three χ² computation methods give consistent results.
"""
function verify_chi2_consistency(r::AbstractVector, Σ::AbstractMatrix; rtol::Float64 = 1e-10)
    γ_direct = compute_chi2_direct(r, Σ)
    γ_cholesky = compute_chi2_cholesky(r, Σ)
    γ_sum = compute_chi2_whitened_sum(r, Σ)

    γ_max = max(γ_direct, γ_cholesky, γ_sum)

    err_dc = abs(γ_direct - γ_cholesky) / max(γ_max, 1e-20)
    err_ds = abs(γ_direct - γ_sum) / max(γ_max, 1e-20)
    err_cs = abs(γ_cholesky - γ_sum) / max(γ_max, 1e-20)

    max_err = max(err_dc, err_ds, err_cs)
    consistent = max_err < rtol

    return (
        consistent = consistent,
        max_relative_error = max_err,
        γ_values = (direct = γ_direct, cholesky = γ_cholesky, sum = γ_sum)
    )
end

# ============================================================================
# Dimension-Specific Whitening
# ============================================================================

"""
    whiten_field_only(r_B, Σ_B)

Whiten B-only residual (d=3).
"""
function whiten_field_only(r_B::AbstractVector, Σ_B::AbstractMatrix)
    if length(r_B) != 3
        throw(DimensionMismatch("B-only residual must be 3-dimensional, got $(length(r_B))"))
    end
    return whiten(r_B, Σ_B)
end

"""
    whiten_field_gradient(r_BG, Σ_BG)

Whiten B+∇B residual (d=8).
"""
function whiten_field_gradient(r_BG::AbstractVector, Σ_BG::AbstractMatrix)
    if length(r_BG) != 8
        throw(DimensionMismatch("B+∇B residual must be 8-dimensional, got $(length(r_BG))"))
    end
    return whiten(r_BG, Σ_BG)
end

"""
    build_combined_covariance(σ_B, σ_G)

Build combined covariance matrix for B+∇B measurements.
"""
function build_combined_covariance(σ_B::Float64, σ_G::Float64)
    Σ = zeros(8, 8)
    for i in 1:3
        Σ[i, i] = σ_B^2
    end
    for i in 4:8
        Σ[i, i] = σ_G^2
    end
    return Σ
end

# ============================================================================
# Total Covariance Components
# ============================================================================

"""
    TotalCovarianceComponents

Components of the total innovation covariance.

CRITICAL FORMULA:
    Σ_total = Σ_meas + Σ_pose + Σ_map + Q_model
           = R + H P Hᵀ + Jα Pα Jαᵀ + Q_model

All four terms are required for correct NEES/NIS calibration!

- Σ_meas (R): Sensor measurement noise
- Σ_pose (HPHᵀ): State uncertainty propagated to measurement space
- Σ_map: Map/tile coefficient uncertainty
- Q_model: Truth-model mismatch (unmodeled physics)
"""
struct TotalCovarianceComponents
    Σ_meas::Matrix{Float64}   # Measurement noise (R)
    Σ_pose::Matrix{Float64}   # Position uncertainty contribution (HPHᵀ)
    Σ_map::Matrix{Float64}    # Map/tile uncertainty contribution
    Q_model::Matrix{Float64}  # Model mismatch uncertainty
    Σ_total::Matrix{Float64}  # Sum of all components
end

"""
    build_total_covariance(Σ_meas, ∇B, P_pos, J_α, P_α; Q_model=nothing)

Build total innovation covariance from all sources.

CRITICAL FORMULA:
    Σ_total = Σ_meas + (∇B) P_pos (∇B)' + J_α P_α J_α' + Q_model
           = R + H P Hᵀ + Σ_map + Q_model

All terms must be included for correct NEES calibration!
"""
function build_total_covariance(
    Σ_meas::AbstractMatrix,
    ∇B::AbstractMatrix,
    P_pos::AbstractMatrix,
    J_α::AbstractMatrix,
    P_α::AbstractMatrix;
    Q_model::Union{AbstractMatrix, Nothing} = nothing
)
    d = size(Σ_meas, 1)
    Σ_meas_mat = Matrix{Float64}(Σ_meas)

    # Term 2: HPHᵀ (position uncertainty in measurement space)
    Σ_pose = if size(∇B, 1) == d && size(∇B, 2) == 3
        ∇B * P_pos * ∇B'
    else
        zeros(d, d)
    end

    # Term 3: Map uncertainty
    Σ_map = if size(J_α, 1) == d && size(P_α, 1) == size(J_α, 2)
        J_α * P_α * J_α'
    else
        zeros(d, d)
    end

    # Term 4: Model mismatch
    Q_model_mat = if Q_model === nothing
        zeros(d, d)
    else
        Matrix{Float64}(Q_model)
    end

    # Sum all terms
    Σ_total = Σ_meas_mat + Σ_pose + Σ_map + Q_model_mat
    Σ_total = (Σ_total + Σ_total') / 2

    return TotalCovarianceComponents(
        Σ_meas_mat,
        Matrix{Float64}(Σ_pose),
        Matrix{Float64}(Σ_map),
        Q_model_mat,
        Σ_total
    )
end

"""
    whiten_with_total_covariance(r, Σ_meas, ∇B, P_pos, J_α, P_α; Q_model=nothing)

Whiten residual using full innovation covariance.

Uses the complete formula:
    Σ_total = Σ_meas + ∇B P_pos ∇B' + J_α P_α J_α' + Q_model
"""
function whiten_with_total_covariance(
    r::AbstractVector,
    Σ_meas::AbstractMatrix,
    ∇B::AbstractMatrix,
    P_pos::AbstractMatrix,
    J_α::AbstractMatrix,
    P_α::AbstractMatrix;
    Q_model::Union{AbstractMatrix, Nothing} = nothing
)
    cov_components = build_total_covariance(Σ_meas, ∇B, P_pos, J_α, P_α; Q_model=Q_model)
    result = whiten_full(r, cov_components.Σ_total)
    return result, cov_components
end

"""
    create_Q_model(d::Int; σ_B::Float64=1e-9, σ_G::Float64=1e-9) -> Matrix

Create model mismatch covariance for d-dimensional measurement.

# What Q_model Represents

Q_model accounts for "truth-model mismatch" - uncertainty that arises because
our model of the magnetic field is imperfect:

1. **Unmodeled sources**: Dipoles, geological features not in the map
2. **Temporal variations**: Tidal effects, diurnal variations, space weather
3. **Discretization errors**: Map resolution, interpolation artifacts
4. **Calibration drift**: Sensor biases that change over time

# Impact on NEES Calibration

Without Q_model, the filter becomes overconfident:
- HPHᵀ captures uncertainty in state estimate
- R captures sensor noise
- But neither captures uncertainty in the world model itself

This causes χ² values to be systematically too high, failing NEES tests.

# Tuning Q_model

- Start with σ_B ≈ σ_G ≈ sensor noise floor (1-5 nT)
- Increase if NEES consistently > 1.0 (overconfident)
- Decrease if NEES consistently < 1.0 (underconfident)
- Can be learned from residual statistics
"""
function create_Q_model(d::Int; σ_B::Float64 = 1e-9, σ_G::Float64 = 1e-9)
    if d == 3
        return Diagonal(fill(σ_B^2, 3))
    elseif d == 8
        variances = vcat(
            fill(σ_B^2, 3),
            fill(σ_G^2, 5)
        )
        return Diagonal(variances)
    else
        error("Unsupported dimension d=$d for Q_model")
    end
end

export create_Q_model
