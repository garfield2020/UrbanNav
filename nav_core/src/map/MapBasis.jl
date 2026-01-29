# ============================================================================
# Map Basis - Harmonic Basis Functions and Fitting (Phase A)
# ============================================================================
#
# Implements harmonic basis expansion for magnetic field representation.
#
# Physics Background:
# - In source-free regions, B = -∇Φ where Φ satisfies Laplace: ∇²Φ = 0
# - Solutions are harmonic functions (solid harmonics in 3D)
# - Linear model: Φ = Φ₀ + a·x + b·y + c·z (constant + linear gradient)
# - Field: B = -∇Φ = [-a, -b, -c] = B₀ (constant field)
# - Gradient: G = -∇²Φ = 0 for linear potential (but we fit G directly)
#
# Phase A Choice: Linear harmonic model
# - B(x) = B₀ + G₀·(x - x_ref)
# - 12 parameters: B₀ (3) + G₀ (9, but traceless symmetric → 5 independent)
# - Actually 8 DOF total: 3 field + 5 gradient
#
# Fitting Method: Weighted Least Squares
# - Fit to B and/or G samples with measurement uncertainties
# - Output: coefficients + coefficient covariance
# ============================================================================

using LinearAlgebra
using StaticArrays
using Statistics

# Import types from MapContract (assumes included in same module)
# using ..MapContract

# ============================================================================
# Type Aliases
# ============================================================================

const Vec3Basis = SVector{3, Float64}
const Mat3Basis = SMatrix{3, 3, Float64, 9}

# ============================================================================
# Linear Harmonic Model
# ============================================================================

"""
    LinearHarmonicModel

Linear harmonic background model: B(x) = B₀ + G₀·(x - x_ref)

# Fields
- `B0::Vec3Basis`: Field at reference point [T]
- `G0::Mat3Basis`: Gradient tensor [T/m] (symmetric, traceless)
- `x_ref::Vec3Basis`: Reference position [m]
- `Σ_coeffs::Matrix{Float64}`: 8×8 covariance of [B0; G5] coefficients

# Physics Notes
- G0 is symmetric: G[i,j] = G[j,i]
- G0 is traceless: tr(G0) = 0 (Maxwell: ∇·B = 0)
- 8 independent parameters: B₀ (3) + G₀ packed (5)
- Packed gradient: [Gxx, Gyy, Gxy, Gxz, Gyz], with Gzz = -(Gxx + Gyy)

# Validity Regime
- Valid within ~tile_size/2 of x_ref (linear approximation)
- Model error grows quadratically with distance from x_ref
- For Phase A: single global tile, valid over entire mission area if gradients
  are approximately constant
"""
struct LinearHarmonicModel
    B0::Vec3Basis           # Field at reference [T]
    G0::Mat3Basis           # Gradient tensor [T/m]
    x_ref::Vec3Basis        # Reference position [m]
    Σ_coeffs::Matrix{Float64}  # 8×8 coefficient covariance
end

"""Create model with diagonal covariance."""
function LinearHarmonicModel(;
    B0::AbstractVector,
    G0::AbstractMatrix,
    x_ref::AbstractVector,
    σ_B::Float64 = 0.0,    # Field coefficient uncertainty [T]
    σ_G::Float64 = 0.0     # Gradient coefficient uncertainty [T/m]
)
    Σ = diagm(vcat(fill(σ_B^2, 3), fill(σ_G^2, 5)))
    LinearHarmonicModel(
        Vec3Basis(B0...),
        Mat3Basis(G0...),
        Vec3Basis(x_ref...),
        Σ
    )
end

"""
    predict_field(model::LinearHarmonicModel, position::AbstractVector) -> Vec3Basis

Predict magnetic field at position using linear model.
"""
function predict_field(model::LinearHarmonicModel, position::AbstractVector)
    dx = Vec3Basis(position...) - model.x_ref
    return model.B0 + model.G0 * dx
end

"""
    predict_gradient(model::LinearHarmonicModel, position::AbstractVector) -> Mat3Basis

Return gradient tensor (constant for linear model).
"""
function predict_gradient(model::LinearHarmonicModel, position::AbstractVector)
    return model.G0
end

"""
    predict_with_uncertainty(model::LinearHarmonicModel, position::AbstractVector)

Predict field and gradient with propagated uncertainties.

Returns (B_pred, G_pred, Σ_B, Σ_G) where:
- B_pred: 3-vector field prediction [T]
- G_pred: 3×3 gradient tensor [T/m]
- Σ_B: 3×3 field prediction covariance [T²]
- Σ_G: 5×5 packed gradient covariance [T²/m²]
"""
function predict_with_uncertainty(model::LinearHarmonicModel, position::AbstractVector)
    dx = Vec3Basis(position...) - model.x_ref

    # Prediction
    B_pred = model.B0 + model.G0 * dx
    G_pred = model.G0

    # Jacobian of [B; G5] w.r.t. coefficients [B0; G5_0]
    # B = B0 + G0 * dx
    # ∂B/∂B0 = I(3)
    # ∂B/∂G5 = position-dependent (need to unpack G5 to G, multiply by dx)
    #
    # For simplicity, use first-order approximation:
    # Σ_B ≈ Σ_B0 + dx ⊗ Σ_G5 ⊗ dx (simplified)

    # Extract covariances
    Σ_B0 = model.Σ_coeffs[1:3, 1:3]
    Σ_G5 = model.Σ_coeffs[4:8, 4:8]

    # Field uncertainty: Σ_B = Σ_B0 + J_G * Σ_G5 * J_G'
    # where J_G is Jacobian of B w.r.t. G5 at position
    J_G = compute_field_gradient_jacobian(dx)
    Σ_B = Mat3Basis(Σ_B0 + J_G * Σ_G5 * J_G')

    # Gradient uncertainty is constant (independent of position for linear model)
    Σ_G = SMatrix{5,5}(Σ_G5)

    return B_pred, G_pred, Σ_B, Σ_G
end

"""
Compute Jacobian ∂B/∂G5 where G5 = [Gxx, Gyy, Gxy, Gxz, Gyz].

B = G * dx where G is unpacked from G5.
"""
function compute_field_gradient_jacobian(dx::AbstractVector)
    x, y, z = dx[1], dx[2], dx[3]

    # B[1] = Gxx*x + Gxy*y + Gxz*z
    # B[2] = Gxy*x + Gyy*y + Gyz*z
    # B[3] = Gxz*x + Gyz*y + Gzz*z = Gxz*x + Gyz*y - (Gxx+Gyy)*z

    J = @SMatrix [
        x    0.0   y    z    0.0;   # ∂B[1]/∂G5
        0.0  y     x    0.0  z  ;   # ∂B[2]/∂G5
        -z   -z    0.0  x    y      # ∂B[3]/∂G5 (using Gzz = -Gxx - Gyy)
    ]
    return J
end

# ============================================================================
# Model Packing/Unpacking (uses pack_gradient_tensor from tile_coefficients.jl)
# ============================================================================

"""Pack LinearHarmonicModel to 8-vector [B0; G5]."""
function pack_model(model::LinearHarmonicModel)
    G5 = pack_gradient_tensor(model.G0)
    return vcat(Vector(model.B0), Vector(G5))
end

"""Unpack 8-vector to LinearHarmonicModel."""
function unpack_model(coeffs::AbstractVector, x_ref::AbstractVector, Σ::AbstractMatrix)
    B0 = Vec3Basis(coeffs[1:3]...)
    G5 = coeffs[4:8]
    G0 = unpack_gradient_tensor(G5)
    return LinearHarmonicModel(B0, G0, Vec3Basis(x_ref...), Matrix{Float64}(Σ))
end

# ============================================================================
# Weighted Least Squares Fitting
# ============================================================================

"""
    MapFitSample

Single sample for map fitting.

# Fields
- `position::Vec3Basis`: Sample position [m] in world frame
- `B::Union{Nothing, Vec3Basis}`: Measured field [T] (nothing if not available)
- `G::Union{Nothing, Mat3Basis}`: Measured gradient [T/m] (nothing if not available)
- `R_B::Union{Nothing, Mat3Basis}`: Field measurement covariance [T²]
- `R_G::Union{Nothing, SMatrix{5,5}}`: Gradient measurement covariance [T²/m²]
"""
struct MapFitSample
    position::Vec3Basis
    B::Union{Nothing, Vec3Basis}
    G::Union{Nothing, Mat3Basis}
    R_B::Union{Nothing, Mat3Basis}
    R_G::Union{Nothing, SMatrix{5, 5, Float64, 25}}
end

"""Create sample with field only."""
function MapFitSample(position::AbstractVector, B::AbstractVector, R_B::AbstractMatrix)
    MapFitSample(
        Vec3Basis(position...),
        Vec3Basis(B...),
        nothing,
        Mat3Basis(R_B...),
        nothing
    )
end

"""Create sample with field and gradient."""
function MapFitSample(position::AbstractVector, B::AbstractVector, G::AbstractMatrix,
                      R_B::AbstractMatrix, R_G::AbstractMatrix)
    MapFitSample(
        Vec3Basis(position...),
        Vec3Basis(B...),
        Mat3Basis(G...),
        Mat3Basis(R_B...),
        SMatrix{5,5}(R_G...)
    )
end

"""
    MapFitResult

Result of weighted least squares map fitting.

# Fields
- `model::LinearHarmonicModel`: Fitted harmonic model
- `residuals::Vector{Float64}`: Fit residuals (stacked)
- `chi2::Float64`: χ² statistic (should be ≈ n_obs if well-calibrated)
- `dof::Int`: Degrees of freedom (n_observations - n_parameters)
- `nees::Float64`: Normalized estimation error squared (χ²/dof, should be ≈ 1)
"""
struct MapFitResult
    model::LinearHarmonicModel
    residuals::Vector{Float64}
    chi2::Float64
    dof::Int
    nees::Float64
end

"""
    fit_linear_harmonic_model(samples::Vector{MapFitSample}; x_ref=nothing) -> MapFitResult

Fit linear harmonic model B(x) = B₀ + G₀·(x - x_ref) to samples.

# Arguments
- `samples`: Vector of MapFitSample with field and/or gradient measurements
- `x_ref`: Reference position for model (default: centroid of sample positions)

# Returns
MapFitResult containing fitted model with coefficient covariances.

# Method
Weighted least squares minimizing:
    J = Σᵢ (zᵢ - h(xᵢ, θ))' Rᵢ⁻¹ (zᵢ - h(xᵢ, θ))

where θ = [B₀; G₅] are the 8 model parameters.

# Statistical Properties
- χ²/dof ≈ 1 indicates well-calibrated measurement noise
- Coefficient covariance enables uncertainty propagation in EKF
"""
function fit_linear_harmonic_model(samples::Vector{MapFitSample}; x_ref = nothing)
    n_samples = length(samples)
    @assert n_samples > 0 "Need at least one sample"

    # Compute reference position (centroid if not specified)
    if x_ref === nothing
        x_ref = mean([s.position for s in samples])
    else
        x_ref = Vec3Basis(x_ref...)
    end

    # Build design matrix and observation vector
    # Model: z = H * θ where θ = [B0; G5] (8 parameters)
    H_blocks = Vector{Matrix{Float64}}()
    z_blocks = Vector{Vector{Float64}}()
    R_blocks = Vector{Matrix{Float64}}()

    for sample in samples
        dx = sample.position - x_ref

        if sample.B !== nothing && sample.R_B !== nothing
            # Field measurement: B_meas = B0 + G0 * dx
            # H_B * [B0; G5] = B_meas
            H_B = zeros(3, 8)
            H_B[1:3, 1:3] = I(3)  # ∂B/∂B0 = I
            H_B[1:3, 4:8] = compute_field_gradient_jacobian(dx)  # ∂B/∂G5

            push!(H_blocks, H_B)
            push!(z_blocks, Vector(sample.B))
            push!(R_blocks, Matrix(sample.R_B))
        end

        if sample.G !== nothing && sample.R_G !== nothing
            # Gradient measurement: G5_meas = G5_0 (constant for linear model)
            # H_G * [B0; G5] = G5_meas
            H_G = zeros(5, 8)
            H_G[1:5, 4:8] = I(5)  # ∂G5/∂G5 = I

            G5_meas = pack_gradient_tensor(sample.G)
            push!(H_blocks, H_G)
            push!(z_blocks, Vector(G5_meas))
            push!(R_blocks, Matrix(sample.R_G))
        end
    end

    @assert length(H_blocks) > 0 "Need at least one valid measurement"

    # Stack into single system
    H = vcat(H_blocks...)
    z = vcat(z_blocks...)
    R = cat(R_blocks...; dims=(1,2))

    n_obs = length(z)
    n_params = 8
    dof = n_obs - n_params

    @assert dof > 0 "Need more observations than parameters (got $n_obs obs, $n_params params)"

    # Weighted least squares: θ = (H'R⁻¹H)⁻¹ H'R⁻¹z
    # Information matrix: I = H'R⁻¹H
    # Covariance: Σ = I⁻¹

    R_inv = inv(R)
    info_matrix = H' * R_inv * H
    Σ_coeffs = inv(info_matrix)
    θ = Σ_coeffs * H' * R_inv * z

    # Compute residuals and χ²
    z_pred = H * θ
    residuals = z - z_pred
    chi2 = residuals' * R_inv * residuals
    nees = chi2 / dof

    # Unpack model
    B0 = Vec3Basis(θ[1:3]...)
    G0 = unpack_gradient_tensor(θ[4:8])
    model = LinearHarmonicModel(B0, G0, x_ref, Σ_coeffs)

    return MapFitResult(model, residuals, chi2, dof, nees)
end

"""
    fit_from_field_samples(positions, fields, R_B; x_ref=nothing) -> MapFitResult

Convenience function to fit from arrays of field samples.

# Arguments
- `positions`: Vector of 3-vectors, sample positions [m]
- `fields`: Vector of 3-vectors, measured fields [T]
- `R_B`: Measurement covariance (scalar variance or 3×3 matrix) [T²]
"""
function fit_from_field_samples(positions::AbstractVector,
                                 fields::AbstractVector,
                                 R_B;
                                 x_ref = nothing)
    n = length(positions)
    @assert length(fields) == n "positions and fields must have same length"

    # Normalize R_B to 3×3 matrix
    if R_B isa Real
        R_B_mat = Mat3Basis(R_B * I)
    else
        R_B_mat = Mat3Basis(R_B...)
    end

    samples = [MapFitSample(positions[i], fields[i], R_B_mat) for i in 1:n]
    return fit_linear_harmonic_model(samples; x_ref=x_ref)
end

"""
    fit_from_field_and_gradient_samples(positions, fields, gradients, R_B, R_G; x_ref=nothing)

Convenience function to fit from arrays of field and gradient samples.

# Arguments
- `positions`: Vector of 3-vectors, sample positions [m]
- `fields`: Vector of 3-vectors, measured fields [T]
- `gradients`: Vector of 3×3 matrices, measured gradient tensors [T/m]
- `R_B`: Field measurement covariance [T²]
- `R_G`: Gradient measurement covariance (5×5) [T²/m²]
"""
function fit_from_field_and_gradient_samples(positions::AbstractVector,
                                              fields::AbstractVector,
                                              gradients::AbstractVector,
                                              R_B, R_G;
                                              x_ref = nothing)
    n = length(positions)
    @assert length(fields) == n "positions and fields must have same length"
    @assert length(gradients) == n "positions and gradients must have same length"

    # Normalize covariances
    if R_B isa Real
        R_B_mat = Mat3Basis(R_B * I)
    else
        R_B_mat = Mat3Basis(R_B...)
    end

    if R_G isa Real
        R_G_mat = SMatrix{5,5}(R_G * I)
    else
        R_G_mat = SMatrix{5,5}(R_G...)
    end

    samples = [MapFitSample(positions[i], fields[i], gradients[i], R_B_mat, R_G_mat)
               for i in 1:n]
    return fit_linear_harmonic_model(samples; x_ref=x_ref)
end

# ============================================================================
# Residual Analysis
# ============================================================================

"""
    compute_fit_residuals(model::LinearHarmonicModel, samples::Vector{MapFitSample})

Compute residuals for all samples against fitted model.

Returns vector of (position, B_residual, G_residual, NIS) tuples.
NIS = Normalized Innovation Squared (should be χ² distributed).
"""
function compute_fit_residuals(model::LinearHarmonicModel, samples::Vector{MapFitSample})
    results = NamedTuple[]

    for sample in samples
        B_pred = predict_field(model, sample.position)
        G_pred = predict_gradient(model, sample.position)

        B_res = sample.B !== nothing ? sample.B - B_pred : nothing
        G_res = sample.G !== nothing ? pack_gradient_tensor(sample.G - G_pred) : nothing

        # Compute NIS
        nis = 0.0
        dof = 0
        if B_res !== nothing && sample.R_B !== nothing
            nis += B_res' * inv(Matrix(sample.R_B)) * B_res
            dof += 3
        end
        if G_res !== nothing && sample.R_G !== nothing
            nis += G_res' * inv(Matrix(sample.R_G)) * G_res
            dof += 5
        end

        push!(results, (
            position = sample.position,
            B_residual = B_res,
            G_residual = G_res,
            nis = nis,
            dof = dof
        ))
    end

    return results
end

# ============================================================================
# Model Quality Metrics
# ============================================================================

"""
    compute_gradient_energy(model::LinearHarmonicModel) -> Float64

Compute gradient tensor Frobenius norm [T/m].

Low gradient energy indicates uninformative map for position updates.
Typical threshold: 5 nT/m minimum for position observability.
"""
function compute_gradient_energy(model::LinearHarmonicModel)
    return norm(model.G0)
end

"""
    compute_gradient_condition(model::LinearHarmonicModel) -> Float64

Compute condition number of gradient tensor.

High condition number indicates directional bias (strong in one direction,
weak in others). Threshold: condition > 10 suggests degraded observability.
"""
function compute_gradient_condition(model::LinearHarmonicModel)
    svd_result = svd(Matrix(model.G0))
    if minimum(svd_result.S) < 1e-15
        return Inf
    end
    return maximum(svd_result.S) / minimum(svd_result.S)
end

"""
    compute_directional_observability(model::LinearHarmonicModel, direction::AbstractVector)

Compute gradient magnitude in specified direction [T/m].

Useful for checking along-track vs cross-track observability.
"""
function compute_directional_observability(model::LinearHarmonicModel, direction::AbstractVector)
    d = Vec3Basis(direction...) / norm(direction)
    return norm(model.G0 * d)
end

# ============================================================================
# Exports
# ============================================================================

export LinearHarmonicModel
export predict_field, predict_gradient, predict_with_uncertainty
export pack_gradient_tensor, unpack_gradient_tensor
export pack_model, unpack_model
export MapFitSample, MapFitResult
export fit_linear_harmonic_model
export fit_from_field_samples, fit_from_field_and_gradient_samples
export compute_fit_residuals
export compute_gradient_energy, compute_gradient_condition
export compute_directional_observability
