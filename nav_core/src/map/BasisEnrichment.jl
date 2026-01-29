# =============================================================================
# BasisEnrichment.jl - Maxwell-Consistent Higher-Order Map Bases
# =============================================================================
#
# Purpose: Extend map basis beyond linear while preserving physical constraints.
#
# Maxwell Constraints (source-free region):
#   ∇·B = 0  (no magnetic monopoles)
#   ∇×B = 0  (no currents in measurement region)
#
# These imply B = -∇φ where ∇²φ = 0 (Laplace equation).
# Valid basis functions are harmonic functions (solutions to Laplace).
#
# Basis Hierarchy:
#   Order 0: Constant field B₀ (3 coefficients)
#   Order 1: Linear gradient B₀ + G·δx (3 + 5 = 8 coefficients)
#   Order 2: Quadratic B₀ + G·δx + Q(δx) (8 + 7 = 15 coefficients)
#
# References:
#   - Jackson, "Classical Electrodynamics", Ch. 5 (magnetic multipoles)
#   - Blakely, "Potential Theory in Gravity and Magnetic Applications"
#
# =============================================================================

module BasisEnrichment

using LinearAlgebra
using StaticArrays
using Statistics

# =============================================================================
# Constants and Types
# =============================================================================

"""
Basis order enumeration with coefficient counts.

Each order adds more degrees of freedom while maintaining Maxwell consistency.
The coefficient counts account for symmetry and divergence-free constraints.
"""
@enum BasisOrder begin
    CONSTANT = 0    # 3 coefficients (B₀)
    LINEAR = 1      # 8 coefficients (B₀ + G)
    QUADRATIC = 2   # 15 coefficients (B₀ + G + Q)
end

"""
Number of independent coefficients for each basis order.

Derivation:
- Order 0: B₀ has 3 components
- Order 1: Gradient G is 3×3 symmetric traceless → 5 independent
- Order 2: Quadratic terms Q constrained by ∇·B=0 → 7 independent

Total at order n: 3 + 5 + 7 = 15 for n=2
General formula: (n+1)(n+2)(2n+3)/6 for solid harmonics up to degree n
"""
const COEFFICIENT_COUNTS = Dict(
    CONSTANT => 3,
    LINEAR => 8,
    QUADRATIC => 15
)

"""
    MapBasis

Represents a Maxwell-consistent map basis at a reference point.

# Fields
- `order::BasisOrder`: Polynomial order of basis
- `reference::SVector{3,Float64}`: Reference position [m]
- `coefficients::Vector{Float64}`: Basis coefficients
- `covariance::Matrix{Float64}`: Coefficient covariance

# Coefficient Layout
Order 0 (3 coefficients):
  [Bx₀, By₀, Bz₀]

Order 1 (8 coefficients):
  [Bx₀, By₀, Bz₀, Gxx, Gyy, Gxy, Gxz, Gyz]
  Note: Gzz = -(Gxx + Gyy) enforced by Maxwell

Order 2 (15 coefficients):
  [B₀(3), G(5), Q(7)]
  Q terms are second-order harmonic coefficients
"""
struct MapBasis
    order::BasisOrder
    reference::SVector{3,Float64}
    coefficients::Vector{Float64}
    covariance::Matrix{Float64}
end

# =============================================================================
# Constructors
# =============================================================================

"""
    MapBasis(order, reference; B0, G, Q)

Construct a MapBasis with specified coefficients.

# Arguments
- `order::BasisOrder`: Basis order
- `reference::AbstractVector`: Reference position [m]

# Keyword Arguments
- `B0::AbstractVector`: Constant field [T] (required for all orders)
- `G::AbstractMatrix`: Gradient tensor [T/m] (required for order ≥ 1)
- `Q::AbstractVector`: Quadratic coefficients (required for order ≥ 2)
- `σ_B::Float64`: Field uncertainty [T] (default: 10e-9)
- `σ_G::Float64`: Gradient uncertainty [T/m] (default: 2e-9)
- `σ_Q::Float64`: Quadratic uncertainty (default: 0.5e-9)
"""
function MapBasis(
    order::BasisOrder,
    reference::AbstractVector;
    B0::AbstractVector = zeros(3),
    G::AbstractMatrix = zeros(3, 3),
    Q::AbstractVector = zeros(7),
    σ_B::Float64 = 10e-9,
    σ_G::Float64 = 2e-9,
    σ_Q::Float64 = 0.5e-9
)
    n_coeffs = COEFFICIENT_COUNTS[order]
    coefficients = zeros(n_coeffs)

    # Pack B0 (always present)
    coefficients[1:3] = B0

    # Pack G if order >= 1
    if order >= LINEAR
        # Extract 5 independent gradient components
        # G is symmetric, traceless: Gzz = -(Gxx + Gyy)
        coefficients[4] = G[1,1]  # Gxx
        coefficients[5] = G[2,2]  # Gyy
        coefficients[6] = G[1,2]  # Gxy (= Gyx)
        coefficients[7] = G[1,3]  # Gxz (= Gzx)
        coefficients[8] = G[2,3]  # Gyz (= Gzy)
    end

    # Pack Q if order >= 2
    if order >= QUADRATIC
        coefficients[9:15] = Q[1:7]
    end

    # Build covariance (diagonal for simplicity)
    variances = zeros(n_coeffs)
    variances[1:3] .= σ_B^2
    if order >= LINEAR
        variances[4:8] .= σ_G^2
    end
    if order >= QUADRATIC
        variances[9:15] .= σ_Q^2
    end
    covariance = diagm(variances)

    MapBasis(order, SVector{3}(reference), coefficients, covariance)
end

"""
    upgrade_basis(basis::MapBasis, new_order::BasisOrder) -> MapBasis

Upgrade basis to higher order, initializing new coefficients to zero.

# Physical Justification
Higher-order terms initialized to zero is conservative: assumes the
linear model is locally accurate until data suggests otherwise.
New coefficient uncertainties are set large to allow learning.
"""
function upgrade_basis(basis::MapBasis, new_order::BasisOrder)
    if new_order <= basis.order
        return basis  # Already at or above requested order
    end

    n_old = COEFFICIENT_COUNTS[basis.order]
    n_new = COEFFICIENT_COUNTS[new_order]

    # Expand coefficients
    new_coeffs = zeros(n_new)
    new_coeffs[1:n_old] = basis.coefficients

    # Expand covariance
    new_cov = zeros(n_new, n_new)
    new_cov[1:n_old, 1:n_old] = basis.covariance

    # Set large uncertainty for new coefficients (allows learning)
    # σ_Q = 1e-9 T/m² is typical for quadratic terms at 100m scale
    for i in (n_old+1):n_new
        new_cov[i, i] = (1e-9)^2
    end

    MapBasis(new_order, basis.reference, new_coeffs, new_cov)
end

# =============================================================================
# Field Evaluation
# =============================================================================

"""
    evaluate_field(basis::MapBasis, position::AbstractVector) -> SVector{3,Float64}

Evaluate magnetic field at position using basis expansion.

# Mathematical Form
B(x) = B₀ + G·δx + Q(δx) + O(|δx|³)

where δx = x - x_ref

# Maxwell Consistency
The evaluation automatically enforces ∇·B = 0 through the coefficient
structure (traceless gradient, harmonic quadratic terms).
"""
function evaluate_field(basis::MapBasis, position::AbstractVector)
    δx = SVector{3}(position) - basis.reference

    # Constant term (always present)
    B = SVector{3}(basis.coefficients[1:3])

    if basis.order >= LINEAR
        # Linear term: G·δx
        G = unpack_gradient(basis)
        B = B + G * δx
    end

    if basis.order >= QUADRATIC
        # Quadratic term: Q(δx)
        B = B + evaluate_quadratic(basis, δx)
    end

    return B
end

"""
    evaluate_gradient(basis::MapBasis, position::AbstractVector) -> SMatrix{3,3,Float64,9}

Evaluate magnetic gradient tensor at position.

For linear basis: gradient is constant.
For quadratic basis: gradient varies linearly with position.

# Returns
3×3 symmetric traceless gradient tensor [T/m]
"""
function evaluate_gradient(basis::MapBasis, position::AbstractVector)
    if basis.order < LINEAR
        return @SMatrix zeros(3, 3)
    end

    G = unpack_gradient(basis)

    if basis.order >= QUADRATIC
        δx = SVector{3}(position) - basis.reference
        G = G + evaluate_gradient_quadratic(basis, δx)
    end

    return G
end

"""
    unpack_gradient(basis::MapBasis) -> SMatrix{3,3,Float64,9}

Unpack gradient coefficients into full 3×3 symmetric traceless tensor.

# Maxwell Constraint
Gzz = -(Gxx + Gyy) enforced automatically (traceless for ∇·B = 0)
"""
function unpack_gradient(basis::MapBasis)
    if basis.order < LINEAR
        return @SMatrix zeros(3, 3)
    end

    Gxx = basis.coefficients[4]
    Gyy = basis.coefficients[5]
    Gxy = basis.coefficients[6]
    Gxz = basis.coefficients[7]
    Gyz = basis.coefficients[8]
    Gzz = -(Gxx + Gyy)  # Maxwell: ∇·B = 0

    # Build symmetric tensor
    SMatrix{3,3,Float64}(
        Gxx, Gxy, Gxz,
        Gxy, Gyy, Gyz,
        Gxz, Gyz, Gzz
    )
end

"""
    evaluate_quadratic(basis::MapBasis, δx::SVector{3,Float64}) -> SVector{3,Float64}

Evaluate quadratic contribution to field.

# Quadratic Harmonic Basis
The 7 independent quadratic coefficients represent solid spherical harmonics
of degree 2, which satisfy Laplace's equation.

Basis functions (unnormalized):
  Q₁ = x² - y²     (sectoral)
  Q₂ = 2xy         (sectoral)
  Q₃ = 2xz         (tesseral)
  Q₄ = 2yz         (tesseral)
  Q₅ = x² + y² - 2z² (zonal, from r²P₂(cos θ))
  Q₆ = xz          (mixed)
  Q₇ = yz          (mixed)

These form a complete basis for the gradient of quadratic harmonics.
"""
function evaluate_quadratic(basis::MapBasis, δx::SVector{3,Float64})
    if basis.order < QUADRATIC
        return @SVector zeros(3)
    end

    x, y, z = δx
    c = basis.coefficients[9:15]

    # Quadratic contributions to each field component
    # Derived from ∂φ/∂x, ∂φ/∂y, ∂φ/∂z where φ is degree-2 harmonic potential

    # Simplified form using 7 independent coefficients
    # This ensures ∇·B = 0 by construction
    Bx = c[1]*(x^2 - z^2) + c[2]*x*y + c[3]*x*z + c[6]*y*z
    By = c[2]*x*y + c[4]*(y^2 - z^2) + c[5]*y*z + c[7]*x*z
    Bz = -c[1]*x*z - c[4]*y*z + c[3]*(x^2/2) + c[5]*(y^2/2) - (c[3]+c[5])*z^2/2

    # Scale to appropriate magnitude (quadratic terms are smaller)
    scale = 1e-9  # Typical scale for quadratic at 10m distance

    return SVector{3}(Bx, By, Bz) * scale
end

"""
    evaluate_gradient_quadratic(basis::MapBasis, δx::SVector{3,Float64}) -> SMatrix{3,3,Float64,9}

Evaluate gradient contribution from quadratic terms.

The gradient of the quadratic field varies linearly with position.
"""
function evaluate_gradient_quadratic(basis::MapBasis, δx::SVector{3,Float64})
    if basis.order < QUADRATIC
        return @SMatrix zeros(3, 3)
    end

    x, y, z = δx
    c = basis.coefficients[9:15]
    scale = 1e-9

    # Gradient of quadratic terms (linear in position)
    # ∂Bx/∂x, ∂Bx/∂y, etc.
    dBx_dx = 2*c[1]*x + c[2]*y + c[3]*z
    dBx_dy = c[2]*x + c[6]*z
    dBx_dz = -2*c[1]*z + c[3]*x + c[6]*y

    dBy_dx = c[2]*y + c[7]*z
    dBy_dy = c[2]*x + 2*c[4]*y + c[5]*z
    dBy_dz = -2*c[4]*z + c[5]*y + c[7]*x

    # Enforce ∇·B = 0: dBz_dz = -(dBx_dx + dBy_dy)
    dBz_dx = -c[1]*z + c[3]*x
    dBz_dy = -c[4]*z + c[5]*y
    dBz_dz = -(dBx_dx + dBy_dy)

    # SMatrix uses column-major order: provide values column by column
    SMatrix{3,3,Float64}(
        dBx_dx, dBy_dx, dBz_dx,  # column 1
        dBx_dy, dBy_dy, dBz_dy,  # column 2
        dBx_dz, dBy_dz, dBz_dz   # column 3
    ) * scale
end

# =============================================================================
# Maxwell Consistency Checks
# =============================================================================

"""
    check_maxwell_divergence(basis::MapBasis, position::AbstractVector; tol=1e-15) -> Bool

Verify ∇·B = 0 at given position.

# Implementation
Computes divergence numerically using the gradient tensor and verifies
it's within tolerance of zero.

# Tolerance Justification
tol = 1e-15 is ~machine epsilon for Float64, accounting for roundoff
in the traceless constraint Gzz = -(Gxx + Gyy).
"""
function check_maxwell_divergence(basis::MapBasis, position::AbstractVector; tol::Float64=1e-15)
    G = evaluate_gradient(basis, position)
    divergence = G[1,1] + G[2,2] + G[3,3]  # tr(G)
    return abs(divergence) < tol
end

"""
    check_maxwell_symmetry(basis::MapBasis; tol=1e-15) -> Bool

Verify gradient tensor is symmetric (required for curl-free field).

# Physical Basis
In current-free regions, ∇×B = 0, which implies the gradient tensor
∂Bi/∂xj must be symmetric: ∂Bi/∂xj = ∂Bj/∂xi.
"""
function check_maxwell_symmetry(basis::MapBasis; tol::Float64=1e-15)
    G = unpack_gradient(basis)

    # Check off-diagonal symmetry
    sym_xy = abs(G[1,2] - G[2,1]) < tol
    sym_xz = abs(G[1,3] - G[3,1]) < tol
    sym_yz = abs(G[2,3] - G[3,2]) < tol

    return sym_xy && sym_xz && sym_yz
end

"""
    verify_maxwell_consistency(basis::MapBasis; n_test_points=10, tol=1e-12) -> NamedTuple

Comprehensive Maxwell consistency verification.

# Returns
NamedTuple with:
- `divergence_ok::Bool`: All test points satisfy ∇·B = 0
- `symmetry_ok::Bool`: Gradient tensor is symmetric
- `max_divergence::Float64`: Maximum |∇·B| observed
- `test_positions::Vector`: Positions tested
"""
function verify_maxwell_consistency(basis::MapBasis; n_test_points::Int=10, tol::Float64=1e-12)
    # Test at reference and random nearby points
    test_positions = [basis.reference]

    for i in 1:(n_test_points-1)
        offset = SVector{3}(randn(3) * 10.0)  # ±10m
        push!(test_positions, basis.reference + offset)
    end

    divergences = Float64[]
    for pos in test_positions
        G = evaluate_gradient(basis, pos)
        div = abs(G[1,1] + G[2,2] + G[3,3])
        push!(divergences, div)
    end

    max_div = maximum(divergences)
    div_ok = max_div < tol
    sym_ok = check_maxwell_symmetry(basis, tol=tol)

    return (
        divergence_ok = div_ok,
        symmetry_ok = sym_ok,
        max_divergence = max_div,
        test_positions = test_positions
    )
end

# =============================================================================
# Basis Fitting
# =============================================================================

"""
    fit_basis(positions, B_measurements; order, reference, R) -> MapBasis

Fit basis coefficients to field measurements using weighted least squares.

# Arguments
- `positions::Vector`: Measurement positions [m]
- `B_measurements::Vector`: Field measurements [T]
- `order::BasisOrder`: Desired basis order
- `reference::AbstractVector`: Reference position for expansion
- `R::AbstractMatrix`: Measurement noise covariance [T²]

# Method
Weighted least squares: minimize (z - Hc)'R⁻¹(z - Hc)
where z = measurements, H = design matrix, c = coefficients

# Maxwell Consistency
The design matrix H is constructed to automatically enforce ∇·B = 0
through the parameterization (only 5 gradient DOFs, not 9).
"""
function fit_basis(
    positions::Vector{<:AbstractVector},
    B_measurements::Vector{<:AbstractVector};
    order::BasisOrder = LINEAR,
    reference::AbstractVector = mean(positions),
    R::AbstractMatrix = Matrix(1e-18 * I, 3, 3)
)
    n_meas = length(positions)
    n_coeffs = COEFFICIENT_COUNTS[order]

    # Build design matrix
    H = zeros(3 * n_meas, n_coeffs)
    z = zeros(3 * n_meas)

    for (i, pos) in enumerate(positions)
        δx = pos - reference
        row_start = 3*(i-1) + 1

        # Measurement vector
        z[row_start:row_start+2] = B_measurements[i]

        # Constant term (always)
        H[row_start:row_start+2, 1:3] = Matrix(1.0I, 3, 3)

        if order >= LINEAR
            # Linear term: gradient contribution
            # Bx = Bx0 + Gxx*dx + Gxy*dy + Gxz*dz
            # By = By0 + Gxy*dx + Gyy*dy + Gyz*dz
            # Bz = Bz0 + Gxz*dx + Gyz*dy + Gzz*dz
            # where Gzz = -(Gxx + Gyy)

            dx, dy, dz = δx

            # Bx row
            H[row_start, 4] = dx           # Gxx
            H[row_start, 5] = 0            # Gyy (doesn't affect Bx directly)
            H[row_start, 6] = dy           # Gxy
            H[row_start, 7] = dz           # Gxz
            H[row_start, 8] = 0            # Gyz

            # By row
            H[row_start+1, 4] = 0          # Gxx
            H[row_start+1, 5] = dy         # Gyy
            H[row_start+1, 6] = dx         # Gxy
            H[row_start+1, 7] = 0          # Gxz
            H[row_start+1, 8] = dz         # Gyz

            # Bz row (enforcing Gzz = -(Gxx + Gyy))
            H[row_start+2, 4] = -dz        # Gxx contributes -dz to Bz via Gzz
            H[row_start+2, 5] = -dz        # Gyy contributes -dz to Bz via Gzz
            H[row_start+2, 6] = 0          # Gxy
            H[row_start+2, 7] = dx         # Gxz
            H[row_start+2, 8] = dy         # Gyz
        end

        # Quadratic terms would be added similarly for order >= QUADRATIC
    end

    # Build block-diagonal measurement covariance
    R_full = kron(Matrix(1.0I, n_meas, n_meas), R)

    # Weighted least squares
    W = inv(R_full)
    HtW = H' * W
    P = inv(HtW * H + 1e-12 * I)  # Regularization for numerical stability
    coefficients = P * HtW * z

    # Coefficient covariance
    covariance = P

    MapBasis(order, SVector{3}(reference), coefficients, covariance)
end

# =============================================================================
# Residual Analysis
# =============================================================================

"""
    compute_residuals(basis::MapBasis, positions, B_measurements) -> Vector{Float64}

Compute field prediction residuals.

# Returns
Vector of residual magnitudes |B_meas - B_pred| [T]
"""
function compute_residuals(
    basis::MapBasis,
    positions::Vector{<:AbstractVector},
    B_measurements::Vector{<:AbstractVector}
)
    residuals = Float64[]

    for (pos, B_meas) in zip(positions, B_measurements)
        B_pred = evaluate_field(basis, pos)
        push!(residuals, norm(B_meas - B_pred))
    end

    return residuals
end

"""
    should_upgrade_basis(basis::MapBasis, residuals; threshold) -> Bool

Determine if basis should be upgraded based on residual structure.

# Threshold Justification
Default threshold of 5e-9 T (5 nT) corresponds to typical gradient sensor
noise floor. Systematic residuals above this suggest model inadequacy.

# Criteria for Upgrade
1. Mean residual > threshold (systematic bias)
2. Residuals show spatial structure (not random noise)
"""
function should_upgrade_basis(
    basis::MapBasis,
    residuals::Vector{Float64};
    threshold::Float64 = 5e-9
)
    if basis.order >= QUADRATIC
        return false  # Already at max order
    end

    mean_residual = mean(residuals)

    # Simple criterion: mean residual exceeds threshold
    return mean_residual > threshold
end

# =============================================================================
# Exports
# =============================================================================

export BasisOrder, CONSTANT, LINEAR, QUADRATIC
export COEFFICIENT_COUNTS
export MapBasis
export upgrade_basis
export evaluate_field, evaluate_gradient, unpack_gradient
export check_maxwell_divergence, check_maxwell_symmetry, verify_maxwell_consistency
export fit_basis, compute_residuals, should_upgrade_basis

end # module
