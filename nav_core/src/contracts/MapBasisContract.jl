# ============================================================================
# MapBasisContract.jl - Single source of truth for magnetic field basis math
# ============================================================================
#
# AUTHORITATIVE CONTRACT: All field prediction and Jacobian computations MUST
# use functions from this contract. Duplicated implementations are prohibited.
#
# Physics grounding (see docs/specs/map_basis_spec.md):
# - Magnetic field in source-free region: B = -∇Φ, ∇²Φ = 0
# - Solutions are solid spherical harmonics (Maxwell-consistent)
# - Gradient tensor is symmetric and traceless: Gzz = -(Gxx + Gyy)
#
# State parameterization:
#   d=3  (MODE_B0):        [B0x, B0y, B0z]
#   d=8  (MODE_LINEAR):    [B0(3), G5(5)] where G5 = [Gxx, Gyy, Gxy, Gxz, Gyz]
#   d=15 (MODE_QUADRATIC): [B0(3), G5(5), Q7(7)]
#
# All computations use NORMALIZED coordinates x̃ = (x - center) / L
# where L is the tile scale (half-width). This makes all basis terms O(1).
#
# Physical unit conversion:
#   G_physical[T/m] = G_normalized / L
#   Q_physical[T/m²] = Q_normalized / L²
# ============================================================================

using LinearAlgebra
using StaticArrays

# Local type alias (same as MapContract.Vec3Map, defined here to avoid load-order dependency)
const Vec3Map = SVector{3, Float64}
const Mat3Map = SMatrix{3, 3, Float64, 9}

# ============================================================================
# Type Definitions
# ============================================================================

"""
    MapModelMode

Explicit map model mode. NOT inferred from coefficient vector length.
The integer value equals the number of active coefficients.
"""
@enum MapModelMode begin
    MODE_B0 = 3         # Constant field only
    MODE_LINEAR = 8     # Constant + gradient (5 independent components)
    MODE_QUADRATIC = 15 # Constant + gradient + quadratic (7 additional)
end

"""
    TileLocalFrame

Defines the local coordinate frame for a tile, enabling normalized coordinates.
All basis evaluations use normalized coordinates x̃ = (x - center) / scale.

# Fields
- `center::Vec3Map`: Tile center in world NED frame [m]
- `scale::Float64`: Characteristic length L [m], typically tile half-width (25m for 50m tiles)
"""
struct TileLocalFrame
    center::Vec3Map
    scale::Float64

    function TileLocalFrame(center::Vec3Map, scale::Float64)
        scale > 0 || throw(ArgumentError("scale must be positive, got $scale"))
        new(center, scale)
    end
end

# Convenience constructor from existing tile
TileLocalFrame(center::Vec3Map; tile_size::Float64=50.0) =
    TileLocalFrame(center, tile_size / 2)

"""Number of coefficients for a given mode."""
n_coefficients(mode::MapModelMode) = Int(mode)

"""
    mode_from_dim(n::Int) -> MapModelMode

Infer model mode from coefficient count (temporary until explicit mode-per-tile in Task 3).
"""
function mode_from_dim(n::Int)
    n >= 15 && return MODE_QUADRATIC
    n >= 8  && return MODE_LINEAR
    n >= 3  && return MODE_B0
    throw(ArgumentError("n=$n too small for any mode (need ≥ 3)"))
end

# ============================================================================
# Coordinate Normalization
# ============================================================================

"""
    normalize_position(frame::TileLocalFrame, x_world::Vec3Map) -> Vec3Map

Convert world position to normalized tile-local coordinates.
Returns x̃ = (x - center) / scale, dimensionless.
"""
normalize_position(frame::TileLocalFrame, x_world::Vec3Map) =
    (x_world - frame.center) / frame.scale

"""
    denormalize_position(frame::TileLocalFrame, x_norm::Vec3Map) -> Vec3Map

Convert normalized coordinates back to world position.
Returns x = center + scale * x̃ [m].
"""
denormalize_position(frame::TileLocalFrame, x_norm::Vec3Map) =
    frame.center + frame.scale * x_norm

# ============================================================================
# Field Evaluation
# ============================================================================

"""
    evaluate_field(α::AbstractVector, x̃::Vec3Map, mode::MapModelMode) -> Vec3Map

Evaluate magnetic field at normalized position x̃.

# Arguments
- `α`: Coefficient vector, length must be ≥ n_coefficients(mode)
- `x̃`: Normalized position (x - center) / L, dimensionless
- `mode`: Model mode determining which terms are active

# Returns
- B: Magnetic field vector [T]

# Physics
The field is the sum of:
- B0 (constant): α[1:3]
- G·x̃ (linear): gradient contribution from α[4:8]
- Q(x̃) (quadratic): curvature contribution from α[9:15]

All terms satisfy Maxwell's equations (∇·B = 0) by construction.
"""
function evaluate_field(α::AbstractVector, x̃::Vec3Map, mode::MapModelMode)
    n = n_coefficients(mode)
    length(α) >= n || throw(ArgumentError("α has $(length(α)) elements, need ≥$n for $mode"))

    # B0 contribution (always present)
    B = Vec3Map(α[1], α[2], α[3])

    if mode == MODE_B0
        return B
    end

    # Gradient contribution (MODE_LINEAR and above)
    # G5 = [Gxx, Gyy, Gxy, Gxz, Gyz], Gzz = -(Gxx + Gyy)
    Gxx, Gyy, Gxy, Gxz, Gyz = α[4], α[5], α[6], α[7], α[8]
    Gzz = -(Gxx + Gyy)

    x, y, z = x̃[1], x̃[2], x̃[3]

    Bx_G = Gxx * x + Gxy * y + Gxz * z
    By_G = Gxy * x + Gyy * y + Gyz * z
    Bz_G = Gxz * x + Gyz * y + Gzz * z

    B = B + Vec3Map(Bx_G, By_G, Bz_G)

    if mode == MODE_LINEAR
        return B
    end

    # Quadratic contribution (MODE_QUADRATIC)
    # Q7 = [Q1, Q2, Q3, Q4, Q5, Q6, Q7] - 7 Maxwell-consistent harmonics
    # Derived from scalar potential Φ satisfying Laplace's equation (∇²Φ = 0):
    #   Φ = Q1*(x² - z²) + Q2*(y² - z²) + Q3*xy + Q4*xz + Q5*yz + Q6*(x² - y²) + Q7*2xyz
    # Field B = -∇Φ guarantees ∇·B = 0 (Maxwell divergence-free constraint)
    Q1, Q2, Q3, Q4, Q5, Q6, Q7 = α[9], α[10], α[11], α[12], α[13], α[14], α[15]

    # B = -∇Φ (each component is -∂Φ/∂{x,y,z})
    Bx_Q = -(2*Q1 + 2*Q6)*x - Q3*y - Q4*z - 2*Q7*y*z
    By_Q = -Q3*x - (2*Q2 - 2*Q6)*y - Q5*z - 2*Q7*x*z
    Bz_Q = -Q4*x - Q5*y + 2*(Q1 + Q2)*z - 2*Q7*x*y

    B = B + Vec3Map(Bx_Q, By_Q, Bz_Q)

    return B
end

"""
    evaluate_gradient(α::AbstractVector, x̃::Vec3Map, mode::MapModelMode) -> SMatrix{3,3}

Evaluate gradient tensor ∂B/∂x̃ at normalized position.

# Returns
- G: 3×3 symmetric, traceless gradient tensor [T] (in normalized coords)
  To convert to physical units [T/m], divide by tile scale L.

Only meaningful for mode ≥ MODE_LINEAR. Returns zero matrix for MODE_B0.
"""
function evaluate_gradient(α::AbstractVector, x̃::Vec3Map, mode::MapModelMode)
    if mode == MODE_B0
        return @SMatrix zeros(3, 3)
    end

    # Base gradient from G5
    Gxx, Gyy, Gxy, Gxz, Gyz = α[4], α[5], α[6], α[7], α[8]
    Gzz = -(Gxx + Gyy)

    G = @SMatrix [
        Gxx  Gxy  Gxz;
        Gxy  Gyy  Gyz;
        Gxz  Gyz  Gzz
    ]

    if mode == MODE_LINEAR
        return G
    end

    # Quadratic contribution to gradient
    # From Maxwell-consistent formulas where B = -∇Φ:
    #   Bx = -(2Q1 + 2Q6)*x - Q3*y - Q4*z - 2*Q7*yz
    #   By = -Q3*x - (2Q2 - 2Q6)*y - Q5*z - 2*Q7*xz
    #   Bz = -Q4*x - Q5*y + 2*(Q1 + Q2)*z - 2*Q7*xy
    Q1, Q2, Q3, Q4, Q5, Q6, Q7 = α[9], α[10], α[11], α[12], α[13], α[14], α[15]
    x, y, z = x̃[1], x̃[2], x̃[3]

    # ∂B/∂x̃ from quadratic terms (gradient is symmetric and traceless)
    dBx_dx = -(2*Q1 + 2*Q6)
    dBx_dy = -Q3 - 2*Q7*z
    dBx_dz = -Q4 - 2*Q7*y

    dBy_dx = -Q3 - 2*Q7*z     # = dBx_dy (symmetric)
    dBy_dy = -(2*Q2 - 2*Q6)
    dBy_dz = -Q5 - 2*Q7*x

    dBz_dx = -Q4 - 2*Q7*y     # = dBx_dz (symmetric)
    dBz_dy = -Q5 - 2*Q7*x     # = dBy_dz (symmetric)
    dBz_dz = 2*(Q1 + Q2)      # Traceless: sum of diagonal = 0

    G_Q = @SMatrix [
        dBx_dx  dBx_dy  dBx_dz;
        dBy_dx  dBy_dy  dBy_dz;
        dBz_dx  dBz_dy  dBz_dz
    ]

    return G + G_Q
end

# ============================================================================
# Jacobian Functions
# ============================================================================

"""
    field_jacobian(x̃::Vec3Map, mode::MapModelMode) -> Matrix{Float64}

Compute Jacobian ∂B/∂α at normalized position x̃.

# Returns
- H: 3×d matrix where d = n_coefficients(mode)

The Jacobian is used in:
- Measurement update: innovation = z - H*α, update α via H'*R⁻¹*H
- Observability analysis: rank(H) determines identifiable parameters

# Coordinate Note
This Jacobian is for the NORMALIZED coordinate system. The scaling by L
is absorbed into the coefficient units, not the Jacobian.
"""
function field_jacobian(x̃::Vec3Map, mode::MapModelMode)
    n = n_coefficients(mode)
    H = zeros(3, n)

    # B0 block (always present): ∂B/∂B0 = I
    H[1, 1] = 1.0
    H[2, 2] = 1.0
    H[3, 3] = 1.0

    if mode == MODE_B0
        return H
    end

    # Gradient block: ∂B/∂G5
    # G5 = [Gxx, Gyy, Gxy, Gxz, Gyz] at indices 4:8
    x, y, z = x̃[1], x̃[2], x̃[3]

    # ∂Bx/∂G5 = [x, 0, y, z, 0]
    H[1, 4] = x    # ∂Bx/∂Gxx
    H[1, 5] = 0.0  # ∂Bx/∂Gyy
    H[1, 6] = y    # ∂Bx/∂Gxy
    H[1, 7] = z    # ∂Bx/∂Gxz
    H[1, 8] = 0.0  # ∂Bx/∂Gyz

    # ∂By/∂G5 = [0, y, x, 0, z]
    H[2, 4] = 0.0  # ∂By/∂Gxx
    H[2, 5] = y    # ∂By/∂Gyy
    H[2, 6] = x    # ∂By/∂Gxy
    H[2, 7] = 0.0  # ∂By/∂Gxz
    H[2, 8] = z    # ∂By/∂Gyz

    # ∂Bz/∂G5 = [-z, -z, 0, x, y] (from Gzz = -(Gxx + Gyy))
    H[3, 4] = -z   # ∂Bz/∂Gxx (via Gzz)
    H[3, 5] = -z   # ∂Bz/∂Gyy (via Gzz)
    H[3, 6] = 0.0  # ∂Bz/∂Gxy
    H[3, 7] = x    # ∂Bz/∂Gxz
    H[3, 8] = y    # ∂Bz/∂Gyz

    if mode == MODE_LINEAR
        return H
    end

    # Quadratic block: ∂B/∂Q7
    # Q7 = [Q1, Q2, Q3, Q4, Q5, Q6, Q7] at indices 9:15
    # Derived from B = -∇Φ where Φ = Q1*(x²-z²) + Q2*(y²-z²) + Q3*xy + Q4*xz + Q5*yz + Q6*(x²-y²) + Q7*2xyz
    #
    # Bx = -(2Q1 + 2Q6)*x - Q3*y - Q4*z - 2*Q7*yz
    # By = -Q3*x - (2Q2 - 2Q6)*y - Q5*z - 2*Q7*xz
    # Bz = -Q4*x - Q5*y + 2*(Q1 + Q2)*z - 2*Q7*xy

    # ∂Bx/∂Q
    H[1,  9] = -2*x      # ∂Bx/∂Q1
    H[1, 10] = 0.0       # ∂Bx/∂Q2
    H[1, 11] = -y        # ∂Bx/∂Q3
    H[1, 12] = -z        # ∂Bx/∂Q4
    H[1, 13] = 0.0       # ∂Bx/∂Q5
    H[1, 14] = -2*x      # ∂Bx/∂Q6
    H[1, 15] = -2*y*z    # ∂Bx/∂Q7

    # ∂By/∂Q
    H[2,  9] = 0.0       # ∂By/∂Q1
    H[2, 10] = -2*y      # ∂By/∂Q2
    H[2, 11] = -x        # ∂By/∂Q3
    H[2, 12] = 0.0       # ∂By/∂Q4
    H[2, 13] = -z        # ∂By/∂Q5
    H[2, 14] = 2*y       # ∂By/∂Q6
    H[2, 15] = -2*x*z    # ∂By/∂Q7

    # ∂Bz/∂Q
    H[3,  9] = 2*z       # ∂Bz/∂Q1
    H[3, 10] = 2*z       # ∂Bz/∂Q2
    H[3, 11] = 0.0       # ∂Bz/∂Q3
    H[3, 12] = -x        # ∂Bz/∂Q4
    H[3, 13] = -y        # ∂Bz/∂Q5
    H[3, 14] = 0.0       # ∂Bz/∂Q6
    H[3, 15] = -2*x*y    # ∂Bz/∂Q7

    return H
end

"""
    gradient_jacobian(x̃::Vec3Map, mode::MapModelMode) -> Matrix{Float64}

Compute Jacobian ∂G5/∂α at normalized position x̃.

# Returns
- J: 5×d matrix mapping coefficients to gradient tensor components
  G5 ordering: [Gxx, Gyy, Gxy, Gxz, Gyz]

Only non-trivial for MODE_QUADRATIC (quadratic terms contribute position-dependent gradient).
"""
function gradient_jacobian(x̃::Vec3Map, mode::MapModelMode)
    n = n_coefficients(mode)
    J = zeros(5, n)

    if mode == MODE_B0
        return J
    end

    # For MODE_LINEAR: ∂G5/∂G5 = I, others zero
    J[1, 4] = 1.0  # ∂Gxx/∂Gxx
    J[2, 5] = 1.0  # ∂Gyy/∂Gyy
    J[3, 6] = 1.0  # ∂Gxy/∂Gxy
    J[4, 7] = 1.0  # ∂Gxz/∂Gxz
    J[5, 8] = 1.0  # ∂Gyz/∂Gyz

    if mode == MODE_LINEAR
        return J
    end

    # For MODE_QUADRATIC: quadratic terms contribute to gradient
    # From Maxwell-consistent formulas, the gradient contributions are:
    #   Gxx_Q = -(2Q1 + 2Q6)
    #   Gyy_Q = -(2Q2 - 2Q6)
    #   Gxy_Q = -Q3 - 2*Q7*z
    #   Gxz_Q = -Q4 - 2*Q7*y
    #   Gyz_Q = -Q5 - 2*Q7*x
    x, y, z = x̃[1], x̃[2], x̃[3]

    # ∂Gxx/∂Q
    J[1, 9] = -2.0    # ∂Gxx/∂Q1
    J[1, 14] = -2.0   # ∂Gxx/∂Q6

    # ∂Gyy/∂Q
    J[2, 10] = -2.0   # ∂Gyy/∂Q2
    J[2, 14] = 2.0    # ∂Gyy/∂Q6

    # ∂Gxy/∂Q (position-dependent via Q7)
    J[3, 11] = -1.0   # ∂Gxy/∂Q3
    J[3, 15] = -2*z   # ∂Gxy/∂Q7

    # ∂Gxz/∂Q (position-dependent via Q7)
    J[4, 12] = -1.0   # ∂Gxz/∂Q4
    J[4, 15] = -2*y   # ∂Gxz/∂Q7

    # ∂Gyz/∂Q (position-dependent via Q7)
    J[5, 13] = -1.0   # ∂Gyz/∂Q5
    J[5, 15] = -2*x   # ∂Gyz/∂Q7

    return J
end

# ============================================================================
# Unit Conversion Utilities
# ============================================================================

"""
    to_physical_gradient(G_normalized, L::Float64) -> SMatrix{3,3}

Convert gradient tensor from normalized coordinates to SI units [T/m].
"""
to_physical_gradient(G::SMatrix{3,3,Float64}, L::Float64) = G / L

"""
    to_physical_curvature(Q_normalized, L::Float64)

Convert quadratic coefficients from normalized coordinates to SI units [T/m²].
"""
to_physical_curvature(Q::AbstractVector, L::Float64) = Q / L^2

"""
    from_physical_gradient(G_physical, L::Float64) -> SMatrix{3,3}

Convert gradient tensor from SI units [T/m] to normalized coordinates.
"""
from_physical_gradient(G::SMatrix{3,3,Float64}, L::Float64) = G * L

"""
    from_physical_curvature(Q_physical, L::Float64)

Convert quadratic coefficients from SI units [T/m²] to normalized coordinates.
"""
from_physical_curvature(Q::AbstractVector, L::Float64) = Q * L^2

# ============================================================================
# Validation Utilities
# ============================================================================

"""
    verify_maxwell_divergence(α::AbstractVector, x̃::Vec3Map, mode::MapModelMode;
                               tol::Float64=1e-10) -> Bool

Verify that ∇·B = 0 (Maxwell's divergence constraint).
Uses finite differences on evaluate_field.
"""
function verify_maxwell_divergence(α::AbstractVector, x̃::Vec3Map, mode::MapModelMode;
                                    δ::Float64=1e-6, tol::Float64=1e-10)
    # Numerical divergence via central differences
    div_B = 0.0
    for i in 1:3
        e_i = Vec3Map(i == 1 ? 1.0 : 0.0, i == 2 ? 1.0 : 0.0, i == 3 ? 1.0 : 0.0)
        B_plus = evaluate_field(α, x̃ + δ * e_i, mode)
        B_minus = evaluate_field(α, x̃ - δ * e_i, mode)
        div_B += (B_plus[i] - B_minus[i]) / (2 * δ)
    end
    return abs(div_B) < tol
end

"""
    verify_jacobian_consistency(α::AbstractVector, x̃::Vec3Map, mode::MapModelMode;
                                 δ::Float64=1e-7, tol::Float64=1e-5) -> Bool

Verify Jacobian matches finite difference of evaluator.
"""
function verify_jacobian_consistency(α::AbstractVector, x̃::Vec3Map, mode::MapModelMode;
                                      δ::Float64=1e-7, tol::Float64=1e-5)
    H = field_jacobian(x̃, mode)
    n = n_coefficients(mode)

    B0 = evaluate_field(α, x̃, mode)

    for j in 1:n
        α_pert = copy(α)
        α_pert[j] += δ
        B_pert = evaluate_field(α_pert, x̃, mode)

        dB_fd = (B_pert - B0) / δ  # Finite difference
        dB_an = H[:, j]            # Analytic

        err = norm(dB_fd - dB_an)
        if err > tol
            return false
        end
    end
    return true
end

"""
    verify_gradient_symmetry(α::AbstractVector, x̃::Vec3Map, mode::MapModelMode;
                              tol::Float64=1e-12) -> Bool

Verify gradient tensor is symmetric (required by Maxwell's curl-free condition).
"""
function verify_gradient_symmetry(α::AbstractVector, x̃::Vec3Map, mode::MapModelMode;
                                   tol::Float64=1e-12)
    if mode == MODE_B0
        return true  # No gradient for B0-only
    end
    G = evaluate_gradient(α, x̃, mode)
    return norm(G - G') < tol
end

"""
    verify_gradient_traceless(α::AbstractVector, x̃::Vec3Map, mode::MapModelMode;
                               tol::Float64=1e-12) -> Bool

Verify gradient tensor is traceless (∇·B = 0).
"""
function verify_gradient_traceless(α::AbstractVector, x̃::Vec3Map, mode::MapModelMode;
                                    tol::Float64=1e-12)
    if mode == MODE_B0
        return true
    end
    G = evaluate_gradient(α, x̃, mode)
    return abs(tr(G)) < tol
end

# ============================================================================
# Prior Policy — physically motivated priors for tile coefficients
# ============================================================================

"""
    PriorPolicy

Specifies prior standard deviations for tile coefficients in PHYSICAL units.
The constructor converts to normalized-unit variances using the tile scale L.

# Physical parameters
- `σ_B0::Float64`: Prior std for B0 components [T]. Default 1e-6 T (1000 nT).
  Rationale: crustal anomalies are typically 100-2000 nT; 1000 nT is weakly informative.

- `σ_G::Float64`: Prior std for gradient components [T/m]. Default 2e-7 T/m (200 nT/m).
  Rationale: typical crustal gradients are 10-200 nT/m over 50m tiles.

- `σ_Q::Float64`: Prior std for quadratic components [T/m²]. Default 1e-8 T/m² (10 nT/m²).
  Rationale: curvature at 50m scale is typically 1-10 nT/m².

# Conversion to normalized units
Coefficients use normalized coordinates x̃ = x/L, so:
- B0 variance (order 0): σ_B0²
- G variance (order 1):  (σ_G × L)²
- Q variance (order 2):  (σ_Q × L²)²

This ensures the prior is neither too tight (suppressing learning) nor too loose
(dominating conditioning) regardless of tile scale.
"""
struct PriorPolicy
    σ_B0::Float64  # [T]
    σ_G::Float64   # [T/m]
    σ_Q::Float64   # [T/m²]
end

"""Default prior policy based on typical crustal magnetic field scales."""
const DEFAULT_PRIOR_POLICY = PriorPolicy(1e-6, 2e-7, 1e-8)

"""
    prior_variances(policy::PriorPolicy, mode::MapModelMode, L::Float64) -> Vector{Float64}

Compute per-coefficient prior variances in normalized units.

# Arguments
- `policy`: Physical-unit prior specification
- `mode`: Determines number of coefficients
- `L`: Tile scale (half-width) [m]

# Returns
Vector of length `n_coefficients(mode)` with prior variances.
"""
function prior_variances(policy::PriorPolicy, mode::MapModelMode, L::Float64)
    n = n_coefficients(mode)
    pv = zeros(n)

    # B0 (order 0): variance = σ_B0²
    pv[1:3] .= policy.σ_B0^2

    if n >= 8
        # Gradient (order 1): normalized σ = σ_G_phys × L
        pv[4:8] .= (policy.σ_G * L)^2
    end

    if n >= 15
        # Quadratic (order 2): normalized σ = σ_Q_phys × L²
        pv[9:15] .= (policy.σ_Q * L^2)^2
    end

    return pv
end

# ============================================================================
# Exports
# ============================================================================
#
# Note: pack_gradient_tensor and unpack_gradient_tensor are defined in
# mapping/tile_coefficients.jl and exported from there to avoid duplication.
# Once migration is complete, they should move here.

export PriorPolicy, DEFAULT_PRIOR_POLICY, prior_variances
export MapModelMode, MODE_B0, MODE_LINEAR, MODE_QUADRATIC
export TileLocalFrame, n_coefficients, mode_from_dim
export normalize_position, denormalize_position
export evaluate_field, evaluate_gradient
export field_jacobian, gradient_jacobian
export to_physical_gradient, to_physical_curvature
export from_physical_gradient, from_physical_curvature
export verify_maxwell_divergence, verify_jacobian_consistency
export verify_gradient_symmetry, verify_gradient_traceless
