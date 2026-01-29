# =============================================================================
# D8 Measurement Contract
# =============================================================================
#
# This file defines the AUTHORITATIVE ordering and units for d=8 magnetic
# measurements. All code that produces or consumes d=8 measurements MUST
# conform to this contract.
#
# =============================================================================

module D8Contract

using StaticArrays
using LinearAlgebra

export D8_ORDERING, D8_UNITS, D8_INDICES
export D8Measurement, D8Covariance, D8AuditTrace
export pack_d8, unpack_d8, validate_d8_ordering
export create_d8_R, create_d8_S_total

# =============================================================================
# Ordering Contract
# =============================================================================

"""
    D8_ORDERING

Canonical ordering of the d=8 measurement vector:
- Indices 1-3: Field components Bx, By, Bz
- Indices 4-8: Gradient components Gxx, Gxy, Gxz, Gyy, Gyz

Note: Gzz = -(Gxx + Gyy) by Maxwell's equations (div B = 0), so only 5
independent gradient components are used.
"""
const D8_ORDERING = [:Bx, :By, :Bz, :Gxx, :Gxy, :Gxz, :Gyy, :Gyz]

"""
    D8_INDICES

Named indices for accessing d=8 vector components.
"""
const D8_INDICES = (
    Bx = 1, By = 2, Bz = 3,
    Gxx = 4, Gxy = 5, Gxz = 6, Gyy = 7, Gyz = 8
)

"""
    D8_UNITS

Units for each component:
- Field B: Tesla (T), typical Earth field ~50 μT = 50e-6 T
- Gradient G: Tesla/meter (T/m), typical gradient ~1e-7 T/m near dipoles

CRITICAL: All covariance matrices MUST use these units consistently.
Mixing nT and T will cause χ² values to be off by factors of 1e18.
"""
const D8_UNITS = (
    B_unit = "T",           # Tesla
    G_unit = "T/m",         # Tesla per meter
    B_typical = 50e-6,      # Typical field magnitude (50 μT)
    G_typical = 1e-7,       # Typical gradient magnitude
    B_noise_floor = 1e-9,   # Sensor noise floor ~1 nT
    G_noise_floor = 1e-9    # Gradient noise floor ~1 nT/m
)

# =============================================================================
# Measurement Type
# =============================================================================

"""
    D8Measurement

A d=8 magnetic measurement with guaranteed ordering.
"""
struct D8Measurement
    z::SVector{8,Float64}  # [Bx, By, Bz, Gxx, Gxy, Gxz, Gyy, Gyz] in T and T/m

    function D8Measurement(z::SVector{8,Float64})
        new(z)
    end
end

"""
    pack_d8(B::SVector{3}, G::SMatrix{3,3}) -> D8Measurement

Pack field vector B and gradient tensor G into canonical d=8 ordering.
B should be in Tesla, G should be in T/m.
"""
function pack_d8(B::SVector{3,Float64}, G::SMatrix{3,3,Float64,9})
    z = SVector{8,Float64}(
        B[1], B[2], B[3],           # Bx, By, Bz
        G[1,1],                      # Gxx
        G[1,2],                      # Gxy (= Gyx by symmetry)
        G[1,3],                      # Gxz (= Gzx by symmetry)
        G[2,2],                      # Gyy
        G[2,3]                       # Gyz (= Gzy by symmetry)
    )
    return D8Measurement(z)
end

"""
    unpack_d8(meas::D8Measurement) -> (B::SVector{3}, G::SMatrix{3,3})

Unpack d=8 measurement to field vector and full gradient tensor.
"""
function unpack_d8(meas::D8Measurement)
    z = meas.z

    B = SVector(z[1], z[2], z[3])

    # Reconstruct full symmetric tensor
    Gxx, Gxy, Gxz, Gyy, Gyz = z[4], z[5], z[6], z[7], z[8]
    Gzz = -(Gxx + Gyy)  # Trace = 0 (div B = 0)

    G = SMatrix{3,3}(
        Gxx, Gxy, Gxz,
        Gxy, Gyy, Gyz,
        Gxz, Gyz, Gzz
    )

    return (B, G)
end

# =============================================================================
# Covariance Types
# =============================================================================

"""
    D8Covariance

Innovation or measurement covariance for d=8 measurements.
Enforces that B and G blocks have correct units.
"""
struct D8Covariance
    Σ::SMatrix{8,8,Float64,64}

    function D8Covariance(Σ::AbstractMatrix)
        @assert size(Σ) == (8, 8) "D8Covariance must be 8x8"
        @assert issymmetric(Σ) || norm(Σ - Σ') < 1e-12 * norm(Σ) "D8Covariance must be symmetric"
        new(SMatrix{8,8}((Σ + Σ') / 2))
    end
end

"""
    create_d8_R(σ_B::Float64, σ_G::Float64) -> D8Covariance

Create measurement noise covariance with specified standard deviations.
- σ_B: Field noise std in Tesla
- σ_G: Gradient noise std in T/m
"""
function create_d8_R(σ_B::Float64, σ_G::Float64)
    R = zeros(8, 8)
    R[1:3, 1:3] = σ_B^2 * I(3)    # B block
    R[4:8, 4:8] = σ_G^2 * I(5)    # G block
    return D8Covariance(R)
end

"""
    create_d8_S_total(H::Matrix, P::Matrix, R::D8Covariance, Q_model::Float64) -> D8Covariance

Create complete innovation covariance:
    S = H P Hᵀ + R + Q_model·I

This is the ONLY correct form for computing χ².

Arguments:
- H: Measurement Jacobian (8 × n_states)
- P: State covariance (n_states × n_states)
- R: Measurement noise covariance
- Q_model: Model mismatch uncertainty (added to diagonal)
"""
function create_d8_S_total(
    H::AbstractMatrix{Float64},
    P::AbstractMatrix{Float64},
    R::D8Covariance;
    Q_model_B::Float64 = 0.0,
    Q_model_G::Float64 = 0.0
)
    HPHt = H * P * H'

    # Add model mismatch (unmodeled field complexity)
    Q = zeros(8, 8)
    Q[1:3, 1:3] = Q_model_B * I(3)
    Q[4:8, 4:8] = Q_model_G * I(5)

    S = HPHt + R.Σ + Q
    return D8Covariance(S)
end

# =============================================================================
# Audit Trace
# =============================================================================

"""
    D8AuditTrace

Complete audit record for a single d=8 update.
Used for diagnosing NEES/NIS calibration issues.
"""
struct D8AuditTrace
    t::Float64                          # Timestamp

    # Raw measurement
    z::SVector{8,Float64}               # Measurement vector
    z_pred::SVector{8,Float64}          # Predicted measurement

    # Residual
    r::SVector{8,Float64}               # Raw residual (z - z_pred)
    r_w::SVector{8,Float64}             # Whitened residual L⁻¹ r

    # Covariances
    R::SMatrix{8,8,Float64,64}          # Measurement noise
    HPHt::SMatrix{8,8,Float64,64}       # State uncertainty in measurement space
    Q_model::SMatrix{8,8,Float64,64}    # Model mismatch
    S::SMatrix{8,8,Float64,64}          # Total innovation covariance

    # Statistics
    NIS::Float64                        # Normalized Innovation Squared = rᵀ S⁻¹ r
    NIS_B::Float64                      # NIS contribution from B (field)
    NIS_G::Float64                      # NIS contribution from G (gradient)

    # Gating decision
    chi2_threshold::Float64             # Threshold used
    gate_decision::Symbol               # :accept, :reject_mild, :reject_strong

    # State at update time
    pos_est::SVector{3,Float64}         # Position estimate
    pos_true::SVector{3,Float64}        # True position (if available)
    NEES_pos::Float64                   # Position NEES
end

"""
    compute_NIS(r::SVector{8}, S::D8Covariance) -> (NIS::Float64, NIS_B::Float64, NIS_G::Float64)

Compute Normalized Innovation Squared with block decomposition.
"""
function compute_NIS(r::SVector{8,Float64}, S::D8Covariance)
    # Total NIS
    S_mat = S.Σ
    NIS = r' * (S_mat \ r)

    # Block decomposition (approximate - assumes block diagonal dominance)
    r_B = r[1:3]
    r_G = r[4:8]
    S_BB = S_mat[1:3, 1:3]
    S_GG = S_mat[4:8, 4:8]

    NIS_B = r_B' * (S_BB \ r_B)
    NIS_G = r_G' * (S_GG \ r_G)

    return (NIS, NIS_B, NIS_G)
end

"""
    whiten_residual(r::SVector{8}, S::D8Covariance) -> SVector{8}

Compute whitened residual r_w = L⁻¹ r where S = L Lᵀ (Cholesky).
"""
function whiten_residual(r::SVector{8,Float64}, S::D8Covariance)
    L = cholesky(Symmetric(S.Σ)).L
    r_w = L \ r
    return SVector{8}(r_w)
end

# =============================================================================
# Validation
# =============================================================================

"""
    validate_d8_ordering(z::AbstractVector) -> Bool

Validate that a vector conforms to d=8 ordering expectations.
Checks that values are within physically reasonable ranges.
"""
function validate_d8_ordering(z::AbstractVector)
    @assert length(z) == 8 "d=8 measurement must have exactly 8 components"

    # Check field magnitudes (should be ~μT range)
    B_mag = norm(z[1:3])
    if B_mag < 1e-9 || B_mag > 1e-3
        @warn "Field magnitude $B_mag T outside typical range [1nT, 1mT]"
        return false
    end

    # Check gradient magnitudes (should be ~nT/m to μT/m range)
    G_vals = z[4:8]
    G_max = maximum(abs.(G_vals))
    if G_max > 1e-3
        @warn "Gradient magnitude $G_max T/m unusually large"
        return false
    end

    return true
end

end # module D8Contract
