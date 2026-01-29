# =============================================================================
# WorldFieldContract.jl - Formal Interface for Magnetic World Models
# =============================================================================
#
# Phase D Step 3: Defines the contract between world generators and consumers.
#
# This contract ensures:
# 1. Consistent API across all world implementations
# 2. Deterministic seeding for reproducibility (INV-05)
# 3. Truth metadata isolation from online learning (INV-07)
# 4. Maxwell consistency validation
#
# CRITICAL: Online learning code must NEVER access truth_metadata().
# This is enforced by test_anticheat.jl grep verification.
# =============================================================================

using LinearAlgebra
using StaticArrays

# =============================================================================
# Type Aliases
# =============================================================================

const Vec3 = SVector{3, Float64}
const Mat3 = SMatrix{3, 3, Float64, 9}

# =============================================================================
# Abstract World Type
# =============================================================================

"""
    AbstractMagneticWorld

Base type for all magnetic world implementations.

Any concrete world must implement:
- `field_at(world, position)::Vec3` - Magnetic field in Tesla
- `gradient_at(world, position)::Mat3` - Field gradient in T/m

Optional implementations:
- `truth_metadata(world)` - Ground truth for validation (INV-07: not for online use)
- `seed_world(world_type, scenario_hash, seed)` - Deterministic construction
"""
abstract type AbstractMagneticWorld end

# =============================================================================
# Required Interface Methods
# =============================================================================

"""
    field_at(world::AbstractMagneticWorld, position::Vec3) -> Vec3

Return the magnetic field vector B at the given position.

# Returns
- `B::Vec3`: Magnetic field in Tesla [T]

# Physics
The field must satisfy Maxwell's equations in source-free regions:
- ∇·B = 0 (no magnetic monopoles)
- ∇×B = 0 (no currents in measurement region)

# Units
- Position: meters [m] in world frame (NED)
- Field: Tesla [T]
"""
function field_at end

"""
    gradient_at(world::AbstractMagneticWorld, position::Vec3) -> Mat3

Return the magnetic field gradient tensor at the given position.

# Returns
- `G::Mat3`: Gradient tensor ∂Bᵢ/∂xⱼ in T/m

# Physics
The gradient tensor has structure due to Maxwell's equations:
- Symmetric: Gᵢⱼ = Gⱼᵢ (curl-free condition)
- Traceless: Gxx + Gyy + Gzz = 0 (divergence-free condition)
- 9 components → 5 independent degrees of freedom

# Units
- Position: meters [m]
- Gradient: Tesla per meter [T/m]
"""
function gradient_at end

# =============================================================================
# Optional Interface Methods
# =============================================================================

"""
    truth_metadata(world::AbstractMagneticWorld) -> WorldMetadata

Return ground truth metadata for validation and analysis.

# CRITICAL SAFETY INVARIANT (INV-07)
This function must NEVER be called by online learning code.
It exists solely for:
- Offline map building (Phase A)
- Validation and testing
- Performance analysis

Violation is detected by test_anticheat.jl grep verification.

# Returns
- `WorldMetadata`: Structure containing source locations, moments, etc.
"""
function truth_metadata end

"""
    seed_world(::Type{W}, scenario_hash::UInt64, seed::Int) where W <: AbstractMagneticWorld

Construct a deterministic world from scenario hash and seed.

# Arguments
- `W`: World type to construct
- `scenario_hash`: Hash of scenario configuration
- `seed`: Random seed for stochastic components

# Returns
- `world::W`: Fully constructed world

# Determinism Requirement (INV-05)
Given identical (scenario_hash, seed), must produce bit-identical world.
This enables reproducible testing and debugging.
"""
function seed_world end

# =============================================================================
# World Metadata Types
# =============================================================================

"""
    WorldMetadata

Ground truth information about world composition.
Used for validation only - NEVER for online navigation.
"""
struct WorldMetadata
    # Background field parameters
    background_intensity_nT::Float64
    background_inclination_deg::Float64
    background_declination_deg::Float64

    # Anomaly information
    n_anomalies::Int
    anomaly_positions::Vector{Vec3}
    anomaly_scales::Vector{Float64}

    # Dipole information
    n_dipoles::Int
    dipole_positions::Vector{Vec3}
    dipole_moments::Vector{Vec3}

    # Field statistics at survey depth
    gradient_mean_nT_m::Float64
    gradient_std_nT_m::Float64
end

"""
    WorldMetadata()

Construct empty metadata (for worlds without truth information).
"""
function WorldMetadata()
    WorldMetadata(
        0.0, 0.0, 0.0,
        0, Vec3[], Float64[],
        0, Vec3[], Vec3[],
        0.0, 0.0
    )
end

# =============================================================================
# Validation Utilities
# =============================================================================

"""
    validate_maxwell_consistency(world::AbstractMagneticWorld, position::Vec3;
                                  h::Float64 = 0.01, tol::Float64 = 1e-10) -> Bool

Verify that the field satisfies Maxwell's equations at the given position.

# Checks
1. Divergence: |∇·B| < tol
2. Gradient symmetry: |Gᵢⱼ - Gⱼᵢ| < tol for all i,j

# Arguments
- `world`: World to validate
- `position`: Point to check
- `h`: Step size for numerical differentiation [m]
- `tol`: Tolerance for Maxwell violation [T/m]

# Returns
- `true` if Maxwell-consistent, `false` otherwise
"""
function validate_maxwell_consistency(
    world::AbstractMagneticWorld,
    position::Vec3;
    h::Float64 = 0.01,
    tol::Float64 = 1e-10
)
    # Compute numerical divergence
    B_xp = field_at(world, position + Vec3(h, 0, 0))
    B_xm = field_at(world, position - Vec3(h, 0, 0))
    B_yp = field_at(world, position + Vec3(0, h, 0))
    B_ym = field_at(world, position - Vec3(0, h, 0))
    B_zp = field_at(world, position + Vec3(0, 0, h))
    B_zm = field_at(world, position - Vec3(0, 0, h))

    dBx_dx = (B_xp[1] - B_xm[1]) / (2h)
    dBy_dy = (B_yp[2] - B_ym[2]) / (2h)
    dBz_dz = (B_zp[3] - B_zm[3]) / (2h)

    divergence = abs(dBx_dx + dBy_dy + dBz_dz)

    if divergence > tol
        return false
    end

    # Check gradient symmetry
    G = gradient_at(world, position)
    for i in 1:3
        for j in (i+1):3
            if abs(G[i,j] - G[j,i]) > tol
                return false
            end
        end
    end

    return true
end

"""
    compute_gradient_statistics(world::AbstractMagneticWorld,
                                 positions::Vector{Vec3}) -> (mean, std)

Compute gradient magnitude statistics over a set of positions.

# Returns
- `mean_grad`: Mean gradient magnitude [T/m]
- `std_grad`: Standard deviation of gradient magnitude [T/m]
"""
function compute_gradient_statistics(
    world::AbstractMagneticWorld,
    positions::Vector{Vec3}
)
    grad_mags = Float64[]

    for pos in positions
        G = gradient_at(world, pos)
        # Frobenius norm as magnitude measure
        grad_mag = norm(G)
        push!(grad_mags, grad_mag)
    end

    return mean(grad_mags), std(grad_mags)
end

# =============================================================================
# Exports
# =============================================================================

export AbstractMagneticWorld
export field_at, gradient_at, truth_metadata, seed_world
export WorldMetadata
export validate_maxwell_consistency, compute_gradient_statistics
