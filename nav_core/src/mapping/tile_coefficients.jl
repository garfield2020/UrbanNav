# ============================================================================
# Tile Coefficients - Harmonic basis for magnetic field representation
# ============================================================================
#
# Ported from AUV-Navigation/src/map_estimation.jl
#
# Each tile stores coefficients α_t for harmonic basis functions φᵢ:
#   Φ(x) = Σᵢ αᵢ φᵢ(x_local)
#   B(x) = -∇Φ
#
# Basis functions satisfy Laplace's equation ∇²φ = 0 (Maxwell).
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Harmonic Basis Functions with Gradients and Hessians
# ============================================================================

"""
    get_harmonic_basis(max_order::Int)

Get harmonic basis functions with their gradients up to specified order.

Returns: Vector of (φ, ∇φ) tuples where:
- φ(x) returns scalar potential value
- ∇φ(x) returns Vec3 gradient

Number of terms: 1 + 3 + 5 + 7 = 16 for order 3.
"""
function get_harmonic_basis(max_order::Int)
    basis = Tuple{Function, Function}[]

    for order in 0:max_order
        for (φ, ∇φ) in _harmonic_basis_order_with_grad(order)
            push!(basis, (φ, ∇φ))
        end
    end

    return basis
end

"""
    get_harmonic_basis_with_hessian(max_order::Int)

Get harmonic basis functions with gradients and Hessians for d=8 support.

Returns: Vector of (φ, ∇φ, Hφ) tuples where:
- φ(x) returns scalar potential value
- ∇φ(x) returns Vec3 gradient
- Hφ(x) returns 3×3 Hessian matrix

The Hessian gives ∂²φ/∂xᵢ∂xⱼ, needed for gradient tensor evaluation.
"""
function get_harmonic_basis_with_hessian(max_order::Int)
    basis = Tuple{Function, Function, Function}[]

    # Order 0: constant
    if max_order >= 0
        φ = p -> 1.0
        ∇φ = p -> Vec3(0.0, 0.0, 0.0)
        Hφ = p -> @SMatrix zeros(3, 3)
        push!(basis, (φ, ∇φ, Hφ))
    end

    # Order 1: linear
    if max_order >= 1
        # x
        push!(basis, (
            p -> p[1],
            p -> Vec3(1.0, 0.0, 0.0),
            p -> @SMatrix zeros(3, 3)
        ))
        # y
        push!(basis, (
            p -> p[2],
            p -> Vec3(0.0, 1.0, 0.0),
            p -> @SMatrix zeros(3, 3)
        ))
        # z
        push!(basis, (
            p -> p[3],
            p -> Vec3(0.0, 0.0, 1.0),
            p -> @SMatrix zeros(3, 3)
        ))
    end

    # Order 2: quadratic
    if max_order >= 2
        # x² - y²
        push!(basis, (
            p -> p[1]^2 - p[2]^2,
            p -> Vec3(2*p[1], -2*p[2], 0.0),
            p -> @SMatrix [2.0 0.0 0.0; 0.0 -2.0 0.0; 0.0 0.0 0.0]
        ))
        # x² - z²
        push!(basis, (
            p -> p[1]^2 - p[3]^2,
            p -> Vec3(2*p[1], 0.0, -2*p[3]),
            p -> @SMatrix [2.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 -2.0]
        ))
        # xy
        push!(basis, (
            p -> p[1]*p[2],
            p -> Vec3(p[2], p[1], 0.0),
            p -> @SMatrix [0.0 1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]
        ))
        # xz
        push!(basis, (
            p -> p[1]*p[3],
            p -> Vec3(p[3], 0.0, p[1]),
            p -> @SMatrix [0.0 0.0 1.0; 0.0 0.0 0.0; 1.0 0.0 0.0]
        ))
        # yz
        push!(basis, (
            p -> p[2]*p[3],
            p -> Vec3(0.0, p[3], p[2]),
            p -> @SMatrix [0.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0]
        ))
    end

    # Order 3: cubic
    if max_order >= 3
        # x³ - 3xy²
        push!(basis, (
            p -> p[1]^3 - 3*p[1]*p[2]^2,
            p -> Vec3(3*p[1]^2 - 3*p[2]^2, -6*p[1]*p[2], 0.0),
            p -> @SMatrix [6*p[1] -6*p[2] 0.0; -6*p[2] -6*p[1] 0.0; 0.0 0.0 0.0]
        ))
        # y³ - 3yx²
        push!(basis, (
            p -> p[2]^3 - 3*p[2]*p[1]^2,
            p -> Vec3(-6*p[1]*p[2], 3*p[2]^2 - 3*p[1]^2, 0.0),
            p -> @SMatrix [-6*p[2] -6*p[1] 0.0; -6*p[1] 6*p[2] 0.0; 0.0 0.0 0.0]
        ))
        # x³ - 3xz²
        push!(basis, (
            p -> p[1]^3 - 3*p[1]*p[3]^2,
            p -> Vec3(3*p[1]^2 - 3*p[3]^2, 0.0, -6*p[1]*p[3]),
            p -> @SMatrix [6*p[1] 0.0 -6*p[3]; 0.0 0.0 0.0; -6*p[3] 0.0 -6*p[1]]
        ))
        # z³ - 3zx²
        push!(basis, (
            p -> p[3]^3 - 3*p[3]*p[1]^2,
            p -> Vec3(-6*p[1]*p[3], 0.0, 3*p[3]^2 - 3*p[1]^2),
            p -> @SMatrix [-6*p[3] 0.0 -6*p[1]; 0.0 0.0 0.0; -6*p[1] 0.0 6*p[3]]
        ))
        # x²y - yz²
        push!(basis, (
            p -> p[1]^2*p[2] - p[2]*p[3]^2,
            p -> Vec3(2*p[1]*p[2], p[1]^2 - p[3]^2, -2*p[2]*p[3]),
            p -> @SMatrix [2*p[2] 2*p[1] 0.0; 2*p[1] 0.0 -2*p[3]; 0.0 -2*p[3] -2*p[2]]
        ))
        # x²z - zy²
        push!(basis, (
            p -> p[1]^2*p[3] - p[3]*p[2]^2,
            p -> Vec3(2*p[1]*p[3], -2*p[2]*p[3], p[1]^2 - p[2]^2),
            p -> @SMatrix [2*p[3] 0.0 2*p[1]; 0.0 -2*p[3] -2*p[2]; 2*p[1] -2*p[2] 0.0]
        ))
        # xyz
        push!(basis, (
            p -> p[1]*p[2]*p[3],
            p -> Vec3(p[2]*p[3], p[1]*p[3], p[1]*p[2]),
            p -> @SMatrix [0.0 p[3] p[2]; p[3] 0.0 p[1]; p[2] p[1] 0.0]
        ))
    end

    return basis
end

"""
    pack_gradient_tensor(G::AbstractMatrix) -> SVector{5}

Pack 3×3 gradient tensor to 5 independent components.

Components: [Gxx, Gyy, Gxy, Gxz, Gyz]
Note: Gzz = -(Gxx + Gyy) from traceless constraint (∇·B = 0).
"""
function pack_gradient_tensor(G::AbstractMatrix)
    return SVector{5, Float64}(G[1,1], G[2,2], G[1,2], G[1,3], G[2,3])
end

"""
    unpack_gradient_tensor(G5::AbstractVector) -> SMatrix{3,3}

Unpack 5 components to full 3×3 symmetric traceless tensor.
"""
function unpack_gradient_tensor(G5::AbstractVector)
    Gxx, Gyy, Gxy, Gxz, Gyz = G5[1], G5[2], G5[3], G5[4], G5[5]
    Gzz = -(Gxx + Gyy)  # Traceless constraint

    return @SMatrix [
        Gxx  Gxy  Gxz;
        Gxy  Gyy  Gyz;
        Gxz  Gyz  Gzz
    ]
end

# ============================================================================
# Tile Coefficients Structure
# ============================================================================

"""
    TileCoefficients

Coefficients for harmonic basis functions in a single tile.

# Fields
- `id::Int`: Unique tile identifier
- `center::Vec3`: Tile center position in world frame
- `size::Float64`: Tile size (meters)
- `coefficients::Vector{Float64}`: Basis function coefficients α_t
- `covariance::Matrix{Float64}`: Coefficient covariance
- `max_order::Int`: Maximum harmonic order
"""
mutable struct TileCoefficients
    id::Int
    center::Vec3
    size::Float64
    coefficients::Vector{Float64}
    covariance::Matrix{Float64}
    max_order::Int
end

"""
    TileCoefficients(; center, size, max_order, id)

Create tile with default (zero) coefficients.
"""
function TileCoefficients(;
    center::AbstractVector = [0.0, 0.0, 0.0],
    size::Real = 50.0,
    max_order::Int = 3,
    id::Int = 1
)
    # Number of coefficients: 1 + 3 + 5 + 7 = 16 for order 3
    n_coeffs = sum(2*k + 1 for k in 0:max_order)

    TileCoefficients(
        id,
        Vec3(center...),
        Float64(size),
        zeros(n_coeffs),
        Matrix{Float64}(I(n_coeffs) * 1e-6),  # Small initial covariance
        max_order
    )
end

"""Number of coefficients in tile."""
n_coefficients(tile::TileCoefficients) = length(tile.coefficients)

"""Convert world position to tile-local coordinates."""
function local_position(tile::TileCoefficients, world_pos::AbstractVector)
    Vec3(world_pos...) - tile.center
end

"""Check if position is within tile bounds."""
function in_tile(tile::TileCoefficients, world_pos::AbstractVector)
    local_pos = local_position(tile, world_pos)
    half_size = tile.size / 2
    return abs(local_pos[1]) <= half_size &&
           abs(local_pos[2]) <= half_size &&
           abs(local_pos[3]) <= half_size
end

# ============================================================================
# Field Evaluation
# ============================================================================

"""
    evaluate_tile_field(tile::TileCoefficients, world_pos::AbstractVector)

Evaluate magnetic field B at world position using tile coefficients.

B = -∇Φ where Φ = Σᵢ αᵢ φᵢ(x_local)
"""
function evaluate_tile_field(tile::TileCoefficients, world_pos::AbstractVector)
    local_pos = local_position(tile, world_pos)
    basis = get_harmonic_basis(tile.max_order)

    B = Vec3(0.0, 0.0, 0.0)
    for (i, (φ, ∇φ)) in enumerate(basis)
        if i <= length(tile.coefficients)
            # B = -∇Φ, so B contribution = -α × ∇φ
            B = B - tile.coefficients[i] * ∇φ(local_pos)
        end
    end

    return B
end

"""
    evaluate_tile_field_jacobian(tile::TileCoefficients, world_pos::AbstractVector)

Compute Jacobian ∂B/∂α of field with respect to coefficients.

Returns 3 × n_coeffs matrix.
"""
function evaluate_tile_field_jacobian(tile::TileCoefficients, world_pos::AbstractVector)
    local_pos = local_position(tile, world_pos)
    basis = get_harmonic_basis(tile.max_order)

    n_coeffs = length(tile.coefficients)
    J = zeros(3, n_coeffs)

    for (i, (φ, ∇φ)) in enumerate(basis)
        if i <= n_coeffs
            # B = -∇Φ, so ∂B/∂αᵢ = -∇φᵢ
            grad = ∇φ(local_pos)
            J[1, i] = -grad[1]
            J[2, i] = -grad[2]
            J[3, i] = -grad[3]
        end
    end

    return J
end

# ============================================================================
# Gradient Tensor Evaluation (d=8 Full Tensor Support)
# ============================================================================

"""
    evaluate_tile_gradient(tile::TileCoefficients, world_pos::AbstractVector)

Evaluate gradient tensor G at world position using tile coefficients.

G = ∂B/∂x = -∇²Φ = -Σᵢ αᵢ Hφᵢ(x_local)

Returns G5: 5-component packed tensor [Gxx, Gyy, Gxy, Gxz, Gyz]
(Gzz = -(Gxx + Gyy) from traceless constraint)
"""
function evaluate_tile_gradient(tile::TileCoefficients, world_pos::AbstractVector)
    local_pos = local_position(tile, world_pos)
    basis = get_harmonic_basis_with_hessian(tile.max_order)

    G = @SMatrix zeros(3, 3)
    for (i, (φ, ∇φ, Hφ)) in enumerate(basis)
        if i <= length(tile.coefficients)
            # G = -∇²Φ, so G contribution = -α × Hφ
            G = G - tile.coefficients[i] * Hφ(local_pos)
        end
    end

    return pack_gradient_tensor(G)
end

"""
    evaluate_tile_gradient_jacobian(tile::TileCoefficients, world_pos::AbstractVector)

Compute Jacobian ∂G/∂α of gradient tensor with respect to coefficients.

Returns 5 × n_coeffs matrix where each column is the packed Hessian.
"""
function evaluate_tile_gradient_jacobian(tile::TileCoefficients, world_pos::AbstractVector)
    local_pos = local_position(tile, world_pos)
    basis = get_harmonic_basis_with_hessian(tile.max_order)

    n_coeffs = length(tile.coefficients)
    J = zeros(5, n_coeffs)

    for (i, (φ, ∇φ, Hφ)) in enumerate(basis)
        if i <= n_coeffs
            # G = -∇²Φ, so ∂G/∂αᵢ = -Hφᵢ (packed)
            H = Hφ(local_pos)
            J[1, i] = -H[1, 1]  # Gxx
            J[2, i] = -H[2, 2]  # Gyy
            J[3, i] = -H[1, 2]  # Gxy
            J[4, i] = -H[1, 3]  # Gxz
            J[5, i] = -H[2, 3]  # Gyz
        end
    end

    return J
end

"""
    evaluate_tile_field_and_gradient(tile::TileCoefficients, world_pos::AbstractVector)

Evaluate both field B and gradient tensor G at world position.

Returns (B, G5) where B ∈ ℝ³ and G5 ∈ ℝ⁵.
"""
function evaluate_tile_field_and_gradient(tile::TileCoefficients, world_pos::AbstractVector)
    local_pos = local_position(tile, world_pos)
    basis = get_harmonic_basis_with_hessian(tile.max_order)

    B = Vec3(0.0, 0.0, 0.0)
    G = @SMatrix zeros(3, 3)

    for (i, (φ, ∇φ, Hφ)) in enumerate(basis)
        if i <= length(tile.coefficients)
            α = tile.coefficients[i]
            # B = -∇Φ
            B = B - α * ∇φ(local_pos)
            # G = -∇²Φ
            G = G - α * Hφ(local_pos)
        end
    end

    return B, pack_gradient_tensor(G)
end

# ============================================================================
# Exports
# ============================================================================

export get_harmonic_basis, get_harmonic_basis_with_hessian
export pack_gradient_tensor, unpack_gradient_tensor
export TileCoefficients, n_coefficients, local_position, in_tile
export evaluate_tile_field, evaluate_tile_field_jacobian
export evaluate_tile_gradient, evaluate_tile_gradient_jacobian
export evaluate_tile_field_and_gradient
