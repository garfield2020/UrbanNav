# ============================================================================
# Sigma Total - Full Innovation Covariance
# ============================================================================
#
# Ported from AUV-Navigation/src/sigma_total.jl
#
# SINGLE implementation of Σ_total (full innovation covariance).
# All code paths computing γ, gating, or confidence MUST use this module.
#
# Formula:
#     Σ_total = Σ_meas + J_p × P_pos × J_p' + J_α × Σ_α × J_α'
#
# Where:
# - Σ_meas = Measurement noise covariance
# - J_p = ∂B/∂p = Field-position Jacobian
# - P_pos = Position covariance from navigation
# - J_α = ∂B/∂α = Field-coefficient Jacobian
# - Σ_α = Tile coefficient covariance
# ============================================================================

using LinearAlgebra
using StaticArrays

export SigmaTotalConfig, DEFAULT_SIGMA_TOTAL_CONFIG
export compute_sigma_meas, compute_sigma_total, compute_sigma_total_simple
export compute_Q_model
export field_position_jacobian, field_coefficient_jacobian
export ensure_spd, log_sigma_components

# ============================================================================
# Configuration
# ============================================================================

"""
    SigmaTotalConfig

Configuration for Σ_total computation.

# Fields
- `noise_B::Float64`: Magnetic field noise std (Tesla)
- `noise_G::Float64`: Gradient noise std (Tesla/m), for d=8
- `Q_model_B::Float64`: Model mismatch noise for field (Tesla²)
- `Q_model_G::Float64`: Model mismatch noise for gradient (T²/m²)
- `regularization_eps::Float64`: Minimum eigenvalue for SPD
- `include_pose_uncertainty::Bool`: Whether to include Σ_pose term
- `include_map_uncertainty::Bool`: Whether to include Σ_map term
- `include_model_mismatch::Bool`: Whether to include Q_model term
- `use_numeric_jacobians::Bool`: Use finite differences
- `jacobian_delta::Float64`: Step size for numeric Jacobians

# The Q_model Term

Q_model represents "truth-model mismatch" - the uncertainty arising from:
1. Unmodeled field complexity (dipoles not in the map)
2. Temporal variations (tidal, diurnal)
3. Sensor calibration drift
4. Numerical discretization errors

This term is CRITICAL for proper NEES calibration. Without it, the filter
becomes overconfident and χ² statistics are consistently too large.

The full innovation covariance formula is:
    S = H P Hᵀ + R + Q_model

Where:
- H P Hᵀ: State uncertainty propagated to measurement space
- R: Measurement noise (sensor uncertainty)
- Q_model: Truth-model mismatch (world model uncertainty)
"""
struct SigmaTotalConfig
    noise_B::Float64
    noise_G::Float64
    Q_model_B::Float64
    Q_model_G::Float64
    regularization_eps::Float64
    include_pose_uncertainty::Bool
    include_map_uncertainty::Bool
    include_model_mismatch::Bool
    use_numeric_jacobians::Bool
    jacobian_delta::Float64
end

function SigmaTotalConfig(;
    noise_B::Float64 = 5e-9,
    noise_G::Float64 = 5e-9,
    Q_model_B::Float64 = 1e-9^2,   # ~1 nT model uncertainty
    Q_model_G::Float64 = 1e-9^2,   # ~1 nT/m model uncertainty
    regularization_eps::Float64 = 1e-20,
    include_pose_uncertainty::Bool = true,
    include_map_uncertainty::Bool = true,
    include_model_mismatch::Bool = true,
    use_numeric_jacobians::Bool = false,
    jacobian_delta::Float64 = 1e-6
)
    SigmaTotalConfig(
        noise_B, noise_G, Q_model_B, Q_model_G, regularization_eps,
        include_pose_uncertainty, include_map_uncertainty, include_model_mismatch,
        use_numeric_jacobians, jacobian_delta
    )
end

const DEFAULT_SIGMA_TOTAL_CONFIG = SigmaTotalConfig()

# ============================================================================
# Measurement Covariance
# ============================================================================

"""
    compute_sigma_meas(d::Int, config::SigmaTotalConfig) -> Matrix

Compute measurement covariance for dimension d.

d=3: B-only measurement [Bx, By, Bz]
d=8: B + ∇B measurement [Bx, By, Bz, Gxx, Gyy, Gxy, Gxz, Gyz]
"""
function compute_sigma_meas(d::Int, config::SigmaTotalConfig = DEFAULT_SIGMA_TOTAL_CONFIG)
    if d == 3
        return Diagonal(fill(config.noise_B^2, 3))
    elseif d == 8
        variances = vcat(
            fill(config.noise_B^2, 3),
            fill(config.noise_G^2, 5)
        )
        return Diagonal(variances)
    else
        error("Unsupported measurement dimension d=$d. Supported: 3, 8")
    end
end

# ============================================================================
# Jacobians
# ============================================================================

"""
    field_position_jacobian(position::Vec3, tile, config; d=3) -> Matrix

Compute ∂B/∂p: how field changes with position.
Returns d×3 matrix.
"""
function field_position_jacobian(
    position::Vec3,
    tile,
    config::SigmaTotalConfig = DEFAULT_SIGMA_TOTAL_CONFIG;
    d::Int = 3
)
    if tile === nothing
        return zeros(d, 3)
    end

    if config.use_numeric_jacobians
        return _numeric_position_jacobian(position, tile, config, d)
    else
        return _analytic_position_jacobian(position, tile, d)
    end
end

function _numeric_position_jacobian(position::Vec3, tile, config::SigmaTotalConfig, d::Int)
    δ = config.jacobian_delta
    J = zeros(d, 3)

    B0 = _evaluate_field_safe(tile, position)
    if d == 3
        for i in 1:3
            p_plus = position + δ * _unit_vec(i)
            B_plus = _evaluate_field_safe(tile, p_plus)
            J[:, i] = (B_plus - B0) / δ
        end
    else
        z0 = vcat(B0, _evaluate_gradient_flat_safe(tile, position))
        for i in 1:3
            p_plus = position + δ * _unit_vec(i)
            B_plus = _evaluate_field_safe(tile, p_plus)
            G_plus = _evaluate_gradient_flat_safe(tile, p_plus)
            z_plus = vcat(B_plus, G_plus)
            J[:, i] = (z_plus - z0) / δ
        end
    end

    return J
end

function _analytic_position_jacobian(position::Vec3, tile, d::Int)
    # Fallback to numeric for now
    return _numeric_position_jacobian(position, tile, DEFAULT_SIGMA_TOTAL_CONFIG, d)
end

function _unit_vec(i::Int)
    v = zeros(3)
    v[i] = 1.0
    return Vec3(v...)
end

"""
    field_coefficient_jacobian(position::Vec3, tile, config; d=3) -> Matrix

Compute ∂B/∂α: how field changes with tile coefficients.
Returns d×n_coeffs matrix.
"""
function field_coefficient_jacobian(
    position::Vec3,
    tile,
    config::SigmaTotalConfig = DEFAULT_SIGMA_TOTAL_CONFIG;
    d::Int = 3
)
    if tile === nothing
        return zeros(d, 16)
    end

    n_coeffs = _get_n_coefficients_safe(tile)

    if d == 3
        return _evaluate_basis_matrix_safe(tile, position)
    else
        J_field = _evaluate_basis_matrix_safe(tile, position)
        J_grad = _evaluate_gradient_basis_matrix_safe(tile, position)
        return vcat(J_field, J_grad)
    end
end

# ============================================================================
# Σ_total Computation
# ============================================================================

"""
    compute_sigma_total(position::Vec3, P_pos, tile, config; d=3) -> Matrix

Compute full innovation covariance Σ_total.

CRITICAL: This implements the correct innovation covariance formula:
    Σ_total = H P Hᵀ + R + Q_model

Where:
- H P Hᵀ = Σ_pose (position uncertainty in measurement space)
- R = Σ_meas (measurement noise)
- Q_model = model mismatch uncertainty

Missing any term causes NEES miscalibration!

# Arguments
- `position`: Current vehicle position
- `P_pos`: Position covariance (3×3)
- `tile`: Active magnetic tile
- `config`: Configuration
- `d`: Measurement dimension (3 or 8)
"""
function compute_sigma_total(
    position::Vec3,
    P_pos::AbstractMatrix,
    tile,
    config::SigmaTotalConfig = DEFAULT_SIGMA_TOTAL_CONFIG;
    d::Int = 3
)
    # Term 1: Measurement noise R (Σ_meas)
    Σ_total = Matrix(compute_sigma_meas(d, config))

    # Term 2: State uncertainty in measurement space H P Hᵀ (Σ_pose)
    if config.include_pose_uncertainty && !iszero(P_pos)
        J_p = field_position_jacobian(position, tile, config; d=d)
        Σ_pose = J_p * P_pos * J_p'
        Σ_total += Σ_pose
    end

    # Term 3: Map/tile coefficient uncertainty (Σ_map)
    if config.include_map_uncertainty && tile !== nothing
        Σ_α = _get_coefficient_covariance_safe(tile)
        if Σ_α !== nothing && !iszero(Σ_α)
            J_α = field_coefficient_jacobian(position, tile, config; d=d)
            Σ_map = J_α * Σ_α * J_α'
            Σ_total += Σ_map
        end
    end

    # Term 4: Model mismatch Q_model (CRITICAL for NEES calibration)
    if config.include_model_mismatch
        Σ_total += compute_Q_model(d, config)
    end

    Σ_total = ensure_spd(Σ_total, config.regularization_eps)

    return Σ_total
end

"""
    compute_Q_model(d::Int, config::SigmaTotalConfig) -> Matrix

Compute model mismatch covariance Q_model.

This term accounts for:
- Unmodeled field sources (dipoles not in map)
- Temporal field variations
- Map discretization errors
- Sensor calibration drift

Without this term, the filter becomes overconfident.
"""
function compute_Q_model(d::Int, config::SigmaTotalConfig)
    if d == 3
        return Diagonal(fill(config.Q_model_B, 3))
    elseif d == 8
        variances = vcat(
            fill(config.Q_model_B, 3),
            fill(config.Q_model_G, 5)
        )
        return Diagonal(variances)
    else
        error("Unsupported measurement dimension d=$d")
    end
end

"""
    compute_sigma_total_simple(Σ_meas, Σ_pose, Σ_map, Q_model; ε=1e-20) -> Matrix

Simplified interface: directly sum pre-computed covariance components.

CRITICAL: The full formula is:
    Σ_total = Σ_meas + Σ_pose + Σ_map + Q_model
           = R + H P Hᵀ + Jα Pα Jαᵀ + Q_model

All four terms must be included for correct NEES calibration!
"""
function compute_sigma_total_simple(
    Σ_meas::AbstractMatrix,
    Σ_pose::AbstractMatrix = zeros(size(Σ_meas)),
    Σ_map::AbstractMatrix = zeros(size(Σ_meas)),
    Q_model::AbstractMatrix = zeros(size(Σ_meas));
    ε::Float64 = 1e-20
)
    Σ_total = Matrix(Σ_meas) + Matrix(Σ_pose) + Matrix(Σ_map) + Matrix(Q_model)
    return ensure_spd(Σ_total, ε)
end

# ============================================================================
# Helpers
# ============================================================================

"""
    ensure_spd(Σ::AbstractMatrix, ε::Float64) -> Matrix

Ensure matrix is symmetric positive definite.
"""
function ensure_spd(Σ::AbstractMatrix, ε::Float64 = 1e-20)
    Σ_sym = (Σ + Σ') / 2

    λs = eigvals(Symmetric(Σ_sym))
    λ_min = minimum(real.(λs))

    if λ_min < ε
        Σ_sym += (ε - λ_min + 1e-15) * I
    end

    return Matrix(Σ_sym)
end

"""
    log_sigma_components(Σ_meas, Σ_pose, Σ_map, Σ_total)

Log covariance components for debugging.
"""
function log_sigma_components(Σ_meas, Σ_pose, Σ_map, Σ_total)
    @info "Σ_total breakdown" begin
        trace_Σ_meas = tr(Σ_meas)
        trace_Σ_pose = tr(Σ_pose)
        trace_Σ_map = tr(Σ_map)
        trace_Σ_total = tr(Σ_total)
        ratio_pose_to_meas = tr(Σ_pose) / tr(Σ_meas)
        ratio_map_to_meas = tr(Σ_map) / tr(Σ_meas)
    end
end

# ============================================================================
# Safe tile interface (fallbacks for nothing)
# ============================================================================

function _evaluate_field_safe(tile::Nothing, position::Vec3)
    zeros(3)
end

function _evaluate_field_safe(tile, position::Vec3)
    if hasmethod(evaluate_field, Tuple{typeof(tile), Vec3})
        return evaluate_field(tile, position)
    else
        return zeros(3)
    end
end

function _evaluate_gradient_flat_safe(tile::Nothing, position::Vec3)
    zeros(5)
end

function _evaluate_gradient_flat_safe(tile, position::Vec3)
    if hasmethod(evaluate_gradient_flat, Tuple{typeof(tile), Vec3})
        return evaluate_gradient_flat(tile, position)
    else
        return zeros(5)
    end
end

function _get_n_coefficients_safe(tile::Nothing)
    16
end

function _get_n_coefficients_safe(tile)
    if hasmethod(get_n_coefficients, Tuple{typeof(tile)})
        return get_n_coefficients(tile)
    else
        return 16
    end
end

function _evaluate_basis_matrix_safe(tile::Nothing, position::Vec3)
    zeros(3, 16)
end

function _evaluate_basis_matrix_safe(tile, position::Vec3)
    if hasmethod(evaluate_basis_matrix, Tuple{typeof(tile), Vec3})
        return evaluate_basis_matrix(tile, position)
    else
        return zeros(3, 16)
    end
end

function _evaluate_gradient_basis_matrix_safe(tile::Nothing, position::Vec3)
    zeros(5, 16)
end

function _evaluate_gradient_basis_matrix_safe(tile, position::Vec3)
    if hasmethod(evaluate_gradient_basis_matrix, Tuple{typeof(tile), Vec3})
        return evaluate_gradient_basis_matrix(tile, position)
    else
        return zeros(5, 16)
    end
end

function _get_coefficient_covariance_safe(tile::Nothing)
    nothing
end

function _get_coefficient_covariance_safe(tile)
    if hasmethod(get_coefficient_covariance, Tuple{typeof(tile)})
        return get_coefficient_covariance(tile)
    else
        return nothing
    end
end
