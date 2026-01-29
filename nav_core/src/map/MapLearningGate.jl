# ============================================================================
# Map Learning Gate (Phase B)
# ============================================================================
#
# Implements the critical separation between navigation and map learning.
#
# Core Principle: "Don't let bad pose states teach the map"
#
# Navigation Update (position correction):
#   S_nav = H_pose · P_pose · H_pose' + R_sensor + Σ_map
#   This updates the position estimate using map predictions
#
# Map Learning Update (coefficient refinement):
#   S_learn = H_map · P_map · H_map' + R_sensor + Σ_pose_contribution
#   This updates map coefficients using measurements at known positions
#
# The key insight is that:
# - Navigation uses map to correct position → requires map uncertainty Σ_map
# - Map learning uses measurements to correct map → requires pose uncertainty
#
# When pose uncertainty is HIGH:
# - Navigation still proceeds (Σ_map inflates S_nav appropriately)
# - Map learning should be INHIBITED (bad pose would corrupt map)
#
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Residual Types (explicit separation)
# ============================================================================

"""
    NavigationResidual

Residual for navigation (pose) update.

# Fields
- `innovation::Vector{Float64}`: z - h(x̂, map) [T] or [T, T/m]
- `H_pose::Matrix{Float64}`: ∂h/∂x (Jacobian wrt position)
- `S_nav::Matrix{Float64}`: Innovation covariance for navigation
- `measurement_dim::Int`: Dimension of measurement (3 or 8)

# Covariance Composition
S_nav = H_pose · P_pose · H_pose' + R_sensor + Σ_map

where:
- P_pose: Position covariance [m²]
- R_sensor: Sensor measurement covariance [T²] or [T², (T/m)²]
- Σ_map: Map prediction covariance [T²] (from map uncertainty)
"""
struct NavigationResidual
    innovation::Vector{Float64}
    H_pose::Matrix{Float64}
    S_nav::Matrix{Float64}
    measurement_dim::Int
end

"""
    MapLearningResidual

Residual for map coefficient update.

# Fields
- `innovation::Vector{Float64}`: z - h(x̂, map) [T] or [T, T/m]
- `H_map::Matrix{Float64}`: ∂h/∂α (Jacobian wrt map coefficients)
- `R_effective::Matrix{Float64}`: Effective measurement covariance for map learning
- `pose_contribution::Float64`: Pose uncertainty contribution [T²]
- `teachable::Bool`: Whether this residual should be used for learning
- `rejection_reason::Symbol`: Reason if not teachable (:ok, :high_pose_uncertainty, etc.)

# Covariance Composition
R_effective = R_sensor + Σ_pose_to_meas

where:
- R_sensor: Sensor measurement covariance [T²]
- Σ_pose_to_meas: Pose uncertainty projected to measurement space

# Teachability
The residual is teachable if pose uncertainty is small enough that
it doesn't dominate the measurement uncertainty. See MapLearningConfig.
"""
struct MapLearningResidual
    innovation::Vector{Float64}
    H_map::Matrix{Float64}
    R_effective::Matrix{Float64}
    pose_contribution::Float64
    teachable::Bool
    rejection_reason::Symbol
end

# ============================================================================
# Configuration
# ============================================================================

"""
    MapLearningConfig

Configuration for map learning gating.

# Fields
- `max_pose_uncertainty_m2::Float64`: Maximum position variance [m²] for learning
- `max_pose_contribution_ratio::Float64`: Max ratio of pose contribution to sensor noise
- `min_gradient_energy::Float64`: Minimum gradient norm [T/m] for learning

# Threshold Justifications

**max_pose_uncertainty_m2 = 1.0 m²** (σ_pos = 1 m):
For typical gradient |G| = 100 nT/m, pose error δx causes field error δB = |G|δx.
At σ_pos = 1 m: σ_B_pose = 100 nT, comparable to typical sensor noise.
More uncertain poses would dominate measurement error.

**max_pose_contribution_ratio = 0.5**:
Pose uncertainty contribution should be < 50% of total measurement uncertainty.
This ensures sensor noise dominates, not pose uncertainty.
Higher values allow more aggressive learning; lower values are more conservative.

**min_gradient_energy = 20e-9 T/m** (20 nT/m):
Gradient provides position observability. Below this threshold:
- Position updates have high uncertainty
- Learning from uncertain position corrupts map
This is set at 20% of typical gradient (100 nT/m).
"""
struct MapLearningConfig
    max_pose_uncertainty_m2::Float64
    max_pose_contribution_ratio::Float64
    min_gradient_energy::Float64

    function MapLearningConfig(;
        max_pose_uncertainty_m2::Float64 = 1.0,
        max_pose_contribution_ratio::Float64 = 0.5,
        min_gradient_energy::Float64 = 20e-9
    )
        @assert max_pose_uncertainty_m2 > 0
        @assert 0 < max_pose_contribution_ratio <= 1.0
        @assert min_gradient_energy >= 0
        new(max_pose_uncertainty_m2, max_pose_contribution_ratio, min_gradient_energy)
    end
end

const DEFAULT_MAP_LEARNING_CONFIG = MapLearningConfig()

# ============================================================================
# Residual Separation
# ============================================================================

"""
    compute_pose_to_measurement_covariance(P_pose::AbstractMatrix,
                                           G::AbstractMatrix)

Compute how pose uncertainty propagates to measurement space.

For field measurement B = B_map(x), the uncertainty in B due to position uncertainty:

    Σ_B_pose = G · P_pose · G'

where G is the gradient tensor ∂B/∂x [T/m].

# Arguments
- `P_pose`: Position covariance [m²], typically 3×3
- `G`: Gradient tensor [T/m], 3×3

# Returns
- `Σ_B_pose`: Field covariance due to pose uncertainty [T²], 3×3
"""
function compute_pose_to_measurement_covariance(P_pose::AbstractMatrix,
                                                 G::AbstractMatrix)
    return G * P_pose * G'
end

"""
    create_navigation_residual(innovation::AbstractVector,
                               H_pose::AbstractMatrix,
                               P_pose::AbstractMatrix,
                               R_sensor::AbstractMatrix,
                               Σ_map::AbstractMatrix)

Create a navigation residual with properly composed covariance.

# Arguments
- `innovation`: z - h(x̂, map), measurement residual [T] or [T, T/m]
- `H_pose`: Jacobian of measurement wrt position, ∂h/∂x
- `P_pose`: Position covariance [m²]
- `R_sensor`: Sensor measurement covariance [T²]
- `Σ_map`: Map prediction covariance [T²]

# Returns
NavigationResidual with innovation covariance:
    S_nav = H_pose · P_pose · H_pose' + R_sensor + Σ_map
"""
function create_navigation_residual(innovation::AbstractVector,
                                    H_pose::AbstractMatrix,
                                    P_pose::AbstractMatrix,
                                    R_sensor::AbstractMatrix,
                                    Σ_map::AbstractMatrix)
    d = length(innovation)

    # Compose innovation covariance for navigation
    # S_nav includes: position uncertainty, sensor noise, and map uncertainty
    S_nav = H_pose * P_pose * H_pose' + R_sensor + Σ_map

    # Ensure symmetry
    S_nav = (S_nav + S_nav') / 2

    return NavigationResidual(
        Vector(innovation),
        Matrix(H_pose),
        Matrix(S_nav),
        d
    )
end

"""
    create_map_learning_residual(innovation::AbstractVector,
                                 H_map::AbstractMatrix,
                                 R_sensor::AbstractMatrix,
                                 P_pose::AbstractMatrix,
                                 G::AbstractMatrix,
                                 config::MapLearningConfig = DEFAULT_MAP_LEARNING_CONFIG)

Create a map learning residual with teachability gating.

# Arguments
- `innovation`: z - h(x̂, map), measurement residual [T] or [T, T/m]
- `H_map`: Jacobian of measurement wrt map coefficients, ∂h/∂α
- `R_sensor`: Sensor measurement covariance [T²]
- `P_pose`: Position covariance [m²], 3×3
- `G`: Gradient tensor [T/m], 3×3 (for projecting pose uncertainty)
- `config`: Learning configuration

# Returns
MapLearningResidual with:
- Effective measurement covariance including pose contribution
- Teachability flag and rejection reason

# Teachability Criteria
1. Position uncertainty < max_pose_uncertainty_m2
2. Pose contribution < max_pose_contribution_ratio × sensor_noise
3. Gradient energy > min_gradient_energy
"""
function create_map_learning_residual(innovation::AbstractVector,
                                      H_map::AbstractMatrix,
                                      R_sensor::AbstractMatrix,
                                      P_pose::AbstractMatrix,
                                      G::AbstractMatrix,
                                      config::MapLearningConfig = DEFAULT_MAP_LEARNING_CONFIG)
    d = length(innovation)

    # Compute pose contribution to measurement uncertainty
    Σ_pose_meas = compute_pose_to_measurement_covariance(P_pose, G)

    # Total position variance (trace of position block)
    pos_variance = tr(P_pose)

    # Pose contribution to field variance (trace of Σ_pose_meas)
    pose_contribution = tr(Σ_pose_meas)

    # Sensor noise level (trace of R_sensor for field components)
    sensor_noise = tr(R_sensor[1:min(3,d), 1:min(3,d)])

    # Gradient energy
    grad_energy = norm(G)

    # Check teachability criteria
    teachable = true
    rejection_reason = :ok

    # Criterion 1: Position uncertainty
    if pos_variance > config.max_pose_uncertainty_m2
        teachable = false
        rejection_reason = :high_pose_uncertainty
    end

    # Criterion 2: Pose contribution ratio
    if teachable && pose_contribution > config.max_pose_contribution_ratio * sensor_noise
        teachable = false
        rejection_reason = :pose_dominates_sensor
    end

    # Criterion 3: Gradient energy
    if teachable && grad_energy < config.min_gradient_energy
        teachable = false
        rejection_reason = :weak_gradient
    end

    # Effective measurement covariance for map learning
    # Includes pose uncertainty contribution
    R_effective = R_sensor + Σ_pose_meas

    # Ensure symmetry
    R_effective = (R_effective + R_effective') / 2

    return MapLearningResidual(
        Vector(innovation),
        Matrix(H_map),
        Matrix(R_effective),
        pose_contribution,
        teachable,
        rejection_reason
    )
end

# ============================================================================
# Learning Rate Adaptation
# ============================================================================

"""
    compute_learning_weight(P_pose::AbstractMatrix,
                           G::AbstractMatrix,
                           R_sensor::AbstractMatrix,
                           config::MapLearningConfig)

Compute a learning weight ∈ [0, 1] based on pose uncertainty.

# Weight Formula
    weight = 1 - clamp(pose_contribution / (max_ratio × sensor_noise), 0, 1)

When pose contribution is:
- Zero: weight = 1 (full learning)
- Equal to max_ratio × sensor_noise: weight = 0 (no learning)
- Between: linear interpolation

This provides smooth degradation of learning rate as pose uncertainty increases,
rather than a hard cutoff.

# Returns
Learning weight ∈ [0, 1]
"""
function compute_learning_weight(P_pose::AbstractMatrix,
                                 G::AbstractMatrix,
                                 R_sensor::AbstractMatrix,
                                 config::MapLearningConfig)
    # Compute pose contribution
    Σ_pose_meas = compute_pose_to_measurement_covariance(P_pose, G)
    pose_contribution = tr(Σ_pose_meas)

    # Sensor noise level
    d = size(R_sensor, 1)
    sensor_noise = tr(R_sensor[1:min(3,d), 1:min(3,d)])

    # Compute weight
    max_contribution = config.max_pose_contribution_ratio * sensor_noise
    if max_contribution <= 0
        return 0.0
    end

    ratio = pose_contribution / max_contribution
    weight = 1.0 - clamp(ratio, 0.0, 1.0)

    return weight
end

# ============================================================================
# Integration Test Helper
# ============================================================================

"""
    LearningGateStatistics

Statistics for monitoring learning gate behavior.
"""
mutable struct LearningGateStatistics
    total_residuals::Int
    teachable_count::Int
    rejection_counts::Dict{Symbol, Int}
    mean_learning_weight::Float64
    mean_pose_contribution::Float64
end

function LearningGateStatistics()
    LearningGateStatistics(
        0, 0,
        Dict{Symbol, Int}(),
        0.0, 0.0
    )
end

"""
    update_statistics!(stats::LearningGateStatistics, residual::MapLearningResidual, weight::Float64)

Update statistics with a new residual.
"""
function update_statistics!(stats::LearningGateStatistics,
                           residual::MapLearningResidual,
                           weight::Float64)
    n = stats.total_residuals
    stats.total_residuals += 1

    if residual.teachable
        stats.teachable_count += 1
    else
        reason = residual.rejection_reason
        stats.rejection_counts[reason] = get(stats.rejection_counts, reason, 0) + 1
    end

    # Running average of weight and pose contribution
    stats.mean_learning_weight = (n * stats.mean_learning_weight + weight) / (n + 1)
    stats.mean_pose_contribution = (n * stats.mean_pose_contribution + residual.pose_contribution) / (n + 1)
end

"""
    teachability_rate(stats::LearningGateStatistics)

Compute fraction of residuals that were teachable.
"""
function teachability_rate(stats::LearningGateStatistics)
    if stats.total_residuals == 0
        return 1.0
    end
    return stats.teachable_count / stats.total_residuals
end

# ============================================================================
# Exports
# ============================================================================

export NavigationResidual, MapLearningResidual
export MapLearningConfig, DEFAULT_MAP_LEARNING_CONFIG
export compute_pose_to_measurement_covariance
export create_navigation_residual, create_map_learning_residual
export compute_learning_weight
export LearningGateStatistics, update_statistics!, teachability_rate
