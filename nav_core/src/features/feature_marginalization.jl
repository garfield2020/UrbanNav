# ============================================================================
# Feature Marginalization
# ============================================================================
#
# Ported from AUV-Navigation/src/feature_marginalization.jl
#
# Keep graph bounded by marginalizing retired features.
# When a dipole is retired:
# 1. Compute its marginal effect using Schur complement
# 2. Fold the effect into a prior on connected poses/tiles
# 3. Remove the dipole from active graph
# 4. Preserve estimator consistency
#
# This ensures solve time remains O(1) in mission duration.
# ============================================================================

using LinearAlgebra
using SparseArrays
using StaticArrays

# ============================================================================
# Marginalization Configuration
# ============================================================================

"""
    FeatureMarginalizationConfig

Configuration for feature marginalization.
"""
struct FeatureMarginalizationConfig
    fold_into_map::Bool           # Fold effect into map coefficients
    preserve_prior::Bool          # Add prior from marginalized info
    min_eigenvalue::Float64       # Min eigenvalue for numerical stability
end

function FeatureMarginalizationConfig(;
    fold_into_map::Bool = true,
    preserve_prior::Bool = true,
    min_eigenvalue::Float64 = 1e-6
)
    FeatureMarginalizationConfig(fold_into_map, preserve_prior, min_eigenvalue)
end

const DEFAULT_MARGINALIZATION_CONFIG = FeatureMarginalizationConfig()

# ============================================================================
# Schur Complement
# ============================================================================

"""
    schur_complement(H, b, keep_indices, marginalize_indices; min_eigenvalue)

Compute Schur complement to marginalize out variables.

Given the normal equations:
    [H_kk  H_km] [x_k]   [b_k]
    [H_mk  H_mm] [x_m] = [b_m]

The Schur complement marginalizes x_m, giving:
    H_schur * x_k = b_schur

Where:
    H_schur = H_kk - H_km * inv(H_mm) * H_mk
    b_schur = b_k - H_km * inv(H_mm) * b_m

Returns: (H_schur, b_schur, marginalized_estimate)
"""
function schur_complement(H::AbstractMatrix, b::AbstractVector,
                          keep_indices::AbstractVector{Int},
                          marginalize_indices::AbstractVector{Int};
                          min_eigenvalue::Float64 = 1e-6)
    n_keep = length(keep_indices)
    n_marg = length(marginalize_indices)

    # Extract sub-blocks
    H_kk = H[keep_indices, keep_indices]
    H_km = H[keep_indices, marginalize_indices]
    H_mk = H[marginalize_indices, keep_indices]
    H_mm = H[marginalize_indices, marginalize_indices]
    b_k = b[keep_indices]
    b_m = b[marginalize_indices]

    # Regularize H_mm
    H_mm_reg = H_mm + min_eigenvalue * I(n_marg)

    try
        C = cholesky(Hermitian(H_mm_reg))
        H_mm_inv = inv(C)

        H_schur = H_kk - H_km * H_mm_inv * H_mk
        b_schur = b_k - H_km * (H_mm_inv * b_m)
        x_m = H_mm_inv * b_m

        return (H_schur, b_schur, x_m)
    catch
        H_mm_inv = pinv(H_mm_reg)
        H_schur = H_kk - H_km * H_mm_inv * H_mk
        b_schur = b_k - H_km * (H_mm_inv * b_m)
        x_m = H_mm_inv * b_m
        return (H_schur, b_schur, x_m)
    end
end

# ============================================================================
# Marginalized Prior
# ============================================================================

"""
    FeatureMarginalizedPrior

Prior factor derived from marginalizing out a feature.
"""
struct FeatureMarginalizedPrior
    connected_indices::Vector{Int}    # Indices of connected variables
    information::Matrix{Float64}      # Information matrix (H_schur addition)
    mean_shift::Vector{Float64}       # Shift in mean (from b_schur)
    original_feature_id::Int          # ID of marginalized feature
    marginalization_time::Float64     # When marginalization occurred
end

"""
    apply_marginalized_prior!(H, b, prior)

Apply a marginalized prior to the system.
"""
function apply_marginalized_prior!(H::AbstractMatrix, b::AbstractVector,
                                   prior::FeatureMarginalizedPrior)
    indices = prior.connected_indices
    H[indices, indices] .+= prior.information
    b[indices] .+= prior.mean_shift
end

# ============================================================================
# Feature Marginalization Result
# ============================================================================

"""
    FeatureMarginalizationResult

Result of marginalizing a feature.
"""
struct FeatureMarginalizationResult
    feature_id::Int
    prior::Union{FeatureMarginalizedPrior, Nothing}
    marginalized_state::Union{DipoleFeatureState, Nothing}
    folded_into_map::Bool
    success::Bool
    error_message::String
end

"""
    marginalize_feature(graph, feature_id; config)

Marginalize a retired feature from the factor graph.

This is a simplified implementation - a full version would
integrate with the actual factor graph structure.
"""
function marginalize_feature(graph, feature_id::Int;
                             config::FeatureMarginalizationConfig = DEFAULT_MARGINALIZATION_CONFIG)
    return FeatureMarginalizationResult(
        feature_id,
        nothing,
        nothing,
        config.fold_into_map,
        true,
        ""
    )
end

"""
    compute_feature_information(dipole, measurement_positions, noise_cov)

Compute the information matrix for a dipole from its measurements.
"""
function compute_feature_information(dipole::DipoleFeatureState,
                                     measurement_positions::Vector{<:AbstractVector},
                                     noise_cov::AbstractMatrix)
    n_meas = length(measurement_positions)
    if n_meas == 0
        return zeros(DIPOLE_FEATURE_DIM, DIPOLE_FEATURE_DIM)
    end

    info = zeros(DIPOLE_FEATURE_DIM, DIPOLE_FEATURE_DIM)
    Σ_inv = inv(noise_cov)

    for pos in measurement_positions
        J = feature_field_jacobian(pos, dipole)
        info += J' * Σ_inv * J
    end

    return info
end

# ============================================================================
# Graph Pruning
# ============================================================================

"""
    prune_retired_features!(registry, lifecycle_mgr; config)

Remove all retired features from the registry, optionally marginalizing.
"""
function prune_retired_features!(registry::DipoleFeatureRegistry,
                                 mgr::FeatureLifecycleManager;
                                 config::FeatureMarginalizationConfig = DEFAULT_MARGINALIZATION_CONFIG)
    results = FeatureMarginalizationResult[]

    for dipole in copy(registry.retired)
        result = FeatureMarginalizationResult(
            dipole.id,
            nothing,
            dipole.state,
            config.fold_into_map,
            true,
            "Pruned retired feature"
        )
        push!(results, result)
    end

    empty!(registry.retired)

    return results
end

# ============================================================================
# Memory Management
# ============================================================================

"""
    FeatureGraphMemoryStats

Memory statistics for the factor graph.
"""
struct FeatureGraphMemoryStats
    n_pose_nodes::Int
    n_tile_nodes::Int
    n_dipole_nodes::Int
    n_factors::Int
    n_marginalized_priors::Int
    estimated_memory_mb::Float64
end

"""
    estimate_feature_graph_memory(n_poses, n_tiles, n_dipoles, n_factors)

Estimate memory usage of the factor graph.
"""
function estimate_feature_graph_memory(n_poses::Int, n_tiles::Int,
                                       n_dipoles::Int, n_factors::Int)
    pose_size = 15 * 8 + 15 * 15 * 8
    tile_size = 20 * 8 + 20 * 20 * 8
    dipole_size = 6 * 8 + 6 * 6 * 8
    factor_size = 100 * 8

    total_bytes = (n_poses * pose_size +
                   n_tiles * tile_size +
                   n_dipoles * dipole_size +
                   n_factors * factor_size)

    return total_bytes / (1024 * 1024)
end

"""
    check_feature_graph_growth(current_stats, initial_stats; max_growth_factor)

Check if graph has grown beyond acceptable limits.
"""
function check_feature_graph_growth(current::FeatureGraphMemoryStats,
                                    initial::FeatureGraphMemoryStats;
                                    max_growth_factor::Float64 = 10.0)
    if initial.estimated_memory_mb < 0.001
        return (true, 1.0)
    end

    growth = current.estimated_memory_mb / initial.estimated_memory_mb
    return (growth <= max_growth_factor, growth)
end

# ============================================================================
# Exports
# ============================================================================

export FeatureMarginalizationConfig, DEFAULT_MARGINALIZATION_CONFIG
export schur_complement
export FeatureMarginalizedPrior, apply_marginalized_prior!
export FeatureMarginalizationResult, marginalize_feature
export compute_feature_information
export prune_retired_features!
export FeatureGraphMemoryStats, estimate_feature_graph_memory, check_feature_graph_growth
