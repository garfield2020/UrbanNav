# ============================================================================
# Feature Disambiguation
# ============================================================================
#
# Ported from AUV-Navigation/src/feature_disambiguation.jl
#
# Handles multiple dipoles in proximity:
# 1. Candidate-to-feature association (Mahalanobis gating)
# 2. Split detection (residuals suggest multiple sources)
# 3. Merge detection (candidates too close to distinguish)
# 4. Rejection of ambiguous assignments
# ============================================================================

using LinearAlgebra
using StaticArrays
using Statistics

# ============================================================================
# Disambiguation Configuration
# ============================================================================

"""
    FeatureDisambiguationConfig

Configuration for feature disambiguation.
"""
struct FeatureDisambiguationConfig
    mahalanobis_gate::Float64      # χ² threshold for association
    min_separation::Float64        # Minimum separation to distinguish (m)
    merge_threshold::Float64       # Distance below which to merge (m)
    split_residual_ratio::Float64  # Residual ratio suggesting split
    ambiguity_threshold::Float64   # Max acceptable ambiguity score
end

function FeatureDisambiguationConfig(;
    mahalanobis_gate::Real = 9.21,   # χ²(3, 0.99)
    min_separation::Real = 2.0,      # 2m minimum
    merge_threshold::Real = 1.0,     # Merge if <1m
    split_residual_ratio::Real = 0.5,
    ambiguity_threshold::Real = 0.7
)
    FeatureDisambiguationConfig(
        Float64(mahalanobis_gate), Float64(min_separation),
        Float64(merge_threshold), Float64(split_residual_ratio),
        Float64(ambiguity_threshold)
    )
end

const DEFAULT_DISAMBIGUATION_CONFIG = FeatureDisambiguationConfig()

# ============================================================================
# Mahalanobis Distance
# ============================================================================

"""
    feature_mahalanobis_distance(x, mean, covariance)

Compute Mahalanobis distance: d = √((x - μ)' Σ⁻¹ (x - μ))
"""
function feature_mahalanobis_distance(x::AbstractVector, mean::AbstractVector,
                                      covariance::AbstractMatrix)
    diff = x - mean
    try
        return sqrt(max(0.0, dot(diff, covariance \ diff)))
    catch
        return sqrt(dot(diff, pinv(covariance) * diff))
    end
end

"""
    feature_mahalanobis_distance_sq(x, mean, covariance)

Squared Mahalanobis distance (avoids sqrt for gating).
"""
function feature_mahalanobis_distance_sq(x::AbstractVector, mean::AbstractVector,
                                         covariance::AbstractMatrix)
    diff = x - mean
    try
        return max(0.0, dot(diff, covariance \ diff))
    catch
        return dot(diff, pinv(covariance) * diff)
    end
end

# ============================================================================
# Association
# ============================================================================

"""
    FeatureAssociationDecision

Result of associating a measurement with features.
"""
@enum FeatureAssociationDecision begin
    ASSOCIATE_CLEAR     # Clear association with single feature
    ASSOCIATE_AMBIGUOUS # Multiple features within gate
    ASSOCIATE_NO_MATCH  # No feature within gate (new candidate)
    ASSOCIATE_SPLIT     # Residual suggests multiple sources
end

"""
    FeatureAssociationResult

Result of measurement-to-feature association.
"""
struct FeatureAssociationResult
    decision::FeatureAssociationDecision
    best_feature_id::Union{Int, Nothing}
    best_distance::Float64
    alternative_ids::Vector{Int}
    alternative_distances::Vector{Float64}
    ambiguity_score::Float64
end

"""
    associate_feature_measurement(measurement_pos, residual, features, config)

Associate a measurement with existing features using Mahalanobis gating.
"""
function associate_feature_measurement(measurement_pos::AbstractVector,
                                       residual::AbstractVector,
                                       features::Dict{Int, DipoleFeatureNode};
                                       config::FeatureDisambiguationConfig = DEFAULT_DISAMBIGUATION_CONFIG)
    if isempty(features)
        return FeatureAssociationResult(ASSOCIATE_NO_MATCH, nothing, Inf, Int[], Float64[], 0.0)
    end

    # Compute distance to each feature
    distances = Tuple{Int, Float64}[]

    for (id, dipole) in features
        pos_cov = dipole.covariance[1:3, 1:3]
        d² = feature_mahalanobis_distance_sq(
            measurement_pos, Vector(dipole.state.position), pos_cov
        )
        push!(distances, (id, d²))
    end

    sort!(distances, by = x -> x[2])

    best_id, best_d² = distances[1]
    best_distance = sqrt(best_d²)

    # Find all features within gate
    within_gate = filter(x -> x[2] < config.mahalanobis_gate^2, distances)

    if isempty(within_gate)
        return FeatureAssociationResult(ASSOCIATE_NO_MATCH, nothing, best_distance,
                                        Int[], Float64[], 0.0)
    end

    if length(within_gate) == 1
        return FeatureAssociationResult(ASSOCIATE_CLEAR, best_id, best_distance,
                                        Int[], Float64[], 0.0)
    end

    # Multiple features within gate
    alt_ids = [x[1] for x in within_gate[2:end]]
    alt_dists = [sqrt(x[2]) for x in within_gate[2:end]]

    second_d = sqrt(within_gate[2][2])
    ambiguity = exp(-abs(second_d - best_distance) / config.min_separation)

    if ambiguity > config.ambiguity_threshold
        return FeatureAssociationResult(ASSOCIATE_AMBIGUOUS, best_id, best_distance,
                                        alt_ids, alt_dists, ambiguity)
    else
        return FeatureAssociationResult(ASSOCIATE_CLEAR, best_id, best_distance,
                                        alt_ids, alt_dists, ambiguity)
    end
end

# ============================================================================
# Split Detection
# ============================================================================

"""
    FeatureSplitResult

Result of checking if a candidate should be split.
"""
struct FeatureSplitResult
    should_split::Bool
    residual_ratio::Float64
    estimated_positions::Vector{SVector{3, Float64}}
    confidence::Float64
end

"""
    detect_feature_split(candidate, existing_fit_residual; config)

Detect if residuals suggest multiple sources should be fit.
"""
function detect_feature_split(candidate::DipoleFeatureCandidate,
                              existing_fit_residual::Float64;
                              config::FeatureDisambiguationConfig = DEFAULT_DISAMBIGUATION_CONFIG)
    n = candidate_support_count(candidate)
    if n < 6
        return FeatureSplitResult(false, 1.0, SVector{3, Float64}[], 0.0)
    end

    positions = candidate.positions
    residuals = candidate.residuals

    # Compute residual magnitudes
    res_mags = [norm(r) for r in residuals]
    mean_res = mean(res_mags)

    # Find high-residual clusters
    high_res_indices = findall(x -> x > 1.5 * mean_res, res_mags)

    if length(high_res_indices) < 3
        return FeatureSplitResult(false, 1.0, SVector{3, Float64}[], 0.0)
    end

    high_res_positions = [positions[i] for i in high_res_indices]

    if length(high_res_positions) >= 4
        centroid1, centroid2, separation = simple_2means_clustering(high_res_positions)

        if separation > config.min_separation
            return FeatureSplitResult(
                true, 0.5,
                [centroid1, centroid2],
                min(1.0, separation / (2 * config.min_separation))
            )
        end
    end

    return FeatureSplitResult(false, 1.0, SVector{3, Float64}[], 0.0)
end

"""
    simple_2means_clustering(points)

Simple 2-means clustering.
"""
function simple_2means_clustering(points::Vector{<:SVector{3}})
    n = length(points)
    if n < 2
        c = points[1]
        return (c, c, 0.0)
    end

    c1 = Vector(points[1])
    c2 = Vector(points[end])

    for iter in 1:10
        cluster1 = SVector{3, Float64}[]
        cluster2 = SVector{3, Float64}[]

        for p in points
            if norm(p - c1) < norm(p - c2)
                push!(cluster1, p)
            else
                push!(cluster2, p)
            end
        end

        if !isempty(cluster1)
            c1 = Vector(sum(cluster1) / length(cluster1))
        end
        if !isempty(cluster2)
            c2 = Vector(sum(cluster2) / length(cluster2))
        end
    end

    separation = norm(c1 - c2)
    return (SVector{3}(c1), SVector{3}(c2), separation)
end

# ============================================================================
# Merge Detection
# ============================================================================

"""
    FeatureMergeResult

Result of checking if two features should be merged.
"""
struct FeatureMergeResult
    should_merge::Bool
    distance::Float64
    merged_state::Union{DipoleFeatureState, Nothing}
    merged_covariance::Union{Matrix{Float64}, Nothing}
end

"""
    detect_feature_merge(dipole1, dipole2; config)

Check if two dipoles are too close and should be merged.
"""
function detect_feature_merge(dipole1::DipoleFeatureNode, dipole2::DipoleFeatureNode;
                              config::FeatureDisambiguationConfig = DEFAULT_DISAMBIGUATION_CONFIG)
    dist = norm(dipole1.state.position - dipole2.state.position)

    if dist > config.merge_threshold
        return FeatureMergeResult(false, dist, nothing, nothing)
    end

    try
        P1 = dipole1.covariance
        P2 = dipole2.covariance

        I1 = inv(P1)
        I2 = inv(P2)

        I_merged = I1 + I2
        P_merged = inv(I_merged)

        x1 = to_state_vector(dipole1.state)
        x2 = to_state_vector(dipole2.state)
        x_merged = P_merged * (I1 * x1 + I2 * x2)

        merged_state = from_state_vector(DipoleFeatureState, x_merged)

        return FeatureMergeResult(true, dist, merged_state, P_merged)
    catch
        pos_avg = (dipole1.state.position + dipole2.state.position) / 2
        mom_avg = (dipole1.state.moment + dipole2.state.moment) / 2
        merged_state = DipoleFeatureState(pos_avg, mom_avg)

        return FeatureMergeResult(true, dist, merged_state, dipole1.covariance)
    end
end

"""
    find_feature_merge_candidates(features; config)

Find all pairs of features that should be merged.
"""
function find_feature_merge_candidates(features::Dict{Int, DipoleFeatureNode};
                                       config::FeatureDisambiguationConfig = DEFAULT_DISAMBIGUATION_CONFIG)
    candidates = Tuple{Int, Int, FeatureMergeResult}[]

    ids = collect(keys(features))
    n = length(ids)

    for i in 1:n
        for j in (i+1):n
            result = detect_feature_merge(features[ids[i]], features[ids[j]]; config = config)
            if result.should_merge
                push!(candidates, (ids[i], ids[j], result))
            end
        end
    end

    sort!(candidates, by = x -> x[3].distance)

    return candidates
end

# ============================================================================
# Disambiguation Manager
# ============================================================================

"""
    FeatureDisambiguationManager

Manages feature disambiguation across the mission.
"""
mutable struct FeatureDisambiguationManager
    config::FeatureDisambiguationConfig
    association_history::Vector{FeatureAssociationResult}
    merge_count::Int
    split_count::Int
    ambiguous_count::Int
end

function FeatureDisambiguationManager(;
    config::FeatureDisambiguationConfig = DEFAULT_DISAMBIGUATION_CONFIG
)
    FeatureDisambiguationManager(config, FeatureAssociationResult[], 0, 0, 0)
end

"""
    process_feature_disambiguation!(mgr, registry)

Process all disambiguation decisions for the registry.
"""
function process_feature_disambiguation!(mgr::FeatureDisambiguationManager,
                                         registry::DipoleFeatureRegistry)
    if !dipoles_enabled(registry)
        return
    end

    merges = find_feature_merge_candidates(registry.features; config = mgr.config)

    merged_ids = Set{Int}()

    for (id1, id2, result) in merges
        if id1 in merged_ids || id2 in merged_ids
            continue
        end

        if haskey(registry.features, id1) && haskey(registry.features, id2)
            dipole1 = registry.features[id1]
            dipole1.state = result.merged_state
            dipole1.covariance = result.merged_covariance
            dipole1.support_count += registry.features[id2].support_count

            retire_dipole_feature!(registry, id2)
            push!(merged_ids, id2)
            mgr.merge_count += 1
        end
    end
end

# ============================================================================
# Exports
# ============================================================================

export FeatureDisambiguationConfig, DEFAULT_DISAMBIGUATION_CONFIG
export feature_mahalanobis_distance, feature_mahalanobis_distance_sq
export FeatureAssociationDecision
export ASSOCIATE_CLEAR, ASSOCIATE_AMBIGUOUS, ASSOCIATE_NO_MATCH, ASSOCIATE_SPLIT
export FeatureAssociationResult, associate_feature_measurement
export FeatureSplitResult, detect_feature_split, simple_2means_clustering
export FeatureMergeResult, detect_feature_merge, find_feature_merge_candidates
export FeatureDisambiguationManager, process_feature_disambiguation!
