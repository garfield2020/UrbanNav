# ============================================================================
# Spatial Clustering for Detection
# ============================================================================
#
# Ported from AUV-Navigation/src/spatial_clustering.jl
#
# Provides spatial clustering of detections with pose uncertainty awareness:
#   - Greedy clustering within merge_radius
#   - Mahalanobis-based clustering under pose uncertainty
#   - Compactness gate for false alarm rejection
# ============================================================================

using LinearAlgebra
using StaticArrays
using Statistics

# ============================================================================
# Detection Representation
# ============================================================================

"""
    SpatialDetection

Single detection event with position, moment estimate, and uncertainty.
"""
struct SpatialDetection
    position::SVector{3, Float64}       # Estimated source position (m)
    moment::SVector{3, Float64}         # Estimated moment (A·m²)
    position_cov::SMatrix{3, 3, Float64, 9}  # Position uncertainty
    timestamp::Float64                   # Time of detection
    snr::Float64                         # Detection SNR
    chi2::Float64                        # χ² statistic
    id::Int                              # Unique detection ID
end

function SpatialDetection(;
    position::AbstractVector,
    moment::AbstractVector = zeros(3),
    position_cov::AbstractMatrix = Matrix(1.0I, 3, 3),
    timestamp::Float64 = 0.0,
    snr::Float64 = 0.0,
    chi2::Float64 = 0.0,
    id::Int = 0
)
    SpatialDetection(
        SVector{3}(position),
        SVector{3}(moment),
        SMatrix{3,3}(position_cov),
        timestamp, snr, chi2, id
    )
end

"""
    SpatialCluster

Cluster of detections believed to originate from the same source.
"""
mutable struct SpatialCluster
    detections::Vector{SpatialDetection}
    centroid::SVector{3, Float64}
    combined_moment::SVector{3, Float64}
    combined_cov::SMatrix{3, 3, Float64, 9}
    mean_snr::Float64
    cluster_id::Int
end

function SpatialCluster(det::SpatialDetection, cluster_id::Int)
    SpatialCluster(
        [det], det.position, det.moment, det.position_cov, det.snr, cluster_id
    )
end

# ============================================================================
# Distance Metrics
# ============================================================================

"""
    euclidean_dist(p1, p2)

Simple Euclidean distance between two positions.
"""
euclidean_dist(p1::AbstractVector, p2::AbstractVector) = norm(p1 - p2)

"""
    mahalanobis_dist(p1, p2, Σ1, Σ2)

Mahalanobis distance accounting for position uncertainties.
d² = (p1 - p2)ᵀ (Σ1 + Σ2)⁻¹ (p1 - p2)
"""
function mahalanobis_dist(p1::AbstractVector, p2::AbstractVector,
                          Σ1::AbstractMatrix, Σ2::AbstractMatrix)
    Δp = p1 - p2
    Σ_combined = Σ1 + Σ2 + 1e-10 * I

    try
        return sqrt(max(0.0, Δp' * (Σ_combined \ Δp)))
    catch
        return norm(Δp)
    end
end

"""
    mahalanobis_dist_sq(p1, p2, Σ1, Σ2)

Squared Mahalanobis distance (avoids sqrt for threshold comparisons).
"""
function mahalanobis_dist_sq(p1::AbstractVector, p2::AbstractVector,
                             Σ1::AbstractMatrix, Σ2::AbstractMatrix)
    Δp = p1 - p2
    Σ_combined = Σ1 + Σ2 + 1e-10 * I

    try
        return max(0.0, Δp' * (Σ_combined \ Δp))
    catch
        return dot(Δp, Δp)
    end
end

# ============================================================================
# Clustering Configuration
# ============================================================================

"""
    SpatialClusteringConfig

Configuration for spatial clustering.
"""
struct SpatialClusteringConfig
    merge_radius::Float64           # Euclidean merge threshold (m)
    mahalanobis_threshold::Float64  # Mahalanobis merge threshold (σ)
    use_mahalanobis::Bool           # Use uncertainty-aware clustering
    min_cluster_size::Int           # Minimum detections to form valid cluster
    max_cluster_radius::Float64     # Maximum cluster extent (m)
end

function SpatialClusteringConfig(;
    merge_radius::Float64 = 2.0,
    mahalanobis_threshold::Float64 = 3.0,  # 3σ gate
    use_mahalanobis::Bool = true,
    min_cluster_size::Int = 1,
    max_cluster_radius::Float64 = 10.0
)
    SpatialClusteringConfig(merge_radius, mahalanobis_threshold, use_mahalanobis,
                            min_cluster_size, max_cluster_radius)
end

const DEFAULT_SPATIAL_CLUSTERING_CONFIG = SpatialClusteringConfig()

# ============================================================================
# Greedy Spatial Clustering
# ============================================================================

"""
    greedy_spatial_cluster(detections, config)

Greedy spatial clustering of detections.

Algorithm:
1. Sort detections by SNR (strongest first)
2. For each unassigned detection, start new cluster
3. Add nearby detections (within merge_radius or Mahalanobis threshold)
4. Compute cluster centroid and combined uncertainty
"""
function greedy_spatial_cluster(detections::Vector{SpatialDetection},
                                config::SpatialClusteringConfig = DEFAULT_SPATIAL_CLUSTERING_CONFIG)
    n = length(detections)
    if n == 0
        return SpatialCluster[]
    end

    # Sort by SNR (strongest first)
    sorted_idx = sortperm([d.snr for d in detections], rev=true)

    used = falses(n)
    clusters = SpatialCluster[]
    cluster_id = 0

    for i in sorted_idx
        if used[i]
            continue
        end

        # Start new cluster
        cluster_id += 1
        cluster = SpatialCluster(detections[i], cluster_id)
        used[i] = true

        # Find nearby detections to add
        for j in sorted_idx
            if used[j]
                continue
            end

            should_merge = false

            if config.use_mahalanobis
                d_sq = mahalanobis_dist_sq(
                    cluster.centroid, detections[j].position,
                    cluster.combined_cov, detections[j].position_cov
                )
                should_merge = d_sq < config.mahalanobis_threshold^2
            else
                d = euclidean_dist(cluster.centroid, detections[j].position)
                should_merge = d < config.merge_radius
            end

            # Check cluster extent
            if should_merge
                new_extent = max_cluster_extent(cluster, detections[j])
                if new_extent > config.max_cluster_radius
                    should_merge = false
                end
            end

            if should_merge
                add_to_spatial_cluster!(cluster, detections[j])
                used[j] = true
            end
        end

        if length(cluster.detections) >= config.min_cluster_size
            push!(clusters, cluster)
        end
    end

    return clusters
end

"""
    add_to_spatial_cluster!(cluster, detection)

Add a detection to a cluster and update cluster statistics.
"""
function add_to_spatial_cluster!(cluster::SpatialCluster, det::SpatialDetection)
    push!(cluster.detections, det)

    # Update centroid (uniform weighting)
    positions = [d.position for d in cluster.detections]
    cluster.centroid = SVector{3}(mean(positions))

    # Combined moment
    moments = [d.moment for d in cluster.detections]
    cluster.combined_moment = SVector{3}(mean(moments))

    # Combined covariance: precision adds
    P_total = sum(inv(d.position_cov + 1e-10*I) for d in cluster.detections)
    cluster.combined_cov = SMatrix{3,3}(inv(P_total + 1e-10*I))

    # Mean SNR
    cluster.mean_snr = mean(d.snr for d in cluster.detections)
end

"""
    max_cluster_extent(cluster, new_detection)

Compute maximum extent if new detection were added to cluster.
"""
function max_cluster_extent(cluster::SpatialCluster, det::SpatialDetection)
    max_dist = 0.0
    for d in cluster.detections
        dist = euclidean_dist(d.position, det.position)
        max_dist = max(max_dist, dist)
    end
    return max_dist
end

# ============================================================================
# Compactness Gate
# ============================================================================

"""
    CompactnessResult

Result of compactness gate evaluation.
"""
struct CompactnessResult
    is_compact::Bool
    scatter::Float64                # RMS scatter from centroid (m)
    mahalanobis_scatter::Float64    # Scatter in Mahalanobis units
    n_detections::Int
    reason::String
end

"""
    evaluate_compactness(cluster; max_scatter, max_mahal)

Evaluate whether a cluster is spatially compact (likely real source).

A real source should produce tightly clustered detections.
Scattered detections suggest noise.
"""
function evaluate_compactness(cluster::SpatialCluster;
                              max_scatter::Float64 = 3.0,
                              max_mahal::Float64 = 5.0)
    n = length(cluster.detections)

    if n < 2
        return CompactnessResult(true, 0.0, 0.0, n, "Single detection")
    end

    # Physical scatter (RMS distance from centroid)
    scatter = sqrt(mean(
        norm(d.position - cluster.centroid)^2 for d in cluster.detections
    ))

    # Mahalanobis scatter (accounting for uncertainty)
    mahal_scatter = sqrt(mean(
        mahalanobis_dist_sq(d.position, cluster.centroid,
                           d.position_cov, cluster.combined_cov)
        for d in cluster.detections
    ))

    is_compact = scatter < max_scatter && mahal_scatter < max_mahal

    reason = if !is_compact && scatter >= max_scatter
        "Physical scatter $(round(scatter, digits=1))m > $(max_scatter)m"
    elseif !is_compact && mahal_scatter >= max_mahal
        "Mahalanobis scatter $(round(mahal_scatter, digits=1))σ > $(max_mahal)σ"
    else
        "Compact"
    end

    return CompactnessResult(is_compact, scatter, mahal_scatter, n, reason)
end

# ============================================================================
# Clustering Pipeline
# ============================================================================

"""
    SpatialClusteringPipeline

Pipeline for clustering detections with pose uncertainty awareness.
"""
mutable struct SpatialClusteringPipeline
    config::SpatialClusteringConfig
    pending_detections::Vector{SpatialDetection}
    clusters::Vector{SpatialCluster}
    next_detection_id::Int
    next_cluster_id::Int
end

function SpatialClusteringPipeline(config::SpatialClusteringConfig = DEFAULT_SPATIAL_CLUSTERING_CONFIG)
    SpatialClusteringPipeline(config, SpatialDetection[], SpatialCluster[], 1, 1)
end

"""
    add_spatial_detection!(pipeline, position, moment, position_cov, timestamp, snr, chi2)

Add a new detection to the pipeline.
"""
function add_spatial_detection!(pipeline::SpatialClusteringPipeline,
                                position::AbstractVector,
                                moment::AbstractVector,
                                position_cov::AbstractMatrix,
                                timestamp::Float64,
                                snr::Float64,
                                chi2::Float64)
    det = SpatialDetection(
        position = position,
        moment = moment,
        position_cov = position_cov,
        timestamp = timestamp,
        snr = snr,
        chi2 = chi2,
        id = pipeline.next_detection_id
    )
    pipeline.next_detection_id += 1

    push!(pipeline.pending_detections, det)
    return det
end

"""
    process_spatial_detections!(pipeline)

Cluster pending detections and evaluate compactness.

Returns vector of (cluster, compactness_result) for clusters that pass the gate.
"""
function process_spatial_detections!(pipeline::SpatialClusteringPipeline)
    if isempty(pipeline.pending_detections)
        return Tuple{SpatialCluster, CompactnessResult}[]
    end

    # Cluster detections
    new_clusters = greedy_spatial_cluster(pipeline.pending_detections, pipeline.config)

    # Renumber clusters
    for cluster in new_clusters
        cluster.cluster_id = pipeline.next_cluster_id
        pipeline.next_cluster_id += 1
    end

    # Evaluate compactness
    results = Tuple{SpatialCluster, CompactnessResult}[]
    for cluster in new_clusters
        compactness = evaluate_compactness(cluster)

        if compactness.is_compact
            push!(pipeline.clusters, cluster)
            push!(results, (cluster, compactness))
        end
    end

    # Clear pending
    empty!(pipeline.pending_detections)

    return results
end

"""
    get_confirmed_cluster_sources(pipeline)

Get all confirmed source positions from compact clusters.
"""
function get_confirmed_cluster_sources(pipeline::SpatialClusteringPipeline)
    return [(c.centroid, c.combined_moment, c.combined_cov) for c in pipeline.clusters]
end

# ============================================================================
# Exports
# ============================================================================

export SpatialDetection, SpatialCluster
export euclidean_dist, mahalanobis_dist, mahalanobis_dist_sq
export SpatialClusteringConfig, DEFAULT_SPATIAL_CLUSTERING_CONFIG
export greedy_spatial_cluster, add_to_spatial_cluster!, max_cluster_extent
export CompactnessResult, evaluate_compactness
export SpatialClusteringPipeline
export add_spatial_detection!, process_spatial_detections!, get_confirmed_cluster_sources
