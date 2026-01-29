# ============================================================================
# SourceDetectionFrontEnd.jl - Detection Front-End (Phase G Step 3)
# ============================================================================
#
# Detects magnetic anomalies from measurement residuals, clusters them
# spatially, and associates with existing source candidates.
#
# Pipeline: residual → chi² test → source subtraction → clustering → association
#
# Reuses: SourceDetectionConfig, SpatialClusteringPipeline, SourceSeparator,
#          ResidualManager chi-square gating
#
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Types
# ============================================================================

"""
    DetectionEvent

A single anomaly detection from residual analysis.

# Fields
- `position::Vec3Map`: Measurement position in world frame [m]
- `residual::Vec3Map`: Residual vector after background subtraction [T]
- `chi2::Float64`: Chi-square statistic of residual
- `snr::Float64`: Signal-to-noise ratio
- `timestamp::Float64`: Detection time [s]
"""
struct DetectionEvent
    position::Vec3Map
    residual::Vec3Map
    chi2::Float64
    snr::Float64
    timestamp::Float64
end

"""
    SourceDetectionFrontEnd

Front-end for detecting magnetic anomalies from measurement residuals.

# Fields
- `config::SourceDetectionConfig`: Detection thresholds (reused from OnlineSourceSLAM)
- `clustering_config::SpatialClusteringConfig`: Spatial clustering parameters
- `max_candidates::Int`: Maximum number of candidates to track
- `detection_count::Int`: Total detections seen
"""
mutable struct SourceDetectionFrontEnd
    config::SourceDetectionConfig
    clustering_config::SpatialClusteringConfig
    max_candidates::Int
    detection_count::Int
end

"""Create detection front-end with default configuration."""
function SourceDetectionFrontEnd(;
    config::SourceDetectionConfig = DEFAULT_SOURCE_DETECTION_CONFIG,
    clustering_config::SpatialClusteringConfig = DEFAULT_SPATIAL_CLUSTERING_CONFIG
)
    SourceDetectionFrontEnd(config, clustering_config, config.max_candidates, 0)
end

# ============================================================================
# Detection
# ============================================================================

"""
    detect_anomalies(fe::SourceDetectionFrontEnd,
                     position::Vec3Map,
                     z_measured::Vec3Map,
                     z_bg_pred::Vec3Map,
                     R::Mat3Map,
                     active_sources::Vector{SlamSourceState},
                     t::Float64) -> Vector{DetectionEvent}

Detect anomalous residuals after subtracting background and active sources.

# Steps
1. Compute raw residual: r = z_measured - z_bg_pred
2. Subtract active source contributions from residual
3. Compute chi-square: γ = r' R⁻¹ r
4. Fire detection if γ > threshold AND |r| > min_residual_norm

# Arguments
- `position`: Measurement position [m]
- `z_measured`: Measured magnetic field [T]
- `z_bg_pred`: Predicted background field [T]
- `R`: Measurement noise covariance [T²]
- `active_sources`: Currently tracked active sources
- `t`: Timestamp [s]

# Returns
Vector of DetectionEvent (0 or 1 events per call)
"""
function detect_anomalies(fe::SourceDetectionFrontEnd,
                           position::Vec3Map,
                           z_measured::Vec3Map,
                           z_bg_pred::Vec3Map,
                           R::Mat3Map,
                           active_sources::Vector{SlamSourceState},
                           t::Float64)
    events = DetectionEvent[]

    # Raw residual
    residual = z_measured - z_bg_pred

    # Subtract active sources
    for src in active_sources
        if src.lifecycle == :active
            B_src = dipole_field_at(position, source_position(src), source_moment(src))
            residual = residual - Vec3Map(B_src...)
        end
    end

    # Chi-square test
    R_mat = Matrix{Float64}(R) + 1e-30 * I(3)  # Regularize
    chi2 = Vector(residual)' * (R_mat \ Vector(residual))

    # SNR
    noise_std = sqrt(tr(R_mat) / 3)
    snr = noise_std > 0 ? norm(residual) / noise_std : 0.0

    # Detection gate
    if chi2 > fe.config.detection_chi2_threshold &&
       norm(residual) > fe.config.min_residual_norm
        fe.detection_count += 1
        push!(events, DetectionEvent(position, residual, chi2, snr, t))
    end

    return events
end

"""
    cluster_detections(fe::SourceDetectionFrontEnd,
                       events::Vector{DetectionEvent}) -> Vector{Vector{DetectionEvent}}

Cluster detection events spatially using Euclidean distance.
"""
function cluster_detections(fe::SourceDetectionFrontEnd,
                             events::Vector{DetectionEvent})
    if isempty(events)
        return Vector{Vector{DetectionEvent}}[]
    end

    clusters = Vector{Vector{DetectionEvent}}()
    assigned = falses(length(events))

    for i in eachindex(events)
        if assigned[i]
            continue
        end

        cluster = [events[i]]
        assigned[i] = true

        for j in (i+1):length(events)
            if !assigned[j] && norm(events[i].position - events[j].position) < fe.config.spatial_clustering_radius
                push!(cluster, events[j])
                assigned[j] = true
            end
        end

        push!(clusters, cluster)
    end

    return clusters
end

"""
    associate_or_create!(fe::SourceDetectionFrontEnd,
                          events::Vector{DetectionEvent},
                          candidates::Dict{Int, SourceCandidate},
                          next_id::Ref{Int}) -> (new_ids::Vector{Int}, updated_ids::Vector{Int})

Associate detection events with existing candidates or create new ones.

Uses Euclidean distance for initial association (Mahalanobis for refinement).
Association threshold: spatial_clustering_radius from config.
"""
function associate_or_create!(fe::SourceDetectionFrontEnd,
                               events::Vector{DetectionEvent},
                               candidates::Dict{Int, SourceCandidate},
                               next_id::Ref{Int})
    new_ids = Int[]
    updated_ids = Int[]

    for event in events
        # Find nearest candidate
        best_id = -1
        best_dist = Inf
        for (id, cand) in candidates
            d = norm(event.position - cand.centroid)
            if d < best_dist
                best_dist = d
                best_id = id
            end
        end

        if best_dist < fe.config.spatial_clustering_radius && best_id > 0
            # Associate with existing candidate
            add_observation!(candidates[best_id], event.position, event.residual,
                           event.timestamp, event.chi2)
            push!(updated_ids, best_id)
        elseif length(candidates) < fe.max_candidates
            # Create new candidate
            id = next_id[]
            next_id[] += 1
            candidates[id] = SourceCandidate(id, event.position, event.residual,
                                             event.timestamp, event.chi2)
            push!(new_ids, id)
        end
    end

    return (new_ids, updated_ids)
end

# ============================================================================
# Exports
# ============================================================================

export DetectionEvent, SourceDetectionFrontEnd
export detect_anomalies, cluster_detections, associate_or_create!
