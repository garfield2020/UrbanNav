# ============================================================================
# Fleet Map Fusion (Phase B)
# ============================================================================
#
# Enables multiple vehicles to share and fuse magnetic map knowledge.
#
# Purpose: "Fleet learns faster than any single vehicle."
#
# The Problem:
# -----------
# Each vehicle builds its own map from its observations. Without fusion:
# - Redundant coverage: Multiple vehicles re-learn the same areas
# - Slow convergence: Each vehicle starts from scratch
# - Inconsistent maps: Vehicles may have conflicting estimates
#
# The Solution:
# ------------
# Fleet Map Fusion using Covariance Intersection (CI):
# - Robust to unknown correlations between vehicle maps
# - No double-counting of shared information
# - Decentralized: No central server required
#
# Key Equation (Information Form):
#   P_fused⁻¹ = ω·P₁⁻¹ + (1-ω)·P₂⁻¹
#   α_fused = P_fused · (ω·P₁⁻¹·α₁ + (1-ω)·P₂⁻¹·α₂)
#
# where ω ∈ [0,1] minimizes det(P_fused) or tr(P_fused).
#
# ============================================================================

using LinearAlgebra
using StaticArrays
using Dates
using Statistics: mean

# ============================================================================
# Fleet Map Message Types
# ============================================================================

"""
    FleetMapMessageType

Type of inter-vehicle map message.

# Types
- `MAP_MSG_TILE_UPDATE`: Share tile coefficients and covariance
- `MAP_MSG_VERSION_QUERY`: Request current map version from peer
- `MAP_MSG_VERSION_RESPONSE`: Respond with map version info
- `MAP_MSG_VALIDATION_RESULT`: Share validation metrics
"""
@enum FleetMapMessageType begin
    MAP_MSG_TILE_UPDATE = 1
    MAP_MSG_VERSION_QUERY = 2
    MAP_MSG_VERSION_RESPONSE = 3
    MAP_MSG_VALIDATION_RESULT = 4
end

"""
    TileUpdatePayload

Payload for sharing tile coefficient updates.

# Fields
- `tile_id::MapTileID`: Which tile this update is for
- `coefficients::Vector{Float64}`: Map coefficients α
- `covariance::Matrix{Float64}`: Coefficient covariance P_α
- `observation_count::Int`: Number of observations in this estimate
- `version::Int`: Map version when this was captured
- `quality_score::Float64`: Confidence in this estimate ∈ [0, 1]

# Quality Score Computation
Quality is based on:
- Observation count (more = better, saturates at N_saturate = 100)
- Covariance trace (smaller = better)
- Validation NEES if available
"""
struct TileUpdatePayload
    tile_id::MapTileID
    coefficients::Vector{Float64}
    covariance::Matrix{Float64}
    observation_count::Int
    version::Int
    quality_score::Float64

    function TileUpdatePayload(
        tile_id::MapTileID,
        coefficients::AbstractVector,
        covariance::AbstractMatrix,
        observation_count::Int,
        version::Int,
        quality_score::Float64
    )
        @assert 0 <= quality_score <= 1 "Quality score must be in [0, 1]"
        new(
            tile_id,
            Vector{Float64}(coefficients),
            Matrix{Float64}(covariance),
            observation_count,
            version,
            quality_score
        )
    end
end

"""Compute quality score for tile data."""
function compute_tile_quality(
    tile::MapTileData;
    validation_nees::Union{Float64, Nothing} = nothing
)
    # Quality based on observation count
    # Saturates at N_saturate = 100 observations
    # Justification: Law of large numbers - σ/√N, so 100 obs gives 10x reduction
    N_saturate = 100
    obs_quality = min(tile.observation_count / N_saturate, 1.0)

    # Quality based on covariance trace
    # Normalize by expected good covariance (1e-12 T² typical for well-observed tile)
    σ_good = 1e-6  # 1 µT field uncertainty = good quality
    cov_trace = tr(tile.covariance)
    n_coef = length(tile.coefficients)
    # Average variance per coefficient
    avg_var = cov_trace / max(n_coef, 1)
    cov_quality = min(σ_good^2 / max(avg_var, 1e-20), 1.0)

    # Quality based on NEES (if available)
    nees_quality = 1.0
    if validation_nees !== nothing
        # NEES should be ≈ 1.0 for well-calibrated filter
        # Penalize if NEES is far from 1
        # Quality = 1 at NEES=1, drops to 0.5 at NEES=0.1 or NEES=10
        nees_quality = exp(-0.5 * (log(validation_nees))^2)
    end

    # Combined quality: geometric mean
    # Geometric mean is appropriate when factors are multiplicative
    quality = (obs_quality * cov_quality * nees_quality)^(1/3)

    return clamp(quality, 0.0, 1.0)
end

"""
    FleetMapMessage

Message for sharing map data between vehicles.

# Fields
- `msg_type::FleetMapMessageType`: Type of message
- `sender::VehicleId`: Sending vehicle
- `receiver::Union{VehicleId, Nothing}`: Target (nothing for broadcast)
- `timestamp::Float64`: Message creation time [s]
- `payload::Union{TileUpdatePayload, ValidationMetrics, MapVersionInfo, Nothing}`
"""
struct FleetMapMessage
    msg_type::FleetMapMessageType
    sender::VehicleId
    receiver::Union{VehicleId, Nothing}
    timestamp::Float64
    payload::Any  # TileUpdatePayload, ValidationMetrics, MapVersionInfo, or Nothing
end

"""Create tile update message."""
function tile_update_message(
    sender::VehicleId,
    tile::MapTileData,
    version::Int,
    timestamp::Float64;
    receiver::Union{VehicleId, Nothing} = nothing,
    validation_nees::Union{Float64, Nothing} = nothing
)
    quality = compute_tile_quality(tile; validation_nees=validation_nees)
    payload = TileUpdatePayload(
        tile.id,
        tile.coefficients,
        tile.covariance,
        tile.observation_count,
        version,
        quality
    )
    FleetMapMessage(MAP_MSG_TILE_UPDATE, sender, receiver, timestamp, payload)
end

"""Create version query message."""
function version_query_message(sender::VehicleId, receiver::VehicleId, timestamp::Float64)
    FleetMapMessage(MAP_MSG_VERSION_QUERY, sender, receiver, timestamp, nothing)
end

"""Create version response message."""
function version_response_message(
    sender::VehicleId,
    receiver::VehicleId,
    version_info::MapVersionInfo,
    timestamp::Float64
)
    FleetMapMessage(MAP_MSG_VERSION_RESPONSE, sender, receiver, timestamp, version_info)
end

# ============================================================================
# Fleet Map Fusion Configuration
# ============================================================================

"""
    FleetMapFusionConfig

Configuration for fleet map fusion.

# Fields
- `omega_initial::Float64`: Initial CI mixing parameter
- `auto_omega::Bool`: Automatically optimize omega
- `min_quality_threshold::Float64`: Minimum quality to accept update
- `max_version_lag::Int`: Maximum version difference to allow fusion
- `stale_timeout::Float64`: Time before peer data is stale [s]
- `min_observation_count::Int`: Minimum observations for reliable estimate
- `fusion_mode::Symbol`: :conservative, :balanced, :aggressive

# Threshold Justifications

**min_quality_threshold = 0.3**:
Below 30% quality, the tile estimate has either:
- Few observations (< 30)
- Large covariance (poor constraint)
- Poor NEES (miscalibrated)
Fusing with such data can degrade the combined estimate.

**max_version_lag = 5**:
If vehicle maps differ by more than 5 versions, they may have
diverged significantly. Require explicit reconciliation rather
than automatic fusion.

**min_observation_count = 10**:
With fewer than 10 observations, the coefficient estimate is
dominated by prior, not data. σ/√N with N=10 gives ~30% of
single-observation uncertainty.

**stale_timeout = 60.0 s**:
For typical AUV operations at 1-2 m/s, 60s corresponds to
60-120m of travel. Older data may be from significantly
different map regions.
"""
struct FleetMapFusionConfig
    omega_initial::Float64
    auto_omega::Bool
    min_quality_threshold::Float64
    max_version_lag::Int
    stale_timeout::Float64
    min_observation_count::Int
    fusion_mode::Symbol

    function FleetMapFusionConfig(;
        omega_initial::Float64 = 0.5,
        auto_omega::Bool = true,
        min_quality_threshold::Float64 = 0.3,
        max_version_lag::Int = 5,
        stale_timeout::Float64 = 60.0,
        min_observation_count::Int = 10,
        fusion_mode::Symbol = :balanced
    )
        @assert 0 < omega_initial < 1 "omega must be in (0, 1)"
        @assert 0 <= min_quality_threshold <= 1
        @assert max_version_lag > 0
        @assert stale_timeout > 0
        @assert min_observation_count > 0
        @assert fusion_mode in [:conservative, :balanced, :aggressive]
        new(omega_initial, auto_omega, min_quality_threshold, max_version_lag,
            stale_timeout, min_observation_count, fusion_mode)
    end
end

const DEFAULT_FLEET_MAP_FUSION_CONFIG = FleetMapFusionConfig()

# ============================================================================
# Covariance Intersection for Map Coefficients
# ============================================================================

"""
    map_covariance_intersection(α₁, P₁, α₂, P₂; omega=0.5)

Perform Covariance Intersection fusion of map coefficients.

# Arguments
- `α₁, α₂`: Coefficient vectors
- `P₁, P₂`: Covariance matrices
- `omega`: Mixing parameter ∈ (0, 1), higher = trust α₁ more

# Returns
- `(α_fused, P_fused)`: Fused coefficients and covariance

# Mathematical Derivation
CI is the optimal linear fusion when cross-correlation is unknown.

Information form:
  P_fused⁻¹ = ω·P₁⁻¹ + (1-ω)·P₂⁻¹
  α_fused = P_fused · (ω·P₁⁻¹·α₁ + (1-ω)·P₂⁻¹·α₂)

Properties:
- Conservative: P_fused ≤ P_i for optimal ω
- No double-counting: Safe even if α₁, α₂ share common information
- Robust: Works for any true correlation

Reference: Julier & Uhlmann (1997) "A Non-divergent Estimation Algorithm
in the Presence of Unknown Correlations"
"""
function map_covariance_intersection(
    α₁::AbstractVector, P₁::AbstractMatrix,
    α₂::AbstractVector, P₂::AbstractMatrix;
    omega::Float64 = 0.5
)
    @assert length(α₁) == length(α₂) "Coefficient vectors must have same length"
    @assert size(P₁) == size(P₂) "Covariance matrices must have same size"

    # Clamp omega to valid range
    ω = clamp(omega, 0.01, 0.99)

    # Compute information matrices
    P₁_reg = P₁ + 1e-12 * I(size(P₁, 1))  # Regularize for numerical stability
    P₂_reg = P₂ + 1e-12 * I(size(P₂, 1))

    try
        P₁_inv = inv(P₁_reg)
        P₂_inv = inv(P₂_reg)

        # CI fusion in information form
        P_fused_inv = ω * P₁_inv + (1 - ω) * P₂_inv
        P_fused = inv(P_fused_inv)

        # Fused coefficients
        α_fused = P_fused * (ω * P₁_inv * α₁ + (1 - ω) * P₂_inv * α₂)

        # Ensure covariance is symmetric
        P_fused = 0.5 * (P_fused + P_fused')

        return (Vector{Float64}(α_fused), Matrix{Float64}(P_fused))
    catch e
        # Fall back to weighted average if matrices are ill-conditioned
        @warn "CI fusion failed, using weighted average" exception=e
        α_fused = ω * α₁ + (1 - ω) * α₂
        P_fused = ω^2 * P₁ + (1 - ω)^2 * P₂
        return (Vector{Float64}(α_fused), Matrix{Float64}(P_fused))
    end
end

"""
    optimize_map_omega(P₁, P₂; criterion=:determinant, num_samples=20)

Find optimal ω that minimizes the fused covariance.

# Arguments
- `P₁, P₂`: Covariance matrices to fuse
- `criterion`: `:determinant` (minimize det) or `:trace` (minimize tr)
- `num_samples`: Number of ω values to evaluate

# Returns
- Optimal ω value

# Criterion Selection
- `:determinant`: Minimizes volume of uncertainty ellipsoid (D-optimal)
- `:trace`: Minimizes sum of variances (A-optimal)

For map fusion, determinant is typically preferred as it accounts
for correlations between coefficients.
"""
function optimize_map_omega(
    P₁::AbstractMatrix, P₂::AbstractMatrix;
    criterion::Symbol = :determinant,
    num_samples::Int = 20
)
    best_omega = 0.5
    best_metric = Inf

    P₁_reg = P₁ + 1e-12 * I(size(P₁, 1))
    P₂_reg = P₂ + 1e-12 * I(size(P₂, 1))

    for ω in range(0.1, 0.9, length=num_samples)
        try
            P₁_inv = inv(P₁_reg)
            P₂_inv = inv(P₂_reg)
            P_fused_inv = ω * P₁_inv + (1 - ω) * P₂_inv
            P_fused = inv(P_fused_inv)

            metric = if criterion == :determinant
                det(P_fused)
            else  # :trace
                tr(P_fused)
            end

            if metric < best_metric && metric > 0
                best_metric = metric
                best_omega = ω
            end
        catch
            continue
        end
    end

    return best_omega
end

# ============================================================================
# Fleet Map Fusion Statistics
# ============================================================================

"""
    FleetMapFusionStatistics

Statistics for monitoring fleet map fusion quality.

# Fields
- `total_fusions::Int`: Total tile fusions performed
- `successful_fusions::Int`: Fusions that improved estimate
- `rejected_low_quality::Int`: Rejected due to low quality
- `rejected_stale::Int`: Rejected due to stale data
- `rejected_version_lag::Int`: Rejected due to version mismatch
- `mean_improvement_ratio::Float64`: Mean covariance reduction ratio
- `vehicles_contributed::Set{Int}`: Vehicle IDs that have contributed
"""
mutable struct FleetMapFusionStatistics
    total_fusions::Int
    successful_fusions::Int
    rejected_low_quality::Int
    rejected_stale::Int
    rejected_version_lag::Int
    mean_improvement_ratio::Float64
    vehicles_contributed::Set{Int}
end

function FleetMapFusionStatistics()
    FleetMapFusionStatistics(0, 0, 0, 0, 0, 0.0, Set{Int}())
end

"""Update statistics after fusion attempt."""
function update_fusion_statistics!(
    stats::FleetMapFusionStatistics,
    success::Bool,
    rejection_reason::Union{Symbol, Nothing},
    improvement_ratio::Float64,
    vehicle_id::Int
)
    stats.total_fusions += 1

    if success
        stats.successful_fusions += 1
        push!(stats.vehicles_contributed, vehicle_id)
        # Running average of improvement ratio
        n = stats.successful_fusions
        stats.mean_improvement_ratio =
            ((n-1) * stats.mean_improvement_ratio + improvement_ratio) / n
    else
        if rejection_reason == :low_quality
            stats.rejected_low_quality += 1
        elseif rejection_reason == :stale
            stats.rejected_stale += 1
        elseif rejection_reason == :version_lag
            stats.rejected_version_lag += 1
        end
    end
end

"""Compute fusion success rate."""
function fusion_success_rate(stats::FleetMapFusionStatistics)
    if stats.total_fusions == 0
        return 1.0
    end
    return stats.successful_fusions / stats.total_fusions
end

# ============================================================================
# Fleet Map Fusion Result
# ============================================================================

"""
    FleetMapFusionResult

Result of fusing map data from multiple vehicles.

# Fields
- `success::Bool`: Whether fusion was performed
- `tile_id::MapTileID`: Tile that was fused
- `fused_coefficients::Vector{Float64}`: Resulting coefficients
- `fused_covariance::Matrix{Float64}`: Resulting covariance
- `improvement_ratio::Float64`: tr(P_before) / tr(P_after)
- `contributing_vehicles::Vector{Int}`: Vehicle IDs that contributed
- `omega_used::Float64`: CI mixing parameter used
- `rejection_reason::Union{Symbol, Nothing}`: Why fusion was rejected
"""
struct FleetMapFusionResult
    success::Bool
    tile_id::MapTileID
    fused_coefficients::Vector{Float64}
    fused_covariance::Matrix{Float64}
    improvement_ratio::Float64
    contributing_vehicles::Vector{Int}
    omega_used::Float64
    rejection_reason::Union{Symbol, Nothing}
end

"""Create successful fusion result."""
function FleetMapFusionResult(
    tile_id::MapTileID,
    coefficients::Vector{Float64},
    covariance::Matrix{Float64},
    improvement_ratio::Float64,
    contributing_vehicles::Vector{Int},
    omega::Float64
)
    FleetMapFusionResult(
        true, tile_id, coefficients, covariance,
        improvement_ratio, contributing_vehicles, omega, nothing
    )
end

"""Create rejected fusion result."""
function FleetMapFusionResult(
    tile_id::MapTileID,
    original_coefficients::Vector{Float64},
    original_covariance::Matrix{Float64},
    reason::Symbol
)
    FleetMapFusionResult(
        false, tile_id, original_coefficients, original_covariance,
        1.0, Int[], 0.0, reason
    )
end

# ============================================================================
# Fleet Map Fusion Manager
# ============================================================================

"""
    FleetMapFusionManager

Manager for fleet-wide map fusion.

# Fields
- `own_id::VehicleId`: This vehicle's ID
- `config::FleetMapFusionConfig`: Fusion configuration
- `peer_tiles::Dict{VehicleId, Dict{MapTileID, TileUpdatePayload}}`: Peer tile data
- `peer_timestamps::Dict{VehicleId, Float64}`: Last heard from each peer
- `statistics::FleetMapFusionStatistics`: Running statistics

# Usage
```julia
manager = FleetMapFusionManager(VehicleId(1))

# Receive update from peer
receive_tile_update!(manager, message)

# Fuse with own tile
result = fuse_tile!(manager, own_tile, current_time)

if result.success
    # Update map with fused coefficients
    update_tile!(map, result.fused_coefficients, result.fused_covariance)
end
```
"""
mutable struct FleetMapFusionManager
    own_id::VehicleId
    config::FleetMapFusionConfig
    peer_tiles::Dict{VehicleId, Dict{MapTileID, TileUpdatePayload}}
    peer_timestamps::Dict{VehicleId, Float64}
    statistics::FleetMapFusionStatistics
end

function FleetMapFusionManager(
    own_id::VehicleId;
    config::FleetMapFusionConfig = DEFAULT_FLEET_MAP_FUSION_CONFIG
)
    FleetMapFusionManager(
        own_id,
        config,
        Dict{VehicleId, Dict{MapTileID, TileUpdatePayload}}(),
        Dict{VehicleId, Float64}(),
        FleetMapFusionStatistics()
    )
end

"""Receive tile update from peer vehicle."""
function receive_tile_update!(
    manager::FleetMapFusionManager,
    message::FleetMapMessage
)
    if message.msg_type != MAP_MSG_TILE_UPDATE
        return false
    end

    sender = message.sender
    if sender == manager.own_id
        return false  # Ignore own messages
    end

    payload = message.payload::TileUpdatePayload

    # Initialize peer tile storage if needed
    if !haskey(manager.peer_tiles, sender)
        manager.peer_tiles[sender] = Dict{MapTileID, TileUpdatePayload}()
    end

    # Store update
    manager.peer_tiles[sender][payload.tile_id] = payload
    manager.peer_timestamps[sender] = message.timestamp

    return true
end

"""
    fuse_tile!(manager, own_tile, own_version, current_time)

Fuse own tile with peer data using Covariance Intersection.

# Arguments
- `manager`: Fleet fusion manager
- `own_tile`: Own tile data
- `own_version`: Current map version
- `current_time`: Current time for staleness check

# Returns
FleetMapFusionResult with fused coefficients or rejection reason.
"""
function fuse_tile!(
    manager::FleetMapFusionManager,
    own_tile::MapTileData,
    own_version::Int,
    current_time::Float64
)
    tile_id = own_tile.id
    config = manager.config

    # Collect valid peer estimates for this tile
    valid_peers = Tuple{VehicleId, TileUpdatePayload}[]

    for (peer_id, tiles) in manager.peer_tiles
        if !haskey(tiles, tile_id)
            continue
        end

        payload = tiles[tile_id]
        peer_time = get(manager.peer_timestamps, peer_id, 0.0)

        # Check staleness
        if current_time - peer_time > config.stale_timeout
            continue
        end

        # Check quality
        if payload.quality_score < config.min_quality_threshold
            update_fusion_statistics!(
                manager.statistics, false, :low_quality, 1.0, peer_id.id
            )
            continue
        end

        # Check version lag
        if abs(payload.version - own_version) > config.max_version_lag
            update_fusion_statistics!(
                manager.statistics, false, :version_lag, 1.0, peer_id.id
            )
            continue
        end

        # Check minimum observations
        if payload.observation_count < config.min_observation_count
            update_fusion_statistics!(
                manager.statistics, false, :low_quality, 1.0, peer_id.id
            )
            continue
        end

        push!(valid_peers, (peer_id, payload))
    end

    if isempty(valid_peers)
        return FleetMapFusionResult(
            tile_id,
            own_tile.coefficients,
            own_tile.covariance,
            :no_valid_peers
        )
    end

    # Sequential CI fusion with all valid peers
    α_fused = copy(own_tile.coefficients)
    P_fused = copy(own_tile.covariance)
    contributing_vehicles = Int[]
    omega_avg = 0.0

    for (peer_id, payload) in valid_peers
        # Ensure compatible dimensions
        if length(payload.coefficients) != length(α_fused)
            continue
        end

        # Find optimal omega
        omega = if config.auto_omega
            optimize_map_omega(P_fused, payload.covariance)
        else
            config.omega_initial
        end

        # Weight by quality scores (adjust omega based on relative quality)
        own_quality = compute_tile_quality(own_tile)
        quality_ratio = own_quality / (own_quality + payload.quality_score)
        omega = config.fusion_mode == :conservative ?
            max(omega, quality_ratio) :
            config.fusion_mode == :aggressive ?
            min(omega, quality_ratio) :
            0.5 * (omega + quality_ratio)

        α_fused, P_fused = map_covariance_intersection(
            α_fused, P_fused,
            payload.coefficients, payload.covariance;
            omega=omega
        )

        push!(contributing_vehicles, peer_id.id)
        omega_avg += omega
    end

    omega_avg /= length(valid_peers)

    # Compute improvement
    tr_before = tr(own_tile.covariance)
    tr_after = tr(P_fused)
    improvement_ratio = tr_before / max(tr_after, 1e-30)

    # Update statistics
    for vid in contributing_vehicles
        update_fusion_statistics!(
            manager.statistics, true, nothing, improvement_ratio, vid
        )
    end

    return FleetMapFusionResult(
        tile_id,
        α_fused,
        P_fused,
        improvement_ratio,
        contributing_vehicles,
        omega_avg
    )
end

"""Remove stale peer data."""
function prune_stale_peers!(manager::FleetMapFusionManager, current_time::Float64)
    stale_peers = VehicleId[]

    for (peer_id, timestamp) in manager.peer_timestamps
        if current_time - timestamp > manager.config.stale_timeout
            push!(stale_peers, peer_id)
        end
    end

    for peer_id in stale_peers
        delete!(manager.peer_tiles, peer_id)
        delete!(manager.peer_timestamps, peer_id)
    end

    return length(stale_peers)
end

"""Get list of peers with data for a tile."""
function peers_with_tile(manager::FleetMapFusionManager, tile_id::MapTileID)
    peers = VehicleId[]
    for (peer_id, tiles) in manager.peer_tiles
        if haskey(tiles, tile_id)
            push!(peers, peer_id)
        end
    end
    return peers
end

# ============================================================================
# Fleet Map Version Reconciliation
# ============================================================================

"""
    MapVersionConflict

Represents a version conflict between vehicles.

# Fields
- `tile_id::MapTileID`: Tile with conflict
- `own_version::Int`: Own map version
- `peer_version::Int`: Peer's map version
- `peer_id::VehicleId`: Which peer has the conflict
- `resolution::Symbol`: :use_own, :use_peer, :merge, :manual
"""
struct MapVersionConflict
    tile_id::MapTileID
    own_version::Int
    peer_version::Int
    peer_id::VehicleId
    resolution::Symbol
end

"""
    detect_version_conflicts(manager, own_version)

Detect version conflicts with peer vehicles.

# Returns
Vector of MapVersionConflict structs.
"""
function detect_version_conflicts(
    manager::FleetMapFusionManager,
    own_version::Int
)
    conflicts = MapVersionConflict[]

    for (peer_id, tiles) in manager.peer_tiles
        for (tile_id, payload) in tiles
            version_diff = payload.version - own_version

            if abs(version_diff) > manager.config.max_version_lag
                resolution = if version_diff > 0
                    :use_peer  # Peer is ahead
                elseif version_diff < 0
                    :use_own   # We are ahead
                else
                    :merge     # Same version but different data
                end

                push!(conflicts, MapVersionConflict(
                    tile_id, own_version, payload.version, peer_id, resolution
                ))
            end
        end
    end

    return conflicts
end

# ============================================================================
# Formatting and Display
# ============================================================================

"""Format fleet map fusion statistics."""
function format_fusion_statistics(stats::FleetMapFusionStatistics)
    lines = String[]
    push!(lines, "Fleet Map Fusion Statistics")
    push!(lines, "=" ^ 40)
    push!(lines, "Total fusion attempts: $(stats.total_fusions)")
    push!(lines, "Successful: $(stats.successful_fusions) ($(round(fusion_success_rate(stats) * 100, digits=1))%)")
    push!(lines, "")
    push!(lines, "Rejection Reasons:")
    push!(lines, "  Low quality: $(stats.rejected_low_quality)")
    push!(lines, "  Stale data: $(stats.rejected_stale)")
    push!(lines, "  Version lag: $(stats.rejected_version_lag)")
    push!(lines, "")
    push!(lines, "Performance:")
    push!(lines, "  Mean improvement: $(round(stats.mean_improvement_ratio, digits=2))x")
    push!(lines, "  Vehicles contributed: $(length(stats.vehicles_contributed))")

    return join(lines, "\n")
end

"""Format fusion result."""
function format_fusion_result(result::FleetMapFusionResult)
    lines = String[]
    push!(lines, "Fleet Map Fusion Result")
    push!(lines, "-" ^ 40)
    push!(lines, "Tile: $(result.tile_id)")

    if result.success
        push!(lines, "Status: ✓ SUCCESS")
        push!(lines, "Improvement: $(round(result.improvement_ratio, digits=2))x")
        push!(lines, "Contributing vehicles: $(result.contributing_vehicles)")
        push!(lines, "Omega used: $(round(result.omega_used, digits=3))")
    else
        push!(lines, "Status: ✗ REJECTED")
        push!(lines, "Reason: $(result.rejection_reason)")
    end

    return join(lines, "\n")
end

# ============================================================================
# Exports
# ============================================================================

export FleetMapMessageType, MAP_MSG_TILE_UPDATE, MAP_MSG_VERSION_QUERY
export MAP_MSG_VERSION_RESPONSE, MAP_MSG_VALIDATION_RESULT
export TileUpdatePayload, compute_tile_quality
export FleetMapMessage, tile_update_message, version_query_message, version_response_message
export FleetMapFusionConfig, DEFAULT_FLEET_MAP_FUSION_CONFIG
export map_covariance_intersection, optimize_map_omega
export FleetMapFusionStatistics, update_fusion_statistics!, fusion_success_rate
export FleetMapFusionResult
export FleetMapFusionManager, receive_tile_update!, fuse_tile!
export prune_stale_peers!, peers_with_tile
export MapVersionConflict, detect_version_conflicts
export format_fusion_statistics, format_fusion_result

