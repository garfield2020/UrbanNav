# ============================================================================
# OnlineFleetLearning.jl - Fleet Online Learning Integration (Phase C Step 9)
# ============================================================================
#
# Extends Phase B FleetMapFusion for real-time online SLAM integration.
#
# Key Additions:
# 1. Online SLAM state fusion (SlamAugmentedState)
# 2. Source (dipole) sharing between vehicles
# 3. Bandwidth-aware communication prioritization
# 4. Fleet convergence tracking
# 5. Distributed safety coordination
#
# Architecture:
# - Integrates with OnlineSlamScheduler for timing
# - Uses OnlineMapProvider for prediction
# - Coordinates with OnlineSafetyController for rollback
#
# Physics:
# - CI fusion remains conservative (INV-10)
# - Source matching requires geometric validation
# - Bandwidth budget constrains message rate
#
# ============================================================================

using LinearAlgebra
using StaticArrays
using Statistics: mean

# ============================================================================
# Fleet Learning Configuration
# ============================================================================

"""
    FleetLearningConfig

Configuration for fleet online learning.

# Fields
## Communication
- `bandwidth_budget_kbps::Float64`: Bandwidth budget per vehicle [kbps]
- `message_priority::Symbol`: Priority strategy (:round_robin, :highest_gain, :nearest)
- `max_messages_per_cycle::Int`: Maximum messages per slow loop cycle

## Tile Sharing
- `min_quality_to_share::Float64`: Minimum tile quality to share
- `min_observation_count::Int`: Minimum observations before sharing
- `stale_threshold_s::Float64`: Tile becomes stale after this time [s]

## Source Sharing
- `share_sources::Bool`: Whether to share source estimates
- `min_source_confidence::Float64`: Minimum confidence to share source
- `source_matching_threshold_m::Float64`: Distance for source matching [m]

## Safety
- `require_validated_checkpoint::Bool`: Only share if validated checkpoint exists
- `max_version_skew::Int`: Maximum version difference to accept
- `reject_degraded_peers::Bool`: Reject updates from degraded peers

# Bandwidth Economics
At 9600 baud acoustic modem (1200 bytes/s):
- Tile update: ~200 bytes (coefficients + covariance)
- Source update: ~100 bytes (position + moment + covariance)
- Budget of 0.5 kbps allows ~5 tile updates per second
"""
struct FleetLearningConfig
    # Communication
    bandwidth_budget_kbps::Float64
    message_priority::Symbol
    max_messages_per_cycle::Int

    # Tile sharing
    min_quality_to_share::Float64
    min_observation_count::Int
    stale_threshold_s::Float64

    # Source sharing
    share_sources::Bool
    min_source_confidence::Float64
    source_matching_threshold_m::Float64

    # Safety
    require_validated_checkpoint::Bool
    max_version_skew::Int
    reject_degraded_peers::Bool

    function FleetLearningConfig(;
        bandwidth_budget_kbps::Float64 = 0.5,
        message_priority::Symbol = :highest_gain,
        max_messages_per_cycle::Int = 3,
        min_quality_to_share::Float64 = 0.3,
        min_observation_count::Int = 10,
        stale_threshold_s::Float64 = 60.0,
        share_sources::Bool = true,
        min_source_confidence::Float64 = 0.7,
        source_matching_threshold_m::Float64 = 5.0,
        require_validated_checkpoint::Bool = true,
        max_version_skew::Int = 5,
        reject_degraded_peers::Bool = true
    )
        @assert bandwidth_budget_kbps > 0
        @assert message_priority in (:round_robin, :highest_gain, :nearest)
        @assert max_messages_per_cycle > 0
        @assert 0 <= min_quality_to_share <= 1
        @assert min_observation_count >= 0
        @assert stale_threshold_s > 0
        @assert 0 <= min_source_confidence <= 1
        @assert source_matching_threshold_m > 0
        @assert max_version_skew > 0

        new(bandwidth_budget_kbps, message_priority, max_messages_per_cycle,
            min_quality_to_share, min_observation_count, stale_threshold_s,
            share_sources, min_source_confidence, source_matching_threshold_m,
            require_validated_checkpoint, max_version_skew, reject_degraded_peers)
    end
end

const DEFAULT_FLEET_LEARNING_CONFIG = FleetLearningConfig()

# ============================================================================
# Fleet Learning Message Types
# ============================================================================

"""
    OnlineMapUpdate

Online map update message for fleet sharing.

Extends Phase B TileUpdatePayload with online SLAM fields.
"""
struct OnlineMapUpdate
    tile_id::MapTileID
    coefficients::Vector{Float64}
    information::Matrix{Float64}  # Information form for efficient fusion
    observation_count::Int
    version::Int
    quality_score::Float64
    sender_nees::Float64          # Sender's NEES for validation
    sender_health::Symbol         # :healthy, :degraded, :warning
    timestamp::Float64
end

"""Create online map update from SLAM tile state."""
function OnlineMapUpdate(tile::SlamTileState, sender_nees::Float64,
                          sender_health::Symbol, timestamp::Float64)
    quality = compute_online_tile_quality(tile, sender_nees)

    OnlineMapUpdate(
        tile.tile_id,
        copy(tile.coefficients),
        copy(tile.information),
        tile.observation_count,
        tile.version,
        quality,
        sender_nees,
        sender_health,
        timestamp
    )
end

"""Compute quality for online tile."""
function compute_online_tile_quality(tile::SlamTileState, nees::Float64)
    # Observation quality
    obs_quality = min(tile.observation_count / 100, 1.0)

    # Covariance quality
    avg_var = tr(tile.covariance) / max(length(tile.coefficients), 1)
    cov_quality = min(1e-12 / max(avg_var, 1e-20), 1.0)

    # NEES quality
    nees_quality = exp(-0.5 * (log(max(nees, 0.1)))^2)

    # Probationary penalty
    prob_penalty = tile.is_probationary ? 0.5 : 1.0

    quality = prob_penalty * (obs_quality * cov_quality * nees_quality)^(1/3)
    return clamp(quality, 0.0, 1.0)
end

"""
    OnlineSourceUpdate

Online source update message for fleet sharing.
"""
struct OnlineSourceUpdate
    source_id::Int
    position::SVector{3, Float64}
    moment::SVector{3, Float64}
    covariance::SMatrix{6, 6, Float64, 36}
    support_count::Int
    confidence::Float64
    sender_id::String
    timestamp::Float64
end

"""Create source update from SLAM source state."""
function OnlineSourceUpdate(source::SlamSourceState, sender_id::String, timestamp::Float64)
    conf = source.is_probationary ? 0.5 : 0.9

    OnlineSourceUpdate(
        source.source_id,
        source_position(source),
        source_moment(source),
        source.covariance,
        source.support_count,
        conf,
        sender_id,
        timestamp
    )
end

# ============================================================================
# Fleet Learning State
# ============================================================================

"""
    PeerLearningState

State tracking for a single peer vehicle.
"""
mutable struct PeerLearningState
    peer_id::String
    last_update_received::Float64
    last_update_sent::Float64
    tiles_received::Int
    tiles_sent::Int
    sources_received::Int
    sources_sent::Int
    fusion_successes::Int
    fusion_rejections::Int
    peer_version::Int
    peer_nees::Float64
    peer_health::Symbol
end

function PeerLearningState(peer_id::String)
    PeerLearningState(
        peer_id, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 1.0, :unknown
    )
end

"""
    FleetLearningState

State of fleet learning system.
"""
mutable struct FleetLearningState
    peers::Dict{String, PeerLearningState}
    bandwidth_used_kbps::Float64
    cycle_count::Int
    tiles_shared_total::Int
    sources_shared_total::Int
    fusions_successful::Int
    fusions_rejected::Int
end

function FleetLearningState()
    FleetLearningState(
        Dict{String, PeerLearningState}(),
        0.0, 0, 0, 0, 0, 0
    )
end

# ============================================================================
# Online Fleet Learning Manager
# ============================================================================

"""
    OnlineFleetLearning

Manager for fleet-based online learning.

# Architecture
- Tracks peer vehicles and their states
- Selects tiles/sources to share based on priority
- Applies CI fusion to received updates
- Coordinates with safety controller

# Usage
```julia
fleet = OnlineFleetLearning()

# On slow loop:
# 1. Generate updates to send
outgoing = generate_updates(fleet, slam_state, peer_positions)

# 2. Process received updates
for update in received_updates
    result = process_update!(fleet, slam_state, update)
end
```
"""
mutable struct OnlineFleetLearning
    config::FleetLearningConfig
    state::FleetLearningState
    my_vehicle_id::String

    # Pending updates (buffered for bandwidth management)
    pending_tile_updates::Vector{OnlineMapUpdate}
    pending_source_updates::Vector{OnlineSourceUpdate}
end

"""Create fleet learning manager."""
function OnlineFleetLearning(vehicle_id::String,
                              config::FleetLearningConfig = DEFAULT_FLEET_LEARNING_CONFIG)
    OnlineFleetLearning(
        config,
        FleetLearningState(),
        vehicle_id,
        OnlineMapUpdate[],
        OnlineSourceUpdate[]
    )
end

# ============================================================================
# Update Generation
# ============================================================================

"""
    FleetUpdatePriority

Priority for sharing an update.
"""
struct FleetUpdatePriority
    tile_id::Union{MapTileID, Nothing}
    source_id::Union{Int, Nothing}
    priority_score::Float64
    information_gain::Float64
    bytes::Int
end

"""
    generate_updates(fleet::OnlineFleetLearning, slam_state::SlamAugmentedState,
                     peer_positions::Dict{String, Vec3Map}, timestamp::Float64;
                     nees::Float64 = 1.0, health::Symbol = :healthy)

Generate updates to share with fleet.

Returns list of updates prioritized by expected information gain.
"""
function generate_updates(fleet::OnlineFleetLearning, slam_state::SlamAugmentedState,
                           peer_positions::Dict{String, Vec3Map}, timestamp::Float64;
                           nees::Float64 = 1.0, health::Symbol = :healthy)
    config = fleet.config

    # Check safety conditions
    if config.require_validated_checkpoint && health != :healthy
        return (tiles = OnlineMapUpdate[], sources = OnlineSourceUpdate[])
    end

    tile_updates = OnlineMapUpdate[]
    source_updates = OnlineSourceUpdate[]

    # Generate tile updates
    for (tile_id, tile) in slam_state.tile_states
        # Check quality threshold
        quality = compute_online_tile_quality(tile, nees)
        if quality < config.min_quality_to_share
            continue
        end

        # Check observation count
        if tile.observation_count < config.min_observation_count
            continue
        end

        # Skip probationary tiles
        if tile.is_probationary
            continue
        end

        update = OnlineMapUpdate(tile, nees, health, timestamp)
        push!(tile_updates, update)
    end

    # Generate source updates (if enabled)
    if config.share_sources
        for source in slam_state.source_states
            if source.lifecycle != :active
                continue
            end

            if source.is_probationary
                continue
            end

            conf = 1.0 - source.is_probationary * 0.5
            if conf < config.min_source_confidence
                continue
            end

            update = OnlineSourceUpdate(source, fleet.my_vehicle_id, timestamp)
            push!(source_updates, update)
        end
    end

    # Prioritize by expected information gain
    tile_updates = prioritize_tile_updates(tile_updates, peer_positions, config)
    source_updates = prioritize_source_updates(source_updates, peer_positions, config)

    # Limit by bandwidth budget
    tile_updates, source_updates = apply_bandwidth_limit(
        tile_updates, source_updates, config
    )

    return (tiles = tile_updates, sources = source_updates)
end

"""Prioritize tile updates by expected information gain."""
function prioritize_tile_updates(updates::Vector{OnlineMapUpdate},
                                  peer_positions::Dict{String, Vec3Map},
                                  config::FleetLearningConfig)
    if isempty(updates)
        return updates
    end

    # Score each update
    scored = [(u, score_tile_update(u, peer_positions, config)) for u in updates]

    # Sort by score (descending)
    sort!(scored, by = x -> x[2], rev = true)

    # Return top updates
    n = min(length(scored), config.max_messages_per_cycle)
    return [s[1] for s in scored[1:n]]
end

"""Score a tile update for prioritization."""
function score_tile_update(update::OnlineMapUpdate, peer_positions::Dict{String, Vec3Map},
                            config::FleetLearningConfig)
    # Base score from quality
    score = update.quality_score

    # Information gain: trace of information matrix
    info_trace = tr(update.information)
    score *= log1p(info_trace)

    # Proximity bonus if peers are near this tile
    tile_center = tile_center_from_id(update.tile_id)
    min_dist = Inf
    for (_, pos) in peer_positions
        d = norm(Vec3Map(tile_center) - pos)
        min_dist = min(min_dist, d)
    end

    # Bonus for nearby peers (within 100m)
    if min_dist < 100.0
        score *= 1.5
    end

    return score
end

"""Prioritize source updates."""
function prioritize_source_updates(updates::Vector{OnlineSourceUpdate},
                                    peer_positions::Dict{String, Vec3Map},
                                    config::FleetLearningConfig)
    if isempty(updates)
        return updates
    end

    # Score by confidence and support count
    scored = [(u, u.confidence * log1p(u.support_count)) for u in updates]
    sort!(scored, by = x -> x[2], rev = true)

    n = min(length(scored), config.max_messages_per_cycle)
    return [s[1] for s in scored[1:n]]
end

"""Apply bandwidth limit to updates."""
function apply_bandwidth_limit(tile_updates::Vector{OnlineMapUpdate},
                                source_updates::Vector{OnlineSourceUpdate},
                                config::FleetLearningConfig)
    # Estimate bytes per update
    bytes_per_tile = 200  # coefficients + information matrix
    bytes_per_source = 100  # position + moment + covariance

    # Total budget per cycle (assuming 1 Hz cycle rate)
    budget_bytes = config.bandwidth_budget_kbps * 1000 / 8

    used = 0
    limited_tiles = OnlineMapUpdate[]
    for u in tile_updates
        if used + bytes_per_tile > budget_bytes
            break
        end
        push!(limited_tiles, u)
        used += bytes_per_tile
    end

    limited_sources = OnlineSourceUpdate[]
    for u in source_updates
        if used + bytes_per_source > budget_bytes
            break
        end
        push!(limited_sources, u)
        used += bytes_per_source
    end

    return (limited_tiles, limited_sources)
end

"""Get tile center from tile ID (placeholder - use actual tile indexing)."""
function tile_center_from_id(tile_id::MapTileID)
    # Simplified: decode from tile_id string or struct
    return Vec3Map(0.0, 0.0, 0.0)  # Placeholder
end

# ============================================================================
# Update Processing
# ============================================================================

"""
    FleetFusionResult

Result of fusing a fleet update.
"""
struct FleetFusionResult
    accepted::Bool
    tile_id::Union{MapTileID, Nothing}
    source_id::Union{Int, Nothing}
    rejection_reason::Symbol  # :none, :version_skew, :low_quality, :degraded_peer, :safety
    covariance_reduction::Float64
    information_gain::Float64
end

"""
    process_tile_update!(fleet::OnlineFleetLearning, slam_state::SlamAugmentedState,
                          update::OnlineMapUpdate, timestamp::Float64) -> FleetFusionResult

Process received tile update from peer.

Uses Covariance Intersection for conservative fusion.
"""
function process_tile_update!(fleet::OnlineFleetLearning, slam_state::SlamAugmentedState,
                               update::OnlineMapUpdate, timestamp::Float64)
    config = fleet.config

    # Get or create peer state
    sender = "peer"  # Would extract from message in real implementation
    if !haskey(fleet.state.peers, sender)
        fleet.state.peers[sender] = PeerLearningState(sender)
    end
    peer = fleet.state.peers[sender]

    # Check peer health
    if config.reject_degraded_peers && update.sender_health == :degraded
        peer.fusion_rejections += 1
        fleet.state.fusions_rejected += 1
        return FleetFusionResult(false, update.tile_id, nothing, :degraded_peer, 0.0, 0.0)
    end

    # Check version skew
    local_tile = get_tile_state(slam_state, update.tile_id)
    if local_tile !== nothing
        version_diff = abs(local_tile.version - update.version)
        if version_diff > config.max_version_skew
            peer.fusion_rejections += 1
            fleet.state.fusions_rejected += 1
            return FleetFusionResult(false, update.tile_id, nothing, :version_skew, 0.0, 0.0)
        end
    end

    # Check quality
    if update.quality_score < config.min_quality_to_share
        peer.fusion_rejections += 1
        fleet.state.fusions_rejected += 1
        return FleetFusionResult(false, update.tile_id, nothing, :low_quality, 0.0, 0.0)
    end

    # Perform CI fusion
    if local_tile === nothing
        # No local tile - create from update
        new_tile = create_tile_from_update(update)
        slam_state.tile_states[update.tile_id] = new_tile
        cov_reduction = 1.0
        info_gain = tr(update.information)
    else
        # Fuse with local tile
        old_trace = tr(local_tile.covariance)

        fuse_tile_ci!(local_tile, update)

        new_trace = tr(local_tile.covariance)
        cov_reduction = (old_trace - new_trace) / old_trace
        info_gain = tr(update.information)
    end

    # Update statistics
    peer.tiles_received += 1
    peer.fusion_successes += 1
    peer.last_update_received = timestamp
    peer.peer_version = update.version
    peer.peer_nees = update.sender_nees
    peer.peer_health = update.sender_health

    fleet.state.fusions_successful += 1

    return FleetFusionResult(true, update.tile_id, nothing, :none, cov_reduction, info_gain)
end

"""Create tile state from received update."""
function create_tile_from_update(update::OnlineMapUpdate)
    n = length(update.coefficients)
    cov = inv(update.information + 1e-12 * I(n))

    SlamTileState(
        update.tile_id,
        tile_center_from_id(update.tile_id),
        DEFAULT_TILE_SCALE,  # scale (tile half-width)
        mode_from_dim(length(update.coefficients)),  # model_mode
        copy(update.coefficients),
        cov,
        copy(update.information),
        update.observation_count,
        update.timestamp,
        update.version,
        true,  # Mark as probationary until validated locally
        Vec3Map(0.0, 0.0, 0.0),  # position_bbox_min
        Vec3Map(0.0, 0.0, 0.0),  # position_bbox_max
        false,  # tier2_active (fleet tiles start locked)
        0       # tier2_relock_count
    )
end

"""Fuse update into local tile using CI."""
function fuse_tile_ci!(local_tile::SlamTileState, update::OnlineMapUpdate)
    # CI fusion in information form:
    # I_fused = ω·I_local + (1-ω)·I_peer
    # α_fused = I_fused⁻¹ · (ω·I_local·α_local + (1-ω)·I_peer·α_peer)

    I_local = local_tile.information
    α_local = local_tile.coefficients
    I_peer = update.information
    α_peer = update.coefficients

    # Optimize ω to minimize trace (simplified - use fixed ω for now)
    ω = optimize_ci_omega(I_local, I_peer)

    # Fuse information matrices
    I_fused = ω * I_local + (1 - ω) * I_peer

    # Fuse state vectors (information-weighted)
    i_fused = ω * (I_local * α_local) + (1 - ω) * (I_peer * α_peer)

    # Recover state and covariance
    I_fused_reg = I_fused + 1e-12 * I(size(I_fused, 1))
    P_fused = inv(I_fused_reg)
    α_fused = P_fused * i_fused

    # Update tile
    local_tile.coefficients = α_fused
    local_tile.covariance = P_fused
    local_tile.information = I_fused
    local_tile.observation_count += update.observation_count ÷ 2  # Conservative
    local_tile.version += 1
end

"""Optimize CI omega to minimize trace."""
function optimize_ci_omega(I1::AbstractMatrix, I2::AbstractMatrix)
    # Simplified: use quality-weighted blend
    # In full implementation, optimize det(P_fused) or tr(P_fused)

    tr1 = tr(I1)
    tr2 = tr(I2)

    if tr1 + tr2 == 0
        return 0.5
    end

    # Weight by information
    ω = tr1 / (tr1 + tr2)
    return clamp(ω, 0.1, 0.9)
end

"""
    process_source_update!(fleet::OnlineFleetLearning, slam_state::SlamAugmentedState,
                            update::OnlineSourceUpdate, timestamp::Float64) -> FleetFusionResult

Process received source update from peer.
"""
function process_source_update!(fleet::OnlineFleetLearning, slam_state::SlamAugmentedState,
                                 update::OnlineSourceUpdate, timestamp::Float64)
    config = fleet.config

    # Try to match with existing source
    matched_source = nothing
    for src in slam_state.source_states
        dist = norm(source_position(src) - update.position)
        if dist < config.source_matching_threshold_m
            matched_source = src
            break
        end
    end

    if matched_source !== nothing
        # Fuse with existing source
        fuse_source_ci!(matched_source, update)
        return FleetFusionResult(true, nothing, matched_source.source_id, :none, 0.0, 0.0)
    else
        # Create new source from update
        max_id = isempty(slam_state.source_states) ? 0 :
                 maximum(s.source_id for s in slam_state.source_states)

        new_source = SlamSourceState(
            max_id + 1000,  # Offset for fleet sources
            update.position,
            update.moment
        )
        new_source.covariance = update.covariance
        new_source.support_count = update.support_count
        new_source.lifecycle = :candidate  # Needs local validation
        new_source.is_probationary = true

        push!(slam_state.source_states, new_source)

        fleet.state.sources_shared_total += 1

        return FleetFusionResult(true, nothing, new_source.source_id, :none, 0.0, 0.0)
    end
end

"""Fuse source update with local source using CI."""
function fuse_source_ci!(local_source::SlamSourceState, update::OnlineSourceUpdate)
    # CI fusion for 6-DOF source state
    P_local = Matrix(local_source.covariance)
    x_local = Vector(local_source.state)
    P_peer = Matrix(update.covariance)
    x_peer = vcat(Vector(update.position), Vector(update.moment))

    # Information form
    I_local = inv(P_local + 1e-12 * I(6))
    I_peer = inv(P_peer + 1e-12 * I(6))

    # Optimize omega
    ω = optimize_ci_omega(I_local, I_peer)

    # Fuse
    I_fused = ω * I_local + (1 - ω) * I_peer
    i_fused = ω * (I_local * x_local) + (1 - ω) * (I_peer * x_peer)

    P_fused = inv(I_fused + 1e-12 * I(6))
    x_fused = P_fused * i_fused

    local_source.state = SVector{6}(x_fused)
    local_source.covariance = SMatrix{6, 6}(P_fused)
    local_source.support_count += update.support_count ÷ 2
end

# ============================================================================
# Fleet Convergence Tracking
# ============================================================================

"""
    FleetConvergenceMetrics

Metrics for fleet-wide learning convergence.
"""
struct FleetConvergenceMetrics
    n_vehicles::Int
    mean_map_trace::Float64
    std_map_trace::Float64
    mean_nees::Float64
    std_nees::Float64
    fusion_rate::Float64
    convergence_progress::Float64
end

"""Compute fleet-wide convergence metrics."""
function compute_fleet_convergence(fleet::OnlineFleetLearning)
    peers = values(fleet.state.peers)
    n = length(peers)

    if n == 0
        return FleetConvergenceMetrics(1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    end

    nees_values = [p.peer_nees for p in peers if p.peer_nees > 0]

    mean_nees = isempty(nees_values) ? 1.0 : mean(nees_values)

    total_fusions = fleet.state.fusions_successful + fleet.state.fusions_rejected
    fusion_rate = total_fusions > 0 ?
        fleet.state.fusions_successful / total_fusions : 0.0

    # Convergence progress based on fusion success rate and NEES
    progress = fusion_rate * (2.0 / (1.0 + mean_nees))

    FleetConvergenceMetrics(
        n + 1,  # Include self
        0.0, 0.0,  # Trace metrics would need state access
        mean_nees,
        0.0,  # std would need more data
        fusion_rate,
        clamp(progress, 0.0, 1.0)
    )
end

# ============================================================================
# Statistics and Formatting
# ============================================================================

"""
    FleetLearningStatistics

Statistics for fleet learning monitoring.
"""
struct FleetLearningStatistics
    n_peers::Int
    tiles_shared::Int
    sources_shared::Int
    fusions_successful::Int
    fusions_rejected::Int
    fusion_rate::Float64
    bandwidth_used_kbps::Float64
end

"""Get fleet learning statistics."""
function get_fleet_statistics(fleet::OnlineFleetLearning)
    total = fleet.state.fusions_successful + fleet.state.fusions_rejected
    rate = total > 0 ? fleet.state.fusions_successful / total : 0.0

    FleetLearningStatistics(
        length(fleet.state.peers),
        fleet.state.tiles_shared_total,
        fleet.state.sources_shared_total,
        fleet.state.fusions_successful,
        fleet.state.fusions_rejected,
        rate,
        fleet.state.bandwidth_used_kbps
    )
end

"""Format fleet learning statistics."""
function format_fleet_statistics(stats::FleetLearningStatistics)
    return """
    Fleet Learning Statistics
    =========================
    Peers: $(stats.n_peers)
    Tiles shared: $(stats.tiles_shared)
    Sources shared: $(stats.sources_shared)

    Fusions: $(stats.fusions_successful) successful, $(stats.fusions_rejected) rejected
    Fusion rate: $(round(stats.fusion_rate * 100, digits=1))%

    Bandwidth: $(round(stats.bandwidth_used_kbps, digits=2)) kbps
    """
end

# ============================================================================
# Exports
# ============================================================================

export FleetLearningConfig, DEFAULT_FLEET_LEARNING_CONFIG
export OnlineMapUpdate, OnlineSourceUpdate
export compute_online_tile_quality
export PeerLearningState, FleetLearningState
export OnlineFleetLearning
export generate_updates, FleetUpdatePriority
export FleetFusionResult, process_tile_update!, process_source_update!
export fuse_tile_ci!, fuse_source_ci!, optimize_ci_omega
export FleetConvergenceMetrics, compute_fleet_convergence
export FleetLearningStatistics, get_fleet_statistics, format_fleet_statistics
