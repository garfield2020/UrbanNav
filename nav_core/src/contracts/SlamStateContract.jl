# ============================================================================
# SlamStateContract.jl - Online MagSLAM State Definitions (Phase C)
# ============================================================================
#
# Defines the augmented state for joint inference over trajectory + map.
#
# State Partitioning:
# A. Vehicle state x_k (from StateContract.jl):
#    - Position p_k (3) in NED world frame [m]
#    - Velocity v_k (3) in world frame [m/s]
#    - Orientation q_k (4 stored, 3 error-state) world-to-body
#    - Gyro bias b_ω (3) [rad/s]
#    - Accel bias b_a (3) [m/s²]
#    Error-state dimension: 15
#
# B. Map state per tile m_j:
#    - Coefficients α_j (n_coef) harmonic basis coefficients
#    - Covariance P_αj (n_coef × n_coef)
#    n_coef depends on basis order (Maxwell-consistent tile counts):
#      CONSTANT=3, LINEAR=8, QUADRATIC=15
#    (Canonical source: BasisEnrichment.COEFFICIENT_COUNTS)
#
# C. Source state per confirmed dipole s_ℓ:
#    - Position μ_ℓ[1:3] (3) in NED world frame [m]
#    - Moment μ_ℓ[4:6] (3) magnetic moment [A·m²]
#    - Covariance P_ℓ (6 × 6)
#    State dimension per source: 6
#
# Full SLAM state dimension:
#   dim = 15 + Σ_j n_coef_j + 6 × n_sources
#
# Architecture Constraint:
# - Online map updater has NO access to truth world
# - All information comes through measurements and their statistics
# - This contract is for real-time online inference
#
# ============================================================================

using LinearAlgebra
using StaticArrays
using Dates

# ============================================================================
# SLAM Configuration
# ============================================================================

"""
    SlamMode

Operating mode for the SLAM system.

- `SLAM_FROZEN`: Map is frozen, navigation only (Phase A mode)
- `SLAM_ONLINE`: Online map learning enabled (Phase C mode)
- `SLAM_SURVEY`: High-rate map updates for survey missions
"""
@enum SlamMode begin
    SLAM_FROZEN = 1   # Map immutable, nav-only
    SLAM_ONLINE = 2   # Online learning enabled
    SLAM_SURVEY = 3   # Survey mode (aggressive updates)
end

"""
    SlamConfig

Configuration for SLAM system behavior.

# Fields
- `mode::SlamMode`: Operating mode (default: SLAM_FROZEN per INV-02)
- `online_learning_enabled::Bool`: Master switch for online learning
- `source_tracking_enabled::Bool`: Enable source (dipole) SLAM
- `rollback_enabled::Bool`: Enable automatic rollback on NEES violation
- `max_sources::Int`: Maximum tracked sources (memory bound)
- `keyframe_interval_s::Float64`: Keyframe creation interval [s]
- `map_update_batch_size::Int`: Observations per batch update
- `min_observability_sv::Float64`: Minimum singular value for updates

# Safety Invariants
- online_learning_enabled defaults to false (INV-02)
- rollback_enabled defaults to true (INV-08)

# Physics Constraints
- keyframe_interval_s ∈ [0.5, 2.0] per timing contract
- max_sources bounded by real-time compute budget
"""
struct SlamConfig
    mode::SlamMode
    online_learning_enabled::Bool
    source_tracking_enabled::Bool
    rollback_enabled::Bool
    max_sources::Int
    keyframe_interval_s::Float64
    map_update_batch_size::Int
    min_observability_sv::Float64

    function SlamConfig(;
        mode::SlamMode = SLAM_FROZEN,
        online_learning_enabled::Bool = false,  # INV-02: Default OFF
        source_tracking_enabled::Bool = false,
        rollback_enabled::Bool = true,          # INV-08: Always available
        max_sources::Int = 50,
        keyframe_interval_s::Float64 = 1.0,
        map_update_batch_size::Int = 10,
        min_observability_sv::Float64 = 1e-6
    )
        # Validate constraints
        @assert 0.5 <= keyframe_interval_s <= 2.0 "Keyframe interval must be in [0.5, 2.0]s"
        @assert max_sources > 0 "max_sources must be positive"
        @assert map_update_batch_size > 0 "batch_size must be positive"
        @assert min_observability_sv > 0 "min_observability_sv must be positive"

        # Mode consistency
        if mode == SLAM_FROZEN
            @assert !online_learning_enabled "SLAM_FROZEN mode requires online_learning_enabled=false"
        end

        new(mode, online_learning_enabled, source_tracking_enabled, rollback_enabled,
            max_sources, keyframe_interval_s, map_update_batch_size, min_observability_sv)
    end
end

"""Default SLAM configuration (Phase C disabled)."""
const DEFAULT_SLAM_CONFIG = SlamConfig()

"""Check if online learning is active."""
is_online_learning(c::SlamConfig) = c.online_learning_enabled && c.mode != SLAM_FROZEN

"""Check if source tracking is active."""
is_source_tracking(c::SlamConfig) = c.source_tracking_enabled && is_online_learning(c)

# ============================================================================
# Map Tile State (Online)
# ============================================================================

"""
    SlamTileState

Online state for a single map tile with versioning.

# Fields
- `tile_id::MapTileID`: Tile identifier
- `coefficients::Vector{Float64}`: Harmonic basis coefficients α_j
- `covariance::Matrix{Float64}`: Coefficient covariance P_αj
- `information::Matrix{Float64}`: Information matrix P_αj⁻¹ (for incremental updates)
- `observation_count::Int`: Number of observations incorporated
- `last_update_time::Float64`: Timestamp of last update [s]
- `version::Int`: Monotonic version counter
- `is_probationary::Bool`: True if tile is in quarantine (INV-08)

# Tile Coefficient Dimensions (Maxwell-consistent, canonical source: BasisEnrichment)
- CONSTANT basis: 3 coefficients (B₀)
- LINEAR basis: 8 coefficients (3 field + 5 traceless symmetric gradient)
- QUADRATIC basis: 15 coefficients (8 linear + 7 quadratic solid harmonics)

# Physics Notes
- Coefficients expand magnetic scalar potential: Φ = Σᵢ αᵢ φᵢ(x - center)
- Each φᵢ satisfies Laplace equation: ∇²φᵢ = 0 (source-free region)
- Field B = -∇Φ automatically satisfies Maxwell (∇·B = 0)
"""
mutable struct SlamTileState
    tile_id::MapTileID
    center::Vec3Map
    scale::Float64               # Tile half-width L [m] for coordinate normalization (default 25.0)
    model_mode::MapModelMode     # Explicit model mode (NOT inferred from coefficient length)
    coefficients::Vector{Float64}
    covariance::Matrix{Float64}
    information::Matrix{Float64}
    observation_count::Int
    last_update_time::Float64
    version::Int
    is_probationary::Bool
    position_bbox_min::Vec3Map   # Running min of observation positions (relative to center)
    position_bbox_max::Vec3Map   # Running max
    tier2_active::Bool           # Quadratic dims (9:15) active in solve + prediction
    tier2_relock_count::Int      # Number of re-locks (diagnostics)
end

# Default tile scale (half-width of 50m tile)
const DEFAULT_TILE_SCALE = 25.0

"""
    get_local_frame(tile::SlamTileState) -> TileLocalFrame

Get the local coordinate frame for this tile.
Used for normalized coordinate computations in MapBasisContract.
"""
get_local_frame(tile::SlamTileState) = TileLocalFrame(tile.center, tile.scale)

"""
    active_mode(tile::SlamTileState) -> MapModelMode

The model mode currently active for solving/prediction.
This is the tile's declared model_mode, NOT inferred from coefficient length.
Use this instead of checking `tile_state_dim(tile) >= 8` etc.
"""
active_mode(tile::SlamTileState) = tile.model_mode

"""Create initial tile state from frozen tile."""
function SlamTileState(tile::MapTileData; version::Int = 0)
    n = length(tile.coefficients)
    info = inv(tile.covariance + 1e-12 * I)  # Regularize for invertibility

    SlamTileState(
        tile.id,
        tile.center,
        DEFAULT_TILE_SCALE,  # scale (tile half-width)
        mode_from_dim(n),    # model_mode inferred from frozen tile
        copy(tile.coefficients),
        copy(tile.covariance),
        info,
        tile.observation_count,
        0.0,  # last_update_time
        version,
        false,  # not probationary
        Vec3Map(0.0, 0.0, 0.0),  # position_bbox_min
        Vec3Map(0.0, 0.0, 0.0),  # position_bbox_max
        false,  # tier2_active
        0       # tier2_relock_count
    )
end

"""Create empty tile state (for new tiles).

Prior variances are computed from a PriorPolicy (physical units) and tile scale L.
This ensures priors are physically meaningful regardless of normalization convention.

See `PriorPolicy` docstring for physical rationale and unit conversion.

# Legacy keyword interface
The old `prior_variance` / `gradient_prior_variance` / `quadratic_prior_variance`
keywords are still accepted for backward compatibility. They specify variances
directly in normalized units (no L conversion). Prefer `prior_policy` for new code.
"""
function SlamTileState(tile_id::MapTileID, center::Vec3Map, n_coef::Int;
                       scale::Float64 = DEFAULT_TILE_SCALE,
                       prior_policy::PriorPolicy = DEFAULT_PRIOR_POLICY,
                       # Legacy interface (overrides prior_policy if any are provided)
                       prior_variance::Union{Float64, Nothing} = nothing,
                       gradient_prior_variance::Union{Float64, Nothing} = nothing,
                       quadratic_prior_variance::Union{Float64, Nothing} = nothing)
    coef = zeros(n_coef)
    mode = mode_from_dim(n_coef)

    # Compute prior variances from policy (physical units → normalized units)
    pv = prior_variances(prior_policy, mode, scale)

    # Legacy overrides: if caller provided explicit variances, use them directly
    # (these are already in normalized units by convention)
    if prior_variance !== nothing
        pv[1:min(3, n_coef)] .= prior_variance
    end
    if gradient_prior_variance !== nothing && n_coef >= 8
        pv[4:8] .= gradient_prior_variance
    end
    if quadratic_prior_variance !== nothing && n_coef >= 15
        pv[9:15] .= quadratic_prior_variance
    end

    cov = diagm(pv)
    info = diagm(1.0 ./ pv)

    SlamTileState(tile_id, center, scale, mode,
                  coef, cov, info, 0, 0.0, 0, true,
                  Vec3Map(0.0, 0.0, 0.0), Vec3Map(0.0, 0.0, 0.0),
                  false, 0)  # probationary, tier2 locked
end

"""Tile coefficient dimension."""
tile_state_dim(t::SlamTileState) = length(t.coefficients)

"""Convert back to frozen tile data."""
function to_tile_data(t::SlamTileState)
    MapTileData(t.tile_id, t.center, t.coefficients, t.covariance, t.observation_count)
end

# ============================================================================
# Source State (Dipole SLAM)
# ============================================================================

"""
    SlamSourceState

State for a tracked magnetic source (dipole).

# Fields
- `source_id::Int`: Unique source identifier
- `state::SVector{6, Float64}`: [x, y, z, mx, my, mz] position + moment
- `covariance::SMatrix{6, 6, Float64, 36}`: 6×6 state covariance
- `lifecycle::Symbol`: :candidate, :active, :retired, :demoted
- `support_count::Int`: Number of supporting observations
- `last_observed::Float64`: Timestamp of last observation [s]
- `is_probationary::Bool`: True if source is in quarantine

# State Vector [6]
- state[1:3]: Position in NED world frame [m]
- state[4:6]: Magnetic moment [A·m²]

# Physics Notes
- Dipole field: B(r) = (μ₀/4π) [3(m·r̂)r̂ - m] / |r|³
- Falls off as 1/r³ (not absorbed by smooth tile basis)
- Confirmed sources become persistent map entities
"""
mutable struct SlamSourceState
    source_id::Int
    state::SVector{6, Float64}
    covariance::SMatrix{6, 6, Float64, 36}
    lifecycle::Symbol
    support_count::Int
    last_observed::Float64
    is_probationary::Bool
end

"""Create source from dipole feature node."""
function SlamSourceState(node::DipoleFeatureNode)
    state = SVector{6}(vcat(node.state.position, node.state.moment))
    cov = SMatrix{6, 6}(node.covariance)

    lifecycle = if node.lifecycle == DIPOLE_CANDIDATE
        :candidate
    elseif node.lifecycle == DIPOLE_ACTIVE
        :active
    elseif node.lifecycle == DIPOLE_RETIRED
        :retired
    else
        :demoted
    end

    SlamSourceState(
        node.id,
        state,
        cov,
        lifecycle,
        node.support_count,
        node.last_observed,
        lifecycle == :candidate
    )
end

"""Create new candidate source."""
function SlamSourceState(id::Int, position::AbstractVector, moment::AbstractVector;
                         position_var::Float64 = 100.0, moment_var::Float64 = 10000.0)
    state = SVector{6}(vcat(position, moment))
    cov = SMatrix{6, 6}(Diagonal([fill(position_var, 3); fill(moment_var, 3)]))

    SlamSourceState(id, state, cov, :candidate, 0, 0.0, true)
end

"""Source state dimension (constant = 6)."""
const SOURCE_STATE_DIM = 6

"""Position component of source state."""
source_position(s::SlamSourceState) = s.state[SVector{3}(1, 2, 3)]

"""Moment component of source state."""
source_moment(s::SlamSourceState) = s.state[SVector{3}(4, 5, 6)]

"""Convert to DipoleFeatureState for field computation."""
function to_dipole_state(s::SlamSourceState)
    DipoleFeatureState(source_position(s), source_moment(s))
end

# ============================================================================
# Augmented SLAM State
# ============================================================================

"""
    SlamAugmentedState

Full SLAM state combining navigation + map + sources.

# Fields
- `nav_state::UrbanNavState`: Vehicle navigation state (15 error-state DOF)
- `tile_states::Dict{MapTileID, SlamTileState}`: Online tile states
- `source_states::Vector{SlamSourceState}`: Tracked source states
- `config::SlamConfig`: SLAM configuration
- `timestamp::Float64`: State timestamp [s]
- `version::Int`: Global state version

# State Dimensions
- Navigation: 15 (error-state: pos, vel, rot, bias_g, bias_a)
- Map: Σ_j dim(tile_j) variable per active tiles
- Sources: 6 × n_sources

Total dimension: 15 + Σ_j n_coef_j + 6 × n_sources

# Frame Convention
- All positions in NED world frame [m]
- All fields in Tesla [T]
- All moments in [A·m²]

# Architecture Notes
- nav_state is the PRIMARY output (real-time navigation)
- tile_states updated on slow loop (keyframes)
- source_states updated on track-before-detect cycle
"""
mutable struct SlamAugmentedState
    nav_state::UrbanNavState
    tile_states::Dict{MapTileID, SlamTileState}
    source_states::Vector{SlamSourceState}
    config::SlamConfig
    timestamp::Float64
    version::Int
end

"""Create SLAM state from navigation state and frozen map."""
function SlamAugmentedState(nav_state::UrbanNavState, map::MapModel;
                            config::SlamConfig = DEFAULT_SLAM_CONFIG)
    # Convert frozen tiles to online tiles
    tile_states = Dict{MapTileID, SlamTileState}()

    for (id, tile) in map.tiles
        tile_states[id] = SlamTileState(tile)
    end

    if map.global_tile !== nothing
        tile_states[map.global_tile.id] = SlamTileState(map.global_tile)
    end

    SlamAugmentedState(
        copy(nav_state),
        tile_states,
        SlamSourceState[],
        config,
        nav_state.timestamp,
        0
    )
end

"""Create navigation-only state (no map, no sources)."""
function SlamAugmentedState(nav_state::UrbanNavState; config::SlamConfig = DEFAULT_SLAM_CONFIG)
    SlamAugmentedState(
        copy(nav_state),
        Dict{MapTileID, SlamTileState}(),
        SlamSourceState[],
        config,
        nav_state.timestamp,
        0
    )
end

# ============================================================================
# State Dimension Accounting
# ============================================================================

"""Navigation state dimension (error-state)."""
nav_state_dim(::SlamAugmentedState) = NAV_STATE_DIM  # 15

"""Total map state dimension."""
function map_state_dim(s::SlamAugmentedState)
    sum(tile_state_dim(t) for t in values(s.tile_states); init=0)
end

"""Total source state dimension."""
source_state_dim(s::SlamAugmentedState) = SOURCE_STATE_DIM * length(s.source_states)

"""Number of active sources."""
n_sources(s::SlamAugmentedState) = length(s.source_states)

"""Number of active tiles."""
n_tiles(s::SlamAugmentedState) = length(s.tile_states)

"""Total augmented state dimension."""
function total_state_dim(s::SlamAugmentedState)
    nav_state_dim(s) + map_state_dim(s) + source_state_dim(s)
end

"""
    state_partition(s::SlamAugmentedState) -> NamedTuple

Return named index ranges into the augmented state vector.

# Returns
- `nav`: Range for navigation state (1:15)
- `tiles`: Dict{MapTileID, Range} for each tile
- `sources`: Vector{Range} for each source
"""
function state_partition(s::SlamAugmentedState)
    # Navigation always first
    nav_range = 1:NAV_STATE_DIM
    idx = NAV_STATE_DIM + 1

    # Tiles (sorted by ID for determinism)
    tile_ranges = Dict{MapTileID, UnitRange{Int}}()
    for id in sort(collect(keys(s.tile_states)))
        tile = s.tile_states[id]
        dim = tile_state_dim(tile)
        tile_ranges[id] = idx:(idx + dim - 1)
        idx += dim
    end

    # Sources (by ID order)
    source_ranges = UnitRange{Int}[]
    for src in s.source_states
        push!(source_ranges, idx:(idx + SOURCE_STATE_DIM - 1))
        idx += SOURCE_STATE_DIM
    end

    return (nav = nav_range, tiles = tile_ranges, sources = source_ranges)
end

# ============================================================================
# State Extraction
# ============================================================================

"""Extract navigation-only state (for fast loop)."""
get_nav_state(s::SlamAugmentedState) = s.nav_state

"""Extract tile state by ID."""
function get_tile_state(s::SlamAugmentedState, tile_id::MapTileID)
    get(s.tile_states, tile_id, nothing)
end

"""Extract source state by ID."""
function get_source_state(s::SlamAugmentedState, source_id::Int)
    idx = findfirst(src -> src.source_id == source_id, s.source_states)
    isnothing(idx) ? nothing : s.source_states[idx]
end

"""Get active (non-probationary) source count."""
function n_active_sources(s::SlamAugmentedState)
    count(src -> !src.is_probationary && src.lifecycle == :active, s.source_states)
end

"""Get probationary source count."""
function n_probationary_sources(s::SlamAugmentedState)
    count(src -> src.is_probationary, s.source_states)
end

# ============================================================================
# Map Query from SLAM State
# ============================================================================

"""
    query_slam_map(state::SlamAugmentedState, position::AbstractVector) -> MapQueryResult

Query the online map at a position, including source contributions.

Returns predicted field B, gradient G, and associated uncertainties.
Uncertainties include both tile uncertainty and source uncertainty.

# Physics
Total field: B_total = B_background + Σ_ℓ B_dipole(source_ℓ)
"""
function query_slam_map(state::SlamAugmentedState, position::AbstractVector,
                        tile_size::Float64 = 50.0)
    pos = Vec3Map(position...)

    # Get relevant tile
    tile_id = tile_id_at(pos, tile_size)
    tile = get_tile_state(state, tile_id)

    if tile === nothing
        # No tile coverage - return high uncertainty
        return MapQueryResult(
            B_pred = zeros(3),
            G_pred = zeros(3, 3),
            σ_B = 1e-3,  # 1000 nT uncertainty
            σ_G = 1e-6,
            in_coverage = false,
            tile_id = tile_id
        )
    end

    # Background field from tile (placeholder - actual implementation in MapProvider)
    # This would evaluate the harmonic basis at position
    B_bg = Vec3Map(tile.coefficients[1:3]...)  # Simplified: first 3 coefs as B0
    G_bg = Mat3Map(zeros(3, 3))  # Simplified

    # Add source contributions
    B_sources = Vec3Map(0.0, 0.0, 0.0)
    for src in state.source_states
        if src.lifecycle == :active
            dipole = to_dipole_state(src)
            B_sources = B_sources + Vec3Map(feature_field(pos, dipole)...)
        end
    end

    B_total = B_bg + B_sources

    # Uncertainty from tile covariance (simplified projection)
    σ_B_tile = sqrt(tr(tile.covariance[1:3, 1:3]) / 3)
    σ_G_tile = sqrt(tr(tile.covariance[4:min(9, end), 4:min(9, end)]) / 5)

    # Add source uncertainty (conservative)
    σ_B_src = 0.0
    for src in state.source_states
        if src.lifecycle == :active
            # Propagate source uncertainty to field (simplified)
            σ_B_src += sqrt(tr(Matrix(src.covariance)[1:3, 1:3]) / 3) * 1e-9
        end
    end

    MapQueryResult(
        B_pred = B_total,
        G_pred = G_bg,
        σ_B = sqrt(σ_B_tile^2 + σ_B_src^2),
        σ_G = σ_G_tile,
        in_coverage = true,
        tile_id = tile_id
    )
end

# ============================================================================
# Checkpoint and Rollback (INV-08)
# ============================================================================

"""
    SlamCheckpoint

Immutable snapshot of SLAM state for rollback capability.

# Fields
- `timestamp::Float64`: Checkpoint timestamp [s]
- `version::Int`: State version at checkpoint
- `nav_state::UrbanNavState`: Navigation state copy
- `tile_data::Dict{MapTileID, MapTileData}`: Frozen tile snapshots
- `source_data::Vector{Tuple{Int, SVector{6, Float64}, SMatrix{6,6,Float64,36}}}`: Source snapshots
- `nees_at_checkpoint::Float64`: NEES value when checkpoint was created
"""
struct SlamCheckpoint
    timestamp::Float64
    version::Int
    nav_state::UrbanNavState
    tile_data::Dict{MapTileID, MapTileData}
    source_data::Vector{Tuple{Int, SVector{6, Float64}, SMatrix{6, 6, Float64, 36}}}
    nees_at_checkpoint::Float64
end

"""Create checkpoint from current state."""
function create_checkpoint(state::SlamAugmentedState, nees::Float64)
    # Snapshot tiles
    tile_data = Dict{MapTileID, MapTileData}()
    for (id, tile) in state.tile_states
        tile_data[id] = to_tile_data(tile)
    end

    # Snapshot sources
    source_data = [(src.source_id, src.state, src.covariance) for src in state.source_states]

    SlamCheckpoint(
        state.timestamp,
        state.version,
        copy(state.nav_state),
        tile_data,
        source_data,
        nees
    )
end

"""Restore state from checkpoint."""
function restore_from_checkpoint!(state::SlamAugmentedState, checkpoint::SlamCheckpoint)
    # Restore navigation
    state.nav_state = copy(checkpoint.nav_state)

    # Restore tiles
    empty!(state.tile_states)
    for (id, tile) in checkpoint.tile_data
        state.tile_states[id] = SlamTileState(tile; version = checkpoint.version)
    end

    # Restore sources
    empty!(state.source_states)
    for (id, s, cov) in checkpoint.source_data
        push!(state.source_states, SlamSourceState(id, s[1:3], s[4:6]))
        state.source_states[end].covariance = cov
    end

    state.version = checkpoint.version
    state.timestamp = checkpoint.timestamp

    return state
end

# ============================================================================
# Held-Out Prediction Test (Model Adequacy)
# ============================================================================

"""
    HeldOutProbe

A single held-out measurement for model adequacy validation.
Stores the raw observation (position, measured field, noise covariance)
so it can be re-evaluated against different model orders.
"""
struct HeldOutProbe
    position::Vec3Map       # World-frame position [m]
    z::Vector{Float64}      # Measured field [T] (3-vector)
    R::Matrix{Float64}      # Measurement noise covariance [T²] (3×3)
    timestamp::Float64      # [s]
end

"""
    HeldOutBuffer

Per-tile ring buffer of held-out measurements for model adequacy testing.

Every Nth measurement (controlled by `holdout_rate`) is diverted to this buffer
instead of being used for training. The buffer has fixed capacity; oldest probes
are overwritten when full.

# Fields
- `probes::Vector{HeldOutProbe}`: Ring buffer of held-out measurements
- `capacity::Int`: Maximum number of probes
- `write_idx::Int`: Next write position (1-based, wraps)
- `count::Int`: Number of valid probes (≤ capacity)
- `measurement_counter::Int`: Running counter for holdout selection
- `holdout_rate::Int`: Keep every Nth measurement as held-out (default 5 → 20%)
"""
mutable struct HeldOutBuffer
    probes::Vector{HeldOutProbe}
    capacity::Int
    write_idx::Int
    count::Int
    measurement_counter::Int
    holdout_rate::Int
end

"""Create empty held-out buffer with given capacity and holdout rate."""
function HeldOutBuffer(; capacity::Int = 40, holdout_rate::Int = 5)
    HeldOutBuffer(Vector{HeldOutProbe}(undef, capacity), capacity, 1, 0, 0, holdout_rate)
end

"""
    should_holdout(buf::HeldOutBuffer) -> Bool

Decide if the next measurement should be held out (not used for training).
Returns true every `holdout_rate` measurements.
"""
function should_holdout(buf::HeldOutBuffer)
    buf.measurement_counter += 1
    return mod(buf.measurement_counter, buf.holdout_rate) == 0
end

"""Add a probe to the ring buffer."""
function add_probe!(buf::HeldOutBuffer, probe::HeldOutProbe)
    buf.probes[buf.write_idx] = probe
    buf.write_idx = mod1(buf.write_idx + 1, buf.capacity)
    buf.count = min(buf.count + 1, buf.capacity)
end

"""Get all valid probes from the buffer."""
function get_probes(buf::HeldOutBuffer)
    if buf.count < buf.capacity
        return buf.probes[1:buf.count]
    else
        return buf.probes
    end
end

"""
    ModelAdequacyResult

Result of comparing model orders on held-out data.

# Fields
- `χ2_b0::Float64`: Mean whitened χ² for B0-only model (d=3)
- `χ2_linear::Float64`: Mean whitened χ² for linear model (d=8)
- `χ2_quadratic::Float64`: Mean whitened χ² for quadratic model (d=15), or NaN if not applicable
- `n_probes::Int`: Number of held-out probes used
- `linear_adequate::Bool`: True if linear model improves over B0
- `quadratic_beneficial::Bool`: True if quadratic improves over linear
- `recommended_mode::MapModelMode`: Best model order for this tile
"""
struct ModelAdequacyResult
    χ2_b0::Float64
    χ2_linear::Float64
    χ2_quadratic::Float64
    n_probes::Int
    linear_adequate::Bool
    quadratic_beneficial::Bool
    recommended_mode::MapModelMode
end

"""
    evaluate_model_adequacy(tile::SlamTileState, probes::Vector{HeldOutProbe};
                            χ2_miscalibration_threshold::Float64 = 5.0) -> ModelAdequacyResult

Compare prediction quality of B0 vs linear (vs quadratic) models on held-out data.

For each probe, computes the noise-whitened prediction error:
  χ² = r' R⁻¹ r / 3
where r = z - B_pred (prediction residual) and R is the measurement noise.

Using R (not R + H P H') is deliberate: we test whether the model's point prediction
matches the data to within measurement noise. A model that absorbs unmodeled physics
into its coefficients (e.g., gradient absorbing curvature) will produce biased
predictions at held-out positions, yielding χ²/dof >> 1 even though its posterior
covariance is small.

# Adequacy decisions
- A model is "adequate" if its mean χ²/dof < `χ2_miscalibration_threshold` (default 5.0)
  AND it improves over the next-lower model.
- A higher-order model is adequate ONLY if it is well-calibrated on held-out data.
- Recommendation selects the highest adequate model order.

# Returns
ModelAdequacyResult with per-model χ² and recommendation.
"""
function evaluate_model_adequacy(tile::SlamTileState, probes::Vector{HeldOutProbe};
                                  χ2_miscalibration_threshold::Float64 = 5.0)
    n = length(probes)
    n_coef = tile_state_dim(tile)
    L = tile.scale

    χ2_b0_sum = 0.0
    χ2_lin_sum = 0.0
    χ2_quad_sum = 0.0
    has_quadratic = n_coef >= 15 && tile.tier2_active

    for probe in probes
        x̃ = Vec3Map((probe.position - tile.center) / L)
        R_inv = inv(probe.R)

        # B0 prediction (d=3)
        B_b0 = evaluate_field(tile.coefficients, x̃, MODE_B0)
        r_b0 = probe.z - Vector(B_b0)
        χ2_b0_sum += r_b0' * R_inv * r_b0

        # Linear prediction (d=8)
        if n_coef >= 8
            B_lin = evaluate_field(tile.coefficients, x̃, MODE_LINEAR)
            r_lin = probe.z - Vector(B_lin)
            χ2_lin_sum += r_lin' * R_inv * r_lin
        else
            χ2_lin_sum = χ2_b0_sum
        end

        # Quadratic prediction (d=15)
        if has_quadratic
            B_quad = evaluate_field(tile.coefficients, x̃, MODE_QUADRATIC)
            r_quad = probe.z - Vector(B_quad)
            χ2_quad_sum += r_quad' * R_inv * r_quad
        end
    end

    # Normalize: mean χ²/dof (dof = 3 per measurement)
    dof = 3.0
    χ2_b0 = χ2_b0_sum / (n * dof)
    χ2_lin = n_coef >= 8 ? χ2_lin_sum / (n * dof) : χ2_b0
    χ2_quad = has_quadratic ? χ2_quad_sum / (n * dof) : NaN

    # Model adequacy decisions:
    # 1. Linear is adequate if BOTH:
    #    (a) It is well-calibrated (χ²/dof not wildly above 1)
    #    (b) It improves over B0
    linear_adequate = n_coef >= 8 &&
                      χ2_lin < χ2_miscalibration_threshold &&
                      χ2_lin < χ2_b0

    # 2. Quadratic is beneficial if BOTH:
    #    (a) It is well-calibrated
    #    (b) It improves over linear
    quadratic_beneficial = has_quadratic &&
                           χ2_quad < χ2_miscalibration_threshold &&
                           χ2_quad < χ2_lin

    # Recommend highest adequate model
    if quadratic_beneficial
        recommended = MODE_QUADRATIC
    elseif linear_adequate
        recommended = MODE_LINEAR
    else
        recommended = MODE_B0
    end

    return ModelAdequacyResult(χ2_b0, χ2_lin, χ2_quad, n, linear_adequate, quadratic_beneficial, recommended)
end

# ============================================================================
# Exports
# ============================================================================

export HeldOutProbe, HeldOutBuffer, should_holdout, add_probe!, get_probes
export ModelAdequacyResult, evaluate_model_adequacy
export SlamMode, SLAM_FROZEN, SLAM_ONLINE, SLAM_SURVEY
export SlamConfig, DEFAULT_SLAM_CONFIG
export is_online_learning, is_source_tracking
export SlamTileState, tile_state_dim, to_tile_data, get_local_frame, DEFAULT_TILE_SCALE
export active_mode
export SlamSourceState, SOURCE_STATE_DIM
export source_position, source_moment, to_dipole_state
export SlamAugmentedState
export nav_state_dim, map_state_dim, source_state_dim, total_state_dim
export n_sources, n_tiles, n_active_sources, n_probationary_sources
export state_partition
export get_nav_state, get_tile_state, get_source_state
export query_slam_map
export SlamCheckpoint, create_checkpoint, restore_from_checkpoint!
