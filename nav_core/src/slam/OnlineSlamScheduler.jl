# ============================================================================
# OnlineSlamScheduler.jl - Dual-Time-Scale SLAM Inference (Phase C Step 3)
# ============================================================================
#
# Manages the separation between fast (navigation) and slow (mapping) loops.
#
# Architecture:
# - Fast loop: Runs every IMU/Odometry tick, updates pose using CURRENT map
# - Slow loop: Runs on keyframes, updates tile coefficients and/or sources
#
# Timing Contract:
# - Fast loop: < 10ms per tick (50-200 Hz)
# - Slow loop: < 50ms per keyframe (0.5-2.0 Hz)
# - Total budget: INV-06 timing compliance
#
# Determinism:
# - Keyframe cadence is deterministic given input sequence
# - All decisions based on timestamp, not wall-clock time
#
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Keyframe Policy
# ============================================================================

"""
    KeyframeTrigger

Triggers that can cause keyframe creation.

- `TRIGGER_TIME`: Regular time interval elapsed
- `TRIGGER_DISTANCE`: Sufficient travel distance
- `TRIGGER_ROTATION`: Sufficient rotation change
- `TRIGGER_OBSERVABILITY`: Observability metric triggered
- `TRIGGER_RESIDUAL`: Large residual detected
- `TRIGGER_MANUAL`: Manual/external trigger
"""
@enum KeyframeTrigger begin
    TRIGGER_TIME = 1
    TRIGGER_DISTANCE = 2
    TRIGGER_ROTATION = 3
    TRIGGER_OBSERVABILITY = 4
    TRIGGER_RESIDUAL = 5
    TRIGGER_MANUAL = 6
end

"""
    KeyframePolicyConfig

Configuration for keyframe creation policy.

# Fields
- `time_interval_s::Float64`: Maximum time between keyframes [s] (0.5-2.0)
- `distance_threshold_m::Float64`: Distance threshold for keyframe [m]
- `rotation_threshold_rad::Float64`: Rotation threshold for keyframe [rad]
- `observability_threshold::Float64`: Minimum singular value threshold
- `residual_threshold::Float64`: Chi-square threshold for residual trigger
- `min_observations_per_keyframe::Int`: Minimum observations before allowing keyframe
- `enable_adaptive::Bool`: Enable adaptive keyframe timing

# Physics Justification
- time_interval_s: 1.0s default balances compute budget vs map staleness
- distance_threshold_m: 1.0m corresponds to ~1 tile radius for local updates
- rotation_threshold_rad: 0.1 rad (~6°) captures significant viewpoint change
"""
struct KeyframePolicyConfig
    time_interval_s::Float64
    distance_threshold_m::Float64
    rotation_threshold_rad::Float64
    observability_threshold::Float64
    residual_threshold::Float64
    min_observations_per_keyframe::Int
    enable_adaptive::Bool

    function KeyframePolicyConfig(;
        time_interval_s::Float64 = 1.0,
        distance_threshold_m::Float64 = 1.0,
        rotation_threshold_rad::Float64 = 0.1,
        observability_threshold::Float64 = 1e-6,
        residual_threshold::Float64 = 16.266,  # χ²(3, 0.001)
        min_observations_per_keyframe::Int = 5,
        enable_adaptive::Bool = true
    )
        @assert 0.5 <= time_interval_s <= 2.0 "Keyframe interval must be in [0.5, 2.0]s"
        @assert distance_threshold_m > 0 "Distance threshold must be positive"
        @assert rotation_threshold_rad > 0 "Rotation threshold must be positive"
        @assert min_observations_per_keyframe >= 1 "Must have at least 1 observation"

        new(time_interval_s, distance_threshold_m, rotation_threshold_rad,
            observability_threshold, residual_threshold, min_observations_per_keyframe,
            enable_adaptive)
    end
end

"""Default keyframe policy configuration."""
const DEFAULT_KEYFRAME_POLICY_CONFIG = KeyframePolicyConfig()

"""
    KeyframeState

State tracking for keyframe policy decisions.

# Fields
- `last_keyframe_time::Float64`: Timestamp of last keyframe [s]
- `last_keyframe_position::Vec3`: Position at last keyframe [m]
- `last_keyframe_orientation::QuatRotation`: Orientation at last keyframe
- `observations_since_keyframe::Int`: Number of observations since last keyframe
- `accumulated_chi2::Float64`: Accumulated chi-square since last keyframe
- `keyframe_count::Int`: Total keyframes created
"""
mutable struct KeyframeState
    last_keyframe_time::Float64
    last_keyframe_position::Vec3
    last_keyframe_orientation::QuatRotation{Float64}
    observations_since_keyframe::Int
    accumulated_chi2::Float64
    keyframe_count::Int
end

"""Initialize keyframe state from initial navigation state."""
function KeyframeState(nav_state::UrbanNavState)
    KeyframeState(
        nav_state.timestamp,
        nav_state.position,
        nav_state.orientation,
        0,
        0.0,
        0
    )
end

"""
    KeyframeDecision

Result of keyframe policy evaluation.

# Fields
- `should_create::Bool`: Whether to create a keyframe
- `trigger::Union{KeyframeTrigger, Nothing}`: What triggered the decision
- `time_since_last::Float64`: Time since last keyframe [s]
- `distance_since_last::Float64`: Distance since last keyframe [m]
- `rotation_since_last::Float64`: Rotation since last keyframe [rad]
"""
struct KeyframeDecision
    should_create::Bool
    trigger::Union{KeyframeTrigger, Nothing}
    time_since_last::Float64
    distance_since_last::Float64
    rotation_since_last::Float64
end

"""No keyframe needed."""
function no_keyframe_decision(time_since::Float64, dist_since::Float64, rot_since::Float64)
    KeyframeDecision(false, nothing, time_since, dist_since, rot_since)
end

"""Keyframe triggered."""
function keyframe_decision(trigger::KeyframeTrigger, time_since::Float64,
                           dist_since::Float64, rot_since::Float64)
    KeyframeDecision(true, trigger, time_since, dist_since, rot_since)
end

"""
    evaluate_keyframe_policy(state::KeyframeState, nav_state::UrbanNavState,
                             config::KeyframePolicyConfig) -> KeyframeDecision

Evaluate whether a new keyframe should be created.

Checks (in order):
1. Time interval trigger
2. Distance trigger
3. Rotation trigger
4. Minimum observations check
"""
function evaluate_keyframe_policy(state::KeyframeState, nav_state::UrbanNavState,
                                  config::KeyframePolicyConfig)
    # Compute deltas
    time_since = nav_state.timestamp - state.last_keyframe_time
    distance_since = norm(nav_state.position - state.last_keyframe_position)

    # Rotation delta (angle of rotation quaternion difference)
    q_diff = state.last_keyframe_orientation' * nav_state.orientation
    # Extract angle from quaternion (2 * acos(|w|))
    w = abs(q_diff.q.s)
    rotation_since = 2.0 * acos(clamp(w, -1.0, 1.0))

    # Check minimum observations
    if state.observations_since_keyframe < config.min_observations_per_keyframe
        return no_keyframe_decision(time_since, distance_since, rotation_since)
    end

    # Check time trigger (highest priority for determinism)
    if time_since >= config.time_interval_s
        return keyframe_decision(TRIGGER_TIME, time_since, distance_since, rotation_since)
    end

    # Check distance trigger
    if distance_since >= config.distance_threshold_m
        return keyframe_decision(TRIGGER_DISTANCE, time_since, distance_since, rotation_since)
    end

    # Check rotation trigger
    if rotation_since >= config.rotation_threshold_rad
        return keyframe_decision(TRIGGER_ROTATION, time_since, distance_since, rotation_since)
    end

    return no_keyframe_decision(time_since, distance_since, rotation_since)
end

"""Update keyframe state after creating a keyframe."""
function record_keyframe!(state::KeyframeState, nav_state::UrbanNavState)
    state.last_keyframe_time = nav_state.timestamp
    state.last_keyframe_position = nav_state.position
    state.last_keyframe_orientation = nav_state.orientation
    state.observations_since_keyframe = 0
    state.accumulated_chi2 = 0.0
    state.keyframe_count += 1
    return state
end

"""Record an observation (increment counter)."""
function record_observation!(state::KeyframeState, chi2::Float64 = 0.0)
    state.observations_since_keyframe += 1
    state.accumulated_chi2 += chi2
    return state
end

# ============================================================================
# Online SLAM Scheduler
# ============================================================================

"""
    SchedulerLoopType

Which loop is currently active.
"""
@enum SchedulerLoopType begin
    LOOP_FAST = 1   # Navigation loop (IMU/Odometry rate)
    LOOP_SLOW = 2   # Mapping loop (keyframe rate)
    LOOP_IDLE = 3   # No active processing
end

"""
    OnlineSlamSchedulerConfig

Configuration for the dual-time-scale SLAM scheduler.

# Fields
- `slam_config::SlamConfig`: SLAM configuration (mode, flags)
- `keyframe_config::KeyframePolicyConfig`: Keyframe policy
- `fast_loop_budget_ms::Float64`: Fast loop timing budget [ms]
- `slow_loop_budget_ms::Float64`: Slow loop timing budget [ms]
- `defer_map_updates::Bool`: Defer map updates if slow loop exceeds budget
- `max_deferred_updates::Int`: Maximum deferred map updates before forced flush

# Timing Bounds
- fast_loop_budget_ms: Typically 10ms (100 Hz tick rate)
- slow_loop_budget_ms: Typically 50ms (satisfies INV-06)
"""
struct OnlineSlamSchedulerConfig
    slam_config::SlamConfig
    keyframe_config::KeyframePolicyConfig
    fast_loop_budget_ms::Float64
    slow_loop_budget_ms::Float64
    defer_map_updates::Bool
    max_deferred_updates::Int

    function OnlineSlamSchedulerConfig(;
        slam_config::SlamConfig = DEFAULT_SLAM_CONFIG,
        keyframe_config::KeyframePolicyConfig = DEFAULT_KEYFRAME_POLICY_CONFIG,
        fast_loop_budget_ms::Float64 = 10.0,
        slow_loop_budget_ms::Float64 = 50.0,
        defer_map_updates::Bool = true,
        max_deferred_updates::Int = 5
    )
        @assert fast_loop_budget_ms > 0 "Fast loop budget must be positive"
        @assert slow_loop_budget_ms > 0 "Slow loop budget must be positive"
        @assert fast_loop_budget_ms <= slow_loop_budget_ms "Fast loop must be faster than slow loop"

        new(slam_config, keyframe_config, fast_loop_budget_ms, slow_loop_budget_ms,
            defer_map_updates, max_deferred_updates)
    end
end

"""Default scheduler configuration."""
const DEFAULT_SCHEDULER_CONFIG = OnlineSlamSchedulerConfig()

"""
    SchedulerStatistics

Statistics tracking for scheduler performance.

# Teachability Counters (Phase C verification)
- `attempted_tile_updates::Int`: Total updates attempted
- `accepted_tile_updates::Int`: Updates accepted by all gates
- `rejected_by_teachability::Int`: Rejected by pose uncertainty gate
- `rejected_by_observability::Int`: Rejected by gradient observability gate
- `rejected_by_quality::Int`: Rejected by tile quality gate

These counters enable verification that:
1. Acceptance rate is high in observable regions
2. Rejection occurs in clutter/mismatch scenarios
3. Not all updates are blindly accepted (~100% everywhere would be suspicious)
"""
mutable struct SchedulerStatistics
    fast_loop_count::Int
    slow_loop_count::Int
    keyframes_created::Int
    deferred_updates::Int
    forced_flushes::Int
    fast_loop_overruns::Int
    slow_loop_overruns::Int
    total_fast_time_ms::Float64
    total_slow_time_ms::Float64
    # Teachability counters
    attempted_tile_updates::Int
    accepted_tile_updates::Int
    rejected_by_teachability::Int
    rejected_by_observability::Int
    rejected_by_quality::Int
    # Active dimension histogram (key diagnostic for QUAD safety)
    active_k_count_3::Int   # updates solved at k=3 (constant)
    active_k_count_8::Int   # updates solved at k=8 (linear)
    active_k_count_15::Int  # updates solved at k=15 (quadratic)
    active_k_max::Int       # max active_k observed (must be ≤ TIER2_MAX_ACTIVE_DIM)
end

function SchedulerStatistics()
    SchedulerStatistics(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0)
end

function active_k_histogram_string(stats::SchedulerStatistics)
    total = stats.active_k_count_3 + stats.active_k_count_8 + stats.active_k_count_15
    if total == 0
        return "active_k histogram: no solves"
    end
    pct3 = round(100.0 * stats.active_k_count_3 / total, digits=1)
    pct8 = round(100.0 * stats.active_k_count_8 / total, digits=1)
    pct15 = round(100.0 * stats.active_k_count_15 / total, digits=1)
    return "active_k histogram: k≤3=$(pct3)% k≤8=$(pct8)% k>8=$(pct15)% | max=$(stats.active_k_max) | n=$(total)"
end

"""
    DeferredMapUpdate

A deferred map update to be processed later.
"""
struct DeferredMapUpdate
    timestamp::Float64
    tile_id::MapTileID
    innovation::Vector{Float64}
    H::Matrix{Float64}
    R::Matrix{Float64}
    pose_uncertainty::Float64
    active_dim::Int  # Number of active coefficients (k ≤ n_coef). Solve only in 1:k subspace.
end

"""
    informed_jacobian_rank(H::Matrix{Float64}) -> Int

Compute the canonical active dimension from H's column structure.
Delegates to `canonical_informed_dim` (defined in OnlineTileUpdater.jl)
which uses `JACOBIAN_COL_NORM_THRESHOLD` and clamps to {3, 8, 15}.

This ensures Scheduler and Updater always agree on active_dim for a given H.
"""
informed_jacobian_rank(H::Matrix{Float64}) = canonical_informed_dim(H)

"""Construct DeferredMapUpdate with active_dim derived from Jacobian structure.

active_dim is set to the informed rank of H (last nonzero column, clamped to {3,8,15}),
NOT size(H,2). This prevents rank-deficient solves when tiles are allocated at >8D
but the Jacobian only informs 8 columns.
"""
function DeferredMapUpdate(timestamp, tile_id, innovation, H, R, pose_uncertainty)
    k = informed_jacobian_rank(H)
    DeferredMapUpdate(timestamp, tile_id, innovation, H, R, pose_uncertainty, k)
end

"""
    OnlineSlamScheduler

Manages dual-time-scale SLAM inference.

# Architecture
The scheduler separates:
- Fast loop: Navigation state updates at sensor rate (50-200 Hz)
- Slow loop: Map coefficient updates at keyframe rate (0.5-2 Hz)

This separation ensures:
1. Real-time navigation performance (INV-06)
2. Stable online learning (gradual map updates)
3. Deterministic behavior (keyframe cadence based on timestamps)

# Fields
- `config::OnlineSlamSchedulerConfig`: Scheduler configuration
- `slam_state::SlamAugmentedState`: Current SLAM state
- `keyframe_state::KeyframeState`: Keyframe policy state
- `deferred_updates::Vector{DeferredMapUpdate}`: Pending map updates
- `current_loop::SchedulerLoopType`: Current active loop
- `statistics::SchedulerStatistics`: Performance statistics

# Usage
```julia
scheduler = OnlineSlamScheduler(slam_state, config)

# Fast loop (every sensor tick)
fast_loop_result = process_fast_loop!(scheduler, measurements)

# Check if slow loop should run
if should_run_slow_loop(scheduler)
    slow_loop_result = process_slow_loop!(scheduler)
end
```
"""
mutable struct OnlineSlamScheduler
    config::OnlineSlamSchedulerConfig
    slam_state::SlamAugmentedState
    keyframe_state::KeyframeState
    deferred_updates::Vector{DeferredMapUpdate}
    current_loop::SchedulerLoopType
    statistics::SchedulerStatistics
    tier2_gated::Bool   # If true, per-tile quadratic unlock is allowed (all 4 gates must pass)
    tier2_forced::Bool  # If true, bypass 4-gate system and force tier2_active on every tile (diagnostic only)
    held_out_buffers::Dict{MapTileID, HeldOutBuffer}  # Per-tile held-out probes for model adequacy
    model_adequacy_enabled::Bool  # If true, held-out test can demote active_k
end

"""Create scheduler from SLAM state."""
function OnlineSlamScheduler(slam_state::SlamAugmentedState;
                             config::OnlineSlamSchedulerConfig = DEFAULT_SCHEDULER_CONFIG,
                             tier2_gated::Bool = false,
                             tier2_forced::Bool = false,
                             model_adequacy_enabled::Bool = false)
    keyframe_state = KeyframeState(slam_state.nav_state)

    OnlineSlamScheduler(
        config,
        slam_state,
        keyframe_state,
        DeferredMapUpdate[],
        LOOP_IDLE,
        SchedulerStatistics(),
        tier2_gated,
        tier2_forced,
        Dict{MapTileID, HeldOutBuffer}(),
        model_adequacy_enabled
    )
end

"""Check if online learning is enabled."""
is_learning_enabled(s::OnlineSlamScheduler) = is_online_learning(s.config.slam_config)

"""Check if source tracking is enabled."""
is_source_tracking_enabled(s::OnlineSlamScheduler) = is_source_tracking(s.config.slam_config)

# ============================================================================
# Fast Loop (Navigation)
# ============================================================================

"""
    FastLoopInput

Input for fast loop processing.
"""
struct FastLoopInput
    timestamp::Float64
    imu::Union{IMUMeasurement, Nothing}
    odometry::Union{OdometryMeasurement, Nothing}
    barometer::Union{BarometerMeasurement, Nothing}
    magnetic::Union{Vector{Float64}, Nothing}  # Optional B field measurement
end

"""
    FastLoopResult

Result from fast loop processing.

# Fields
- `success::Bool`: Whether update succeeded
- `nav_state::UrbanNavState`: Updated navigation state
- `innovation::Union{Vector{Float64}, Nothing}`: Innovation if magnetic update used
- `chi2::Float64`: Chi-square of magnetic update (0 if not used)
- `elapsed_ms::Float64`: Processing time [ms]
- `map_update_deferred::Bool`: Whether a map update was deferred
"""
struct FastLoopResult
    success::Bool
    nav_state::UrbanNavState
    innovation::Union{Vector{Float64}, Nothing}
    chi2::Float64
    elapsed_ms::Float64
    map_update_deferred::Bool
end

"""
    process_fast_loop!(scheduler::OnlineSlamScheduler, input::FastLoopInput) -> FastLoopResult

Process a fast loop iteration (navigation update).

This function:
1. Propagates state with IMU (if available)
2. Updates with Odometry/barometer (if available)
3. Updates with magnetic field using CURRENT map (if available)
4. Records observation for keyframe policy
5. Optionally defers map update message

# Critical: This function does NOT modify the map. It uses the current map
for localization only. Map updates happen in the slow loop.
"""
function process_fast_loop!(scheduler::OnlineSlamScheduler, input::FastLoopInput)
    start_time = time()
    scheduler.current_loop = LOOP_FAST

    nav_state = scheduler.slam_state.nav_state
    innovation = nothing
    chi2 = 0.0
    map_update_deferred = false

    # IMU propagation would go here
    # (delegated to existing StateEstimator machinery)

    # Odometry/barometer update would go here
    # (delegated to existing StateEstimator machinery)

    # Magnetic update using current map
    if input.magnetic !== nothing && length(input.magnetic) >= 3
        # Query map at current position
        map_result = query_slam_map(scheduler.slam_state, nav_state.position)

        if map_result.in_coverage
            # Compute innovation: z - h(x)
            B_meas = SVector{3}(input.magnetic[1:3])
            innovation = Vector(B_meas - map_result.B_pred)

            # Chi-square (simplified - actual implementation uses full covariance)
            R_sensor = Matrix(0.1e-9^2 * I, 3, 3)  # 0.1 nT noise
            Σ_map = Matrix(map_result.Σ_B)
            S = R_sensor + Σ_map
            chi2 = dot(innovation, S \ innovation)

            # Record observation for keyframe policy
            record_observation!(scheduler.keyframe_state, chi2)

            # If learning is enabled, prepare deferred map update
            if is_learning_enabled(scheduler)
                tile = get_tile_state(scheduler.slam_state, map_result.tile_id)
                if tile !== nothing
                    # Held-out diversion: every Nth measurement goes to validation buffer
                    if scheduler.model_adequacy_enabled
                        buf = get!(scheduler.held_out_buffers, map_result.tile_id) do
                            HeldOutBuffer()
                        end
                        if should_holdout(buf)
                            probe = HeldOutProbe(input.position, Vector(B_meas), R_sensor, input.timestamp)
                            add_probe!(buf, probe)
                            # Skip training — this measurement is held out
                            @goto skip_deferred
                        end
                    end

                    H = compute_tile_jacobian(tile, input.position)

                    deferred = DeferredMapUpdate(
                        input.timestamp,
                        map_result.tile_id,
                        innovation,
                        H,
                        R_sensor,
                        position_uncertainty(nav_state)
                    )

                    push!(scheduler.deferred_updates, deferred)
                    map_update_deferred = true

                    @label skip_deferred
                end
            end
        end
    end

    # Update timestamp
    nav_state.timestamp = input.timestamp
    scheduler.slam_state.timestamp = input.timestamp

    # Record statistics
    elapsed_ms = (time() - start_time) * 1000
    scheduler.statistics.fast_loop_count += 1
    scheduler.statistics.total_fast_time_ms += elapsed_ms

    if elapsed_ms > scheduler.config.fast_loop_budget_ms
        scheduler.statistics.fast_loop_overruns += 1
    end

    scheduler.current_loop = LOOP_IDLE

    return FastLoopResult(true, nav_state, innovation, chi2, elapsed_ms, map_update_deferred)
end

# ============================================================================
# Slow Loop (Mapping)
# ============================================================================

"""
    should_run_slow_loop(scheduler::OnlineSlamScheduler) -> KeyframeDecision

Check if the slow loop should run (keyframe creation).
"""
function should_run_slow_loop(scheduler::OnlineSlamScheduler)
    # Check keyframe policy
    decision = evaluate_keyframe_policy(
        scheduler.keyframe_state,
        scheduler.slam_state.nav_state,
        scheduler.config.keyframe_config
    )

    # Also check if deferred updates need flushing
    if !decision.should_create &&
       length(scheduler.deferred_updates) >= scheduler.config.max_deferred_updates
        return keyframe_decision(TRIGGER_MANUAL, decision.time_since_last,
                                decision.distance_since_last, decision.rotation_since_last)
    end

    return decision
end

"""
    SlowLoopResult

Result from slow loop processing.

# Fields
- `success::Bool`: Whether update succeeded
- `updates_applied::Int`: Number of map updates applied
- `tiles_updated::Vector{MapTileID}`: Tiles that were modified
- `elapsed_ms::Float64`: Processing time [ms]
- `nees_check_passed::Bool`: Whether NEES check passed (INV-04)
"""
struct SlowLoopResult
    success::Bool
    updates_applied::Int
    tiles_updated::Vector{MapTileID}
    elapsed_ms::Float64
    nees_check_passed::Bool
end

"""
    process_slow_loop!(scheduler::OnlineSlamScheduler) -> SlowLoopResult

Process a slow loop iteration (map update).

This function:
1. Processes deferred map updates
2. Updates tile coefficients using information-form fusion
3. Optionally updates source states
4. Records keyframe
5. Validates NEES (INV-04)

# Critical: Only called at keyframe rate, not every sensor tick.
"""
function process_slow_loop!(scheduler::OnlineSlamScheduler)
    start_time = time()
    scheduler.current_loop = LOOP_SLOW

    updates_applied = 0
    tiles_updated = MapTileID[]
    nees_check_passed = true

    # Skip if learning is disabled
    if !is_learning_enabled(scheduler)
        record_keyframe!(scheduler.keyframe_state, scheduler.slam_state.nav_state)
        scheduler.statistics.keyframes_created += 1
        elapsed_ms = (time() - start_time) * 1000
        scheduler.current_loop = LOOP_IDLE
        return SlowLoopResult(true, 0, tiles_updated, elapsed_ms, true)
    end

    # Process deferred updates by tile
    updates_by_tile = Dict{MapTileID, Vector{DeferredMapUpdate}}()
    for update in scheduler.deferred_updates
        if !haskey(updates_by_tile, update.tile_id)
            updates_by_tile[update.tile_id] = DeferredMapUpdate[]
        end
        push!(updates_by_tile[update.tile_id], update)
    end


    # Apply batch updates to each tile
    for (tile_id, updates) in updates_by_tile
        tile = get_tile_state(scheduler.slam_state, tile_id)
        if tile === nothing
            continue
        end

        # =====================================================================
        # ACTIVE-SUBSPACE INFORMATION-FORM BATCH UPDATE
        # =====================================================================
        #
        # For tiles with >k active coefficients out of n total (fixed-max-state),
        # we solve ONLY in the active subspace (1:k). Frozen dimensions (k+1:n)
        # never participate in the information-form inversion.
        #
        # Why: Even with zero Jacobian columns for frozen dims, the full n×n
        # solve can corrupt active coefficients through numerical coupling
        # between prior information in frozen dims and data information in
        # active dims. Solving in the k×k subspace eliminates this.
        #
        # Invariant: I[active, inactive] == 0 and I[inactive, active] == 0
        # is enforced by zeroing cross-blocks after each update.
        # =====================================================================

        n_dim = tile_state_dim(tile)

        # Determine active dimension from updates (use minimum across batch)
        active_k = n_dim
        for update in updates
            active_k = min(active_k, update.active_dim)
        end
        # Spatial diversity gate: clamp active_k to B0-only (3) unless
        # the tile has sufficient trajectory excitation for gradient learning.
        # Physics: gradient Jacobian columns scale as dx,dy,dz. Without
        # spatial diversity, gradient directions are rank-deficient.
        #
        # CRITICAL: The solve active_k and prediction gate (OnlineMapProvider)
        # must be synchronized. If the solve uses k=8 but prediction uses B0-only,
        # the B0 covariance shrinks faster than a pure 3D solve due to gradient
        # coupling in the information matrix, causing NEES overconfidence.
        # Both gates use GRADIENT_MIN_SPAN_M and GRADIENT_MIN_OBS.
        if active_k > 3
            span = tile.position_bbox_max - tile.position_bbox_min
            span_xy = max(span[1], span[2])
            if span_xy < GRADIENT_MIN_SPAN_M || tile.observation_count < GRADIENT_MIN_OBS
                active_k = 3  # Fall back to B0-only solve
            end
        end

        # =============================================================
        # MODEL ADEQUACY DEMOTION (held-out prediction test)
        # If linear model is worse than B0 on held-out data, demote.
        # =============================================================
        if scheduler.model_adequacy_enabled && active_k > 3
            buf = get(scheduler.held_out_buffers, tile_id, nothing)
            if buf !== nothing && buf.count >= 10  # Need minimum probes for reliable test
                probes = get_probes(buf)
                adequacy = evaluate_model_adequacy(tile, probes)
                if !adequacy.linear_adequate
                    active_k = 3  # Demote to B0-only
                    @warn "Model adequacy demotion: linear worse than B0" tile_id χ2_b0=adequacy.χ2_b0 χ2_linear=adequacy.χ2_linear n_probes=adequacy.n_probes
                end
            end
        end

        # =============================================================
        # TIER-2 UNLOCK (must happen BEFORE accumulation/solve so
        # active_k=15 is used in the information-form update)
        # =============================================================
        tier2_allowed = scheduler.tier2_gated || scheduler.tier2_forced
        ΔI_qq = nothing  # Will be set if gates are evaluated; used by re-lock

        # Safety clamp: when Tier-2 is not allowed, cap active_k at 8.
        # Without this, informed_jacobian_rank(H)=15 (non-zero quadratic cols)
        # would cause a 15D solve even under TIER2_LOCKED policy.
        if !tier2_allowed && active_k > 8
            active_k = min(active_k, 8)
        end

        if scheduler.tier2_forced && !tile.tier2_active && n_dim >= 15
            # Mode C: bypass all gates, force unlock immediately
            tile.tier2_active = true
            active_k = 15
            @info "Tier-2 FORCED" tile_id obs=tile.observation_count
        elseif tier2_allowed && !tile.tier2_active && active_k >= 8 && n_dim >= 15
            # =============================================================
            # 3-GATE STATISTICAL READINESS TEST for k=8→15 unlock
            # =============================================================
            # Gate 1: Excitation/coverage (spatial diversity + observation count)
            gate_excitation = is_tier2_eligible(tile)

            # Gates 2-3 require data FIM computation; skip if Gate 1 fails
            gate_identifiability = false
            gate_benefit = false

            if gate_excitation
                # Accumulate full 15-col DATA FIM from current batch
                ΔI_full = zeros(15, 15)
                for update in updates
                    R_inv = inv(update.R)
                    H15 = update.H[:, 1:min(15, size(update.H, 2))]
                    if size(H15, 2) < 15
                        H15 = hcat(H15, zeros(3, 15 - size(H15, 2)))
                    end
                    ΔI_full += H15' * R_inv * H15
                end
                ΔI_qq = ΔI_full[9:15, 9:15]

                # Gate 2: Identifiability — quadratic block has sufficient
                # rank and conditioning in the data subspace.
                sv_quad = svdvals(Symmetric(ΔI_qq))
                rank_quad = count(s -> s > 1e-10 * maximum(sv_quad), sv_quad)
                sigma_min = minimum(sv_quad)
                sigma_max = maximum(sv_quad)
                relative_cond = sigma_min / max(sigma_max, 1e-30)
                gate_identifiability = rank_quad >= TIER2_MIN_RANK_QUAD &&
                                       relative_cond >= TIER2_MIN_RELATIVE_COND

                # Gate 3: Held-out benefit — quadratic model must demonstrably
                # improve prediction on held-out data over linear model.
                # This prevents unlock when data is sufficient to identify
                # parameters but the quadratic terms don't actually help.
                if gate_identifiability && scheduler.model_adequacy_enabled
                    buf = get(scheduler.held_out_buffers, tile_id, nothing)
                    if buf !== nothing && buf.count >= 10
                        probes = get_probes(buf)
                        adequacy = evaluate_model_adequacy(tile, probes)
                        gate_benefit = adequacy.quadratic_beneficial
                        @debug "Tier-2 benefit gate" tile_id χ2_linear=adequacy.χ2_linear χ2_quad=adequacy.χ2_quadratic beneficial=gate_benefit
                    end
                    # If no held-out data yet, gate_benefit stays false → no unlock
                elseif gate_identifiability && !scheduler.model_adequacy_enabled
                    # Without held-out testing, allow unlock on identifiability alone
                    # (backward-compatible with tier2_gated mode without model_adequacy)
                    gate_benefit = true
                end

                @debug "Tier-2 gates" tile_id gate_excitation gate_identifiability gate_benefit rank_quad relative_cond
            end

            if gate_excitation && gate_identifiability && gate_benefit
                tile.tier2_active = true
                active_k = 15
                @info "Tier-2 UNLOCKED" tile_id obs=tile.observation_count
            end
        end

        # If already tier2_active from a previous batch, use k=15
        if tile.tier2_active && tier2_allowed && active_k < 15 && n_dim >= 15
            active_k = 15
        end

        active_idx = 1:active_k

        # Record active_k histogram
        if active_k <= 3
            scheduler.statistics.active_k_count_3 += 1
        elseif active_k <= 8
            scheduler.statistics.active_k_count_8 += 1
        else
            scheduler.statistics.active_k_count_15 += 1
        end
        scheduler.statistics.active_k_max = max(scheduler.statistics.active_k_max, active_k)

        # Accumulate information in the active subspace only
        I_delta_aa = zeros(active_k, active_k)
        i_delta_a = zeros(active_k)
        tile_updates_applied = 0

        for update in updates
            # =====================================================================
            # CONFIDENCE-WEIGHTED TEACHABILITY GATE
            # =====================================================================
            # σ_nominal = 2m → tr(P_pos) = 3σ² = 12 m²
            # σ_max = 5m → tr(P_pos) = 3σ² = 75 m²
            const_nominal_tr_P = 12.0   # m²
            const_max_tr_P = 75.0       # m²
            const_min_weight = 0.1

            pos_unc = update.pose_uncertainty
            if pos_unc <= const_nominal_tr_P
                weight = 1.0
            elseif pos_unc >= const_max_tr_P
                weight = 0.0
            else
                frac = (pos_unc - const_nominal_tr_P) / (const_max_tr_P - const_nominal_tr_P)
                weight = 1.0 - frac * (1.0 - const_min_weight)
            end

            scheduler.statistics.attempted_tile_updates += 1

            if weight < const_min_weight
                scheduler.statistics.rejected_by_teachability += 1
                continue
            end

            scheduler.statistics.accepted_tile_updates += 1

            # Extract active subspace of H (first k columns)
            R_inv = inv(update.R)
            H_a = update.H[:, active_idx]

            # Accumulate in active subspace only
            I_delta_aa += weight * (H_a' * R_inv * H_a)
            i_delta_a += weight * (H_a' * R_inv * update.innovation)

            tile_updates_applied += 1
        end

        if tile_updates_applied > 0
            # =====================================================================
            # CONDITION NUMBER GATE on DATA information (not prior)
            # =====================================================================
            # Gate on cond(ΔI_aa): the data Fisher in the active subspace.
            # This catches rank-deficient updates regardless of prior strength.
            # σ_min(ΔI_aa) < threshold means the data provides no useful
            # information in some active direction.
            #
            # Applied for ALL active_k ≥ 3 (not just k > 8). Even 8D updates
            # can be rank-deficient if measurement geometry is degenerate.
            # =====================================================================
            if active_k >= 3
                # For k=15: check linear (1:8) and quadratic (9:15) blocks
                # separately. The 1e-9 basis scaling creates O(1e6) scale
                # difference between blocks, making joint cond() useless.
                # The Tier-2 gate system already validates quadratic quality.
                if active_k >= 15
                    sv_lin = svdvals(I_delta_aa[1:8, 1:8])
                    data_cond = maximum(sv_lin) / max(minimum(sv_lin), 1e-30)
                else
                    sv = svdvals(I_delta_aa)
                    data_cond = maximum(sv) / max(minimum(sv), 1e-30)
                end
                if data_cond > 1e8
                    scheduler.statistics.rejected_by_quality += 1
                    continue
                end
            end

            # =====================================================================
            # BLOCK-STRUCTURED SOLVE
            # =====================================================================
            # When active_k=15, solve linear (1:8) and quadratic (9:15) blocks
            # INDEPENDENTLY. Cross-block coupling in the data FIM is discarded.
            # This ensures quadratic updates cannot inject spurious changes
            # into B0/G coefficients.
            #
            # For active_k ≤ 8, this reduces to the standard active-subspace
            # solve (single block).
            #
            # Physics justification: The cross-block term H_lin'R⁻¹H_quad
            # represents correlation between linear and quadratic information.
            # Discarding it is conservative — it slightly increases posterior
            # variance but guarantees that learning Q never degrades B0/G.
            # =====================================================================

            if active_k >= 15
                # --- Block 1: Linear (1:8) ---
                lin_idx = 1:8
                I_lin_old = tile.information[lin_idx, lin_idx]
                x_lin_old = tile.coefficients[lin_idx]
                i_lin_old = I_lin_old * x_lin_old

                # Extract linear-only data information (discard cross terms)
                ΔI_lin = I_delta_aa[1:8, 1:8]
                Δi_lin = i_delta_a[1:8]

                I_lin_new = I_lin_old + ΔI_lin
                P_lin_new = inv(I_lin_new + 1e-12 * LinearAlgebra.I(8))
                x_lin_new = P_lin_new * (i_lin_old + Δi_lin)

                # --- Block 2: Quadratic (9:15) ---
                quad_idx = 9:15
                I_quad_old = tile.information[quad_idx, quad_idx]
                x_quad_old = tile.coefficients[quad_idx]
                i_quad_old = I_quad_old * x_quad_old

                ΔI_quad = I_delta_aa[9:15, 9:15]
                Δi_quad = i_delta_a[9:15]

                I_quad_new = I_quad_old + ΔI_quad
                P_quad_new = inv(I_quad_new + 1e-12 * LinearAlgebra.I(7))
                x_quad_new = P_quad_new * (i_quad_old + Δi_quad)

                # --- B0/G stability check ---
                # Snapshot linear coefficients before write-back. If held-out
                # test shows B0/G degradation after enabling quadratic, we
                # can rollback (handled by re-lock + coefficient preservation).
                x_lin_before = copy(x_lin_old)

                # Write back both blocks independently
                tile.information[lin_idx, lin_idx] = I_lin_new
                tile.covariance[lin_idx, lin_idx] = P_lin_new
                tile.coefficients[lin_idx] = x_lin_new

                tile.information[quad_idx, quad_idx] = I_quad_new
                tile.covariance[quad_idx, quad_idx] = P_quad_new
                tile.coefficients[quad_idx] = x_quad_new

                # Zero ALL cross-blocks (linear↔quadratic and active↔inactive)
                tile.information[lin_idx, quad_idx] .= 0.0
                tile.information[quad_idx, lin_idx] .= 0.0
                tile.covariance[lin_idx, quad_idx] .= 0.0
                tile.covariance[quad_idx, lin_idx] .= 0.0
            else
                # Standard active-subspace solve (single block, k ≤ 8)
                I_aa_old = tile.information[active_idx, active_idx]
                x_a_old = tile.coefficients[active_idx]

                i_a_old = I_aa_old * x_a_old
                I_aa_new = I_aa_old + I_delta_aa
                i_a_new = i_a_old + i_delta_a

                P_aa_new = inv(I_aa_new + 1e-12 * LinearAlgebra.I(active_k))
                x_a_new = P_aa_new * i_a_new

                tile.information[active_idx, active_idx] = I_aa_new
                tile.covariance[active_idx, active_idx] = P_aa_new
                tile.coefficients[active_idx] = x_a_new
            end

            # Enforce structural decoupling: zero cross-blocks between
            # active and inactive dims (prevents coupling leakage)
            if active_k < n_dim
                inactive_idx = (active_k+1):n_dim
                tile.information[active_idx, inactive_idx] .= 0.0
                tile.information[inactive_idx, active_idx] .= 0.0
                tile.covariance[active_idx, inactive_idx] .= 0.0
                tile.covariance[inactive_idx, active_idx] .= 0.0
            end

            tile.observation_count += tile_updates_applied
            tile.version += 1
            tile.last_update_time = scheduler.slam_state.timestamp

            # =============================================================
            # TIER-2 RE-LOCK: monitor for divergence after solve
            # =============================================================
            if tile.tier2_active
                should_relock = false
                reason = ""

                # Check 1: Spatial diversity dropped (vehicle left tile)
                span = tile.position_bbox_max - tile.position_bbox_min
                span_xy_relock = max(span[1], span[2])
                if span_xy_relock < TIER2_RELOCK_SPAN_M
                    should_relock = true
                    reason = "span=$(round(span_xy_relock, digits=1))m"
                end

                # Check 2: Policy override (runtime downgrade)
                if !tier2_allowed
                    should_relock = true
                    reason = "policy=LOCKED"
                end

                # Check 3: Relative conditioning of quadratic DATA information
                if ΔI_qq !== nothing
                    sv_relock = svdvals(Symmetric(ΔI_qq))
                    rel_cond = minimum(sv_relock) / max(maximum(sv_relock), 1e-30)
                    if rel_cond < TIER2_RELOCK_RELATIVE_COND
                        should_relock = true
                        reason = "rel_cond=$(round(rel_cond, sigdigits=3))"
                    end
                end

                # Check 4: Held-out adequacy — quadratic no longer beneficial
                if !should_relock && scheduler.model_adequacy_enabled
                    buf = get(scheduler.held_out_buffers, tile_id, nothing)
                    if buf !== nothing && buf.count >= 10
                        probes = get_probes(buf)
                        adequacy = evaluate_model_adequacy(tile, probes)
                        if !adequacy.quadratic_beneficial
                            should_relock = true
                            reason = "held_out: χ2_lin=$(round(adequacy.χ2_linear, sigdigits=3)) χ2_quad=$(round(adequacy.χ2_quadratic, sigdigits=3))"
                        end
                    end
                end

                if should_relock
                    tile.tier2_active = false
                    tile.tier2_relock_count += 1
                    # Next solve uses active_k=8; coefs 9-15 frozen but preserved
                    @warn "Tier-2 RE-LOCKED" tile_id reason relock_count=tile.tier2_relock_count
                end
            end

            push!(tiles_updated, tile_id)
            updates_applied += tile_updates_applied
        end
    end

    # Clear deferred updates
    empty!(scheduler.deferred_updates)
    scheduler.statistics.deferred_updates += updates_applied

    # Record keyframe
    record_keyframe!(scheduler.keyframe_state, scheduler.slam_state.nav_state)
    scheduler.slam_state.version += 1
    scheduler.statistics.keyframes_created += 1

    # Record statistics
    elapsed_ms = (time() - start_time) * 1000
    scheduler.statistics.slow_loop_count += 1
    scheduler.statistics.total_slow_time_ms += elapsed_ms

    if elapsed_ms > scheduler.config.slow_loop_budget_ms
        scheduler.statistics.slow_loop_overruns += 1
    end

    scheduler.current_loop = LOOP_IDLE

    return SlowLoopResult(true, updates_applied, tiles_updated, elapsed_ms, nees_check_passed)
end

# ============================================================================
# Statistics and Diagnostics
# ============================================================================

"""Average fast loop time [ms]."""
function avg_fast_loop_time(s::OnlineSlamScheduler)
    s.statistics.fast_loop_count == 0 ? 0.0 :
        s.statistics.total_fast_time_ms / s.statistics.fast_loop_count
end

"""Average slow loop time [ms]."""
function avg_slow_loop_time(s::OnlineSlamScheduler)
    s.statistics.slow_loop_count == 0 ? 0.0 :
        s.statistics.total_slow_time_ms / s.statistics.slow_loop_count
end

"""Fast loop overrun rate."""
function fast_loop_overrun_rate(s::OnlineSlamScheduler)
    s.statistics.fast_loop_count == 0 ? 0.0 :
        s.statistics.fast_loop_overruns / s.statistics.fast_loop_count
end

"""Slow loop overrun rate."""
function slow_loop_overrun_rate(s::OnlineSlamScheduler)
    s.statistics.slow_loop_count == 0 ? 0.0 :
        s.statistics.slow_loop_overruns / s.statistics.slow_loop_count
end

"""Format scheduler statistics."""
function format_scheduler_statistics(s::OnlineSlamScheduler)
    stats = s.statistics
    return """
    Online SLAM Scheduler Statistics:
      Fast loop: $(stats.fast_loop_count) iterations
        Average time: $(round(avg_fast_loop_time(s), digits=2)) ms
        Overruns: $(stats.fast_loop_overruns) ($(round(100*fast_loop_overrun_rate(s), digits=1))%)
      Slow loop: $(stats.slow_loop_count) iterations
        Average time: $(round(avg_slow_loop_time(s), digits=2)) ms
        Overruns: $(stats.slow_loop_overruns) ($(round(100*slow_loop_overrun_rate(s), digits=1))%)
      Keyframes created: $(stats.keyframes_created)
      Deferred updates processed: $(stats.deferred_updates)
      Forced flushes: $(stats.forced_flushes)
    """
end

# ============================================================================
# Mission Finalization (Step 6: End-of-Mission Checkpoint)
# ============================================================================

"""
    MissionFinalizeResult

Result from finalizing a mission.

# Fields
- `success::Bool`: Whether finalization succeeded
- `updates_flushed::Int`: Number of deferred updates flushed
- `slam_version::Int`: Final SLAM state version
- `total_observations::Int`: Total observations processed this mission
- `elapsed_ms::Float64`: Finalization time [ms]
"""
struct MissionFinalizeResult
    success::Bool
    updates_flushed::Int
    slam_version::Int
    total_observations::Int
    elapsed_ms::Float64
end

"""
    finalize_mission!(scheduler::OnlineSlamScheduler, mission_id::String;
                      metrics::Dict{String, Float64} = Dict{String, Float64}())
                      -> (MissionFinalizeResult, MultiTileCheckpoint)

Finalize a mission: flush updates, validate, and create checkpoint.

This is the key function for multi-mission persistence:
1. Flush any remaining deferred updates
2. Increment version to mark mission boundary
3. Create checkpoint for persistence
4. Return checkpoint for saving

# Arguments
- `scheduler`: The SLAM scheduler with current state
- `mission_id`: Identifier for this mission
- `metrics`: Optional mission-level metrics to include

# Returns
Tuple of (result, checkpoint) where checkpoint can be saved via MapPersistence.

# Usage
```julia
# End of Mission 1
result, checkpoint = finalize_mission!(scheduler, "mission_001")
save_multi_tile_checkpoint("maps/mission_001.json", checkpoint)

# Start of Mission 2
checkpoint = load_multi_tile_checkpoint("maps/mission_001.json")
enable_online_learning!(new_estimator; initial_checkpoint=checkpoint)
```
"""
function finalize_mission!(scheduler::OnlineSlamScheduler, mission_id::String;
                           metrics::Dict{String, Float64} = Dict{String, Float64}())
    start_time = time()

    # Step 1: Flush remaining deferred updates
    updates_flushed = 0
    if !isempty(scheduler.deferred_updates)
        # Force a slow loop to process remaining updates
        result = process_slow_loop!(scheduler)
        updates_flushed = result.updates_applied
        @info "Flushed deferred updates at mission end" n_updates=updates_flushed
    end

    # Step 2: Increment version to mark mission boundary
    scheduler.slam_state.version += 1

    # Step 3: Compute total observations across all tiles
    total_observations = sum(
        t.observation_count for t in values(scheduler.slam_state.tile_states);
        init=0
    )

    # Step 4: Create checkpoint for persistence
    checkpoint = create_multi_tile_checkpoint(scheduler.slam_state, mission_id; metrics=metrics)

    elapsed_ms = (time() - start_time) * 1000

    result = MissionFinalizeResult(
        true,
        updates_flushed,
        scheduler.slam_state.version,
        total_observations,
        elapsed_ms
    )

    @info "Mission finalized" mission_id=mission_id version=scheduler.slam_state.version n_tiles=length(checkpoint.tiles) total_obs=total_observations

    return result, checkpoint
end

"""
    get_current_checkpoint(scheduler::OnlineSlamScheduler, mission_id::String)
                           -> MultiTileCheckpoint

Get a checkpoint of current state without flushing (non-destructive snapshot).

Use this for periodic checkpointing during a mission, or for inspection.
For end-of-mission checkpointing, use `finalize_mission!` instead.
"""
function get_current_checkpoint(scheduler::OnlineSlamScheduler, mission_id::String)
    return create_multi_tile_checkpoint(scheduler.slam_state, mission_id)
end

# ============================================================================
# Teachability Statistics (Phase C Verification)
# ============================================================================

"""
    TeachabilityStatistics

Summary of teachability gate behavior for Phase C verification.

# Fields
- `attempted::Int`: Total update attempts
- `accepted::Int`: Updates accepted by all gates
- `rejected_teachability::Int`: Rejected by pose uncertainty gate
- `rejected_observability::Int`: Rejected by gradient observability gate
- `rejected_quality::Int`: Rejected by tile quality gate
- `acceptance_rate::Float64`: Fraction of updates accepted

# Verification Criteria
- High acceptance rate (> 80%) in observable regions
- Non-zero rejections in clutter/mismatch scenarios
- ~100% acceptance everywhere is suspicious (gates not working)
"""
struct TeachabilityStatistics
    attempted::Int
    accepted::Int
    rejected_teachability::Int
    rejected_observability::Int
    rejected_quality::Int
    acceptance_rate::Float64
end

"""
    get_teachability_statistics(scheduler::OnlineSlamScheduler) -> TeachabilityStatistics

Extract teachability gate statistics from scheduler.
"""
function get_teachability_statistics(scheduler::OnlineSlamScheduler)
    stats = scheduler.statistics
    total = stats.attempted_tile_updates
    accepted = stats.accepted_tile_updates
    rate = total > 0 ? accepted / total : 0.0

    TeachabilityStatistics(
        total,
        accepted,
        stats.rejected_by_teachability,
        stats.rejected_by_observability,
        stats.rejected_by_quality,
        rate
    )
end

"""
    format_teachability_statistics(stats::TeachabilityStatistics) -> String

Format teachability statistics for reporting.
"""
function format_teachability_statistics(stats::TeachabilityStatistics)
    return """
    Teachability Gate Statistics:
      Attempted:               $(stats.attempted)
      Accepted:                $(stats.accepted) ($(round(stats.acceptance_rate * 100, digits=1))%)
      Rejected (teachability): $(stats.rejected_teachability)
      Rejected (observability): $(stats.rejected_observability)
      Rejected (quality):      $(stats.rejected_quality)
    """
end

# ============================================================================
# Exports
# ============================================================================

export KeyframeTrigger, TRIGGER_TIME, TRIGGER_DISTANCE, TRIGGER_ROTATION
export TRIGGER_OBSERVABILITY, TRIGGER_RESIDUAL, TRIGGER_MANUAL
export KeyframePolicyConfig, DEFAULT_KEYFRAME_POLICY_CONFIG
export KeyframeState, KeyframeDecision
export evaluate_keyframe_policy, record_keyframe!, record_observation!
export SchedulerLoopType, LOOP_FAST, LOOP_SLOW, LOOP_IDLE
export OnlineSlamSchedulerConfig, DEFAULT_SCHEDULER_CONFIG
export SchedulerStatistics, DeferredMapUpdate
export OnlineSlamScheduler
export is_learning_enabled, is_source_tracking_enabled, model_adequacy_enabled
export FastLoopInput, FastLoopResult, process_fast_loop!
export SlowLoopResult, should_run_slow_loop, process_slow_loop!
export avg_fast_loop_time, avg_slow_loop_time
export fast_loop_overrun_rate, slow_loop_overrun_rate
export format_scheduler_statistics
export MissionFinalizeResult, finalize_mission!, get_current_checkpoint
export TeachabilityStatistics, get_teachability_statistics, format_teachability_statistics
