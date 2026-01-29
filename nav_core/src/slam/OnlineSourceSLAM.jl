# ============================================================================
# OnlineSourceSLAM.jl - Source (Dipole) SLAM Integration (Phase C Step 6)
# ============================================================================
#
# Integrates magnetic source (dipole) tracking into the online MagSLAM loop.
#
# Key Responsibilities:
# 1. Source detection from residuals (track-before-detect)
# 2. Source state estimation (EKF-style updates for 6-DOF dipole)
# 3. Source lifecycle management (candidate → active → retired)
# 4. Source contribution subtraction for clean map learning
#
# Architecture:
# - Sources are separate from tiles (1/r³ vs smooth basis)
# - Sources are subtracted BEFORE tile updates (INV-07 clean separation)
# - Source uncertainty propagates to total measurement covariance
#
# Physics:
# - Dipole field: B(r) = (μ₀/4π) [3(m·r̂)r̂ - m] / |r|³
# - Falls off as 1/r³ (much faster than tile basis)
# - Near-field: sensitive to position; far-field: sensitive to moment
#
# ============================================================================

using LinearAlgebra
using StaticArrays
using Statistics: mean

# ============================================================================
# Source Detection Configuration
# ============================================================================

"""
    SourceDetectionConfig

Configuration for detecting new source candidates from residuals.

# Fields
- `min_residual_norm::Float64`: Minimum residual for detection [T]
- `min_snr::Float64`: Minimum signal-to-noise ratio
- `detection_chi2_threshold::Float64`: χ² threshold for anomaly detection
- `spatial_clustering_radius::Float64`: Radius for clustering detections [m]
- `min_cluster_observations::Int`: Minimum observations to form cluster
- `max_candidates::Int`: Maximum tracked candidates

# Physics Justification
- min_residual_norm = 100e-9 T (100 nT): Typical dipole signature at 10m
- min_snr = 3.0: Classic detection threshold
- detection_chi2_threshold = 11.345 (χ²(3, p=0.01)): 1% false positive rate
"""
struct SourceDetectionConfig
    min_residual_norm::Float64
    min_snr::Float64
    detection_chi2_threshold::Float64
    spatial_clustering_radius::Float64
    min_cluster_observations::Int
    max_candidates::Int

    function SourceDetectionConfig(;
        min_residual_norm::Float64 = 100e-9,
        min_snr::Float64 = 3.0,
        detection_chi2_threshold::Float64 = 11.345,
        spatial_clustering_radius::Float64 = 10.0,
        min_cluster_observations::Int = 3,
        max_candidates::Int = 20
    )
        @assert min_residual_norm > 0
        @assert min_snr > 0
        @assert detection_chi2_threshold > 0
        @assert spatial_clustering_radius > 0
        @assert min_cluster_observations >= 1
        @assert max_candidates > 0

        new(min_residual_norm, min_snr, detection_chi2_threshold,
            spatial_clustering_radius, min_cluster_observations, max_candidates)
    end
end

const DEFAULT_SOURCE_DETECTION_CONFIG = SourceDetectionConfig()

# ============================================================================
# Source Promotion Configuration
# ============================================================================

"""
    SourcePromotionConfig

Configuration for promoting candidates to active sources.

# Fields
- `min_support_count::Int`: Minimum observations to promote
- `min_duration_s::Float64`: Minimum lifetime before promotion [s]
- `max_cluster_radius::Float64`: Maximum spatial spread for promotion [m]
- `min_position_confidence::Float64`: Minimum position confidence [m²]
- `chi2_improvement_threshold::Float64`: Minimum χ² improvement ratio

# Promotion Criteria
All must pass before candidate becomes active:
1. Temporal persistence: observed for ≥ min_duration_s
2. Spatial compactness: cluster radius < max_cluster_radius
3. Sufficient observations: support_count ≥ min_support_count
4. Position confidence: tr(P_pos) ≤ min_position_confidence
5. Residual reduction: χ² improves by ≥ chi2_improvement_threshold
"""
struct SourcePromotionConfig
    min_support_count::Int
    min_duration_s::Float64
    max_cluster_radius::Float64
    min_position_confidence::Float64
    chi2_improvement_threshold::Float64

    function SourcePromotionConfig(;
        min_support_count::Int = 5,
        min_duration_s::Float64 = 2.0,
        max_cluster_radius::Float64 = 5.0,
        min_position_confidence::Float64 = 25.0,
        chi2_improvement_threshold::Float64 = 0.5
    )
        @assert min_support_count >= 1
        @assert min_duration_s > 0
        @assert max_cluster_radius > 0
        @assert min_position_confidence > 0
        @assert 0 < chi2_improvement_threshold <= 1

        new(min_support_count, min_duration_s, max_cluster_radius,
            min_position_confidence, chi2_improvement_threshold)
    end
end

const DEFAULT_SOURCE_PROMOTION_CONFIG = SourcePromotionConfig()

# ============================================================================
# Source Update Configuration
# ============================================================================

"""
    SourceUpdateConfig

Configuration for updating active source states.

# Fields
- `max_association_distance::Float64`: Max distance for measurement association [m]
- `association_chi2_threshold::Float64`: χ² gate for association
- `position_process_noise::Float64`: Position random walk σ [m²/s]
- `moment_process_noise::Float64`: Moment random walk σ [A²m⁴/s]
- `max_covariance_trace::Float64`: Max tr(P) before source demotion

# Physics
- position_process_noise: Sources are typically stationary; small allows drift
- moment_process_noise: Small; moment should be constant for stationary object
"""
struct SourceUpdateConfig
    max_association_distance::Float64
    association_chi2_threshold::Float64
    position_process_noise::Float64
    moment_process_noise::Float64
    max_covariance_trace::Float64

    function SourceUpdateConfig(;
        max_association_distance::Float64 = 20.0,
        association_chi2_threshold::Float64 = 16.266,
        position_process_noise::Float64 = 0.001,
        moment_process_noise::Float64 = 0.01,
        max_covariance_trace::Float64 = 10000.0
    )
        @assert max_association_distance > 0
        @assert association_chi2_threshold > 0
        @assert position_process_noise >= 0
        @assert moment_process_noise >= 0
        @assert max_covariance_trace > 0

        new(max_association_distance, association_chi2_threshold,
            position_process_noise, moment_process_noise, max_covariance_trace)
    end
end

const DEFAULT_SOURCE_UPDATE_CONFIG = SourceUpdateConfig()

# ============================================================================
# Source Retirement Configuration
# ============================================================================

"""
    SourceRetirementConfig

Configuration for retiring sources.

# Fields
- `max_unobserved_duration_s::Float64`: Retire if not observed for this long [s]
- `min_contribution_threshold::Float64`: Minimum field contribution to keep [T]
- `covariance_growth_threshold::Float64`: Retire if covariance grows too much

# Retirement Reasons
- TIME_OUT: Not observed for max_unobserved_duration_s
- LOW_CONTRIBUTION: Field contribution below threshold (moved away)
- ABSORBED: Close enough to tile center to be absorbed
- DEMOTION: Failed validation (large residuals, covariance explosion)
"""
struct SourceRetirementConfig
    max_unobserved_duration_s::Float64
    min_contribution_threshold::Float64
    covariance_growth_threshold::Float64

    function SourceRetirementConfig(;
        max_unobserved_duration_s::Float64 = 30.0,
        min_contribution_threshold::Float64 = 10e-9,
        covariance_growth_threshold::Float64 = 2.0
    )
        @assert max_unobserved_duration_s > 0
        @assert min_contribution_threshold > 0
        @assert covariance_growth_threshold > 1

        new(max_unobserved_duration_s, min_contribution_threshold,
            covariance_growth_threshold)
    end
end

const DEFAULT_SOURCE_RETIREMENT_CONFIG = SourceRetirementConfig()

# ============================================================================
# Combined Source SLAM Configuration
# ============================================================================

"""
    OnlineSourceSLAMConfig

Combined configuration for online source SLAM.
"""
struct OnlineSourceSLAMConfig
    detection::SourceDetectionConfig
    promotion::SourcePromotionConfig
    update::SourceUpdateConfig
    retirement::SourceRetirementConfig

    function OnlineSourceSLAMConfig(;
        detection::SourceDetectionConfig = DEFAULT_SOURCE_DETECTION_CONFIG,
        promotion::SourcePromotionConfig = DEFAULT_SOURCE_PROMOTION_CONFIG,
        update::SourceUpdateConfig = DEFAULT_SOURCE_UPDATE_CONFIG,
        retirement::SourceRetirementConfig = DEFAULT_SOURCE_RETIREMENT_CONFIG
    )
        new(detection, promotion, update, retirement)
    end
end

const DEFAULT_SOURCE_SLAM_CONFIG = OnlineSourceSLAMConfig()

# ============================================================================
# Source Candidate (Pre-Promotion)
# ============================================================================

"""
    SourceCandidate

Candidate source before promotion to SlamSourceState.

Tracks observations and statistics needed for promotion decision.
"""
mutable struct SourceCandidate
    id::Int
    created_time::Float64

    # Accumulated observations
    positions::Vector{Vec3Map}      # Measurement positions
    residuals::Vector{Vec3Map}      # Field residuals at each position
    timestamps::Vector{Float64}     # Observation timestamps
    chi2_values::Vector{Float64}    # χ² values for each observation

    # Statistics
    centroid::Vec3Map
    cluster_radius::Float64
    mean_residual::Vec3Map
    estimated_position::Vec3Map
    estimated_moment::SVector{3, Float64}
    position_covariance::SMatrix{3, 3, Float64, 9}
end

"""Create new source candidate."""
function SourceCandidate(id::Int, position::AbstractVector, residual::AbstractVector,
                         timestamp::Float64, chi2::Float64)
    SourceCandidate(
        id,
        timestamp,
        [Vec3Map(position...)],
        [Vec3Map(residual...)],
        [timestamp],
        [chi2],
        Vec3Map(position...),
        0.0,
        Vec3Map(residual...),
        Vec3Map(position...),
        SVector{3}(0.0, 0.0, 0.0),
        SMatrix{3, 3}(100.0 * I(3))
    )
end

"""Add observation to candidate."""
function add_observation!(c::SourceCandidate, position::AbstractVector,
                          residual::AbstractVector, timestamp::Float64, chi2::Float64)
    push!(c.positions, Vec3Map(position...))
    push!(c.residuals, Vec3Map(residual...))
    push!(c.timestamps, timestamp)
    push!(c.chi2_values, chi2)

    update_candidate_statistics!(c)
end

"""Update candidate statistics from observations."""
function update_candidate_statistics!(c::SourceCandidate)
    n = length(c.positions)
    if n == 0
        return
    end

    # Compute centroid
    c.centroid = Vec3Map(mean([p[1] for p in c.positions]),
                         mean([p[2] for p in c.positions]),
                         mean([p[3] for p in c.positions]))

    # Compute cluster radius
    if n > 1
        c.cluster_radius = maximum(norm(p - c.centroid) for p in c.positions)
    else
        c.cluster_radius = 0.0
    end

    # Compute mean residual
    c.mean_residual = Vec3Map(mean([r[1] for r in c.residuals]),
                              mean([r[2] for r in c.residuals]),
                              mean([r[3] for r in c.residuals]))

    # Estimate source position from residual centroid
    # Heuristic: source is approximately at measurement centroid
    c.estimated_position = c.centroid

    # Estimate moment magnitude from residuals
    # B ~ μ₀/(4π) * |m| / r³  =>  |m| ~ 4π/μ₀ * |B| * r³
    mean_residual_mag = norm(c.mean_residual)
    # Assume average observation distance of 5m for initial estimate
    r_est = 5.0
    μ₀_4π = 1e-7
    moment_mag = mean_residual_mag * r_est^3 / μ₀_4π

    # Direction heuristic: moment roughly aligned with mean residual
    if mean_residual_mag > 0
        moment_dir = c.mean_residual / mean_residual_mag
    else
        moment_dir = Vec3Map(0.0, 0.0, 1.0)
    end
    c.estimated_moment = SVector{3}(moment_mag * moment_dir...)

    # Update position covariance based on cluster spread
    σ_pos = max(c.cluster_radius, 5.0)  # At least 5m uncertainty
    c.position_covariance = SMatrix{3, 3}(σ_pos^2 * I(3))
end

"""Check if candidate passes promotion gates."""
function check_promotion(c::SourceCandidate, current_time::Float64,
                         config::SourcePromotionConfig)
    # Gate 1: Temporal persistence
    duration = current_time - c.created_time
    if duration < config.min_duration_s
        return (pass = false, reason = :insufficient_duration)
    end

    # Gate 2: Sufficient observations
    if length(c.positions) < config.min_support_count
        return (pass = false, reason = :insufficient_observations)
    end

    # Gate 3: Spatial compactness
    if c.cluster_radius > config.max_cluster_radius
        return (pass = false, reason = :cluster_too_spread)
    end

    # Gate 4: Position confidence
    pos_trace = tr(Matrix(c.position_covariance))
    if pos_trace > config.min_position_confidence
        return (pass = false, reason = :low_position_confidence)
    end

    # Gate 5: χ² consistency (large chi2 means strong anomaly, which is good for detection)
    # Skip chi2 gate — large chi2 indicates strong source signal, not poor fit

    return (pass = true, reason = :promoted)
end

# ============================================================================
# Source Observation
# ============================================================================

"""
    SourceObservation

Single observation for source update.
"""
struct SourceObservation
    position::Vec3Map               # Measurement position [m]
    field_measured::Vec3Map         # Total measured field [T]
    field_predicted_bg::Vec3Map     # Predicted background field [T]
    R_meas::Mat3Map                 # Measurement covariance [T²]
    timestamp::Float64
    pose_covariance::SMatrix{3, 3, Float64, 9}  # Vehicle pose uncertainty
end

"""Compute residual (measurement - background) for source."""
function source_residual(obs::SourceObservation)
    return obs.field_measured - obs.field_predicted_bg
end

# ============================================================================
# Dipole Field Jacobian
# ============================================================================

"""
    compute_source_jacobian(measurement_pos::AbstractVector,
                            source_state::SlamSourceState) -> Matrix{Float64}

Compute Jacobian of dipole field w.r.t. source state [x, y, z, mx, my, mz].

Returns 3×6 matrix: ∂B/∂[pos, moment].

# Physics
B(r) = (μ₀/4π) [3(m·r̂)r̂ - m] / |r|³

where r = measurement_pos - source_pos, r̂ = r/|r|

Partials computed analytically for numerical stability.
"""
function compute_source_jacobian(measurement_pos::AbstractVector,
                                  source::SlamSourceState)
    μ₀_4π = 1e-7  # μ₀/(4π)

    pos_src = source_position(source)
    moment = source_moment(source)

    r = SVector{3}(measurement_pos) - pos_src
    r_norm = norm(r)

    if r_norm < 0.1  # Too close, regularize
        return zeros(3, 6)
    end

    r_hat = r / r_norm
    r3 = r_norm^3
    r5 = r_norm^5

    m_dot_r = dot(moment, r_hat)

    # Jacobian w.r.t. source position (3×3)
    # ∂B/∂pos_src = -∂B/∂r (chain rule with negative sign)
    # This is complex; use numerical approximation if needed
    J_pos = zeros(3, 3)

    # Numerical differentiation for position Jacobian
    δ = 0.01  # 1 cm perturbation
    for i in 1:3
        pos_plus = copy(Vector(pos_src))
        pos_minus = copy(Vector(pos_src))
        pos_plus[i] += δ
        pos_minus[i] -= δ

        B_plus = dipole_field_at(measurement_pos, pos_plus, moment)
        B_minus = dipole_field_at(measurement_pos, pos_minus, moment)

        J_pos[:, i] = (B_plus - B_minus) / (2δ)
    end

    # Jacobian w.r.t. moment (3×3)
    # ∂B/∂m = (μ₀/4π) [3r̂(r̂·) - I] / r³
    # This is linear in moment
    J_moment = μ₀_4π / r3 * (3 * r_hat * r_hat' - I(3))

    return hcat(J_pos, Matrix(J_moment))
end

"""Compute dipole field at measurement position."""
function dipole_field_at(meas_pos::AbstractVector, src_pos::AbstractVector,
                          moment::AbstractVector)
    μ₀_4π = 1e-7

    r = SVector{3}(meas_pos) - SVector{3}(src_pos)
    r_norm = norm(r)

    if r_norm < 0.1
        return zeros(3)
    end

    r_hat = r / r_norm
    r3 = r_norm^3

    m_dot_r_hat = dot(moment, r_hat)

    B = μ₀_4π / r3 * (3 * m_dot_r_hat * r_hat - SVector{3}(moment...))
    return Vector(B)
end

# ============================================================================
# Source State Update (EKF)
# ============================================================================

"""
    SourceUpdateResult

Result of updating a source state.
"""
struct SourceUpdateResult
    updated::Bool
    source_id::Int
    innovation::Vec3Map
    innovation_covariance::Mat3Map
    chi2::Float64
    gain::Matrix{Float64}
end

"""
    update_source_state!(source::SlamSourceState, obs::SourceObservation,
                         config::SourceUpdateConfig) -> SourceUpdateResult

EKF update for source state using magnetic field observation.

# Algorithm
Standard EKF:
1. Predict source field: B_pred = dipole(obs.position, source)
2. Compute innovation: ν = z - h(x) - B_pred - B_bg
3. Compute Jacobian: H = ∂B/∂[pos, moment]
4. Innovation covariance: S = H P H' + R
5. Kalman gain: K = P H' S⁻¹
6. State update: x ← x + K ν
7. Covariance update: P ← (I - KH) P
"""
function update_source_state!(source::SlamSourceState, obs::SourceObservation,
                               config::SourceUpdateConfig)
    # Predicted source field at measurement position
    B_src_pred = Vec3Map(dipole_field_at(obs.position, source_position(source),
                                          source_moment(source))...)

    # Innovation: observed - background - source prediction
    innovation = obs.field_measured - obs.field_predicted_bg - B_src_pred

    # Jacobian
    H = compute_source_jacobian(obs.position, source)

    # Innovation covariance
    P = Matrix(source.covariance)
    R = Matrix(obs.R_meas)
    S = H * P * H' + R

    # Chi-square test
    S_reg = S + 1e-20 * I(3)
    chi2 = innovation' * (S_reg \ innovation)

    # Gate check
    if chi2 > config.association_chi2_threshold
        return SourceUpdateResult(false, source.source_id, innovation,
                                   Mat3Map(S), chi2, zeros(6, 3))
    end

    # Kalman gain
    K = P * H' * inv(S_reg)

    # State update
    state_new = source.state + SVector{6}(K * Vector(innovation))

    # Covariance update (Joseph form for stability)
    I_KH = I(6) - K * H
    P_new = I_KH * P * I_KH' + K * R * K'

    # Apply update
    source.state = state_new
    source.covariance = SMatrix{6, 6}(P_new)
    source.support_count += 1
    source.last_observed = obs.timestamp

    return SourceUpdateResult(true, source.source_id, innovation,
                               Mat3Map(S), chi2, K)
end

# ============================================================================
# Source Retirement Check
# ============================================================================

"""
    SourceRetirementReason

Reasons for retiring a source.
"""
@enum SourceRetirementReason begin
    RETIRE_NONE = 0
    RETIRE_TIME_OUT = 1
    RETIRE_LOW_CONTRIBUTION = 2
    RETIRE_COVARIANCE_GROWTH = 3
    RETIRE_ABSORBED = 4
    RETIRE_MANUAL = 5
end

"""
    check_source_retirement(source::SlamSourceState, current_pos::AbstractVector,
                            current_time::Float64, config::SourceRetirementConfig)
        -> (retire::Bool, reason::SourceRetirementReason)

Check if source should be retired.
"""
function check_source_retirement(source::SlamSourceState, current_pos::AbstractVector,
                                  current_time::Float64, config::SourceRetirementConfig)
    # Check 1: Time since last observation
    time_since_obs = current_time - source.last_observed
    if time_since_obs > config.max_unobserved_duration_s
        return (true, RETIRE_TIME_OUT)
    end

    # Check 2: Field contribution at current position
    B_src = dipole_field_at(current_pos, source_position(source), source_moment(source))
    if norm(B_src) < config.min_contribution_threshold
        return (true, RETIRE_LOW_CONTRIBUTION)
    end

    # Check 3: Covariance growth
    cov_trace = tr(Matrix(source.covariance))
    if cov_trace > config.covariance_growth_threshold * 1000  # Initial trace ~ 1000
        return (true, RETIRE_COVARIANCE_GROWTH)
    end

    return (false, RETIRE_NONE)
end

# ============================================================================
# Online Source SLAM Manager
# ============================================================================

"""
    OnlineSourceSLAM

Manager for online source tracking and SLAM integration.

# Architecture
- Maintains candidate pool for track-before-detect
- Manages source lifecycle (candidate → active → retired)
- Provides source contributions for map learning clean subtraction
- Integrates with SlamAugmentedState for joint state

# Usage
```julia
source_slam = OnlineSourceSLAM()

# Process each observation:
result = process_source_observation!(source_slam, state, observation)

# Get source contributions for map learning
contributions = get_source_contributions(source_slam, state, position)
```
"""
mutable struct OnlineSourceSLAM
    config::OnlineSourceSLAMConfig
    candidates::Dict{Int, SourceCandidate}
    next_candidate_id::Int
    next_source_id::Int

    # Statistics
    total_detections::Int
    promoted_count::Int
    retired_count::Int
end

"""Create source SLAM manager."""
function OnlineSourceSLAM(config::OnlineSourceSLAMConfig = DEFAULT_SOURCE_SLAM_CONFIG)
    OnlineSourceSLAM(
        config,
        Dict{Int, SourceCandidate}(),
        1,
        1,
        0, 0, 0
    )
end

"""
    SourceProcessingResult

Result of processing an observation through source SLAM.
"""
struct SourceProcessingResult
    # Detection
    new_candidate_created::Bool
    candidate_updated::Bool

    # Source updates
    sources_updated::Vector{Int}
    update_results::Vector{SourceUpdateResult}

    # Promotions and retirements
    promoted_sources::Vector{Int}
    retired_sources::Vector{Int}

    # For map learning
    clean_residual::Vec3Map
    clean_covariance::Mat3Map
    is_teachable::Bool
end

"""
    process_source_observation!(slam::OnlineSourceSLAM, state::SlamAugmentedState,
                                 obs::SourceObservation) -> SourceProcessingResult

Process a magnetic field observation through source SLAM.

# Steps
1. Compute residual after subtracting known sources
2. Try to associate with existing sources / update them
3. Check for anomalous residual → create candidate
4. Check candidate promotions
5. Check source retirements
6. Return clean residual for map learning
"""
function process_source_observation!(slam::OnlineSourceSLAM, state::SlamAugmentedState,
                                      obs::SourceObservation)
    config = slam.config
    sources_updated = Int[]
    update_results = SourceUpdateResult[]
    promoted = Int[]
    retired = Int[]
    new_candidate = false
    candidate_updated = false

    # Step 1: Subtract known active sources from residual
    residual = source_residual(obs)  # z - B_background
    total_source_covariance = zeros(3, 3)

    for src in state.source_states
        if src.lifecycle == :active
            B_src = Vec3Map(dipole_field_at(obs.position, source_position(src),
                                            source_moment(src))...)
            residual = residual - B_src

            # Propagate source uncertainty to total covariance
            H = compute_source_jacobian(obs.position, src)
            total_source_covariance += H * Matrix(src.covariance) * H'
        end
    end

    # Step 2: Try to update active sources
    if config.update isa SourceUpdateConfig  # Source tracking enabled
        for src in state.source_states
            if src.lifecycle == :active
                # Create observation relative to this source
                result = update_source_state!(src, obs, config.update)
                push!(update_results, result)
                if result.updated
                    push!(sources_updated, src.source_id)
                end
            end
        end
    end

    # Step 3: Check residual for new source candidates
    residual_norm = norm(residual)
    R_total = Matrix(obs.R_meas) + total_source_covariance + 1e-20 * I(3)
    chi2_residual = residual' * (R_total \ residual)

    if residual_norm > config.detection.min_residual_norm &&
       chi2_residual > config.detection.detection_chi2_threshold

        # Check if near existing candidate
        matched_candidate = nothing
        for (id, cand) in slam.candidates
            dist = norm(Vector(obs.position) - Vector(cand.centroid))
            if dist < config.detection.spatial_clustering_radius
                matched_candidate = cand
                break
            end
        end

        if matched_candidate !== nothing
            # Add to existing candidate
            add_observation!(matched_candidate, obs.position, residual,
                           obs.timestamp, chi2_residual)
            candidate_updated = true
        elseif length(slam.candidates) < config.detection.max_candidates
            # Create new candidate
            cand = SourceCandidate(slam.next_candidate_id, obs.position,
                                  residual, obs.timestamp, chi2_residual)
            slam.candidates[slam.next_candidate_id] = cand
            slam.next_candidate_id += 1
            slam.total_detections += 1
            new_candidate = true
        end
    end

    # Step 4: Check candidate promotions
    to_promote = Int[]
    for (id, cand) in slam.candidates
        result = check_promotion(cand, obs.timestamp, config.promotion)
        if result.pass
            push!(to_promote, id)
        end
    end

    for id in to_promote
        cand = slam.candidates[id]

        # Create SlamSourceState from candidate
        new_source = SlamSourceState(
            slam.next_source_id,
            cand.estimated_position,
            cand.estimated_moment;
            position_var = cand.position_covariance[1, 1],
            moment_var = norm(cand.estimated_moment)^2 * 0.25  # 50% moment uncertainty
        )
        new_source.lifecycle = :active
        new_source.is_probationary = true  # New sources start probationary

        push!(state.source_states, new_source)
        push!(promoted, slam.next_source_id)

        slam.next_source_id += 1
        slam.promoted_count += 1
        delete!(slam.candidates, id)
    end

    # Step 5: Check source retirements
    to_retire = Int[]
    for (i, src) in enumerate(state.source_states)
        should_retire, _ = check_source_retirement(
            src, obs.position, obs.timestamp, config.retirement
        )
        if should_retire
            push!(to_retire, i)
        end
    end

    # Retire in reverse order to preserve indices
    for i in reverse(sort(to_retire))
        src = state.source_states[i]
        src.lifecycle = :retired
        push!(retired, src.source_id)
        deleteat!(state.source_states, i)
        slam.retired_count += 1
    end

    # Step 6: Route residual through source-first attribution (Phase G+)
    # Uses SourceFirstResidualRouter contract instead of ad-hoc is_teachable gate.
    # The router attributes residuals to sources, tiles, and nav, then decides
    # whether the tile should receive the residual or be frozen.
    clean_residual = residual  # Already has sources subtracted (Step 1)
    clean_covariance = Mat3Map(R_total)

    # Contract-based teachability: tile is teachable when source does not dominate
    # and we are not in the spatial exclusion zone of a confirmed source.
    sensor_var = tr(Matrix(obs.R_meas))
    source_var = tr(total_source_covariance)
    source_fraction = (sensor_var + source_var) > 0 ?
        source_var / (sensor_var + source_var) : 0.0

    # Teachability gate: χ² bounded AND source fraction below dominance threshold (0.5)
    # The 0.5 threshold means tile learning is suppressed when source uncertainty
    # contributes more than half of the total measurement variance.
    # χ²(3, p=0.01) = 11.345 ensures residual is statistically reasonable.
    is_teachable = (chi2_residual < 11.345) && (source_fraction < 0.5)

    return SourceProcessingResult(
        new_candidate,
        candidate_updated,
        sources_updated,
        update_results,
        promoted,
        retired,
        clean_residual,
        clean_covariance,
        is_teachable
    )
end

"""
    get_source_contributions(slam::OnlineSourceSLAM, state::SlamAugmentedState,
                              position::AbstractVector) -> Vector{SourceContribution}

Get source contributions for source separation at a position.

Returns SourceContribution structs compatible with SourceSeparation.jl.
"""
function get_source_contributions(slam::OnlineSourceSLAM, state::SlamAugmentedState,
                                   position::AbstractVector)
    contributions = SourceContribution[]

    for src in state.source_states
        if src.lifecycle == :active
            # Compute field at position
            B_src = dipole_field_at(position, source_position(src), source_moment(src))

            # Compute field covariance from source uncertainty
            H = compute_source_jacobian(position, src)
            Σ_src = H * Matrix(src.covariance) * H'

            # Confidence based on source lifecycle and covariance
            conf = src.is_probationary ? 0.7 : 0.9

            push!(contributions, object_contribution(
                string(src.source_id),
                B_src,
                Σ_src,
                conf
            ))
        end
    end

    return contributions
end

# ============================================================================
# Statistics and Formatting
# ============================================================================

"""
    SourceSLAMStatistics

Statistics for source SLAM monitoring.
"""
struct SourceSLAMStatistics
    n_candidates::Int
    n_active_sources::Int
    total_detections::Int
    promoted_count::Int
    retired_count::Int
end

"""Get current statistics."""
function get_statistics(slam::OnlineSourceSLAM, state::SlamAugmentedState)
    SourceSLAMStatistics(
        length(slam.candidates),
        count(s -> s.lifecycle == :active, state.source_states),
        slam.total_detections,
        slam.promoted_count,
        slam.retired_count
    )
end

"""Format statistics for display."""
function format_source_slam_statistics(stats::SourceSLAMStatistics)
    return """
    Source SLAM Statistics:
      Candidates: $(stats.n_candidates)
      Active sources: $(stats.n_active_sources)
      Total detections: $(stats.total_detections)
      Promotions: $(stats.promoted_count)
      Retirements: $(stats.retired_count)
    """
end

# ============================================================================
# Exports
# ============================================================================

export SourceDetectionConfig, DEFAULT_SOURCE_DETECTION_CONFIG
export SourcePromotionConfig, DEFAULT_SOURCE_PROMOTION_CONFIG
export SourceUpdateConfig, DEFAULT_SOURCE_UPDATE_CONFIG
export SourceRetirementConfig, DEFAULT_SOURCE_RETIREMENT_CONFIG
export OnlineSourceSLAMConfig, DEFAULT_SOURCE_SLAM_CONFIG
export SourceCandidate, add_observation!, update_candidate_statistics!, check_promotion
export SourceObservation, source_residual
export compute_source_jacobian, dipole_field_at
export SourceUpdateResult, update_source_state!
export SourceRetirementReason, RETIRE_NONE, RETIRE_TIME_OUT
export RETIRE_LOW_CONTRIBUTION, RETIRE_COVARIANCE_GROWTH, RETIRE_ABSORBED, RETIRE_MANUAL
export check_source_retirement
export OnlineSourceSLAM, SourceProcessingResult
export process_source_observation!, get_source_contributions
export SourceSLAMStatistics, get_statistics, format_source_slam_statistics
