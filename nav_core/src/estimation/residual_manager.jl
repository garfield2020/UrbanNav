# ============================================================================
# Residual Management for Anomaly Detection
# ============================================================================
#
# Ported from AUV-Navigation/src/residual.jl
#
# Chi-square gating for outlier detection and feature promotion ladder
# for persistent anomaly identification.
#
# Key concepts:
# - γ = r' * Σ^{-1} * r follows χ²(d) distribution
# - Gating thresholds at p=0.01 and p=0.001
# - Feature promotion: Candidate → Confirmed → Demoted → Absorbed
# ============================================================================

using LinearAlgebra
using Distributions: Chisq, cdf, quantile
using Statistics: mean

export GatingDecision, INLIER, MILD_OUTLIER, STRONG_OUTLIER
export GatingConfig, CHI2_D3_MILD_THRESHOLD, CHI2_D3_STRONG_THRESHOLD
export chi2_threshold, compute_chi2, gate_measurement
export ResidualStatistics, compute_statistics
export PhysicsGateConfig, DEFAULT_PHYSICS_GATE_CONFIG, DISABLED_PHYSICS_GATE_CONFIG
export PhysicsGatedStatistics, compute_statistics_with_physics_gate
export FeatureStatus, CANDIDATE, CONFIRMED, DEMOTED, ABSORBED
export FeatureCandidate, ResidualManager
export process_measurement!, get_confirmed_features, get_active_candidates
export get_statistics_summary

# ============================================================================
# Gating Decisions
# ============================================================================

"""
    GatingDecision

Result of chi-square gating.
"""
@enum GatingDecision begin
    INLIER          # Below p=0.01 threshold
    MILD_OUTLIER    # Between p=0.01 and p=0.001
    STRONG_OUTLIER  # Above p=0.001 threshold
end

# ============================================================================
# Chi-Square Thresholds
# ============================================================================

# Hardcoded for d=3 (B-only measurement)
# χ²(3, 0.99)  = 11.3449 → P(γ > 11.3449) = 0.01
# χ²(3, 0.999) = 16.2662 → P(γ > 16.2662) = 0.001
const CHI2_D3_MILD_THRESHOLD = 11.3449
const CHI2_D3_STRONG_THRESHOLD = 16.2662

"""
    GatingConfig

Configuration for chi-square gating.

# Fields
- `dof::Int`: Degrees of freedom (measurement dimension)
- `threshold_mild::Float64`: Chi-square threshold for mild outlier (p=0.01)
- `threshold_strong::Float64`: Chi-square threshold for strong outlier (p=0.001)
- `min_observations::Int`: Minimum observations to promote candidate
- `confirmation_threshold::Int`: Observations needed for confirmation
- `demotion_threshold::Int`: Inlier observations to demote feature
"""
struct GatingConfig
    dof::Int
    threshold_mild::Float64
    threshold_strong::Float64
    min_observations::Int
    confirmation_threshold::Int
    demotion_threshold::Int
end

function GatingConfig(;
    dof::Int = 3,
    threshold_mild::Real = CHI2_D3_MILD_THRESHOLD,
    threshold_strong::Real = CHI2_D3_STRONG_THRESHOLD,
    min_observations::Int = 3,
    confirmation_threshold::Int = 5,
    demotion_threshold::Int = 10
)
    if dof != 3 && threshold_mild == CHI2_D3_MILD_THRESHOLD
        @warn "GatingConfig: dof=$dof but using d=3 thresholds"
    end

    GatingConfig(dof, Float64(threshold_mild), Float64(threshold_strong),
                 min_observations, confirmation_threshold, demotion_threshold)
end

"""
    chi2_threshold(dof::Int, p_value::Real)

Get chi-square threshold for given degrees of freedom and p-value.
"""
function chi2_threshold(dof::Int, p_value::Real)
    quantile(Chisq(dof), 1 - p_value)
end

"""
    compute_chi2(residual::AbstractVector, covariance::AbstractMatrix)

Compute chi-square statistic: γ = r' * Σ^{-1} * r
"""
function compute_chi2(residual::AbstractVector, covariance::AbstractMatrix)
    result = whiten_full(residual, covariance; check_shapes=true, check_spd=false)
    return result.γ
end

"""
    gate_measurement(gamma::Real, config::GatingConfig)

Apply chi-square gating to determine if measurement is an outlier.
"""
function gate_measurement(gamma::Real, config::GatingConfig)
    if gamma > config.threshold_strong
        return STRONG_OUTLIER
    elseif gamma > config.threshold_mild
        return MILD_OUTLIER
    else
        return INLIER
    end
end

# ============================================================================
# Residual Statistics
# ============================================================================

"""
    ResidualStatistics

Statistics computed from a measurement residual.

# Fields
- `chi2::Float64`: Chi-square statistic (γ)
- `dof::Int`: Degrees of freedom
- `p_value::Float64`: p-value for the chi-square statistic
- `decision::GatingDecision`: Gating decision
- `position::Vec3`: Position where measurement was taken
- `timestamp::Float64`: Measurement timestamp
- `residual::Vector{Float64}`: Raw residual vector
"""
struct ResidualStatistics
    chi2::Float64
    dof::Int
    p_value::Float64
    decision::GatingDecision
    position::Vec3
    timestamp::Float64
    residual::Vector{Float64}
end

"""
    compute_statistics(residual, covariance, position, timestamp, config)

Compute residual statistics with chi-square gating.
"""
function compute_statistics(
    residual::AbstractVector,
    covariance::AbstractMatrix,
    position::AbstractVector,
    timestamp::Real,
    config::GatingConfig = GatingConfig()
)
    dof = length(residual)

    if dof != config.dof
        @warn "compute_statistics: residual dimension ($dof) != config.dof ($(config.dof))"
    end

    gamma = compute_chi2(residual, covariance)
    p_value = 1 - cdf(Chisq(dof), gamma)
    decision = gate_measurement(gamma, config)

    ResidualStatistics(
        Float64(gamma),
        dof,
        p_value,
        decision,
        Vec3(position...),
        Float64(timestamp),
        Vector{Float64}(residual)
    )
end

# ============================================================================
# Physics Gate Configuration
# ============================================================================

"""
    PhysicsGateConfig

Configuration for Maxwell physics gate.
"""
struct PhysicsGateConfig
    enabled::Bool
    σ_B::Float64
    σ_G::Float64
    eigenvalue_ratio_min::Float64
    eigenvalue_ratio_max::Float64
    max_fg_ratio::Float64
    temporal_sigma::Float64
    snr_threshold::Float64
end

function PhysicsGateConfig(;
    enabled::Bool = true,
    σ_B::Real = 5e-9,
    σ_G::Real = 50e-9,
    eigenvalue_ratio_min::Real = 0.1,
    eigenvalue_ratio_max::Real = 0.95,
    max_fg_ratio::Real = 5.0,
    temporal_sigma::Real = 4.0,
    snr_threshold::Real = 5.0
)
    PhysicsGateConfig(
        enabled, Float64(σ_B), Float64(σ_G),
        Float64(eigenvalue_ratio_min), Float64(eigenvalue_ratio_max),
        Float64(max_fg_ratio), Float64(temporal_sigma), Float64(snr_threshold)
    )
end

const DEFAULT_PHYSICS_GATE_CONFIG = PhysicsGateConfig()
const DISABLED_PHYSICS_GATE_CONFIG = PhysicsGateConfig(enabled=false)

"""
    PhysicsGatedStatistics

Extended statistics including Maxwell physics gate result.
"""
struct PhysicsGatedStatistics
    stats::ResidualStatistics
    physics_passed::Bool
    physics_result::Union{Nothing, MaxwellGateResult}
end

"""
    compute_statistics_with_physics_gate(...)

Compute residual statistics with Maxwell physics gate pre-filtering.
"""
function compute_statistics_with_physics_gate(
    residual_B::AbstractVector,
    residual_G::AbstractMatrix,
    covariance::AbstractMatrix,
    position::AbstractVector,
    timestamp::Real;
    gating_config::GatingConfig = GatingConfig(),
    physics_config::PhysicsGateConfig = DEFAULT_PHYSICS_GATE_CONFIG,
    B_previous::Union{Nothing, AbstractVector} = nothing,
    G_previous::Union{Nothing, AbstractMatrix} = nothing,
    pos_previous::Union{Nothing, AbstractVector} = nothing
)
    stats = compute_statistics(residual_B, covariance, position, timestamp, gating_config)

    physics_passed = true
    physics_result = nothing

    if physics_config.enabled
        Δpos = nothing
        if !isnothing(pos_previous)
            Δpos = Vec3(position...) - Vec3(pos_previous...)
        end

        physics_result = apply_maxwell_gate(
            residual_B,
            residual_G;
            σ_B = physics_config.σ_B,
            σ_G = physics_config.σ_G,
            B_previous = B_previous,
            G_previous = G_previous,
            Δpos = Δpos,
            snr_threshold = physics_config.snr_threshold
        )

        physics_passed = physics_result.passes
    end

    return PhysicsGatedStatistics(stats, physics_passed, physics_result)
end

# ============================================================================
# Feature Promotion Ladder
# ============================================================================

"""
    FeatureStatus

Status in the feature promotion ladder.
"""
@enum FeatureStatus begin
    CANDIDATE       # Initial detection
    CONFIRMED       # Multiple consistent observations
    DEMOTED         # Was confirmed but now seeing inliers
    ABSORBED        # Incorporated into background model
end

"""
    FeatureCandidate

A candidate magnetic feature (potential anomaly).

# Fields
- `id::Int`: Unique identifier
- `position::Vec3`: Estimated position
- `status::FeatureStatus`: Current ladder status
- `observations::Int`: Number of outlier observations
- `inlier_count::Int`: Consecutive inlier observations
- `chi2_history::Vector{Float64}`: History of chi-square values
- `first_seen::Float64`: Timestamp of first detection
- `last_seen::Float64`: Timestamp of most recent detection
"""
mutable struct FeatureCandidate
    id::Int
    position::Vec3
    status::FeatureStatus
    observations::Int
    inlier_count::Int
    chi2_history::Vector{Float64}
    first_seen::Float64
    last_seen::Float64
end

function FeatureCandidate(id::Int, position::AbstractVector, timestamp::Real)
    FeatureCandidate(
        id,
        Vec3(position...),
        CANDIDATE,
        1,
        0,
        Float64[],
        Float64(timestamp),
        Float64(timestamp)
    )
end

# ============================================================================
# Residual Manager
# ============================================================================

"""
    ResidualManager

Manages residual statistics and feature promotion.

# Fields
- `config::GatingConfig`: Gating configuration
- `candidates::Dict{Int, FeatureCandidate}`: Active feature candidates
- `confirmed_features::Vector{FeatureCandidate}`: Confirmed features
- `next_id::Int`: Next feature ID to assign
- `association_radius::Float64`: Maximum distance to associate observation
- `statistics_history::Vector{ResidualStatistics}`: Recent statistics
- `max_history::Int`: Maximum history to keep
"""
mutable struct ResidualManager
    config::GatingConfig
    candidates::Dict{Int, FeatureCandidate}
    confirmed_features::Vector{FeatureCandidate}
    next_id::Int
    association_radius::Float64
    statistics_history::Vector{ResidualStatistics}
    max_history::Int
end

function ResidualManager(;
    config::GatingConfig = GatingConfig(),
    association_radius::Real = 5.0,
    max_history::Int = 1000
)
    ResidualManager(
        config,
        Dict{Int, FeatureCandidate}(),
        FeatureCandidate[],
        1,
        Float64(association_radius),
        ResidualStatistics[],
        max_history
    )
end

"""
    process_measurement!(manager::ResidualManager, stats::ResidualStatistics)

Process a measurement through the feature promotion ladder.
"""
function process_measurement!(manager::ResidualManager, stats::ResidualStatistics)
    push!(manager.statistics_history, stats)
    if length(manager.statistics_history) > manager.max_history
        popfirst!(manager.statistics_history)
    end

    if stats.decision == INLIER
        _update_candidates_inlier!(manager, stats)
        return nothing
    end

    candidate = _find_nearby_candidate(manager, stats.position)

    if candidate === nothing
        candidate = FeatureCandidate(manager.next_id, stats.position, stats.timestamp)
        manager.candidates[manager.next_id] = candidate
        manager.next_id += 1
    else
        _update_candidate!(manager, candidate, stats)
    end

    push!(candidate.chi2_history, stats.chi2)
    _check_promotion!(manager, candidate)

    return candidate
end

function _find_nearby_candidate(manager::ResidualManager, position::Vec3)
    best_candidate = nothing
    best_distance = Inf

    for (_, candidate) in manager.candidates
        d = norm(candidate.position - position)
        if d < manager.association_radius && d < best_distance
            best_distance = d
            best_candidate = candidate
        end
    end

    return best_candidate
end

function _update_candidate!(manager::ResidualManager, candidate::FeatureCandidate, stats::ResidualStatistics)
    candidate.observations += 1
    candidate.inlier_count = 0
    candidate.last_seen = stats.timestamp

    α = 1.0 / candidate.observations
    candidate.position = (1 - α) * candidate.position + α * stats.position
end

function _update_candidates_inlier!(manager::ResidualManager, stats::ResidualStatistics)
    for (id, candidate) in manager.candidates
        d = norm(candidate.position - stats.position)
        if d < manager.association_radius
            candidate.inlier_count += 1

            if candidate.inlier_count >= manager.config.demotion_threshold
                if candidate.status == CONFIRMED
                    candidate.status = DEMOTED
                end
            end
        end
    end
end

function _check_promotion!(manager::ResidualManager, candidate::FeatureCandidate)
    if candidate.status == CANDIDATE &&
       candidate.observations >= manager.config.confirmation_threshold
        candidate.status = CONFIRMED
        push!(manager.confirmed_features, candidate)
    end
end

"""
    get_confirmed_features(manager::ResidualManager)

Get all confirmed feature candidates.
"""
get_confirmed_features(manager::ResidualManager) = manager.confirmed_features

"""
    get_active_candidates(manager::ResidualManager)

Get all active (non-confirmed) candidates.
"""
get_active_candidates(manager::ResidualManager) = [c for (_, c) in manager.candidates if c.status == CANDIDATE]

"""
    get_statistics_summary(manager::ResidualManager)

Get summary statistics of residual history.
"""
function get_statistics_summary(manager::ResidualManager)
    if isempty(manager.statistics_history)
        return (
            mean_chi2 = 0.0,
            max_chi2 = 0.0,
            outlier_fraction = 0.0,
            n_measurements = 0
        )
    end

    chi2_vals = [s.chi2 for s in manager.statistics_history]
    n_outliers = count(s -> s.decision != INLIER, manager.statistics_history)

    return (
        mean_chi2 = mean(chi2_vals),
        max_chi2 = maximum(chi2_vals),
        outlier_fraction = n_outliers / length(manager.statistics_history),
        n_measurements = length(manager.statistics_history)
    )
end
