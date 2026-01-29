# ============================================================================
# Feature Lifecycle & Retirement
# ============================================================================
#
# Ported from AUV-Navigation/src/feature_lifecycle.jl
#
# Manages feature aging to prevent long-term map pollution.
# Features must:
# - Track support decay (exponential)
# - Monitor contribution-to-cost
# - Retire when not re-observed or contribution drops
# - Demote when confidence degrades
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Lifecycle Configuration
# ============================================================================

"""
    FeatureLifecycleConfig

Configuration for feature lifecycle management.
"""
struct FeatureLifecycleConfig
    T_retire::Float64                    # Time without observation → retire
    decay_rate::Float64                  # Support decay rate (1/s)
    min_contribution::Float64            # Minimum marginal contribution
    low_contribution_streak_max::Int     # Max consecutive low contributions
    low_confidence_streak_max::Int       # Max consecutive low confidence
    min_effective_support::Float64       # Minimum effective support
    min_observation_count::Int              # Minimum observations before promotion
end

# Tuned for urban dynamics: elevators, vehicles, doors, and other
# transient magnetic sources that appear/disappear on short timescales.
function FeatureLifecycleConfig(;
    T_retire::Real = 15.0,               # 15s without observation → retire (urban sources vanish fast)
    decay_rate::Real = 0.15,             # ~4.6s half-life (fast decay for transient urban sources)
    min_contribution::Real = 0.01,       # 1% of cost (kept for urban SNR levels)
    low_contribution_streak_max::Int = 10,
    low_confidence_streak_max::Int = 5,
    min_effective_support::Float64 = 1.0, # At least 1 "effective" measurement
    min_observation_count::Int = 3        # Faster promotion in dense urban environments
)
    FeatureLifecycleConfig(
        Float64(T_retire), Float64(decay_rate), Float64(min_contribution),
        low_contribution_streak_max, low_confidence_streak_max, min_effective_support,
        min_observation_count
    )
end

const DEFAULT_LIFECYCLE_CONFIG = FeatureLifecycleConfig()

# ============================================================================
# Lifecycle State per Feature
# ============================================================================

"""
    FeatureLifecycleState

Tracks lifecycle-specific state for a dipole feature.
"""
mutable struct FeatureLifecycleState
    effective_support::Float64
    marginal_contribution::Float64
    contribution_history::Vector{Float64}
    low_contribution_streak::Int
    low_confidence_streak::Int
    total_cost_reduction::Float64
    observation_count::Int
end

"""Create default lifecycle state."""
function FeatureLifecycleState()
    FeatureLifecycleState(
        1.0,              # Initial support of 1
        1.0,              # Assume 100% initial contribution
        Float64[],        # Empty history
        0, 0,             # No streaks
        0.0, 0            # No observations yet
    )
end

# ============================================================================
# Lifecycle Manager
# ============================================================================

"""
    FeatureLifecycleManager

Manages lifecycle state for all dipoles in the registry.
"""
mutable struct FeatureLifecycleManager
    config::FeatureLifecycleConfig
    states::Dict{Int, FeatureLifecycleState}  # dipole_id → lifecycle state
    last_update_time::Float64
end

function FeatureLifecycleManager(; config::FeatureLifecycleConfig = DEFAULT_LIFECYCLE_CONFIG)
    FeatureLifecycleManager(config, Dict{Int, FeatureLifecycleState}(), 0.0)
end

"""Initialize lifecycle tracking for a dipole."""
function init_feature_lifecycle!(mgr::FeatureLifecycleManager, dipole_id::Int)
    mgr.states[dipole_id] = FeatureLifecycleState()
end

"""Get lifecycle state for a dipole."""
function get_feature_lifecycle(mgr::FeatureLifecycleManager, dipole_id::Int)
    get(mgr.states, dipole_id, nothing)
end

"""Remove lifecycle state (on retirement/demotion)."""
function remove_feature_lifecycle!(mgr::FeatureLifecycleManager, dipole_id::Int)
    delete!(mgr.states, dipole_id)
end

# ============================================================================
# Support Decay
# ============================================================================

"""
    compute_effective_support(initial_support, decay_rate, dt)

Compute effective support after time dt with exponential decay.
"""
function compute_effective_support(initial_support::Float64, decay_rate::Float64, dt::Float64)
    initial_support * exp(-decay_rate * dt)
end

"""
    decay_all_feature_support!(mgr, current_time)

Apply exponential decay to all dipole support levels.
"""
function decay_all_feature_support!(mgr::FeatureLifecycleManager, current_time::Float64)
    dt = current_time - mgr.last_update_time
    if dt <= 0
        return
    end

    for (id, state) in mgr.states
        state.effective_support = compute_effective_support(
            state.effective_support, mgr.config.decay_rate, dt
        )
    end

    mgr.last_update_time = current_time
end

"""
    add_feature_observation!(mgr, dipole_id, contribution, confidence)

Record an observation of a dipole, boosting effective support.
"""
function add_feature_observation!(mgr::FeatureLifecycleManager, dipole_id::Int,
                                  contribution::Float64, confidence::Float64)
    state = get_feature_lifecycle(mgr, dipole_id)
    isnothing(state) && return

    # Boost effective support
    state.effective_support += 1.0

    # Update contribution tracking
    state.marginal_contribution = contribution
    push!(state.contribution_history, contribution)

    # Keep only last 10 contributions
    if length(state.contribution_history) > 10
        popfirst!(state.contribution_history)
    end

    # Update streaks
    if contribution < mgr.config.min_contribution
        state.low_contribution_streak += 1
    else
        state.low_contribution_streak = 0
    end

    min_confidence = 0.5  # MEDIUM threshold
    if confidence < min_confidence
        state.low_confidence_streak += 1
    else
        state.low_confidence_streak = 0
    end

    # Accumulate totals
    state.total_cost_reduction += contribution
    state.observation_count += 1
end

# ============================================================================
# Contribution Metric
# ============================================================================

"""
    compute_contribution_metric(residual_with, residual_without)

Compute the marginal contribution of a dipole to cost reduction.
"""
function compute_contribution_metric(residual_with::Float64, residual_without::Float64)
    if residual_without < 1e-10
        return 0.0
    end
    contribution = (residual_without - residual_with) / residual_without
    return clamp(contribution, 0.0, 1.0)
end

"""
    evaluate_feature_contribution(feature_state, measurement_pos, measurement_residual, noise_cov)

Evaluate how much a dipole explains a measurement residual.
"""
function evaluate_feature_contribution(feature_state::DipoleFeatureState,
                                       measurement_pos::AbstractVector,
                                       measurement_residual::AbstractVector,
                                       noise_cov::AbstractMatrix)
    # Predict dipole contribution
    B_dipole = feature_field(measurement_pos, feature_state)

    # Residual without dipole
    r_without = measurement_residual
    cost_without = dot(r_without, noise_cov \ r_without)

    # Residual with dipole subtracted
    r_with = r_without - B_dipole
    cost_with = dot(r_with, noise_cov \ r_with)

    contribution = compute_contribution_metric(cost_with, cost_without)
    return (cost_with, contribution)
end

# ============================================================================
# Retirement & Demotion Rules
# ============================================================================

"""
    FeatureRetirementReason

Reason for feature retirement or demotion.
"""
@enum FeatureRetirementReason begin
    FEATURE_NOT_OBSERVED      # Not re-observed after T_retire
    FEATURE_LOW_CONTRIBUTION  # Marginal contribution < ε for N iterations
    FEATURE_LOW_SUPPORT       # Effective support dropped below threshold
    FEATURE_CONFIDENCE_DEGRADED # Confidence dropped persistently
    FEATURE_ABSORBED_BY_MAP   # Absorbed into basis functions
    FEATURE_MANUAL            # Manually retired
end

"""
    FeatureLifecycleDecision

Decision about a feature's lifecycle.
"""
struct FeatureLifecycleDecision
    dipole_id::Int
    action::Symbol          # :keep, :retire, :demote
    reason::Union{FeatureRetirementReason, Nothing}
    effective_support::Float64
    marginal_contribution::Float64
    time_since_observation::Float64
end

"""
    check_feature_retirement(dipole, lifecycle_state, config, current_time)

Check if a dipole should be retired or demoted.
"""
function check_feature_retirement(dipole::DipoleFeatureNode, state::FeatureLifecycleState,
                                  config::FeatureLifecycleConfig, current_time::Float64)
    time_since_obs = current_time - dipole.last_observed

    # Rule 1: Not observed after T_retire
    if time_since_obs > config.T_retire
        return FeatureLifecycleDecision(
            dipole.id, :retire, FEATURE_NOT_OBSERVED,
            state.effective_support, state.marginal_contribution, time_since_obs
        )
    end

    # Rule 2: Low contribution for too long
    if state.low_contribution_streak >= config.low_contribution_streak_max
        return FeatureLifecycleDecision(
            dipole.id, :retire, FEATURE_LOW_CONTRIBUTION,
            state.effective_support, state.marginal_contribution, time_since_obs
        )
    end

    # Rule 3: Effective support dropped too low
    if state.effective_support < config.min_effective_support
        return FeatureLifecycleDecision(
            dipole.id, :retire, FEATURE_LOW_SUPPORT,
            state.effective_support, state.marginal_contribution, time_since_obs
        )
    end

    # Rule 4: Confidence degraded persistently → demote
    if state.low_confidence_streak >= config.low_confidence_streak_max
        return FeatureLifecycleDecision(
            dipole.id, :demote, FEATURE_CONFIDENCE_DEGRADED,
            state.effective_support, state.marginal_contribution, time_since_obs
        )
    end

    # Feature is healthy
    return FeatureLifecycleDecision(
        dipole.id, :keep, nothing,
        state.effective_support, state.marginal_contribution, time_since_obs
    )
end

"""
    process_feature_lifecycle!(registry, lifecycle_mgr, current_time)

Process all dipoles and retire/demote as needed.

Returns vector of FeatureLifecycleDecision for all processed dipoles.
"""
function process_feature_lifecycle!(registry::DipoleFeatureRegistry,
                                    mgr::FeatureLifecycleManager,
                                    current_time::Float64)
    # First, apply support decay
    decay_all_feature_support!(mgr, current_time)

    decisions = FeatureLifecycleDecision[]

    # Check each active dipole
    for (id, dipole) in registry.features
        state = get_feature_lifecycle(mgr, id)
        isnothing(state) && continue

        decision = check_feature_retirement(dipole, state, mgr.config, current_time)
        push!(decisions, decision)

        # Execute decision
        if decision.action == :retire
            retire_dipole_feature!(registry, id)
            remove_feature_lifecycle!(mgr, id)
        elseif decision.action == :demote
            demote_dipole_feature!(registry, id)
            remove_feature_lifecycle!(mgr, id)
        end
    end

    return decisions
end

# ============================================================================
# Observation Update (called from estimator)
# ============================================================================

"""
    update_feature_observation!(registry, lifecycle_mgr, dipole_id,
                                measurement_pos, measurement_residual,
                                noise_cov, confidence, current_time)

Update a dipole with a new observation.
"""
function update_feature_observation!(registry::DipoleFeatureRegistry,
                                     mgr::FeatureLifecycleManager,
                                     dipole_id::Int,
                                     measurement_pos::AbstractVector,
                                     measurement_residual::AbstractVector,
                                     noise_cov::AbstractMatrix,
                                     confidence::Float64,
                                     current_time::Float64)
    dipole = get_dipole_feature(registry, dipole_id)
    isnothing(dipole) && return nothing

    # Update dipole observation time
    dipole.last_observed = current_time
    dipole.support_count += 1

    # Compute contribution
    _, contribution = evaluate_feature_contribution(
        dipole.state, measurement_pos, measurement_residual, noise_cov
    )

    # Update lifecycle state
    add_feature_observation!(mgr, dipole_id, contribution, confidence)

    return contribution
end

# ============================================================================
# Lifecycle Statistics
# ============================================================================

"""
    FeatureLifecycleStats

Summary statistics for lifecycle management.
"""
struct FeatureLifecycleStats
    n_active::Int
    n_retired::Int
    n_demoted::Int
    mean_effective_support::Float64
    mean_contribution::Float64
    oldest_dipole_age::Float64
end

"""
    get_feature_lifecycle_stats(registry, lifecycle_mgr, current_time)

Get summary statistics for feature lifecycle.
"""
function get_feature_lifecycle_stats(registry::DipoleFeatureRegistry,
                                     mgr::FeatureLifecycleManager,
                                     current_time::Float64)
    n_active = n_dipole_features(registry)
    n_retired = length(registry.retired)
    n_demoted = length(registry.demoted)

    if n_active == 0
        return FeatureLifecycleStats(0, n_retired, n_demoted, 0.0, 0.0, 0.0)
    end

    total_support = 0.0
    total_contribution = 0.0
    oldest_age = 0.0

    for (id, dipole) in registry.features
        state = get_feature_lifecycle(mgr, id)
        if !isnothing(state)
            total_support += state.effective_support
            total_contribution += state.marginal_contribution
        end

        age = current_time - dipole.created_time
        oldest_age = max(oldest_age, age)
    end

    mean_support = total_support / n_active
    mean_contribution = total_contribution / n_active

    return FeatureLifecycleStats(
        n_active, n_retired, n_demoted,
        mean_support, mean_contribution, oldest_age
    )
end

# ============================================================================
# Exports
# ============================================================================

export FeatureLifecycleConfig, DEFAULT_LIFECYCLE_CONFIG
export FeatureLifecycleState, FeatureLifecycleManager
export init_feature_lifecycle!, get_feature_lifecycle, remove_feature_lifecycle!
export compute_effective_support, decay_all_feature_support!, add_feature_observation!
export compute_contribution_metric, evaluate_feature_contribution
export FeatureRetirementReason
export FEATURE_NOT_OBSERVED, FEATURE_LOW_CONTRIBUTION, FEATURE_LOW_SUPPORT
export FEATURE_CONFIDENCE_DEGRADED, FEATURE_ABSORBED_BY_MAP, FEATURE_MANUAL
export FeatureLifecycleDecision, check_feature_retirement, process_feature_lifecycle!
export update_feature_observation!
export FeatureLifecycleStats, get_feature_lifecycle_stats
