# ============================================================================
# Map Update Policy - Learn vs Freeze Decision Logic
# ============================================================================
#
# Ported from AUV-Navigation/src/map_update_policy.jl
#
# Implements adaptive policies for deciding when to update map coefficients
# versus freezing them.
#
# Key principles:
# 1. Only update when navigation confidence is high
# 2. Freeze near active features or anomalies
# 3. Require minimum observations before updating
# 4. Bound innovation to prevent corruption from outliers
# ============================================================================

using LinearAlgebra
using Statistics: mean

# ============================================================================
# Policy Types
# ============================================================================

"""
    Abstract type for map update policies.
"""
abstract type UpdatePolicy end

"""
    TileUpdateDecision

Decision about whether/how to update a tile.

# Fields
- `should_update::Bool`: Whether to apply any update
- `reason::Symbol`: Reason for decision
- `weight::Float64`: Blending weight (0=freeze, 1=full update)
- `details::Dict{String, Any}`: Additional diagnostic info
"""
struct TileUpdateDecision
    should_update::Bool
    reason::Symbol
    weight::Float64
    details::Dict{String, Any}
end

"""Create a freeze decision."""
freeze_decision(reason::Symbol; details=Dict{String,Any}()) =
    TileUpdateDecision(false, reason, 0.0, details)

"""Create a full update decision."""
update_decision(; details=Dict{String,Any}()) =
    TileUpdateDecision(true, :full_update, 1.0, details)

"""Create a partial update decision."""
partial_update_decision(weight::Float64, reason::Symbol; details=Dict{String,Any}()) =
    TileUpdateDecision(true, reason, weight, details)

# ============================================================================
# Update Context
# ============================================================================

"""
    TileUpdateContext

Context for making update decisions.

# Fields
- `tile_idx::Tuple{Int,Int}`: Tile index
- `n_observations::Int`: Number of observations in this tile
- `nav_confidence::Float64`: Current navigation confidence (0-1)
- `innovation_sigma::Float64`: Innovation in units of σ
- `near_active_feature::Bool`: Whether near an active detected feature
- `current_coefficients::Vector{Float64}`: Current tile coefficients
- `proposed_coefficients::Vector{Float64}`: Proposed new coefficients
- `coefficient_covariance::Matrix{Float64}`: Current coefficient covariance
"""
struct TileUpdateContext
    tile_idx::Tuple{Int,Int}
    n_observations::Int
    nav_confidence::Float64
    innovation_sigma::Float64
    near_active_feature::Bool
    current_coefficients::Vector{Float64}
    proposed_coefficients::Vector{Float64}
    coefficient_covariance::Matrix{Float64}
end

# ============================================================================
# Adaptive Policy
# ============================================================================

"""
    AdaptiveUpdatePolicy <: UpdatePolicy

Adaptive update policy based on multiple criteria.

# Fields
- `min_observations::Int`: Minimum observations before update (default: 10)
- `confidence_threshold::Float64`: Nav confidence required (default: 0.8)
- `innovation_threshold::Float64`: Max allowed innovation in σ (default: 3.0)
- `freeze_on_anomaly::Bool`: Freeze when feature detected (default: true)
- `partial_update_enabled::Bool`: Allow partial updates (default: true)
- `min_partial_weight::Float64`: Minimum weight for partial updates (default: 0.1)
- `max_coefficient_drift::Float64`: Max allowed coefficient change per update (default: 0.5)
- `feature_proximity_distance::Float64`: Distance to freeze near features (default: 10.0)
"""
struct AdaptiveUpdatePolicy <: UpdatePolicy
    min_observations::Int
    confidence_threshold::Float64
    innovation_threshold::Float64
    freeze_on_anomaly::Bool
    partial_update_enabled::Bool
    min_partial_weight::Float64
    max_coefficient_drift::Float64
    feature_proximity_distance::Float64
end

function AdaptiveUpdatePolicy(;
    min_observations::Int = 10,
    confidence_threshold::Float64 = 0.8,
    innovation_threshold::Float64 = 3.0,
    freeze_on_anomaly::Bool = true,
    partial_update_enabled::Bool = true,
    min_partial_weight::Float64 = 0.1,
    max_coefficient_drift::Float64 = 0.5,
    feature_proximity_distance::Float64 = 10.0
)
    AdaptiveUpdatePolicy(
        min_observations, confidence_threshold, innovation_threshold,
        freeze_on_anomaly, partial_update_enabled, min_partial_weight,
        max_coefficient_drift, feature_proximity_distance
    )
end

const DEFAULT_UPDATE_POLICY = AdaptiveUpdatePolicy()

"""
    decide_update(policy::AdaptiveUpdatePolicy, ctx::TileUpdateContext)

Decide whether to update a tile based on the adaptive policy.
"""
function decide_update(policy::AdaptiveUpdatePolicy, ctx::TileUpdateContext)
    details = Dict{String, Any}(
        "tile_idx" => ctx.tile_idx,
        "n_observations" => ctx.n_observations,
        "nav_confidence" => ctx.nav_confidence,
        "innovation_sigma" => ctx.innovation_sigma,
        "near_active_feature" => ctx.near_active_feature
    )

    # Check hard freeze conditions first

    # 1. Freeze if near active feature and policy says so
    if policy.freeze_on_anomaly && ctx.near_active_feature
        return freeze_decision(:near_active_feature; details)
    end

    # 2. Freeze if navigation confidence too low
    if ctx.nav_confidence < policy.confidence_threshold * 0.5
        return freeze_decision(:very_low_confidence; details)
    end

    # 3. Freeze if innovation too high (possible outlier or environment change)
    if ctx.innovation_sigma > 2 * policy.innovation_threshold
        return freeze_decision(:high_innovation; details)
    end

    # 4. Freeze if insufficient observations
    if ctx.n_observations < policy.min_observations ÷ 2
        return freeze_decision(:insufficient_observations; details)
    end

    # Check for partial update conditions

    # 5. Partial update if confidence is moderate
    if ctx.nav_confidence < policy.confidence_threshold
        if !policy.partial_update_enabled
            return freeze_decision(:low_confidence; details)
        end

        # Weight proportional to confidence
        weight = max(
            policy.min_partial_weight,
            (ctx.nav_confidence - policy.confidence_threshold * 0.5) /
            (policy.confidence_threshold * 0.5)
        )
        details["computed_weight"] = weight
        return partial_update_decision(weight, :moderate_confidence; details)
    end

    # 6. Partial update if innovation is elevated
    if ctx.innovation_sigma > policy.innovation_threshold
        if !policy.partial_update_enabled
            return freeze_decision(:elevated_innovation; details)
        end

        # Weight inversely proportional to innovation
        weight = max(
            policy.min_partial_weight,
            1.0 - (ctx.innovation_sigma - policy.innovation_threshold) /
                  policy.innovation_threshold
        )
        details["computed_weight"] = weight
        return partial_update_decision(weight, :elevated_innovation; details)
    end

    # 7. Partial update if observations below minimum
    if ctx.n_observations < policy.min_observations
        if !policy.partial_update_enabled
            return freeze_decision(:few_observations; details)
        end

        weight = max(
            policy.min_partial_weight,
            ctx.n_observations / policy.min_observations
        )
        details["computed_weight"] = weight
        return partial_update_decision(weight, :building_observations; details)
    end

    # 8. Check coefficient drift bound
    if length(ctx.current_coefficients) > 0 && length(ctx.proposed_coefficients) > 0
        drift = norm(ctx.proposed_coefficients - ctx.current_coefficients)
        coeff_norm = max(norm(ctx.current_coefficients), 1e-10)
        relative_drift = drift / coeff_norm

        details["coefficient_drift"] = drift
        details["relative_drift"] = relative_drift

        if relative_drift > policy.max_coefficient_drift
            if !policy.partial_update_enabled
                return freeze_decision(:excessive_drift; details)
            end

            # Clamp update to respect drift bound
            weight = min(1.0, policy.max_coefficient_drift / relative_drift)
            details["computed_weight"] = weight
            return partial_update_decision(weight, :drift_limited; details)
        end
    end

    # All checks passed - full update
    details["computed_weight"] = 1.0
    return update_decision(; details)
end

# ============================================================================
# Conservative Policy
# ============================================================================

"""
    ConservativeUpdatePolicy <: UpdatePolicy

Conservative policy that prioritizes stability over learning speed.
Only updates when conditions are highly favorable.
"""
struct ConservativeUpdatePolicy <: UpdatePolicy
    min_observations::Int
    confidence_threshold::Float64
    innovation_threshold::Float64
end

function ConservativeUpdatePolicy(;
    min_observations::Int = 20,
    confidence_threshold::Float64 = 0.95,
    innovation_threshold::Float64 = 2.0
)
    ConservativeUpdatePolicy(min_observations, confidence_threshold, innovation_threshold)
end

function decide_update(policy::ConservativeUpdatePolicy, ctx::TileUpdateContext)
    details = Dict{String, Any}()

    # Conservative: require all conditions to be met
    if ctx.near_active_feature
        return freeze_decision(:near_feature; details)
    end

    if ctx.nav_confidence < policy.confidence_threshold
        return freeze_decision(:low_confidence; details)
    end

    if ctx.innovation_sigma > policy.innovation_threshold
        return freeze_decision(:high_innovation; details)
    end

    if ctx.n_observations < policy.min_observations
        return freeze_decision(:insufficient_observations; details)
    end

    return update_decision(; details)
end

# ============================================================================
# Aggressive Policy
# ============================================================================

"""
    AggressiveUpdatePolicy <: UpdatePolicy

Aggressive policy that prioritizes learning speed.
Updates whenever minimally safe. Use for initial map learning.
"""
struct AggressiveUpdatePolicy <: UpdatePolicy
    min_observations::Int
    min_confidence::Float64
end

function AggressiveUpdatePolicy(;
    min_observations::Int = 3,
    min_confidence::Float64 = 0.3
)
    AggressiveUpdatePolicy(min_observations, min_confidence)
end

function decide_update(policy::AggressiveUpdatePolicy, ctx::TileUpdateContext)
    details = Dict{String, Any}()

    # Only check minimum requirements
    if ctx.nav_confidence < policy.min_confidence
        return freeze_decision(:very_low_confidence; details)
    end

    if ctx.n_observations < policy.min_observations
        return freeze_decision(:too_few_observations; details)
    end

    return update_decision(; details)
end

# ============================================================================
# Manual Freeze Policy
# ============================================================================

"""
    ManualFreezePolicy <: UpdatePolicy

Policy with explicit freeze list for specific tiles.
Useful for known problematic areas or during testing.
"""
struct ManualFreezePolicy <: UpdatePolicy
    frozen_tiles::Set{Tuple{Int,Int}}
    default_policy::UpdatePolicy
end

function ManualFreezePolicy(frozen_tiles::Vector{Tuple{Int,Int}};
                            default_policy::UpdatePolicy = AdaptiveUpdatePolicy())
    ManualFreezePolicy(Set(frozen_tiles), default_policy)
end

function decide_update(policy::ManualFreezePolicy, ctx::TileUpdateContext)
    if ctx.tile_idx in policy.frozen_tiles
        return freeze_decision(:manually_frozen)
    end
    return decide_update(policy.default_policy, ctx)
end

"""Add a tile to the freeze list."""
function freeze_tile!(policy::ManualFreezePolicy, tile_idx::Tuple{Int,Int})
    push!(policy.frozen_tiles, tile_idx)
end

"""Remove a tile from the freeze list."""
function unfreeze_tile!(policy::ManualFreezePolicy, tile_idx::Tuple{Int,Int})
    delete!(policy.frozen_tiles, tile_idx)
end

# ============================================================================
# Policy Manager
# ============================================================================

"""
    PolicyManager

Manages update decisions across all tiles with statistics tracking.

# Fields
- `policy::UpdatePolicy`: The active update policy
- `decision_history::Vector{TileUpdateDecision}`: Recent decisions
- `history_limit::Int`: Max decisions to keep
- `statistics::Dict{Symbol, Int}`: Counts of decision reasons
"""
mutable struct PolicyManager
    policy::UpdatePolicy
    decision_history::Vector{TileUpdateDecision}
    history_limit::Int
    statistics::Dict{Symbol, Int}
end

function PolicyManager(policy::UpdatePolicy; history_limit::Int = 1000)
    PolicyManager(
        policy,
        TileUpdateDecision[],
        history_limit,
        Dict{Symbol, Int}()
    )
end

"""Make and record an update decision."""
function decide!(manager::PolicyManager, ctx::TileUpdateContext)
    decision = decide_update(manager.policy, ctx)

    # Record decision
    push!(manager.decision_history, decision)
    if length(manager.decision_history) > manager.history_limit
        popfirst!(manager.decision_history)
    end

    # Update statistics
    manager.statistics[decision.reason] =
        get(manager.statistics, decision.reason, 0) + 1

    return decision
end

"""Get summary statistics of decisions."""
function get_policy_statistics(manager::PolicyManager)
    total = sum(values(manager.statistics))
    if total == 0
        return (total=0, update_rate=0.0, freeze_rate=0.0, mean_weight=0.0, reasons=Dict{Symbol,Float64}())
    end

    n_updates = sum(d.should_update for d in manager.decision_history; init=0)

    reason_rates = Dict{Symbol, Float64}()
    for (reason, count) in manager.statistics
        reason_rates[reason] = count / total
    end

    return (
        total = total,
        update_rate = n_updates / length(manager.decision_history),
        freeze_rate = (length(manager.decision_history) - n_updates) / length(manager.decision_history),
        mean_weight = mean(d.weight for d in manager.decision_history),
        reasons = reason_rates
    )
end

"""Reset decision statistics."""
function reset_policy_statistics!(manager::PolicyManager)
    empty!(manager.decision_history)
    empty!(manager.statistics)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    apply_weighted_update(current, proposed, weight)

Apply weighted coefficient update.
"""
function apply_weighted_update(current::Vector{Float64}, proposed::Vector{Float64},
                               weight::Float64)
    return current + weight * (proposed - current)
end

"""
    apply_weighted_covariance_update(current, proposed, weight)

Apply weighted covariance update with guaranteed positive definiteness.
"""
function apply_weighted_covariance_update(current::Matrix{Float64},
                                          proposed::Matrix{Float64},
                                          weight::Float64)
    # Simple linear interpolation
    result = current + weight * (proposed - current)

    # Ensure positive definiteness
    result = 0.5 * (result + result')  # Symmetrize

    # Check eigenvalues and fix if needed
    ev = eigvals(Symmetric(result))
    min_ev = minimum(ev)
    if min_ev < 1e-10
        result += (1e-10 - min_ev) * I(size(result, 1))
    end

    return result
end

"""
    compute_innovation_sigma(residual, covariance)

Compute innovation in units of standard deviation.
"""
function compute_innovation_sigma(residual::Vector{Float64},
                                  covariance::Matrix{Float64})
    # Mahalanobis-like measure
    try
        n = length(residual)
        sigma_sq = residual' * (covariance \ residual)
        return sqrt(max(sigma_sq / n, 0.0))
    catch
        # Fallback to simple norm-based measure
        trace_cov = tr(covariance)
        n = size(covariance, 1)
        avg_var = trace_cov / n
        return norm(residual) / sqrt(max(avg_var, 1e-20))
    end
end

"""
    recommend_policy(mission_number, has_prior_map)

Recommend a policy based on mission context.
"""
function recommend_policy(mission_number::Int, has_prior_map::Bool)
    if !has_prior_map || mission_number <= 2
        # Early missions: learn aggressively
        return AggressiveUpdatePolicy()
    elseif mission_number <= 10
        # Building confidence: balanced approach
        return AdaptiveUpdatePolicy()
    else
        # Mature map: be conservative
        return ConservativeUpdatePolicy()
    end
end

# ============================================================================
# Exports
# ============================================================================

export UpdatePolicy, TileUpdateDecision, TileUpdateContext
export AdaptiveUpdatePolicy, ConservativeUpdatePolicy, AggressiveUpdatePolicy
export ManualFreezePolicy, PolicyManager
export DEFAULT_UPDATE_POLICY
export decide_update, decide!, get_policy_statistics, reset_policy_statistics!
export freeze_decision, update_decision, partial_update_decision
export freeze_tile!, unfreeze_tile!
export apply_weighted_update, apply_weighted_covariance_update
export compute_innovation_sigma, recommend_policy
