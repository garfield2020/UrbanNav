# ============================================================================
# Feature-Map Absorption
# ============================================================================
#
# Ported from AUV-Navigation/src/feature_absorption.jl
#
# When a feature has been stable for long enough, it should be absorbed
# into the map basis functions rather than remain as a discrete feature.
#
# This provides:
# 1. Gradual blending without discontinuity
# 2. Reduced computational cost
# 3. Long-term map consistency
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Absorption Configuration
# ============================================================================

"""
    FeatureAbsorptionConfig

Configuration for feature-to-map absorption.
"""
struct FeatureAbsorptionConfig
    min_stable_time::Float64           # Time feature must be stable (s)
    max_position_uncertainty::Float64  # Max position std for absorption (m)
    max_moment_uncertainty::Float64    # Max moment std for absorption (A·m²)
    blend_duration::Float64            # Duration of gradual blend (s)
    contribution_stability::Float64    # Max contribution variance for stable
end

function FeatureAbsorptionConfig(;
    min_stable_time::Real = 60.0,
    max_position_uncertainty::Real = 1.0,
    max_moment_uncertainty::Real = 5.0,
    blend_duration::Real = 10.0,
    contribution_stability::Real = 0.1
)
    FeatureAbsorptionConfig(
        Float64(min_stable_time), Float64(max_position_uncertainty),
        Float64(max_moment_uncertainty), Float64(blend_duration),
        Float64(contribution_stability)
    )
end

const DEFAULT_ABSORPTION_CONFIG = FeatureAbsorptionConfig()

# ============================================================================
# Absorption Criteria
# ============================================================================

"""
    FeatureAbsorptionCriteria

Result of checking absorption criteria.
"""
struct FeatureAbsorptionCriteria
    feature_id::Int
    is_stable::Bool
    stability_time::Float64
    position_uncertainty::Float64
    moment_uncertainty::Float64
    contribution_variance::Float64
    ready_for_absorption::Bool
    reason::String
end

"""
    check_feature_absorption_criteria(dipole, lifecycle_state, config, current_time)

Check if a dipole is ready to be absorbed into the map.
"""
function check_feature_absorption_criteria(dipole::DipoleFeatureNode,
                                           lifecycle_state::FeatureLifecycleState,
                                           config::FeatureAbsorptionConfig,
                                           current_time::Float64)
    stability_time = current_time - dipole.created_time

    # Position uncertainty
    pos_var = diag(dipole.covariance)[1:3]
    pos_uncertainty = sqrt(maximum(pos_var))

    # Moment uncertainty
    mom_var = diag(dipole.covariance)[4:6]
    mom_uncertainty = sqrt(maximum(mom_var))

    # Contribution variance
    contribution_variance = 0.0
    if length(lifecycle_state.contribution_history) >= 3
        contributions = lifecycle_state.contribution_history
        mean_contrib = sum(contributions) / length(contributions)
        contribution_variance = sqrt(sum((c - mean_contrib)^2 for c in contributions) / length(contributions))
    end

    # Check criteria
    time_stable = stability_time >= config.min_stable_time
    pos_determined = pos_uncertainty <= config.max_position_uncertainty
    mom_determined = mom_uncertainty <= config.max_moment_uncertainty
    contrib_stable = contribution_variance <= config.contribution_stability

    ready = time_stable && pos_determined && mom_determined && contrib_stable

    reason = if ready
        "Ready for absorption"
    elseif !time_stable
        "Not stable long enough ($(round(stability_time, digits=1))s < $(config.min_stable_time)s)"
    elseif !pos_determined
        "Position uncertainty too high ($(round(pos_uncertainty, digits=2))m > $(config.max_position_uncertainty)m)"
    elseif !mom_determined
        "Moment uncertainty too high ($(round(mom_uncertainty, digits=1)) > $(config.max_moment_uncertainty))"
    else
        "Contribution not stable"
    end

    return FeatureAbsorptionCriteria(
        dipole.id,
        time_stable && pos_determined && mom_determined,
        stability_time, pos_uncertainty, mom_uncertainty, contribution_variance,
        ready, reason
    )
end

# ============================================================================
# Absorption Blending
# ============================================================================

"""
    FeatureAbsorptionState

Tracks the state of an ongoing absorption process.
"""
mutable struct FeatureAbsorptionState
    feature_id::Int
    start_time::Float64
    end_time::Float64
    initial_state::DipoleFeatureState
    blend_factor::Float64  # 0 = full feature, 1 = fully absorbed
    completed::Bool
end

function FeatureAbsorptionState(feature_id::Int, state::DipoleFeatureState,
                                start_time::Float64, duration::Float64)
    FeatureAbsorptionState(
        feature_id, start_time, start_time + duration,
        state, 0.0, false
    )
end

"""
    update_absorption_blend!(absorption, current_time)

Update the blend factor based on current time.
"""
function update_absorption_blend!(absorption::FeatureAbsorptionState, current_time::Float64)
    if current_time >= absorption.end_time
        absorption.blend_factor = 1.0
        absorption.completed = true
    elseif current_time <= absorption.start_time
        absorption.blend_factor = 0.0
    else
        progress = (current_time - absorption.start_time) /
                   (absorption.end_time - absorption.start_time)
        absorption.blend_factor = clamp(progress, 0.0, 1.0)
    end
    return absorption.blend_factor
end

"""
    get_absorption_blended_field(pos, absorption)

Get the field contribution from a feature being absorbed.
During blending: B = (1 - blend_factor) * B_feature
"""
function get_absorption_blended_field(pos::AbstractVector, absorption::FeatureAbsorptionState)
    if absorption.completed
        return zeros(SVector{3})
    end

    B_full = feature_field(pos, absorption.initial_state)
    return (1.0 - absorption.blend_factor) * B_full
end

# ============================================================================
# Absorption Manager
# ============================================================================

"""
    FeatureAbsorptionManager

Manages the absorption of features into the map.
"""
mutable struct FeatureAbsorptionManager
    config::FeatureAbsorptionConfig
    active_absorptions::Dict{Int, FeatureAbsorptionState}
    completed_absorptions::Vector{FeatureAbsorptionState}
    absorption_count::Int
end

function FeatureAbsorptionManager(; config::FeatureAbsorptionConfig = DEFAULT_ABSORPTION_CONFIG)
    FeatureAbsorptionManager(
        config,
        Dict{Int, FeatureAbsorptionState}(),
        FeatureAbsorptionState[],
        0
    )
end

"""
    start_feature_absorption!(mgr, dipole, current_time)

Start absorbing a dipole into the map.
"""
function start_feature_absorption!(mgr::FeatureAbsorptionManager,
                                   dipole::DipoleFeatureNode,
                                   current_time::Float64)
    if haskey(mgr.active_absorptions, dipole.id)
        return mgr.active_absorptions[dipole.id]
    end

    absorption = FeatureAbsorptionState(
        dipole.id, dipole.state, current_time, mgr.config.blend_duration
    )

    mgr.active_absorptions[dipole.id] = absorption
    mgr.absorption_count += 1

    return absorption
end

"""
    update_feature_absorptions!(mgr, current_time)

Update all active absorptions and complete finished ones.
"""
function update_feature_absorptions!(mgr::FeatureAbsorptionManager, current_time::Float64)
    completed_ids = Int[]

    for (id, absorption) in mgr.active_absorptions
        update_absorption_blend!(absorption, current_time)

        if absorption.completed
            push!(completed_ids, id)
            push!(mgr.completed_absorptions, absorption)
        end
    end

    for id in completed_ids
        delete!(mgr.active_absorptions, id)
    end

    return completed_ids
end

"""
    get_total_absorption_field(mgr, pos)

Get total field from all features being absorbed.
"""
function get_total_absorption_field(mgr::FeatureAbsorptionManager, pos::AbstractVector)
    B_total = zeros(SVector{3})

    for (_, absorption) in mgr.active_absorptions
        B_total += get_absorption_blended_field(pos, absorption)
    end

    return B_total
end

# ============================================================================
# Integrated Absorption Pipeline
# ============================================================================

"""
    process_feature_absorptions!(registry, lifecycle_mgr, absorption_mgr, current_time)

Process all dipoles for potential absorption.

Returns (started_ids, completed_ids, criteria_results)
"""
function process_feature_absorptions!(registry::DipoleFeatureRegistry,
                                      lifecycle_mgr::FeatureLifecycleManager,
                                      absorption_mgr::FeatureAbsorptionManager,
                                      current_time::Float64)
    started_ids = Int[]
    criteria_results = FeatureAbsorptionCriteria[]

    # Check each active dipole
    for (id, dipole) in registry.features
        if haskey(absorption_mgr.active_absorptions, id)
            continue
        end

        lifecycle_state = get_feature_lifecycle(lifecycle_mgr, id)
        if isnothing(lifecycle_state)
            continue
        end

        criteria = check_feature_absorption_criteria(
            dipole, lifecycle_state, absorption_mgr.config, current_time
        )
        push!(criteria_results, criteria)

        if criteria.ready_for_absorption
            start_feature_absorption!(absorption_mgr, dipole, current_time)
            push!(started_ids, id)
        end
    end

    # Update ongoing absorptions
    completed_ids = update_feature_absorptions!(absorption_mgr, current_time)

    # Retire completed absorptions
    for id in completed_ids
        dipole = get_dipole_feature(registry, id)
        if !isnothing(dipole)
            dipole.lifecycle = DIPOLE_RETIRED
            push!(registry.retired, dipole)
            delete!(registry.features, id)
        end
    end

    return (started_ids, completed_ids, criteria_results)
end

# ============================================================================
# Exports
# ============================================================================

export FeatureAbsorptionConfig, DEFAULT_ABSORPTION_CONFIG
export FeatureAbsorptionCriteria, check_feature_absorption_criteria
export FeatureAbsorptionState, update_absorption_blend!, get_absorption_blended_field
export FeatureAbsorptionManager
export start_feature_absorption!, update_feature_absorptions!, get_total_absorption_field
export process_feature_absorptions!
