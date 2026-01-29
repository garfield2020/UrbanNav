# ============================================================================
# SourceRefinement.jl - Source Refinement (Phase G Step 7)
# ============================================================================
#
# Periodic refinement of source parameter estimates using:
# 1. LM refit (from dipole_fitter.jl)
# 2. Multi-source coordinate descent (from dipole_mle.jl)
# 3. AIC model selection (decide if source is real)
#
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Types
# ============================================================================

"""
    RefinementConfig

Configuration for source refinement.

# Fields
- `refit_interval_obs::Int`: Observations between refits (default 20)
- `multi_source_radius_m::Float64`: Cluster nearby sources for joint fit [m]
- `max_lm_iterations::Int`: Max LM iterations per refit
- `aic_model_selection::Bool`: Use AIC for model selection
- `convergence_tol::Float64`: LM convergence tolerance
"""
struct RefinementConfig
    refit_interval_obs::Int
    multi_source_radius_m::Float64
    max_lm_iterations::Int
    aic_model_selection::Bool
    convergence_tol::Float64

    function RefinementConfig(;
        refit_interval_obs::Int = 20,
        multi_source_radius_m::Float64 = 30.0,
        max_lm_iterations::Int = 50,
        aic_model_selection::Bool = true,
        convergence_tol::Float64 = 1e-8
    )
        @assert refit_interval_obs >= 1
        @assert multi_source_radius_m > 0
        @assert max_lm_iterations >= 1
        @assert convergence_tol > 0
        new(refit_interval_obs, multi_source_radius_m, max_lm_iterations,
            aic_model_selection, convergence_tol)
    end
end

const DEFAULT_REFINEMENT_CONFIG = RefinementConfig()

"""
    RefinementResult

Result of refining a single source.
"""
struct RefinementResult
    source_id::Int
    pre_cost::Float64
    post_cost::Float64
    position_shift_m::Float64
    crlb_improved::Bool
    aic_delta::Float64
end

# ============================================================================
# Refinement
# ============================================================================

"""
    refine_sources!(tracker::SourceTracker, config::RefinementConfig)
        -> Vector{RefinementResult}

Refine active source estimates periodically.

# Logic
For each active track with update_count divisible by refit_interval_obs:
1. Refit using LM fitter
2. Compute AIC for model selection (0 vs 1 source)
3. Demote sources with worsening AIC
"""
function refine_sources!(tracker::SourceTracker, config::RefinementConfig)
    results = RefinementResult[]

    for (id, track) in tracker.tracks
        if track.status in (TRACK_PROVISIONAL, TRACK_CONFIRMED, TRACK_LOCKED) &&
           track.update_count > 0 &&
           mod(track.update_count, config.refit_interval_obs) == 0

            pre_cov_trace = tr(Matrix(track.source_state.covariance))
            pre_pos = Vector(source_position(track.source_state))

            # Attempt refit
            fit = refit_track!(tracker, id)

            if fit !== nothing && fit.converged
                post_cov_trace = tr(Matrix(track.source_state.covariance))
                post_pos = Vector(source_position(track.source_state))
                pos_shift = norm(post_pos - pre_pos)

                # AIC computation (simplified)
                n_obs = length(track.observation_positions)
                k_params = 6  # position + moment
                aic_with_source = 2 * k_params + n_obs * log(fit.final_cost + 1e-30)
                aic_no_source = n_obs * log(fit.initial_cost + 1e-30)
                aic_delta = aic_no_source - aic_with_source  # Positive = source helps

                crlb_improved = post_cov_trace < pre_cov_trace

                push!(results, RefinementResult(
                    id, fit.initial_cost, fit.final_cost,
                    pos_shift, crlb_improved, aic_delta))

                # Demote if AIC worsens
                if config.aic_model_selection && aic_delta < 0
                    track.status = TRACK_DEMOTED
                    track.source_state.lifecycle = :demoted
                    track.coupling_mode = SOURCE_SHADOW
                end
            end
        end
    end

    return results
end

"""
    model_selection_aic(tracks::Dict{Int, SourceTrack},
                        measurements::Vector{SourceObservation})
        -> (keep::Vector{Int}, remove::Vector{Int})

Use AIC to decide which sources to keep vs remove.
"""
function model_selection_aic(tracks::Dict{Int, SourceTrack},
                              measurements::Vector{SourceObservation})
    keep = Int[]
    remove = Int[]

    n = length(measurements)
    if n == 0
        return (keep, remove)
    end

    for (id, track) in tracks
        if track.status in (TRACK_PROVISIONAL, TRACK_CONFIRMED, TRACK_LOCKED)
            # Cost with source
            cost_with = 0.0
            cost_without = 0.0
            for obs in measurements
                r_with = source_residual(obs)
                B_src = dipole_field_at(obs.position, source_position(track.source_state),
                                        source_moment(track.source_state))
                r_with = r_with - Vec3Map(B_src...)
                cost_with += norm(r_with)^2

                r_without = source_residual(obs)
                cost_without += norm(r_without)^2
            end

            aic_with = 2 * 6 + n * log(cost_with / n + 1e-30)
            aic_without = n * log(cost_without / n + 1e-30)

            if aic_with < aic_without
                push!(keep, id)
            else
                push!(remove, id)
            end
        end
    end

    return (keep, remove)
end

# ============================================================================
# Exports
# ============================================================================

export RefinementConfig, DEFAULT_REFINEMENT_CONFIG
export RefinementResult
export refine_sources!, model_selection_aic
