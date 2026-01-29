# ============================================================================
# source_metrics.jl - Source-Centric Metrics (Phase G Step 8)
# ============================================================================
#
# Metrics for evaluating source detection, localization, and classification.
#
# Key metrics:
# - Position error (median, P90)
# - Detection probability (Pd) and false alarm rate (Pfa)
# - NEES consistency
# - CRLB efficiency
# - Time-to-detect and time-to-localize
#
# ============================================================================

using LinearAlgebra
using StaticArrays
using Statistics: mean, median, quantile

# ============================================================================
# Per-Source Metrics
# ============================================================================

"""
    SourceLocalizationMetrics

Metrics for a single source localization.

# Fields
- `position_error_m::Float64`: Euclidean position error [m]
- `moment_error_ratio::Float64`: |m_est - m_true| / |m_true|
- `position_nees::Float64`: Normalized estimation error squared (position block)
- `crlb_efficiency::Float64`: σ_est / σ_crlb (≥ 1.0; 1.0 = optimal)
- `time_to_detect_s::Float64`: Time from source appearance to first detection [s]
- `time_to_localize_s::Float64`: Time from first detection to position_error < threshold [s]
- `coupling_mode_at_end::SourceCouplingMode`: Final coupling mode
"""
struct SourceLocalizationMetrics
    position_error_m::Float64
    moment_error_ratio::Float64
    position_nees::Float64
    crlb_efficiency::Float64
    time_to_detect_s::Float64
    time_to_localize_s::Float64
    coupling_mode_at_end::SourceCouplingMode
end

# ============================================================================
# Scenario Metrics
# ============================================================================

"""
    ScenarioSourceMetrics

Aggregate metrics for a source localization scenario.

# Fields
- `n_targets::Int`: Number of ground truth targets
- `n_detected::Int`: Number detected (Pd = n_detected / n_targets)
- `n_false_alarms::Int`: Tracks with no matching truth
- `n_confirmed::Int`: Tracks reaching CONFIRMED status
- `n_locked::Int`: Tracks reaching LOCKED status
- `median_position_error_m::Float64`: Median position error [m]
- `p90_position_error_m::Float64`: 90th percentile position error [m]
- `median_time_to_localize_s::Float64`: Median time to localize [s]
- `per_source::Vector{SourceLocalizationMetrics}`: Per-source breakdown
"""
struct ScenarioSourceMetrics
    n_targets::Int
    n_detected::Int
    n_false_alarms::Int
    n_confirmed::Int
    n_locked::Int
    median_position_error_m::Float64
    p90_position_error_m::Float64
    median_time_to_localize_s::Float64
    per_source::Vector{SourceLocalizationMetrics}
end

# ============================================================================
# NEES
# ============================================================================

"""
    source_nees(track::SourceTrack, truth_pos::Vec3Map, truth_moment::SVector{3,Float64})
        -> Float64

Compute NEES for a source track against truth.

NEES = (x_est - x_true)' P⁻¹ (x_est - x_true)

For a well-calibrated estimator, E[NEES] = dim (=6).
"""
function source_nees(track::SourceTrack, truth_pos::Vec3Map,
                     truth_moment::SVector{3, Float64})
    x_est = track.source_state.state
    x_true = SVector{6}(vcat(Vector(truth_pos), Vector(truth_moment)))
    err = x_est - x_true
    P = Matrix(track.source_state.covariance) + 1e-15 * I(6)
    return err' * (P \ err)
end

"""Position-only NEES (3-DOF)."""
function source_position_nees(track::SourceTrack, truth_pos::Vec3Map)
    pos_est = source_position(track.source_state)
    err = pos_est - truth_pos
    P_pos = Matrix(track.source_state.covariance)[1:3, 1:3] + 1e-15 * I(3)
    return Vector(err)' * (P_pos \ Vector(err))
end

# ============================================================================
# Evaluation
# ============================================================================

"""
    evaluate_source_metrics(tracks::Dict{Int, SourceTrack},
                             truth::Vector{SourceTruth},
                             association_threshold_m::Float64 = 5.0)
        -> ScenarioSourceMetrics

Evaluate source localization metrics by associating tracks with truth.

# Association
Uses nearest-neighbor assignment with threshold. Each truth source
is matched to at most one track (greedy by distance).
"""
function evaluate_source_metrics(tracks::Dict{Int, SourceTrack},
                                  truth,  # Vector{SourceTruth} or similar
                                  association_threshold_m::Float64 = 5.0)
    # Extract truth positions and moments
    truth_positions = Vec3Map[]
    truth_moments = SVector{3, Float64}[]
    truth_labels = Symbol[]

    for src in truth
        if hasproperty(src, :dipole)
            push!(truth_positions, Vec3Map(src.dipole.position...))
            push!(truth_moments, SVector{3}(src.dipole.moment...))
            push!(truth_labels, src.label)
        end
    end

    n_targets = count(l -> l == :target, truth_labels)
    target_indices = findall(l -> l == :target, truth_labels)

    # Greedy nearest-neighbor association
    active_tracks = [(id, t) for (id, t) in tracks
                     if t.status in (TRACK_PROVISIONAL, TRACK_CONFIRMED, TRACK_LOCKED, TRACK_RETIRED)]

    matched = Dict{Int, Int}()  # track_id → truth_index
    used_truth = Set{Int}()

    for (track_id, track) in sort(active_tracks, by=x->x[1])
        est_pos = source_position(track.source_state)
        best_idx = -1
        best_dist = Inf

        for i in eachindex(truth_positions)
            if i in used_truth
                continue
            end
            d = norm(est_pos - truth_positions[i])
            if d < best_dist && d < association_threshold_m
                best_dist = d
                best_idx = i
            end
        end

        if best_idx > 0
            matched[track_id] = best_idx
            push!(used_truth, best_idx)
        end
    end

    # Compute per-source metrics
    per_source = SourceLocalizationMetrics[]
    detected_targets = 0

    for (track_id, truth_idx) in matched
        track = tracks[track_id]
        truth_pos = truth_positions[truth_idx]
        truth_mom = truth_moments[truth_idx]

        pos_err = norm(source_position(track.source_state) - truth_pos)
        mom_err = norm(source_moment(track.source_state) - truth_mom) /
                  max(norm(truth_mom), 1e-30)

        nees = source_position_nees(track, truth_pos)

        # CRLB efficiency
        pos_std = sqrt(tr(Matrix(track.source_state.covariance)[1:3, 1:3]) / 3)
        crlb_std = track.observability !== nothing ?
                   mean(track.observability.position_crlb) : Inf
        efficiency = crlb_std > 0 ? pos_std / crlb_std : Inf

        if truth_labels[truth_idx] == :target
            detected_targets += 1
        end

        push!(per_source, SourceLocalizationMetrics(
            pos_err, mom_err, nees, max(efficiency, 1.0),
            track.creation_time, 0.0,  # Simplified times
            track.coupling_mode
        ))
    end

    # False alarms: tracks not matched to any truth
    n_false_alarms = length(active_tracks) - length(matched)

    # Aggregate
    pos_errors = [m.position_error_m for m in per_source]
    med_err = isempty(pos_errors) ? NaN : median(pos_errors)
    p90_err = isempty(pos_errors) ? NaN : quantile(pos_errors, 0.9)

    ttl = [m.time_to_localize_s for m in per_source]
    med_ttl = isempty(ttl) ? NaN : median(ttl)

    n_confirmed = count(p -> p.second.status in (TRACK_CONFIRMED, TRACK_LOCKED),
                        collect(tracks))
    n_locked = count(p -> p.second.status == TRACK_LOCKED, collect(tracks))

    return ScenarioSourceMetrics(
        n_targets, detected_targets, n_false_alarms,
        n_confirmed, n_locked,
        med_err, p90_err, med_ttl, per_source
    )
end

# ============================================================================
# Exports
# ============================================================================

# ============================================================================
# Phase G+ Step 8: Conditioned Metrics
# ============================================================================

"""
    ConditionedSourceMetrics

Metrics broken down by SNR regime, coupling mode, and lifecycle stage.

# Fields
- `by_snr_regime`: Metrics per SNR regime (:high, :medium, :low)
- `by_coupling_mode`: Metrics per coupling mode
- `by_lifecycle`: Count of tracks per lifecycle status
- `tile_rmse_with_sources`: Tile RMSE in regions with sources [T]
- `tile_rmse_baseline`: Tile RMSE in source-free regions [T]
- `tile_regression_pct`: Percentage tile degradation due to sources

# SNR Regime Boundaries
- :high   — SNR ≥ 10 at 10m standoff (strong signal, easy detection)
- :medium — 3 ≤ SNR < 10 (detectable, challenging localization)
- :low    — SNR < 3 (near or below detection threshold)

These boundaries correspond to the detection threshold (SNR=3, 3σ)
and comfortable detection (SNR=10, margin for real-world degradation).
"""
struct ConditionedSourceMetrics
    by_snr_regime::Dict{Symbol, ScenarioSourceMetrics}
    by_coupling_mode::Dict{SourceCouplingMode, ScenarioSourceMetrics}
    by_lifecycle::Dict{SourceTrackStatus, Int}
    tile_rmse_with_sources::Float64
    tile_rmse_baseline::Float64
    tile_regression_pct::Float64
end

"""
    snr_regime(snr::Float64; high_threshold::Float64=10.0,
               low_threshold::Float64=3.0) → Symbol

Classify SNR into regime.
- high_threshold: SNR=10 (10× noise floor, comfortable detection margin)
- low_threshold: SNR=3 (classic 3σ detection boundary)
"""
function snr_regime(snr::Float64; high_threshold::Float64 = 10.0,
                    low_threshold::Float64 = 3.0)
    if snr >= high_threshold
        return :high
    elseif snr >= low_threshold
        return :medium
    else
        return :low
    end
end

"""
    evaluate_conditioned_metrics(tracks, truth, tile_states, baseline_rmse;
                                  high_snr=10.0, low_snr=3.0)
        → ConditionedSourceMetrics

Break down metrics by SNR regime, coupling mode, and lifecycle stage.

# Arguments
- `tracks`: Dict{Int, SourceTrack} from tracker
- `truth`: Vector of SourceTruth
- `tile_states`: Optional vector of tile RMSE values near sources (Float64[])
- `baseline_rmse`: Tile RMSE from source-free baseline run [T]
"""
function evaluate_conditioned_metrics(tracks::Dict{Int, SourceTrack},
                                       truth,
                                       tile_rmse_values::Vector{Float64},
                                       baseline_rmse::Float64;
                                       high_snr::Float64 = 10.0,
                                       low_snr::Float64 = 3.0)
    # Partition truth by SNR regime
    regime_truth = Dict{Symbol, Vector}(:high => [], :medium => [], :low => [])
    for src in truth
        if hasproperty(src, :snr_at_10m)
            regime = snr_regime(src.snr_at_10m; high_threshold=high_snr, low_threshold=low_snr)
            push!(regime_truth[regime], src)
        end
    end

    # Evaluate metrics per regime
    by_snr = Dict{Symbol, ScenarioSourceMetrics}()
    for (regime, regime_sources) in regime_truth
        if !isempty(regime_sources)
            by_snr[regime] = evaluate_source_metrics(regime_sources, 5.0)
        else
            by_snr[regime] = ScenarioSourceMetrics(0, 0, 0, 0, 0, 0.0, 0.0, 0.0,
                                                    SourceLocalizationMetrics[])
        end
    end

    # Partition by coupling mode
    by_coupling = Dict{SourceCouplingMode, ScenarioSourceMetrics}()
    for mode in (SOURCE_SHADOW, SOURCE_COV_ONLY, SOURCE_SUBTRACT)
        mode_tracks = Dict{Int, SourceTrack}()
        for (id, track) in tracks
            if track.coupling_mode == mode
                mode_tracks[id] = track
            end
        end
        if !isempty(mode_tracks)
            by_coupling[mode] = evaluate_source_metrics(mode_tracks, truth, 5.0)
        end
    end

    # Count by lifecycle status
    by_lifecycle = Dict{SourceTrackStatus, Int}()
    for status in instances(SourceTrackStatus)
        by_lifecycle[status] = count(t -> t.second.status == status, collect(tracks))
    end

    # Tile regression
    tile_rmse_with = isempty(tile_rmse_values) ? 0.0 : sqrt(mean(v^2 for v in tile_rmse_values))
    tile_regression = baseline_rmse > 0 ?
        100.0 * (tile_rmse_with - baseline_rmse) / baseline_rmse : 0.0

    return ConditionedSourceMetrics(by_snr, by_coupling, by_lifecycle,
                                     tile_rmse_with, baseline_rmse,
                                     max(tile_regression, 0.0))
end

# ============================================================================
# Exports
# ============================================================================

export SourceLocalizationMetrics, ScenarioSourceMetrics
export source_nees, source_position_nees
export evaluate_source_metrics
export ConditionedSourceMetrics, snr_regime, evaluate_conditioned_metrics
