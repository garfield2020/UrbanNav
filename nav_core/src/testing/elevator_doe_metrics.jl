# ============================================================================
# elevator_doe_metrics.jl - Elevator-specific DOE metrics
# ============================================================================
#
# Computes position error quantiles, do-no-harm ratios, innovation burst
# metrics, map contamination scores, and segment-scoped analysis for
# elevator DOE testing.
# ============================================================================

export ElevatorDOEMetrics, SegmentMetrics
export compute_elevator_metrics, compute_segment_metrics
export compute_do_no_harm, compute_map_contamination, compute_innovation_burst

using LinearAlgebra
using Statistics

# ============================================================================
# Elevator DOE Metrics
# ============================================================================

"""
    ElevatorDOEMetrics

Position error and diagnostic metrics for an elevator DOE run.

# Fields
- `rmse::Float64`: Root mean square position error (m).
- `p50_error::Float64`: Median position error (m).
- `p90_error::Float64`: 90th-percentile position error (m).
- `max_error::Float64`: Maximum position error (m).
- `do_no_harm_ratio::Float64`: P90(with elevator) / P90(without). ≤1.10 required.
- `innovation_burst_peak::Float64`: Peak normalized innovation during elevator motion.
- `map_contamination_score::Float64`: Correlation between tile updates and elevator position.
- `false_source_count_per_km::Float64`: Spurious source detections per km walked.
"""
struct ElevatorDOEMetrics
    rmse::Float64
    p50_error::Float64
    p90_error::Float64
    max_error::Float64
    do_no_harm_ratio::Float64
    innovation_burst_peak::Float64
    map_contamination_score::Float64
    false_source_count_per_km::Float64
end

"""
    SegmentMetrics

ElevatorDOEMetrics broken down by proximity/motion segment.

# Fields
- `near_shaft::ElevatorDOEMetrics`: Metrics when pedestrian is <5m from shaft.
- `during_motion::ElevatorDOEMetrics`: Metrics when elevator velocity > 0.
- `during_stops::ElevatorDOEMetrics`: Metrics when elevator is stopped but recently moved.
- `full_mission::ElevatorDOEMetrics`: Metrics over the entire mission.
"""
struct SegmentMetrics
    near_shaft::ElevatorDOEMetrics
    during_motion::ElevatorDOEMetrics
    during_stops::ElevatorDOEMetrics
    full_mission::ElevatorDOEMetrics
end

# ============================================================================
# Metric Computation
# ============================================================================

"""
    _quantile_sorted(sorted_vals, q) -> Float64

Compute quantile from pre-sorted values.
"""
function _quantile_sorted(sorted_vals::Vector{Float64}, q::Float64)
    n = length(sorted_vals)
    n == 0 && return 0.0
    n == 1 && return sorted_vals[1]
    idx = q * (n - 1) + 1.0
    lo = floor(Int, idx)
    hi = ceil(Int, idx)
    lo = clamp(lo, 1, n)
    hi = clamp(hi, 1, n)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac
end

"""
    compute_elevator_metrics(position_errors, innovations, tile_updates,
                             elevator_positions, path_length;
                             do_no_harm_ratio, false_source_count) -> ElevatorDOEMetrics

Compute elevator-specific metrics from raw data.

# Arguments
- `position_errors::Vector{Float64}`: Position error at each timestep (m).
- `innovations::Vector{Float64}`: Normalized innovation magnitudes.
- `tile_updates::Vector{Float64}`: Tile coefficient change magnitudes per step.
- `elevator_positions::Vector{<:AbstractVector}`: Elevator 3D positions per step.
- `path_length::Float64`: Total path length walked (m).
- `do_no_harm_ratio::Float64`: Pre-computed P90 ratio.
- `false_source_count::Int`: Number of false source detections.
"""
function compute_elevator_metrics(
    position_errors::Vector{Float64},
    innovations::Vector{Float64},
    tile_updates::Vector{Float64},
    elevator_positions::Vector,
    path_length::Float64;
    do_no_harm_ratio::Float64 = 1.0,
    false_source_count::Int = 0,
)
    sorted_errors = sort(position_errors)
    rmse = isempty(position_errors) ? 0.0 : sqrt(mean(position_errors .^ 2))
    p50 = _quantile_sorted(sorted_errors, 0.5)
    p90 = _quantile_sorted(sorted_errors, 0.9)
    max_err = isempty(sorted_errors) ? 0.0 : sorted_errors[end]

    burst_peak = isempty(innovations) ? 0.0 : maximum(innovations)
    contamination = compute_map_contamination(tile_updates, elevator_positions)
    false_per_km = path_length > 0 ? false_source_count / (path_length / 1000.0) : 0.0

    return ElevatorDOEMetrics(
        rmse, p50, p90, max_err, do_no_harm_ratio,
        burst_peak, contamination, false_per_km,
    )
end

"""
    compute_segment_metrics(position_errors, innovations, tile_updates,
                            elevator_positions, pedestrian_positions,
                            elevator_velocities, timestamps, path_length;
                            do_no_harm_ratio, false_source_count,
                            near_shaft_threshold) -> SegmentMetrics

Compute metrics segmented by shaft proximity and elevator motion state.
"""
function compute_segment_metrics(
    position_errors::Vector{Float64},
    innovations::Vector{Float64},
    tile_updates::Vector{Float64},
    elevator_positions::Vector,
    pedestrian_positions::Vector,
    elevator_velocities::Vector{Float64},
    timestamps::Vector{Float64},
    path_length::Float64;
    do_no_harm_ratio::Float64 = 1.0,
    false_source_count::Int = 0,
    near_shaft_threshold::Float64 = 5.0,
    recent_motion_window::Float64 = 30.0,
)
    n = length(position_errors)

    # Classify each timestep
    near_mask = falses(n)
    motion_mask = falses(n)
    stop_mask = falses(n)

    for i in 1:n
        # Near shaft: horizontal distance < threshold
        if !isempty(elevator_positions) && i <= length(elevator_positions)
            ped_xy = pedestrian_positions[i][1:2]
            elev_xy = elevator_positions[i][1:2]
            dist = sqrt(sum((ped_xy .- elev_xy).^2))
            near_mask[i] = dist < near_shaft_threshold
        end

        # Motion state
        if i <= length(elevator_velocities)
            if abs(elevator_velocities[i]) > 0.01
                motion_mask[i] = true
            else
                # Check if elevator recently moved
                t_now = timestamps[i]
                recently_moved = false
                for j in max(1, i-1):-1:1
                    if t_now - timestamps[j] > recent_motion_window
                        break
                    end
                    if abs(elevator_velocities[j]) > 0.01
                        recently_moved = true
                        break
                    end
                end
                stop_mask[i] = recently_moved
            end
        end
    end

    _make = (mask) -> begin
        idx = findall(mask)
        if isempty(idx)
            return ElevatorDOEMetrics(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        end
        pe = position_errors[idx]
        inn = !isempty(innovations) ? innovations[clamp.(idx, 1, length(innovations))] : Float64[]
        tu = !isempty(tile_updates) ? tile_updates[clamp.(idx, 1, length(tile_updates))] : Float64[]
        ep = !isempty(elevator_positions) ? elevator_positions[clamp.(idx, 1, length(elevator_positions))] : []
        compute_elevator_metrics(pe, inn, tu, ep, path_length;
                                 do_no_harm_ratio=do_no_harm_ratio,
                                 false_source_count=0)
    end

    full = compute_elevator_metrics(
        position_errors, innovations, tile_updates, elevator_positions, path_length;
        do_no_harm_ratio=do_no_harm_ratio, false_source_count=false_source_count,
    )

    return SegmentMetrics(
        _make(near_mask),
        _make(motion_mask),
        _make(stop_mask),
        full,
    )
end

# ============================================================================
# Do-No-Harm Ratio
# ============================================================================

"""
    compute_do_no_harm(errors_with::Vector{Float64}, errors_without::Vector{Float64}) -> Float64

Compute the do-no-harm ratio: P90(with elevator) / P90(without elevator).
Must be ≤ 1.10 to pass.
"""
function compute_do_no_harm(errors_with::Vector{Float64}, errors_without::Vector{Float64})
    isempty(errors_with) && return 1.0
    isempty(errors_without) && return 1.0

    p90_with = _quantile_sorted(sort(errors_with), 0.9)
    p90_without = _quantile_sorted(sort(errors_without), 0.9)

    p90_without < 1e-10 && return 1.0
    return p90_with / p90_without
end

# ============================================================================
# Map Contamination
# ============================================================================

"""
    compute_map_contamination(tile_updates, elevator_positions) -> Float64

Compute correlation between tile update magnitudes and inverse distance to
elevator. High values indicate the map is being contaminated by elevator field.
Returns a value in [0, 1]. Near-zero is desired.
"""
function compute_map_contamination(
    tile_updates::Vector{Float64},
    elevator_positions::Vector,
)
    n = min(length(tile_updates), length(elevator_positions))
    n < 3 && return 0.0

    # Use tile update magnitude as one variable
    tu = tile_updates[1:n]

    # Use proximity signal (1/distance) as other variable
    # Use norm of elevator position change as proxy for motion correlation
    prox = Float64[]
    for i in 1:n
        ep = elevator_positions[i]
        push!(prox, 1.0 / (norm(ep) + 1.0))
    end

    mu_tu = mean(tu)
    mu_prox = mean(prox)
    std_tu = std(tu; corrected=false)
    std_prox = std(prox; corrected=false)

    if std_tu < 1e-12 || std_prox < 1e-12
        return 0.0
    end

    corr = mean((tu .- mu_tu) .* (prox .- mu_prox)) / (std_tu * std_prox)
    return clamp(abs(corr), 0.0, 1.0)
end

# ============================================================================
# Innovation Burst
# ============================================================================

"""
    compute_innovation_burst(innovations, threshold) -> (peak, containment_ratio)

Compute the peak normalized innovation and the fraction of innovations
that remain below the threshold.

# Returns
- `peak::Float64`: Maximum normalized innovation.
- `containment_ratio::Float64`: Fraction of innovations below threshold.
"""
function compute_innovation_burst(innovations::Vector{Float64}, threshold::Float64 = 3.0)
    isempty(innovations) && return (0.0, 1.0)
    peak = maximum(innovations)
    contained = count(x -> x <= threshold, innovations) / length(innovations)
    return (peak, contained)
end
