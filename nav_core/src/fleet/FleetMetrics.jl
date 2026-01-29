# ============================================================================
# Fleet Metrics (Phase B)
# ============================================================================
#
# Defines what "success" means for fleet fusion qualification.
#
# Per claude.md: All thresholds must have physical or statistical justification.
#
# Key Metrics:
# 1. Position RMSE [m] - accuracy
# 2. NEES - filter honesty (must be ~1.0)
# 3. NEES consistency - fraction of NEES in acceptable range
# 4. Map quality score - fusion quality
# 5. Improvement ratio - fleet vs solo
#
# ============================================================================

using LinearAlgebra
using Statistics: mean, std, quantile

# ============================================================================
# Per-Vehicle Metrics
# ============================================================================

"""
    VehicleMetrics

Metrics collected for a single vehicle during a mission.

# Fields
- `vehicle_id::Int`: Vehicle identifier
- `position_errors::Vector{Float64}`: Position error at each timestep [m]
- `nees_values::Vector{Float64}`: NEES at each timestep
- `map_residuals::Vector{Float64}`: Map prediction residuals [T]
- `coverage_fraction::Float64`: Fraction of area covered
- `updates_applied::Int`: Number of map updates applied
- `updates_rejected::Int`: Number of map updates rejected
- `fusion_count::Int`: Number of successful fleet fusions
- `fusion_rejected::Int`: Number of rejected fusion attempts

# Derived Metrics (computed on demand)
- RMSE: sqrt(mean(position_errors.^2))
- NEES mean: mean(nees_values)
- NEES consistency: fraction of NEES in [χ²_low, χ²_high]
"""
mutable struct VehicleMetrics
    vehicle_id::Int
    position_errors::Vector{Float64}
    nees_values::Vector{Float64}
    map_residuals::Vector{Float64}
    coverage_fraction::Float64
    updates_applied::Int
    updates_rejected::Int
    fusion_count::Int
    fusion_rejected::Int
    timestamps::Vector{Float64}
end

function VehicleMetrics(vehicle_id::Int)
    VehicleMetrics(
        vehicle_id,
        Float64[],
        Float64[],
        Float64[],
        0.0,
        0, 0, 0, 0,
        Float64[]
    )
end

"""Record a position measurement."""
function record_position!(m::VehicleMetrics, t::Float64, error::Float64, nees::Float64)
    push!(m.timestamps, t)
    push!(m.position_errors, error)
    push!(m.nees_values, nees)
end

"""Record a map residual."""
function record_residual!(m::VehicleMetrics, residual::Float64)
    push!(m.map_residuals, residual)
end

"""Record map update result."""
function record_update!(m::VehicleMetrics, applied::Bool)
    if applied
        m.updates_applied += 1
    else
        m.updates_rejected += 1
    end
end

"""Record fusion result."""
function record_fusion!(m::VehicleMetrics, success::Bool)
    if success
        m.fusion_count += 1
    else
        m.fusion_rejected += 1
    end
end

"""Compute position RMSE [m]."""
function compute_rmse(m::VehicleMetrics)
    if isempty(m.position_errors)
        return NaN
    end
    return sqrt(mean(m.position_errors.^2))
end

"""Compute mean NEES."""
function compute_nees_mean(m::VehicleMetrics)
    if isempty(m.nees_values)
        return NaN
    end
    return mean(m.nees_values)
end

"""
    compute_nees_consistency(m::VehicleMetrics; dof::Int=3, p_low::Float64=0.025, p_high::Float64=0.975)

Compute fraction of NEES values within acceptable chi-square bounds.

# Arguments
- `dof`: Degrees of freedom (3 for position)
- `p_low`: Lower tail probability
- `p_high`: Upper tail probability

# Bounds Justification
For χ²(3):
- p=0.025: χ² = 0.216 (2.5th percentile)
- p=0.975: χ² = 9.348 (97.5th percentile)

A well-calibrated filter should have 95% of NEES values in this range.
We use 70% as minimum acceptable to allow for model mismatch.
"""
function compute_nees_consistency(m::VehicleMetrics; dof::Int=3, p_low::Float64=0.025, p_high::Float64=0.975)
    if isempty(m.nees_values)
        return NaN
    end

    # Chi-square critical values
    # For dof=3: χ²(0.025) ≈ 0.216, χ²(0.975) ≈ 9.348
    # Using approximate values from chi-square table
    chi2_bounds = Dict(
        3 => (0.216, 9.348),   # 95% interval
        8 => (2.180, 17.535)   # For d=8 mode
    )

    bounds = get(chi2_bounds, dof, (0.1, 10.0))  # Fallback
    chi2_low, chi2_high = bounds

    in_bounds = count(x -> chi2_low <= x <= chi2_high, m.nees_values)
    return in_bounds / length(m.nees_values)
end

"""Compute map residual quantiles."""
function compute_residual_quantiles(m::VehicleMetrics)
    if isempty(m.map_residuals)
        return (q50 = NaN, q90 = NaN, q99 = NaN)
    end
    abs_res = abs.(m.map_residuals)
    return (
        q50 = quantile(abs_res, 0.50),
        q90 = quantile(abs_res, 0.90),
        q99 = quantile(abs_res, 0.99)
    )
end

"""Compute update acceptance rate."""
function compute_update_acceptance_rate(m::VehicleMetrics)
    total = m.updates_applied + m.updates_rejected
    if total == 0
        return NaN
    end
    return m.updates_applied / total
end

"""Compute fusion success rate."""
function compute_fusion_success_rate(m::VehicleMetrics)
    total = m.fusion_count + m.fusion_rejected
    if total == 0
        return NaN
    end
    return m.fusion_count / total
end

# ============================================================================
# Fleet Aggregate Metrics
# ============================================================================

"""
    FleetMetrics

Aggregated metrics across all vehicles in fleet.

# Fields
- `vehicle_metrics::Vector{VehicleMetrics}`: Per-vehicle metrics
- `mission_duration::Float64`: Total mission time [s]
- `comms_messages_sent::Int`: Total messages transmitted
- `comms_bytes_sent::Int`: Total bytes transmitted
- `version_conflicts::Int`: Number of version conflicts detected
- `rollbacks_triggered::Int`: Number of rollbacks performed

# Fleet-Level Metrics (computed on demand)
- Mean RMSE across vehicles
- Worst-case RMSE (tail risk)
- Mean NEES consistency
- Fleet improvement vs solo baseline
"""
mutable struct FleetMetrics
    vehicle_metrics::Vector{VehicleMetrics}
    mission_duration::Float64
    comms_messages_sent::Int
    comms_bytes_sent::Int
    version_conflicts::Int
    rollbacks_triggered::Int
    distance_traveled::Float64  # Total fleet distance [m]
end

function FleetMetrics(num_vehicles::Int)
    FleetMetrics(
        [VehicleMetrics(i) for i in 1:num_vehicles],
        0.0,
        0, 0,
        0, 0,
        0.0
    )
end

"""Get metrics for specific vehicle."""
function get_vehicle_metrics(fm::FleetMetrics, vehicle_id::Int)
    return fm.vehicle_metrics[vehicle_id]
end

"""Record communication."""
function record_comms!(fm::FleetMetrics, bytes::Int)
    fm.comms_messages_sent += 1
    fm.comms_bytes_sent += bytes
end

"""Record version conflict."""
function record_conflict!(fm::FleetMetrics)
    fm.version_conflicts += 1
end

"""Record rollback."""
function record_rollback!(fm::FleetMetrics)
    fm.rollbacks_triggered += 1
end

# ============================================================================
# Fleet Aggregate Computations
# ============================================================================

"""Compute mean RMSE across all vehicles [m]."""
function compute_fleet_rmse_mean(fm::FleetMetrics)
    rmses = [compute_rmse(vm) for vm in fm.vehicle_metrics]
    valid = filter(!isnan, rmses)
    if isempty(valid)
        return NaN
    end
    return mean(valid)
end

"""Compute worst-case (max) RMSE across vehicles [m]."""
function compute_fleet_rmse_worst(fm::FleetMetrics)
    rmses = [compute_rmse(vm) for vm in fm.vehicle_metrics]
    valid = filter(!isnan, rmses)
    if isempty(valid)
        return NaN
    end
    return maximum(valid)
end

"""Compute mean NEES across all vehicles."""
function compute_fleet_nees_mean(fm::FleetMetrics)
    nees_means = [compute_nees_mean(vm) for vm in fm.vehicle_metrics]
    valid = filter(!isnan, nees_means)
    if isempty(valid)
        return NaN
    end
    return mean(valid)
end

"""Compute mean NEES consistency across all vehicles."""
function compute_fleet_nees_consistency(fm::FleetMetrics)
    consistencies = [compute_nees_consistency(vm) for vm in fm.vehicle_metrics]
    valid = filter(!isnan, consistencies)
    if isempty(valid)
        return NaN
    end
    return mean(valid)
end

"""Compute percentage of missions with rollback."""
function compute_rollback_rate(fm::FleetMetrics)
    # One mission per vehicle
    n_vehicles = length(fm.vehicle_metrics)
    if n_vehicles == 0
        return NaN
    end
    return fm.rollbacks_triggered / n_vehicles
end

"""Compute mean map quality score after fusion."""
function compute_fleet_map_quality(fm::FleetMetrics)
    fusion_rates = [compute_fusion_success_rate(vm) for vm in fm.vehicle_metrics]
    valid = filter(!isnan, fusion_rates)
    if isempty(valid)
        return NaN
    end
    return mean(valid)
end

"""Compute communication load [messages/km]."""
function compute_comms_per_km(fm::FleetMetrics)
    if fm.distance_traveled <= 0
        return NaN
    end
    return fm.comms_messages_sent / (fm.distance_traveled / 1000.0)
end

"""Compute communication load [bytes/km]."""
function compute_bytes_per_km(fm::FleetMetrics)
    if fm.distance_traveled <= 0
        return NaN
    end
    return fm.comms_bytes_sent / (fm.distance_traveled / 1000.0)
end

# ============================================================================
# Improvement Metrics
# ============================================================================

"""
    FleetImprovementMetrics

Comparison of fleet performance vs solo baseline.

# Fields
- `fleet_rmse::Float64`: Fleet RMSE [m]
- `solo_rmse::Float64`: Best solo vehicle RMSE [m]
- `improvement_fraction::Float64`: (solo - fleet) / solo
- `fleet_nees::Float64`: Fleet mean NEES
- `solo_nees::Float64`: Solo mean NEES
- `honesty_preserved::Bool`: Fleet NEES not worse than solo by >20%
"""
struct FleetImprovementMetrics
    fleet_rmse::Float64
    solo_rmse::Float64
    improvement_fraction::Float64
    fleet_nees::Float64
    solo_nees::Float64
    honesty_preserved::Bool
end

"""
    compute_improvement(fleet_metrics, solo_metrics)

Compare fleet fusion performance against solo baseline.

# Arguments
- `fleet_metrics`: Metrics from fleet fusion run
- `solo_metrics`: Metrics from solo (no fusion) run

# Returns
FleetImprovementMetrics with improvement analysis.

# Honesty Criterion
Fleet is "honest" if mean NEES does not exceed solo NEES by more than 20%.
This allows for some CI conservatism overhead while catching dishonest fusion.

Justification: CI should be conservative, so NEES should not be worse.
20% margin allows for statistical variation and omega optimization effects.
"""
function compute_improvement(fleet_metrics::FleetMetrics, solo_metrics::FleetMetrics)
    fleet_rmse = compute_fleet_rmse_mean(fleet_metrics)
    solo_rmse = compute_fleet_rmse_mean(solo_metrics)

    improvement = if solo_rmse > 0 && !isnan(solo_rmse) && !isnan(fleet_rmse)
        (solo_rmse - fleet_rmse) / solo_rmse
    else
        NaN
    end

    fleet_nees = compute_fleet_nees_mean(fleet_metrics)
    solo_nees = compute_fleet_nees_mean(solo_metrics)

    # Honesty: fleet NEES should not be worse than solo by >20%
    honesty = if !isnan(fleet_nees) && !isnan(solo_nees)
        fleet_nees <= solo_nees * 1.20
    else
        false
    end

    FleetImprovementMetrics(
        fleet_rmse,
        solo_rmse,
        improvement,
        fleet_nees,
        solo_nees,
        honesty
    )
end

# ============================================================================
# Pass/Fail Criteria
# ============================================================================

"""
    FleetQualificationResult

Result of fleet qualification test.

# Fields
- `passed::Bool`: Overall pass/fail
- `rmse_passed::Bool`: RMSE within threshold
- `nees_passed::Bool`: NEES within acceptable range
- `consistency_passed::Bool`: NEES consistency above threshold
- `improvement_passed::Bool`: Improvement vs solo above threshold
- `honesty_passed::Bool`: No dishonest covariance shrinkage
- `no_corruption::Bool`: No version conflicts or rollbacks
- `details::Dict{Symbol, Any}`: Detailed results

# Pass Criteria Justification

**max_rmse = 5.0 m**:
Phase A qualified at <5m RMSE. Fleet should not degrade this.

**min_nees_consistency = 0.70**:
95% of NEES should be in χ²(3) 95% interval for perfect filter.
70% allows for model mismatch while catching systematic issues.

**max_nees = 3.0**:
Mean NEES should be ~1.0. Upper bound of 3.0 catches overconfidence.
χ²(3) has E[X]=3, so NEES=3 means 3× expected variance.

**min_improvement = 0.0**:
Fleet should not be worse than solo. Some scenarios may have 0 expected
improvement (e.g., version conflicts, quality gating).
"""
struct FleetQualificationResult
    passed::Bool
    rmse_passed::Bool
    nees_passed::Bool
    consistency_passed::Bool
    improvement_passed::Bool
    honesty_passed::Bool
    no_corruption::Bool
    details::Dict{Symbol, Any}
end

"""
    evaluate_qualification(fleet_metrics, solo_metrics, scenario)

Evaluate fleet against qualification criteria.

# Arguments
- `fleet_metrics`: Metrics from fleet fusion run
- `solo_metrics`: Metrics from solo (no fusion) run
- `scenario`: Scenario specification with pass criteria

# Returns
FleetQualificationResult with pass/fail determination.
"""
function evaluate_qualification(
    fleet_metrics::FleetMetrics,
    solo_metrics::FleetMetrics;
    max_rmse::Float64 = 5.0,
    min_nees_consistency::Float64 = 0.70,
    max_nees::Float64 = 3.0,
    min_improvement::Float64 = 0.0
)
    # Compute metrics
    fleet_rmse = compute_fleet_rmse_mean(fleet_metrics)
    fleet_nees = compute_fleet_nees_mean(fleet_metrics)
    fleet_consistency = compute_fleet_nees_consistency(fleet_metrics)

    improvement = compute_improvement(fleet_metrics, solo_metrics)

    # Evaluate criteria
    rmse_passed = !isnan(fleet_rmse) && fleet_rmse <= max_rmse
    nees_passed = !isnan(fleet_nees) && fleet_nees <= max_nees
    consistency_passed = !isnan(fleet_consistency) && fleet_consistency >= min_nees_consistency
    improvement_passed = !isnan(improvement.improvement_fraction) &&
                         improvement.improvement_fraction >= min_improvement
    honesty_passed = improvement.honesty_preserved

    # No corruption: no rollbacks and no unresolved version conflicts
    no_corruption = fleet_metrics.rollbacks_triggered == 0 &&
                    fleet_metrics.version_conflicts == 0

    # Overall pass
    passed = rmse_passed && nees_passed && consistency_passed &&
             improvement_passed && honesty_passed && no_corruption

    details = Dict{Symbol, Any}(
        :fleet_rmse => fleet_rmse,
        :fleet_nees => fleet_nees,
        :fleet_consistency => fleet_consistency,
        :solo_rmse => improvement.solo_rmse,
        :improvement_fraction => improvement.improvement_fraction,
        :rollbacks => fleet_metrics.rollbacks_triggered,
        :version_conflicts => fleet_metrics.version_conflicts,
        :comms_per_km => compute_comms_per_km(fleet_metrics)
    )

    FleetQualificationResult(
        passed,
        rmse_passed,
        nees_passed,
        consistency_passed,
        improvement_passed,
        honesty_passed,
        no_corruption,
        details
    )
end

# ============================================================================
# Formatting
# ============================================================================

"""Format vehicle metrics for display."""
function format_vehicle_metrics(m::VehicleMetrics)
    lines = String[]
    push!(lines, "Vehicle $(m.vehicle_id) Metrics")
    push!(lines, "-" ^ 40)
    push!(lines, "Position RMSE: $(round(compute_rmse(m), digits=3)) m")
    push!(lines, "NEES mean: $(round(compute_nees_mean(m), digits=3))")
    push!(lines, "NEES consistency: $(round(compute_nees_consistency(m) * 100, digits=1))%")
    push!(lines, "Updates: $(m.updates_applied) applied, $(m.updates_rejected) rejected")
    push!(lines, "Fusions: $(m.fusion_count) success, $(m.fusion_rejected) rejected")
    push!(lines, "Coverage: $(round(m.coverage_fraction * 100, digits=1))%")
    return join(lines, "\n")
end

"""Format fleet metrics for display."""
function format_fleet_metrics(fm::FleetMetrics)
    lines = String[]
    push!(lines, "Fleet Metrics Summary")
    push!(lines, "=" ^ 50)
    push!(lines, "Vehicles: $(length(fm.vehicle_metrics))")
    push!(lines, "Duration: $(round(fm.mission_duration, digits=1)) s")
    push!(lines, "Distance: $(round(fm.distance_traveled, digits=1)) m")
    push!(lines, "")
    push!(lines, "Accuracy:")
    push!(lines, "  Mean RMSE: $(round(compute_fleet_rmse_mean(fm), digits=3)) m")
    push!(lines, "  Worst RMSE: $(round(compute_fleet_rmse_worst(fm), digits=3)) m")
    push!(lines, "")
    push!(lines, "Consistency:")
    push!(lines, "  Mean NEES: $(round(compute_fleet_nees_mean(fm), digits=3))")
    push!(lines, "  NEES consistency: $(round(compute_fleet_nees_consistency(fm) * 100, digits=1))%")
    push!(lines, "")
    push!(lines, "Communication:")
    push!(lines, "  Messages: $(fm.comms_messages_sent)")
    push!(lines, "  Bytes: $(fm.comms_bytes_sent)")
    push!(lines, "  Messages/km: $(round(compute_comms_per_km(fm), digits=1))")
    push!(lines, "")
    push!(lines, "Issues:")
    push!(lines, "  Version conflicts: $(fm.version_conflicts)")
    push!(lines, "  Rollbacks: $(fm.rollbacks_triggered)")
    return join(lines, "\n")
end

"""Format qualification result for display."""
function format_qualification_result(r::FleetQualificationResult)
    lines = String[]
    push!(lines, "Fleet Qualification Result")
    push!(lines, "=" ^ 50)
    status = r.passed ? "PASSED" : "FAILED"
    push!(lines, "Overall: $status")
    push!(lines, "")
    push!(lines, "Criteria:")
    push!(lines, "  [$(r.rmse_passed ? "✓" : "✗")] RMSE: $(round(r.details[:fleet_rmse], digits=3)) m")
    push!(lines, "  [$(r.nees_passed ? "✓" : "✗")] NEES: $(round(r.details[:fleet_nees], digits=3))")
    push!(lines, "  [$(r.consistency_passed ? "✓" : "✗")] Consistency: $(round(r.details[:fleet_consistency] * 100, digits=1))%")
    push!(lines, "  [$(r.improvement_passed ? "✓" : "✗")] Improvement: $(round(r.details[:improvement_fraction] * 100, digits=1))%")
    push!(lines, "  [$(r.honesty_passed ? "✓" : "✗")] Honesty preserved")
    push!(lines, "  [$(r.no_corruption ? "✓" : "✗")] No corruption")
    return join(lines, "\n")
end

# ============================================================================
# Exports
# ============================================================================

export VehicleMetrics, record_position!, record_residual!, record_update!, record_fusion!
export compute_rmse, compute_nees_mean, compute_nees_consistency
export compute_residual_quantiles, compute_update_acceptance_rate, compute_fusion_success_rate
export FleetMetrics, get_vehicle_metrics, record_comms!, record_conflict!, record_rollback!
export compute_fleet_rmse_mean, compute_fleet_rmse_worst
export compute_fleet_nees_mean, compute_fleet_nees_consistency
export compute_rollback_rate, compute_fleet_map_quality
export compute_comms_per_km, compute_bytes_per_km
export FleetImprovementMetrics, compute_improvement
export FleetQualificationResult, evaluate_qualification
export format_vehicle_metrics, format_fleet_metrics, format_qualification_result

