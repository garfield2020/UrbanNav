# ============================================================================
# Manifold Collapse Metrics (Phase B)
# ============================================================================
#
# Defines "manifold collapse" as a measurable phenomenon.
#
# Purpose: Prove "Mission 2 is better than Mission 1" with rigorous metrics.
#
# What is Manifold Collapse?
# --------------------------
# The "manifold" is the space of possible map coefficients α ∈ ℝⁿ.
# Initially, our uncertainty spans a large region (high covariance P_α).
# As observations accumulate, the posterior "collapses" onto a smaller region.
#
# Mathematically:
#   Prior:     P(α) ~ N(α₀, P₀)         [large uncertainty]
#   Posterior: P(α|z₁...zₖ) ~ N(α̂, P̂)  [smaller uncertainty]
#
# Collapse is healthy when:
#   1. Covariance reduces: tr(P̂) < tr(P₀)
#   2. NEES remains calibrated: E[NEES] ≈ 1
#   3. Reduction is information-theoretic: ΔH = ½ log|P₀/P̂| > 0
#
# Key Metrics:
# - Uncertainty Volume: det(P)^(1/2n) [geometric mean of eigenvalues]
# - Information Gain: log|P_prior| - log|P_posterior| [bits]
# - Effective DOF: number of eigenvalues above threshold
# - Convergence Rate: d(tr(P))/d(observations)
#
# ============================================================================

using LinearAlgebra
using Statistics: mean, std

# ============================================================================
# Core Metric Types
# ============================================================================

"""
    CollapseSnapshot

Snapshot of map state at a point in learning trajectory.

# Fields
- `mission_id::String`: Which mission this snapshot is from
- `observation_count::Int`: Total observations incorporated
- `timestamp::Float64`: Time within mission [s]
- `covariance_trace::Float64`: tr(P_α) [varies by unit]
- `covariance_det::Float64`: det(P_α) (volume measure)
- `eigenvalues::Vector{Float64}`: Eigenvalues of P_α (sorted descending)
- `rmse::Float64`: Position RMSE at this point [m]
- `nees::Float64`: NEES at this point (should be ≈1)
- `position_error::Vector{Float64}`: Position error [m] (for detailed analysis)
"""
struct CollapseSnapshot
    mission_id::String
    observation_count::Int
    timestamp::Float64
    covariance_trace::Float64
    covariance_det::Float64
    eigenvalues::Vector{Float64}
    rmse::Float64
    nees::Float64
    position_error::Vector{Float64}
end

"""Create snapshot from current map state and navigation error."""
function CollapseSnapshot(
    mission_id::String,
    observation_count::Int,
    timestamp::Float64,
    P_map::AbstractMatrix,
    position_error::AbstractVector,
    P_position::AbstractMatrix
)
    # Eigenvalue decomposition (sorted descending)
    eigs = eigvals(Hermitian(P_map))
    eigs_sorted = sort(real.(eigs), rev=true)

    # NEES: e' P⁻¹ e where e is position error
    # Handle near-singular covariance
    P_reg = P_position + 1e-20 * I(size(P_position, 1))
    nees = dot(position_error, P_reg \ position_error)

    CollapseSnapshot(
        mission_id,
        observation_count,
        timestamp,
        tr(P_map),
        det(P_map),
        eigs_sorted,
        norm(position_error),
        nees,
        Vector(position_error)
    )
end

"""
    CollapseTrajectory

Complete learning trajectory across one or more missions.

# Fields
- `snapshots::Vector{CollapseSnapshot}`: Time-ordered snapshots
- `mission_sequence::Vector{String}`: Order of missions
- `baseline_trace::Float64`: Initial covariance trace (for normalization)
- `baseline_det::Float64`: Initial covariance determinant
"""
mutable struct CollapseTrajectory
    snapshots::Vector{CollapseSnapshot}
    mission_sequence::Vector{String}
    baseline_trace::Float64
    baseline_det::Float64
end

"""Create empty trajectory."""
function CollapseTrajectory()
    CollapseTrajectory(CollapseSnapshot[], String[], 0.0, 0.0)
end

"""Create trajectory with initial baseline."""
function CollapseTrajectory(P_initial::AbstractMatrix)
    CollapseTrajectory(
        CollapseSnapshot[],
        String[],
        tr(P_initial),
        det(P_initial)
    )
end

"""Add snapshot to trajectory."""
function add_snapshot!(traj::CollapseTrajectory, snap::CollapseSnapshot)
    push!(traj.snapshots, snap)
    if !(snap.mission_id in traj.mission_sequence)
        push!(traj.mission_sequence, snap.mission_id)
    end

    # Update baseline if this is the first snapshot
    if length(traj.snapshots) == 1
        traj.baseline_trace = snap.covariance_trace
        traj.baseline_det = snap.covariance_det
    end
end

# ============================================================================
# Collapse Metrics
# ============================================================================

"""
    CollapseMetrics

Summary metrics for manifold collapse analysis.

# Fields
- `total_observations::Int`: Total observations across all missions
- `n_missions::Int`: Number of missions
- `trace_reduction_pct::Float64`: (baseline - final) / baseline × 100
- `volume_reduction_pct::Float64`: (baseline_det^(1/n) - final_det^(1/n)) / baseline^(1/n) × 100
- `information_gain_nats::Float64`: ½ × (log|P₀| - log|P_final|)
- `effective_dof::Int`: Number of well-constrained dimensions
- `mean_nees::Float64`: Mean NEES across trajectory (should be ≈1)
- `nees_consistency::Float64`: Fraction of NEES in [0.1, 10]
- `final_rmse::Float64`: Final position RMSE [m]
- `convergence_rate::Float64`: Average trace reduction per 100 observations

# Health Thresholds
- trace_reduction_pct > 0: Learning is happening
- trace_reduction_pct > 50: Significant learning
- mean_nees ∈ [0.5, 2.0]: Well-calibrated filter
- nees_consistency > 0.8: Consistent behavior
"""
struct CollapseMetrics
    total_observations::Int
    n_missions::Int
    trace_reduction_pct::Float64
    volume_reduction_pct::Float64
    information_gain_nats::Float64
    effective_dof::Int
    mean_nees::Float64
    nees_consistency::Float64
    final_rmse::Float64
    convergence_rate::Float64
end

"""
    compute_collapse_metrics(traj::CollapseTrajectory;
                             dof_threshold::Float64 = 0.01)

Compute summary metrics from collapse trajectory.

# Arguments
- `traj`: Collapse trajectory with snapshots
- `dof_threshold`: Eigenvalue threshold for counting effective DOF
                   (fraction of largest eigenvalue)

# Returns
CollapseMetrics summary
"""
function compute_collapse_metrics(traj::CollapseTrajectory;
                                  dof_threshold::Float64 = 0.01)
    if isempty(traj.snapshots)
        return CollapseMetrics(0, 0, 0.0, 0.0, 0.0, 0, 1.0, 1.0, 0.0, 0.0)
    end

    first_snap = traj.snapshots[1]
    last_snap = traj.snapshots[end]

    # Total observations
    total_obs = last_snap.observation_count

    # Number of missions
    n_missions = length(traj.mission_sequence)

    # Trace reduction
    baseline_trace = traj.baseline_trace > 0 ? traj.baseline_trace : first_snap.covariance_trace
    trace_reduction = (baseline_trace - last_snap.covariance_trace) / baseline_trace * 100

    # Volume reduction (geometric mean of eigenvalues)
    n_dim = length(first_snap.eigenvalues)
    if n_dim > 0 && traj.baseline_det > 0 && last_snap.covariance_det > 0
        baseline_vol = traj.baseline_det^(1/n_dim)
        final_vol = last_snap.covariance_det^(1/n_dim)
        volume_reduction = (baseline_vol - final_vol) / baseline_vol * 100
    else
        volume_reduction = 0.0
    end

    # Information gain (in nats, not bits)
    # ΔI = ½ × (log|P_prior| - log|P_posterior|)
    if traj.baseline_det > 0 && last_snap.covariance_det > 0
        info_gain = 0.5 * (log(traj.baseline_det) - log(last_snap.covariance_det))
    else
        info_gain = 0.0
    end

    # Effective degrees of freedom
    if !isempty(last_snap.eigenvalues)
        max_eig = maximum(last_snap.eigenvalues)
        threshold = dof_threshold * max_eig
        effective_dof = count(e -> e > threshold, last_snap.eigenvalues)
    else
        effective_dof = 0
    end

    # NEES statistics
    nees_values = [s.nees for s in traj.snapshots]
    mean_nees = mean(nees_values)
    nees_consistency = mean(0.1 .< nees_values .< 10.0)

    # Final RMSE
    final_rmse = last_snap.rmse

    # Convergence rate: trace reduction per 100 observations
    if total_obs > 0
        convergence_rate = (baseline_trace - last_snap.covariance_trace) / (total_obs / 100)
    else
        convergence_rate = 0.0
    end

    CollapseMetrics(
        total_obs,
        n_missions,
        trace_reduction,
        volume_reduction,
        info_gain,
        effective_dof,
        mean_nees,
        nees_consistency,
        final_rmse,
        convergence_rate
    )
end

# ============================================================================
# Convergence Criteria
# ============================================================================

"""
    ConvergenceCriteria

Criteria for determining when map learning has converged.

# Fields
- `max_trace::Float64`: Maximum acceptable covariance trace
- `min_trace_reduction_pct::Float64`: Minimum required trace reduction (%)
- `max_eigenvalue_ratio::Float64`: Maximum ratio of largest to smallest eigenvalue
- `nees_min::Float64`: Minimum acceptable mean NEES
- `nees_max::Float64`: Maximum acceptable mean NEES
- `max_rmse::Float64`: Maximum acceptable position RMSE [m]

# Physical Justification

**max_trace = 1e-14 T²** (default):
For 8 map coefficients with σ_B = 10 nT = 10e-9 T:
  σ² = (10e-9)² = 1e-16 T²
  tr(P) = 8 × 1e-16 = 8e-16 T²
A threshold of 1e-14 allows ~12× this noise floor.

**min_trace_reduction_pct = 50%**:
Learning should at least halve the prior uncertainty.
If this isn't achieved, either:
- Prior was already tight (Phase A map was good)
- Observations aren't informative (weak gradient)
- Learning is misconfigured

**max_eigenvalue_ratio = 100**:
Well-conditioned covariance should have bounded condition number.
Ratio > 100 indicates near-singular directions.

**NEES bounds [0.5, 2.0]**:
Tighter than validation bounds [0.1, 5.0] because converged
filter should be well-calibrated.

**max_rmse = 5.0 m**:
Operational requirement for most AUV missions.
"""
struct ConvergenceCriteria
    max_trace::Float64
    min_trace_reduction_pct::Float64
    max_eigenvalue_ratio::Float64
    nees_min::Float64
    nees_max::Float64
    max_rmse::Float64

    function ConvergenceCriteria(;
        max_trace::Float64 = 1e-14,
        min_trace_reduction_pct::Float64 = 50.0,
        max_eigenvalue_ratio::Float64 = 100.0,
        nees_min::Float64 = 0.5,
        nees_max::Float64 = 2.0,
        max_rmse::Float64 = 5.0
    )
        @assert max_trace > 0
        @assert 0 <= min_trace_reduction_pct <= 100
        @assert max_eigenvalue_ratio > 1
        @assert 0 < nees_min < nees_max
        @assert max_rmse > 0
        new(max_trace, min_trace_reduction_pct, max_eigenvalue_ratio,
            nees_min, nees_max, max_rmse)
    end
end

const DEFAULT_CONVERGENCE_CRITERIA = ConvergenceCriteria()

"""
    ConvergenceResult

Result of checking convergence criteria.

# Fields
- `converged::Bool`: Whether all criteria are met
- `criteria_met::Dict{Symbol, Bool}`: Which specific criteria passed
- `metrics::CollapseMetrics`: The metrics that were checked
- `failure_reasons::Vector{Symbol}`: Why convergence wasn't achieved
"""
struct ConvergenceResult
    converged::Bool
    criteria_met::Dict{Symbol, Bool}
    metrics::CollapseMetrics
    failure_reasons::Vector{Symbol}
end

"""
    check_convergence(traj::CollapseTrajectory, criteria::ConvergenceCriteria)

Check if collapse trajectory meets convergence criteria.

Returns ConvergenceResult with detailed pass/fail information.
"""
function check_convergence(traj::CollapseTrajectory,
                          criteria::ConvergenceCriteria = DEFAULT_CONVERGENCE_CRITERIA)
    metrics = compute_collapse_metrics(traj)

    criteria_met = Dict{Symbol, Bool}()
    failure_reasons = Symbol[]

    # Check each criterion
    last_snap = isempty(traj.snapshots) ? nothing : traj.snapshots[end]

    # 1. Maximum trace
    if last_snap !== nothing
        trace_ok = last_snap.covariance_trace < criteria.max_trace
        criteria_met[:max_trace] = trace_ok
        if !trace_ok
            push!(failure_reasons, :trace_too_high)
        end
    else
        criteria_met[:max_trace] = false
        push!(failure_reasons, :no_data)
    end

    # 2. Minimum trace reduction
    reduction_ok = metrics.trace_reduction_pct >= criteria.min_trace_reduction_pct
    criteria_met[:trace_reduction] = reduction_ok
    if !reduction_ok
        push!(failure_reasons, :insufficient_reduction)
    end

    # 3. Eigenvalue ratio (condition number)
    if last_snap !== nothing && !isempty(last_snap.eigenvalues)
        max_eig = maximum(last_snap.eigenvalues)
        min_eig = max(minimum(last_snap.eigenvalues), 1e-30)  # Avoid division by zero
        ratio = max_eig / min_eig
        ratio_ok = ratio < criteria.max_eigenvalue_ratio
        criteria_met[:eigenvalue_ratio] = ratio_ok
        if !ratio_ok
            push!(failure_reasons, :ill_conditioned)
        end
    else
        criteria_met[:eigenvalue_ratio] = false
        push!(failure_reasons, :no_eigenvalues)
    end

    # 4. NEES bounds
    nees_ok = criteria.nees_min <= metrics.mean_nees <= criteria.nees_max
    criteria_met[:nees_bounds] = nees_ok
    if !nees_ok
        if metrics.mean_nees < criteria.nees_min
            push!(failure_reasons, :nees_too_low)
        else
            push!(failure_reasons, :nees_too_high)
        end
    end

    # 5. Maximum RMSE
    rmse_ok = metrics.final_rmse < criteria.max_rmse
    criteria_met[:max_rmse] = rmse_ok
    if !rmse_ok
        push!(failure_reasons, :rmse_too_high)
    end

    converged = all(values(criteria_met))

    ConvergenceResult(converged, criteria_met, metrics, failure_reasons)
end

# ============================================================================
# Mission Comparison
# ============================================================================

"""
    MissionComparison

Compare two missions on the same map.

# Fields
- `mission_1::String`: First mission ID (typically survey)
- `mission_2::String`: Second mission ID (re-traversal)
- `rmse_1::Float64`: RMSE for mission 1 [m]
- `rmse_2::Float64`: RMSE for mission 2 [m]
- `nees_1::Float64`: Mean NEES for mission 1
- `nees_2::Float64`: Mean NEES for mission 2
- `improvement_pct::Float64`: RMSE improvement (positive = better)
- `mission_2_better::Bool`: Whether mission 2 has lower RMSE

# Interpretation
- improvement_pct > 0: Second mission is better
- improvement_pct > 20: Significant improvement
- improvement_pct > 50: Major improvement
"""
struct MissionComparison
    mission_1::String
    mission_2::String
    rmse_1::Float64
    rmse_2::Float64
    nees_1::Float64
    nees_2::Float64
    improvement_pct::Float64
    mission_2_better::Bool
end

"""
    compare_missions(traj::CollapseTrajectory, mission_1::String, mission_2::String)

Compare performance between two missions.

Returns MissionComparison if both missions found, nothing otherwise.
"""
function compare_missions(traj::CollapseTrajectory, mission_1::String, mission_2::String)
    # Find snapshots for each mission
    snaps_1 = filter(s -> s.mission_id == mission_1, traj.snapshots)
    snaps_2 = filter(s -> s.mission_id == mission_2, traj.snapshots)

    if isempty(snaps_1) || isempty(snaps_2)
        return nothing
    end

    # Compute mean RMSE and NEES for each mission
    rmse_1 = mean([s.rmse for s in snaps_1])
    rmse_2 = mean([s.rmse for s in snaps_2])
    nees_1 = mean([s.nees for s in snaps_1])
    nees_2 = mean([s.nees for s in snaps_2])

    # Compute improvement
    improvement_pct = (rmse_1 - rmse_2) / rmse_1 * 100

    MissionComparison(
        mission_1, mission_2,
        rmse_1, rmse_2,
        nees_1, nees_2,
        improvement_pct,
        rmse_2 < rmse_1
    )
end

"""
    prove_mission_improvement(traj::CollapseTrajectory;
                              min_improvement_pct::Float64 = 10.0,
                              max_nees_degradation::Float64 = 1.0)

Prove that later missions perform better than earlier ones.

# Arguments
- `traj`: Collapse trajectory spanning multiple missions
- `min_improvement_pct`: Minimum required RMSE improvement (default: 10%)
- `max_nees_degradation`: Maximum allowed NEES increase (default: 1.0)

# Returns
Named tuple with:
- `proven::Bool`: Whether improvement is proven
- `comparisons::Vector{MissionComparison}`: Pairwise comparisons
- `summary::String`: Human-readable summary
"""
function prove_mission_improvement(traj::CollapseTrajectory;
                                   min_improvement_pct::Float64 = 10.0,
                                   max_nees_degradation::Float64 = 1.0)
    missions = traj.mission_sequence
    comparisons = MissionComparison[]

    if length(missions) < 2
        return (
            proven = false,
            comparisons = comparisons,
            summary = "Need at least 2 missions to prove improvement"
        )
    end

    # Compare consecutive missions
    for i in 1:(length(missions)-1)
        comp = compare_missions(traj, missions[i], missions[i+1])
        if comp !== nothing
            push!(comparisons, comp)
        end
    end

    if isempty(comparisons)
        return (
            proven = false,
            comparisons = comparisons,
            summary = "No valid mission comparisons found"
        )
    end

    # Check improvement criteria
    all_improved = all(c -> c.improvement_pct >= min_improvement_pct, comparisons)
    nees_stable = all(c -> c.nees_2 - c.nees_1 <= max_nees_degradation, comparisons)

    proven = all_improved && nees_stable

    # Build summary
    lines = String[]
    push!(lines, "Mission Improvement Analysis")
    push!(lines, "=" ^ 40)

    for comp in comparisons
        status = comp.mission_2_better ? "✓" : "✗"
        push!(lines, "$status $(comp.mission_1) → $(comp.mission_2)")
        push!(lines, "  RMSE: $(round(comp.rmse_1, digits=2))m → $(round(comp.rmse_2, digits=2))m ($(round(comp.improvement_pct, digits=1))%)")
        push!(lines, "  NEES: $(round(comp.nees_1, digits=2)) → $(round(comp.nees_2, digits=2))")
    end

    push!(lines, "")
    if proven
        push!(lines, "✓ PROVEN: Later missions perform better")
    else
        push!(lines, "✗ NOT PROVEN:")
        if !all_improved
            push!(lines, "  - Some missions did not improve by ≥$(min_improvement_pct)%")
        end
        if !nees_stable
            push!(lines, "  - NEES degraded by more than $(max_nees_degradation)")
        end
    end

    return (
        proven = proven,
        comparisons = comparisons,
        summary = join(lines, "\n")
    )
end

# ============================================================================
# Acceptance Curve Data
# ============================================================================

"""
    AcceptanceCurve

Data for plotting acceptance curves.

# Fields
- `observation_counts::Vector{Int}`: X-axis: observation count
- `trace_values::Vector{Float64}`: Covariance trace over time
- `rmse_values::Vector{Float64}`: RMSE over time [m]
- `nees_values::Vector{Float64}`: NEES over time
- `mission_boundaries::Vector{Int}`: Observation counts at mission transitions

# Usage
This struct provides the raw data needed for external plotting.
Generate curves with `extract_acceptance_curves()`.
"""
struct AcceptanceCurve
    observation_counts::Vector{Int}
    trace_values::Vector{Float64}
    rmse_values::Vector{Float64}
    nees_values::Vector{Float64}
    mission_boundaries::Vector{Int}
end

"""
    extract_acceptance_curves(traj::CollapseTrajectory)

Extract data for acceptance curve plotting.
"""
function extract_acceptance_curves(traj::CollapseTrajectory)
    obs_counts = [s.observation_count for s in traj.snapshots]
    traces = [s.covariance_trace for s in traj.snapshots]
    rmses = [s.rmse for s in traj.snapshots]
    neess = [s.nees for s in traj.snapshots]

    # Find mission boundaries
    boundaries = Int[]
    current_mission = ""
    for (i, snap) in enumerate(traj.snapshots)
        if snap.mission_id != current_mission
            push!(boundaries, snap.observation_count)
            current_mission = snap.mission_id
        end
    end

    AcceptanceCurve(obs_counts, traces, rmses, neess, boundaries)
end

"""
    normalized_acceptance_curve(traj::CollapseTrajectory)

Extract acceptance curve with normalized values (0-1 scale).

Useful for comparing across different scenarios.
"""
function normalized_acceptance_curve(traj::CollapseTrajectory)
    curves = extract_acceptance_curves(traj)

    # Normalize to [0, 1]
    max_trace = maximum(curves.trace_values)
    max_rmse = maximum(curves.rmse_values)

    normalized_traces = max_trace > 0 ? curves.trace_values ./ max_trace : curves.trace_values
    normalized_rmses = max_rmse > 0 ? curves.rmse_values ./ max_rmse : curves.rmse_values

    AcceptanceCurve(
        curves.observation_counts,
        normalized_traces,
        normalized_rmses,
        curves.nees_values,  # NEES is already normalized around 1
        curves.mission_boundaries
    )
end

# ============================================================================
# Formatting and Display
# ============================================================================

"""Format collapse metrics as human-readable string."""
function format_collapse_metrics(metrics::CollapseMetrics)
    lines = String[]
    push!(lines, "Manifold Collapse Metrics")
    push!(lines, "=" ^ 40)
    push!(lines, "Observations: $(metrics.total_observations) across $(metrics.n_missions) mission(s)")
    push!(lines, "")
    push!(lines, "Uncertainty Reduction:")
    push!(lines, "  Trace reduction: $(round(metrics.trace_reduction_pct, digits=1))%")
    push!(lines, "  Volume reduction: $(round(metrics.volume_reduction_pct, digits=1))%")
    push!(lines, "  Information gain: $(round(metrics.information_gain_nats, digits=2)) nats")
    push!(lines, "  Effective DOF: $(metrics.effective_dof)")
    push!(lines, "")
    push!(lines, "Filter Health:")
    push!(lines, "  Mean NEES: $(round(metrics.mean_nees, digits=2)) (target: 1.0)")
    push!(lines, "  NEES consistency: $(round(metrics.nees_consistency * 100, digits=1))%")
    push!(lines, "  Final RMSE: $(round(metrics.final_rmse, digits=2)) m")
    push!(lines, "")
    push!(lines, "Convergence rate: $(round(metrics.convergence_rate, digits=4)) per 100 obs")

    return join(lines, "\n")
end

"""Format convergence result as human-readable string."""
function format_convergence_result(result::ConvergenceResult)
    lines = String[]
    status = result.converged ? "✓ CONVERGED" : "✗ NOT CONVERGED"
    push!(lines, "Convergence Check: $status")
    push!(lines, "-" ^ 40)

    for (criterion, passed) in result.criteria_met
        mark = passed ? "✓" : "✗"
        push!(lines, "  $mark $criterion")
    end

    if !result.converged
        push!(lines, "")
        push!(lines, "Failure reasons:")
        for reason in result.failure_reasons
            push!(lines, "  - $reason")
        end
    end

    return join(lines, "\n")
end

# ============================================================================
# Exports
# ============================================================================

export CollapseSnapshot, CollapseTrajectory
export add_snapshot!
export CollapseMetrics, compute_collapse_metrics
export ConvergenceCriteria, DEFAULT_CONVERGENCE_CRITERIA
export ConvergenceResult, check_convergence
export MissionComparison, compare_missions, prove_mission_improvement
export AcceptanceCurve, extract_acceptance_curves, normalized_acceptance_curve
export format_collapse_metrics, format_convergence_result
