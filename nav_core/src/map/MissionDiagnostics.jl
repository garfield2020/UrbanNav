# ============================================================================
# MissionDiagnostics.jl - Time-Series Learning Diagnostics
# ============================================================================
#
# Provides time-resolved diagnostics to answer:
# "Why didn't it learn more?" without archaeology.
#
# Tracks per-timestep:
# - Update attempts and outcomes
# - Rejection reasons (teachability, observability, quality)
# - Observability metrics (min singular value, gradient norm)
# - Uncertainty metrics (Σ_map, innovation covariance)
# - Residual statistics
#
# Output formats:
# - Compact JSON for programmatic analysis
# - CSV for spreadsheet/plotting tools
#
# ============================================================================

using Dates
using Statistics: mean, std, quantile
using LinearAlgebra
using JSON3

# ============================================================================
# Rejection Reasons (enumerated for analysis)
# ============================================================================

"""
    UpdateRejectionReason

Why a learning update was rejected.

# Values
- `ACCEPTED`: Update was accepted
- `LOW_TEACHABILITY`: Tile uncertainty too low (nothing to learn)
- `LOW_OBSERVABILITY`: Gradient too weak (can't constrain position)
- `HIGH_INNOVATION`: Innovation too large (outlier/model mismatch)
- `NEES_VIOLATION`: Would violate NEES bounds
- `QUALITY_GATE`: Failed quality gate (chi-square test)
- `COVARIANCE_SINGULAR`: Covariance matrix ill-conditioned
- `OUTSIDE_COVERAGE`: Position outside map coverage
"""
@enum UpdateRejectionReason begin
    ACCEPTED = 0
    LOW_TEACHABILITY = 1
    LOW_OBSERVABILITY = 2
    HIGH_INNOVATION = 3
    NEES_VIOLATION = 4
    QUALITY_GATE = 5
    COVARIANCE_SINGULAR = 6
    OUTSIDE_COVERAGE = 7
end

# ============================================================================
# Per-Timestep Record
# ============================================================================

"""
    DiagnosticRecord

Single timestep diagnostic record.

All quantities in SI units (meters, Tesla, seconds).
"""
struct DiagnosticRecord
    # Time
    t::Float64                      # Mission time [s]

    # Position
    position_est::Vector{Float64}   # Estimated position [m]
    position_error::Float64         # True position error [m] (if truth available)

    # Update outcome
    update_attempted::Bool
    update_accepted::Bool
    rejection_reason::UpdateRejectionReason

    # Observability metrics
    gradient_norm::Float64          # |G| [T/m]
    min_singular_value::Float64     # σ_min of H [T/m]
    condition_number::Float64       # κ(H) = σ_max/σ_min

    # Innovation statistics
    innovation_norm::Float64        # |z - h(x)| [T]
    innovation_normalized::Float64  # Mahalanobis distance (χ² statistic)

    # Uncertainty metrics
    sigma_map::Float64              # √tr(Σ_map) [T]
    sigma_position::Float64         # √tr(P_pos) [m]
    nees::Float64                   # Normalized estimation error squared

    # Learning metrics (if update attempted)
    teachability::Float64           # How much can we learn? [0-1]
    information_gain::Float64       # Δlog|P| from update
end

"""Create a diagnostic record."""
function DiagnosticRecord(;
    t::Float64,
    position_est::Vector{Float64},
    position_error::Float64 = NaN,
    update_attempted::Bool = false,
    update_accepted::Bool = false,
    rejection_reason::UpdateRejectionReason = ACCEPTED,
    gradient_norm::Float64 = 0.0,
    min_singular_value::Float64 = 0.0,
    condition_number::Float64 = Inf,
    innovation_norm::Float64 = 0.0,
    innovation_normalized::Float64 = 0.0,
    sigma_map::Float64 = 0.0,
    sigma_position::Float64 = 0.0,
    nees::Float64 = 0.0,
    teachability::Float64 = 0.0,
    information_gain::Float64 = 0.0
)
    DiagnosticRecord(
        t, position_est, position_error,
        update_attempted, update_accepted, rejection_reason,
        gradient_norm, min_singular_value, condition_number,
        innovation_norm, innovation_normalized,
        sigma_map, sigma_position, nees,
        teachability, information_gain
    )
end

# ============================================================================
# Mission Diagnostic Log
# ============================================================================

"""
    MissionDiagnosticLog

Complete diagnostic log for a mission.
"""
mutable struct MissionDiagnosticLog
    # Metadata
    mission_id::String
    world_type::Symbol
    seed::Int
    start_time::DateTime

    # Time-series records
    records::Vector{DiagnosticRecord}

    # Aggregated statistics (computed on finalize)
    finalized::Bool
    summary::Union{Nothing, Dict{String, Any}}
end

"""Create a new diagnostic log."""
function MissionDiagnosticLog(mission_id::String; world_type::Symbol = :unknown, seed::Int = 0)
    MissionDiagnosticLog(
        mission_id,
        world_type,
        seed,
        now(UTC),
        DiagnosticRecord[],
        false,
        nothing
    )
end

"""Add a diagnostic record to the log."""
function record!(log::MissionDiagnosticLog, rec::DiagnosticRecord)
    @assert !log.finalized "Cannot add records to finalized log"
    push!(log.records, rec)
end

"""Add a diagnostic record using keyword arguments."""
function record!(log::MissionDiagnosticLog; kwargs...)
    record!(log, DiagnosticRecord(; kwargs...))
end

# ============================================================================
# Summary Statistics
# ============================================================================

"""
    compute_summary(log::MissionDiagnosticLog) -> Dict

Compute summary statistics from diagnostic records.
"""
function compute_summary(log::MissionDiagnosticLog)
    records = log.records
    n = length(records)

    if n == 0
        return Dict{String, Any}()
    end

    # Update statistics
    n_attempted = count(r -> r.update_attempted, records)
    n_accepted = count(r -> r.update_accepted, records)

    # Rejection breakdown
    rejection_counts = Dict{UpdateRejectionReason, Int}()
    for r in records
        if r.update_attempted && !r.update_accepted
            rejection_counts[r.rejection_reason] = get(rejection_counts, r.rejection_reason, 0) + 1
        end
    end

    # Position error statistics (if truth available)
    errors = [r.position_error for r in records if !isnan(r.position_error)]
    error_stats = if !isempty(errors)
        Dict(
            "rmse_m" => sqrt(mean(errors.^2)),
            "mean_m" => mean(errors),
            "std_m" => std(errors),
            "p50_m" => quantile(errors, 0.50),
            "p90_m" => quantile(errors, 0.90),
            "p99_m" => quantile(errors, 0.99)
        )
    else
        Dict{String, Float64}()
    end

    # NEES statistics
    nees_values = [r.nees for r in records if r.nees > 0]
    nees_stats = if !isempty(nees_values)
        Dict(
            "mean" => mean(nees_values),
            "std" => std(nees_values),
            "p50" => quantile(nees_values, 0.50),
            "p90" => quantile(nees_values, 0.90),
            "fraction_in_bounds" => mean(0.1 .< nees_values .< 10.0)
        )
    else
        Dict{String, Float64}()
    end

    # Observability statistics
    grad_norms = [r.gradient_norm for r in records if r.gradient_norm > 0]
    obs_stats = if !isempty(grad_norms)
        Dict(
            "gradient_norm_mean_nT_m" => mean(grad_norms) * 1e9,
            "gradient_norm_min_nT_m" => minimum(grad_norms) * 1e9,
            "min_sv_mean_nT_m" => mean(r.min_singular_value for r in records) * 1e9,
            "condition_mean" => mean(r.condition_number for r in records if isfinite(r.condition_number))
        )
    else
        Dict{String, Float64}()
    end

    # Innovation statistics
    innov_norms = [r.innovation_norm for r in records if r.innovation_norm > 0]
    innov_stats = if !isempty(innov_norms)
        Dict(
            "mean_nT" => mean(innov_norms) * 1e9,
            "std_nT" => std(innov_norms) * 1e9,
            "p90_nT" => quantile(innov_norms, 0.90) * 1e9,
            "normalized_mean" => mean(r.innovation_normalized for r in records if r.innovation_normalized > 0)
        )
    else
        Dict{String, Float64}()
    end

    # Teachability statistics
    teach_values = [r.teachability for r in records if r.teachability > 0]
    teach_stats = if !isempty(teach_values)
        Dict(
            "mean" => mean(teach_values),
            "min" => minimum(teach_values),
            "max" => maximum(teach_values)
        )
    else
        Dict{String, Float64}()
    end

    # Dominant rejection reason
    dominant_rejection = if !isempty(rejection_counts)
        argmax(rejection_counts)
    else
        nothing
    end

    return Dict{String, Any}(
        "mission_id" => log.mission_id,
        "world_type" => string(log.world_type),
        "seed" => log.seed,
        "n_timesteps" => n,
        "duration_s" => records[end].t - records[1].t,
        "updates" => Dict(
            "attempted" => n_attempted,
            "accepted" => n_accepted,
            "acceptance_rate" => n_attempted > 0 ? n_accepted / n_attempted : 0.0,
            "rejection_breakdown" => Dict(string(k) => v for (k, v) in rejection_counts)
        ),
        "dominant_rejection" => dominant_rejection === nothing ? "none" : string(dominant_rejection),
        "position_error" => error_stats,
        "nees" => nees_stats,
        "observability" => obs_stats,
        "innovation" => innov_stats,
        "teachability" => teach_stats
    )
end

"""Finalize the log and compute summary."""
function finalize!(log::MissionDiagnosticLog)
    if !log.finalized
        log.summary = compute_summary(log)
        log.finalized = true
    end
    return log.summary
end

# ============================================================================
# Export Formats
# ============================================================================

"""
    to_csv(log::MissionDiagnosticLog, path::String)

Export diagnostic log to CSV for spreadsheet/plotting tools.
"""
function to_csv(log::MissionDiagnosticLog, path::String)
    open(path, "w") do io
        # Header
        println(io, "t_s,pos_x_m,pos_y_m,pos_z_m,pos_error_m,update_attempted,update_accepted,rejection_reason,gradient_norm_nT_m,min_sv_nT_m,condition,innovation_nT,innovation_chi2,sigma_map_nT,sigma_pos_m,nees,teachability,info_gain")

        # Records
        for r in log.records
            println(io, join([
                r.t,
                r.position_est[1], r.position_est[2], r.position_est[3],
                isnan(r.position_error) ? "" : r.position_error,
                r.update_attempted ? 1 : 0,
                r.update_accepted ? 1 : 0,
                Int(r.rejection_reason),
                r.gradient_norm * 1e9,
                r.min_singular_value * 1e9,
                isfinite(r.condition_number) ? r.condition_number : "",
                r.innovation_norm * 1e9,
                r.innovation_normalized,
                r.sigma_map * 1e9,
                r.sigma_position,
                r.nees,
                r.teachability,
                r.information_gain
            ], ","))
        end
    end
    return path
end

"""
    to_json(log::MissionDiagnosticLog, path::String; include_records::Bool = false)

Export diagnostic log to JSON.

If include_records is false, only exports summary (more compact).
"""
function to_json(log::MissionDiagnosticLog, path::String; include_records::Bool = false)
    !log.finalized && finalize!(log)

    data = copy(log.summary)
    data["start_time"] = string(log.start_time)

    if include_records
        data["records"] = [
            Dict(
                "t" => r.t,
                "position_est" => r.position_est,
                "position_error" => r.position_error,
                "update_attempted" => r.update_attempted,
                "update_accepted" => r.update_accepted,
                "rejection_reason" => string(r.rejection_reason),
                "gradient_norm" => r.gradient_norm,
                "min_singular_value" => r.min_singular_value,
                "condition_number" => r.condition_number,
                "innovation_norm" => r.innovation_norm,
                "innovation_normalized" => r.innovation_normalized,
                "sigma_map" => r.sigma_map,
                "sigma_position" => r.sigma_position,
                "nees" => r.nees,
                "teachability" => r.teachability,
                "information_gain" => r.information_gain
            )
            for r in log.records
        ]
    end

    open(path, "w") do io
        JSON3.write(io, data)
    end
    return path
end

"""
    print_summary(log::MissionDiagnosticLog)

Print human-readable summary to stdout.
"""
function print_summary(log::MissionDiagnosticLog)
    !log.finalized && finalize!(log)
    s = log.summary

    println("=" ^ 60)
    println("MISSION DIAGNOSTIC SUMMARY: $(s["mission_id"])")
    println("=" ^ 60)
    println("World: $(s["world_type"]), Seed: $(s["seed"])")
    println("Duration: $(round(s["duration_s"], digits=1))s, Timesteps: $(s["n_timesteps"])")
    println()

    # Updates
    u = s["updates"]
    println("Learning Updates:")
    println("  Attempted: $(u["attempted"])")
    println("  Accepted:  $(u["accepted"]) ($(round(u["acceptance_rate"] * 100, digits=1))%)")
    if !isempty(u["rejection_breakdown"])
        println("  Rejections:")
        for (reason, count) in u["rejection_breakdown"]
            println("    $reason: $count")
        end
    end
    println("  Dominant rejection: $(s["dominant_rejection"])")
    println()

    # Position error
    if !isempty(s["position_error"])
        pe = s["position_error"]
        println("Position Error:")
        println("  RMSE: $(round(pe["rmse_m"], digits=2)) m")
        println("  P50:  $(round(pe["p50_m"], digits=2)) m")
        println("  P90:  $(round(pe["p90_m"], digits=2)) m")
        println()
    end

    # NEES
    if !isempty(s["nees"])
        n = s["nees"]
        println("NEES (filter consistency):")
        println("  Mean: $(round(n["mean"], digits=2))")
        println("  P90:  $(round(n["p90"], digits=2))")
        println("  In bounds [0.1, 10]: $(round(n["fraction_in_bounds"] * 100, digits=1))%")
        println()
    end

    # Observability
    if !isempty(s["observability"])
        o = s["observability"]
        println("Observability:")
        println("  Gradient norm: $(round(o["gradient_norm_mean_nT_m"], digits=1)) nT/m (mean)")
        println("  Min singular value: $(round(o["min_sv_mean_nT_m"], digits=1)) nT/m (mean)")
        println()
    end

    # Innovation
    if !isempty(s["innovation"])
        i = s["innovation"]
        println("Innovation:")
        println("  Mean: $(round(i["mean_nT"], digits=1)) nT")
        println("  P90:  $(round(i["p90_nT"], digits=1)) nT")
        println("  χ² mean: $(round(i["normalized_mean"], digits=2))")
        println()
    end

    println("=" ^ 60)
end

# ============================================================================
# Bits-Per-Meter Analysis
# ============================================================================

"""
    BitsPerMeterAnalysis

Information efficiency metrics for mission.

# Fields
- `total_distance_m::Float64`: Total distance traveled [m]
- `total_information_bits::Float64`: Total information gained [bits]
- `bits_per_meter::Float64`: Information efficiency [bits/m]
- `bits_per_second::Float64`: Information rate [bits/s]
- `effective_updates::Int`: Updates that contributed information
- `info_per_update_bits::Float64`: Average information per accepted update [bits]
"""
struct BitsPerMeterAnalysis
    total_distance_m::Float64
    total_information_bits::Float64
    bits_per_meter::Float64
    bits_per_second::Float64
    effective_updates::Int
    info_per_update_bits::Float64
end

"""
    compute_bits_per_meter(log::MissionDiagnosticLog) -> BitsPerMeterAnalysis

Compute information efficiency from diagnostic log.

Information gain is measured in bits via:
    bits = -0.5 * log2(|P_post|/|P_prior|)

For Kalman updates, information_gain field stores Δlog|P|,
so bits = -0.5 * log2(exp(info_gain)) = -0.5 * info_gain / log(2)
"""
function compute_bits_per_meter(log::MissionDiagnosticLog)
    records = log.records
    n = length(records)

    if n < 2
        return BitsPerMeterAnalysis(0.0, 0.0, 0.0, 0.0, 0, 0.0)
    end

    # Compute total distance traveled
    total_distance = 0.0
    for i in 2:n
        dx = records[i].position_est - records[i-1].position_est
        total_distance += norm(dx)
    end

    # Compute total information gain in bits
    # info_gain stores Δlog|P| (negative when uncertainty decreases)
    # bits = -0.5 * Δlog|P| / log(2) = -Δlog|P| / (2*log(2))
    LOG2 = Base.log(2.0)
    total_info_bits = 0.0
    effective_updates = 0

    for r in records
        if r.update_accepted && r.information_gain < 0
            # Negative info_gain means uncertainty decreased
            bits = -r.information_gain / (2 * LOG2)
            total_info_bits += bits
            effective_updates += 1
        end
    end

    # Duration
    duration = records[end].t - records[1].t

    # Compute rates
    bits_per_meter = total_distance > 0 ? total_info_bits / total_distance : 0.0
    bits_per_second = duration > 0 ? total_info_bits / duration : 0.0
    info_per_update = effective_updates > 0 ? total_info_bits / effective_updates : 0.0

    return BitsPerMeterAnalysis(
        total_distance,
        total_info_bits,
        bits_per_meter,
        bits_per_second,
        effective_updates,
        info_per_update
    )
end

"""
    compute_bits_per_meter(positions::Vector{Vector{Float64}},
                           info_gains::Vector{Float64}) -> BitsPerMeterAnalysis

Compute bits-per-meter from position and information gain arrays.
"""
function compute_bits_per_meter(positions::Vector{<:AbstractVector},
                                 info_gains::Vector{Float64})
    n = length(positions)
    @assert n == length(info_gains) "Position and info_gain arrays must have same length"

    if n < 2
        return BitsPerMeterAnalysis(0.0, 0.0, 0.0, 0.0, 0, 0.0)
    end

    # Total distance
    total_distance = sum(norm(positions[i] - positions[i-1]) for i in 2:n)

    # Total information in bits
    LOG2 = log(2.0)
    total_bits = 0.0
    effective = 0
    for ig in info_gains
        if ig < 0  # Negative means uncertainty decreased
            total_bits += -ig / (2 * LOG2)
            effective += 1
        end
    end

    bits_per_meter = total_distance > 0 ? total_bits / total_distance : 0.0
    info_per_update = effective > 0 ? total_bits / effective : 0.0

    return BitsPerMeterAnalysis(
        total_distance,
        total_bits,
        bits_per_meter,
        0.0,  # No time info
        effective,
        info_per_update
    )
end

"""
    print_bits_per_meter(analysis::BitsPerMeterAnalysis)

Print human-readable bits-per-meter analysis.
"""
function print_bits_per_meter(analysis::BitsPerMeterAnalysis)
    println("=" ^ 50)
    println("BITS-PER-METER ANALYSIS")
    println("=" ^ 50)
    println("Distance traveled:    $(round(analysis.total_distance_m, digits=1)) m")
    println("Total information:    $(round(analysis.total_information_bits, digits=2)) bits")
    println("Effective updates:    $(analysis.effective_updates)")
    println("-" ^ 50)
    println("Bits per meter:       $(round(analysis.bits_per_meter * 1000, digits=3)) mbits/m")
    println("Bits per second:      $(round(analysis.bits_per_second * 1000, digits=3)) mbits/s")
    println("Bits per update:      $(round(analysis.info_per_update_bits * 1000, digits=3)) mbits")
    println("=" ^ 50)
end

# ============================================================================
# Diagnostic Helpers
# ============================================================================

"""
    compute_teachability(P_map::Matrix, R_meas::Matrix, H::Matrix) -> Float64

Compute teachability metric: how much can we learn from this measurement?

Returns value in [0, 1] where:
- 0 = nothing to learn (P_map already small relative to R_meas)
- 1 = maximum learning potential

Based on information ratio: tr(K*H*P) / tr(P)
"""
function compute_teachability(P_map::AbstractMatrix, R_meas::AbstractMatrix, H::AbstractMatrix)
    # Innovation covariance
    S = H * P_map * H' + R_meas

    # Kalman gain
    K = P_map * H' / S

    # Information gain ratio
    P_reduction = K * H * P_map
    # Division guard: 1e-20 prevents division by zero when P_map is numerically zero.
    # This is well below Float64 precision (~2.2e-16) so doesn't affect valid results.
    teachability = tr(P_reduction) / (tr(P_map) + 1e-20)

    return clamp(teachability, 0.0, 1.0)
end

"""
    compute_observability_metrics(H::Matrix) -> (min_sv, condition)

Compute observability metrics from measurement Jacobian.

Returns (min_singular_value, condition_number).

# Units
- min_sv: Same units as H (typically T/m for gradient Jacobians)
- condition: Dimensionless ratio σ_max/σ_min
"""
function compute_observability_metrics(H::AbstractMatrix)
    svs = svdvals(H)
    min_sv = minimum(svs)
    max_sv = maximum(svs)
    # Division guard: 1e-20 prevents Inf when min_sv is numerically zero.
    condition = max_sv / (min_sv + 1e-20)
    return (min_sv, condition)
end

"""
    diagnose_rejection(; kwargs...) -> UpdateRejectionReason

Determine why an update would be rejected based on metrics.

# Threshold Justifications (per claude.md quality standards)

## teachability_threshold = 0.01
Statistical basis: Teachability measures tr(K*H*P)/tr(P), the fraction of prior
uncertainty reduced by the update. At 1%, the expected information gain is
negligible relative to computational cost. Derived from requiring at least
0.01 bits of information per update (ln(1/(1-0.01))/2ln(2) ≈ 0.007 bits).

## observability_threshold = 5e-9 T/m
Physical basis: Minimum gradient magnitude needed for position observability.
At 50m altitude, Earth's field gradient is ~3 nT/m. Local anomalies add
10-50 nT/m. Threshold of 5 nT/m ensures SNR > 1 with typical sensor noise
of 2-5 nT/m (gradient derived from differenced magnetometers at ~10cm baseline).

## chi2_threshold = 11.345
Statistical basis: Chi-square critical value for d=3 degrees of freedom at
α=0.01 significance level. From χ²₃(0.99) = 11.345. At this threshold,
1% of valid measurements are incorrectly rejected (Type I error).
Reference: Abramowitz & Stegun, Table 26.8.

## nees_max = 10.0
Statistical basis: For d=3 state dimensions, E[NEES]=3 and Var[NEES]=6.
Upper bound of 10 is ~2.9σ above mean: P(NEES > 10 | consistent) ≈ 0.02.
This provides early warning of filter divergence while tolerating normal
statistical variation. Reference: Bar-Shalom et al., "Estimation with
Applications to Tracking and Navigation", Ch. 5.
"""
function diagnose_rejection(;
    teachability::Float64 = 1.0,
    min_singular_value::Float64 = 1.0,
    innovation_normalized::Float64 = 0.0,
    nees::Float64 = 1.0,
    in_coverage::Bool = true,
    covariance_ok::Bool = true,
    # Threshold: 1% learning gain minimum (see docstring for justification)
    teachability_threshold::Float64 = 0.01,
    # Threshold: 5 nT/m minimum gradient for observability (see docstring)
    observability_threshold::Float64 = 5e-9,
    # Threshold: χ²₃(0.99) = 11.345 for d=3 at α=0.01 (see docstring)
    chi2_threshold::Float64 = 11.345,
    # Threshold: ~2.9σ above E[NEES] for d=3 (see docstring)
    nees_max::Float64 = 10.0
)
    !in_coverage && return OUTSIDE_COVERAGE
    !covariance_ok && return COVARIANCE_SINGULAR
    teachability < teachability_threshold && return LOW_TEACHABILITY
    min_singular_value < observability_threshold && return LOW_OBSERVABILITY
    innovation_normalized > chi2_threshold && return QUALITY_GATE
    nees > nees_max && return NEES_VIOLATION
    return ACCEPTED
end

# ============================================================================
# Exports
# ============================================================================

export UpdateRejectionReason
export ACCEPTED, LOW_TEACHABILITY, LOW_OBSERVABILITY, HIGH_INNOVATION
export NEES_VIOLATION, QUALITY_GATE, COVARIANCE_SINGULAR, OUTSIDE_COVERAGE
export DiagnosticRecord, MissionDiagnosticLog
export record!, finalize!, compute_summary
export to_csv, to_json, print_summary
export compute_teachability, compute_observability_metrics, diagnose_rejection
export BitsPerMeterAnalysis, compute_bits_per_meter, print_bits_per_meter
