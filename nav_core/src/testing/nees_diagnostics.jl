# ============================================================================
# nees_diagnostics.jl - NEES Deep-Dive Diagnostics
# ============================================================================
#
# V1.0 Qualification Closure Plan Step 4:
# Comprehensive NEES analysis tools for identifying calibration issues.
#
# This module provides:
# 1. Per-component NEES breakdown (position, velocity, attitude, bias)
# 2. Time-series analysis (trends, windows, autocorrelation)
# 3. Chi-squared distribution checking (goodness-of-fit)
# 4. Covariance consistency diagnostics
# 5. Root cause identification
# 6. Calibration recommendations
#
# NEES = (x_true - x_est)' * P^{-1} * (x_true - x_est)
#
# For a well-calibrated filter:
# - E[NEES] = n (state dimension)
# - NEES ~ χ²(n)
# - 95% of NEES values in [χ²_{0.025}(n), χ²_{0.975}(n)]
# ============================================================================

using LinearAlgebra
using Statistics
using Distributions: Chisq, Normal, cdf, quantile

export NEESComponents, compute_nees_components
export NEESSample, NEESTimeSeries, add_sample!
export NEESWindowStats, compute_window_stats
export ChiSquaredTest, chi2_cdf, chi2_quantile, run_chi2_test
export NEESConsistencyResult, check_nees_consistency
export NEESTrend, detect_nees_trend
export CovarianceDiagnostics, diagnose_covariance
export NEESRootCause, CAUSE_NONE, CAUSE_Q_TOO_SMALL, CAUSE_Q_TOO_LARGE
export CAUSE_R_TOO_SMALL, CAUSE_R_TOO_LARGE, CAUSE_BIAS_UNMODELED
export CAUSE_OBSERVABILITY, CAUSE_DIVERGENCE
export NEESDiagnosticResult, run_nees_diagnostics
export CalibrationRecommendation, generate_recommendations
export format_nees_diagnostic_report

# ============================================================================
# UrbanNav model)
# ============================================================================

# Position: states 1-3
const POS_INDICES = 1:3
# Velocity: states 4-6
const VEL_INDICES = 4:6
# Attitude: states 7-9 (error state: small angles)
const ATT_INDICES = 7:9
# Gyro bias: states 10-12
const GYRO_BIAS_INDICES = 10:12
# Accel bias: states 13-15
const ACCEL_BIAS_INDICES = 13:15

# ============================================================================
# NEES Component Breakdown
# ============================================================================

"""
    NEESComponents

Per-component NEES breakdown.
"""
struct NEESComponents
    total::Float64
    position::Float64
    velocity::Float64
    attitude::Float64
    gyro_bias::Float64
    accel_bias::Float64

    # Dimensions for each component
    n_total::Int
    n_position::Int
    n_velocity::Int
    n_attitude::Int
    n_gyro_bias::Int
    n_accel_bias::Int
end

"""
    compute_nees_components(error::Vector{Float64}, P::AbstractMatrix) -> NEESComponents

Compute NEES broken down by state component.

# Arguments
- `error`: State error vector (x_true - x_est)
- `P`: State covariance matrix

# Returns
NEESComponents with total and per-component NEES values.
"""
function compute_nees_components(error::AbstractVector{Float64}, P::AbstractMatrix)
    n = length(error)

    # Handle different state dimensions
    if n == 15
        # Full 15-state model
        pos_idx = POS_INDICES
        vel_idx = VEL_INDICES
        att_idx = ATT_INDICES
        gyro_idx = GYRO_BIAS_INDICES
        accel_idx = ACCEL_BIAS_INDICES
    elseif n == 9
        # 9-state model (pos, vel, att)
        pos_idx = 1:3
        vel_idx = 4:6
        att_idx = 7:9
        gyro_idx = 1:0  # Empty
        accel_idx = 1:0
    elseif n == 6
        # 6-state model (pos, vel)
        pos_idx = 1:3
        vel_idx = 4:6
        att_idx = 1:0
        gyro_idx = 1:0
        accel_idx = 1:0
    elseif n == 3
        # 3-state model (pos only)
        pos_idx = 1:3
        vel_idx = 1:0
        att_idx = 1:0
        gyro_idx = 1:0
        accel_idx = 1:0
    else
        # Unknown - treat all as position
        pos_idx = 1:n
        vel_idx = 1:0
        att_idx = 1:0
        gyro_idx = 1:0
        accel_idx = 1:0
    end

    # Compute total NEES
    P_inv = inv(P)
    nees_total = error' * P_inv * error

    # Compute per-component NEES
    function component_nees(idx)
        if isempty(idx)
            return 0.0
        end
        e = error[idx]
        P_sub = P[idx, idx]
        if det(P_sub) < 1e-30
            return Inf
        end
        e' * inv(P_sub) * e
    end

    NEESComponents(
        nees_total,
        component_nees(pos_idx),
        component_nees(vel_idx),
        component_nees(att_idx),
        component_nees(gyro_idx),
        component_nees(accel_idx),
        n,
        length(pos_idx),
        length(vel_idx),
        length(att_idx),
        length(gyro_idx),
        length(accel_idx)
    )
end

"""
    normalized_component(comp::NEESComponents, component::Symbol) -> Float64

Get NEES normalized by dimension (should be ≈1 for well-calibrated).
"""
function normalized_component(comp::NEESComponents, component::Symbol)
    if component == :total
        return comp.n_total > 0 ? comp.total / comp.n_total : NaN
    elseif component == :position
        return comp.n_position > 0 ? comp.position / comp.n_position : NaN
    elseif component == :velocity
        return comp.n_velocity > 0 ? comp.velocity / comp.n_velocity : NaN
    elseif component == :attitude
        return comp.n_attitude > 0 ? comp.attitude / comp.n_attitude : NaN
    elseif component == :gyro_bias
        return comp.n_gyro_bias > 0 ? comp.gyro_bias / comp.n_gyro_bias : NaN
    elseif component == :accel_bias
        return comp.n_accel_bias > 0 ? comp.accel_bias / comp.n_accel_bias : NaN
    else
        return NaN
    end
end

# ============================================================================
# NEES Time Series
# ============================================================================

"""
    NEESSample

Single NEES sample with timestamp.
"""
struct NEESSample
    time::Float64
    components::NEESComponents
end

"""
    NEESTimeSeries

Time series of NEES samples for analysis.
"""
mutable struct NEESTimeSeries
    samples::Vector{NEESSample}
    max_samples::Int
end

function NEESTimeSeries(max_samples::Int = 10000)
    NEESTimeSeries(NEESSample[], max_samples)
end

"""
    add_sample!(ts::NEESTimeSeries, time::Float64, error::Vector, P::Matrix)

Add a new NEES sample to the time series.
"""
function add_sample!(ts::NEESTimeSeries, time::Float64,
                     error::AbstractVector{Float64}, P::AbstractMatrix)
    comp = compute_nees_components(error, P)
    sample = NEESSample(time, comp)
    push!(ts.samples, sample)

    # Trim if over capacity
    if length(ts.samples) > ts.max_samples
        popfirst!(ts.samples)
    end
end

"""
    get_component_series(ts::NEESTimeSeries, component::Symbol) -> Vector{Float64}

Extract a specific component as a time series.
"""
function get_component_series(ts::NEESTimeSeries, component::Symbol)
    if component == :total
        return [s.components.total for s in ts.samples]
    elseif component == :position
        return [s.components.position for s in ts.samples]
    elseif component == :velocity
        return [s.components.velocity for s in ts.samples]
    elseif component == :attitude
        return [s.components.attitude for s in ts.samples]
    elseif component == :gyro_bias
        return [s.components.gyro_bias for s in ts.samples]
    elseif component == :accel_bias
        return [s.components.accel_bias for s in ts.samples]
    else
        error("Unknown component: $component")
    end
end

# ============================================================================
# Windowed Statistics
# ============================================================================

"""
    NEESWindowStats

NEES statistics computed over a sliding window.
"""
struct NEESWindowStats
    window_size::Int
    n_samples::Int

    # Per-component statistics
    mean_total::Float64
    std_total::Float64
    mean_position::Float64
    mean_velocity::Float64
    mean_attitude::Float64

    # Consistency (fraction in chi-squared bounds)
    consistency_total::Float64
    consistency_position::Float64
    consistency_velocity::Float64

    # Expected values
    expected_total::Float64
    expected_position::Float64
    expected_velocity::Float64
end

"""
    compute_window_stats(ts::NEESTimeSeries; window_size::Int=100) -> NEESWindowStats

Compute statistics over the last `window_size` samples.
"""
function compute_window_stats(ts::NEESTimeSeries; window_size::Int = 100)
    n = length(ts.samples)
    if n == 0
        return NEESWindowStats(window_size, 0, NaN, NaN, NaN, NaN, NaN, 0.0, 0.0, 0.0, 0, 0, 0)
    end

    # Get last window_size samples
    start_idx = max(1, n - window_size + 1)
    window = ts.samples[start_idx:n]
    n_window = length(window)

    # Extract series
    total = [s.components.total for s in window]
    pos = [s.components.position for s in window]
    vel = [s.components.velocity for s in window]
    att = [s.components.attitude for s in window]

    # Get dimensions from first sample
    first = window[1].components
    n_total = first.n_total
    n_pos = first.n_position
    n_vel = first.n_velocity

    # Compute means
    mean_total = mean(filter(!isinf, total))
    std_total = n_window > 1 ? std(filter(!isinf, total)) : 0.0
    mean_pos = mean(filter(!isinf, pos))
    mean_vel = n_vel > 0 ? mean(filter(!isinf, vel)) : NaN
    mean_att = first.n_attitude > 0 ? mean(filter(!isinf, att)) : NaN

    # Compute consistency (fraction in 95% chi-squared bounds)
    function consistency(values, n_dof)
        if n_dof == 0 || isempty(values)
            return 0.0
        end
        lower = chi2_quantile(n_dof, 0.025)
        upper = chi2_quantile(n_dof, 0.975)
        valid = filter(!isinf, values)
        if isempty(valid)
            return 0.0
        end
        count(x -> lower <= x <= upper, valid) / length(valid)
    end

    NEESWindowStats(
        window_size,
        n_window,
        mean_total,
        std_total,
        mean_pos,
        mean_vel,
        mean_att,
        consistency(total, n_total),
        consistency(pos, n_pos),
        consistency(vel, n_vel),
        n_total,
        n_pos,
        n_vel
    )
end

# ============================================================================
# Chi-Squared Distribution Utilities
# ============================================================================

"""
    chi2_cdf(x::Float64, k::Int) -> Float64

Chi-squared CDF using Distributions.jl.
"""
function chi2_cdf(x::Float64, k::Int)
    if k <= 0 || x < 0
        return 0.0
    end
    cdf(Chisq(k), x)
end

"""
    chi2_quantile(k::Int, p::Float64) -> Float64

Chi-squared quantile using Distributions.jl.
"""
function chi2_quantile(k::Int, p::Float64)
    if k <= 0
        return NaN
    end
    if p <= 0
        return 0.0
    end
    if p >= 1
        return Inf
    end
    quantile(Chisq(k), p)
end

"""
    ChiSquaredTest

Result of chi-squared goodness-of-fit test.
"""
struct ChiSquaredTest
    n_samples::Int
    n_dof::Int
    observed_mean::Float64
    expected_mean::Float64
    observed_variance::Float64
    expected_variance::Float64

    # Test statistics
    mean_ratio::Float64        # observed/expected (should be ≈1)
    variance_ratio::Float64    # observed/expected (should be ≈1)

    # P-values
    p_value_mean::Float64      # From t-test on mean
    p_value_ks::Float64        # Kolmogorov-Smirnov (if available)

    # Verdict
    passes_mean::Bool
    passes_variance::Bool
    passes_overall::Bool
end

"""
    run_chi2_test(values::Vector{Float64}, n_dof::Int; α::Float64=0.05) -> ChiSquaredTest

Run chi-squared distribution test on NEES values.
"""
function run_chi2_test(values::Vector{Float64}, n_dof::Int; α::Float64 = 0.05)
    valid = filter(x -> !isnan(x) && !isinf(x), values)
    n = length(valid)

    if n < 10 || n_dof <= 0
        return ChiSquaredTest(n, n_dof, NaN, NaN, NaN, NaN, NaN, NaN, 1.0, 1.0, true, true, true)
    end

    # Expected values for χ²(n_dof)
    expected_mean = Float64(n_dof)
    expected_var = 2.0 * n_dof

    # Observed values
    obs_mean = mean(valid)
    obs_var = var(valid)

    # Ratios
    mean_ratio = obs_mean / expected_mean
    var_ratio = obs_var / expected_var

    # T-test for mean
    # Under H0: mean = n_dof, Var(NEES) = 2*n_dof
    # Standard error of mean = sqrt(2*n_dof / n)
    se_mean = sqrt(2.0 * n_dof / n)
    t_stat = (obs_mean - expected_mean) / se_mean
    # Two-tailed p-value using normal distribution
    p_mean = 2 * (1 - cdf(Normal(), abs(t_stat)))

    # KS test placeholder (simplified)
    p_ks = 1.0  # Would need proper KS implementation

    # Pass criteria
    # Mean should be within ±20% of expected
    passes_mean = 0.8 <= mean_ratio <= 1.2
    # Variance should be within ±50% of expected
    passes_var = 0.5 <= var_ratio <= 1.5

    ChiSquaredTest(
        n, n_dof,
        obs_mean, expected_mean,
        obs_var, expected_var,
        mean_ratio, var_ratio,
        p_mean, p_ks,
        passes_mean, passes_var,
        passes_mean && passes_var
    )
end

# ============================================================================
# NEES Consistency Check
# ============================================================================

"""
    NEESConsistencyResult

Result of NEES consistency analysis.
"""
struct NEESConsistencyResult
    # Sample counts
    n_samples::Int
    n_dof::Int

    # Overall statistics
    mean_nees::Float64
    std_nees::Float64
    consistency::Float64     # Fraction in 95% bounds

    # Per-component consistency
    consistency_position::Float64
    consistency_velocity::Float64
    consistency_attitude::Float64

    # Bounds
    lower_bound::Float64     # χ²_{0.025}(n)
    upper_bound::Float64     # χ²_{0.975}(n)

    # Verdict
    is_consistent::Bool
    issues::Vector{String}
end

"""
    check_nees_consistency(ts::NEESTimeSeries; min_samples::Int=50) -> NEESConsistencyResult

Check NEES consistency across time series.
"""
function check_nees_consistency(ts::NEESTimeSeries; min_samples::Int = 50)
    n = length(ts.samples)
    issues = String[]

    if n < min_samples
        push!(issues, "Insufficient samples: $n < $min_samples")
        return NEESConsistencyResult(n, 0, NaN, NaN, 0.0, 0.0, 0.0, 0.0, 0.0, Inf, false, issues)
    end

    # Get dimension from first sample
    first = ts.samples[1].components
    n_dof = first.n_total

    # Extract total NEES
    total = [s.components.total for s in ts.samples]
    valid = filter(!isinf, total)

    # Compute statistics
    mean_nees = mean(valid)
    std_nees = std(valid)

    # Chi-squared bounds
    lower = chi2_quantile(n_dof, 0.025)
    upper = chi2_quantile(n_dof, 0.975)

    # Consistency
    consistency = count(x -> lower <= x <= upper, valid) / length(valid)

    # Per-component consistency
    pos = [s.components.position for s in ts.samples]
    vel = [s.components.velocity for s in ts.samples]
    att = [s.components.attitude for s in ts.samples]

    n_pos = first.n_position
    n_vel = first.n_velocity
    n_att = first.n_attitude

    function comp_consistency(vals, dof)
        if dof == 0
            return NaN
        end
        lo = chi2_quantile(dof, 0.025)
        hi = chi2_quantile(dof, 0.975)
        v = filter(!isinf, vals)
        isempty(v) ? 0.0 : count(x -> lo <= x <= hi, v) / length(v)
    end

    cons_pos = comp_consistency(pos, n_pos)
    cons_vel = comp_consistency(vel, n_vel)
    cons_att = comp_consistency(att, n_att)

    # Check for issues
    if mean_nees > n_dof * 1.5
        push!(issues, "NEES mean too high: $(round(mean_nees, digits=2)) vs expected $n_dof")
    end
    if mean_nees < n_dof * 0.5
        push!(issues, "NEES mean too low: $(round(mean_nees, digits=2)) vs expected $n_dof (overconfident)")
    end
    if consistency < 0.85
        push!(issues, "Consistency below 85%: $(round(consistency * 100, digits=1))%")
    end
    if !isnan(cons_pos) && cons_pos < 0.80
        push!(issues, "Position consistency low: $(round(cons_pos * 100, digits=1))%")
    end
    if !isnan(cons_vel) && cons_vel < 0.80
        push!(issues, "Velocity consistency low: $(round(cons_vel * 100, digits=1))%")
    end

    is_consistent = consistency >= 0.85 && 0.7 <= mean_nees / n_dof <= 1.3

    NEESConsistencyResult(
        n, n_dof,
        mean_nees, std_nees, consistency,
        cons_pos, cons_vel, cons_att,
        lower, upper,
        is_consistent, issues
    )
end

# ============================================================================
# Trend Detection
# ============================================================================

"""
    NEESTrend

Result of NEES trend analysis.
"""
struct NEESTrend
    has_trend::Bool
    slope::Float64           # NEES per second
    r_squared::Float64       # Fit quality
    trend_direction::Symbol  # :increasing, :decreasing, :stable
    significance::Float64    # 0-1, how significant the trend is
end

"""
    detect_nees_trend(ts::NEESTimeSeries; window::Int=100) -> NEESTrend

Detect if NEES has a temporal trend (indicating divergence or convergence).
"""
function detect_nees_trend(ts::NEESTimeSeries; window::Int = 100)
    n = length(ts.samples)
    if n < 20
        return NEESTrend(false, 0.0, 0.0, :stable, 0.0)
    end

    # Get last window samples
    start_idx = max(1, n - window + 1)
    window_samples = ts.samples[start_idx:n]

    # Extract times and NEES values
    times = [s.time for s in window_samples]
    nees = [s.components.total for s in window_samples]

    # Filter out infinities
    valid_mask = .!isinf.(nees)
    if sum(valid_mask) < 10
        return NEESTrend(false, 0.0, 0.0, :stable, 0.0)
    end

    t = times[valid_mask]
    y = nees[valid_mask]

    # Normalize time
    t0 = t[1]
    t_norm = t .- t0

    # Linear regression: y = a + b*t
    n_valid = length(t_norm)
    mean_t = mean(t_norm)
    mean_y = mean(y)

    # Slope
    num = sum((t_norm .- mean_t) .* (y .- mean_y))
    den = sum((t_norm .- mean_t).^2)
    slope = den > 1e-10 ? num / den : 0.0

    # R-squared
    y_pred = mean_y .+ slope .* (t_norm .- mean_t)
    ss_res = sum((y .- y_pred).^2)
    ss_tot = sum((y .- mean_y).^2)
    r_squared = ss_tot > 1e-10 ? 1 - ss_res / ss_tot : 0.0

    # Determine trend direction
    direction = if abs(slope) < 0.01
        :stable
    elseif slope > 0
        :increasing
    else
        :decreasing
    end

    # Significance based on R² and magnitude
    significance = r_squared * min(1.0, abs(slope) / 0.1)

    # Has significant trend if R² > 0.3 and slope is meaningful
    has_trend = r_squared > 0.3 && abs(slope) > 0.01

    NEESTrend(has_trend, slope, r_squared, direction, significance)
end

# ============================================================================
# Covariance Diagnostics
# ============================================================================

"""
    CovarianceDiagnostics

Diagnostics for covariance matrix health.
"""
struct CovarianceDiagnostics
    # Condition number
    condition_number::Float64
    is_well_conditioned::Bool

    # Eigenvalue analysis
    min_eigenvalue::Float64
    max_eigenvalue::Float64
    eigenvalue_ratio::Float64

    # Component-wise analysis
    position_std::Vector{Float64}    # σ for x, y, z
    velocity_std::Vector{Float64}
    attitude_std::Vector{Float64}

    # Flags
    has_negative_eigenvalues::Bool
    has_zero_eigenvalues::Bool
    is_symmetric::Bool

    # Issues
    issues::Vector{String}
end

"""
    diagnose_covariance(P::AbstractMatrix) -> CovarianceDiagnostics

Diagnose covariance matrix for issues.
"""
function diagnose_covariance(P::AbstractMatrix)
    issues = String[]
    n = size(P, 1)

    # Symmetry check
    is_symmetric = maximum(abs.(P - P')) < 1e-10

    if !is_symmetric
        push!(issues, "Covariance matrix not symmetric")
    end

    # Make symmetric for analysis
    P_sym = 0.5 * (P + P')

    # Eigenvalue analysis
    eig_vals = eigvals(Hermitian(P_sym))
    min_eig = minimum(eig_vals)
    max_eig = maximum(eig_vals)

    has_negative = min_eig < -1e-10
    has_zero = min_eig < 1e-15

    if has_negative
        push!(issues, "Negative eigenvalues detected: min = $(min_eig)")
    end
    if has_zero && !has_negative
        push!(issues, "Near-zero eigenvalues: min = $(min_eig)")
    end

    # Condition number
    cond = max_eig / max(abs(min_eig), 1e-15)
    is_well_cond = cond < 1e8

    if !is_well_cond
        push!(issues, "Poor conditioning: κ = $(round(cond, sigdigits=3))")
    end

    # Extract component standard deviations
    diag_P = diag(P_sym)

    pos_std = n >= 3 ? sqrt.(max.(diag_P[1:3], 0.0)) : Float64[]
    vel_std = n >= 6 ? sqrt.(max.(diag_P[4:6], 0.0)) : Float64[]
    att_std = n >= 9 ? sqrt.(max.(diag_P[7:9], 0.0)) : Float64[]

    CovarianceDiagnostics(
        cond, is_well_cond,
        min_eig, max_eig, max_eig / max(abs(min_eig), 1e-15),
        pos_std, vel_std, att_std,
        has_negative, has_zero, is_symmetric,
        issues
    )
end

# ============================================================================
# Root Cause Analysis
# ============================================================================

"""
    NEESRootCause

Identified root cause for NEES issues.
"""
@enum NEESRootCause begin
    CAUSE_NONE = 0
    CAUSE_Q_TOO_SMALL = 1     # Process noise too small (overconfident)
    CAUSE_Q_TOO_LARGE = 2     # Process noise too large (underconfident)
    CAUSE_R_TOO_SMALL = 3     # Measurement noise too small
    CAUSE_R_TOO_LARGE = 4     # Measurement noise too large
    CAUSE_BIAS_UNMODELED = 5  # Unmodeled bias causing drift
    CAUSE_OBSERVABILITY = 6   # Observability issues
    CAUSE_DIVERGENCE = 7      # Filter divergence
end

"""
    NEESDiagnosticResult

Complete NEES diagnostic result.
"""
struct NEESDiagnosticResult
    # Input summary
    n_samples::Int
    n_dof::Int
    analysis_window_s::Float64

    # Test results
    chi2_test::ChiSquaredTest
    consistency::NEESConsistencyResult
    trend::NEESTrend
    covariance::Union{Nothing, CovarianceDiagnostics}

    # Root cause analysis
    primary_cause::NEESRootCause
    secondary_causes::Vector{NEESRootCause}
    confidence::Float64  # 0-1

    # Overall health
    is_healthy::Bool
    severity::Symbol     # :none, :warning, :critical
    summary::String
end

"""
    run_nees_diagnostics(ts::NEESTimeSeries;
                         P_latest::Union{Nothing, Matrix}=nothing) -> NEESDiagnosticResult

Run comprehensive NEES diagnostics.
"""
function run_nees_diagnostics(ts::NEESTimeSeries;
                              P_latest::Union{Nothing, AbstractMatrix} = nothing)
    n = length(ts.samples)

    if n == 0
        return NEESDiagnosticResult(
            0, 0, 0.0,
            ChiSquaredTest(0, 0, NaN, NaN, NaN, NaN, NaN, NaN, 1.0, 1.0, true, true, true),
            NEESConsistencyResult(0, 0, NaN, NaN, 0.0, 0.0, 0.0, 0.0, 0.0, Inf, false, ["No samples"]),
            NEESTrend(false, 0.0, 0.0, :stable, 0.0),
            nothing,
            CAUSE_NONE, NEESRootCause[], 0.0,
            true, :none, "No data for analysis"
        )
    end

    # Get dimensions
    first = ts.samples[1].components
    n_dof = first.n_total

    # Time window
    t_start = ts.samples[1].time
    t_end = ts.samples[end].time
    window_s = t_end - t_start

    # Run tests
    total_nees = [s.components.total for s in ts.samples]
    chi2_test = run_chi2_test(total_nees, n_dof)
    consistency = check_nees_consistency(ts)
    trend = detect_nees_trend(ts)
    cov_diag = P_latest !== nothing ? diagnose_covariance(P_latest) : nothing

    # Root cause analysis
    primary_cause = CAUSE_NONE
    secondary_causes = NEESRootCause[]
    confidence = 0.0

    mean_nees = consistency.mean_nees
    expected = Float64(n_dof)

    if !isnan(mean_nees)
        ratio = mean_nees / expected

        if ratio > 1.5
            # NEES too high - covariance underestimates error
            if trend.has_trend && trend.trend_direction == :increasing
                primary_cause = CAUSE_DIVERGENCE
                confidence = trend.significance
            else
                primary_cause = CAUSE_Q_TOO_SMALL
                confidence = min(1.0, (ratio - 1.0) / 0.5)
            end

            # Check for bias
            if consistency.consistency_position < 0.7
                push!(secondary_causes, CAUSE_BIAS_UNMODELED)
            end

        elseif ratio < 0.5
            # NEES too low - covariance overestimates error
            primary_cause = CAUSE_Q_TOO_LARGE
            confidence = min(1.0, (1.0 - ratio) / 0.5)

        elseif !consistency.is_consistent
            # Mean OK but consistency bad
            if cov_diag !== nothing && !cov_diag.is_well_conditioned
                primary_cause = CAUSE_OBSERVABILITY
                confidence = 0.7
            else
                primary_cause = CAUSE_R_TOO_SMALL
                confidence = 0.5
            end
        end
    end

    # Determine severity
    severity = if primary_cause == CAUSE_NONE
        :none
    elseif primary_cause == CAUSE_DIVERGENCE
        :critical
    elseif !consistency.is_consistent
        :warning
    else
        :warning
    end

    is_healthy = primary_cause == CAUSE_NONE

    # Generate summary
    summary = if is_healthy
        "NEES calibration healthy: mean=$(round(mean_nees, digits=2)), consistency=$(round(consistency.consistency * 100, digits=1))%"
    else
        cause_str = string(primary_cause)
        "NEES issue detected: $cause_str (confidence: $(round(confidence * 100, digits=0))%)"
    end

    NEESDiagnosticResult(
        n, n_dof, window_s,
        chi2_test, consistency, trend, cov_diag,
        primary_cause, secondary_causes, confidence,
        is_healthy, severity, summary
    )
end

# ============================================================================
# Calibration Recommendations
# ============================================================================

"""
    CalibrationRecommendation

Specific recommendation for improving NEES calibration.
"""
struct CalibrationRecommendation
    priority::Int           # 1 = highest
    category::Symbol        # :process_noise, :measurement_noise, :model, :observability
    parameter::String       # Which parameter to adjust
    action::Symbol          # :increase, :decrease, :investigate
    magnitude::Float64      # Suggested factor (e.g., 1.5 = increase by 50%)
    rationale::String
end

"""
    generate_recommendations(diag::NEESDiagnosticResult) -> Vector{CalibrationRecommendation}

Generate calibration recommendations based on diagnostics.
"""
function generate_recommendations(diag::NEESDiagnosticResult)
    recs = CalibrationRecommendation[]

    if diag.is_healthy
        return recs
    end

    cause = diag.primary_cause
    ratio = diag.consistency.mean_nees / max(diag.n_dof, 1)

    if cause == CAUSE_Q_TOO_SMALL
        push!(recs, CalibrationRecommendation(
            1, :process_noise, "Q_position",
            :increase, sqrt(ratio),
            "NEES $(round(ratio, digits=2))× expected - increase process noise"
        ))
        push!(recs, CalibrationRecommendation(
            2, :process_noise, "Q_velocity",
            :increase, sqrt(ratio),
            "Velocity process noise likely also underestimated"
        ))

    elseif cause == CAUSE_Q_TOO_LARGE
        push!(recs, CalibrationRecommendation(
            1, :process_noise, "Q_position",
            :decrease, sqrt(ratio),
            "NEES $(round(ratio, digits=2))× expected - decrease process noise"
        ))

    elseif cause == CAUSE_R_TOO_SMALL
        push!(recs, CalibrationRecommendation(
            1, :measurement_noise, "R_magnetic",
            :increase, 1.5,
            "Poor consistency suggests measurement noise underestimated"
        ))

    elseif cause == CAUSE_R_TOO_LARGE
        push!(recs, CalibrationRecommendation(
            1, :measurement_noise, "R_magnetic",
            :decrease, 0.7,
            "Overconfident covariance suggests measurement noise overestimated"
        ))

    elseif cause == CAUSE_BIAS_UNMODELED
        push!(recs, CalibrationRecommendation(
            1, :model, "bias_estimation",
            :investigate,
            1.0,
            "Unmodeled bias causing position drift - check sensor calibration"
        ))

    elseif cause == CAUSE_OBSERVABILITY
        push!(recs, CalibrationRecommendation(
            1, :observability, "trajectory",
            :investigate,
            1.0,
            "Observability issues - consider trajectory redesign or additional sensors"
        ))

    elseif cause == CAUSE_DIVERGENCE
        push!(recs, CalibrationRecommendation(
            1, :model, "filter_reset",
            :investigate,
            1.0,
            "Filter divergence detected - immediate attention required"
        ))
        push!(recs, CalibrationRecommendation(
            2, :process_noise, "Q_all",
            :increase, 2.0,
            "Temporary: increase all process noise to prevent divergence"
        ))
    end

    # Add secondary cause recommendations
    for secondary in diag.secondary_causes
        if secondary == CAUSE_BIAS_UNMODELED && cause != CAUSE_BIAS_UNMODELED
            push!(recs, CalibrationRecommendation(
                3, :model, "bias_model",
                :investigate,
                1.0,
                "Secondary: check for unmodeled biases"
            ))
        end
    end

    # Sort by priority
    sort!(recs, by = r -> r.priority)

    recs
end

# ============================================================================
# Reporting
# ============================================================================

"""
    format_nees_diagnostic_report(diag::NEESDiagnosticResult) -> String

Generate a formatted diagnostic report.
"""
function format_nees_diagnostic_report(diag::NEESDiagnosticResult)
    lines = String[]

    push!(lines, "=" ^ 70)
    push!(lines, "NEES DIAGNOSTIC REPORT")
    push!(lines, "=" ^ 70)
    push!(lines, "")

    # Summary
    status = if diag.is_healthy
        "✓ HEALTHY"
    elseif diag.severity == :warning
        "⚠ WARNING"
    else
        "✗ CRITICAL"
    end
    push!(lines, "STATUS: $status")
    push!(lines, diag.summary)
    push!(lines, "")

    # Sample info
    push!(lines, "-" ^ 70)
    push!(lines, "DATA SUMMARY")
    push!(lines, "-" ^ 70)
    push!(lines, "Samples: $(diag.n_samples)")
    push!(lines, "State dimension: $(diag.n_dof)")
    push!(lines, "Analysis window: $(round(diag.analysis_window_s, digits=1))s")
    push!(lines, "")

    # Chi-squared test
    chi2 = diag.chi2_test
    push!(lines, "-" ^ 70)
    push!(lines, "CHI-SQUARED TEST")
    push!(lines, "-" ^ 70)
    push!(lines, "Mean NEES: $(round(chi2.observed_mean, digits=2)) (expected: $(chi2.expected_mean))")
    push!(lines, "Mean ratio: $(round(chi2.mean_ratio, digits=2)) (should be ≈1.0)")
    push!(lines, "Variance ratio: $(round(chi2.variance_ratio, digits=2)) (should be ≈1.0)")
    push!(lines, "Mean test: $(chi2.passes_mean ? "PASS" : "FAIL")")
    push!(lines, "Variance test: $(chi2.passes_variance ? "PASS" : "FAIL")")
    push!(lines, "")

    # Consistency
    cons = diag.consistency
    push!(lines, "-" ^ 70)
    push!(lines, "CONSISTENCY ANALYSIS")
    push!(lines, "-" ^ 70)
    push!(lines, "Overall: $(round(cons.consistency * 100, digits=1))% (target: ≥85%)")
    if !isnan(cons.consistency_position)
        push!(lines, "Position: $(round(cons.consistency_position * 100, digits=1))%")
    end
    if !isnan(cons.consistency_velocity)
        push!(lines, "Velocity: $(round(cons.consistency_velocity * 100, digits=1))%")
    end
    push!(lines, "95% bounds: [$(round(cons.lower_bound, digits=1)), $(round(cons.upper_bound, digits=1))]")
    if !isempty(cons.issues)
        push!(lines, "Issues:")
        for issue in cons.issues
            push!(lines, "  - $issue")
        end
    end
    push!(lines, "")

    # Trend
    trend = diag.trend
    push!(lines, "-" ^ 70)
    push!(lines, "TREND ANALYSIS")
    push!(lines, "-" ^ 70)
    if trend.has_trend
        push!(lines, "⚠ Trend detected: $(trend.trend_direction)")
        push!(lines, "  Slope: $(round(trend.slope, digits=4)) NEES/s")
        push!(lines, "  R²: $(round(trend.r_squared, digits=3))")
    else
        push!(lines, "No significant trend detected")
    end
    push!(lines, "")

    # Root cause
    if diag.primary_cause != CAUSE_NONE
        push!(lines, "-" ^ 70)
        push!(lines, "ROOT CAUSE ANALYSIS")
        push!(lines, "-" ^ 70)
        push!(lines, "Primary cause: $(diag.primary_cause)")
        push!(lines, "Confidence: $(round(diag.confidence * 100, digits=0))%")
        if !isempty(diag.secondary_causes)
            push!(lines, "Secondary causes: $(join(string.(diag.secondary_causes), ", "))")
        end
        push!(lines, "")
    end

    # Recommendations
    recs = generate_recommendations(diag)
    if !isempty(recs)
        push!(lines, "-" ^ 70)
        push!(lines, "RECOMMENDATIONS")
        push!(lines, "-" ^ 70)
        for rec in recs
            action_str = rec.action == :increase ? "↑ Increase" :
                        rec.action == :decrease ? "↓ Decrease" : "🔍 Investigate"
            mag_str = rec.action in [:increase, :decrease] ? " by $(round((rec.magnitude - 1) * 100, digits=0))%" : ""
            push!(lines, "[P$(rec.priority)] $action_str $(rec.parameter)$mag_str")
            push!(lines, "     $(rec.rationale)")
        end
        push!(lines, "")
    end

    push!(lines, "=" ^ 70)

    join(lines, "\n")
end
