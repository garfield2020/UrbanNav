# ============================================================================
# Temporal Coherence - Detection Validation via Temporal Consistency
# ============================================================================
#
# Ported from AUV-Navigation/src/temporal_coherence.jl
#
# Validates detections using temporal consistency:
#   - Signal persistence: Real sources persist over multiple measurements
#   - Profile consistency: Signal follows expected 1/r⁴ gradient falloff
#   - Spectral stability: Tensor eigenvalue ratios remain stable
#   - Causal consistency: No unphysical signal jumps
#
# Key insight: False alarms from noise are temporally incoherent - they
# don't follow the expected spatial/temporal signature of a real dipole.
# ============================================================================

using LinearAlgebra
using StaticArrays
using Statistics: mean, std

# ============================================================================
# Temporal Measurement Buffer
# ============================================================================

"""
    TemporalMeasurement

Single measurement with timestamp and position for temporal analysis.
"""
struct TemporalMeasurement
    timestamp::Float64
    position::SVector{3, Float64}    # Vehicle position
    field::SVector{3, Float64}       # Measured B field
    gradient::SVector{5, Float64}    # Measured gradient tensor (5 independent)
    snr::Float64                     # Instantaneous SNR
    chi2::Float64                    # χ² residual
end

function TemporalMeasurement(timestamp::Float64, position::AbstractVector,
                              field::AbstractVector, gradient::AbstractVector,
                              snr::Float64, chi2::Float64)
    TemporalMeasurement(
        timestamp,
        SVector{3}(position...),
        SVector{3}(field...),
        SVector{5}(gradient...),
        snr, chi2
    )
end

"""
    MeasurementBuffer

Circular buffer of recent measurements for temporal analysis.
"""
mutable struct MeasurementBuffer
    measurements::Vector{TemporalMeasurement}
    max_size::Int
    window_duration::Float64         # Time window to keep (seconds)
end

function MeasurementBuffer(; max_size::Int = 100, window_duration::Float64 = 30.0)
    MeasurementBuffer(TemporalMeasurement[], max_size, window_duration)
end

"""Add measurement to buffer, pruning old entries."""
function add_to_buffer!(buffer::MeasurementBuffer, m::TemporalMeasurement)
    push!(buffer.measurements, m)

    # Prune by size
    while length(buffer.measurements) > buffer.max_size
        popfirst!(buffer.measurements)
    end

    # Prune by time
    if !isempty(buffer.measurements)
        current_time = m.timestamp
        cutoff = current_time - buffer.window_duration
        filter!(x -> x.timestamp >= cutoff, buffer.measurements)
    end
end

"""Get measurements in time window."""
function get_time_window(buffer::MeasurementBuffer, start_time::Float64, end_time::Float64)
    filter(m -> start_time <= m.timestamp <= end_time, buffer.measurements)
end

"""Get measurements within spatial radius of a point."""
function get_spatial_window(buffer::MeasurementBuffer, center::AbstractVector, radius::Float64)
    filter(m -> norm(m.position - SVector{3}(center...)) <= radius, buffer.measurements)
end

# ============================================================================
# Signal Persistence Test
# ============================================================================

"""
    PersistenceConfig

Configuration for persistence validation.
"""
struct PersistenceConfig
    min_observations::Int            # Minimum number of confirming observations
    snr_threshold::Float64           # Minimum SNR to count as observation
    min_duration::Float64            # Minimum time duration of signal (s)
    max_gap::Float64                 # Maximum gap between observations (s)
end

function PersistenceConfig(;
    min_observations::Int = 3,
    snr_threshold::Float64 = 3.0,
    min_duration::Float64 = 1.0,
    max_gap::Float64 = 2.0
)
    PersistenceConfig(min_observations, snr_threshold, min_duration, max_gap)
end

"""
    PersistenceResult

Result of persistence test.
"""
struct PersistenceResult
    is_persistent::Bool
    n_observations::Int
    duration::Float64               # Total time span (s)
    max_gap::Float64                # Largest gap between observations (s)
    mean_snr::Float64
    reason::String
end

"""
    test_persistence(measurements; config)

Test whether measurements show persistent signal.

A real source should produce multiple above-threshold detections
over a contiguous time period as the vehicle passes by.
"""
function test_persistence(measurements::Vector{TemporalMeasurement};
                          config::PersistenceConfig = PersistenceConfig())
    # Filter to above-threshold measurements
    above_thresh = filter(m -> m.snr >= config.snr_threshold, measurements)
    n = length(above_thresh)

    if n == 0
        return PersistenceResult(false, 0, 0.0, Inf, 0.0, "No observations above threshold")
    end

    # Check count
    if n < config.min_observations
        return PersistenceResult(false, n, 0.0, 0.0, 0.0,
            "Too few observations: $n < $(config.min_observations)")
    end

    # Sort by time
    sorted = sort(above_thresh, by=m -> m.timestamp)

    # Compute duration
    duration = sorted[end].timestamp - sorted[1].timestamp

    # Compute max gap
    max_gap = 0.0
    for i in 2:length(sorted)
        gap = sorted[i].timestamp - sorted[i-1].timestamp
        max_gap = max(max_gap, gap)
    end

    # Check duration
    if duration < config.min_duration
        return PersistenceResult(false, n, duration, max_gap, mean(m.snr for m in above_thresh),
            "Duration too short: $(round(duration, digits=2))s < $(config.min_duration)s")
    end

    # Check gap
    if max_gap > config.max_gap
        return PersistenceResult(false, n, duration, max_gap, mean(m.snr for m in above_thresh),
            "Gap too large: $(round(max_gap, digits=2))s > $(config.max_gap)s")
    end

    return PersistenceResult(true, n, duration, max_gap,
        mean(m.snr for m in above_thresh), "Persistent signal")
end

# ============================================================================
# Profile Consistency Test
# ============================================================================

"""
    ProfileConfig

Configuration for profile consistency test.
"""
struct ProfileConfig
    expected_exponent::Float64       # Expected field falloff exponent (3 for B, 4 for G)
    tolerance::Float64               # Tolerance on exponent estimate
    min_points::Int                  # Minimum points for regression
    position_tolerance::Float64      # Tolerance for source position estimate (m)
end

function ProfileConfig(;
    expected_exponent::Float64 = 4.0,  # Gradient falloff
    tolerance::Float64 = 0.5,          # Allow 3.5-4.5
    min_points::Int = 5,
    position_tolerance::Float64 = 2.0
)
    ProfileConfig(expected_exponent, tolerance, min_points, position_tolerance)
end

"""
    ProfileResult

Result of profile consistency test.
"""
struct ProfileResult
    is_consistent::Bool
    estimated_exponent::Float64
    estimated_source_pos::SVector{3, Float64}
    fit_residual::Float64           # RMS fit residual
    r_squared::Float64              # Coefficient of determination
    reason::String
end

"""
    test_profile_consistency(measurements, estimated_source; config)

Test whether signal magnitude follows expected 1/r^n falloff.
"""
function test_profile_consistency(measurements::Vector{TemporalMeasurement},
                                   estimated_source::AbstractVector;
                                   config::ProfileConfig = ProfileConfig())
    n = length(measurements)

    if n < config.min_points
        return ProfileResult(false, 0.0, SVector{3}(estimated_source...), Inf, 0.0,
            "Too few points: $n < $(config.min_points)")
    end

    # Compute distances to estimated source
    distances = Float64[]
    magnitudes = Float64[]

    for m in measurements
        r = norm(m.position - SVector{3}(estimated_source...))
        if r < 0.5  # Avoid singularity
            continue
        end

        # Use gradient magnitude as signal
        G_mag = norm(m.gradient)
        push!(distances, r)
        push!(magnitudes, G_mag)
    end

    if length(distances) < config.min_points
        return ProfileResult(false, 0.0, SVector{3}(estimated_source...), Inf, 0.0,
            "Too few valid points")
    end

    # Log-log regression: log(|G|) = -n × log(r) + C
    log_r = log.(distances)
    log_G = log.(magnitudes)

    # Filter out invalid values
    valid_idx = isfinite.(log_r) .& isfinite.(log_G)
    log_r = log_r[valid_idx]
    log_G = log_G[valid_idx]

    if length(log_r) < config.min_points
        return ProfileResult(false, 0.0, SVector{3}(estimated_source...), Inf, 0.0,
            "Too few valid log points")
    end

    # Linear regression
    mean_x = mean(log_r)
    mean_y = mean(log_G)

    Sxy = sum((log_r .- mean_x) .* (log_G .- mean_y))
    Sxx = sum((log_r .- mean_x).^2)
    Syy = sum((log_G .- mean_y).^2)

    if Sxx < 1e-10
        return ProfileResult(false, 0.0, SVector{3}(estimated_source...), Inf, 0.0,
            "Degenerate data")
    end

    slope = Sxy / Sxx
    estimated_exponent = -slope  # Negative because |G| ∝ 1/r^n

    # R² coefficient
    r_squared = Syy > 1e-10 ? (Sxy^2 / (Sxx * Syy)) : 0.0

    # Compute residuals
    intercept = mean_y - slope * mean_x
    predicted = slope .* log_r .+ intercept
    residuals = log_G .- predicted
    rms_residual = sqrt(mean(residuals.^2))

    # Check exponent
    exponent_error = abs(estimated_exponent - config.expected_exponent)
    is_consistent = exponent_error <= config.tolerance && r_squared > 0.5

    reason = if is_consistent
        "Exponent $(round(estimated_exponent, digits=2)) ≈ $(config.expected_exponent) (R²=$(round(r_squared, digits=2)))"
    elseif exponent_error > config.tolerance
        "Wrong exponent: $(round(estimated_exponent, digits=2)) vs $(config.expected_exponent)"
    else
        "Poor fit: R²=$(round(r_squared, digits=2)) < 0.5"
    end

    return ProfileResult(is_consistent, estimated_exponent, SVector{3}(estimated_source...),
                         rms_residual, r_squared, reason)
end

# ============================================================================
# Spectral Stability Test
# ============================================================================

"""
    SpectralConfig

Configuration for spectral stability test.
"""
struct SpectralConfig
    eigenvalue_ratio_tolerance::Float64   # Tolerance on λ₁/λ₂ ratio
    direction_tolerance_deg::Float64      # Tolerance on principal direction (degrees)
    min_points::Int
end

function SpectralConfig(;
    eigenvalue_ratio_tolerance::Float64 = 0.3,
    direction_tolerance_deg::Float64 = 30.0,
    min_points::Int = 3
)
    SpectralConfig(eigenvalue_ratio_tolerance, direction_tolerance_deg, min_points)
end

"""
    SpectralResult

Result of spectral stability test.
"""
struct SpectralResult
    is_stable::Bool
    mean_eigenvalue_ratio::Float64
    eigenvalue_ratio_std::Float64
    direction_spread_deg::Float64   # Angular spread of principal directions
    reason::String
end

"""
    test_spectral_stability(measurements; config)

Test whether tensor eigenvalue structure is stable over measurements.
"""
function test_spectral_stability(measurements::Vector{TemporalMeasurement};
                                  config::SpectralConfig = SpectralConfig())
    n = length(measurements)

    if n < config.min_points
        return SpectralResult(false, 0.0, Inf, Inf,
            "Too few points: $n < $(config.min_points)")
    end

    eigenvalue_ratios = Float64[]
    principal_directions = SVector{3, Float64}[]

    for m in measurements
        # Unpack gradient to full tensor
        G5 = m.gradient
        Gxx, Gyy, Gxy, Gxz, Gyz = G5[1], G5[2], G5[3], G5[4], G5[5]
        Gzz = -(Gxx + Gyy)

        G = @SMatrix [Gxx Gxy Gxz;
                      Gxy Gyy Gyz;
                      Gxz Gyz Gzz]

        try
            eigenvalues = eigvals(Symmetric(Matrix(G)))
            eigenvectors = eigvecs(Symmetric(Matrix(G)))

            # Sort by absolute value (largest first)
            idx = sortperm(abs.(eigenvalues), rev=true)
            sorted_eig = eigenvalues[idx]
            sorted_vec = eigenvectors[:, idx]

            # Eigenvalue ratio
            if abs(sorted_eig[2]) > 1e-15
                ratio = abs(sorted_eig[1]) / abs(sorted_eig[2])
                push!(eigenvalue_ratios, ratio)
            end

            # Principal direction
            push!(principal_directions, SVector{3}(sorted_vec[:, 1]))
        catch
            continue
        end
    end

    if length(eigenvalue_ratios) < config.min_points
        return SpectralResult(false, 0.0, Inf, Inf, "Too few valid decompositions")
    end

    # Eigenvalue ratio statistics
    mean_ratio = mean(eigenvalue_ratios)
    std_ratio = std(eigenvalue_ratios)

    # Direction spread
    mean_dir = mean(principal_directions)
    mean_dir_norm = normalize(mean_dir)

    angular_devs = Float64[]
    for d in principal_directions
        d_norm = normalize(d)
        # Handle sign ambiguity in eigenvector
        cos_angle = abs(dot(d_norm, mean_dir_norm))
        cos_angle = clamp(cos_angle, -1.0, 1.0)
        angle_deg = acosd(cos_angle)
        push!(angular_devs, angle_deg)
    end
    direction_spread = std(angular_devs)

    # Check criteria
    ratio_stable = std_ratio < config.eigenvalue_ratio_tolerance * mean_ratio
    direction_stable = direction_spread < config.direction_tolerance_deg

    is_stable = ratio_stable && direction_stable

    reason = if is_stable
        "Stable (ratio=$(round(mean_ratio, digits=1))±$(round(std_ratio, digits=1)), dir=$(round(direction_spread, digits=1))°)"
    elseif !ratio_stable
        "Unstable ratio: std/mean = $(round(std_ratio/mean_ratio, digits=2)) > $(config.eigenvalue_ratio_tolerance)"
    else
        "Unstable direction: spread = $(round(direction_spread, digits=1))° > $(config.direction_tolerance_deg)°"
    end

    return SpectralResult(is_stable, mean_ratio, std_ratio, direction_spread, reason)
end

# ============================================================================
# Causal Consistency Test
# ============================================================================

"""
    CausalConfig

Configuration for causal consistency test.
"""
struct CausalConfig
    max_snr_jump::Float64           # Maximum SNR change per sample
    max_position_jump::Float64      # Maximum inferred position jump (m)
    smoothness_threshold::Float64   # Max acceleration in SNR curve
end

function CausalConfig(;
    max_snr_jump::Float64 = 5.0,
    max_position_jump::Float64 = 3.0,
    smoothness_threshold::Float64 = 10.0
)
    CausalConfig(max_snr_jump, max_position_jump, smoothness_threshold)
end

"""
    CausalResult

Result of causal consistency test.
"""
struct CausalResult
    is_causal::Bool
    max_snr_jump::Float64
    smoothness_score::Float64       # Lower is better
    n_anomalous_jumps::Int
    reason::String
end

"""
    test_causal_consistency(measurements; config)

Test whether signal evolution is physically plausible (smooth).
"""
function test_causal_consistency(measurements::Vector{TemporalMeasurement};
                                  config::CausalConfig = CausalConfig())
    n = length(measurements)

    if n < 3
        return CausalResult(true, 0.0, 0.0, 0, "Too few points for causal test")
    end

    # Sort by time
    sorted = sort(measurements, by=m -> m.timestamp)

    # Compute SNR jumps
    snr_jumps = Float64[]
    for i in 2:n
        dt = sorted[i].timestamp - sorted[i-1].timestamp
        if dt > 0
            jump = abs(sorted[i].snr - sorted[i-1].snr)
            push!(snr_jumps, jump)
        end
    end

    max_jump = isempty(snr_jumps) ? 0.0 : maximum(snr_jumps)

    # Compute "acceleration" (second derivative of SNR)
    accelerations = Float64[]
    if n >= 3
        for i in 2:(n-1)
            dt1 = sorted[i].timestamp - sorted[i-1].timestamp
            dt2 = sorted[i+1].timestamp - sorted[i].timestamp

            if dt1 > 0 && dt2 > 0
                v1 = (sorted[i].snr - sorted[i-1].snr) / dt1
                v2 = (sorted[i+1].snr - sorted[i].snr) / dt2
                acc = abs(v2 - v1) / ((dt1 + dt2) / 2)
                push!(accelerations, acc)
            end
        end
    end

    smoothness = isempty(accelerations) ? 0.0 : sqrt(mean(accelerations.^2))

    # Count anomalous jumps
    n_anomalous = count(j -> j > config.max_snr_jump, snr_jumps)

    # Check criteria
    jump_ok = max_jump <= config.max_snr_jump * 2
    smooth_ok = smoothness <= config.smoothness_threshold

    is_causal = jump_ok && smooth_ok && n_anomalous <= 1

    reason = if is_causal
        "Causal (max_jump=$(round(max_jump, digits=1)), smoothness=$(round(smoothness, digits=1)))"
    elseif !jump_ok
        "Large SNR jump: $(round(max_jump, digits=1)) > $(config.max_snr_jump * 2)"
    elseif !smooth_ok
        "Non-smooth: $(round(smoothness, digits=1)) > $(config.smoothness_threshold)"
    else
        "$n_anomalous anomalous jumps"
    end

    return CausalResult(is_causal, max_jump, smoothness, n_anomalous, reason)
end

# ============================================================================
# Combined Temporal Coherence Gate
# ============================================================================

"""
    TemporalCoherenceConfig

Combined configuration for all temporal tests.
"""
struct TemporalCoherenceConfig
    persistence::PersistenceConfig
    profile::ProfileConfig
    spectral::SpectralConfig
    causal::CausalConfig
    require_all::Bool               # Require all tests to pass?
    min_tests_passed::Int           # Minimum tests that must pass
end

function TemporalCoherenceConfig(;
    persistence::PersistenceConfig = PersistenceConfig(),
    profile::ProfileConfig = ProfileConfig(),
    spectral::SpectralConfig = SpectralConfig(),
    causal::CausalConfig = CausalConfig(),
    require_all::Bool = false,
    min_tests_passed::Int = 3
)
    TemporalCoherenceConfig(persistence, profile, spectral, causal, require_all, min_tests_passed)
end

const DEFAULT_TEMPORAL_COHERENCE_CONFIG = TemporalCoherenceConfig()

"""
    TemporalCoherenceResult

Combined result of all temporal coherence tests.
"""
struct TemporalCoherenceResult
    is_coherent::Bool
    persistence::PersistenceResult
    profile::ProfileResult
    spectral::SpectralResult
    causal::CausalResult
    n_tests_passed::Int
    confidence::Float64             # 0-1 confidence score
end

"""
    test_temporal_coherence(measurements, estimated_source; config)

Run all temporal coherence tests and combine results.
"""
function test_temporal_coherence(measurements::Vector{TemporalMeasurement},
                                  estimated_source::AbstractVector;
                                  config::TemporalCoherenceConfig = DEFAULT_TEMPORAL_COHERENCE_CONFIG)
    # Run individual tests
    persistence = test_persistence(measurements; config=config.persistence)
    profile = test_profile_consistency(measurements, estimated_source; config=config.profile)
    spectral = test_spectral_stability(measurements; config=config.spectral)
    causal = test_causal_consistency(measurements; config=config.causal)

    # Count passes
    tests = [persistence.is_persistent, profile.is_consistent,
             spectral.is_stable, causal.is_causal]
    n_passed = count(tests)

    # Determine overall coherence
    is_coherent = if config.require_all
        all(tests)
    else
        n_passed >= config.min_tests_passed
    end

    # Compute confidence (weighted by test strength)
    confidence = 0.0

    if persistence.is_persistent
        confidence += 0.25 * min(1.0, persistence.n_observations / 5)
    end

    if profile.is_consistent
        confidence += 0.25 * profile.r_squared
    end

    if spectral.is_stable
        # Lower std = higher confidence
        spec_conf = 1.0 / (1.0 + spectral.eigenvalue_ratio_std)
        confidence += 0.25 * spec_conf
    end

    if causal.is_causal
        # Lower smoothness score = higher confidence
        causal_conf = 1.0 / (1.0 + causal.smoothness_score / 10)
        confidence += 0.25 * causal_conf
    end

    return TemporalCoherenceResult(
        is_coherent, persistence, profile, spectral, causal,
        n_passed, confidence
    )
end

# ============================================================================
# Exports
# ============================================================================

export TemporalMeasurement, MeasurementBuffer
export add_to_buffer!, get_time_window, get_spatial_window
export PersistenceConfig, PersistenceResult, test_persistence
export ProfileConfig, ProfileResult, test_profile_consistency
export SpectralConfig, SpectralResult, test_spectral_stability
export CausalConfig, CausalResult, test_causal_consistency
export TemporalCoherenceConfig, DEFAULT_TEMPORAL_COHERENCE_CONFIG
export TemporalCoherenceResult, test_temporal_coherence
