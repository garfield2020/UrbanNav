# =============================================================================
# InformationOptimalLearning.jl - Information-Theoretic Map Learning
# =============================================================================
#
# Purpose: Adapt map learning rates based on Fisher Information content
# of each measurement. Standard update policies gate on confidence and
# innovation thresholds; this module weights updates by their expected
# information contribution to the map.
#
# Theory:
# For a map coefficient vector θ with prior P(θ) ~ N(θ₀, Σ₀), and
# measurement z with likelihood P(z|θ) ~ N(h(θ), R):
#
#   Posterior information: Λ_post = Λ_prior + H' R⁻¹ H
#
# where H = ∂h/∂θ is the Jacobian of the measurement model w.r.t.
# map coefficients.
#
# The information gain from this measurement is:
#   ΔI = ½ log det(Λ_post) - ½ log det(Λ_prior)
#
# A measurement is "informative" if ΔI exceeds a threshold.
# The optimal learning weight is derived from the ratio of
# measurement information to prior information.
#
# Integration:
# Works alongside existing AdaptiveUpdatePolicy from map_update_policy.jl.
# Provides an additional information-based weight that modulates the
# policy's confidence/innovation-based weight.
#
# Adapted from patterns in:
# - AUV-Navigation/src/tensor_selectivity.jl (Fisher gain analysis)
# - AUV-Navigation/src/observability.jl (convergence tracking)
# - ALT4/navigation/information.py (FIM computation)
#
# =============================================================================

module InformationOptimalLearning

using LinearAlgebra
using Statistics

# =============================================================================
# Types
# =============================================================================

"""
Configuration for information-optimal learning.

# Parameter Justifications

## min_information_gain_bits = 0.01
A measurement contributing < 0.01 bits is negligible (1/100 of a binary
decision). At this level, the measurement doesn't meaningfully reduce
map uncertainty. Derived from: for d=3 coefficients, 0.01 bits
corresponds to ~0.5% covariance reduction.

## saturation_gain_bits = 1.0
Above 1 bit of gain, the measurement is highly informative and should
receive full weight. 1 bit = factor of 2 in one dimension of
uncertainty volume.

## exploitation_exploration_ratio = 0.7
Fraction of learning budget allocated to exploitation (updating
well-observed coefficients) vs exploration (seeking new information).
0.7 favors exploitation, appropriate for mature maps (Phase E).

## max_learning_rate = 1.0
Upper bound on information-based weight. Values above 1.0 would
over-weight measurements relative to prior, violating Bayesian
consistency.

## coefficient_staleness_threshold_s = 60.0
If a coefficient hasn't been updated in 60 seconds, apply a bonus
to its learning rate. At 2 m/s AUV speed, 60s = 120m of travel.
Magnetic correlation lengths are typically 10-50m, so stale
coefficients likely need refreshing.
"""
struct InformationLearningConfig
    min_information_gain_bits::Float64
    saturation_gain_bits::Float64
    exploitation_exploration_ratio::Float64
    max_learning_rate::Float64
    coefficient_staleness_threshold_s::Float64

    function InformationLearningConfig(;
        min_information_gain_bits::Float64 = 0.01,
        saturation_gain_bits::Float64 = 1.0,
        exploitation_exploration_ratio::Float64 = 0.7,
        max_learning_rate::Float64 = 1.0,
        coefficient_staleness_threshold_s::Float64 = 60.0
    )
        @assert 0 < min_information_gain_bits < saturation_gain_bits
        @assert 0 < exploitation_exploration_ratio <= 1.0
        @assert 0 < max_learning_rate <= 1.0
        new(min_information_gain_bits, saturation_gain_bits,
            exploitation_exploration_ratio, max_learning_rate,
            coefficient_staleness_threshold_s)
    end
end

"""
Result of information-based learning decision.
"""
struct LearningDecision
    should_learn::Bool
    information_gain_bits::Float64
    learning_weight::Float64          # 0 to max_learning_rate
    reason::String
    coefficient_weights::Vector{Float64}  # Per-coefficient weights
end

"""
State tracker for coefficient learning history.

Maintains running information metrics for adaptive learning.
"""
mutable struct LearningTracker
    config::InformationLearningConfig
    n_coefficients::Int
    cumulative_information::Matrix{Float64}    # Accumulated FIM for coefficients
    last_update_times::Vector{Float64}         # Per-coefficient last update
    update_counts::Vector{Int}                 # Per-coefficient update count
    total_gain_bits::Float64                   # Running total of information gained
    n_measurements_processed::Int
    n_measurements_accepted::Int
end

function LearningTracker(config::InformationLearningConfig, n_coefficients::Int)
    LearningTracker(
        config,
        n_coefficients,
        zeros(n_coefficients, n_coefficients),  # Accumulated FIM
        fill(-Inf, n_coefficients),             # Never updated
        zeros(Int, n_coefficients),
        0.0,
        0,
        0
    )
end

# =============================================================================
# Information Gain Computation
# =============================================================================

"""
    compute_measurement_information(
        H::AbstractMatrix,
        R::AbstractMatrix,
        prior_covariance::AbstractMatrix
    ) -> (gain_bits::Float64, fim_contribution::Matrix{Float64})

Compute information gain from a single measurement for map coefficients.

# Arguments
- `H`: Jacobian of measurement w.r.t. map coefficients (m × n)
- `R`: Measurement noise covariance (m × m)
- `prior_covariance`: Current coefficient covariance (n × n)

# Returns
- `gain_bits`: Information gain in bits
- `fim_contribution`: H' R⁻¹ H (the measurement's FIM contribution)

# Theory
The information gain is:
  ΔI = ½ log₂ det(I + Σ₀ H' R⁻¹ H)

This is the mutual information I(z; θ) between the measurement z
and the parameters θ, evaluated at the current estimate.
"""
function compute_measurement_information(
    H::AbstractMatrix,
    R::AbstractMatrix,
    prior_covariance::AbstractMatrix
)
    n = size(prior_covariance, 1)

    # FIM contribution
    R_inv = inv(Matrix(R))
    fim = H' * R_inv * H

    # Information gain: ½ log₂ det(I + Σ₀ H' R⁻¹ H)
    M = I(n) + Matrix(prior_covariance) * fim

    # Use eigenvalues for numerical stability
    eigs = eigvals(Symmetric(M))
    log_det = sum(log.(max.(eigs, 1e-300)))

    gain_bits = log_det / (2.0 * log(2.0))

    return (gain_bits = gain_bits, fim_contribution = Matrix(fim))
end

"""
    compute_per_coefficient_gain(
        H::AbstractMatrix,
        R::AbstractMatrix,
        prior_covariance::AbstractMatrix
    ) -> Vector{Float64}

Compute per-coefficient information gain.

Returns the diagonal contribution of the FIM, normalized by
the prior variance of each coefficient. This indicates which
coefficients benefit most from this measurement.

Per-coefficient gain: Δσ²_i / σ²_i ≈ (H' R⁻¹ H)_ii × σ²_i
"""
function compute_per_coefficient_gain(
    H::AbstractMatrix,
    R::AbstractMatrix,
    prior_covariance::AbstractMatrix
)
    R_inv = inv(Matrix(R))
    fim = H' * R_inv * H

    # Normalized gain per coefficient
    n = size(prior_covariance, 1)
    gains = zeros(n)
    for i in 1:n
        prior_var = prior_covariance[i, i]
        if prior_var > 1e-20
            gains[i] = fim[i, i] * prior_var
        end
    end

    return gains
end

# =============================================================================
# Learning Decision
# =============================================================================

"""
    decide_learning(
        tracker::LearningTracker,
        H::AbstractMatrix,
        R::AbstractMatrix,
        prior_covariance::AbstractMatrix,
        current_time::Float64
    ) -> LearningDecision

Decide whether and how much to learn from a measurement.

# Decision Logic
1. Compute information gain from measurement
2. If gain < min threshold → reject (not informative)
3. Compute per-coefficient weights based on:
   a. Information gain (higher gain → higher weight)
   b. Staleness (stale coefficients get bonus)
   c. Exploitation/exploration balance
4. Return learning decision with per-coefficient weights
"""
function decide_learning(
    tracker::LearningTracker,
    H::AbstractMatrix,
    R::AbstractMatrix,
    prior_covariance::AbstractMatrix,
    current_time::Float64
)
    config = tracker.config
    tracker.n_measurements_processed += 1

    # Compute total information gain
    result = compute_measurement_information(H, R, prior_covariance)
    gain_bits = result.gain_bits
    fim = result.fim_contribution

    # Gate: reject uninformative measurements
    if gain_bits < config.min_information_gain_bits
        return LearningDecision(
            false,
            gain_bits,
            0.0,
            "Below information threshold ($(round(gain_bits, digits=4)) < $(config.min_information_gain_bits) bits)",
            zeros(tracker.n_coefficients)
        )
    end

    # Compute base learning weight from information gain
    # Saturating ramp: 0 at min threshold, 1 at saturation
    base_weight = clamp(
        (gain_bits - config.min_information_gain_bits) /
        (config.saturation_gain_bits - config.min_information_gain_bits),
        0.0, 1.0
    )

    # Per-coefficient weights from measurement information
    per_coeff = compute_per_coefficient_gain(H, R, prior_covariance)

    # Track which coefficients received actual measurement information
    has_measurement_info = per_coeff .> 0

    # Normalize per-coefficient gains to [0, 1]
    max_gain = maximum(per_coeff)
    if max_gain > 0
        per_coeff ./= max_gain
    end

    # Apply staleness bonus only to coefficients with measurement information.
    # Rationale: staleness should increase learning rate for observed coefficients,
    # but cannot create information where the measurement provides none.
    for i in 1:tracker.n_coefficients
        if !has_measurement_info[i]
            continue
        end
        elapsed = current_time - tracker.last_update_times[i]
        if elapsed > config.coefficient_staleness_threshold_s
            staleness_factor = min(elapsed / config.coefficient_staleness_threshold_s, 3.0)
            exploration_bonus = (1.0 - config.exploitation_exploration_ratio) * staleness_factor
            per_coeff[i] = min(per_coeff[i] + exploration_bonus, 1.0)
        end
    end

    # Scale by base weight
    coeff_weights = per_coeff .* base_weight .* config.max_learning_rate

    # Update tracker: only count coefficients with actual measurement information
    tracker.n_measurements_accepted += 1
    tracker.total_gain_bits += gain_bits
    tracker.cumulative_information .+= fim

    for i in 1:tracker.n_coefficients
        if coeff_weights[i] > 0.01 && has_measurement_info[i]
            tracker.last_update_times[i] = current_time
            tracker.update_counts[i] += 1
        end
    end

    return LearningDecision(
        true,
        gain_bits,
        base_weight * config.max_learning_rate,
        "Accepted ($(round(gain_bits, digits=3)) bits)",
        coeff_weights
    )
end

# =============================================================================
# Tracker Diagnostics
# =============================================================================

"""
    learning_efficiency(tracker::LearningTracker) -> Float64

Fraction of measurements that were accepted for learning.

# Interpretation
- < 0.3: Map is mature, most measurements uninformative
- 0.3-0.7: Active learning phase
- > 0.7: Early learning or high-information environment
"""
function learning_efficiency(tracker::LearningTracker)
    if tracker.n_measurements_processed == 0
        return 0.0
    end
    return tracker.n_measurements_accepted / tracker.n_measurements_processed
end

"""
    coefficient_coverage(tracker::LearningTracker) -> Float64

Fraction of coefficients that have been updated at least once.

Below 1.0 indicates unobserved coefficients that may need
different trajectory or measurement strategy.
"""
function coefficient_coverage(tracker::LearningTracker)
    if tracker.n_coefficients == 0
        return 1.0
    end
    return count(c -> c > 0, tracker.update_counts) / tracker.n_coefficients
end

"""
    mean_information_rate(tracker::LearningTracker) -> Float64

Average information gain per accepted measurement (bits).

Declining rate indicates diminishing returns from continued
measurement — the map is approaching convergence.
"""
function mean_information_rate(tracker::LearningTracker)
    if tracker.n_measurements_accepted == 0
        return 0.0
    end
    return tracker.total_gain_bits / tracker.n_measurements_accepted
end

"""
    least_updated_coefficients(tracker::LearningTracker, n::Int=3) -> Vector{Int}

Identify the n least-updated coefficients.

These are candidates for targeted exploration (trajectory design
to improve their observability).
"""
function least_updated_coefficients(tracker::LearningTracker, n::Int=3)
    n = min(n, tracker.n_coefficients)
    perm = sortperm(tracker.update_counts)
    return perm[1:n]
end

"""
    information_summary(tracker::LearningTracker) -> NamedTuple

Summary diagnostics for the learning tracker.
"""
function information_summary(tracker::LearningTracker)
    return (
        n_processed = tracker.n_measurements_processed,
        n_accepted = tracker.n_measurements_accepted,
        efficiency = learning_efficiency(tracker),
        total_bits = tracker.total_gain_bits,
        mean_rate = mean_information_rate(tracker),
        coverage = coefficient_coverage(tracker),
        least_updated = least_updated_coefficients(tracker)
    )
end

# =============================================================================
# Exports
# =============================================================================

export InformationLearningConfig, LearningDecision, LearningTracker
export compute_measurement_information, compute_per_coefficient_gain
export decide_learning
export learning_efficiency, coefficient_coverage, mean_information_rate
export least_updated_coefficients, information_summary

end # module
