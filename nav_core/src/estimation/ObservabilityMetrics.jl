# =============================================================================
# ObservabilityMetrics.jl - Observability-Aware Trajectory Analysis
# =============================================================================
#
# Purpose: Evaluate and compare trajectory candidates for magnetic navigation
# observability. The magnetic field gradient acts as the measurement Jacobian
# with respect to position, so observability depends on gradient structure
# along the trajectory.
#
# Theory:
# For magnetic navigation, the measurement model is:
#   z = h(x) + v,  where h(x) = B_map(x) + noise
#
# The Jacobian w.r.t. position is:
#   H = ∂B/∂x = G (the magnetic field gradient tensor)
#
# The Fisher Information Matrix for position is:
#   F_nav = Σ_k  G_k' R⁻¹ G_k
#
# where G_k is the 3×3 gradient at measurement k and R is sensor noise.
#
# Observability requires F_nav to be full rank (3 for 3D position).
# The eigenvalues of F_nav indicate:
#   - λ_min: worst-case position accuracy (σ_worst = 1/√λ_min)
#   - condition number: anisotropy of position accuracy
#   - log det(F): total information content
#
# For d=8 (field + gradient tensor measurements):
#   The gradient tensor itself provides additional constraints via
#   the second spatial derivatives (curvature), extending observability.
#
# Adapted from ALT4/navigation/information.py (Python → Julia).
#
# References:
#   - Bar-Shalom et al., "Estimation with Applications to Tracking and
#     Navigation", Ch. 2 (Fisher Information)
#   - Canciani & Raquet, "Absolute Position Determination Using Magnetic
#     Field Measurements", Navigation 2016
#
# =============================================================================

module ObservabilityMetrics

using LinearAlgebra
using StaticArrays
using Statistics

# =============================================================================
# Types
# =============================================================================

"""
Observability analysis result for a single position or trajectory segment.

# Fields
- `fim`: Fisher Information Matrix for position (3×3)
- `eigenvalues`: Eigenvalues of FIM, sorted descending
- `eigenvectors`: Corresponding eigenvectors (columns)
- `condition_number`: λ_max / λ_min (∞ if unobservable)
- `total_information`: log det(FIM) in nats
- `observable_dims`: Number of dimensions with eigenvalue > threshold
- `sigma_worst`: Worst-case position std (1/√λ_min) [m]
- `sigma_best`: Best-case position std (1/√λ_max) [m]
"""
struct ObservabilityResult
    fim::SMatrix{3,3,Float64,9}
    eigenvalues::SVector{3,Float64}
    eigenvectors::SMatrix{3,3,Float64,9}
    condition_number::Float64
    total_information::Float64
    observable_dims::Int
    sigma_worst::Float64
    sigma_best::Float64
end

"""
Trajectory observability profile.

Cumulative observability metrics evaluated along a trajectory.
"""
struct TrajectoryObservability
    positions::Vector{SVector{3,Float64}}
    cumulative_info::Vector{Float64}
    cumulative_observable_dims::Vector{Int}
    cumulative_sigma_worst::Vector{Float64}
    cumulative_condition_number::Vector{Float64}
    segment_info_rate::Vector{Float64}       # Information gain per meter
    final_result::ObservabilityResult
end

"""
Configuration for observability analysis.

# Parameter Justifications

## eigenvalue_threshold_relative = 1e-10
Eigenvalues below 1e-10 × λ_max are numerically indistinguishable
from zero in Float64 arithmetic. This threshold determines which
dimensions are considered "observable".

## field_noise_T = 5e-9
Typical magnetometer noise floor: 5 nT (Ripka, "Magnetic Sensors
and Magnetometers", 2001). Fluxgate sensors achieve 1-10 nT.

## gradient_noise_T_per_m = 2e-9
Gradiometer noise: 2 nT/m for 0.1m baseline with 5 nT sensors.
σ_G = √2 × σ_B / baseline = √2 × 5e-9 / 0.1 ≈ 70e-9 T/m
for single-axis, but multi-axis averaging and longer baselines
reduce to ~2 nT/m effective.

## min_gradient_nT_per_m = 5.0
Minimum gradient magnitude for useful navigation.
At 5 nT/m with 2 nT/m noise → SNR = 2.5, providing
marginally useful position information. Below this,
magnetic updates contribute negligible information.
"""
struct ObservabilityConfig
    eigenvalue_threshold_relative::Float64
    field_noise_T::Float64
    gradient_noise_T_per_m::Float64
    min_gradient_nT_per_m::Float64
    use_gradient_tensor::Bool     # d=8 vs d=3 mode

    function ObservabilityConfig(;
        eigenvalue_threshold_relative::Float64 = 1e-10,
        field_noise_T::Float64 = 5e-9,
        gradient_noise_T_per_m::Float64 = 2e-9,
        min_gradient_nT_per_m::Float64 = 5.0,
        use_gradient_tensor::Bool = true
    )
        new(eigenvalue_threshold_relative, field_noise_T,
            gradient_noise_T_per_m, min_gradient_nT_per_m,
            use_gradient_tensor)
    end
end

"""
Trajectory comparison result.
"""
struct TrajectoryComparison
    names::Vector{String}
    final_info::Vector{Float64}
    final_sigma_worst::Vector{Float64}
    final_condition_number::Vector{Float64}
    convergence_distances::Vector{Float64}  # Distance to reach σ < threshold
    best_trajectory_index::Int
end

# =============================================================================
# Core: Fisher Information from Gradient
# =============================================================================

"""
    compute_position_fim(gradient::AbstractMatrix, config::ObservabilityConfig) -> SMatrix{3,3}

Compute Fisher Information Matrix contribution from a single gradient measurement.

# Physics
The magnetic measurement Jacobian w.r.t. position is:
  H = ∂B/∂x = G (the 3×3 gradient tensor)

For d=3 (field only):
  F += G' R_B⁻¹ G

For d=8 (field + gradient tensor):
  F += G' R_B⁻¹ G + (∂G/∂x)' R_G⁻¹ (∂G/∂x)

The second term adds curvature information, but requires the second
spatial derivatives which we approximate from gradient variation.
"""
function compute_position_fim(
    gradient::AbstractMatrix,
    config::ObservabilityConfig
)
    G = SMatrix{3,3,Float64,9}(gradient)

    # Field contribution: G' R_B⁻¹ G
    R_B_inv = SMatrix{3,3,Float64,9}(
        1/config.field_noise_T^2, 0.0, 0.0,
        0.0, 1/config.field_noise_T^2, 0.0,
        0.0, 0.0, 1/config.field_noise_T^2
    )

    F = G' * R_B_inv * G

    return F
end

"""
    accumulate_fim(gradients::Vector, config::ObservabilityConfig) -> SMatrix{3,3}

Accumulate Fisher Information from multiple gradient measurements.

F_total = Σ_k G_k' R⁻¹ G_k
"""
function accumulate_fim(
    gradients::Vector,
    config::ObservabilityConfig
)
    F = SMatrix{3,3,Float64,9}(zeros(3,3))

    for G in gradients
        F = F + compute_position_fim(G, config)
    end

    return F
end

# =============================================================================
# Observability Analysis
# =============================================================================

"""
    analyze_observability(fim::AbstractMatrix, config::ObservabilityConfig) -> ObservabilityResult

Analyze a Fisher Information Matrix for position observability.

Performs eigendecomposition and computes observability metrics.
"""
function analyze_observability(
    fim::AbstractMatrix,
    config::ObservabilityConfig
)
    F = SMatrix{3,3,Float64,9}(fim)

    # Eigendecomposition
    eig = eigen(Symmetric(Matrix(F)))
    eigenvalues = reverse(eig.values)            # Descending
    eigenvectors = eig.vectors[:, reverse(1:3)]  # Match order

    # Observability threshold
    max_eig = maximum(abs.(eigenvalues))
    threshold = config.eigenvalue_threshold_relative * max(max_eig, 1e-30)

    # Observable dimensions
    observable = eigenvalues .> threshold
    n_observable = sum(observable)

    # Condition number
    if n_observable > 1
        nonzero_eigs = eigenvalues[observable]
        cond = nonzero_eigs[1] / nonzero_eigs[end]
    else
        cond = Inf
    end

    # Total information (log det of observable subspace)
    if n_observable > 0
        total_info = sum(log.(eigenvalues[observable] .+ 1e-300))
    else
        total_info = -Inf
    end

    # Position accuracy bounds (Cramér-Rao)
    # σ = 1/√λ for each eigenvalue
    sigma_worst = n_observable > 0 ? 1.0 / sqrt(eigenvalues[n_observable]) : Inf
    sigma_best = eigenvalues[1] > threshold ? 1.0 / sqrt(eigenvalues[1]) : Inf

    return ObservabilityResult(
        F,
        SVector{3}(eigenvalues),
        SMatrix{3,3}(eigenvectors),
        cond,
        total_info,
        n_observable,
        sigma_worst,
        sigma_best
    )
end

"""
    information_gain(fim_before::AbstractMatrix, fim_after::AbstractMatrix) -> Float64

Compute information gain in bits from adding measurements.

Gain = ½ log₂(det(F_after) / det(F_before))

This is the mutual information between the new measurements
and the position parameters.
"""
function information_gain(
    fim_before::AbstractMatrix,
    fim_after::AbstractMatrix
)
    reg = 1e-10 * I(3)
    logdet_before = logdet(Matrix(fim_before) + reg)
    logdet_after = logdet(Matrix(fim_after) + reg)

    # Convert from nats to bits: divide by 2ln(2)
    return (logdet_after - logdet_before) / (2 * log(2))
end

# =============================================================================
# Trajectory Analysis
# =============================================================================

"""
    analyze_trajectory(
        positions::Vector,
        gradients::Vector,
        config::ObservabilityConfig
    ) -> TrajectoryObservability

Analyze observability evolution along a trajectory.

Computes cumulative FIM at each position and tracks how observability
develops as the vehicle traverses the trajectory.

# Arguments
- `positions`: Vehicle positions along trajectory (N × 3D vectors)
- `gradients`: Magnetic gradient tensors at each position (N × 3×3 matrices)
- `config`: Analysis configuration

# Returns
TrajectoryObservability with cumulative metrics at each position.
"""
function analyze_trajectory(
    positions::Vector,
    gradients::Vector,
    config::ObservabilityConfig
)
    n = length(positions)
    @assert length(gradients) == n "positions and gradients must have same length"

    pos_sv = [SVector{3,Float64}(p) for p in positions]

    cumulative_info = zeros(n)
    cumulative_dims = zeros(Int, n)
    cumulative_sigma = fill(Inf, n)
    cumulative_cond = fill(Inf, n)
    segment_rate = zeros(n)

    F = SMatrix{3,3,Float64,9}(zeros(3,3))

    for k in 1:n
        F = F + compute_position_fim(gradients[k], config)

        result = analyze_observability(F, config)
        cumulative_info[k] = result.total_information
        cumulative_dims[k] = result.observable_dims
        cumulative_sigma[k] = result.sigma_worst
        cumulative_cond[k] = result.condition_number

        # Information rate (gain per meter)
        if k > 1
            dist = norm(pos_sv[k] - pos_sv[k-1])
            if dist > 1e-6
                info_delta = cumulative_info[k] - cumulative_info[k-1]
                segment_rate[k] = info_delta / dist
            end
        end
    end

    final = analyze_observability(F, config)

    return TrajectoryObservability(
        pos_sv,
        cumulative_info,
        cumulative_dims,
        cumulative_sigma,
        cumulative_cond,
        segment_rate,
        final
    )
end

"""
    find_convergence_distance(
        profile::TrajectoryObservability,
        sigma_threshold::Float64
    ) -> Float64

Find distance traveled before position uncertainty drops below threshold.

Returns Inf if threshold is never achieved.
"""
function find_convergence_distance(
    profile::TrajectoryObservability,
    sigma_threshold::Float64
)
    total_dist = 0.0

    for k in 2:length(profile.positions)
        total_dist += norm(profile.positions[k] - profile.positions[k-1])

        if profile.cumulative_sigma_worst[k] < sigma_threshold
            return total_dist
        end
    end

    return Inf
end

"""
    identify_low_observability_segments(
        profile::TrajectoryObservability,
        config::ObservabilityConfig
    ) -> Vector{Tuple{Int,Int}}

Identify trajectory segments where observability is poor.

A segment is "low observability" where the information rate drops
below a threshold, meaning the gradient is too weak to provide
useful position updates.

Returns vector of (start_index, end_index) pairs.
"""
function identify_low_observability_segments(
    profile::TrajectoryObservability,
    config::ObservabilityConfig
)
    min_rate = 0.1  # Minimum info rate (nats/m) for useful navigation
    # Justification: 0.1 nats/m means ~14m needed for 1 bit of information,
    # UrbanNav speed is ~7 seconds per bit — barely useful.

    segments = Tuple{Int,Int}[]
    in_low = false
    start_idx = 0

    for k in 2:length(profile.segment_info_rate)
        if profile.segment_info_rate[k] < min_rate
            if !in_low
                in_low = true
                start_idx = k
            end
        else
            if in_low
                push!(segments, (start_idx, k - 1))
                in_low = false
            end
        end
    end

    if in_low
        push!(segments, (start_idx, length(profile.segment_info_rate)))
    end

    return segments
end

# =============================================================================
# Trajectory Comparison
# =============================================================================

"""
    compare_trajectories(
        names::Vector{String},
        trajectories::Vector{<:Vector},
        gradient_fields::Vector{<:Vector},
        config::ObservabilityConfig;
        convergence_threshold::Float64 = 5.0
    ) -> TrajectoryComparison

Compare multiple candidate trajectories for navigation observability.

# Criterion
The "best" trajectory minimizes σ_worst (worst-case position accuracy)
at mission end, subject to achieving convergence within the shortest distance.

This is a conservative criterion: it optimizes for the weakest dimension,
not the average. This is appropriate for safety-critical navigation.

# Arguments
- `convergence_threshold`: Position uncertainty threshold [m] for convergence
"""
function compare_trajectories(
    names::Vector{String},
    trajectories::Vector{<:Vector},
    gradient_fields::Vector{<:Vector},
    config::ObservabilityConfig;
    convergence_threshold::Float64 = 5.0
)
    n_traj = length(trajectories)
    final_info = zeros(n_traj)
    final_sigma = fill(Inf, n_traj)
    final_cond = fill(Inf, n_traj)
    conv_dist = fill(Inf, n_traj)

    for i in 1:n_traj
        profile = analyze_trajectory(trajectories[i], gradient_fields[i], config)
        final_info[i] = profile.final_result.total_information
        final_sigma[i] = profile.final_result.sigma_worst
        final_cond[i] = profile.final_result.condition_number
        conv_dist[i] = find_convergence_distance(profile, convergence_threshold)
    end

    # Best = lowest σ_worst
    best_idx = argmin(final_sigma)

    return TrajectoryComparison(
        names,
        final_info,
        final_sigma,
        final_cond,
        conv_dist,
        best_idx
    )
end

"""
    gradient_quality_metric(gradient::AbstractMatrix) -> Float64

Quick scalar metric for gradient quality at a single point.

Returns the minimum singular value of the gradient tensor,
which determines the worst-case position observability.

# Interpretation
- < 5 nT/m: Poor observability (σ_pos > 1m per measurement)
- 5-20 nT/m: Moderate observability
- > 20 nT/m: Good observability (σ_pos < 0.25m per measurement)

These thresholds assume 5 nT sensor noise:
σ_pos = σ_B / σ_min(G) = 5e-9 / (5e-9) = 1.0m at 5 nT/m
"""
function gradient_quality_metric(gradient::AbstractMatrix)
    G = Matrix{Float64}(gradient)
    sv = svdvals(G)
    return minimum(sv)
end

# =============================================================================
# Exports
# =============================================================================

export ObservabilityConfig, ObservabilityResult
export TrajectoryObservability, TrajectoryComparison
export compute_position_fim, accumulate_fim
export analyze_observability, information_gain
export analyze_trajectory, find_convergence_distance
export identify_low_observability_segments
export compare_trajectories, gradient_quality_metric

end # module
