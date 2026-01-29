# ============================================================================
# Coordinate Descent Multi-Source MLE
# ============================================================================
#
# Ported from AUV-Navigation/src/coordinate_descent_mle.jl
#
# Implements coordinate descent optimization for localizing multiple discrete
# magnetic dipole sources. This complements the tile-based field learning by
# handling localized anomalies (debris, cables, etc.) as discrete sources.
#
# Algorithm: Iterate over sources, fitting each to residuals from others.
# ============================================================================

using LinearAlgebra
using Statistics

# ============================================================================
# Multi-Source Types
# ============================================================================

"""
    DipoleSourceEstimate

Estimated dipole source with position, moment, and ID.
"""
mutable struct DipoleSourceEstimate
    position::Vector{Float64}   # [x, y, z] in meters
    moment::Vector{Float64}     # [mx, my, mz] in A·m²
    id::Int
end

DipoleSourceEstimate(position::Vector{Float64}, moment::Vector{Float64}) =
    DipoleSourceEstimate(position, moment, 0)

"""
    FieldMeasurementSet

Collection of magnetic field measurements for source localization.
"""
struct FieldMeasurementSet
    positions::Vector{Vector{Float64}}   # Measurement positions
    fields::Vector{Vector{Float64}}       # Measured field vectors (Tesla)
    noise_std::Float64                    # Measurement noise standard deviation
end

# ============================================================================
# Single-Source MLE
# ============================================================================

"""
    SingleSourceResult

Result of single-source maximum likelihood estimation.
"""
struct SingleSourceResult
    source::DipoleSourceEstimate
    converged::Bool
    iterations::Int
    final_cost::Float64
    position_std::Vector{Float64}
    moment_std::Vector{Float64}
end

"""
    dipole_field_estimate(source, obs_position)

Compute magnetic field B at observation position from a dipole source.
Physics: B = (μ₀/4π) × [3(m·r̂)r̂ - m] / |r|³
"""
function dipole_field_estimate(source::DipoleSourceEstimate, obs_position::Vector{Float64})
    r_vec = obs_position - source.position
    r_mag = norm(r_vec)

    if r_mag < 1e-6
        return [0.0, 0.0, 0.0]
    end

    r_hat = r_vec / r_mag
    m = source.moment
    m_dot_r = dot(m, r_hat)

    B = μ₀_4π * (3 * m_dot_r * r_hat - m) / r_mag^3
    return B
end

"""
    dipole_field_jacobians(source, obs_position)

Compute Jacobians of field with respect to source position and moment.

Returns (J_pos, J_mom) where:
- J_pos: 3×3 matrix, ∂B/∂p
- J_mom: 3×3 matrix, ∂B/∂m
"""
function dipole_field_jacobians(source::DipoleSourceEstimate, obs_position::Vector{Float64})
    r_vec = obs_position - source.position
    r_mag = norm(r_vec)

    if r_mag < 1e-6
        return zeros(3, 3), zeros(3, 3)
    end

    r_hat = r_vec / r_mag

    # ∂B/∂m = (μ₀/4π) × [3 r̂ ⊗ r̂ - I] / r³
    J_mom = μ₀_4π * (3 * r_hat * r_hat' - I(3)) / r_mag^3

    # ∂B/∂p via finite differences
    ε = 1e-6
    J_pos = zeros(3, 3)
    for i in 1:3
        p_plus = copy(source.position)
        p_plus[i] += ε
        p_minus = copy(source.position)
        p_minus[i] -= ε

        B_plus = dipole_field_estimate(DipoleSourceEstimate(p_plus, source.moment), obs_position)
        B_minus = dipole_field_estimate(DipoleSourceEstimate(p_minus, source.moment), obs_position)

        J_pos[:, i] = (B_plus - B_minus) / (2ε)
    end

    return J_pos, J_mom
end

"""
    total_field_from_sources(sources, obs_position)

Compute total magnetic field from all sources at observation position.
"""
function total_field_from_sources(sources::Vector{DipoleSourceEstimate}, obs_position::Vector{Float64})
    B = zeros(3)
    for source in sources
        B .+= dipole_field_estimate(source, obs_position)
    end
    return B
end

"""
    field_excluding_source(sources, exclude_idx, obs_position)

Compute total field from all sources EXCEPT the one at exclude_idx.
"""
function field_excluding_source(sources::Vector{DipoleSourceEstimate}, exclude_idx::Int,
                                obs_position::Vector{Float64})
    B = zeros(3)
    for (i, source) in enumerate(sources)
        if i != exclude_idx
            B .+= dipole_field_estimate(source, obs_position)
        end
    end
    return B
end

"""
    single_source_mle(measurements, initial; max_iterations, tolerance)

Estimate a single dipole source from measurements using Levenberg-Marquardt.
"""
function single_source_mle(measurements::FieldMeasurementSet, initial::DipoleSourceEstimate;
                           max_iterations::Int = 50, tolerance::Float64 = 1e-8)
    # State vector: [px, py, pz, mx, my, mz]
    x = vcat(initial.position, initial.moment)
    n_params = 6

    # LM damping
    λ = 1e-3
    λ_up = 10.0
    λ_down = 0.1

    prev_cost = Inf

    for iter in 1:max_iterations
        source = DipoleSourceEstimate(x[1:3], x[4:6])

        # Build normal equations
        H = zeros(n_params, n_params)
        g = zeros(n_params)
        total_cost = 0.0

        for (pos, B_meas) in zip(measurements.positions, measurements.fields)
            B_pred = dipole_field_estimate(source, pos)
            J_pos, J_mom = dipole_field_jacobians(source, pos)
            J = hcat(J_pos, J_mom)

            r = B_meas - B_pred
            total_cost += dot(r, r)

            w = 1.0 / measurements.noise_std^2
            H .+= w * (J' * J)
            g .+= w * (J' * r)
        end

        # Check convergence
        if abs(total_cost - prev_cost) < tolerance * max(1.0, prev_cost)
            local position_std, moment_std
            try
                H_inv = inv(H + 1e-10 * I(n_params))
                param_std = sqrt.(max.(diag(H_inv), 0.0))
                position_std = param_std[1:3]
                moment_std = param_std[4:6]
            catch
                position_std = fill(Inf, 3)
                moment_std = fill(Inf, 3)
            end

            return SingleSourceResult(
                DipoleSourceEstimate(x[1:3], x[4:6], initial.id),
                true, iter, total_cost, position_std, moment_std
            )
        end

        # LM update with damping
        H_damped = H + λ * Diagonal(diag(H) .+ 1e-10)

        try
            Δx = H_damped \ g

            # Limit step size
            pos_step = norm(Δx[1:3])
            if pos_step > 5.0
                Δx[1:3] .*= 5.0 / pos_step
            end

            mom_step = norm(Δx[4:6])
            if mom_step > 500.0
                Δx[4:6] .*= 500.0 / mom_step
            end

            # Evaluate new cost
            x_new = x + Δx
            source_new = DipoleSourceEstimate(x_new[1:3], x_new[4:6])
            new_cost = 0.0
            for (pos, B_meas) in zip(measurements.positions, measurements.fields)
                B_pred = dipole_field_estimate(source_new, pos)
                r = B_meas - B_pred
                new_cost += dot(r, r)
            end

            if new_cost < total_cost
                x .= x_new
                λ *= λ_down
                prev_cost = new_cost
            else
                λ *= λ_up
            end

            λ = clamp(λ, 1e-10, 1e10)
        catch
            λ *= λ_up
        end
    end

    # Did not converge
    return SingleSourceResult(
        DipoleSourceEstimate(x[1:3], x[4:6], initial.id),
        false, max_iterations, prev_cost, fill(Inf, 3), fill(Inf, 3)
    )
end

# ============================================================================
# Coordinate Descent Configuration
# ============================================================================

"""
    CoordinateDescentConfig

Configuration for coordinate descent optimization.
"""
struct CoordinateDescentConfig
    max_sweeps::Int
    position_tolerance::Float64
    max_position_change::Float64
    max_moment_change::Float64
    verbose::Bool
end

function CoordinateDescentConfig(;
    max_sweeps::Int = 10,
    position_tolerance::Float64 = 0.1,
    max_position_change::Float64 = 10.0,
    max_moment_change::Float64 = 1000.0,
    verbose::Bool = false
)
    CoordinateDescentConfig(max_sweeps, position_tolerance, max_position_change,
                            max_moment_change, verbose)
end

const DEFAULT_COORDINATE_DESCENT_CONFIG = CoordinateDescentConfig()

# ============================================================================
# Multi-Source MLE
# ============================================================================

"""
    MultiSourceResult

Result of multi-source coordinate descent MLE.
"""
struct MultiSourceResult
    sources::Vector{DipoleSourceEstimate}
    converged::Bool
    sweeps::Int
    max_position_change::Float64
    per_source_results::Vector{SingleSourceResult}
end

"""
    coordinate_descent_mle(measurements, initial_sources; config)

Estimate multiple dipole sources using coordinate descent.

For each sweep:
1. For each source i:
   - Compute residual = measurements - field from all other sources
   - Fit source i to residual using single-source MLE
   - Update source i if fit is valid
2. Check convergence (max position change < tolerance)
"""
function coordinate_descent_mle(measurements::FieldMeasurementSet,
                                initial_sources::Vector{DipoleSourceEstimate};
                                config::CoordinateDescentConfig = DEFAULT_COORDINATE_DESCENT_CONFIG)
    n_sources = length(initial_sources)
    sources = deepcopy(initial_sources)

    per_source_results = Vector{SingleSourceResult}(undef, n_sources)

    for sweep in 1:config.max_sweeps
        max_pos_change = 0.0

        for i in 1:n_sources
            old_position = copy(sources[i].position)

            # Compute residual: measurements minus field from all OTHER sources
            residual_fields = Vector{Vector{Float64}}()
            for (pos, B_meas) in zip(measurements.positions, measurements.fields)
                B_others = field_excluding_source(sources, i, pos)
                push!(residual_fields, B_meas - B_others)
            end

            residual_measurements = FieldMeasurementSet(
                measurements.positions,
                residual_fields,
                measurements.noise_std
            )

            # Fit source i to residual
            result = single_source_mle(residual_measurements, sources[i])
            per_source_results[i] = result

            # Validate and apply update
            pos_change = norm(result.source.position - old_position)
            mom_change = norm(result.source.moment - sources[i].moment)

            if result.converged &&
               pos_change < config.max_position_change &&
               mom_change < config.max_moment_change
                sources[i] = result.source
                max_pos_change = max(max_pos_change, pos_change)
            end

            if config.verbose
                status = result.converged ? "✓" : "✗"
                println("  Sweep $sweep, Source $i: Δpos=$(round(pos_change, digits=3))m [$status]")
            end
        end

        if config.verbose
            println("Sweep $sweep complete: max_pos_change=$(round(max_pos_change, digits=3))m")
        end

        # Check convergence
        if max_pos_change < config.position_tolerance
            return MultiSourceResult(sources, true, sweep, max_pos_change, per_source_results)
        end
    end

    # Did not converge
    return MultiSourceResult(sources, false, config.max_sweeps, Inf, per_source_results)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    source_separation(sources)

Compute minimum separation between any two sources.
"""
function source_separation(sources::Vector{DipoleSourceEstimate})
    n = length(sources)
    if n < 2
        return Inf
    end

    min_sep = Inf
    for i in 1:n
        for j in (i+1):n
            sep = norm(sources[i].position - sources[j].position)
            min_sep = min(min_sep, sep)
        end
    end
    return min_sep
end

# ============================================================================
# Exports
# ============================================================================

export DipoleSourceEstimate, FieldMeasurementSet
export dipole_field_estimate, dipole_field_jacobians
export total_field_from_sources, field_excluding_source
export SingleSourceResult, single_source_mle
export CoordinateDescentConfig, DEFAULT_COORDINATE_DESCENT_CONFIG
export MultiSourceResult, coordinate_descent_mle
export source_separation
