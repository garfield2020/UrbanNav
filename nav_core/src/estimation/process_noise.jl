# ============================================================================
# process_noise.jl - Physically-motivated process noise for NEES calibration
# ============================================================================
#
# Step 7 of NEES Recovery Plan:
# Fix process noise and bias models so NEES calibration is meaningful.
#
# The current simple `Q * dt` scaling ignores:
# 1. Different process noise for different state components
# 2. Bias random walk models for IMU sensors
# 3. Correlation between position/velocity process noise
# 4. Physically-motivated noise spectral densities
#
# This module provides:
# - ProcessNoiseConfig: Full specification of continuous-time noise densities
# - BiasRandomWalkConfig: Bias instability and random walk parameters
# - discretize_process_noise: dt-dependent discretization
# - Integration with CalibrationMonitor for online tuning
# ============================================================================

export ProcessNoiseConfig, DEFAULT_PROCESS_NOISE_CONFIG
export BiasRandomWalkConfig, DEFAULT_BIAS_RANDOM_WALK_CONFIG
export ProcessNoiseState, ProcessNoiseResult
export discretize_process_noise, compute_position_process_noise
export compute_velocity_process_noise, compute_attitude_process_noise
export compute_bias_process_noise, compute_full_process_noise
export ProcessNoiseCalibrator, update_process_noise_calibration!
export recommend_process_noise_tuning

using LinearAlgebra
using StaticArrays

# ============================================================================
# Bias Random Walk Configuration
# ============================================================================

"""
    BiasRandomWalkConfig

Configuration for sensor bias random walk models.

# Physical Interpretation
- `σ_rw`: Random walk noise density (units/√s)
- `σ_instability`: Bias instability (units, at 1-hour Allan deviation)
- `τ_corr`: Correlation time for Gauss-Markov approximation (s)

# Fields
- `σ_gyro_rw::Float64`: Gyro bias random walk (rad/s/√Hz)
- `σ_gyro_instability::Float64`: Gyro bias instability (rad/s)
- `τ_gyro::Float64`: Gyro bias correlation time (s)
- `σ_accel_rw::Float64`: Accelerometer bias random walk (m/s²/√Hz)
- `σ_accel_instability::Float64`: Accel bias instability (m/s²)
- `τ_accel::Float64`: Accel bias correlation time (s)
- `σ_mag_rw::Float64`: Magnetometer bias random walk (T/√Hz)
- `σ_mag_instability::Float64`: Magnetometer bias instability (T)
- `τ_mag::Float64`: Magnetometer bias correlation time (s)

# Notes
Typical tactical-grade IMU values:
- Gyro: σ_rw ≈ 1e-5 rad/s/√Hz, instability ≈ 1e-5 rad/s
- Accel: σ_rw ≈ 1e-4 m/s²/√Hz, instability ≈ 1e-4 m/s²
"""
Base.@kwdef struct BiasRandomWalkConfig
    # Gyroscope
    σ_gyro_rw::Float64 = 1e-5           # rad/s/√Hz (angle random walk)
    σ_gyro_instability::Float64 = 1e-5  # rad/s
    τ_gyro::Float64 = 3600.0            # s (1 hour correlation time)

    # Accelerometer
    σ_accel_rw::Float64 = 1e-4          # m/s²/√Hz (velocity random walk)
    σ_accel_instability::Float64 = 1e-4 # m/s²
    τ_accel::Float64 = 3600.0           # s

    # Magnetometer (hard iron drift)
    σ_mag_rw::Float64 = 1e-12           # T/√Hz
    σ_mag_instability::Float64 = 5e-11  # T
    τ_mag::Float64 = 7200.0             # s (2 hour correlation time)
end

const DEFAULT_BIAS_RANDOM_WALK_CONFIG = BiasRandomWalkConfig()

# ============================================================================
# Process Noise Configuration
# ============================================================================

"""
    ProcessNoiseConfig

Full specification of continuous-time process noise spectral densities.

# Physical Model
Process noise represents unmodeled accelerations and rate changes:
- Position: σ_p represents random velocity (m/√s)
- Velocity: σ_v represents random acceleration (m/s/√s = m/s^1.5)
- Attitude: σ_θ represents random angular rate (rad/√s)

# Discretization
For time step dt:
- Q_pos = σ_p² × dt (position variance grows linearly)
- Q_vel = σ_v² × dt + σ_a² × (dt³/3) (velocity from random accel)
- Q_θ = σ_θ² × dt (attitude variance grows linearly)

# Fields
- `σ_position::Float64`: Position process noise density (m/√s)
- `σ_velocity::Float64`: Velocity process noise density (m/s^1.5)
- `σ_attitude::Float64`: Attitude process noise density (rad/√s)
- `bias_config::BiasRandomWalkConfig`: Bias random walk parameters
- `use_correlated_pv::Bool`: Use correlated position-velocity noise
- `scale_with_velocity::Bool`: Scale process noise with vehicle speed
- `velocity_scale_factor::Float64`: How much velocity affects noise
"""
Base.@kwdef struct ProcessNoiseConfig
    # Core process noise densities
    σ_position::Float64 = 0.01          # m/√s
    σ_velocity::Float64 = 0.1           # m/s^1.5
    σ_attitude::Float64 = 0.001         # rad/√s

    # Bias random walk
    bias_config::BiasRandomWalkConfig = DEFAULT_BIAS_RANDOM_WALK_CONFIG

    # Advanced options
    use_correlated_pv::Bool = true      # Position-velocity correlation
    scale_with_velocity::Bool = false   # Dynamic scaling
    velocity_scale_factor::Float64 = 0.1

    # Regularization
    min_process_noise::Float64 = 1e-12  # Minimum variance floor
end

const DEFAULT_PROCESS_NOISE_CONFIG = ProcessNoiseConfig()

# ============================================================================
# Process Noise State (for online tracking)
# ============================================================================

"""
    ProcessNoiseState

Runtime state for adaptive process noise.

Tracks accumulated noise and enables online calibration.
"""
mutable struct ProcessNoiseState
    config::ProcessNoiseConfig

    # Current adaptive multipliers (initialized to 1.0)
    position_multiplier::Float64
    velocity_multiplier::Float64
    attitude_multiplier::Float64
    gyro_bias_multiplier::Float64
    accel_bias_multiplier::Float64

    # Statistics for calibration
    total_time::Float64
    n_updates::Int
end

function ProcessNoiseState(config::ProcessNoiseConfig = DEFAULT_PROCESS_NOISE_CONFIG)
    ProcessNoiseState(
        config,
        1.0, 1.0, 1.0, 1.0, 1.0,  # multipliers
        0.0, 0                     # statistics
    )
end

# ============================================================================
# Process Noise Result
# ============================================================================

"""
    ProcessNoiseResult

Result of process noise discretization for a time step.

# Fields
- `Q_position::SMatrix{3,3}`: 3×3 position process noise
- `Q_velocity::SMatrix{3,3}`: 3×3 velocity process noise
- `Q_attitude::SMatrix{3,3}`: 3×3 attitude process noise
- `Q_bias_gyro::SMatrix{3,3}`: 3×3 gyro bias process noise
- `Q_bias_accel::SMatrix{3,3}`: 3×3 accel bias process noise
- `Q_pv_cross::SMatrix{3,3}`: 3×3 position-velocity cross-correlation
- `dt::Float64`: Time step used for discretization
"""
struct ProcessNoiseResult
    Q_position::SMatrix{3,3,Float64,9}
    Q_velocity::SMatrix{3,3,Float64,9}
    Q_attitude::SMatrix{3,3,Float64,9}
    Q_bias_gyro::SMatrix{3,3,Float64,9}
    Q_bias_accel::SMatrix{3,3,Float64,9}
    Q_pv_cross::SMatrix{3,3,Float64,9}
    dt::Float64
end

# ============================================================================
# Discretization Functions
# ============================================================================

"""
    compute_position_process_noise(σ_p, dt; min_var=1e-12) -> SMatrix{3,3}

Compute 3×3 position process noise for time step dt.

Position variance grows as σ_p² × dt (random walk).
"""
function compute_position_process_noise(σ_p::Float64, dt::Float64;
                                         min_var::Float64 = 1e-12)
    var_p = max(σ_p^2 * dt, min_var)
    return SMatrix{3,3,Float64,9}(var_p * I)
end

"""
    compute_velocity_process_noise(σ_v, dt; min_var=1e-12) -> SMatrix{3,3}

Compute 3×3 velocity process noise for time step dt.

Velocity variance grows as σ_v² × dt from random acceleration.
"""
function compute_velocity_process_noise(σ_v::Float64, dt::Float64;
                                         min_var::Float64 = 1e-12)
    var_v = max(σ_v^2 * dt, min_var)
    return SMatrix{3,3,Float64,9}(var_v * I)
end

"""
    compute_attitude_process_noise(σ_θ, dt; min_var=1e-12) -> SMatrix{3,3}

Compute 3×3 attitude process noise for time step dt.

Attitude variance grows as σ_θ² × dt from random angular rate.
"""
function compute_attitude_process_noise(σ_θ::Float64, dt::Float64;
                                         min_var::Float64 = 1e-12)
    var_θ = max(σ_θ^2 * dt, min_var)
    return SMatrix{3,3,Float64,9}(var_θ * I)
end

"""
    compute_bias_process_noise(σ_rw, τ, dt; min_var=1e-12) -> SMatrix{3,3}

Compute 3×3 bias process noise using first-order Gauss-Markov model.

# Model
Bias evolves as: db/dt = -b/τ + w, where w ~ N(0, σ_rw²)

For discrete time step dt:
- If dt << τ: Q_b ≈ σ_rw² × dt (pure random walk)
- If dt >> τ: Q_b ≈ σ_rw² × τ (steady-state)

# Arguments
- `σ_rw`: Random walk noise density (units/√Hz)
- `τ`: Correlation time (s)
- `dt`: Time step (s)
"""
function compute_bias_process_noise(σ_rw::Float64, τ::Float64, dt::Float64;
                                    min_var::Float64 = 1e-12)
    # First-order Gauss-Markov discretization
    if τ > 0 && dt > 0
        β = 1.0 / τ
        exp_term = exp(-2 * β * dt)
        # Q = (σ²/2β) × (1 - exp(-2βdt))
        var_b = (σ_rw^2 / (2 * β)) * (1 - exp_term)
    else
        # Pure random walk if no correlation time specified
        var_b = σ_rw^2 * dt
    end

    var_b = max(var_b, min_var)
    return SMatrix{3,3,Float64,9}(var_b * I)
end

"""
    discretize_process_noise(config, dt; velocity_magnitude=0.0, state=nothing) -> ProcessNoiseResult

Discretize continuous-time process noise for a given time step.

# Arguments
- `config::ProcessNoiseConfig`: Process noise configuration
- `dt::Float64`: Time step in seconds
- `velocity_magnitude::Float64`: Current vehicle speed (optional, for adaptive scaling)
- `state::ProcessNoiseState`: Optional state for adaptive multipliers

# Returns
ProcessNoiseResult with all process noise matrices.
"""
function discretize_process_noise(config::ProcessNoiseConfig, dt::Float64;
                                  velocity_magnitude::Float64 = 0.0,
                                  state::Union{Nothing, ProcessNoiseState} = nothing)

    min_var = config.min_process_noise

    # Get multipliers from state (if provided)
    p_mult = state !== nothing ? state.position_multiplier : 1.0
    v_mult = state !== nothing ? state.velocity_multiplier : 1.0
    θ_mult = state !== nothing ? state.attitude_multiplier : 1.0
    bg_mult = state !== nothing ? state.gyro_bias_multiplier : 1.0
    ba_mult = state !== nothing ? state.accel_bias_multiplier : 1.0

    # Velocity-based scaling (if enabled)
    velocity_scale = 1.0
    if config.scale_with_velocity && velocity_magnitude > 0
        velocity_scale = 1.0 + config.velocity_scale_factor * velocity_magnitude
    end

    # Position process noise
    σ_p = config.σ_position * p_mult * velocity_scale
    Q_position = compute_position_process_noise(σ_p, dt; min_var=min_var)

    # Velocity process noise
    σ_v = config.σ_velocity * v_mult * velocity_scale
    Q_velocity = compute_velocity_process_noise(σ_v, dt; min_var=min_var)

    # Attitude process noise
    σ_θ = config.σ_attitude * θ_mult
    Q_attitude = compute_attitude_process_noise(σ_θ, dt; min_var=min_var)

    # Bias process noise (gyro)
    bc = config.bias_config
    σ_bg = bc.σ_gyro_rw * bg_mult
    Q_bias_gyro = compute_bias_process_noise(σ_bg, bc.τ_gyro, dt; min_var=min_var)

    # Bias process noise (accel)
    σ_ba = bc.σ_accel_rw * ba_mult
    Q_bias_accel = compute_bias_process_noise(σ_ba, bc.τ_accel, dt; min_var=min_var)

    # Position-velocity cross-correlation
    Q_pv_cross = SMatrix{3,3,Float64,9}(zeros(3,3))
    if config.use_correlated_pv
        # Cross-covariance from correlated random walk model
        # Cov(p,v) = σ_p × σ_v × dt^1.5 / 2 (approximation)
        cross_cov = σ_p * σ_v * dt^1.5 / 2
        Q_pv_cross = SMatrix{3,3,Float64,9}(cross_cov * I)
    end

    # Update state if provided
    if state !== nothing
        state.total_time += dt
        state.n_updates += 1
    end

    return ProcessNoiseResult(
        Q_position, Q_velocity, Q_attitude,
        Q_bias_gyro, Q_bias_accel, Q_pv_cross, dt
    )
end

"""
    compute_full_process_noise(result::ProcessNoiseResult, dim::Int) -> Matrix

Build the full process noise matrix for state dimension dim.

# UrbanNav state)
[position(3), velocity(3), attitude(3), bias_gyro(3), bias_accel(3)]
"""
function compute_full_process_noise(result::ProcessNoiseResult, dim::Int = 15)
    if dim == 15
        # UrbanNav state
        Q = zeros(15, 15)

        # Diagonal blocks
        Q[1:3, 1:3] = result.Q_position
        Q[4:6, 4:6] = result.Q_velocity
        Q[7:9, 7:9] = result.Q_attitude
        Q[10:12, 10:12] = result.Q_bias_gyro
        Q[13:15, 13:15] = result.Q_bias_accel

        # Cross-correlations
        Q[1:3, 4:6] = result.Q_pv_cross
        Q[4:6, 1:3] = result.Q_pv_cross'

        return Matrix(Q)

    elseif dim == 9
        # Position, velocity, attitude only
        Q = zeros(9, 9)
        Q[1:3, 1:3] = result.Q_position
        Q[4:6, 4:6] = result.Q_velocity
        Q[7:9, 7:9] = result.Q_attitude
        Q[1:3, 4:6] = result.Q_pv_cross
        Q[4:6, 1:3] = result.Q_pv_cross'
        return Matrix(Q)

    elseif dim == 6
        # Position, velocity only
        Q = zeros(6, 6)
        Q[1:3, 1:3] = result.Q_position
        Q[4:6, 4:6] = result.Q_velocity
        Q[1:3, 4:6] = result.Q_pv_cross
        Q[4:6, 1:3] = result.Q_pv_cross'
        return Matrix(Q)

    elseif dim == 3
        # Position only
        return Matrix(result.Q_position)
    else
        error("Unsupported state dimension: $dim")
    end
end

# ============================================================================
# Process Noise Calibration
# ============================================================================

"""
    ProcessNoiseCalibrator

Online calibration of process noise based on innovation statistics.

# Principle
If process noise is too low:
- NEES for process (between-factor) will be > 1
- Innovation variance will be higher than predicted

If process noise is too high:
- NEES will be < 1
- Filter is over-conservative (uncertainty too large)

# Algorithm
Uses exponential smoothing of normalized innovations to adjust multipliers.
"""
mutable struct ProcessNoiseCalibrator
    # Target NEES (should be ≈1 for well-calibrated filter)
    target_nees::Float64

    # Smoothing factor (0 < α < 1, higher = faster adaptation)
    α::Float64

    # Current smoothed NEES estimates per component
    nees_position::Float64
    nees_velocity::Float64
    nees_attitude::Float64
    nees_bias::Float64

    # Update counts
    n_position::Int
    n_velocity::Int
    n_attitude::Int
    n_bias::Int

    # Constraints
    min_multiplier::Float64
    max_multiplier::Float64
end

function ProcessNoiseCalibrator(;
    target_nees::Float64 = 1.0,
    α::Float64 = 0.05,
    min_multiplier::Float64 = 0.1,
    max_multiplier::Float64 = 10.0
)
    ProcessNoiseCalibrator(
        target_nees, α,
        1.0, 1.0, 1.0, 1.0,  # Initial NEES estimates
        0, 0, 0, 0,           # Update counts
        min_multiplier, max_multiplier
    )
end

"""
    update_process_noise_calibration!(calibrator, state, component, nees)

Update process noise calibration based on observed NEES.

# Arguments
- `calibrator`: ProcessNoiseCalibrator instance
- `state`: ProcessNoiseState to update multipliers
- `component`: Symbol (:position, :velocity, :attitude, :bias_gyro, :bias_accel)
- `nees`: Observed normalized estimation error squared

# Returns
Updated multiplier for the component.
"""
function update_process_noise_calibration!(
    calibrator::ProcessNoiseCalibrator,
    state::ProcessNoiseState,
    component::Symbol,
    nees::Float64
)
    α = calibrator.α
    target = calibrator.target_nees

    # Update smoothed NEES estimate
    if component == :position
        calibrator.nees_position = (1 - α) * calibrator.nees_position + α * nees
        calibrator.n_position += 1
        smoothed_nees = calibrator.nees_position
    elseif component == :velocity
        calibrator.nees_velocity = (1 - α) * calibrator.nees_velocity + α * nees
        calibrator.n_velocity += 1
        smoothed_nees = calibrator.nees_velocity
    elseif component == :attitude
        calibrator.nees_attitude = (1 - α) * calibrator.nees_attitude + α * nees
        calibrator.n_attitude += 1
        smoothed_nees = calibrator.nees_attitude
    elseif component in (:bias_gyro, :bias_accel, :bias)
        calibrator.nees_bias = (1 - α) * calibrator.nees_bias + α * nees
        calibrator.n_bias += 1
        smoothed_nees = calibrator.nees_bias
    else
        return 1.0
    end

    # Compute multiplier adjustment
    # If NEES > target, we need more process noise (multiplier > 1)
    # If NEES < target, we need less process noise (multiplier < 1)
    # Use sqrt for smooth adjustment
    if smoothed_nees > 0
        adjustment = sqrt(smoothed_nees / target)
    else
        adjustment = 1.0
    end

    # Clamp to allowed range
    adjustment = clamp(adjustment, calibrator.min_multiplier, calibrator.max_multiplier)

    # Apply to state
    if component == :position
        state.position_multiplier *= adjustment^α  # Gradual adjustment
        state.position_multiplier = clamp(state.position_multiplier,
                                          calibrator.min_multiplier,
                                          calibrator.max_multiplier)
        return state.position_multiplier
    elseif component == :velocity
        state.velocity_multiplier *= adjustment^α
        state.velocity_multiplier = clamp(state.velocity_multiplier,
                                          calibrator.min_multiplier,
                                          calibrator.max_multiplier)
        return state.velocity_multiplier
    elseif component == :attitude
        state.attitude_multiplier *= adjustment^α
        state.attitude_multiplier = clamp(state.attitude_multiplier,
                                          calibrator.min_multiplier,
                                          calibrator.max_multiplier)
        return state.attitude_multiplier
    elseif component == :bias_gyro
        state.gyro_bias_multiplier *= adjustment^α
        state.gyro_bias_multiplier = clamp(state.gyro_bias_multiplier,
                                           calibrator.min_multiplier,
                                           calibrator.max_multiplier)
        return state.gyro_bias_multiplier
    elseif component == :bias_accel
        state.accel_bias_multiplier *= adjustment^α
        state.accel_bias_multiplier = clamp(state.accel_bias_multiplier,
                                            calibrator.min_multiplier,
                                            calibrator.max_multiplier)
        return state.accel_bias_multiplier
    end

    return 1.0
end

"""
    recommend_process_noise_tuning(calibrator) -> Dict{Symbol, Float64}

Get recommended multiplier adjustments based on calibration history.

Returns a dictionary of recommended multipliers for each component.
"""
function recommend_process_noise_tuning(calibrator::ProcessNoiseCalibrator)
    target = calibrator.target_nees

    recommendations = Dict{Symbol, Float64}()

    # Position recommendation
    if calibrator.n_position > 10
        recommendations[:position] = sqrt(calibrator.nees_position / target)
    end

    # Velocity recommendation
    if calibrator.n_velocity > 10
        recommendations[:velocity] = sqrt(calibrator.nees_velocity / target)
    end

    # Attitude recommendation
    if calibrator.n_attitude > 10
        recommendations[:attitude] = sqrt(calibrator.nees_attitude / target)
    end

    # Bias recommendation
    if calibrator.n_bias > 10
        recommendations[:bias] = sqrt(calibrator.nees_bias / target)
    end

    return recommendations
end
