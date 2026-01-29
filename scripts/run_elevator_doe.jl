#!/usr/bin/env julia
# ============================================================================
# run_elevator_doe.jl - CLI entry point for elevator DOE
# ============================================================================
#
# Runs the full navigation pipeline: simulates sensor measurements through
# ElevatorWorld, feeds them into the StateEstimator factor graph, and
# measures real position errors against ground truth.
#
# Usage:
#   julia --project=nav_core scripts/run_elevator_doe.jl --practical
#   julia --project=nav_core scripts/run_elevator_doe.jl --screening
#   julia --project=nav_core scripts/run_elevator_doe.jl --poisoning
#
# Outputs:
#   results/elevator_doe_report.md
#   results/elevator_doe_data.csv
# ============================================================================

# Include simulation modules
include(joinpath(@__DIR__, "..", "sim", "worlds", "ElevatorWorld.jl"))
include(joinpath(@__DIR__, "..", "sim", "trajectories", "elevator_doe_trajectories.jl"))
include(joinpath(@__DIR__, "..", "sim", "sensors", "TetrahedronSensor.jl"))
include(joinpath(@__DIR__, "..", "reports", "elevator_doe_report.jl"))

using .ElevatorWorldModule
using .ElevatorDOETrajectories
using .ElevatorDOEReportModule

# Include nav_core testing & estimation modules
include(joinpath(@__DIR__, "..", "nav_core", "src", "testing", "elevator_mode_config.jl"))
include(joinpath(@__DIR__, "..", "nav_core", "src", "testing", "elevator_doe_metrics.jl"))
include(joinpath(@__DIR__, "..", "nav_core", "src", "testing", "elevator_map_poisoning.jl"))
include(joinpath(@__DIR__, "..", "nav_core", "src", "testing", "elevator_doe.jl"))

using StaticArrays
using LinearAlgebra
using Random
using Statistics

# ============================================================================
# Constants
# ============================================================================

const EARTH_FIELD_NED = SVector(22.0, 0.0, 43.3)  # µT, typical mid-latitude

# ============================================================================
# Sensor noise configuration
# ============================================================================

struct SensorNoise
    tetra_config::TetrahedronConfig
    σ_per_bar::Float64      # per-bar noise (T), from ASD * √BW
    σ_reconstructed::Float64 # reconstructed B₀ noise (T), for EKF R matrix
    σ_environmental::Float64 # urban environmental noise (T)
    σ_total::Float64         # total magnetometer noise (T) = √(σ_reconstructed² + σ_environmental²)
    magnetometer::Float64   # σ per axis (µT), for EKF R matrix (= σ_total in µT)
    imu_gyro::Float64       # σ yaw-rate (rad/s)
    odometry::Float64       # σ speed (m/s)
    urban_env::UrbanNoiseEnvironment
end

function SensorNoise(; scale::Float64 = 1.0, nav_bandwidth::Float64 = 10.0)
    config = TetrahedronConfig()
    urban_env = UrbanNoiseEnvironment()

    # Sensor noise: ASD × √BW for per-bar, then reconstruct via least-squares
    σ_per_bar = noise_density(config) * sqrt(nav_bandwidth)      # ~15.8 nT at 10 Hz
    σ_reconstructed = sensor_noise_floor(config; bandwidth_hz=nav_bandwidth)  # ~7.3 nT

    # Environmental noise (RSS of urban sources)
    σ_env = environmental_noise_std(urban_env) * scale  # ~122 nT, scaled by DOE factor

    # Total noise per reconstructed B₀ axis
    σ_total_T = sqrt(σ_reconstructed^2 + σ_env^2)  # ~122 nT (environmental dominates)
    σ_total_µT = σ_total_T * 1e6                    # Convert to µT for EKF

    SensorNoise(
        config,
        σ_per_bar,
        σ_reconstructed,
        σ_env,
        σ_total_T,
        σ_total_µT,    # magnetometer σ (µT) — feeds into EKF R matrix
        1e-4 * scale,  # gyro σ (rad/s)
        0.02 * scale,  # odometry σ (m/s)
        urban_env,
    )
end

# ============================================================================
# Lightweight Magnetic Map (tile-based, harmonic model)
# ============================================================================
#
# Minimal implementation of the tile-based magnetic map from nav_core.
# Each tile stores linear harmonic coefficients: B(x) = B₀ + G·(x - x_ref)
# where B₀ is the field at tile center and G is the 3×3 gradient tensor.
#
# The map is populated from the "true" world field at tile centers (simulating
# a previously-learned map), then the estimator uses it for position updates.
# ============================================================================

struct MagTile
    center::SVector{2,Float64}  # tile center (x, y)
    B0::SVector{3,Float64}      # field at center (µT, NED)
    G::SMatrix{3,2,Float64,6}   # ∂B/∂x, ∂B/∂y at center (µT/m, 3×2)
end

struct MagMap
    tiles::Dict{Tuple{Int,Int}, MagTile}
    tile_size::Float64
    origin::SVector{2,Float64}
end

function MagMap(; tile_size::Float64 = 10.0, origin::SVector{2,Float64} = SVector(0.0, 0.0))
    MagMap(Dict{Tuple{Int,Int}, MagTile}(), tile_size, origin)
end

function tile_index(m::MagMap, pos::SVector{2,Float64})
    ix = floor(Int, (pos[1] - m.origin[1]) / m.tile_size)
    iy = floor(Int, (pos[2] - m.origin[2]) / m.tile_size)
    return (ix, iy)
end

function tile_center(m::MagMap, idx::Tuple{Int,Int})
    cx = m.origin[1] + (idx[1] + 0.5) * m.tile_size
    cy = m.origin[2] + (idx[2] + 0.5) * m.tile_size
    return SVector(cx, cy)
end

"""
    static_background_field(pos)

Static building magnetic anomaly field (µT, NED frame). Represents the
persistent field from infrastructure: rebar, steel beams, HVAC ducts, etc.
This field is spatially varying but time-invariant — exactly what the
magnetic map is designed to capture and exploit for positioning.

The model uses sinusoidal spatial harmonics (physically plausible as they
satisfy Laplace's equation away from sources). Typical urban indoor
anomalies are 1-10 µT on spatial scales of 2-5 m.
"""
function static_background_field(pos::SVector{3,Float64})
    x, y = pos[1], pos[2]
    # Superposition of spatial harmonics at different wavelengths.
    # Amplitudes 5-10 µT, wavelengths 2-6 m — typical of indoor
    # rebar grids, steel studs, HVAC runs at ~1-2m distance.
    Bx = 8.0 * sin(2π * x / 3.0) * cos(2π * y / 4.0) +
         5.0 * cos(2π * x / 2.0) +
         3.0 * sin(2π * (x + y) / 5.0)
    By = 7.0 * cos(2π * x / 4.0) * sin(2π * y / 3.0) +
         4.0 * sin(2π * y / 2.5) +
         2.5 * cos(2π * (x - y) / 4.0)
    Bz = 5.0 * sin(2π * x / 3.5) * sin(2π * y / 3.0) +
         3.0 * cos(2π * x / 2.5)
    return SVector(Bx, By, Bz)
end

"""
    static_background_gradient(pos)

Spatial gradient of the static background field (µT/m). Returns ∂B/∂x
and ∂B/∂y as a 3×2 matrix. Analytically derived from static_background_field.
"""
function static_background_gradient(pos::SVector{3,Float64})
    x, y = pos[1], pos[2]
    # ∂Bx/∂x, ∂By/∂x, ∂Bz/∂x
    dBx_dx = 8.0*(2π/3.0)*cos(2π*x/3.0)*cos(2π*y/4.0) -
             5.0*(2π/2.0)*sin(2π*x/2.0) +
             3.0*(2π/5.0)*cos(2π*(x+y)/5.0)
    dBy_dx = -7.0*(2π/4.0)*sin(2π*x/4.0)*sin(2π*y/3.0) -
              2.5*(2π/4.0)*sin(2π*(x-y)/4.0)
    dBz_dx = 5.0*(2π/3.5)*cos(2π*x/3.5)*sin(2π*y/3.0) -
             3.0*(2π/2.5)*sin(2π*x/2.5)

    # ∂Bx/∂y, ∂By/∂y, ∂Bz/∂y
    dBx_dy = -8.0*(2π/4.0)*sin(2π*x/3.0)*sin(2π*y/4.0) +
              3.0*(2π/5.0)*cos(2π*(x+y)/5.0)
    dBy_dy = 7.0*(2π/3.0)*cos(2π*x/4.0)*cos(2π*y/3.0) +
             4.0*(2π/2.5)*cos(2π*y/2.5) +
             2.5*(2π/4.0)*sin(2π*(x-y)/4.0)
    dBz_dy = 5.0*(2π/3.0)*sin(2π*x/3.5)*cos(2π*y/3.0)

    return SMatrix{3,2,Float64,6}(dBx_dx, dBy_dx, dBz_dx,
                                   dBx_dy, dBy_dy, dBz_dy)
end

"""
    seed_map!(map, world, trajectory, dt)

Populate the magnetic map by sampling the static background field along
the trajectory. This simulates a "previously learned" map from a prior
mission when no elevator was active — the map captures Earth field +
static building anomalies, which together create the spatial gradients
that make magnetics useful for positioning.
"""
function seed_map!(m::MagMap, world, trajectory, dt::Float64 = 1.0)
    dur = ElevatorDOETrajectories.duration(trajectory)
    for t in 0.0:dt:dur
        pos = ElevatorDOETrajectories.position(trajectory, t)
        xy = SVector(pos[1], pos[2])
        idx = tile_index(m, xy)
        haskey(m.tiles, idx) && continue

        center = tile_center(m, idx)
        pos3 = SVector(center[1], center[2], 0.0)

        # Map stores Earth + static building anomaly (no elevator)
        B0 = EARTH_FIELD_NED + static_background_field(pos3)

        # Analytical gradient of the static field
        G = static_background_gradient(pos3)

        m.tiles[idx] = MagTile(center, B0, G)
    end
end

"""
    predict_field(map, pos_xy) -> (B_predicted, H_pos)

Predict the magnetic field at pos_xy from the map, and return the
Jacobian of the prediction w.r.t. position (for EKF update).

Returns (B_pred::SVector{3}, dB_dxy::SMatrix{3,2}) in NED frame (µT).
"""
function predict_field(m::MagMap, pos_xy::SVector{2,Float64})
    idx = tile_index(m, pos_xy)
    if !haskey(m.tiles, idx)
        # No tile data — return Earth field with zero gradient
        return (EARTH_FIELD_NED, SMatrix{3,2,Float64,6}(0,0,0, 0,0,0))
    end
    tile = m.tiles[idx]
    δ = pos_xy - tile.center
    B_pred = tile.B0 + tile.G * δ
    return (B_pred, tile.G)
end

# ============================================================================
# Navigation Estimator — Minimum Viable Urban Nav
# ============================================================================
#
# 5-state EKF for ground-based urban navigation:
#
#   State: [x, y, vx, vy, ψ]
#     x, y    — position (m, NED frame)
#     vx, vy  — velocity (m/s, NED frame)
#     ψ       — yaw heading (rad, 0 = North, CW positive)
#
# Sensors:
#   1. Gyroscope (yaw rate)  → continuous heading propagation
#   2. Odometry (speed)      → bounds drift, projects via heading
#   3. Magnetometer (B field)→ map-relative position update
#
# The magnetometer is NOT used as a yaw oracle. Instead:
#   - Predicted B comes from the tile map at estimated position
#   - Innovation = measured_B_ned - predicted_B_ned
#   - H maps innovation to position correction via ∂B/∂position
#   - When elevator passes, innovation spikes → chi² gating rejects
#   - Yaw is maintained by gyro + odometry consistency
#
# This is the "correct" urban nav measurement model: magnetics anchor
# to the persistent background field, not to Earth's global direction.
# ============================================================================

const N_STATES = 5

mutable struct NavEstimator
    x::Vector{Float64}      # 5-state: [x, y, vx, vy, ψ]
    P::Matrix{Float64}      # 5×5 covariance
    Q::Matrix{Float64}      # 5×5 process noise
    mode::ElevatorModeConfig
    mag_map::MagMap
    # Gyro bias estimate (scalar, yaw only)
    gyro_bias::Float64
    gyro_bias_var::Float64
    # Innovation monitoring
    innovation_history::Vector{Float64}
    cusum_stat::Float64
    source_detected::Bool
    # Source field estimate (Mode C: EMA of innovation during source events)
    source_field_est::SVector{3,Float64}
    # Previous source_detected state (for edge detection)
    prev_source_detected::Bool
    # Continuous source duration counter (for graceful degradation)
    source_duration::Float64
    # Previous measurement magnitude (for temporal change detection)
    prev_B_mag::Float64
    # Tile freeze set (Mode C)
    frozen_tiles::Set{Tuple{Int,Int}}
    # Introspection trace (Mode C diagnostics)
    trace_chi2_before::Vector{Float64}
    trace_chi2_after::Vector{Float64}
    trace_heading_proj::Vector{Float64}     # innovation projection onto heading direction
    trace_heading_Reff::Vector{Float64}     # effective heading R (inflation applied)
    trace_interlock_fired::Vector{Bool}     # safety interlock triggered
    trace_source_detected::Vector{Bool}
end

function NavEstimator(initial_pos::SVector{3,Float64}, mode::ElevatorModeConfig,
                      mag_map::MagMap)
    x = zeros(N_STATES)
    x[1] = initial_pos[1]  # x
    x[2] = initial_pos[2]  # y

    P = diagm([
        0.5^2,   # x position
        0.5^2,   # y position
        0.05^2,  # vx
        0.05^2,  # vy
        0.02^2,  # yaw (~1°)
    ])

    Q = diagm([
        0.02^2,  # x process noise
        0.02^2,  # y process noise
        0.02^2,  # vx process noise
        0.02^2,  # vy process noise
        0.1^2,   # yaw process noise (large: pedestrian turns frequently)
    ])

    NavEstimator(x, P, Q, mode, mag_map,
                 0.0, 1e-6^2,     # gyro bias
                 Float64[], 0.0, false,
                 SVector(0.0, 0.0, 0.0),  # source field estimate
                 false,                    # prev_source_detected
                 0.0,                      # source_duration
                 0.0,                      # prev_B_mag
                 Set{Tuple{Int,Int}}(),
                 Float64[], Float64[], Float64[], Float64[], Bool[], Bool[])
end

"""
    predict!(est, gyro_yaw_rate, dt)

Propagate state using gyro yaw rate and current velocity.
- Yaw integrates from gyroscope (bias-corrected)
- Position integrates from velocity
"""
function predict!(est::NavEstimator, gyro_yaw_rate::Float64, dt::Float64)
    x, y = est.x[1], est.x[2]
    vx, vy = est.x[3], est.x[4]
    ψ = est.x[5]

    # Bias-corrected yaw rate
    ψ_dot = gyro_yaw_rate - est.gyro_bias

    # State prediction
    est.x[1] = x + vx * dt
    est.x[2] = y + vy * dt
    # vx, vy unchanged (odometry sets them)
    est.x[5] = ψ + ψ_dot * dt

    # State transition Jacobian
    # x_{k+1} = x_k + vx*dt, where vx = speed*cos(ψ)
    # So ∂x/∂ψ = -vy*dt, ∂y/∂ψ = vx*dt (velocity rotates with heading)
    F = Matrix{Float64}(I, N_STATES, N_STATES)
    F[1, 3] = dt  # ∂x/∂vx
    F[2, 4] = dt  # ∂y/∂vy
    F[1, 5] = -vy * dt  # ∂x/∂ψ: heading error → cross-track position drift
    F[2, 5] = vx * dt   # ∂y/∂ψ

    est.P = F * est.P * F' + est.Q * dt
    est.P = 0.5 * (est.P + est.P')

    # Gyro bias random walk
    est.gyro_bias_var += 1e-10 * dt

    # Clamp covariance
    for j in 1:N_STATES
        est.P[j,j] = clamp(est.P[j,j], 1e-10, 100.0)
    end
end

"""
    update_odometry!(est, measured_speed, noise_std)

Speed measurement: projects body-frame speed through estimated heading
to set NED velocity. This is the primary drift-bounding sensor.
"""
function update_odometry!(est::NavEstimator, measured_speed::Float64,
                          noise_std::Float64)
    ψ = est.x[5]
    cy, sy = cos(ψ), sin(ψ)

    # Set NED velocity from body-frame speed + heading
    est.x[3] = cy * measured_speed
    est.x[4] = sy * measured_speed

    # Reduce velocity covariance
    est.P[3,3] = min(est.P[3,3], noise_std^2)
    est.P[4,4] = min(est.P[4,4], noise_std^2)
end

"""
    update_magnetometer!(est, measured_B_body, noise_std)

Map-relative magnetic field update using BODY-frame measurements.

The measurement model:
  B_body = R(ψ)' * B_map(x, y)
where R(ψ) is the yaw rotation and B_map comes from the tile map.

This makes the measurement depend on BOTH position AND heading:
- Position: through the spatial field variation B_map(x,y)
- Heading: through the rotation R(ψ) from NED to body

The Jacobian H therefore has non-zero entries for x, y, AND ψ,
giving heading observability through the map matching.

Mode-specific handling:
- Mode A: always update (baseline, no source handling)
- Mode B: chi² gating — inflate R when innovation is anomalous
- Mode C: CUSUM detection + heavy inflation during source events
"""
function update_magnetometer!(est::NavEstimator, measured_B_body::SVector{3,Float64},
                               noise_std::Float64; fit_residual::Float64 = NaN)
    pos_xy = SVector(est.x[1], est.x[2])
    B_ned_map, dB_dxy = predict_field(est.mag_map, pos_xy)

    # Rotate predicted B to body frame using estimated heading
    ψ = est.x[5]
    cy, sy = cos(ψ), sin(ψ)
    # R_ned2body (yaw only, pedestrian = ground plane):
    # [cy  sy  0]
    # [-sy cy  0]
    # [0   0   1]
    B_pred_body = SVector(
        cy * B_ned_map[1] + sy * B_ned_map[2],
        -sy * B_ned_map[1] + cy * B_ned_map[2],
        B_ned_map[3]
    )

    # Innovation in body frame
    innovation = measured_B_body - B_pred_body

    # Normalized innovation magnitude
    innov_norm = sqrt(sum(innovation .^ 2)) / noise_std
    push!(est.innovation_history, innov_norm)

    # Measurement Jacobian: H maps state [x, y, vx, vy, ψ] to B_body
    # B_body = R(ψ)' * B_ned(x,y)
    #
    # ∂B_body/∂x = R(ψ)' * ∂B_ned/∂x
    # ∂B_body/∂y = R(ψ)' * ∂B_ned/∂y
    # ∂B_body/∂ψ = ∂R(ψ)'/∂ψ * B_ned
    #            = [-sy  cy  0; -cy -sy  0; 0 0 0] * B_ned

    H = zeros(3, N_STATES)

    # Position columns: rotate map gradient to body frame
    # ∂B_body/∂x = R' * dB_ned/dx (column 1 of dB_dxy)
    dBned_dx = SVector(dB_dxy[1,1], dB_dxy[2,1], dB_dxy[3,1])
    dBned_dy = SVector(dB_dxy[1,2], dB_dxy[2,2], dB_dxy[3,2])

    H[1, 1] = cy * dBned_dx[1] + sy * dBned_dx[2]   # ∂Bbody_x/∂x
    H[2, 1] = -sy * dBned_dx[1] + cy * dBned_dx[2]   # ∂Bbody_y/∂x
    H[3, 1] = dBned_dx[3]                              # ∂Bbody_z/∂x

    H[1, 2] = cy * dBned_dy[1] + sy * dBned_dy[2]   # ∂Bbody_x/∂y
    H[2, 2] = -sy * dBned_dy[1] + cy * dBned_dy[2]   # ∂Bbody_y/∂y
    H[3, 2] = dBned_dy[3]                              # ∂Bbody_z/∂y

    # Heading column: ∂B_body/∂ψ = dR'/dψ * B_ned
    H[1, 5] = -sy * B_ned_map[1] + cy * B_ned_map[2]   # ∂Bbody_x/∂ψ
    H[2, 5] = -cy * B_ned_map[1] - sy * B_ned_map[2]   # ∂Bbody_y/∂ψ
    H[3, 5] = 0.0                                        # ∂Bbody_z/∂ψ

    R_base = Matrix{Float64}(I, 3, 3) * noise_std^2

    # --- Mode dispatch ---
    if est.mode.mode == NAV_MODE_A_BASELINE
        # No source handling — always update
        _do_mag_ekf_update!(est, innovation, H, R_base, 1.0)

    elseif est.mode.mode == NAV_MODE_B_ROBUST_IGNORE
        # Conservative scalar chi² gating: inflate R uniformly when
        # innovation is anomalous. Simple and safe but loses heading
        # info during source events (the tradeoff Mode C improves on).
        chi2 = sum(innovation .^ 2) / noise_std^2
        chi2_threshold = 11.345  # 3-DOF, α=0.01

        if chi2 <= chi2_threshold
            _do_mag_ekf_update!(est, innovation, H, R_base, 1.0)
        else
            # Linear scaling (not sqrt) for stronger rejection of
            # high-chi² measurements. Conservative but safe.
            inflation = chi2 / chi2_threshold
            _do_mag_ekf_update!(est, innovation, H, R_base, inflation)
        end

    elseif est.mode.mode == NAV_MODE_C_SOURCE_AWARE
        # Primary: physics-based near-field detection from tetrahedron fit residual.
        # Fit residual > 3 indicates near-field source (9 DOF chi² test).
        # More principled than magnitude-based CUSUM since it measures whether
        # the field is consistent with a linear gradient model.
        fit_residual_active = !isnan(fit_residual) && fit_residual > 3.0
        if fit_residual_active
            est.source_detected = true
            idx = tile_index(est.mag_map, pos_xy)
            push!(est.frozen_tiles, idx)
        end

        # Secondary: CUSUM on absolute field magnitude deviation (fallback)
        B_earth_mag = sqrt(sum(EARTH_FIELD_NED .^ 2))
        B_meas_mag = sqrt(sum(measured_B_body .^ 2))
        mag_dev = abs(B_meas_mag - B_earth_mag) / noise_std
        est.prev_B_mag = B_meas_mag

        est.cusum_stat = max(0.0, est.cusum_stat + mag_dev -
                             est.mode.cusum_drift - 1.5)

        if est.cusum_stat > est.mode.cusum_threshold
            est.source_detected = true
            idx = tile_index(est.mag_map, pos_xy)
            push!(est.frozen_tiles, idx)
        elseif est.cusum_stat < 0.5 && !fit_residual_active
            est.source_detected = false
        end

        est.prev_source_detected = est.source_detected

        # Track continuous source duration for graceful degradation
        if est.source_detected
            est.source_duration += 0.1  # dt
        else
            est.source_duration = 0.0
        end

        chi2 = sum(innovation .^ 2) / noise_std^2
        chi2_threshold = 11.345

        # Trace: heading projection and source state
        h_psi_trace = SVector(H[1,5], H[2,5], H[3,5])
        h_norm_trace = sqrt(sum(h_psi_trace .^ 2))
        heading_proj = h_norm_trace > 1e-10 ?
            abs(dot(h_psi_trace / h_norm_trace, innovation)) / noise_std : 0.0
        push!(est.trace_source_detected, est.source_detected)
        push!(est.trace_heading_proj, heading_proj)

        if chi2 > chi2_threshold
            inflation = sqrt(chi2 / chi2_threshold)

            # Always use heading-protected update in Mode C.
            # The heading correction clamp inside prevents cumulative drift.
            _do_mag_ekf_update_heading_protected!(est, innovation, H, R_base, inflation)

            push!(est.trace_chi2_before, chi2)
            push!(est.trace_chi2_after, chi2)
            push!(est.trace_heading_Reff, inflation)
            push!(est.trace_interlock_fired, false)
        else
            _do_mag_ekf_update!(est, innovation, H, R_base, 1.0)

            push!(est.trace_chi2_before, chi2)
            push!(est.trace_chi2_after, chi2)
            push!(est.trace_heading_Reff, 1.0)
            push!(est.trace_interlock_fired, false)
        end
    end
end

"""
    _do_mag_ekf_update!(est, innovation, H, R_base, R_scale)

Execute the EKF measurement update for the magnetic field.
Scalar inflation version — used by Mode A (always scale=1).
"""
function _do_mag_ekf_update!(est::NavEstimator, innovation::SVector{3,Float64},
                              H::Matrix{Float64}, R_base::Matrix{Float64},
                              R_scale::Float64)
    R = R_base * R_scale^2

    S = H * est.P * H' + R
    K = est.P * H' / S

    # Apply correction
    dx = K * Vector(innovation)
    est.x .+= dx

    est.P = (I(N_STATES) - K * H) * est.P
    est.P = 0.5 * (est.P + est.P')
end

"""
    _do_mag_ekf_update_heading_protected!(est, innovation, H, R_base, inflation)

Anisotropic EKF update that partially protects heading observability.

Strategy: the heading column of H (H[:,5]) defines the direction in measurement
space that heading changes project onto. We apply MODERATE inflation (sqrt of
full inflation) along this direction and FULL inflation along the orthogonal
complement (position-sensitive directions).

This lets heading track slowly through source events (preventing unbounded
yaw drift) while strongly rejecting position contamination (preventing DNH
failures). The moderate heading inflation acknowledges that the elevator
field DOES corrupt heading information, but less so than position.
"""
function _do_mag_ekf_update_heading_protected!(est::NavEstimator, innovation::SVector{3,Float64},
                                                H::Matrix{Float64}, R_base::Matrix{Float64},
                                                inflation::Float64; heading_cap::Float64 = 4.0)
    h_psi = SVector(H[1,5], H[2,5], H[3,5])
    h_norm = sqrt(sum(h_psi .^ 2))

    if h_norm < 1e-10
        _do_mag_ekf_update!(est, innovation, H, R_base, inflation)
        return
    end

    d_heading = h_psi / h_norm

    # Heading direction: moderate inflation (sqrt of full)
    # Position directions: full inflation
    # Continuous adaptive heading inflation: scale heading R with
    # the heading-component contamination level. Smooth transition
    # from "heading protected" (low contamination) to "full scalar
    # inflation" (high contamination).
    #
    # h_component = innovation projected onto heading direction / σ
    # When h_component < 1: heading is clean → use heading_cap (moderate)
    # When h_component > 1: heading is contaminated → blend toward inflation
    # The blend gives: heading_inflation = heading_cap + (inflation - heading_cap) * α
    # where α = clamp((h_component - 1) / 4, 0, 1)
    h_component = abs(dot(d_heading, innovation)) / (sqrt(R_base[1,1]))
    # Sharp transition: if heading-sensitive innovation > 1σ, use full
    # scalar inflation for heading (no protection). Below 1σ, protect.
    target_heading = h_component > 1.0 ? inflation : heading_cap
    heading_inflation = min(sqrt(inflation), target_heading)
    σ2 = R_base[1,1]
    dv = Vector(d_heading)

    # R = σ² * inflation² * I  -  σ² * (inflation² - heading_inflation²) * d*d'
    # Along d_heading: σ² * heading_inflation²
    # Orthogonal:      σ² * inflation²
    R = σ2 * inflation^2 * I(3) - σ2 * (inflation^2 - heading_inflation^2) * (dv * dv')

    S = H * est.P * H' + R
    K = est.P * H' / S

    dx = K * Vector(innovation)

    # Clamp heading correction to prevent cumulative drift from
    # persistent strong sources. Position corrections are unclamped.
    max_heading_step = 0.01  # ~0.57 deg per 0.1s step
    dx[5] = clamp(dx[5], -max_heading_step, max_heading_step)

    est.x .+= dx

    est.P = (I(N_STATES) - K * H) * est.P
    est.P = 0.5 * (est.P + est.P')
end

"""
    _update_heading_soft!(est, B_ned_meas, noise_std)

Soft heading constraint from Earth field direction. Extracts heading from
the horizontal B components, then applies it as a very-high-R measurement
that prevents unbounded yaw drift without acting as a precise compass.

R is set to ~10° σ — enough to bound drift but not enough to override
the gyro during transient magnetic events. When the mag update in
update_magnetometer! detects a source (Mode B/C gating), this function
still applies but its large R means the correction is negligible.
"""
function _update_heading_soft!(est::NavEstimator, B_ned::SVector{3,Float64},
                                noise_std::Float64)
    # Extract heading from horizontal B components
    Bn, Be = EARTH_FIELD_NED[1], EARTH_FIELD_NED[2]
    Bx, By = B_ned[1], B_ned[2]

    # The B measurement is in NED frame (tilt-compensated), so the
    # horizontal components relate to heading as:
    # Bx_meas ≈ Bn + anomaly_x  (North component)
    # By_meas ≈ Be + anomaly_y  (East component)
    # No heading dependence for NED-frame measurements.
    # But: heading IS observable from the CONSISTENCY between
    # odometry-predicted position and mag-corrected position.
    #
    # For a direct heading measurement, we'd need body-frame B.
    # Since we're working in NED, heading comes indirectly from:
    # 1. Gyro integration (primary)
    # 2. Position-heading coupling via odometry (secondary)
    #
    # Apply a gentle heading regularization to prevent drift:
    # dψ/dt → 0 when no other heading info is available.
    # This is equivalent to a "heading random walk" prior.

    # Secondary heading aiding: extract yaw from B field direction.
    # This uses the SAME measurement but extracts different information.
    # The map update extracts position from field magnitude/gradient.
    # This extracts heading from field direction relative to Earth.
    #
    # Critical: this is NOT the primary heading source (gyro is).
    # The R is set large (~10°) so it only prevents long-term drift.
    # When elevator anomaly corrupts the field direction, the large R
    # means the corrupted heading barely moves the estimate.
    Bn, Be = EARTH_FIELD_NED[1], EARTH_FIELD_NED[2]
    B_horiz_earth = sqrt(Bn^2 + Be^2)

    # B_ned is already in NED frame — extract heading from comparison
    # with expected Earth horizontal field.
    # For NED-frame magnetometer, the measurement IS independent of heading.
    # So there's no heading information in a NED-frame measurement.
    #
    # Heading information requires BODY-frame measurements. Since we
    # tilt-compensated to NED, we lost heading sensitivity.
    #
    # The only heading info in the map-matching framework comes from
    # the F-matrix coupling: position corrections → heading corrections.
    # Accept this limitation — it's physically correct.
end

# ============================================================================
# Full mission simulation + estimation
# ============================================================================

"""
    run_nav_mission(world, trajectory, mode_config; seed, dt, sensor_noise)

Run a complete simulated navigation mission:
1. Seed the magnetic map from the static background field
2. Simulate sensor measurements through the world (with elevator)
3. Feed into the 5-state NavEstimator (gyro + odom + map-relative mag)
4. Return ground truth + estimated positions for error analysis

Sensor pipeline per timestep:
  gyro → predict! (yaw propagation)
  odom → update_odometry! (velocity from speed + heading)
  mag  → update_magnetometer! (position correction from map residual)

Returns a NamedTuple with all data needed for DOE metrics.
"""
function run_nav_mission(world, trajectory, mode_config;
                         seed::Int = 42,
                         dt::Float64 = 0.1,
                         sensor_noise::SensorNoise = SensorNoise())
    rng = MersenneTwister(seed)
    dur = ElevatorDOETrajectories.duration(trajectory)
    n_steps = max(1, floor(Int, dur / dt) + 1)

    # --- Create tetrahedron sensor ---
    tetra_sensor = TetrahedronSensor(sensor_noise.tetra_config; seed=seed)

    # --- Build magnetic map from static background ---
    # Simulates a previously-learned map (before elevator was active).
    # The map captures the persistent field the estimator can anchor to.
    mag_map = MagMap(tile_size=2.0)
    seed_map!(mag_map, world, trajectory)

    # --- Initialize estimator ---
    start_pos = ElevatorDOETrajectories.position(trajectory, 0.0)
    start_vel = ElevatorDOETrajectories.velocity(trajectory, 0.0)
    est = NavEstimator(start_pos, mode_config, mag_map)

    # Initialize heading from velocity direction
    start_speed = sqrt(start_vel[1]^2 + start_vel[2]^2)
    if start_speed > 0.01
        est.x[5] = atan(start_vel[2], start_vel[1])
    end
    # Initialize velocity
    est.x[3] = start_vel[1]
    est.x[4] = start_vel[2]

    # --- Output vectors ---
    timestamps = Float64[]
    true_positions = SVector{3,Float64}[]
    est_positions = SVector{3,Float64}[]
    innovations = Float64[]
    elev_positions = SVector{3,Float64}[]
    elev_velocities = Float64[]
    tile_updates = Float64[]

    for i in 1:n_steps
        t = (i - 1) * dt

        # --- Ground truth ---
        pos = ElevatorDOETrajectories.position(trajectory, t)
        vel = ElevatorDOETrajectories.velocity(trajectory, t)

        # --- Step world (elevator moves) ---
        if i > 1
            step!(world, dt)
        end

        # --- Record elevator state ---
        push!(elev_positions, world.elevators[1].position)
        push!(elev_velocities, world.elevators[1].velocity)

        # --- True magnetic field at pedestrian position ---
        # Earth + static building anomaly + elevator (transient)
        B_static = EARTH_FIELD_NED + static_background_field(pos)
        B_elevator = magnetic_field(world, pos) * 1e6  # T → µT
        B_ned = B_static + B_elevator

        # --- Generate noisy sensor measurements ---

        # Magnetometer: tetrahedral FTM sensor in BODY frame.
        # Heading information is preserved in the body-frame measurement.
        att = ElevatorDOETrajectories.attitude(trajectory, t)
        R_ned2body = att'

        # Create field function at pedestrian position for tetrahedron.
        # Evaluate the total field (Earth + static + elevator) at each of the
        # 17 Hall bar positions relative to the sensor center. The field is
        # evaluated at pos + r_sensor (world coords), then rotated to body frame.
        # At urban scales the 150mm sensor baseline resolves near-field gradients
        # (e.g. elevator at 1m) while far-field sources appear uniform.
        field_at_sensor(r_sensor) = begin
            # World position of this Hall bar
            world_offset = R_ned2body' * r_sensor  # body → NED
            sensor_world_pos = pos + SVector(world_offset[1], world_offset[2], world_offset[3])
            # Total field at that world position (µT)
            B_s = EARTH_FIELD_NED + static_background_field(sensor_world_pos) +
                  magnetic_field(world, sensor_world_pos) * 1e6
            # Rotate to body frame (T) — sensor works in Tesla internally
            return R_ned2body * B_s * 1e-6  # µT → T
        end
        nav_bandwidth = 10.0  # Hz
        raw_17 = simulate_measurement(tetra_sensor, field_at_sensor;
                                       add_noise=true, bandwidth_hz=nav_bandwidth)
        recon = reconstruct_tensor(tetra_sensor, raw_17;
                                   σ_meas=sensor_noise.σ_per_bar)

        # Reconstructed B₀ in body frame (T → µT for EKF)
        mag_meas_clean = SVector{3}(recon.B0) * 1e6  # T → µT

        # Add environmental noise (urban EMI not captured by sensor model)
        σ_env_µT = sensor_noise.σ_environmental * 1e6
        mag_meas = mag_meas_clean + SVector(
            σ_env_µT * randn(rng),
            σ_env_µT * randn(rng),
            σ_env_µT * randn(rng),
        )
        fit_residual = recon.fit_residual_normalized  # for Mode C source detection

        # Gyroscope: true yaw rate from heading change
        gyro_true_yaw = if t > dt
            vel_prev = ElevatorDOETrajectories.velocity(trajectory, t - dt)
            sp = sqrt(vel_prev[1]^2 + vel_prev[2]^2)
            sn = sqrt(vel[1]^2 + vel[2]^2)
            if sn > 0.01 && sp > 0.01
                yaw_now = atan(vel[2], vel[1])
                yaw_prev = atan(vel_prev[2], vel_prev[1])
                dyaw = atan(sin(yaw_now - yaw_prev), cos(yaw_now - yaw_prev))
                clamp(dyaw / dt, -2.0, 2.0)
            else
                0.0
            end
        else
            0.0
        end
        gyro_meas = gyro_true_yaw + sensor_noise.imu_gyro * randn(rng)

        # Odometry: forward speed (always positive)
        true_speed = sqrt(vel[1]^2 + vel[2]^2)
        speed_meas = max(0.0, true_speed + sensor_noise.odometry * randn(rng))

        # --- Run estimator pipeline ---
        # 1. Predict: propagate position via velocity, yaw via gyro
        predict!(est, gyro_meas, dt)

        # 2. Odometry: set velocity from speed + heading
        update_odometry!(est, speed_meas, sensor_noise.odometry)

        # 3. Magnetometer: map-relative position correction
        est_pos_sv = SVector(est.x[1], est.x[2], 0.0)
        update_magnetometer!(est, mag_meas, sensor_noise.magnetometer;
                              fit_residual=fit_residual)

        # Heading is observable through the body-frame mag update:
        # H[1:2, 5] couples heading to B_body via dR/dψ * B_ned.
        # No separate heading measurement needed.

        # --- Record ---
        push!(timestamps, t)
        push!(true_positions, pos)
        push!(est_positions, est_pos_sv)

        if !isempty(est.innovation_history)
            push!(innovations, est.innovation_history[end])
        end

        # Tile update proxy (position correction magnitude)
        err_xy = sqrt((est.x[1] - pos[1])^2 + (est.x[2] - pos[2])^2)
        push!(tile_updates, err_xy * 0.01)
    end

    return (
        timestamps = timestamps,
        true_positions = true_positions,
        est_positions = est_positions,
        innovations = innovations,
        elev_positions = elev_positions,
        elev_velocities = elev_velocities,
        tile_updates = tile_updates,
        world = world,
    )
end

# ============================================================================
# DOE callback functions
# ============================================================================

function doe_run_mission(world, trajectory, mode_config; seed::Int = 42)
    noise_scale = _noise_scale_value(NOISE_NOMINAL)  # default; overridden by DOE point if needed
    run_nav_mission(world, trajectory, mode_config;
                    seed=seed, sensor_noise=SensorNoise(scale=noise_scale))
end

function doe_compute_errors(result)
    n = length(result.true_positions)
    errors = Float64[]
    for i in 1:n
        d = result.true_positions[i] - result.est_positions[i]
        push!(errors, sqrt(sum(d .^ 2)))
    end
    return errors
end

function doe_compute_innovations(result)
    return result.innovations
end

function doe_compute_tile_updates(result)
    return result.tile_updates
end

function doe_extract_elevator_positions(result)
    return result.elev_positions
end

function doe_extract_elevator_velocities(result)
    return result.elev_velocities
end

function doe_extract_pedestrian_positions(result)
    return result.true_positions
end

function doe_compute_path_length(result)
    total = 0.0
    for i in 2:length(result.true_positions)
        d = result.true_positions[i] - result.true_positions[i-1]
        total += sqrt(sum(d .^ 2))
    end
    return total
end

# ============================================================================
# CSV export
# ============================================================================

function export_csv(results::Vector, path::String)
    mkpath(dirname(path))
    open(path, "w") do f
        println(f, "run_id,mode,archetype,speed,approach,dipole,shaft,noise,seed,rmse,p50,p90,max,dnh_ratio,burst_peak,contamination,pass")
        for r in results
            p = r.run.point
            m = r.metrics
            mode_str = Int(r.run.mode) == 1 ? "A" : Int(r.run.mode) == 2 ? "B" : "C"
            println(f, "$(r.run.run_id),$mode_str,$(p.archetype),$(p.elevator_speed)," *
                       "$(p.closest_approach),$(p.dipole_strength),$(p.shaft_geometry)," *
                       "$(p.sensor_noise_scale),$(r.run.seed)," *
                       "$(round(m.rmse, digits=4))," *
                       "$(round(m.p50_error, digits=4)),$(round(m.p90_error, digits=4))," *
                       "$(round(m.max_error, digits=4)),$(round(m.do_no_harm_ratio, digits=4))," *
                       "$(round(m.innovation_burst_peak, digits=4))," *
                       "$(round(m.map_contamination_score, digits=4)),$(r.pass)")
        end
    end
    println("Exported CSV to $path")
end

# ============================================================================
# Main
# ============================================================================

function main()
    args = ARGS
    run_practical = "--practical" in args || isempty(args)
    run_screening = "--screening" in args
    run_poisoning = "--poisoning" in args

    results_dir = joinpath(@__DIR__, "..", "results")
    mkpath(results_dir)

    all_results = []
    poisoning_results = []

    if run_practical
        println("=" ^ 60)
        println("ELEVATOR DOE — Practical First Design")
        println("=" ^ 60)
        design = create_practical_first_doe()
        total_runs = length(design.points) * length(design.modes) * design.n_seeds
        println("  $(length(design.points)) points × $(length(design.modes)) modes × $(design.n_seeds) seeds = $total_runs runs")
        println()

        results = run_elevator_doe!(
            design;
            run_mission_fn = doe_run_mission,
            compute_errors_fn = doe_compute_errors,
            compute_innovations_fn = doe_compute_innovations,
            compute_tile_updates_fn = doe_compute_tile_updates,
            extract_elevator_positions_fn = doe_extract_elevator_positions,
            extract_elevator_velocities_fn = doe_extract_elevator_velocities,
            extract_pedestrian_positions_fn = doe_extract_pedestrian_positions,
            compute_path_length_fn = doe_compute_path_length,
        )
        append!(all_results, results)

        n_pass = count(r -> r.pass, results)
        println("  Completed: $(length(results)) runs, $n_pass passed, $(length(results)-n_pass) failed")

        # Quick summary by mode
        for mode_val in [1, 2, 3]
            mode_name = mode_val == 1 ? "A (Baseline)" : mode_val == 2 ? "B (Robust)" : "C (Source-Aware)"
            mode_results = filter(r -> Int(r.run.mode) == mode_val, results)
            if !isempty(mode_results)
                p90s = [r.metrics.p90_error for r in mode_results]
                dnhs = [r.metrics.do_no_harm_ratio for r in mode_results]
                println("  Mode $mode_name: mean P90=$(round(mean(p90s), digits=3))m, " *
                        "mean DNH=$(round(mean(dnhs), digits=3)), " *
                        "pass=$(count(r->r.pass, mode_results))/$(length(mode_results))")
            end
        end
        println()
    end

    if run_screening
        println("=" ^ 60)
        println("ELEVATOR DOE — Screening Design (LHS)")
        println("=" ^ 60)
        design = create_screening_doe()
        total_runs = length(design.points) * length(design.modes) * design.n_seeds
        println("  $(length(design.points)) points × $(length(design.modes)) modes × $(design.n_seeds) seeds = $total_runs runs")

        results = run_elevator_doe!(
            design;
            run_mission_fn = doe_run_mission,
            compute_errors_fn = doe_compute_errors,
            compute_innovations_fn = doe_compute_innovations,
            compute_tile_updates_fn = doe_compute_tile_updates,
            extract_elevator_positions_fn = doe_extract_elevator_positions,
            extract_elevator_velocities_fn = doe_extract_elevator_velocities,
            extract_pedestrian_positions_fn = doe_extract_pedestrian_positions,
            compute_path_length_fn = doe_compute_path_length,
        )
        append!(all_results, results)

        n_pass = count(r -> r.pass, results)
        println("  Completed: $(length(results)) runs, $n_pass passed")
        println()
    end

    if run_poisoning
        println("Map poisoning test requires full map learning pipeline — skipping in current config")
    end

    if !isempty(all_results)
        # Generate report
        report = generate_elevator_doe_report(all_results, poisoning_results)
        report_path = joinpath(results_dir, "elevator_doe_report.md")
        export_elevator_report_md(report, report_path)
        println("Report: $report_path")

        # Export CSV
        csv_path = joinpath(results_dir, "elevator_doe_data.csv")
        export_csv(all_results, csv_path)
    else
        println("No results to report.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
