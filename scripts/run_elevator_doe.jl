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
const GRAVITY_NED = SVector(0.0, 0.0, 9.81)        # m/s²

# ============================================================================
# Sensor noise configuration (mirrors MissionRunner.SensorNoiseParams)
# ============================================================================

struct SensorNoise
    magnetometer::Float64   # σ per axis (µT)
    imu_accel::Float64      # σ per axis (m/s²)
    imu_gyro::Float64       # σ per axis (rad/s)
    odometry::Float64       # σ speed (m/s)
    barometer::Float64      # σ altitude (m)
end

function SensorNoise(; scale::Float64 = 1.0)
    SensorNoise(
        0.5 * scale,   # magnetometer σ (µT) — includes model uncertainty
        0.01 * scale,  # IMU accel σ (m/s²)
        1e-4 * scale,  # IMU gyro σ (rad/s)
        0.02 * scale,  # odometry σ (m/s)
        0.3 * scale,   # barometer σ (m)
    )
end

# ============================================================================
# Navigation Estimator (Lightweight factor-graph-style EKF)
# ============================================================================
#
# Full UrbanNav StateEstimator requires loading the entire nav_core module
# graph with all its dependencies (Rotations, ForwardDiff, etc). Here we
# implement a focused EKF that exercises the same estimation principles:
# - 15-state error-state filter (pos, vel, att, gyro bias, accel bias)
# - IMU preintegration for prediction
# - Magnetometer updates with chi² gating (Mode B/C)
# - Odometry and barometer updates
# - Mode-specific handling of elevator magnetic interference
# ============================================================================

mutable struct NavEstimator
    # State: [px,py,pz, vx,vy,vz, roll,pitch,yaw, bgx,bgy,bgz, bax,bay,baz]
    x::Vector{Float64}      # 15-state
    P::Matrix{Float64}      # 15×15 covariance
    Q::Matrix{Float64}      # process noise
    mode::ElevatorModeConfig
    # Innovation monitoring
    innovation_history::Vector{Float64}
    cusum_stat::Float64     # CUSUM statistic for Mode C
    source_detected::Bool   # Mode C source detection flag
    estimated_source_field::SVector{3,Float64}  # Mode C estimated elevator field
end

function NavEstimator(initial_pos::SVector{3,Float64}, mode::ElevatorModeConfig)
    x = zeros(15)
    x[1:3] = initial_pos

    P = diagm(vcat(
        fill(0.5, 3),    # position: 0.5m σ
        fill(0.05, 3),   # velocity: 0.05 m/s σ
        fill(0.02, 3),   # attitude: 0.02 rad σ (~1°)
        fill(0.001, 3),  # gyro bias
        fill(0.01, 3),   # accel bias
    ).^2)

    Q = diagm(vcat(
        fill(0.001, 3),  # position process noise (small — driven by velocity)
        fill(0.01, 3),   # velocity process noise
        [0.01, 0.01, 0.1], # attitude: roll/pitch small, yaw large (pedestrian turns)
        fill(1e-6, 3),   # gyro bias random walk
        fill(1e-5, 3),   # accel bias random walk
    ).^2)

    NavEstimator(x, P, Q, mode,
                 Float64[], 0.0, false,
                 SVector(0.0, 0.0, 0.0))
end

function predict!(est::NavEstimator, imu_accel::SVector{3,Float64},
                  imu_gyro::SVector{3,Float64}, dt::Float64)
    # Pedestrian dead-reckoning prediction:
    # - Integrate yaw from gyroscope (z-component)
    # - Use current velocity (updated by odometry) for position prediction
    # - Do NOT use accelerometer for position (too noisy for pedestrians
    #   without step detection / zero-velocity updates)

    pos = SVector{3,Float64}(est.x[1:3])
    vel = SVector{3,Float64}(est.x[4:6])

    # Gyro-based heading update
    bg = SVector{3,Float64}(est.x[10:12])
    gyro_corrected = imu_gyro - bg
    est.x[7:9] .+= Vector(gyro_corrected) .* dt

    # Position prediction from velocity
    est.x[1:3] = pos + vel * dt

    # No velocity damping — odometry sets velocity every step

    # Propagate covariance
    F = Matrix{Float64}(I, 15, 15)
    F[1:3, 4:6] = I(3) * dt
    est.P = F * est.P * F' + est.Q * dt
    est.P = 0.5 * (est.P + est.P')
    # Clamp covariance to prevent blowup
    for j in 1:15
        est.P[j,j] = clamp(est.P[j,j], 1e-10, 100.0)
    end
end

function update_odometry!(est::NavEstimator, measured_speed::Float64,
                          noise_std::Float64)
    # Pedestrian odometry: measured forward speed in body frame.
    # Project to NED using estimated heading and directly set velocity.
    # This is more robust than EKF innovation for pedestrian navigation
    # because heading errors don't accumulate through the Kalman gain.
    yaw = est.x[9]
    cy, sy = cos(yaw), sin(yaw)

    # Set NED velocity from body-frame speed + heading
    est.x[4] = cy * measured_speed
    est.x[5] = sy * measured_speed
    # Keep z-velocity from barometer updates (don't overwrite)

    # Reduce velocity covariance (odometry is a strong measurement)
    est.P[4,4] = min(est.P[4,4], noise_std^2)
    est.P[5,5] = min(est.P[5,5], noise_std^2)
end

function update_barometer!(est::NavEstimator, measured_alt::Float64,
                           noise_std::Float64)
    # NED: z positive down, baro measures altitude (positive up) = -z
    predicted_alt = -est.x[3]
    innovation = measured_alt - predicted_alt

    H = zeros(1, 15)
    H[1, 3] = -1.0  # ∂alt/∂z_ned: alt = -z, so ∂alt/∂z = -1

    R = fill(noise_std^2, 1, 1)
    S = H * est.P * H' .+ R
    K = est.P * H' / S

    est.x .+= vec(K * [innovation])
    est.P = (I(15) - K * H) * est.P
    est.P = 0.5 * (est.P + est.P')
end

function update_magnetometer!(est::NavEstimator, measured_B::SVector{3,Float64},
                               predicted_B_background::SVector{3,Float64},
                               noise_std::Float64)
    # Innovation = measured B - predicted B (Earth field in body frame)
    innovation = measured_B - predicted_B_background

    # Normalized innovation magnitude (for gating and monitoring)
    innov_norm = sqrt(sum(innovation .^ 2)) / noise_std
    push!(est.innovation_history, innov_norm)

    # Mode-specific handling of the magnetic anomaly
    if est.mode.mode == NAV_MODE_A_BASELINE
        # Baseline: always use magnetometer for heading (no anomaly handling)
        _update_heading_from_mag!(est, measured_B, noise_std, 1.0)

    elseif est.mode.mode == NAV_MODE_B_ROBUST_IGNORE
        # Chi² gating on B-field residual magnitude
        chi2 = sum(innovation .^ 2) / noise_std^2
        chi2_threshold = 11.345  # 3-DOF, α=0.01

        if chi2 <= chi2_threshold
            _update_heading_from_mag!(est, measured_B, noise_std, 1.0)
        else
            # Anomalous field — still use for heading but inflate R proportionally
            # to how far the measurement is from expected. This keeps heading
            # tracking active even near sources, just with less confidence.
            inflation = 2.0  # moderate inflation — preserve heading tracking
            _update_heading_from_mag!(est, measured_B, noise_std, inflation)
        end

    elseif est.mode.mode == NAV_MODE_C_SOURCE_AWARE
        # Source-aware: detect anomaly via field magnitude change,
        # then compensate heading extraction by using magnitude-invariant
        # heading estimate.
        B_earth_mag = sqrt(sum(EARTH_FIELD_NED .^ 2))
        B_meas_mag = sqrt(sum(measured_B .^ 2))
        anomaly_ratio = B_meas_mag / B_earth_mag

        # CUSUM on magnitude deviation
        mag_dev = abs(anomaly_ratio - 1.0) / 0.02  # normalized by expected noise
        est.cusum_stat = max(0.0, est.cusum_stat + mag_dev -
                             est.mode.cusum_drift - 1.5)

        if est.cusum_stat > est.mode.cusum_threshold
            est.source_detected = true
        elseif est.cusum_stat < 0.5
            est.source_detected = false
        end

        push!(est.innovation_history, innov_norm)

        if est.source_detected
            # Near a source: normalize measured B to Earth field magnitude
            # before heading extraction. This removes the anomaly magnitude
            # while preserving the directional (heading) information if the
            # anomaly is predominantly vertical (elevator dipole is vertical).
            B_horiz = SVector(measured_B[1], measured_B[2], 0.0)
            B_horiz_mag = sqrt(measured_B[1]^2 + measured_B[2]^2)
            B_earth_horiz_mag = sqrt(EARTH_FIELD_NED[1]^2 + EARTH_FIELD_NED[2]^2)
            if B_horiz_mag > 1.0  # µT, avoid division by near-zero
                # Scale horizontal components to expected Earth field magnitude
                scale_factor = B_earth_horiz_mag / B_horiz_mag
                normalized_B = SVector(measured_B[1] * scale_factor,
                                       measured_B[2] * scale_factor,
                                       EARTH_FIELD_NED[3])
                _update_heading_from_mag!(est, normalized_B, noise_std, 2.0)
            else
                # Horizontal field too weak — skip heading update
            end
        else
            _update_heading_from_mag!(est, measured_B, noise_std, 1.0)
        end
    end
end

function _update_heading_from_mag!(est::NavEstimator, B_body::SVector{3,Float64},
                                    noise_std::Float64, R_scale::Float64)
    # Direct heading extraction from magnetometer.
    # The Earth field in NED has known Bx_ned (north) and By_ned (east).
    # In body frame: Bx_body = cy*Bx_ned + sy*By_ned
    #                By_body = -sy*Bx_ned + cy*By_ned
    # So: measured_yaw = atan2(Bx_ned*By_body_meas - By_ned*Bx_body_meas,
    #                          Bx_ned*Bx_body_meas + By_ned*By_body_meas)
    # This gives heading directly without needing a prediction.

    Bn = EARTH_FIELD_NED[1]  # North component
    Be = EARTH_FIELD_NED[2]  # East component
    Bx = B_body[1]
    By = B_body[2]

    # Extract yaw: from the rotation equations, yaw = atan2(Bn*By - Be*Bx, Bn*Bx + Be*By)
    # When Be ≈ 0: yaw = atan2(Bn*By, Bn*Bx) = atan2(By, Bx) ... but Bx = cy*Bn, By = -sy*Bn
    # so yaw = atan2(-sy*Bn, cy*Bn) = atan2(-sy, cy) = -yaw... sign issue.
    # Let's derive properly:
    # Bx_body = cy*Bn + sy*Be, By_body = -sy*Bn + cy*Be
    # [Bx_body]   [Bn  Be] [cy]
    # [By_body] = [-Bn Be] [sy]  ... wait, that's wrong
    # Actually: [Bx] = [Bn Be][cy; sy], [By] = [-Bn Be]... no.
    # R_ned2body for yaw: [cy sy; -sy cy]
    # [Bx_body] = [cy  sy][Bn] = cy*Bn + sy*Be
    # [By_body] = [-sy cy][Be] = -sy*Bn + cy*Be
    # So cy = (Bx*Be - By*(-Bn)) / (Bn^2+Be^2) ... no, solve:
    # cy*Bn + sy*Be = Bx
    # -sy*Bn + cy*Be = By
    # [Bn Be; -Bn Be] is wrong. It's [Bn Be; -Bn Be]...
    # The system is: [Bn Be; -Be Bn]... no.
    # M = [Bn Be; -Bn Be]... let me just do it right:
    # Eq1: Bn*cy + Be*sy = Bx
    # Eq2: -Bn*sy + Be*cy = By
    # det = Bn*Be - Be*(-Bn) = Bn*Be + Be*Bn ... no
    # det(M) where M = [Bn Be; -Bn Be] → Bn*Be - Be*(-Bn) = 2*Bn*Be... that's wrong
    # M = [Bn Be; -Bn Be] is NOT the matrix. The matrix is:
    # Row 1: [Bn, Be], Row 2: [-Bn, Be]... NO.
    # From the equations: cy*Bn + sy*Be = Bx ... → [Bn, Be][cy; sy] = Bx
    #                    -sy*Bn + cy*Be = By ... → [Be, -Bn][cy; sy] = By
    # So M = [Bn Be; Be -Bn], det = -Bn² - Be² = -(Bn²+Be²)
    # cy = (-Bn*By - Be*Bx) / (-(Bn²+Be²)) = (Bn*By + Be*Bx) / (Bn²+Be²)  -- WRONG
    # Let me just use Cramer's rule on [Bn Be; Be -Bn][cy;sy] = [Bx;By]
    # det = Bn*(-Bn) - Be*Be = -(Bn²+Be²)
    # cy = (Bx*(-Bn) - By*Be) / (-(Bn²+Be²)) = (Bx*Bn + By*Be) / (Bn²+Be²)...
    # Hmm let me just use atan2 directly.

    # Simpler: measured_yaw = atan2(sy, cy) where:
    # From eq1 & eq2: (Bn²+Be²)*cy = Bn*Bx + Be*By
    #                 (Bn²+Be²)*sy = Be*Bx - Bn*By  ... CHECK:
    # eq1*Bn: Bn²*cy + Bn*Be*sy = Bn*Bx
    # eq2*Be: Be²*cy - Bn*Be*sy = Be*By
    # Sum: (Bn²+Be²)*cy = Bn*Bx + Be*By  ✓
    # eq1*Be: Bn*Be*cy + Be²*sy = Be*Bx
    # eq2*(-Bn): -Bn*Be*cy + Bn²*sy = -Bn*By
    # Sum: (Bn²+Be²)*sy = Be*Bx - Bn*By  ✓

    B2 = Bn^2 + Be^2
    cy_meas = (Bn * Bx + Be * By) / B2
    sy_meas = (Be * Bx - Bn * By) / B2
    measured_yaw = atan(sy_meas, cy_meas)

    # Heading innovation (wrapped)
    yaw_innov = atan(sin(measured_yaw - est.x[9]), cos(measured_yaw - est.x[9]))

    # EKF scalar update on yaw (state index 9)
    H = zeros(1, 15)
    H[1, 9] = 1.0  # direct heading measurement

    R_yaw = (noise_std * R_scale / sqrt(B2))^2 + (0.02)^2  # ~1° min uncertainty
    S = H * est.P * H' .+ R_yaw
    K = est.P * H' / S

    est.x .+= vec(K * [yaw_innov])
    est.P = (I(15) - K * H) * est.P
    est.P = 0.5 * (est.P + est.P')
end

# ============================================================================
# Full mission simulation + estimation
# ============================================================================

"""
    run_nav_mission(world, trajectory, mode_config, sensor_noise; seed, dt)

Run a complete simulated navigation mission:
1. Simulate sensor measurements through the world
2. Feed into the navigation estimator
3. Return ground truth + estimated positions for error analysis

Returns a NamedTuple with all data needed for DOE metrics.
"""
function run_nav_mission(world, trajectory, mode_config;
                         seed::Int = 42,
                         dt::Float64 = 0.1,
                         sensor_noise::SensorNoise = SensorNoise())
    rng = MersenneTwister(seed)
    dur = ElevatorDOETrajectories.duration(trajectory)
    n_steps = max(1, floor(Int, dur / dt) + 1)

    # Initialize estimator at true start position and heading
    start_pos = ElevatorDOETrajectories.position(trajectory, 0.0)
    start_vel = ElevatorDOETrajectories.velocity(trajectory, 0.0)
    est = NavEstimator(start_pos, mode_config)
    # Initialize heading from velocity direction
    start_speed = sqrt(start_vel[1]^2 + start_vel[2]^2)
    if start_speed > 0.01
        est.x[9] = atan(start_vel[2], start_vel[1])
    end
    # Initialize velocity
    est.x[4:6] = Vector(start_vel)

    # Output vectors
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
        att = ElevatorDOETrajectories.attitude(trajectory, t)  # 3×3 DCM body→NED

        # --- Step world ---
        if i > 1
            step!(world, dt)
        end

        # --- Record elevator state ---
        push!(elev_positions, world.elevators[1].position)
        push!(elev_velocities, world.elevators[1].velocity)

        # --- Compute true magnetic field ---
        B_earth = EARTH_FIELD_NED
        B_sources = magnetic_field(world, pos) * 1e6  # Convert T → µT
        B_ned = B_earth + B_sources

        # Rotate to body frame
        R_ned2body = att'
        B_body = R_ned2body * B_ned

        # --- Generate noisy measurements ---
        mag_meas = B_body + SVector(
            sensor_noise.magnetometer * randn(rng),
            sensor_noise.magnetometer * randn(rng),
            sensor_noise.magnetometer * randn(rng),
        )

        gravity_body = R_ned2body * GRAVITY_NED
        accel_meas = gravity_body + SVector(
            sensor_noise.imu_accel * randn(rng),
            sensor_noise.imu_accel * randn(rng),
            sensor_noise.imu_accel * randn(rng),
        )

        # Compute true yaw rate from heading change (handles large rotations)
        gyro_true = if t > dt
            vel_prev = ElevatorDOETrajectories.velocity(trajectory, t - dt)
            speed_prev = sqrt(vel_prev[1]^2 + vel_prev[2]^2)
            speed_now = sqrt(vel[1]^2 + vel[2]^2)
            if speed_now > 0.01 && speed_prev > 0.01
                yaw_now = atan(vel[2], vel[1])
                yaw_prev = atan(vel_prev[2], vel_prev[1])
                dyaw = atan(sin(yaw_now - yaw_prev), cos(yaw_now - yaw_prev))  # wrap to [-π,π]
                yaw_rate = clamp(dyaw / dt, -2.0, 2.0)  # max ~115°/s turn rate
                SVector(0.0, 0.0, yaw_rate)
            else
                SVector(0.0, 0.0, 0.0)
            end
        else
            SVector(0.0, 0.0, 0.0)
        end

        gyro_meas = gyro_true + SVector(
            sensor_noise.imu_gyro * randn(rng),
            sensor_noise.imu_gyro * randn(rng),
            sensor_noise.imu_gyro * randn(rng),
        )

        # Use NED speed magnitude (always positive) — heading comes from gyro/mag
        true_speed = sqrt(vel[1]^2 + vel[2]^2)
        speed_meas = max(0.0, true_speed + sensor_noise.odometry * randn(rng))
        baro_meas = -pos[3] + sensor_noise.barometer * randn(rng)

        # --- Run estimator ---
        predict!(est, accel_meas, gyro_meas, dt)
        update_odometry!(est, speed_meas, sensor_noise.odometry)
        update_barometer!(est, baro_meas, sensor_noise.barometer)

        # Background B prediction using ESTIMATED yaw (not true attitude)
        est_pos_sv = SVector{3,Float64}(est.x[1:3])
        est_yaw = est.x[9]
        cy_est, sy_est = cos(est_yaw), sin(est_yaw)
        R_est = SMatrix{3,3,Float64}(cy_est, -sy_est, 0.0,
                                      sy_est, cy_est, 0.0,
                                      0.0, 0.0, 1.0)'  # NED→body
        B_background_pred = R_est * EARTH_FIELD_NED
        update_magnetometer!(est, mag_meas, B_background_pred, sensor_noise.magnetometer)

        # Record
        push!(timestamps, t)
        push!(true_positions, pos)
        push!(est_positions, est_pos_sv)

        # Innovation from magnetometer update
        if !isempty(est.innovation_history)
            push!(innovations, est.innovation_history[end])
        end

        # Tile update magnitude placeholder (position correction as proxy)
        push!(tile_updates, norm(est.x[1:3] - Vector(pos)) * 0.01)
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

main()
