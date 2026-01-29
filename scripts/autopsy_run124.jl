#!/usr/bin/env julia
# ============================================================================
# autopsy_run124.jl — Single-run failure trace for DOE run 124
#
# Replays run 124 (stop_and_go, APPROACH_NEAR, ELEV_FAST, Mode C)
# with full introspection to classify the failure mechanism:
#   A: bad subtraction (χ² increases after update)
#   B: heading leakage (heading-sensitive projection large, heading R tight)
#   C: dynamics lag (subtraction helps then fails during fast segments)
# ============================================================================

include(joinpath(@__DIR__, "run_elevator_doe.jl"))

using Statistics
using Printf

function run_autopsy()
    # Reconstruct run 124's exact configuration
    # DOE: 24 points × 3 modes × 3 seeds. Run 124 = ...
    # From the DOE runner: run_id increments per seed, so:
    #   point 1: runs 1-9 (3 modes × 3 seeds)
    #   point N: runs 9*(N-1)+1 to 9*N
    # Run 124: ceil(124/9) = point 14, within that: ((124-1)%9)+1 = 7th sub-run
    # Sub-run 7: mode_idx = ceil(7/3) = 3 → Mode C, seed_idx = ((7-1)%3)+1 = 1
    #
    # But let's reconstruct from the DOE framework directly.

    design = create_practical_first_doe()

    # Find the exact run configuration
    run_id = 0
    target_point = nothing
    target_mode = nothing
    target_seed = 0

    for point in design.points
        for mode in design.modes
            for seed_idx in 1:design.n_seeds
                run_id += 1
                if run_id == 124
                    target_point = point
                    target_mode = mode
                    target_seed = 1000 * run_id + seed_idx
                    break
                end
            end
            run_id == 124 && break
        end
        run_id == 124 && break
    end

    println("=" ^ 70)
    println("AUTOPSY — Run 124")
    println("=" ^ 70)
    println("  Archetype:  $(target_point.archetype)")
    println("  Approach:   $(target_point.closest_approach)")
    println("  Speed:      $(target_point.elevator_speed)")
    println("  Mode:       $(target_mode)")
    println("  Seed:       $target_seed")
    println()

    # Run with elevator
    mode_config = configure_elevator_mode(target_mode)
    traj = build_trajectory(target_point)
    world_active = build_elevator_world(target_point; seed=target_seed)

    result_with = run_nav_mission(world_active, traj, mode_config;
                                   seed=target_seed,
                                   sensor_noise=SensorNoise(scale=1.0))

    # Run without elevator (frozen)
    world_frozen = build_control_world(target_point; seed=target_seed)
    result_without = run_nav_mission(world_frozen, traj, mode_config;
                                      seed=target_seed,
                                      sensor_noise=SensorNoise(scale=1.0))

    # Compute errors
    errors_with = doe_compute_errors(result_with)
    errors_without = doe_compute_errors(result_without)

    p90_with = length(errors_with) > 0 ? sort(errors_with)[ceil(Int, 0.9*length(errors_with))] : 0.0
    p90_without = length(errors_without) > 0 ? sort(errors_without)[ceil(Int, 0.9*length(errors_without))] : 0.0
    dnh = p90_without > 0 ? p90_with / p90_without : 1.0

    println("METRICS:")
    println("  P90 with elevator:    $(round(p90_with, digits=3)) m")
    println("  P90 without elevator: $(round(p90_without, digits=3)) m")
    println("  DNH ratio:            $(round(dnh, digits=4))")
    println("  RMSE with:            $(round(sqrt(mean(errors_with .^ 2)), digits=3)) m")
    println("  RMSE without:         $(round(sqrt(mean(errors_without .^ 2)), digits=3)) m")
    println()

    # Extract the estimator from the active run — it's embedded in the mission result
    # We need to re-run with access to the estimator internals.
    # Let's do a manual replay to get the trace.

    rng = MersenneTwister(target_seed)
    dur = ElevatorDOETrajectories.duration(traj)
    dt = 0.1
    n_steps = max(1, floor(Int, dur / dt) + 1)

    mag_map = MagMap(tile_size=2.0)
    world2 = build_elevator_world(target_point; seed=target_seed)
    seed_map!(mag_map, world2, traj)

    start_pos = ElevatorDOETrajectories.position(traj, 0.0)
    start_vel = ElevatorDOETrajectories.velocity(traj, 0.0)
    est = NavEstimator(start_pos, mode_config, mag_map)
    start_speed = sqrt(start_vel[1]^2 + start_vel[2]^2)
    if start_speed > 0.01
        est.x[5] = atan(start_vel[2], start_vel[1])
    end
    est.x[3] = start_vel[1]
    est.x[4] = start_vel[2]

    noise = SensorNoise(scale=1.0)

    # Introspection data
    println("TIMESTEP TRACE (filtered to high-chi² events):")
    println("-" ^ 120)
    println("  t(s)   | pos_err(m) | χ²_before | χ²_after | heading_proj | Rψ_eff | interlock | source | elev_v | elev_dist")
    println("-" ^ 120)

    for i in 1:n_steps
        t = (i - 1) * dt
        pos = ElevatorDOETrajectories.position(traj, t)
        vel = ElevatorDOETrajectories.velocity(traj, t)

        if i > 1
            step!(world2, dt)
        end

        # Sensor simulation (must match run_nav_mission exactly)
        B_static = EARTH_FIELD_NED + static_background_field(pos)
        B_elevator = magnetic_field(world2, pos) * 1e6
        B_ned = B_static + B_elevator
        att = ElevatorDOETrajectories.attitude(traj, t)
        R_ned2body = att'
        B_body = R_ned2body * B_ned
        mag_meas = B_body + SVector(
            noise.magnetometer * randn(rng),
            noise.magnetometer * randn(rng),
            noise.magnetometer * randn(rng),
        )

        gyro_true_yaw = if t > dt
            vel_prev = ElevatorDOETrajectories.velocity(traj, t - dt)
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
        gyro_meas = gyro_true_yaw + noise.imu_gyro * randn(rng)
        true_speed = sqrt(vel[1]^2 + vel[2]^2)
        speed_meas = max(0.0, true_speed + noise.odometry * randn(rng))

        # Run estimator
        predict!(est, gyro_meas, dt)
        update_odometry!(est, speed_meas, noise.odometry)
        update_magnetometer!(est, mag_meas, noise.magnetometer)

        # Position error
        pos_err = sqrt((est.x[1] - pos[1])^2 + (est.x[2] - pos[2])^2)

        # Elevator state
        elev_pos = world2.elevators[1].position
        elev_vel = world2.elevators[1].velocity
        elev_dist = sqrt((pos[1] - elev_pos[1])^2 + (pos[2] - elev_pos[2])^2)

        # Print trace for high-chi² events or high error
        n_trace = length(est.trace_chi2_before)
        if n_trace > 0
            chi2_b = est.trace_chi2_before[end]
            chi2_a = est.trace_chi2_after[end]
            h_proj = length(est.trace_heading_proj) > 0 ? est.trace_heading_proj[end] : 0.0
            r_eff = est.trace_heading_Reff[end]
            intlk = length(est.trace_interlock_fired) > 0 ? est.trace_interlock_fired[end] : false
            src = length(est.trace_source_detected) > 0 ? est.trace_source_detected[end] : false

            if chi2_b > 11.345 || pos_err > 5.0
                @printf("  %6.1f  | %9.3f  | %9.1f | %8.1f | %11.2f  | %6.2f | %9s | %6s | %6.2f | %9.2f\n",
                        t, pos_err, chi2_b, chi2_a, h_proj, r_eff,
                        intlk ? "YES" : "no", src ? "YES" : "no",
                        elev_vel, elev_dist)
            end
        end
    end

    # Summary statistics
    println()
    println("TRACE SUMMARY:")
    n_high_chi2 = count(x -> x > 11.345, est.trace_chi2_before)
    n_interlock = count(est.trace_interlock_fired)
    n_source = count(est.trace_source_detected)
    high_heading = filter(x -> x > 2.0, est.trace_heading_proj)

    println("  Total timesteps:     $n_steps")
    println("  High χ² events:     $n_high_chi2")
    println("  Interlock fired:    $n_interlock")
    println("  Source detected:    $n_source")
    println("  Heading proj > 2σ: $(length(high_heading))")
    if !isempty(high_heading)
        println("  Max heading proj:   $(round(maximum(high_heading), digits=2))σ")
    end

    # Classify failure
    println()
    println("FAILURE CLASSIFICATION:")
    if n_interlock > 0
        println("  → Type A (bad subtraction): interlock caught $n_interlock harmful updates")
    elseif !isempty(high_heading) && maximum(high_heading) > 3.0
        println("  → Type B (heading leakage): heading-sensitive innovation reached $(round(maximum(high_heading), digits=1))σ")
    else
        println("  → Type C (dynamics lag): source detection may lag fast elevator changes")
    end
end

run_autopsy()
