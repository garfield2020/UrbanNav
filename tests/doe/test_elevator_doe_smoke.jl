#!/usr/bin/env julia
# ============================================================================
# test_elevator_doe_smoke.jl - Smoke tests for elevator DOE
# ============================================================================
# Verifies that a single DOE point runs without crashing for each archetype.

using Test

# Include modules directly for standalone testing
include(joinpath(@__DIR__, "..", "..", "sim", "worlds", "ElevatorWorld.jl"))
include(joinpath(@__DIR__, "..", "..", "sim", "trajectories", "elevator_doe_trajectories.jl"))

using .ElevatorWorldModule
using .ElevatorDOETrajectories
using StaticArrays

# Include testing modules
include(joinpath(@__DIR__, "..", "..", "nav_core", "src", "testing", "elevator_mode_config.jl"))
include(joinpath(@__DIR__, "..", "..", "nav_core", "src", "testing", "elevator_doe_metrics.jl"))
include(joinpath(@__DIR__, "..", "..", "nav_core", "src", "testing", "elevator_map_poisoning.jl"))
include(joinpath(@__DIR__, "..", "..", "nav_core", "src", "testing", "elevator_doe.jl"))

using Random, LinearAlgebra

# Stub mission runner for smoke testing — uses real mag field + simple estimation
function smoke_run_mission(world, trajectory, mode_config; seed::Int = 42)
    rng = MersenneTwister(seed)
    dt = 0.1
    dur = ElevatorDOETrajectories.duration(trajectory)
    n_steps = max(1, floor(Int, dur / dt) + 1)

    true_pos = SVector{3,Float64}[]
    est_pos = SVector{3,Float64}[]
    innov = Float64[]
    elev_pos = SVector{3,Float64}[]
    elev_vel = Float64[]

    for i in 1:n_steps
        t = (i - 1) * dt
        pos = ElevatorDOETrajectories.position(trajectory, t)
        push!(true_pos, pos)
        if i > 1
            step!(world, dt)
        end

        # Query real magnetic field at pedestrian position
        B = magnetic_field(world, pos)
        B_mag = sqrt(sum(B .^ 2))

        # Mode-dependent error scaling using actual field strength
        noise_scale = if mode_config.mode == NAV_MODE_A_BASELINE
            1.0 + B_mag * 1e4
        elseif mode_config.mode == NAV_MODE_B_ROBUST_IGNORE
            1.0 + B_mag * 3e3
        else
            1.0 + B_mag * 5e2
        end

        push!(est_pos, pos + SVector(noise_scale*0.1*randn(rng),
                                      noise_scale*0.1*randn(rng),
                                      noise_scale*0.05*randn(rng)))
        push!(innov, B_mag / 0.1)
        push!(elev_pos, world.elevators[1].position)
        push!(elev_vel, world.elevators[1].velocity)
    end

    return (true_positions=true_pos, est_positions=est_pos,
            innovations=innov, elev_positions=elev_pos,
            elev_velocities=elev_vel, tile_updates=fill(0.01, n_steps),
            world=world)
end

function smoke_compute_errors(result)
    [sqrt(sum((result.true_positions[i] - result.est_positions[i]).^2))
     for i in 1:length(result.true_positions)]
end

@testset "Elevator DOE Smoke Tests" begin

    @testset "Trajectory archetypes construct and run" begin
        shaft = SVector(0.0, 0.0)

        trajectories = [
            ("CorridorParallel", CorridorParallel(shaft; offset=3.0)),
            ("PerpendicularCrossing", PerpendicularCrossing(shaft)),
            ("ShaftLoop", ShaftLoop(shaft; radius=5.0)),
            ("StopAndGo", StopAndGo(shaft; offset=3.0)),
            ("MultiFloorWalk", MultiFloorWalk(shaft)),
            ("DualShaftPath", DualShaftPath(shaft, SVector(8.0, 0.0))),
        ]

        for (name, traj) in trajectories
            @testset "$name" begin
                dur = ElevatorDOETrajectories.duration(traj)
                @test dur > 0.0
                @test isfinite(dur)

                # Position at start, middle, end
                p0 = ElevatorDOETrajectories.position(traj, 0.0)
                pm = ElevatorDOETrajectories.position(traj, dur / 2.0)
                pe = ElevatorDOETrajectories.position(traj, dur)
                @test all(isfinite, p0)
                @test all(isfinite, pm)
                @test all(isfinite, pe)

                # Velocity
                v0 = ElevatorDOETrajectories.velocity(traj, 0.0)
                @test all(isfinite, v0)
            end
        end
    end

    @testset "DOE world construction" begin
        world = create_doe_elevator_world(
            speed=1.5, dwell_time=15.0,
            shaft_positions=[SVector(0.0, 0.0)],
            dipole_moment=200.0,
        )
        @test length(world.elevators) == 1
        @test world.elevators[1].max_velocity == 1.5
        @test world.elevators[1].dwell_time == 15.0

        # Frozen world
        frozen = create_doe_elevator_world(frozen=true)
        @test frozen.elevators[1].dwell_time > 1e10
    end

    @testset "Practical DOE design" begin
        design = create_practical_first_doe()
        @test length(design.points) == 24  # 6 archetypes × 2 approach × 2 speed
        @test length(design.modes) == 3
        @test design.n_seeds == 3
    end

    @testset "Single DOE point runs for each archetype" begin
        archetypes = [:corridor, :perpendicular, :loop, :stop_and_go, :multi_floor, :dual_shaft]

        for arch in archetypes
            @testset "Archetype: $arch" begin
                geom = arch == :dual_shaft ? SHAFT_DUAL : SHAFT_SINGLE
                point = ElevatorDOEPoint(
                    ELEV_MEDIUM, STOP_NORMAL, APPROACH_MEDIUM, DIPOLE_NOMINAL,
                    geom, RICHNESS_STRAIGHT, NOISE_NOMINAL, arch,
                )

                doe_run = ElevatorDOERun(point, NAV_MODE_B_ROBUST_IGNORE, 42, 1)
                result = run_single_point(
                    doe_run;
                    run_mission_fn = smoke_run_mission,
                    compute_errors_fn = smoke_compute_errors,
                )

                @test result isa ElevatorDOEResult
                @test result.metrics.p50_error >= 0.0
                @test result.metrics.p90_error >= result.metrics.p50_error
                @test isfinite(result.metrics.do_no_harm_ratio)
            end
        end
    end

    @testset "Mode configurations" begin
        a = configure_mode_a()
        @test a.mode == NAV_MODE_A_BASELINE
        @test a.covariance_inflation_factor == 1.0

        b = configure_mode_b()
        @test b.mode == NAV_MODE_B_ROBUST_IGNORE
        @test b.chi2_gating_alpha == 0.01
        @test b.covariance_inflation_factor == 3.0

        c = configure_mode_c()
        @test c.mode == NAV_MODE_C_SOURCE_AWARE
        @test c.augmented_state_dipole == true
        @test c.freeze_tiles_near_source == true
    end
end

println("All smoke tests passed.")
