#!/usr/bin/env julia
# ============================================================================
# test_elevator_doe_determinism.jl - Determinism tests for elevator DOE
# ============================================================================
# Verifies that the same design + seed produces identical results.

using Test

include(joinpath(@__DIR__, "..", "..", "sim", "worlds", "ElevatorWorld.jl"))
include(joinpath(@__DIR__, "..", "..", "sim", "trajectories", "elevator_doe_trajectories.jl"))

using .ElevatorWorldModule
using .ElevatorDOETrajectories
using StaticArrays, Random, LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "nav_core", "src", "testing", "elevator_mode_config.jl"))
include(joinpath(@__DIR__, "..", "..", "nav_core", "src", "testing", "elevator_doe_metrics.jl"))
include(joinpath(@__DIR__, "..", "..", "nav_core", "src", "testing", "elevator_map_poisoning.jl"))
include(joinpath(@__DIR__, "..", "..", "nav_core", "src", "testing", "elevator_doe.jl"))

function det_run_mission(world, trajectory, mode_config; seed::Int = 42)
    rng = MersenneTwister(seed)
    dt = 0.1
    dur = duration(trajectory)
    n_steps = max(1, floor(Int, dur / dt) + 1)

    true_pos = SVector{3,Float64}[]
    est_pos = SVector{3,Float64}[]

    for i in 1:n_steps
        t = (i - 1) * dt
        pos = position(trajectory, t)
        push!(true_pos, pos)
        if i > 1
            step!(world, dt)
        end
        B = magnetic_field(world, pos)
        B_mag = sqrt(sum(B .^ 2))
        scale = 1.0 + B_mag * 1e3
        push!(est_pos, pos + SVector(scale*0.1*randn(rng), scale*0.1*randn(rng), scale*0.05*randn(rng)))
    end

    return (true_positions=true_pos, est_positions=est_pos, world=world)
end

function det_compute_errors(result)
    [sqrt(sum((result.true_positions[i] - result.est_positions[i]).^2))
     for i in 1:length(result.true_positions)]
end

@testset "Elevator DOE Determinism" begin

    @testset "Same seed produces identical results" begin
        point = ElevatorDOEPoint(
            ELEV_MEDIUM, STOP_NORMAL, APPROACH_MEDIUM, DIPOLE_NOMINAL,
            SHAFT_SINGLE, RICHNESS_STRAIGHT, NOISE_NOMINAL, :corridor,
        )

        run1 = ElevatorDOERun(point, NAV_MODE_B_ROBUST_IGNORE, 42, 1)
        result1 = run_single_point(run1;
            run_mission_fn=det_run_mission, compute_errors_fn=det_compute_errors)

        run2 = ElevatorDOERun(point, NAV_MODE_B_ROBUST_IGNORE, 42, 2)
        result2 = run_single_point(run2;
            run_mission_fn=det_run_mission, compute_errors_fn=det_compute_errors)

        @test result1.metrics.p50_error == result2.metrics.p50_error
        @test result1.metrics.p90_error == result2.metrics.p90_error
        @test result1.metrics.max_error == result2.metrics.max_error
        @test result1.metrics.do_no_harm_ratio == result2.metrics.do_no_harm_ratio
        @test result1.pass == result2.pass
    end

    @testset "Different seed produces different results" begin
        point = ElevatorDOEPoint(
            ELEV_FAST, STOP_FREQUENT, APPROACH_NEAR, DIPOLE_STRONG,
            SHAFT_SINGLE, RICHNESS_LOOP, NOISE_NOMINAL, :loop,
        )

        run1 = ElevatorDOERun(point, NAV_MODE_A_BASELINE, 42, 1)
        result1 = run_single_point(run1;
            run_mission_fn=det_run_mission, compute_errors_fn=det_compute_errors)

        run2 = ElevatorDOERun(point, NAV_MODE_A_BASELINE, 999, 2)
        result2 = run_single_point(run2;
            run_mission_fn=det_run_mission, compute_errors_fn=det_compute_errors)

        # Very unlikely to be identical with different seeds
        @test result1.metrics.p50_error != result2.metrics.p50_error
    end

    @testset "ElevatorWorld is deterministic" begin
        w1 = create_doe_elevator_world(speed=2.0, dwell_time=5.0, seed=77)
        w2 = create_doe_elevator_world(speed=2.0, dwell_time=5.0, seed=77)

        for _ in 1:100
            step!(w1, 0.1)
            step!(w2, 0.1)
        end

        @test w1.elevators[1].position == w2.elevators[1].position
        @test w1.elevators[1].velocity == w2.elevators[1].velocity
        @test w1.elevators[1].phase == w2.elevators[1].phase
    end
end

println("All determinism tests passed.")
