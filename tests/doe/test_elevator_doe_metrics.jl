#!/usr/bin/env julia
# ============================================================================
# test_elevator_doe_metrics.jl - Metric computation tests
# ============================================================================
# Tests metric computation on known data: do-no-harm ratio, segment
# classification, contamination score.

using Test

include(joinpath(@__DIR__, "..", "..", "nav_core", "src", "testing", "elevator_mode_config.jl"))
include(joinpath(@__DIR__, "..", "..", "nav_core", "src", "testing", "elevator_doe_metrics.jl"))

using StaticArrays

@testset "Elevator DOE Metrics" begin

    @testset "Do-no-harm ratio" begin
        # Identical errors → ratio = 1.0
        errors = collect(range(0.1, 1.0, length=100))
        @test compute_do_no_harm(errors, errors) ≈ 1.0

        # Worse with elevator → ratio > 1.0
        errors_worse = errors .* 1.2
        ratio = compute_do_no_harm(errors_worse, errors)
        @test ratio > 1.0
        @test ratio ≈ 1.2 atol=0.05

        # Better with elevator (unlikely but valid) → ratio < 1.0
        errors_better = errors .* 0.9
        @test compute_do_no_harm(errors_better, errors) < 1.0

        # Empty inputs
        @test compute_do_no_harm(Float64[], errors) ≈ 1.0
        @test compute_do_no_harm(errors, Float64[]) ≈ 1.0
    end

    @testset "Innovation burst" begin
        innovations = [1.0, 2.0, 3.0, 4.0, 5.0]
        peak, ratio = compute_innovation_burst(innovations, 3.5)
        @test peak == 5.0
        @test ratio ≈ 3/5  # 3 out of 5 below 3.5

        # All contained
        _, ratio2 = compute_innovation_burst([1.0, 1.0], 5.0)
        @test ratio2 ≈ 1.0

        # Empty
        peak_e, ratio_e = compute_innovation_burst(Float64[])
        @test peak_e == 0.0
        @test ratio_e ≈ 1.0
    end

    @testset "Map contamination score" begin
        # No correlation → near zero
        tile_updates = collect(range(0.1, 1.0, length=50))
        elev_positions = [SVector(10.0 + 0.1*i, 0.0, 3.5*mod(i,10)) for i in 1:50]
        score = compute_map_contamination(tile_updates, elev_positions)
        @test 0.0 <= score <= 1.0

        # Too few samples
        @test compute_map_contamination(Float64[0.1, 0.2], [SVector(0.0,0.0,0.0), SVector(0.0,0.0,3.5)]) == 0.0
    end

    @testset "Quantile computation" begin
        errors = collect(1.0:100.0)
        metrics = compute_elevator_metrics(
            errors, Float64[], Float64[], [], 1000.0;
            do_no_harm_ratio=1.05, false_source_count=2,
        )
        @test metrics.p50_error ≈ 50.5 atol=1.0
        @test metrics.p90_error ≈ 90.5 atol=1.0
        @test metrics.max_error == 100.0
        @test metrics.do_no_harm_ratio == 1.05
        @test metrics.false_source_count_per_km ≈ 2.0  # 2 / (1000/1000)
    end

    @testset "Segment classification" begin
        n = 100
        errors = fill(1.0, n)
        innovations = fill(1.5, n)
        tile_updates = fill(0.01, n)

        # Elevator at (0,0,z), pedestrian moves from (-10,0,0) to (10,0,0)
        elev_pos = [SVector(0.0, 0.0, 3.5) for _ in 1:n]
        ped_pos = [SVector(-10.0 + 20.0 * i / n, 0.0, 0.0) for i in 1:n]
        elev_vel = vcat(fill(0.0, 50), fill(1.5, 50))  # stopped then moving
        timestamps = collect(range(0.0, step=0.1, length=n))

        seg = compute_segment_metrics(
            errors, innovations, tile_updates,
            elev_pos, ped_pos, elev_vel, timestamps, 20.0;
        )

        @test seg.full_mission.p50_error ≈ 1.0
        @test seg isa SegmentMetrics
    end
end

println("All metric tests passed.")
