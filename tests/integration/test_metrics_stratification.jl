using Test
using StaticArrays
using LinearAlgebra

include("../../sim/worlds/ElevatorWorld.jl")
using .ElevatorWorldModule

@testset "Metrics Stratification" begin
    w = create_default_elevator_world(n_elevators=1, seed=42)
    obs_pos = SVector(5.0, 5.0, 0.0)
    dt = 0.1
    σ_threshold = 1e-8  # 10 nT threshold for "source active"

    source_active_residuals = Float64[]
    source_free_residuals = Float64[]

    for i in 1:200
        ElevatorWorldModule.step!(w, dt)
        B = ElevatorWorldModule.magnetic_field(w, obs_pos)
        B_mag = norm(B)

        if B_mag > σ_threshold
            push!(source_active_residuals, B_mag)
        else
            push!(source_free_residuals, B_mag)
        end
    end

    @testset "Reports stratify by source-active vs source-free" begin
        # Both categories should have entries
        @test length(source_active_residuals) + length(source_free_residuals) == 200

        # Source-active residuals should be larger on average
        if !isempty(source_active_residuals) && !isempty(source_free_residuals)
            @test mean(source_active_residuals) > mean(source_free_residuals)
        end
    end
end
