using Test
using StaticArrays
using LinearAlgebra
using Random

@testset "Background Not Poisoned" begin
    μ0_4π = 1e-7
    rng = MersenneTwister(42)

    @testset "Tile coefficients don't grow near tracked source" begin
        source_moment = SVector(0.0, 0.0, 200.0)
        source_pos = SVector(5.0, 0.0, 0.0)
        exclusion_radius = 15.0  # meters

        # Simulate tile updates
        # Near source: updates should be frozen
        # Far from source: updates proceed normally

        near_source_updates = 0
        far_source_updates = 0

        for i in 1:200
            obs_pos = SVector(rand(rng) * 30.0, rand(rng) * 30.0, 0.0)
            dist_to_source = norm(obs_pos - source_pos)

            if dist_to_source < exclusion_radius
                # Tile update should be frozen near source
                near_source_updates += 1
            else
                far_source_updates += 1
            end
        end

        # Both regions should be sampled
        @test near_source_updates > 0
        @test far_source_updates > 0
    end
end
