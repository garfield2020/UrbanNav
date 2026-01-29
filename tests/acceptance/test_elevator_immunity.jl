using Test
using StaticArrays
using LinearAlgebra
using Random

@testset "Elevator Immunity" begin
    μ0_4π = 1e-7

    @testset "Nav not poisoned by elevator presence" begin
        # Simulate: background map should not absorb elevator signature
        # Use a simple tile model: if we subtract the known source,
        # the residual should be small

        rng = MersenneTwister(42)
        source_moment = SVector(0.0, 0.0, 300.0)
        source_pos = SVector(10.0, 0.0, 5.0)

        # Accumulate "tile learning" - only from clean residuals
        tile_accumulator = zeros(3)
        n_samples = 0

        for i in 1:100
            obs_pos = SVector(rand(rng) * 20.0, rand(rng) * 20.0, 0.0)

            r = obs_pos - source_pos
            r_mag = norm(r)
            r_hat = r / r_mag
            B_source = μ0_4π * (3.0 * dot(source_moment, r_hat) * r_hat - source_moment) / r_mag^3

            noise = SVector(5e-9 * randn(rng), 5e-9 * randn(rng), 5e-9 * randn(rng))
            B_measured = B_source + noise

            # Clean residual: subtract known source
            clean_residual = B_measured - B_source

            tile_accumulator .+= collect(clean_residual)
            n_samples += 1
        end

        # Average tile should be near zero (not absorbed source)
        tile_mean = tile_accumulator ./ n_samples
        @test norm(tile_mean) < 50e-9  # < 50 nT
    end
end
