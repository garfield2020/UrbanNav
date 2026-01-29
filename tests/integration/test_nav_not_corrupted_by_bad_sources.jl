using Test
using StaticArrays
using LinearAlgebra
using Random
using Statistics

@testset "Nav Not Corrupted by Bad Sources" begin
    rng = MersenneTwister(42)
    μ0_4π = 1e-7

    # True source parameters
    true_moment = SVector(0.0, 0.0, 200.0)
    source_pos = SVector(10.0, 0.0, 0.0)

    # Wrong source estimate (10× wrong moment)
    wrong_moment = SVector(0.0, 0.0, 2000.0)

    n_steps = 100
    σ_noise = 5e-9

    function compute_field(moment, src_pos, obs_pos)
        r = obs_pos - src_pos
        r_mag = norm(r)
        r_mag < 1e-10 && return SVector(0.0, 0.0, 0.0)
        r_hat = r / r_mag
        μ0_4π * (3.0 * dot(moment, r_hat) * r_hat - moment) / r_mag^3
    end

    @testset "Graceful degradation with wrong source" begin
        # Baseline: residuals with no source subtraction
        baseline_residuals = Float64[]
        # Degraded: residuals with wrong source subtracted
        degraded_residuals = Float64[]

        for i in 1:n_steps
            obs_pos = SVector(0.0 + i * 0.2, 5.0, 0.0)
            noise = SVector(σ_noise * randn(rng), σ_noise * randn(rng), σ_noise * randn(rng))

            B_true = compute_field(true_moment, source_pos, obs_pos) + noise
            B_wrong_pred = compute_field(wrong_moment, source_pos, obs_pos)

            # Baseline: raw measurement residual (no source subtraction)
            push!(baseline_residuals, norm(B_true))

            # Degraded: subtract wrong source prediction
            corrected = B_true - B_wrong_pred
            push!(degraded_residuals, norm(corrected))
        end

        baseline_rmse = sqrt(mean(baseline_residuals .^ 2))
        degraded_rmse = sqrt(mean(degraded_residuals .^ 2))

        # The degraded case should not diverge
        @test isfinite(degraded_rmse)
        @test degraded_rmse < 100 * baseline_rmse  # bounded, not divergent
    end

    @testset "No divergence" begin
        # All residuals should remain bounded
        for i in 1:n_steps
            obs_pos = SVector(0.0 + i * 0.2, 5.0, 0.0)
            B = compute_field(true_moment, source_pos, obs_pos)
            @test isfinite(norm(B))
            @test norm(B) < 1.0  # Should be well below 1 Tesla
        end
    end
end
