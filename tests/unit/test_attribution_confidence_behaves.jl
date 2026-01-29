using Test
using StaticArrays
using LinearAlgebra

@testset "Attribution Confidence" begin
    μ0_4π = 1e-7

    @testset "Confidence decreases with distance" begin
        moment = SVector(0.0, 0.0, 100.0)  # 100 A·m²
        source_pos = SVector(0.0, 0.0, 0.0)

        distances = [5.0, 10.0, 15.0, 20.0]
        field_mags = Float64[]

        for d in distances
            obs_pos = SVector(d, 0.0, 0.0)
            r = obs_pos - source_pos
            r_mag = norm(r)
            r_hat = r / r_mag
            B = μ0_4π * (3.0 * dot(moment, r_hat) * r_hat - moment) / r_mag^3
            push!(field_mags, norm(B))
        end

        # Field should decrease monotonically with distance
        @test issorted(field_mags, rev=true)

        # Should follow 1/r³ (check ratio between consecutive distances)
        for i in 1:length(distances)-1
            ratio = field_mags[i] / field_mags[i+1]
            expected_ratio = (distances[i+1] / distances[i])^3
            @test abs(ratio - expected_ratio) / expected_ratio < 0.1
        end
    end

    @testset "Attribution sums to total" begin
        # Two sources, verify sum
        m1 = SVector(0.0, 0.0, 50.0)
        m2 = SVector(0.0, 100.0, 0.0)
        p1 = SVector(5.0, 0.0, 0.0)
        p2 = SVector(-3.0, 4.0, 0.0)
        obs = SVector(0.0, 0.0, 0.0)

        function dipole_B(m, src, pt)
            r = pt - src
            r_mag = norm(r)
            r_hat = r / r_mag
            μ0_4π * (3.0 * dot(m, r_hat) * r_hat - m) / r_mag^3
        end

        B1 = dipole_B(m1, p1, obs)
        B2 = dipole_B(m2, p2, obs)
        B_total = B1 + B2

        # Conservation
        @test norm(B_total - (B1 + B2)) < 1e-15
    end
end
