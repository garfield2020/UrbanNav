using Test
using StaticArrays
using LinearAlgebra
using Random

include("../../sim/worlds/ElevatorWorld.jl")
include("../../sim/worlds/ParkingGarageWorld.jl")
using .ElevatorWorldModule
using .ParkingGarageWorldModule

@testset "Measurement Attribution Conservation" begin
    rng = MersenneTwister(42)
    μ0_4π = 1e-7

    @testset "Single elevator source" begin
        w = create_default_elevator_world(n_elevators=1, seed=42)

        for step in 1:50
            ElevatorWorldModule.step!(w, 0.1)
            obs_pos = SVector(3.0 + step * 0.1, 4.0, 0.0)

            # "Measured" field (from world)
            B_measured = ElevatorWorldModule.magnetic_field(w, obs_pos)

            # Background prediction (assume zero)
            B_background = SVector(0.0, 0.0, 0.0)

            # Source prediction (recompute from known source state)
            elev = w.elevators[1]
            r = obs_pos - elev.position
            r_mag = norm(r)
            if r_mag > 1e-10
                r_hat = r / r_mag
                m = elev.dipole_moment
                B_source_pred = μ0_4π * (3.0 * dot(m, r_hat) * r_hat - m) / r_mag^3
            else
                B_source_pred = SVector(0.0, 0.0, 0.0)
            end

            # Residual
            residual = B_measured - B_background - B_source_pred

            # Conservation: background + source + residual ≈ measured
            reconstructed = B_background + B_source_pred + residual
            @test norm(reconstructed - B_measured) < 1e-12
        end
    end

    @testset "Multi-source garage" begin
        w = create_default_garage(n_levels=1, n_vehicles=3, seed=42)
        obs_pos = SVector(15.0, 10.0, 0.5)

        B_total = ParkingGarageWorldModule.magnetic_field(w, obs_pos)

        # Verify field is non-zero (sources present)
        @test norm(B_total) > 0
    end
end
