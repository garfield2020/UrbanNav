using Test
include("../../sim/worlds/ElevatorWorld.jl")
using .ElevatorWorldModule
using StaticArrays
using LinearAlgebra

@testset "Source Injection SNR" begin
    @testset "Elevator dipole field magnitude" begin
        # Single elevator at origin with known moment
        w = create_default_elevator_world(n_elevators=1, seed=1)
        # Field from dipole: B ~ μ₀/(4π) * m / r³
        # For m=200 A·m², r=10m: B ≈ 1e-7 * 200 / 1000 = 2e-8 T = 20 nT

        # Check that field decays as 1/r³
        pos_near = SVector(5.0, 0.0, 0.0)
        pos_far = SVector(10.0, 0.0, 0.0)
        B_near = norm(magnetic_field(w, pos_near))
        B_far = norm(magnetic_field(w, pos_far))

        # Ratio should be approximately (10/5)³ = 8
        ratio = B_near / B_far
        @test 6.0 < ratio < 10.0  # Allow some tolerance for dipole orientation
    end

    @testset "SNR above detection threshold" begin
        σ_noise = 5e-9  # 5 nT sensor noise
        w = create_default_elevator_world(n_elevators=1, seed=1)

        # At 10m, elevator should be detectable
        pos = SVector(10.0, 0.0, 0.0)
        B = norm(magnetic_field(w, pos))
        snr = B / σ_noise
        @test snr > 1.0  # Should be detectable at reasonable range
    end
end
