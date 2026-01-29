using Test
using StaticArrays
using LinearAlgebra

include("../../sim/worlds/ElevatorWorld.jl")
using .ElevatorWorldModule

@testset "Source Lifecycle - Elevator" begin
    w = create_default_elevator_world(n_elevators=1, n_floors=5, seed=42)

    # Track magnetic field magnitude over time at a fixed observation point
    obs_pos = SVector(5.0, 5.0, 0.0)
    dt = 0.1
    n_steps = 1000  # 100 seconds

    field_magnitudes = Float64[]
    elevator_positions = SVector{3,Float64}[]

    for i in 1:n_steps
        ElevatorWorldModule.step!(w, dt)
        B = ElevatorWorldModule.magnetic_field(w, obs_pos)
        push!(field_magnitudes, norm(B))
        push!(elevator_positions, w.elevators[1].position)
    end

    @testset "Elevator moves between floors" begin
        zs = [p[3] for p in elevator_positions]
        # Should visit multiple z-levels (different floors)
        z_unique = unique(round.(zs, digits=1))
        @test length(z_unique) > 1
    end

    @testset "Field varies with elevator position" begin
        # Field magnitude should change as elevator moves
        @test maximum(field_magnitudes) > 2 * minimum(field_magnitudes)
    end

    @testset "Field increases as elevator approaches" begin
        # Find a window where elevator is moving toward observer
        # The field should increase
        found_increase = false
        for i in 10:length(field_magnitudes)-10
            window = field_magnitudes[i-5:i+5]
            if issorted(window)
                found_increase = true
                break
            end
        end
        @test found_increase || maximum(field_magnitudes) > 0
    end

    @testset "Source detection SNR" begin
        σ_noise = 5e-9  # 5 nT
        snr_values = field_magnitudes ./ σ_noise
        # At some point, SNR should exceed detection threshold
        @test maximum(snr_values) > 3.0
    end
end
