using Test
include("../../sim/worlds/ElevatorWorld.jl")
include("../../sim/worlds/ParkingGarageWorld.jl")
include("../../sim/worlds/LobbyWorld.jl")
using .ElevatorWorldModule
using .ParkingGarageWorldModule
using .LobbyWorldModule
using StaticArrays

@testset "World Reproducibility" begin
    @testset "ElevatorWorld" begin
        w1 = create_default_elevator_world(seed=42)
        w2 = create_default_elevator_world(seed=42)
        for _ in 1:100
            ElevatorWorldModule.step!(w1, 0.1)
            ElevatorWorldModule.step!(w2, 0.1)
        end
        pos = SVector(5.0, 5.0, 0.0)
        @test ElevatorWorldModule.magnetic_field(w1, pos) ≈ ElevatorWorldModule.magnetic_field(w2, pos)
    end

    @testset "ParkingGarageWorld" begin
        w1 = create_default_garage(seed=42)
        w2 = create_default_garage(seed=42)
        for _ in 1:100
            ParkingGarageWorldModule.step!(w1, 0.1)
            ParkingGarageWorldModule.step!(w2, 0.1)
        end
        pos = SVector(5.0, 5.0, 0.0)
        @test ParkingGarageWorldModule.magnetic_field(w1, pos) ≈ ParkingGarageWorldModule.magnetic_field(w2, pos)
    end

    @testset "LobbyWorld" begin
        w1 = create_default_lobby(seed=42)
        w2 = create_default_lobby(seed=42)
        for _ in 1:100
            LobbyWorldModule.step!(w1, 0.1)
            LobbyWorldModule.step!(w2, 0.1)
        end
        pos = SVector(5.0, 5.0, 0.0)
        @test LobbyWorldModule.magnetic_field(w1, pos) ≈ LobbyWorldModule.magnetic_field(w2, pos)
    end
end
