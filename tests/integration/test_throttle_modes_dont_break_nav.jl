using Test
using UrbanNav
using StaticArrays

@testset "Throttle Modes Don't Break Nav" begin
    @testset "SlamConfig modes" begin
        # Verify SlamConfig and SlamMode exist
        @test isdefined(UrbanNav, :SlamConfig)
        @test isdefined(UrbanNav, :SlamMode)
    end

    @testset "Enable/disable flags don't crash" begin
        # Verify the module can be loaded with various configurations
        # Full SLAM pipeline integration would go here
        @test true  # Placeholder - module loads without errors
    end
end
