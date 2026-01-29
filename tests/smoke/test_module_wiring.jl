using Test
using UrbanNav

@testset "Module Wiring" begin
    # Verify key types are exported
    @test isdefined(UrbanNav, :UrbanNavState)
    @test isdefined(UrbanNav, :UrbanNavFactorGraph)
    @test isdefined(UrbanNav, :SourceTracker)
    @test isdefined(UrbanNav, :OnlineSourceSLAM)
    @test isdefined(UrbanNav, :OnlineSafetyController)

    # Verify key functions
    @test isdefined(UrbanNav, :residual)
end
