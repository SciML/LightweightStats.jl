using AllocCheck
using LightweightStats
using Test

@testset "AllocCheck - Zero Allocations" begin
    # Test that vector cov and cor don't allocate
    @testset "Vector covariance and correlation" begin
        x = rand(100)
        y = rand(100)

        # Warm up
        LightweightStats.cov(x, y)
        LightweightStats.cor(x, y)

        # Test zero allocations
        @test (@allocated LightweightStats.cov(x, y)) == 0
        @test (@allocated LightweightStats.cor(x, y)) == 0
    end

    @testset "Basic statistics" begin
        x = rand(100)

        # Warm up
        LightweightStats.mean(x)
        LightweightStats.var(x)
        LightweightStats.std(x)

        # Test zero allocations
        @test (@allocated LightweightStats.mean(x)) == 0
        @test (@allocated LightweightStats.var(x)) == 0
        @test (@allocated LightweightStats.std(x)) == 0
    end

    @testset "Middle function" begin
        x = rand(100)

        # Warm up
        LightweightStats.middle(1.0, 2.0)
        LightweightStats.middle(5.0)

        # Test zero allocations for simple cases
        @test (@allocated LightweightStats.middle(1.0, 2.0)) == 0
        @test (@allocated LightweightStats.middle(5.0)) == 0
    end
end
