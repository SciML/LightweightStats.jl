using LightweightStats
using Test
using Random

# Import functions from LightweightStats for cleaner test code
using LightweightStats: mean, median, std, var, cov, cor, quantile, middle

@testset "LightweightStats.jl" begin
    # Include regression tests against Statistics.jl
    include("regression_tests.jl")
    # Include interface compatibility tests
    include("interface_tests.jl")
    # Include JET static analysis tests
    include("jet_tests.jl")
    # Include explicit imports tests
    include("explicit_imports_tests.jl")
    @testset "mean" begin
        @test mean([1, 2, 3, 4, 5]) ≈ 3.0
        @test mean([1.5, 2.5, 3.5]) ≈ 2.5
        @test mean(Float32[1, 2, 3]) ≈ 2.0f0

        @test mean(x -> x^2, [1, 2, 3]) ≈ 14/3

        A = [1 2 3; 4 5 6]
        @test mean(A) ≈ 3.5
        @test mean(A; dims = 1) ≈ [2.5 3.5 4.5]
        @test mean(A; dims = 2) ≈ reshape([2.0, 5.0], 2, 1)

        @test_throws ArgumentError mean([])
    end

    @testset "median" begin
        @test median([1, 2, 3, 4, 5]) == 3
        @test median([1, 2, 3, 4]) == 2.5
        @test median([3, 1, 2]) == 2
        @test median([1]) == 1

        A = [1 2 3; 4 5 6]
        @test median(A) == 3.5
        @test median(A; dims = 1) == [2.5 3.5 4.5]
        @test median(A; dims = 2) == reshape([2.0, 5.0], 2, 1)

        @test_throws ArgumentError median([])
    end

    @testset "var and std" begin
        x = [1, 2, 3, 4, 5]
        @test var(x) ≈ 2.5
        @test var(x; corrected = false) ≈ 2.0
        @test std(x) ≈ sqrt(2.5)
        @test std(x; corrected = false) ≈ sqrt(2.0)

        @test var(x; mean = 3) ≈ 2.5
        @test std(x; mean = 3) ≈ sqrt(2.5)

        A = [1 2 3; 4 5 6]
        @test var(A) ≈ 3.5
        @test std(A) ≈ sqrt(3.5)

        @test isnan(var(Float64[]))
    end

    @testset "cov" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]

        @test cov(x, y) ≈ 5.0
        @test cov(x, y; corrected = false) ≈ 4.0
        @test cov(x) ≈ var(x)

        X = [1 2; 3 4; 5 6]
        C = cov(X; dims = 1)
        @test size(C) == (2, 2)
        @test C[1, 1] ≈ var(X[:, 1])
        @test C[2, 2] ≈ var(X[:, 2])
        @test C[1, 2] ≈ C[2, 1]

        C2 = cov(X; dims = 2)
        @test size(C2) == (3, 3)

        @test_throws DimensionMismatch cov([1, 2], [1, 2, 3])
    end

    @testset "cor" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]

        @test cor(x, y) ≈ 1.0
        @test cor(x, -y) ≈ -1.0

        X = [1 2; 3 4; 5 6]
        R = cor(X; dims = 1)
        @test size(R) == (2, 2)
        @test R[1, 1] ≈ 1.0
        @test R[2, 2] ≈ 1.0
        @test R[1, 2] ≈ R[2, 1]

        @test isnan(cor([1, 1, 1], [1, 2, 3]))

        @test_throws DimensionMismatch cor([1, 2], [1, 2, 3])
    end

    @testset "quantile" begin
        v = [1, 2, 3, 4, 5]

        @test quantile(v, 0.0) == 1
        @test quantile(v, 0.25) ≈ 2.0
        @test quantile(v, 0.5) ≈ 3.0
        @test quantile(v, 0.75) ≈ 4.0
        @test quantile(v, 1.0) == 5

        @test quantile(v, [0.25, 0.5, 0.75]) ≈ [2.0, 3.0, 4.0]

        @test_throws ArgumentError quantile(v, -0.1)
        @test_throws ArgumentError quantile(v, 1.1)
        @test_throws ArgumentError quantile([], 0.5)
    end

    @testset "middle" begin
        @test middle(1, 5) == 3
        @test middle(1.0, 5.0) == 3.0
        @test middle([1, 2, 3, 4, 5]) == 3
        @test middle([5, 1, 3]) == 3
        @test middle(10) == 10

        @test_throws ArgumentError middle([])
    end

    @testset "Type stability" begin
        @test typeof(mean(Int[1, 2, 3])) == Float64
        @test typeof(mean(Float32[1, 2, 3])) == Float32
        @test typeof(std(Int[1, 2, 3])) == Float64
        @test typeof(median(Int[1, 2, 3])) == Int
        @test typeof(median(Float64[1, 2, 3])) == Float64
    end

    @testset "Edge cases" begin
        @test mean([42]) == 42
        @test median([42]) == 42
        @test var([42]) |> isnan
        @test std([42]) |> isnan

        @test mean([1, 1, 1, 1]) == 1
        @test var([1, 1, 1, 1]; corrected = false) == 0
        @test std([1, 1, 1, 1]; corrected = false) == 0
    end
end
