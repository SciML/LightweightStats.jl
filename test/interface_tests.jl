using LightweightStats
using Test

@testset "Interface Compatibility" begin
    @testset "BigFloat support" begin
        x_bf = BigFloat[1.0, 2.0, 3.0, 4.0, 5.0]
        y_bf = BigFloat[2.0, 4.0, 6.0, 8.0, 10.0]
        A_bf = BigFloat[1.0 2.0 3.0; 4.0 5.0 6.0]

        # mean
        @test mean(x_bf) isa BigFloat
        @test eltype(mean(A_bf; dims = 1)) == BigFloat
        @test mean(x -> x^2, x_bf) isa BigFloat

        # var and std
        @test var(x_bf) isa BigFloat
        @test std(x_bf) isa BigFloat

        # cov
        @test cov(x_bf, y_bf) isa BigFloat
        @test cov(x_bf) isa BigFloat
        @test eltype(cov(A_bf; dims = 1)) == BigFloat
        @test eltype(cov(A_bf; dims = 2)) == BigFloat

        # cor
        @test cor(x_bf, y_bf) isa BigFloat
        @test eltype(cor(A_bf; dims = 1)) == BigFloat

        # median and quantile
        @test median(x_bf) isa BigFloat
        @test quantile(x_bf, 0.5) isa BigFloat

        # middle
        @test middle(x_bf) isa BigFloat
    end

    @testset "Float32 type preservation" begin
        x32 = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
        y32 = Float32[2.0, 4.0, 6.0, 8.0, 10.0]
        A32 = Float32[1.0 2.0 3.0; 4.0 5.0 6.0]

        # mean preserves Float32
        @test mean(x32) isa Float32
        @test eltype(mean(A32; dims = 1)) == Float32
        @test mean(x -> x^2, x32) isa Float32

        # var and std preserve Float32
        @test var(x32) isa Float32
        @test std(x32) isa Float32

        # cov preserves Float32
        @test cov(x32, y32) isa Float32
        @test cov(x32) isa Float32
        @test eltype(cov(A32; dims = 1)) == Float32
        @test eltype(cov(A32; dims = 2)) == Float32

        # cor preserves Float32
        @test cor(x32, y32) isa Float32
        @test eltype(cor(A32; dims = 1)) == Float32

        # median preserves Float32
        @test median(x32) isa Float32

        # quantile preserves Float32
        @test quantile(x32, 0.5) isa Float32

        # middle preserves Float32
        @test middle(x32) isa Float32
    end

    @testset "Int promotion to Float64" begin
        x_int = [1, 2, 3, 4, 5]
        A_int = [1 2 3; 4 5 6]

        # Int should promote to Float64 for division operations
        @test mean(x_int) isa Float64
        @test var(x_int) isa Float64
        @test std(x_int) isa Float64
        @test eltype(cov(A_int; dims = 1)) == Float64
        @test eltype(cor(A_int; dims = 1)) == Float64
    end
end
