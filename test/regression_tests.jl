using LightweightStats
using Statistics
using Test
using Random

@testset "Regression tests against Statistics.jl" begin
    # Set seed for reproducibility
    Random.seed!(12345)

    # Test data of various types and sizes
    test_vectors = [
        [1, 2, 3, 4, 5],
        Float64[1.5, 2.7, 3.1, 4.9, 5.2],
        Float32[1.0, 2.0, 3.0, 4.0],
        randn(100),
        randn(Float32, 50),
        collect(1:10),
        collect(1:1000),
        [1],  # single element
    ]

    test_matrices = [
        [1 2 3; 4 5 6],
        randn(10, 5),
        randn(Float32, 8, 3),
        collect(reshape(1:24, 4, 6)),  # Convert to concrete Array
        ones(3, 3),
        [1;;],  # 1x1 matrix
    ]

    @testset "mean - vectors" begin
        for v in test_vectors
            @test LightweightStats.mean(v) ≈ Statistics.mean(v) rtol = 1.0e-8
            # Test with function argument
            @test LightweightStats.mean(x -> x^2, v) ≈ Statistics.mean(x -> x^2, v) rtol = 1.0e-8
            @test LightweightStats.mean(abs, v) ≈ Statistics.mean(abs, v) rtol = 1.0e-8
        end

        # Test empty array error
        @test_throws ArgumentError LightweightStats.mean(Int[])
        # Note: Statistics.mean returns NaN for empty arrays, not an error
        @test isnan(Statistics.mean(Float64[]))
    end

    @testset "mean - matrices with dims" begin
        for m in test_matrices
            @test LightweightStats.mean(m) ≈ Statistics.mean(m) rtol = 1.0e-8
            @test LightweightStats.mean(m; dims = 1) ≈ Statistics.mean(m; dims = 1) rtol = 1.0e-8
            @test LightweightStats.mean(m; dims = 2) ≈ Statistics.mean(m; dims = 2) rtol = 1.0e-8
        end
    end

    @testset "median - vectors" begin
        for v in test_vectors
            # Compare values, not types (Statistics.jl may return different types)
            @test LightweightStats.median(v) ≈ Statistics.median(v) rtol = 1.0e-8
        end

        # Test odd and even length vectors
        @test LightweightStats.median([1, 2, 3]) == Statistics.median([1, 2, 3])
        @test LightweightStats.median([1, 2, 3, 4]) == Statistics.median([1, 2, 3, 4])

        # Test empty array error
        @test_throws ArgumentError LightweightStats.median(Int[])
        @test_throws ArgumentError Statistics.median(Int[])
    end

    @testset "median - matrices with dims" begin
        for m in test_matrices
            @test LightweightStats.median(m) ≈ Statistics.median(m) rtol = 1.0e-8
            @test LightweightStats.median(m; dims = 1) ≈ Statistics.median(m; dims = 1) rtol = 1.0e-8
            @test LightweightStats.median(m; dims = 2) ≈ Statistics.median(m; dims = 2) rtol = 1.0e-8
        end
    end

    @testset "var - variance" begin
        for v in test_vectors
            lw_var = LightweightStats.var(v)
            st_var = Statistics.var(v)
            # Handle NaN comparisons for single element arrays
            if isnan(lw_var) && isnan(st_var)
                @test true  # Both are NaN, which is correct
            else
                @test lw_var ≈ st_var rtol = 1.0e-8
            end

            # Test corrected parameter
            @test LightweightStats.var(v; corrected = false) ≈ Statistics.var(v; corrected = false) rtol = 1.0e-8

            # With known mean
            m = Statistics.mean(v)
            lw_var_m = LightweightStats.var(v; mean = m)
            st_var_m = Statistics.var(v; mean = m)
            if isnan(lw_var_m) && isnan(st_var_m)
                @test true
            else
                @test lw_var_m ≈ st_var_m rtol = 1.0e-8
            end
        end

        # Test empty array returns NaN
        @test isnan(LightweightStats.var(Float64[]))
        @test isnan(Statistics.var(Float64[]))
    end

    @testset "var - matrices with dims" begin
        for m in test_matrices
            # Compare overall variance
            lw_var = LightweightStats.var(m)
            st_var = Statistics.var(m)
            if isnan(lw_var) && isnan(st_var)
                @test true
            else
                @test lw_var ≈ st_var rtol = 1.0e-8
            end

            # Compare along dimensions, handling NaN arrays
            for dims in [1, 2]
                lw_var_dims = LightweightStats.var(m; dims = dims)
                st_var_dims = Statistics.var(m; dims = dims)
                # Element-wise comparison handling NaN
                @test all(
                    i -> (isnan(lw_var_dims[i]) && isnan(st_var_dims[i])) ||
                        (lw_var_dims[i] ≈ st_var_dims[i]), eachindex(lw_var_dims)
                )
            end
        end
    end

    @testset "std - standard deviation" begin
        for v in test_vectors
            lw_std = LightweightStats.std(v)
            st_std = Statistics.std(v)
            if isnan(lw_std) && isnan(st_std)
                @test true
            else
                @test lw_std ≈ st_std rtol = 1.0e-8
            end

            # With known mean
            m = Statistics.mean(v)
            lw_std_m = LightweightStats.std(v; mean = m)
            st_std_m = Statistics.std(v; mean = m)
            if isnan(lw_std_m) && isnan(st_std_m)
                @test true
            else
                @test lw_std_m ≈ st_std_m rtol = 1.0e-8
            end
        end
    end

    @testset "std - matrices with dims" begin
        for m in test_matrices
            lw_std = LightweightStats.std(m)
            st_std = Statistics.std(m)
            if isnan(lw_std) && isnan(st_std)
                @test true
            else
                @test lw_std ≈ st_std rtol = 1.0e-8
            end

            for dims in [1, 2]
                lw_std_dims = LightweightStats.std(m; dims = dims)
                st_std_dims = Statistics.std(m; dims = dims)
                @test all(
                    i -> (isnan(lw_std_dims[i]) && isnan(st_std_dims[i])) ||
                        (lw_std_dims[i] ≈ st_std_dims[i]), eachindex(lw_std_dims)
                )
            end
        end
    end

    @testset "cov - covariance vectors" begin
        x = randn(20)
        y = randn(20)

        @test LightweightStats.cov(x, y) ≈ Statistics.cov(x, y) rtol = 1.0e-8
        @test LightweightStats.cov(x, y; corrected = true) ≈ Statistics.cov(x, y; corrected = true) rtol = 1.0e-8
        @test LightweightStats.cov(x, y; corrected = false) ≈ Statistics.cov(x, y; corrected = false) rtol = 1.0e-8

        # Self-covariance equals variance
        @test LightweightStats.cov(x) ≈ Statistics.cov(x) rtol = 1.0e-8
        @test LightweightStats.cov(x) ≈ LightweightStats.var(x) rtol = 1.0e-8

        # Test dimension mismatch
        @test_throws DimensionMismatch LightweightStats.cov(x[1:10], y)
        @test_throws DimensionMismatch Statistics.cov(x[1:10], y)
    end

    @testset "cov - covariance matrices" begin
        for m in test_matrices[1:4]  # Skip single element cases
            C_lw = LightweightStats.cov(m; dims = 1)
            C_st = Statistics.cov(m; dims = 1)
            # Element-wise comparison handling potential NaN
            @test all(
                i -> (isnan(C_lw[i]) && isnan(C_st[i])) ||
                    (C_lw[i] ≈ C_st[i]), eachindex(C_lw)
            )
            @test size(C_lw) == size(C_st)

            C_lw = LightweightStats.cov(m; dims = 2)
            C_st = Statistics.cov(m; dims = 2)
            @test all(
                i -> (isnan(C_lw[i]) && isnan(C_st[i])) ||
                    (C_lw[i] ≈ C_st[i]), eachindex(C_lw)
            )
            @test size(C_lw) == size(C_st)
        end
    end

    @testset "cor - correlation vectors" begin
        x = randn(30)
        y = randn(30)

        @test LightweightStats.cor(x, y) ≈ Statistics.cor(x, y) rtol = 1.0e-8
        @test LightweightStats.cor(x, x) ≈ 1.0 rtol = 1.0e-8
        @test Statistics.cor(x, x) ≈ 1.0 rtol = 1.0e-8

        # Perfect positive and negative correlation
        z = 2 * x .+ 3
        @test LightweightStats.cor(x, z) ≈ Statistics.cor(x, z) rtol = 1.0e-8
        @test LightweightStats.cor(x, z) ≈ 1.0 rtol = 1.0e-8

        w = -2 * x .+ 5
        @test LightweightStats.cor(x, w) ≈ Statistics.cor(x, w) rtol = 1.0e-8
        @test LightweightStats.cor(x, w) ≈ -1.0 rtol = 1.0e-8

        # Test zero variance case
        constant = ones(10)
        @test isnan(LightweightStats.cor(constant, randn(10)))
        @test isnan(Statistics.cor(constant, randn(10)))
    end

    @testset "cor - correlation matrices" begin
        for m in test_matrices[1:4]
            R_lw = LightweightStats.cor(m; dims = 1)
            R_st = Statistics.cor(m; dims = 1)
            @test all(
                i -> (isnan(R_lw[i]) && isnan(R_st[i])) ||
                    (R_lw[i] ≈ R_st[i]), eachindex(R_lw)
            )
            @test size(R_lw) == size(R_st)

            # Check diagonal elements are 1 (or NaN for zero variance)
            for i in 1:size(R_lw, 1)
                if !isnan(R_lw[i, i])
                    @test R_lw[i, i] ≈ 1.0 rtol = 1.0e-6
                end
            end

            R_lw = LightweightStats.cor(m; dims = 2)
            R_st = Statistics.cor(m; dims = 2)
            @test all(
                i -> (isnan(R_lw[i]) && isnan(R_st[i])) ||
                    (R_lw[i] ≈ R_st[i]), eachindex(R_lw)
            )
        end
    end

    @testset "quantile - single quantile" begin
        for v in test_vectors
            for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
                @test LightweightStats.quantile(v, p) ≈ Statistics.quantile(v, p) rtol = 1.0e-8
            end
        end

        # Test edge cases
        v = [1, 2, 3, 4, 5]
        @test LightweightStats.quantile(v, 0.0) == Statistics.quantile(v, 0.0)
        @test LightweightStats.quantile(v, 1.0) == Statistics.quantile(v, 1.0)

        # Test errors
        @test_throws ArgumentError LightweightStats.quantile(v, -0.1)
        @test_throws ArgumentError LightweightStats.quantile(v, 1.1)
        @test_throws ArgumentError LightweightStats.quantile([], 0.5)
        @test_throws ArgumentError Statistics.quantile(v, -0.1)
        @test_throws ArgumentError Statistics.quantile(v, 1.1)
        @test_throws ArgumentError Statistics.quantile([], 0.5)
    end

    @testset "quantile - multiple quantiles" begin
        for v in test_vectors
            ps = [0.25, 0.5, 0.75]
            @test LightweightStats.quantile(v, ps) ≈ Statistics.quantile(v, ps) rtol = 1.0e-8

            ps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            @test LightweightStats.quantile(v, ps) ≈ Statistics.quantile(v, ps) rtol = 1.0e-8
        end
    end

    @testset "middle" begin
        for v in test_vectors
            @test LightweightStats.middle(v) ≈ Statistics.middle(v) rtol = 1.0e-8
        end

        # Test with explicit values
        @test LightweightStats.middle(1, 5) == Statistics.middle(1, 5)
        @test LightweightStats.middle(1.0, 5.0) == Statistics.middle(1.0, 5.0)
        @test LightweightStats.middle(-10, 10) == Statistics.middle(-10, 10)

        # Test single value
        @test LightweightStats.middle(42) == Statistics.middle(42)
        @test LightweightStats.middle(3.14) == Statistics.middle(3.14)

        # Test error for empty
        @test_throws ArgumentError LightweightStats.middle([])
        # Note: Statistics.jl may throw different error types on different platforms/versions
        # so we just verify it throws some error
        @test_throws Exception Statistics.middle([])
    end

    @testset "Edge cases and special values" begin
        # NaN handling
        v_nan = [1.0, 2.0, NaN, 4.0, 5.0]
        @test isnan(LightweightStats.mean(v_nan))
        @test isnan(Statistics.mean(v_nan))
        @test isnan(LightweightStats.std(v_nan))
        @test isnan(Statistics.std(v_nan))

        # Inf handling
        v_inf = [1.0, 2.0, Inf, 4.0, 5.0]
        @test LightweightStats.mean(v_inf) == Statistics.mean(v_inf)
        @test isinf(LightweightStats.mean(v_inf))

        # Very large numbers
        v_large = [1.0e307, 2.0e307, 3.0e307]
        @test LightweightStats.mean(v_large) ≈ Statistics.mean(v_large) rtol = 1.0e-8
        @test LightweightStats.std(v_large) ≈ Statistics.std(v_large) rtol = 1.0e-8

        # Very small numbers
        v_small = [1.0e-307, 2.0e-307, 3.0e-307]
        @test LightweightStats.mean(v_small) ≈ Statistics.mean(v_small) rtol = 1.0e-8
        @test LightweightStats.std(v_small) ≈ Statistics.std(v_small) rtol = 1.0e-8
    end
end
