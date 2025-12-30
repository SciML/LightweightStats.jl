using LightweightStats
using Test

# JET tests are optional and only run on stable Julia releases 1.11+
# JET has strict Julia version requirements and tight coupling with the compiler:
# - JET 0.9.x works with Julia 1.10/1.11
# - JET 0.10.x/0.11.x works with Julia 1.12+
# We skip JET tests on:
# - Julia < 1.11 (LTS may have JET API differences)
# - Julia pre-release/nightly (may not have JET support yet)
const IS_STABLE_JULIA = VERSION >= v"1.11" && isempty(VERSION.prerelease)
const JET_AVAILABLE = IS_STABLE_JULIA && try
    @eval using JET
    true
catch e
    @info "JET not available: $e"
    false
end

@testset "JET static analysis" begin
    if !JET_AVAILABLE
        @info "JET tests skipped on Julia $(VERSION)"
        @test_skip true  # Mark test as skipped
        return
    end
    @testset "JET error analysis" begin
        # Test key entry points for static errors using report_call
        rep = JET.report_call(mean, (Vector{Float64},))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(mean, (Vector{Int},))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(median, (Vector{Float64},))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(var, (Vector{Float64},))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(std, (Vector{Float64},))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(cov, (Vector{Float64}, Vector{Float64}))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(cov, (Matrix{Float64},))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(cor, (Vector{Float64}, Vector{Float64}))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(cor, (Matrix{Float64},))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(quantile, (Vector{Float64}, Float64))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(middle, (Vector{Float64},))
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "JET optimization analysis" begin
        # Test key entry points for type stability using report_opt
        # Use target_modules to filter to LightweightStats only
        # (Base.mapslices has known runtime dispatches that are not our concern)
        rep = JET.report_opt(mean, (Vector{Float64},); target_modules = (LightweightStats,))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_opt(median, (Vector{Float64},); target_modules = (LightweightStats,))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_opt(var, (Vector{Float64},); target_modules = (LightweightStats,))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_opt(std, (Vector{Float64},); target_modules = (LightweightStats,))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_opt(cov, (Vector{Float64}, Vector{Float64}); target_modules = (LightweightStats,))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_opt(cov, (Matrix{Float64},); target_modules = (LightweightStats,))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_opt(cor, (Vector{Float64}, Vector{Float64}); target_modules = (LightweightStats,))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_opt(cor, (Matrix{Float64},); target_modules = (LightweightStats,))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_opt(quantile, (Vector{Float64}, Float64); target_modules = (LightweightStats,))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_opt(middle, (Vector{Float64},); target_modules = (LightweightStats,))
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "JET analysis with different numeric types" begin
        # Float32 type preservation
        rep = JET.report_call(mean, (Vector{Float32},))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(var, (Vector{Float32},))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(cov, (Matrix{Float32},))
        @test length(JET.get_reports(rep)) == 0

        # BigFloat support
        rep = JET.report_call(mean, (Vector{BigFloat},))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(var, (Vector{BigFloat},))
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "JET analysis with complex numbers" begin
        rep = JET.report_call(mean, (Vector{ComplexF64},))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(cov, (Vector{ComplexF64}, Vector{ComplexF64}))
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(cov, (Matrix{ComplexF64},))
        @test length(JET.get_reports(rep)) == 0
    end
end
