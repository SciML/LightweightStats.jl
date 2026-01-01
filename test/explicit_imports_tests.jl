using ExplicitImports
using LightweightStats
using Test

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(LightweightStats) === nothing
    @test check_no_stale_explicit_imports(LightweightStats) === nothing
end
