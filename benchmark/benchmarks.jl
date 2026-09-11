using LightweightStats, BenchmarkTools
using StableRNGs

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

x = rand(rng, 10_000)
y = rand(rng, 10_000)
xm = rand(rng, 200, 200)

# =============================================================================
# Scalar reductions
# =============================================================================

SUITE["scalar"] = BenchmarkGroup()

SUITE["scalar"]["mean"] = @benchmarkable LightweightStats.mean($x)
SUITE["scalar"]["median"] = @benchmarkable LightweightStats.median($x)
SUITE["scalar"]["std"] = @benchmarkable LightweightStats.std($x)
SUITE["scalar"]["var"] = @benchmarkable LightweightStats.var($x)
SUITE["scalar"]["middle"] = @benchmarkable LightweightStats.middle($x)
SUITE["scalar"]["quantile"] = @benchmarkable LightweightStats.quantile(
    $x, [0.25, 0.5, 0.75]
)

# =============================================================================
# Two-argument statistics
# =============================================================================

SUITE["paired"] = BenchmarkGroup()

SUITE["paired"]["cov"] = @benchmarkable LightweightStats.cov($x, $y)
SUITE["paired"]["cor"] = @benchmarkable LightweightStats.cor($x, $y)

# =============================================================================
# Matrix statistics
# =============================================================================

SUITE["matrix"] = BenchmarkGroup()

SUITE["matrix"]["cov"] = @benchmarkable LightweightStats.cov($xm)
SUITE["matrix"]["cor"] = @benchmarkable LightweightStats.cor($xm)
SUITE["matrix"]["mean_dims1"] = @benchmarkable LightweightStats.mean($xm; dims = 1)
