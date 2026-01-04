using LightweightStats
using Documenter

DocMeta.setdocmeta!(LightweightStats, :DocTestSetup, :(using LightweightStats); recursive = true)

makedocs(;
    modules = [LightweightStats],
    authors = "ChrisRackauckas <contact@chrisrackauckas.com>",
    sitename = "LightweightStats.jl",
    format = Documenter.HTML(;
        canonical = "https://SciML.github.io/LightweightStats.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/SciML/LightweightStats.jl",
    devbranch = "main",
)
