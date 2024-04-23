using MLNanoShaper
using Documenter

DocMeta.setdocmeta!(MLNanoShaper, :DocTestSetup, :(using MLNanoShaper); recursive = true)

makedocs(;
    modules = [MLNanoShaper],
    authors = "tristan <tristan.hacquard@polytechnique.org> and contributors",
    sitename = "MLNanoShaper.jl",
    format = Documenter.HTML(;
        canonical = "https://hack-hard.github.io/MLNanoShaper.jl",
        edit_link = "main",
        assets = String[],
        mathengine = MathJax3()),
    pages = [
        "Home" => "index.md",
    ],)

deploydocs(;
    repo = "github.com/hack-hard/MLNanoShaper.jl",
    devbranch = "main",)
