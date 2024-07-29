using MLNanoShaper
using Documenter

DocMeta.setdocmeta!(MLNanoShaper, :DocTestSetup, :(using MLNanoShaper); recursive = true)

makedocs(;
    modules = [MLNanoShaper],
    authors = "tristan hacquard<tristan.hacquard@polytechnique.org> and contributors",
    sitename = "MLNanoShaper.jl",
    format = Documenter.HTML(;
        canonical = "https://concept-lab.github.io/MLNanoShaper.jl",
        edit_link = "main",
        assets = String[],
        mathengine = MathJax3()),
    pages = [
        "Home" => "index.md",
        "Custom Loss" => "loss.md",
    ],)

deploydocs(;
    repo = "https://github.com/concept-lab/MLNanoShaper.jl.git",
) 
