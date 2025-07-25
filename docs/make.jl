using MLNanoShaper
using Documenter

DocMeta.setdocmeta!(MLNanoShaper, :DocTestSetup, :(using MLNanoShaper); recursive = true)

makedocs(;
    format = Documenter.LaTeX(),
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
        "Building Custom Models" => "model.md",
        "CLI Interface" => "cli.md",
        "model evaluation" => "evaluation.md",
        "C Interface" => "so.md"
    ])

deploydocs(;
    repo = "https://github.com/concept-lab/MLNanoShaper.jl.git",
)
