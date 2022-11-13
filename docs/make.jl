push!(LOAD_PATH,"../src/")

using Documenter
using ModelPredictiveControl

DocMeta.setdocmeta!(
    ModelPredictiveControl, 
    :DocTestSetup, 
    :(using ModelPredictiveControl, ControlSystemsBase); 
    recursive=true
)
makedocs(
    sitename    = "ModelPredictiveControl.jl",
    modules     = [ModelPredictiveControl],
    doctest     = true,
    pages = [
        "Home" => "index.md",
    ]
)

deploydocs(
    repo = "github.com/franckgaga/ModelPredictiveControl.jl.git",
    devbranch = "main",
)