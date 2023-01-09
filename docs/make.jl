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
    format      = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        edit_link = "main"
    ),
    pages = [
        "Home" => "index.md",
        "Manual" => "manual.md",
        "Functions" => [
            "Public" => [
                "Specifying Plant Models" => "public/sim_model.md",
                "State Estimator Design" => "public/state_estim.md",
                "Predictive Controller Design" => "public/predictive_control.md",
                "Generic Functions" => "public/generic_func.md",
            ],
            "Internals" => [
                "SimModel" => "internals/sim_model.md",
                "StateEstimator" => "internals/state_estim.md",
                "PredictiveController" => "internals/predictive_control.md",
            ],
        ],  
        "Index" => "func_index.md"
    ]
)

deploydocs(
    repo = "github.com/franckgaga/ModelPredictiveControl.jl.git",
    devbranch = "main",
)