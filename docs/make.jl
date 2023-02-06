ENV["PLOTS_TEST"] = "true"
ENV["GKS_WSTYPE"]=100

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
                "Plant Models" => "public/sim_model.md",
                "State Estimators" => "public/state_estim.md",
                "Predictive Controllers" => "public/predictive_control.md",
                "Generic Functions" => "public/generic_func.md",
            ],
            "Internals" => [
                "Plant Models" => "internals/sim_model.md",
                "State Estimators" => "internals/state_estim.md",
                "Predictive Controllers" => "internals/predictive_control.md",
            ],
        ],  
        "Index" => "func_index.md"
    ]
)

deploydocs(
    repo = "github.com/franckgaga/ModelPredictiveControl.jl.git",
    devbranch = "main",
)