# debug bad cropping Plots in manual:
ENV["PLOTS_TEST"] = "true"
ENV["GKSwstype"] = "nul"
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
    #format = Documenter.LaTeX(platform = "none"),
    doctest     = true,
    format      = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        edit_link = "main"
    ),
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Installation" => "manual/installation.md",
            "Examples" => [
                "Linear Design" => "manual/linmpc.md",
                "Nonlinear Design" => "manual/nonlinmpc.md",
                "ModelingToolkit" =>  "manual/mtk.md",
            ],
        ],
        "Functions" => [
            "Public" => [
                "Plant Models" => "public/sim_model.md",
                "State Estimators" => "public/state_estim.md",
                "Predictive Controllers" => "public/predictive_control.md",
                "Generic Functions" => "public/generic_func.md",
                "Simulations and Plots" => "public/plot_sim.md",
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
    repo = "github.com/JuliaControl/ModelPredictiveControl.jl.git",
    devbranch = "main",
    push_preview = true
)
