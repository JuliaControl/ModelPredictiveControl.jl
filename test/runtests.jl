# spell-checker: disable

using ModelPredictiveControl
using Documenter
using Test, TestItemRunner

@run_package_tests(verbose=true)

include("0_test_module.jl")
include("1_test_sim_model.jl")
include("2_test_state_estim.jl")
include("3_test_predictive_control.jl")
include("4_test_plot_sim.jl")

old_debug_level = get(ENV, "JULIA_DEBUG", "")
DocMeta.setdocmeta!(
    ModelPredictiveControl, 
    :DocTestSetup, 
    :(
        using ModelPredictiveControl, ControlSystemsBase;
        ENV["JULIA_DEBUG"] = ""; # temporarily disable @debug logging for the doctests
    ); 
    recursive=true,
    warn=false
)
doctest(ModelPredictiveControl, testset="DocTest")
ENV["JULIA_DEBUG"] = old_debug_level

nothing