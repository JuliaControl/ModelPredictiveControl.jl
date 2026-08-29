using ModelPredictiveControl
using Test, TestItems, TestItemRunner

@run_package_tests

# Not needed for TestItems discovery, but including the files means `Pkg.test` parses them 
# and catches syntax errors early:
include("0_test_module.jl")
include("1_test_sim_model.jl")
include("2_test_state_estim.jl")
include("3_test_predictive_control.jl")
include("4_test_plot_sim.jl")
include("5_test_extensions.jl")
include("6_test_doctest.jl")
include("7_test_aqua.jl")

nothing