using ModelPredictiveControl
using Test, TestItems, TestItemRunner

@run_package_tests(verbose=true)

include("0_test_module.jl")
include("1_test_sim_model.jl")
include("2_test_state_estim.jl")
include("3_test_predictive_control.jl")
include("4_test_plot_sim.jl")
include("5_test_doctest.jl")

nothing