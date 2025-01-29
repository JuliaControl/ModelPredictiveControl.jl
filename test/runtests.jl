# spell-checker: disable

using ModelPredictiveControl
using ControlSystemsBase
using Documenter
using LinearAlgebra
using Random: randn
using JuMP, OSQP, Ipopt, DAQP, ForwardDiff
using Plots
using Test

@testset "ModelPredictiveControl.jl" begin
include("test_sim_model.jl")
include("test_state_estim.jl")
include("test_predictive_control.jl")
include("test_plot_sim.jl")

old_debug_level = ENV["JULIA_DEBUG"]
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

end;

nothing