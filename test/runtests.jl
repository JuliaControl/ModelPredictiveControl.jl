# spell-checker: disable

using ModelPredictiveControl
using ControlSystemsBase
using Documenter
using LinearAlgebra
using Random: randn
using JuMP, OSQP, Ipopt, DAQP, ForwardDiff
using Plots
using Test, TestItemRunner

@run_package_tests 

@testset "ModelPredictiveControl.jl" begin

@testmodule SetupMPCtests begin
    using ControlSystemsBase
    Ts = 400.0
    sys = [ tf(1.90,[1800.0,1])   tf(1.90,[1800.0,1])   tf(1.90,[1800.0,1]);
            tf(-0.74,[800.0,1])   tf(0.74,[800.0,1])    tf(-0.74,[800.0,1])   ] 
    sys_ss = minreal(ss(sys))
    Gss = c2d(sys_ss[:,1:2], Ts, :zoh)
    Gss2 = c2d(sys_ss[:,1:2], 0.5Ts, :zoh)
    export Ts, sys, sys_ss, Gss, Gss2
end

include("test_sim_model.jl")
include("test_state_estim.jl")
include("test_predictive_control.jl")
include("test_plot_sim.jl")

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

end;

nothing