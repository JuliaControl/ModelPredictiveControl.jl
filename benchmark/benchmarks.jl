using BenchmarkTools
using ModelPredictiveControl, ControlSystemsBase, LinearAlgebra
using JuMP, OSQP, DAQP, Ipopt, MadNLP, UnoSolver

const SUITE = BenchmarkGroup(["ModelPredictiveControl"])

SUITE["UNIT TESTS"]   = BenchmarkGroup(["allocation-free", "allocations", "single call"])
SUITE["CASE STUDIES"] = BenchmarkGroup(["performance", "speed" ,"integration"])

include("0_bench_setup.jl")
include("1_bench_sim_model.jl")
include("2_bench_state_estim.jl")
include("3_bench_predictive_control.jl")
