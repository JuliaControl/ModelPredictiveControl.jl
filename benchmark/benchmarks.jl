using BenchmarkTools
using ModelPredictiveControl, ControlSystemsBase, LinearAlgebra
using JuMP, OSQP, DAQP, Ipopt, MadNLP

const SUITE = BenchmarkGroup(["ModelPredictiveControl"])

const ALLOC   = SUITE["allocation"] = BenchmarkGroup(["allocation-free","no-allocation"])
const RUNTIME = SUITE["runtime"]    = BenchmarkGroup(["performance","speed"])

include("0_bench_setup.jl")
include("1_bench_sim_model.jl")
include("2_bench_state_estim.jl")
include("3_bench_predictive_control.jl")
