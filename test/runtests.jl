# spell-checker: disable

using ControlSystemsBase
using Documenter
using LinearAlgebra
using ModelPredictiveControl
using Test


@testset "ModelPredictiveControl.jl" begin
include("test_sim_model.jl")
include("test_state_estim.jl")
include("test_predictive_control.jl")

DocMeta.setdocmeta!(
    ModelPredictiveControl, 
    :DocTestSetup, 
    :(using ModelPredictiveControl, ControlSystemsBase); 
    recursive=true,
    warn=false
)
doctest(ModelPredictiveControl, testset="DocTest")


end