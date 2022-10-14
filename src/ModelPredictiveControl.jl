module ModelPredictiveControl

using JuMP
using LinearAlgebra

export greet, LinModel, NonLinModel

greet() = "Hello World!"

include("sim_models.jl")

end