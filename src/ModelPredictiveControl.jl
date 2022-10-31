module ModelPredictiveControl

using JuMP
using LinearAlgebra
using ControlSystemsBase

export LinModel, NonLinModel, setop!

include("sim_models.jl")

end