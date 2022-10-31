module ModelPredictiveControl

using JuMP
using LinearAlgebra
using ControlSystemsBase

export LinModel, NonLinModel, setop!, updatestate, evaloutput

include("sim_models.jl")

end