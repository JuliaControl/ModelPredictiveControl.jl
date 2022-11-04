module ModelPredictiveControl

using JuMP
using LinearAlgebra
using ControlSystemsBase

export LinModel, NonLinModel, setop!, updatestate, evaloutput
export InternalModel

include("sim_models.jl")
include("state_estim.jl")

end