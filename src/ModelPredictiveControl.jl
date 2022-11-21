module ModelPredictiveControl

#using JuMP
using LinearAlgebra
using ControlSystemsBase

export SimModel, LinModel, NonLinModel, setop!, updatestate!, evaloutput
export InternalModel

include("sim_model.jl")
include("state_estim.jl")
#include("predictive_control.jl")

end