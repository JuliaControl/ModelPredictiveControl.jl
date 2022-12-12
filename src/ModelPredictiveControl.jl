module ModelPredictiveControl

#using JuMP
using LinearAlgebra
using ControlSystemsBase

export SimModel, LinModel, NonLinModel, setop!, setstate!, updatestate!, evaloutput
export StateEstimator, InternalModel, SteadyKalmanFilter, KalmanFilter
export initstate!
export LinMPC, setconstraint!

include("sim_model.jl")
include("state_estim.jl")

export LinMPC

include("predictive_control.jl")

end