module ModelPredictiveControl

#using JuMP
using LinearAlgebra
using ControlSystemsBase
using OSQP
using SparseArrays

export SimModel, LinModel, NonLinModel, setop!, setstate!, updatestate!, evaloutput
export StateEstimator, InternalModel, SteadyKalmanFilter, KalmanFilter
export initstate!
export PredictiveController, LinMPC, setconstraint!, moveinput!

include("sim_model.jl")
include("state_estim.jl")

export LinMPC

include("predictive_control.jl")

end