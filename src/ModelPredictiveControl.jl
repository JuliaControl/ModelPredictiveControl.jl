module ModelPredictiveControl

using LinearAlgebra
using ControlSystemsBase
using JuMP, OSQP, Ipopt

export SimModel, LinModel, NonLinModel, setop!, setstate!, updatestate!, evaloutput
export StateEstimator, InternalModel, SteadyKalmanFilter, KalmanFilter
export initstate!
export PredictiveController, LinMPC, setconstraint!, moveinput!

include("sim_model.jl")
include("state_estim.jl")
include("predictive_control.jl")

end