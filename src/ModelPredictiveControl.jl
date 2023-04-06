module ModelPredictiveControl

using LinearAlgebra
using ControlSystemsBase
using JuMP
import OSQP, Ipopt

export SimModel, LinModel, NonLinModel, setop!, setstate!, updatestate!, evaloutput
export StateEstimator, InternalModel
export SteadyKalmanFilter, KalmanFilter, UnscentedKalmanFilter
export initstate!
export PredictiveController, LinMPC, NonLinMPC, setconstraint!, moveinput!

include("sim_model.jl")
include("state_estim.jl")
include("predictive_control.jl")

end