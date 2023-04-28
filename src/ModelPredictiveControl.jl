module ModelPredictiveControl

using PrecompileTools

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

@setup_workload begin
    # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
    # precompile file and potentially make loading faster.
    @compile_workload begin
        # all calls in this block will be precompiled, regardless of whether
        # they belong to your package or not (on Julia 1.8 and higher)
        include("precompile_calls.jl")
    end
end

end