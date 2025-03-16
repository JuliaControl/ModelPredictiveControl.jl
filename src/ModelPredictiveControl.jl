module ModelPredictiveControl

using PrecompileTools
using LinearAlgebra
using Random: randn

using RecipesBase
using ProgressLogging
using ForwardDiff

import ControlSystemsBase
import ControlSystemsBase: ss, tf, delay
import ControlSystemsBase: Continuous, Discrete
import ControlSystemsBase: StateSpace, TransferFunction, DelayLtiSystem, LTISystem
import ControlSystemsBase: iscontinuous, isdiscrete, sminreal, minreal, c2d, d2c

import JuMP
import JuMP: MOIU, MOI, GenericModel, Model, optimizer_with_attributes, register
import JuMP: @variable, @operator, @constraint, @objective

import PreallocationTools: DiffCache, get_tmp

import OSQP, Ipopt

export SimModel, LinModel, NonLinModel
export DiffSolver, RungeKutta, ForwardEuler
export setop!, setname!
export setstate!, setmodel!, preparestate!, updatestate!, evaloutput, linearize, linearize!
export savetime!, periodsleep
export StateEstimator, InternalModel, Luenberger
export SteadyKalmanFilter, KalmanFilter, UnscentedKalmanFilter, ExtendedKalmanFilter
export MovingHorizonEstimator
export default_nint, initstate!
export PredictiveController, ExplicitMPC, LinMPC, NonLinMPC, setconstraint!, moveinput!
export TranscriptionMethod, SingleShooting, MultipleShooting
export SimResult, getinfo, sim!

include("general.jl")
include("sim_model.jl")
include("state_estim.jl")
include("predictive_control.jl")
include("plot_sim.jl")

@setup_workload begin
    # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the
    # size of the precompile file and potentially make loading faster.
    @compile_workload begin
        # all calls in this block will be precompiled, regardless of whether
        # they belong to your package or not (on Julia 1.8 and higher)
        include("precompile.jl")
    end
end

end