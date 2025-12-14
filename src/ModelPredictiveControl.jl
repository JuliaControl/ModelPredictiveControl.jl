module ModelPredictiveControl

using PrecompileTools 
using LinearAlgebra, SparseArrays
using Random: randn
using StableRNGs: StableRNG

using RecipesBase

using DifferentiationInterface: ADTypes.AbstractADType, AutoForwardDiff
using DifferentiationInterface: AutoSparse, SecondOrder
using DifferentiationInterface: gradient, jacobian, hessian
using DifferentiationInterface: value_and_gradient, value_and_jacobian
using DifferentiationInterface: value_gradient_and_hessian
using DifferentiationInterface: gradient!, value_and_gradient!, prepare_gradient
using DifferentiationInterface: jacobian!, value_and_jacobian!, prepare_jacobian 
using DifferentiationInterface: hessian!, value_gradient_and_hessian!, prepare_hessian
using DifferentiationInterface: Constant, Cache
using SparseConnectivityTracer: TracerSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm, sparsity_pattern
using SparseMatrixColorings: NaturalOrder, LargestFirst, SmallestLast
using SparseMatrixColorings: IncidenceDegree, DynamicLargestFirst, RandomOrder

import ProgressLogging

import ForwardDiff

import ControlSystemsBase
import ControlSystemsBase: ss, tf, delay
import ControlSystemsBase: Continuous, Discrete
import ControlSystemsBase: StateSpace, TransferFunction, DelayLtiSystem, LTISystem
import ControlSystemsBase: iscontinuous, isdiscrete, sminreal, minreal, c2d, d2c

import JuMP
import JuMP: MOIU, MOI, GenericModel, Model, optimizer_with_attributes, register
import JuMP: @variable, @operator, @constraint, @objective

import OSQP, Ipopt

export SimModel, LinModel, NonLinModel
export DiffSolver, RungeKutta, ForwardEuler
export setop!, setname!
export setstate!, setmodel!, preparestate!, updatestate!, evaloutput, linearize, linearize!
export savetime!, periodsleep
export StateEstimator, InternalModel, Luenberger
export SteadyKalmanFilter, KalmanFilter, UnscentedKalmanFilter, ExtendedKalmanFilter
export MovingHorizonEstimator
export ManualEstimator
export default_nint, initstate!
export PredictiveController, ExplicitMPC, LinMPC, NonLinMPC, setconstraint!, moveinput!
export TranscriptionMethod, SingleShooting, MultipleShooting, TrapezoidalCollocation
export SimResult, getinfo, sim!

export hi

struct NonExt end
struct Ext end

hi(::NonExt) = println("hello wold with a number")

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