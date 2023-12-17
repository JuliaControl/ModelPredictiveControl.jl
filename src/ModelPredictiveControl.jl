module ModelPredictiveControl

using PrecompileTools

using LinearAlgebra, Random
using RecipesBase
using ControlSystemsBase
using ForwardDiff
using JuMP
using PreallocationTools
import OSQP, Ipopt


export SimModel, LinModel, NonLinModel
export setop!, setstate!, updatestate!, evaloutput, linearize
export StateEstimator, InternalModel, Luenberger
export SteadyKalmanFilter, KalmanFilter, UnscentedKalmanFilter, ExtendedKalmanFilter
export MovingHorizonEstimator
export default_nint, initstate!
export PredictiveController, ExplicitMPC, LinMPC, NonLinMPC, setconstraint!, moveinput!
export SimResult, getinfo, sim!

"Termination status that means 'no solution available'."
const FATAL_STATUSES = [
    INFEASIBLE, DUAL_INFEASIBLE, LOCALLY_INFEASIBLE, INFEASIBLE_OR_UNBOUNDED, 
    NUMERICAL_ERROR, INVALID_MODEL, INVALID_OPTION, INTERRUPTED, 
    OTHER_ERROR
]

"Evaluate the quadratic programming objective function `0.5x'*H*x + q'*x` at `x`."
obj_quadprog(x, H, q) = 0.5*dot(x, H, x) + q'*x  # dot(x, H, x) is faster than x'*H*x

"Generate a block diagonal matrix repeating `n` times the matrix `A`."
repeatdiag(A, n::Int) = kron(I(n), A)

include("sim_model.jl")
include("state_estim.jl")
include("predictive_control.jl")
include("plot_sim.jl")

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