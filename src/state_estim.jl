@doc raw"""
Abstract supertype of all state estimators.

---

    (estim::StateEstimator)(d=[]) -> ŷ

Functor allowing callable `StateEstimator` object as an alias for [`evaloutput`](@ref).

# Examples
```jldoctest
julia> kf = KalmanFilter(setop!(LinModel(tf(3, [10, 1]), 2), yop=[20]));

julia> ŷ = kf() 
1-element Vector{Float64}:
 20.0
```
"""
abstract type StateEstimator{NT<:Real} end

const IntVectorOrInt = Union{Int, Vector{Int}}

"""
    setstate!(estim::StateEstimator, x̂)

Set `estim.x̂` states to values specified by `x̂`. 
"""
function setstate!(estim::StateEstimator, x̂)
    size(x̂) == (estim.nx̂,) || error("x̂ size must be $((estim.nx̂,))")
    estim.x̂[:] = x̂
    return estim
end

function Base.show(io::IO, estim::StateEstimator)
    nu, nd = estim.model.nu, estim.model.nd
    nx̂, nym, nyu = estim.nx̂, estim.nym, estim.nyu
    n = maximum(ndigits.((nu, nx̂, nym, nyu, nd))) + 1
    println(io, "$(typeof(estim).name.name) estimator with a sample time "*
                "Ts = $(estim.model.Ts) s, $(typeof(estim.model).name.name) and:")
    print_estim_dim(io, estim, n)
end

"Print the overall dimensions of the state estimator `estim` with left padding `n`."
function print_estim_dim(io::IO, estim::StateEstimator, n)
    nu, nd = estim.model.nu, estim.model.nd
    nx̂, nym, nyu = estim.nx̂, estim.nym, estim.nyu
    println(io, "$(lpad(nu, n)) manipulated inputs u ($(sum(estim.nint_u)) integrating states)")
    println(io, "$(lpad(nx̂, n)) states x̂")
    println(io, "$(lpad(nym, n)) measured outputs ym ($(sum(estim.nint_ym)) integrating states)")
    println(io, "$(lpad(nyu, n)) unmeasured outputs yu")
    print(io,   "$(lpad(nd, n)) measured disturbances d")
end

include("estimator/construct.jl")
include("estimator/execute.jl")
include("estimator/kalman.jl")
include("estimator/luenberger.jl")
include("estimator/mhe.jl")
include("estimator/internal_model.jl")

"""
    evalŷ(estim::StateEstimator, _ , d) -> ŷ

Evaluate [`StateEstimator`](@ref) output `ŷ` from measured disturbance `d` and `estim.x̂`.

Second argument is ignored, except for [`InternalModel`](@ref).
"""
evalŷ(estim::StateEstimator, _ , d) = evaloutput(estim, d)

@doc raw"""
    evalŷ(estim::InternalModel, ym, d) -> ŷ

Get [`InternalModel`](@ref) output `ŷ` from current measured outputs `ym` and dist. `d`.

[`InternalModel`](@ref) estimator needs current measured outputs ``\mathbf{y^m}(k)`` to 
estimate its outputs ``\mathbf{ŷ}(k)``, since the strategy imposes that 
``\mathbf{ŷ^m}(k) = \mathbf{y^m}(k)`` is always true.
"""
function evalŷ(estim::InternalModel, ym, d)
    ŷ = h(estim.model, estim.x̂d, d - estim.model.dop) + estim.model.yop
    ŷ[estim.i_ym] = ym
    return ŷ
end
    