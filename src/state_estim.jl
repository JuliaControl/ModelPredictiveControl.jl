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

struct StateEstimatorBuffer{NT<:Real}
    u ::Vector{NT}
    x̂ ::Vector{NT}
    ym::Vector{NT}
    ŷ ::Vector{NT}
    d ::Vector{NT}
    empty::Vector{NT}
end

@doc raw"""
    StateEstimatorBuffer(nx̂::Int, nym::Int) -> StateEstimatorBuffer{NT}

Create a buffer for `StateEstimator` objects for estimated states and measured outputs.

The buffer is used to store intermediate results during simulation without allocating.
"""
function StateEstimatorBuffer{NT}(
    nu::Int, nx̂::Int, nym::Int, ny::Int, nd::Int
) where NT <: Real
    u  = Vector{NT}(undef, nu)
    x̂  = Vector{NT}(undef, nx̂)
    ym = Vector{NT}(undef, nym)
    ŷ  = Vector{NT}(undef, ny)
    d  = Vector{NT}(undef, nd)
    empty = Vector{NT}(undef, 0)
    return StateEstimatorBuffer{NT}(u, x̂, ym, ŷ, d, empty)
end

const IntVectorOrInt = Union{Int, Vector{Int}}

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
    println(io, "$(lpad(nx̂, n)) estimated states x̂")
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

Evaluate [`StateEstimator`](@ref) output `ŷ` from measured disturbance `d` and `estim.x̂0`.

Second argument is ignored, except for [`InternalModel`](@ref).
"""
evalŷ(estim::StateEstimator, _ , d) = evaloutput(estim, d)
    