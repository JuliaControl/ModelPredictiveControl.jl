@doc raw"""
Abstract supertype of all state estimators.

---

    (estim::StateEstimator)(d=[]) -> ŷ

Functor allowing callable `StateEstimator` object as an alias for [`evaloutput`](@ref).

# Examples
```jldoctest
julia> kf = KalmanFilter(setop!(LinModel(tf(3, [10, 1]), 2), yop=[20]), direct=false);

julia> ŷ = kf() 
1-element Vector{Float64}:
 20.0
```
"""
abstract type StateEstimator{NT<:Real} end

const IntVectorOrInt = Union{Int, Vector{Int}}

include("estimator/construct.jl")
include("estimator/execute.jl")
include("estimator/kalman.jl")
include("estimator/luenberger.jl")
include("estimator/mhe.jl")
include("estimator/internal_model.jl")
include("estimator/manual.jl")

function Base.show(io::IO, estim::StateEstimator)
    model = estim.model
    nu, nd = model.nu, model.nd
    nx̂, nym, nyu = estim.nx̂, estim.nym, estim.nyu
    n = maximum(ndigits.((nu, nx̂, nym, nyu, nd))) + 1
    println(io, "$(nameof(typeof(estim))) estimator with a sample time Ts = $(model.Ts) s:")
    println(io, "├ model: $(nameof(typeof(estim.model)))")
    println(io, "└ dimensions:")
    print_estim_dim(io, estim, n)
end

"Print the overall dimensions of the state estimator `estim` with left padding `n`."
function print_estim_dim(io::IO, estim::StateEstimator, n)
    nu, nd = estim.model.nu, estim.model.nd
    nx̂, nym, nyu = estim.nx̂, estim.nym, estim.nyu
    niu, niym = sum(estim.nint_u), sum(estim.nint_ym)
    println(io, "  ├$(lpad(nu, n)) manipulated inputs u ($niu integrating states)")
    println(io, "  ├$(lpad(nx̂, n)) estimated states x̂")
    println(io, "  ├$(lpad(nym, n)) measured outputs ym ($niym integrating states)")
    println(io, "  ├$(lpad(nyu, n)) unmeasured outputs yu")
    print(io,   "  └$(lpad(nd, n)) measured disturbances d")
end