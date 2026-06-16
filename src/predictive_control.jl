@doc raw"""
Abstract supertype of all predictive controllers.

---

    (mpc::PredictiveController)(ry, d=[]; kwargs...) -> u

Functor allowing callable `PredictiveController` object as an alias for [`moveinput!`](@ref).

# Examples
```jldoctest
julia> mpc = LinMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1000, Hc=1, direct=false);

julia> u = mpc([5]); round.(u, digits=3)
1-element Vector{Float64}:
 1.0
```

"""
abstract type PredictiveController{NT<:Real} end

include("controller/transcription.jl")
include("controller/construct.jl")
include("controller/execute.jl")
include("controller/explicitmpc.jl")
include("controller/linmpc.jl")
include("controller/nonlinmpc.jl")

function Base.show(io::IO, mpc::PredictiveController)
    estim, model = mpc.estim, mpc.estim.model
    Hp, Hc = mpc.Hp, mpc.Hc
    nu, nd = model.nu, model.nd
    nx̂, nym, nyu = estim.nx̂, estim.nym, estim.nyu
    other_dims = get_other_dims(estim)
    n = maximum(ndigits.((Hp, Hc, nu, nx̂, nym, nyu, nd, other_dims...))) + 1
    println(io, "$(nameof(typeof(mpc))) controller with a sample time Ts = $(model.Ts) s:")
    println(io, "├ estimator: $(nameof(typeof(mpc.estim)))")
    println(io, "├ model: $(nameof(typeof(model)))")
    println(io, "├ optimizer: $(JuMP.solver_name(mpc.optim)) ")
    println(io, "├ transcription: $(nameof(typeof(mpc.transcription)))")
    print_backends(io, mpc)
    println(io, "└ dimensions:")
    println(io, "  ├ variable: ")
    println(io, "  │ ├$(lpad(Hp, n)) prediction steps Hp")
    println(io, "  │ ├$(lpad(Hc, n)) control steps Hc")
    print_estim_dim(io, mpc.estim, n, firstchars="  │")
    println(io) # add a linebreak since `print_estim_dim` ends with a `print` (w/o ln) 
    print_optim_dim(io, mpc)
end

"No differentiation backends to print for a `PredictiveController` by default."
print_backends(::IO, ::PredictiveController) = nothing

"No dimensions related to the optimization problem by default."
print_optim_dim(io::IO, ::PredictiveController) =  println(io, "  └ optimization: nothing")

"Functor allowing callable `PredictiveController` object as an alias for `moveinput!`."
function (mpc::PredictiveController)(
    ry::AbstractVector = mpc.estim.model.yop, 
    d ::AbstractVector = mpc.estim.buffer.empty;
    kwargs...
)
    return moveinput!(mpc, ry, d; kwargs...)
end