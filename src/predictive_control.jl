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
    Hp, Hc, nϵ = mpc.Hp, mpc.Hc, mpc.nϵ
    nu, nd = mpc.estim.model.nu, mpc.estim.model.nd
    nx̂, nym, nyu = mpc.estim.nx̂, mpc.estim.nym, mpc.estim.nyu
    n = maximum(ndigits.((Hp, Hc, nu, nx̂, nym, nyu, nd))) + 1
    println(io, "$(nameof(typeof(mpc))) controller with a sample time Ts = "*
                "$(mpc.estim.model.Ts) s, $(JuMP.solver_name(mpc.optim)) optimizer, "*
                "$(nameof(typeof(mpc.transcription))) transcription, "*
                "$(nameof(typeof(mpc.estim))) estimator and:")
    println(io, "$(lpad(Hp, n)) prediction steps Hp")
    println(io, "$(lpad(Hc, n)) control steps Hc")
    println(io, "$(lpad(nϵ, n)) slack variable ϵ (control constraints)")
    print_estim_dim(io, mpc.estim, n)
end

"Functor allowing callable `PredictiveController` object as an alias for `moveinput!`."
function (mpc::PredictiveController)(
    ry::Vector = mpc.estim.model.yop, 
    d ::Vector = mpc.estim.buffer.empty;
    kwargs...
)
    return moveinput!(mpc, ry, d; kwargs...)
end