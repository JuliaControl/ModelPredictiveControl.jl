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

struct PredictiveControllerBuffer{NT<:Real}
    u ::Vector{NT}
    R̂y::Vector{NT}
    D̂ ::Vector{NT}
    Cy::Vector{NT}
    Cu::Vector{NT}
    empty::Vector{NT}
end

@doc raw"""
    PredictiveControllerBuffer{NT}(nu::Int, ny::Int, nd::Int)

Create a buffer for `PredictiveController` objects.

The buffer is used to store intermediate results during computation without allocating.
"""
function PredictiveControllerBuffer{NT}(nu::Int, ny::Int, nd::Int, Hp::Int) where NT <: Real
    u  = Vector{NT}(undef, nu)
    R̂y = Vector{NT}(undef, ny*Hp)
    D̂  = Vector{NT}(undef, nd*Hp)
    Cy = Vector{NT}(undef, ny*Hp)
    Cu = Vector{NT}(undef, nu*Hp)
    empty = Vector{NT}(undef, 0)
    return PredictiveControllerBuffer{NT}(u, R̂y, D̂, Cy, Cu, empty)
end

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
    println(io, "$(typeof(mpc).name.name) controller with a sample time Ts = "*
                "$(mpc.estim.model.Ts) s, $(JuMP.solver_name(mpc.optim)) optimizer, "*
                "$(typeof(mpc.estim).name.name) estimator and:")
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