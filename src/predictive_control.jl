@doc raw"""
Abstract supertype of all predictive controllers.

---

    (mpc::PredictiveController)(ry, d=[]; kwargs...) -> u

Functor allowing callable `PredictiveController` object as an alias for [`moveinput!`](@ref).

# Examples
```jldoctest
julia> mpc = LinMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1000, Hc=1);

julia> u = mpc([5]); round.(u, digits=3)
1-element Vector{Float64}:
 1.0
```

"""
abstract type PredictiveController{NT<:Real} end

const DEFAULT_HP0 = 10
const DEFAULT_HC  = 2
const DEFAULT_MWT = 1.0
const DEFAULT_NWT = 0.1
const DEFAULT_LWT = 0.0
const DEFAULT_CWT = 1e5
const DEFAULT_EWT = 0.0

"Include all the data for the constraints of [`PredictiveController`](@ref)"
struct ControllerConstraint{NT<:Real}
    ẽx̂      ::Matrix{NT}
    fx̂      ::Vector{NT}
    gx̂      ::Matrix{NT}
    jx̂      ::Matrix{NT}
    kx̂      ::Matrix{NT}
    vx̂      ::Matrix{NT}
    Umin    ::Vector{NT}
    Umax    ::Vector{NT}
    ΔŨmin   ::Vector{NT}
    ΔŨmax   ::Vector{NT}
    Ymin    ::Vector{NT}
    Ymax    ::Vector{NT}
    x̂min    ::Vector{NT}
    x̂max    ::Vector{NT}
    A_Umin  ::Matrix{NT}
    A_Umax  ::Matrix{NT}
    A_ΔŨmin ::Matrix{NT}
    A_ΔŨmax ::Matrix{NT}
    A_Ymin  ::Matrix{NT}
    A_Ymax  ::Matrix{NT}
    A_x̂min  ::Matrix{NT}
    A_x̂max  ::Matrix{NT}
    A       ::Matrix{NT}
    b       ::Vector{NT}
    i_b     ::BitVector
    C_ymin  ::Vector{NT}
    C_ymax  ::Vector{NT}
    c_x̂min  ::Vector{NT}
    c_x̂max  ::Vector{NT}
    i_g     ::BitVector
end

"""
    setstate!(mpc::PredictiveController, x̂)

Set the estimate at `mpc.estim.x̂`.
"""
setstate!(mpc::PredictiveController, x̂) = (setstate!(mpc.estim, x̂); return mpc)

function Base.show(io::IO, mpc::PredictiveController)
    Hp, Hc = mpc.Hp, mpc.Hc
    nu, nd = mpc.estim.model.nu, mpc.estim.model.nd
    nx̂, nym, nyu = mpc.estim.nx̂, mpc.estim.nym, mpc.estim.nyu
    n = maximum(ndigits.((Hp, Hc, nu, nx̂, nym, nyu, nd))) + 1
    println(io, "$(typeof(mpc).name.name) controller with a sample time Ts = "*
                "$(mpc.estim.model.Ts) s, $(solver_name(mpc.optim)) optimizer, "*
                "$(typeof(mpc.estim).name.name) estimator and:")
    println(io, "$(lpad(Hp, n)) prediction steps Hp")
    println(io, "$(lpad(Hc, n)) control steps Hc")
    print_estim_dim(io, mpc.estim, n)
end

"Functor allowing callable `PredictiveController` object as an alias for `moveinput!`."
function (mpc::PredictiveController)(
    ry::Vector = mpc.estim.model.yop, 
    d ::Vector = empty(mpc.estim.x̂);
    kwargs...
)
    return moveinput!(mpc, ry, d; kwargs...)
end

include("controller/construct.jl")
include("controller/execute.jl")
include("controller/explicitmpc.jl")
include("controller/linmpc.jl")
include("controller/nonlinmpc.jl")
