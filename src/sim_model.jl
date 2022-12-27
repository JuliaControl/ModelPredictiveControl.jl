@doc raw"""
Abstract supertype of [`LinModel`](@ref) and [`NonLinModel`](@ref) types.

---

    (model::SimModel)(d=Float64[])

Functor allowing callable `SimModel` object as an alias for [`evaloutput`](@ref).

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_)->-x + u, (x,_)->x .+ 20, 10, 1, 1, 1);

julia> y = model()
1-element Vector{Float64}:
 20.0
```
"""
abstract type SimModel end

struct LinModel <: SimModel
    A   ::Matrix{Float64}
    Bu  ::Matrix{Float64}
    C   ::Matrix{Float64}
    Bd  ::Matrix{Float64}
    Dd  ::Matrix{Float64}
    x::Vector{Float64}
    f::Function
    h::Function
    Ts::Float64
    nu::Int
    nx::Int
    ny::Int
    nd::Int
    uop::Vector{Float64}
    yop::Vector{Float64}
    dop::Vector{Float64}
    function LinModel(A, Bu, C, Bd, Dd, Ts, nu, nx, ny, nd)
        size(A)  == (nx,nx) || error("A size must be $((nx,nx))")
        size(Bu) == (nx,nu) || error("Bu size must be $((nx,nu))")
        size(C)  == (ny,nx) || error("C size must be $((ny,nx))")
        size(Bd) == (nx,nd) || error("Bd size must be $((nx,nd))")
        size(Dd) == (ny,nd) || error("Dd size must be $((ny,nd))")
        Ts > 0 || error("Sampling time Ts must be positive")
        f(x,u,d) = A*x + Bu*u + Bd*d
        h(x,d) = C*x + Dd*d
        uop = zeros(nu)
        yop = zeros(ny)
        dop = zeros(nd)
        x = zeros(nx)
        return new(A, Bu, C, Bd, Dd, x, f, h, Ts, nu, nx, ny, nd, uop, yop, dop)
    end
end


const IntRangeOrVector = Union{UnitRange{Int}, Vector{Int}}

@doc raw"""
    LinModel(sys::StateSpace[, Ts]; i_u=1:size(sys,2), i_d=Int[])

Construct a `LinModel` from state-space model `sys` with sampling time `Ts` in second.

`Ts` can be omitted when `sys` is discrete-time. Its state-space matrices are:
```math
\begin{aligned}
    \mathbf{x}(k+1) &= \mathbf{A x}(k) + \mathbf{B z}(k) \\
    \mathbf{y}(k)   &= \mathbf{C x}(k) + \mathbf{D z}(k)
\end{aligned}
```
with the state ``\mathbf{x}`` and output ``\mathbf{y}`` vectors. The ``\mathbf{z}`` vector 
comprises the manipulated inputs ``\mathbf{u}`` and measured disturbances ``\mathbf{d}``, 
in any order. `i_u` provides the indices of ``\mathbf{z}`` that are manipulated, and `i_d`, 
the measured disturbances. See Extended Help if `sys` is continuous-time.

See also [`ss`](https://juliacontrol.github.io/ControlSystems.jl/stable/lib/constructors/#ControlSystemsBase.ss),
[`tf`](https://juliacontrol.github.io/ControlSystems.jl/stable/lib/constructors/#ControlSystemsBase.tf).

# Examples
```jldoctest
julia> model = LinModel(ss(0.4, 0.2, 0.3, 0, 0.1))
Discrete-time linear model with a sample time Ts = 0.1 s and:
 1 manipulated inputs u
 1 states x
 1 outputs y
 0 measured disturbances d
```

# Extended Help
State-space matrices are similar if `sys` is continuous (replace ``\mathbf{x}(k+1)`` with 
``\mathbf{ẋ}(t)`` and ``k`` with ``t`` on the LHS). In such a case, it's discretized with 
[`c2d`](https://juliacontrol.github.io/ControlSystems.jl/stable/lib/constructors/#ControlSystemsBase.c2d)
and `:zoh` for manipulated inputs, and `:tustin`, for measured disturbances. 

The constructor transforms the system to a more practical form (``\mathbf{D_u=0}`` because 
of the zero-order hold):
```math
\begin{aligned}
    \mathbf{x}(k+1) &=  \mathbf{A x}(k) + \mathbf{B_u u}(k) + \mathbf{B_d d}(k) \\
    \mathbf{y}(k)   &=  \mathbf{C x}(k) + \mathbf{D_d d}(k)
\end{aligned}
```
"""
function LinModel(
    sys::StateSpace,
    Ts::Union{Real,Nothing} = nothing;
    i_u::IntRangeOrVector = 1:size(sys,2),
    i_d::IntRangeOrVector = Int[]
    )
    if !isempty(i_d)
        # common indexes in i_u and i_d are interpreted as measured disturbances d :
        i_u = collect(i_u);
        map(i -> deleteat!(i_u, i_u .== i), i_d);
    end
    if length(unique(i_u)) ≠ length(i_u)
        error("Manipulated input indices i_u should contains valid and unique indices")
    end
    if length(unique(i_d)) ≠ length(i_d)
        error("Measured disturbances indices i_d should contains valid and unique indices")
    end
    sysu = sminreal(sys[:,i_u])  # remove states associated to measured disturbances d
    sysd = sminreal(sys[:,i_d])  # remove states associated to manipulates inputs u
    if !iszero(sysu.D)
        error("State matrix D must be 0 for columns associated to manipulated inputs u")
    end
    if iscontinuous(sys)
        isnothing(Ts) && error("Sample time Ts must be specified if sys is continuous")
        # manipulated inputs : zero-order hold discretization 
        sysu_dis = c2d(sysu,Ts,:zoh);
        # measured disturbances : tustin discretization (continuous signals with ADCs)
        sysd_dis = c2d(sysd,Ts,:tustin)
    else
        if !isnothing(Ts)
            #TODO: Resample discrete system instead of throwing an error
            sys.Ts == Ts || error("Sample time Ts must be identical to sys.Ts")
        else
            Ts = sys.Ts
        end
        sysu_dis = sysu
        sysd_dis = sysd     
    end
    sys_dis = sminreal([sysu_dis sysd_dis]) # merge common poles if possible
    nx = size(sys_dis.A,1)
    nu = length(i_u)
    ny = size(sys_dis,1)
    nd = length(i_d)
    A   = sys_dis.A
    Bu  = sys_dis.B[:,1:nu]
    Bd  = sys_dis.B[:,nu+1:end]
    C   = sys_dis.C;
    Dd  = sys_dis.D[:,nu+1:end]
    return LinModel(A, Bu, C, Bd, Dd, Ts, nu, nx, ny, nd)
end

@doc raw"""
    LinModel(sys::TransferFunction[, Ts]; i_u=1:size(sys,2), i_d=Int[])

Convert to minimal realization state-space when `sys` is a transfer function.

`sys` is equal to ``\frac{\mathbf{y}(s)}{\mathbf{z}(s)}`` for continuous-time, and 
``\frac{\mathbf{y}(z)}{\mathbf{z}(z)}``, for discrete-time.

# Examples
```jldoctest
julia> model = LinModel([tf(3, [30, 1]) tf(-2, [5, 1])], 0.5, i_d=[2])
Discrete-time linear model with a sample time Ts = 0.5 s and:
 1 manipulated inputs u
 2 states x
 1 outputs y
 1 measured disturbances d
```
"""
function LinModel(sys::TransferFunction, Ts::Union{Real,Nothing} = nothing; kwargs...)
    sys_min = minreal(ss(sys)) # remove useless states with pole-zero cancellation
    return LinModel(sys_min, Ts; kwargs...)
end


"""
    LinModel(sys::DelayLtiSystem, Ts; i_u=1:size(sys,2), i_d=Int[])

Discretize with zero-order hold when `sys` is a continuous system with delays.

The delays must be multiples of the sample time `Ts`.

# Examples
```jldoctest
julia> model = LinModel(tf(4, [10, 1])*delay(2), 0.5)
Discrete-time linear model with a sample time Ts = 0.5 s and:
 1 manipulated inputs u
 5 states x
 1 outputs y
 0 measured disturbances d
```
"""
function LinModel(sys::DelayLtiSystem, Ts::Real; kwargs...)
    sys_dis = minreal(c2d(sys, Ts, :zoh)) # c2d only supports :zoh for DelayLtiSystem
    return LinModel(sys_dis, Ts; kwargs...)
end


"Evaluate the steady-state vector when `model` is a [`LinModel`](@ref)."
function steadystate(model::LinModel, u, d=Float64[])
    return (I - model.A) \ (model.Bu*(u - model.uop) + model.Bd*(d - model.dop))
end


@doc raw"""
    NonLinModel(f, h, Ts::Real, nu::Int, nx::Int, ny::Int, nd::Int=0)

Construct a `NonLinModel` from discrete-time state-space functions `f` and `h`.

The state update ``\mathbf{f}`` and output ``\mathbf{h}`` functions are defined as :
```math
    \begin{aligned}
    \mathbf{x}(k+1) &= \mathbf{f}\Big( \mathbf{x}(k), \mathbf{u}(k), \mathbf{d}(k) \Big) \\
    \mathbf{y}(k)   &= \mathbf{h}\Big( \mathbf{x}(k), \mathbf{d}(k) \Big)
    \end{aligned}
```
`Ts` is the sampling time in second. `nu`, `nx`, `ny` and `nd` are the respective number of 
manipulated inputs, states, outputs and measured disturbances. 

!!! tip
    Replace the `d` argument with `_` if `nd = 0` (see Examples below).  

Nonlinear continuous-time state-space functions are not supported for now. In such a case, 
manually call a differential equation solver in the `f` function (e.g.: Euler method).

See also [`LinModel`](@ref).

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_)->-x+u, (x,_)->2x, 10, 1, 1, 1)
Discrete-time nonlinear model with a sample time Ts = 10.0 s and:
 1 manipulated inputs u
 1 states x
 1 outputs y
 0 measured disturbances d
```
"""
struct NonLinModel <: SimModel
    x::Vector{Float64}
    f::Function
    h::Function
    Ts::Float64
    nu::Int
    nx::Int
    ny::Int
    nd::Int
    uop::Vector{Float64}
    yop::Vector{Float64}
    dop::Vector{Float64}
    function NonLinModel(
            f, h, Ts::Real, 
            nu::Int, nx::Int, ny::Int, nd::Int = 0; 
            x0 = nothing
    )
        Ts > 0 || error("Sampling time Ts must be positive")
        validate_fcts(f, h)
        uop = zeros(nu)
        yop = zeros(ny)
        dop = zeros(nd)
        x = zeros(nx)
        return new(x, f, h, Ts, nu, nx, ny, nd, uop, yop, dop)
    end
end

function validate_fcts(f, h)
    fargsvalid1 = hasmethod(f,
        Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    )
    fargsvalid2 = hasmethod(f,
        Tuple{Vector{ComplexF64}, Vector{Float64}, Vector{Float64}}
    )
    if !fargsvalid1 && !fargsvalid2
        error("state function has no method of type "*
            "f(x::Vector{Float64}, u::Vector{Float64}, d::Vector{Float64}) or "*
            "f(x::Vector{ComplexF64}, u::Vector{Float64}, d::Vector{Float64})")
    end
    hargsvalid1 = hasmethod(h,Tuple{Vector{Float64}, Vector{Float64}})
    hargsvalid2 = hasmethod(h,Tuple{Vector{ComplexF64}, Vector{Float64}})
    if !hargsvalid1 && !hargsvalid2
        error("output function has no method of type "*
            "h(x::Vector{Float64}, d::Vector{Float64}) or "*
            "h(x::Vector{ComplexF64}, d::Vector{Float64})")
    end
end


@doc raw"""
    setop!(model::SimModel; uop=nothing, yop=nothing, dop=nothing)

Set `model` inputs `uop`, outputs `yop` and measured disturbances `dop` operating points.

The state-space model with operating points (a.k.a. nominal values) is:
```math
\begin{aligned}
    \mathbf{x}(k+1) &=  \mathbf{A x}(k) + \mathbf{B_u u_0}(k) + \mathbf{B_d d_0}(k) \\
    \mathbf{y_0}(k) &=  \mathbf{C x}(k) + \mathbf{D_d d_0}(k)
\end{aligned}
```
in which the `uop`, `yop` and `dop` vectors evaluate :
```math
\begin{aligned}
    \mathbf{u_0}(k) &= \mathbf{u}(k) - \mathbf{u_{op}} \\
    \mathbf{y_0}(k) &= \mathbf{y}(k) - \mathbf{y_{op}} \\
    \mathbf{d_0}(k) &= \mathbf{d}(k) - \mathbf{d_{op}} 
\end{aligned}
```
The structure is similar if `model` is a `NonLinModel`:
```math
\begin{aligned}
    \mathbf{x}(k+1) &= \mathbf{f}\Big(\mathbf{x}(k), \mathbf{u_0}(k), \mathbf{d_0}(k)\Big)\\
    \mathbf{y_0}(k) &= \mathbf{h}\Big(\mathbf{x}(k), \mathbf{d_0}(k)\Big)
\end{aligned}
```

# Examples
```jldoctest
julia> model = setop!(LinModel(tf(3, [10, 1]), 2), uop=[50], yop=[20])
Discrete-time linear model with a sample time Ts = 2.0 s and:
 1 manipulated inputs u
 1 states x
 1 outputs y
 0 measured disturbances d
```

"""
function setop!(model::SimModel; uop = nothing, yop = nothing, dop = nothing)
    if !isnothing(uop) 
        size(uop) == (model.nu,) || error("uop size must be $((model.nu,))")
        model.uop[:] = uop
    end
    if !isnothing(yop)
        size(yop) == (model.ny,) || error("yop size must be $((model.ny,))")
        model.yop[:] = yop
    end
    if !isnothing(dop)
        size(dop) == (model.nd,) || error("dop size must be $((model.nd,))")
        model.dop[:] = dop
    end
    return model
end


"""
    setstate!(model::SimModel, x)

Set `model.x` states to values specified by `x`. 
"""
function setstate!(model::SimModel, x)
    size(x) == (model.nx,) || error("x size must be $((model.nx,))")
    model.x[:] = x
    return model
end

function Base.show(io::IO, model::SimModel)
    println(io, "Discrete-time $(typestr(model)) model with "*
                "a sample time Ts = $(model.Ts) s and:")
    println(io, " $(model.nu) manipulated inputs u")
    println(io, " $(model.nx) states x")
    println(io, " $(model.ny) outputs y")
    print(io,   " $(model.nd) measured disturbances d")
end
typestr(model::LinModel) = "linear"
typestr(model::NonLinModel) = "nonlinear"

"""
    updatestate!(model::SimModel, u, d=Float64[])

Update `model.x` states with current inputs `u` and measured disturbances `d`.
"""
function updatestate!(model::SimModel, u, d=Float64[])
    model.x[:] = model.f(model.x, u - model.uop, d - model.dop)
    return model.x
end

"""
    evaloutput(model::SimModel, d=Float64[])

Evaluate `SimModel` outputs `y` from `model.x` states and measured disturbances `d`.
"""
evaloutput(model::SimModel, d=Float64[]) = model.h(model.x, d - model.dop) + model.yop


"Functor allowing callable `SimModel` object as an alias for `evaloutput`."
(model::SimModel)(d=Float64[]) = evaloutput(model::SimModel, d)
