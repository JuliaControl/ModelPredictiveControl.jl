abstract type SimModel end

struct LinModel <: SimModel
    A   ::Matrix{Float64}
    Bu  ::Matrix{Float64}
    C   ::Matrix{Float64}
    Bd  ::Matrix{Float64}
    Dd  ::Matrix{Float64}
    f   ::Function
    h   ::Function
    Ts  ::Float64
    nu  ::Int
    nx  ::Int
    ny  ::Int
    nd  ::Int
    uop::Vector{Float64}
    yop::Vector{Float64}
    dop::Vector{Float64}
    function LinModel(A,Bu,C,Bd,Dd,Ts,nu,nx,ny,nd)
        size(A)  == (nx,nx) || error("A size must be $((nx,nx))")
        size(Bu) == (nx,nu) || error("Bu size must be $((nx,nu))")
        size(C)  == (ny,nx) || error("C size must be $((ny,nx))")
        size(Bd) == (nx,nd) || error("Bd size must be $((nx,nd))")
        size(Dd) == (ny,nd) || error("Dd size must be $((ny,nd))")
        Ts > 0 || error("Sampling time Ts must be positive")
        f(x,u,d)  = A*x + Bu*u + Bd*d
        h(x,d)    = C*x + Dd*d
        uop = zeros(nu,)
        yop = zeros(ny,)
        dop = zeros(nd,)
        return new(A,Bu,C,Bd,Dd,f,h,Ts,nu,nx,ny,nd,uop,yop,dop)
    end
end


IntRangeOrVector = Union{UnitRange{Int}, Vector{Int}}

@doc raw"""
    LinModel(sys::StateSpace, Ts=NaN; i_u=1:size(sys,2), i_d=Int[])

Construct a `LinModel` from state-space model `sys` with sampling time `Ts` in second.

`Ts` can be omitted when `sys` is discrete-time. Its state-space matrices are:
```math
\begin{align*}
    \mathbf{x}(k+1) &= \mathbf{A} \mathbf{x}(k) + \mathbf{B} \mathbf{z}(k) \\
    \mathbf{y}(k)   &= \mathbf{C} \mathbf{x}(k) + \mathbf{D} \mathbf{z}(k)
\end{align*}
```
with the state ``\mathbf{x}`` and output ``\mathbf{y}`` vectors. The ``\mathbf{z}`` vector 
comprises the manipulated inputs ``\mathbf{u}`` and measured disturbances ``\mathbf{d}``, 
in any order. `i_u` provides the indices of ``\mathbf{z}`` that are manipulated, and `i_d`, 
the measured disturbances. The state-space matrices are similar if `sys` is continuous-time 
(replace ``\mathbf{x}(k+1)`` with ``\dot{\mathbf{x}}(t)``). In such a case, it's discretized 
with [`c2d`](https://juliacontrol.github.io/ControlSystems.jl/latest/lib/constructors/#ControlSystemsBase.c2d)
and `:zoh` for manipulated inputs, and `:tustin`, for measured disturbances. 
    
The constructor transforms the system to a more practical form (**Dᵤ = 0** because of the 
zero-order hold):
```math
\begin{align*}
    \mathbf{x}(k+1) &=  \mathbf{A} \mathbf{x}(k) + 
                        \mathbf{B_u} \mathbf{u}(k) + \mathbf{B_d} \mathbf{d}(k) \\
    \mathbf{y}(k)   &=  \mathbf{C} \mathbf{x}(k) + \mathbf{D_d} \mathbf{d}(k)
\end{align*}
```

See also [`ss`](https://juliacontrol.github.io/ControlSystems.jl/latest/lib/constructors/#ControlSystemsBase.ss),
[`tf`](https://juliacontrol.github.io/ControlSystems.jl/latest/lib/constructors/#ControlSystemsBase.tf).

# Examples
```jldoctest
julia> model = LinModel(ss(0.4, 0.2, 0.3, 0, 0.1))
Discrete-time linear model with a sample time Ts = 0.1 s and:
- 1 manipulated inputs u
- 1 states x
- 1 outputs y
- 0 measured disturbances d
```
"""
function LinModel(
    sys::StateSpace,
    Ts::Real = NaN;
    i_u::IntRangeOrVector = 1:size(sys,2),
    i_d::IntRangeOrVector = Int[]
    )
    if ~isempty(i_d)
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
    if ~iszero(sysu.D)
        error("State matrix D must be 0 for columns associated to manipulated inputs u")
    end
    if iscontinuous(sys)
        isnan(Ts) && error("Sample time Ts must be specified if sys is continuous")
        # manipulated inputs : zero-order hold discretization 
        sysu_dis = c2d(sysu,Ts,:zoh);
        # measured disturbances : tustin discretization (continuous signals with ADCs)
        sysd_dis = c2d(sysd,Ts,:tustin)
    else
        if ~isnan(Ts)
            #TODO: Resample discrete system instead of throwing an error
            sys.Ts == Ts || error("Sample time Ts must be identical to sys.Ts")
        else
            Ts = sys.Ts
        end
        sysu_dis = sysu
        sysd_dis = sysd     
    end
    sys_dis = [sysu_dis sysd_dis]
    nx = size(sys_dis.A,1)
    nu = length(i_u)
    ny = size(sys_dis,1)
    nd = length(i_d)
    A   = sys_dis.A
    Bu  = sys_dis.B[:,1:nu]
    Bd  = sys_dis.B[:,nu+1:end]
    C   = sys_dis.C;
    Dd  = sys_dis.D[:,nu+1:end]
    return LinModel(A,Bu,C,Bd,Dd,Ts,nu,nx,ny,nd)
end

@doc raw"""
    LinModel(sys::TransferFunction, Ts=NaN; <keyword arguments>)

Convert to minimal realization state-space when `sys` is a transfer function.

`sys` is equal to ``\frac{\mathbf{y}(s)}{\mathbf{z}(s)}`` for continuous-time, and 
``\frac{\mathbf{y}(z)}{\mathbf{z}(z)}``, for discrete-time.

# Examples
```jldoctest
julia> model = LinModel([tf(3, [30, 1]) tf(-2, [5, 1])], 2, i_d=[2])
Discrete-time linear model with a sample time Ts = 2.0 s and:
- 1 manipulated inputs u
- 2 states x
- 1 outputs y
- 1 measured disturbances d
```
"""
function LinModel(sys::TransferFunction, Ts::Real = NaN; kwargs...)
    sys_min = minreal(ss(sys)) # remove useless states with pole-zero cancellation
    return LinModel(sys_min, Ts; kwargs...)
end

@doc raw"""
    NonLinModel(f, h, Ts::Real, nu::Int, nx::Int, ny::Int, nd::Int=0)

Construct a `NonLinModel` from discrete-time state-space functions `f` and `h`.

The state update ``\mathbf{f}`` and output ``\mathbf{h}`` functions are defined as :
```math
    \begin{align*}
    \mathbf{x}(k+1) &= \mathbf{f}\Big( \mathbf{x}(k), \mathbf{u}(k), \mathbf{d}(k) \Big) \\
    \mathbf{y}(k)   &= \mathbf{h}\Big( \mathbf{x}(k), \mathbf{d}(k) \Big)
    \end{align*}
```
`Ts` is the sampling time in second. `nu`, `nx`, `ny` and `nd` are the respective number of 
manipulated inputs, states, outputs and measured disturbances. Replace the `d` argument
with `_` if `nd=0` (see Examples below). Nonlinear continuous-time state-space functions 
are not supported for the time being. In such a case, manually call a differential equation 
solver in the `f` function (e.g.: Euler method).

See also [`LinModel`](@ref).

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_)->-x+u, (x,_)->2x, 10, 1 , 1 , 1)
Discrete-time nonlinear model with a sample time Ts = 10.0 s and:
- 1 manipulated inputs u
- 1 states x
- 1 outputs y
- 0 measured disturbances d
```
"""
struct NonLinModel <: SimModel
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
        f,
        h,
        Ts::Real,
        nu::Int,
        nx::Int,
        ny::Int,   
        nd::Int = 0;
        )
        Ts > 0 || error("Sampling time Ts must be positive")
        validate_fcts(f, h)
        uop = zeros(nu,)
        yop = zeros(ny,)
        dop = zeros(nd,)
        return new(f,h,Ts,nu,nx,ny,nd,uop,yop,dop)
    end
end

function validate_fcts(f, h)
    fargsvalid1 = hasmethod(f,
        Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    )
    fargsvalid2 = hasmethod(f,
        Tuple{Vector{ComplexF64}, Vector{Float64}, Vector{Float64}}
    )
    if ~fargsvalid1 && ~fargsvalid2
        error("state function has no method of type "*
            "f(x::Vector{Float64}, u::Vector{Float64}, d::Vector{Float64}) or "*
            "f(x::Vector{ComplexF64}, u::Vector{Float64}, d::Vector{Float64})")
    end
    hargsvalid1 = hasmethod(h,Tuple{Vector{Float64}, Vector{Float64}})
    hargsvalid2 = hasmethod(h,Tuple{Vector{ComplexF64}, Vector{Float64}})
    if ~hargsvalid1 && ~hargsvalid2
        error("output function has no method of type "*
            "h(x::Vector{Float64}, d::Vector{Float64}) or "*
            "h(x::Vector{ComplexF64}, d::Vector{Float64})")
    end
end


@doc raw"""
    setop!(model::SimModel; uop=Float64[], yop=Float64[], dop=Float64[])

Set `model` inputs `uop`, outputs `yop` and measured disturbances `dop` operating points.

The state-space model including operating points (a.k.a nominal values) is:
```math
\begin{align*}
    \mathbf{x}(k+1) &=  \mathbf{A} \mathbf{x}(k) + 
    \mathbf{B_u} \mathbf{u_0}(k) + \mathbf{B_d} \mathbf{d_0}(k) \\
    \mathbf{y_0}(k) &=  \mathbf{C} \mathbf{x}(k) + \mathbf{D_d} \mathbf{d_0}(k)
\end{align*}
```
where
```math
\begin{align*}
    \mathbf{u_0}(k) &= \mathbf{u}(k) - \mathbf{u_{op}}(k) \\
    \mathbf{y_0}(k) &= \mathbf{y}(k) - \mathbf{y_{op}}(k) \\
    \mathbf{d_0}(k) &= \mathbf{d}(k) - \mathbf{d_{op}}(k) 
\end{align*}
```

The structure is similar if `model` is a `NonLinModel`:
```math
\begin{align*}
    \mathbf{x}(k+1) &= \mathbf{f}\Big(\mathbf{x}(k), \mathbf{u_0}(k), \mathbf{d_0}(k)\Big)\\
    \mathbf{y_0}(k) &= \mathbf{h}\Big(\mathbf{x}(k), \mathbf{d_0}(k)\Big)
\end{align*}
```

# Examples
```jldoctest
julia> model = LinModel(tf(3, [10, 1]), 2);

julia> setop!(model, uop=[50], yop=[20])

```

"""
function setop!(
    model::SimModel;
    uop::Vector{<:Real} = Float64[],
    yop::Vector{<:Real} = Float64[],
    dop::Vector{<:Real} = Float64[]
)
    if ~isempty(uop) 
        size(uop)  == (model.nu,) || error("uop size must be $((model.nu,))")
        model.uop[:] = uop
    end
    if ~isempty(yop)
        size(yop)  == (model.ny,) || error("yop size must be $((model.ny,))")
        model.yop[:] = yop
    end
    if ~isempty(dop)
        size(dop)  == (model.nd,) || error("dop size must be $((model.nd,))")
        model.dop[:] = dop
    end
end

function Base.show(io::IO, model::SimModel)
    println(io,   "Discrete-time $(typestr(model)) model with "*
                "a sample time Ts = $(model.Ts) s and:")
    println(io, "- $(model.nu) manipulated inputs u")
    println(io, "- $(model.nx) states x")
    println(io, "- $(model.ny) outputs y")
    print(io,   "- $(model.nd) measured disturbances d")
end
typestr(model::LinModel) = "linear"
typestr(model::NonLinModel) = "nonlinear"

"""
    updatestate(sys::SimModel, x, u, d=Float64[])

Update states `x` of `sys` with current inputs `u` and measured disturbances `d`.
"""
updatestate(sys::SimModel, x, u, d=Float64[]) = sys.f(x, u-sys.uop, d-sys.dop)

"""
    evaloutput(sys::SimModel, x, d=Float64[])

Evaluate output `y` of `sys` with current state `x` and measured disturbances `d`.
"""
evaloutput(sys::SimModel, x, d=Float64[]) = sys.h(x, d-sys.dop)

