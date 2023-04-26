struct LinModel <: SimModel
    A   ::Matrix{Float64}
    Bu  ::Matrix{Float64}
    C   ::Matrix{Float64}
    Bd  ::Matrix{Float64}
    Dd  ::Matrix{Float64}
    x::Vector{Float64}
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
        uop = zeros(nu)
        yop = zeros(ny)
        dop = zeros(nd)
        x = zeros(nx)
        return new(A, Bu, C, Bd, Dd, x, Ts, nu, nx, ny, nd, uop, yop, dop)
    end
end

@doc raw"""
    LinModel(sys::StateSpace[, Ts]; i_u=1:size(sys,2), i_d=Int[])

Construct a linear model from state-space model `sys` with sampling time `Ts` in second.

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
the measured disturbances. See Extended Help if `sys` is continuous-time, or discrete-time
and `Ts ≠ sys.Ts`.

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
and `:zoh` for manipulated inputs, and `:tustin`, for measured disturbances. Lastly, if 
`sys` is discrete and the provided argument `Ts ≠ sys.Ts`, the system is resampled by using 
the aforementioned discretization methods.

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
        if !isnothing(Ts) && !(Ts ≈ sys.Ts)
            @info "LinModel: resampling linear model from Ts = $(sys.Ts) to $Ts s..."
            sysu = d2c(sysu, :zoh)
            sysd = d2c(sysd, :tustin)
            sysu_dis = c2d(sysu, Ts, :zoh)
            sysd_dis = c2d(sysd, Ts, :tustin)
        else
            Ts = sys.Ts
            sysu_dis = sysu
            sysd_dis = sysd
        end     
    end
    sys_dis = sminreal([sysu_dis sysd_dis]) # merge common poles if possible
    nx = size(sys_dis.A,1)
    nu = length(i_u)
    ny = size(sys_dis,1)
    nd = length(i_d)
    A   = sys_dis.A
    Bu  = sys_dis.B[:,1:nu]
    Bd  = sys_dis.B[:,nu+1:end]
    C   = sys_dis.C
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

@doc raw"""
    steadystate(model::LinModel, u, d=Float64[])

Evaluate the steady-state vector when `model` is a [`LinModel`](@ref).

Omitting the operating points, the method evaluates the equilibrium ``\mathbf{x}(∞)`` from:
```math
    \mathbf{x}(∞) = \mathbf{(I - A)^{-1}(B_u u + B_d d)}
```
with the manipulated inputs held constant at ``\mathbf{u}`` and, the measured disturbances, 
at ``\mathbf{d}``. The Moore-Penrose pseudo-inverse computes ``\mathbf{(I - A)^{-1}}``
to support integrating `model` (integrator states will be 0).
"""
function steadystate(model::LinModel, u, d=Float64[])
    return pinv(I - model.A)*(model.Bu*(u - model.uop) + model.Bd*(d - model.dop))
end

"""
    f(model::LinModel, x, u, d)

Evaluate ``\\mathbf{A x + B_u u + B_d d}`` when `model` is a [`LinModel`](@ref).
"""
f(model::LinModel, x, u, d) = model.A * x + model.Bu * u + model.Bd * d


"""
    h(model::LinModel, x, u, d)

Evaluate ``\\mathbf{C x + D_d d}`` when `model` is a [`LinModel`](@ref).
"""
h(model::LinModel, x, d) = model.C*x + model.Dd*d

typestr(model::LinModel) = "linear"