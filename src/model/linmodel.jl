struct LinModel{NT<:Real} <: SimModel{NT}
    A   ::Matrix{NT}
    Bu  ::Matrix{NT}
    C   ::Matrix{NT}
    Bd  ::Matrix{NT}
    Dd  ::Matrix{NT}
    x::Vector{NT}
    Ts::NT
    nu::Int
    nx::Int
    ny::Int
    nd::Int
    uop::Vector{NT}
    yop::Vector{NT}
    dop::Vector{NT}
    function LinModel{NT}(A, Bu, C, Bd, Dd, Ts) where {NT<:Real}
        A, Bu, C, Bd, Dd = to_mat(A), to_mat(Bu), to_mat(C), to_mat(Bd), to_mat(Dd)
        nu, nx, ny, nd = size(Bu,2), size(A,2), size(C,1), size(Bd,2)
        size(A)  == (nx,nx) || error("A size must be $((nx,nx))")
        size(Bu) == (nx,nu) || error("Bu size must be $((nx,nu))")
        size(C)  == (ny,nx) || error("C size must be $((ny,nx))")
        size(Bd) == (nx,nd) || error("Bd size must be $((nx,nd))")
        size(Dd) == (ny,nd) || error("Dd size must be $((ny,nd))")
        Ts > 0 || error("Sampling time Ts must be positive")
        uop = zeros(NT, nu)
        yop = zeros(NT, ny)
        dop = zeros(NT, nd)
        x = zeros(NT, nx)
        return new(A, Bu, C, Bd, Dd, x, Ts, nu, nx, ny, nd, uop, yop, dop)
    end
end

@doc raw"""
    LinModel(sys::StateSpace[, Ts]; i_u=1:size(sys,2), i_d=Int[])

Construct a linear model from state-space model `sys` with sampling time `Ts` in second.

The system `sys` can be continuous or discrete-time (`Ts` can be omitted for the latter).
For continuous dynamics, its state-space equations are (discrete case in Extended Help):
```math
\begin{aligned}
    \mathbf{ẋ}(t) &= \mathbf{A x}(t) + \mathbf{B z}(t) \\
    \mathbf{y}(t) &= \mathbf{C x}(t) + \mathbf{D z}(t)
\end{aligned}
```
with the state ``\mathbf{x}`` and output ``\mathbf{y}`` vectors. The ``\mathbf{z}`` vector 
comprises the manipulated inputs ``\mathbf{u}`` and measured disturbances ``\mathbf{d}``, 
in any order. `i_u` provides the indices of ``\mathbf{z}`` that are manipulated, and `i_d`, 
the measured disturbances. The constructor automatically discretizes continuous systems,
resamples discrete ones if `Ts ≠ sys.Ts`, computes a new realization if not minimal, and 
separates the ``\mathbf{z}`` terms in two parts (details in Extended Help). The rest of the
documentation assumes discrete dynamics since all systems end up in this form.

See also [`ss`](https://juliacontrol.github.io/ControlSystems.jl/stable/lib/constructors/#ControlSystemsBase.ss)

# Examples
```jldoctest
julia> model1 = LinModel(ss(-0.1, 1.0, 1.0, 0), 2.0) # continuous-time StateSpace
LinModel with a sample time Ts = 2.0 s and:
 1 manipulated inputs u
 1 states x
 1 outputs y
 0 measured disturbances d

julia> model2 = LinModel(ss(0.4, 0.2, 0.3, 0, 0.1)) # discrete-time StateSpace
LinModel with a sample time Ts = 0.1 s and:
 1 manipulated inputs u
 1 states x
 1 outputs y
 0 measured disturbances d
```

# Extended Help
!!! details "Extended Help"
    State-space equations are similar if `sys` is discrete-time (replace ``\mathbf{ẋ}(t)``
    with ``\mathbf{x}(k+1)`` and ``k`` with ``t`` on the LHS). Continuous dynamics are 
    internally discretized using [`c2d`](https://juliacontrol.github.io/ControlSystems.jl/stable/lib/constructors/#ControlSystemsBase.c2d)
    and `:zoh` for manipulated inputs, and `:tustin`, for measured disturbances. Lastly, if 
    `sys` is discrete and the provided argument `Ts ≠ sys.Ts`, the system is resampled by
    using the aforementioned discretization methods.

    Note that the constructor transforms the system to its minimal realization using [`minreal`](https://juliacontrol.github.io/ControlSystems.jl/stable/lib/constructors/#ControlSystemsBase.minreal)
    for controllability/observability. As a consequence, the final state-space
    representation may be different from the one provided in `sys`. It is also converted 
    into a more practical form (``\mathbf{D_u=0}`` because of the zero-order hold):
    ```math
    \begin{aligned}
        \mathbf{x}(k+1) &=  \mathbf{A x}(k) + \mathbf{B_u u}(k) + \mathbf{B_d d}(k) \\
        \mathbf{y}(k)   &=  \mathbf{C x}(k) + \mathbf{D_d d}(k)
    \end{aligned}
    ```
    Use the syntax [`LinModel{NT}(A, Bu, C, Bd, Dd, Ts)`](@ref) to force a specific
    state-space representation.
"""
function LinModel(
    sys::StateSpace{E, NT},
    Ts::Union{Real,Nothing} = nothing;
    i_u::IntRangeOrVector = 1:size(sys,2),
    i_d::IntRangeOrVector = Int[]
) where {E, NT<:Real}
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
        sysu_dis = c2d(sysu,Ts,:zoh)
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
    # minreal to merge common poles if possible and ensure controllability/observability:
    sys_dis = minreal([sysu_dis sysd_dis]) # same realization if already minimal
    nu  = length(i_u)
    A   = sys_dis.A
    Bu  = sys_dis.B[:,1:nu]
    Bd  = sys_dis.B[:,nu+1:end]
    C   = sys_dis.C
    Dd  = sys_dis.D[:,nu+1:end]
    return LinModel{NT}(A, Bu, C, Bd, Dd, Ts)
end


@doc raw"""
    LinModel(sys::TransferFunction[, Ts]; i_u=1:size(sys,2), i_d=Int[])

Convert to minimal realization state-space when `sys` is a transfer function.

`sys` is equal to ``\frac{\mathbf{y}(s)}{\mathbf{z}(s)}`` for continuous-time, and 
``\frac{\mathbf{y}(z)}{\mathbf{z}(z)}``, for discrete-time.

See also [`tf`](https://juliacontrol.github.io/ControlSystems.jl/stable/lib/constructors/#ControlSystemsBase.tf)

# Examples
```jldoctest
julia> model = LinModel([tf(3, [30, 1]) tf(-2, [5, 1])], 0.5, i_d=[2])
LinModel with a sample time Ts = 0.5 s and:
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
"""
function LinModel(sys::DelayLtiSystem, Ts::Real; kwargs...)
    sys_dis = minreal(c2d(sys, Ts, :zoh)) # c2d only supports :zoh for DelayLtiSystem
    return LinModel(sys_dis, Ts; kwargs...)
end

@doc raw"""
    LinModel{NT}(A, Bu, C, Bd, Dd, Ts)

Construct the model from the discrete state-space matrices `A, Bu, C, Bd, Dd` directly.

See [`LinModel(::StateSpace)`](@ref) Extended Help for the meaning of the matrices. This
syntax do not modify the state-space representation provided in argument (`minreal` is not
called). Care must be taken to ensure that the model is controllable and observable. The
optional parameter `NT` explicitly specifies the number type of the matrices.
"""
LinModel{NT}(A, Bu, C, Bd, Dd, Ts) where NT<:Real

function LinModel(
    A::Array{NT}, Bu::Array{NT}, C::Array{NT}, Bd::Array{NT}, Dd::Array{NT}, Ts::Real
) where {NT<:Real} 
    return LinModel{NT}(A, Bu, C, Bd, Dd, Ts)
end

function LinModel(
    A::Array{<:Real}, 
    Bu::Array{<:Real}, 
    C::Array{<:Real}, 
    Bd::Array{<:Real}, 
    Dd::Array{<:Real},
    Ts::Real
)
    A, Bu, C, Bd, Dd = to_mat(A), to_mat(Bu), to_mat(C), to_mat(Bd), to_mat(Dd)
    A, Bu, C, Bd, Dd = promote(A, Bu, C, Bd, Dd)
    return LinModel(A, Bu, C, Bd, Dd, Ts)
end

@doc raw"""
    steadystate!(model::LinModel, u, d)

Set `model.x` to `u` and `d` steady-state if `model` is a [`LinModel`](@ref).

Following [`setop!`](@ref) notation, the method evaluates the equilibrium ``\mathbf{x}``
from:
```math
    \mathbf{x} = \mathbf{(I - A)^{-1}(B_u u_0 + B_d d_0)}
```
with constant manipulated inputs ``\mathbf{u_0 = u - u_{op}}`` and measured
disturbances ``\mathbf{d_0 = d - d_{op}}``. The Moore-Penrose pseudo-inverse computes 
``\mathbf{(I - A)^{-1}}`` to support integrating `model` (integrator states will be 0).
"""
function steadystate!(model::LinModel, u, d)
    M = I - model.A
    rtol = sqrt(eps(real(float(oneunit(eltype(M)))))) # pinv docstring recommendation
    model.x .= pinv(M; rtol)*(model.Bu*(u - model.uop) + model.Bd*(d - model.dop))
    return nothing
end

"""
    f!(xnext, model::LinModel, x, u, d) -> nothing

Evaluate `xnext = A*x + Bu*u + Bd*d` in-place when `model` is a [`LinModel`](@ref).
"""
function f!(xnext, model::LinModel, x, u, d)
    mul!(xnext, model.A,  x)
    mul!(xnext, model.Bu, u, 1, 1)
    mul!(xnext, model.Bd, d, 1, 1)
    return nothing
end


"""
    h!(y, model::LinModel, x, d) -> nothing

Evaluate `y = C*x + Dd*d` in-place when `model` is a [`LinModel`](@ref).
"""
function h!(y, model::LinModel, x, d)
    mul!(y, model.C,  x)
    mul!(y, model.Dd, d, 1, 1)
    return nothing
end