struct NonLinModel{NT<:Real, F<:Function, H<:Function, DS<:DiffSolver} <: SimModel{NT}
    x::Vector{NT}
    f!::F
    h!::H
    solver::DS
    Ts::NT
    nu::Int
    nx::Int
    ny::Int
    nd::Int
    uop::Vector{NT}
    yop::Vector{NT}
    dop::Vector{NT}
    function NonLinModel{NT, F, H, DS}(
        f!::F, h!::H, solver::DS, Ts, nu, nx, ny, nd
    ) where {NT<:Real, F<:Function, H<:Function, DS<:DiffSolver}
        Ts > 0 || error("Sampling time Ts must be positive")
        uop = zeros(NT, nu)
        yop = zeros(NT, ny)
        dop = zeros(NT, nd)
        x = zeros(NT, nx)
        return new{NT, F, H, DS}(x, f!, h!, solver, Ts, nu, nx, ny, nd, uop, yop, dop)
    end
end

@doc raw"""
    NonLinModel{NT}(f::Function,  h::Function,  Ts, nu, nx, ny, nd=0; solver=nothing)
    NonLinModel{NT}(f!::Function, h!::Function, Ts, nu, nx, ny, nd=0; solver=nothing)

Construct a nonlinear model from state-space functions `f`/`f!` and `h`/`h!`.

Default arguments assume discrete-time dynamics, in which the state update ``\mathbf{f}``
and output ``\mathbf{h}`` functions are defined as :
```math
    \begin{aligned}
    \mathbf{x}(k+1) &= \mathbf{f}\Big( \mathbf{x}(k), \mathbf{u}(k), \mathbf{d}(k) \Big) \\
    \mathbf{y}(k)   &= \mathbf{h}\Big( \mathbf{x}(k), \mathbf{d}(k) \Big)
    \end{aligned}
```
Denoting ``\mathbf{x}(k+1)`` as `xnext`, they can be implemented in two possible ways:

- Non-mutating functions (out-of-place): they must be defined as `f(x, u, d) -> xnext` and
  `h(x, d) -> y`. This syntax is simple and intuitive but it allocates more memory.
- Mutating functions (in-place): they must be defined as `f!(xnext, x, u, d) -> nothing` and
  `h!(y, x, d) -> nothing`. This syntax reduces the allocations and potentially the 
  computational burden as well.

`Ts` is the sampling time in second. `nu`, `nx`, `ny` and `nd` are the respective number of 
manipulated inputs, states, outputs and measured disturbances. 

!!! tip
    Replace the `d` argument with `_` if `nd = 0` (see Examples below).

Specifying a differential equation solver with the the `solver` keyword argument implies a
continuous-time model (see Extended Help for details). The optional parameter `NT`
explicitly set the number type of vectors (default to `Float64`).

!!! warning
    The two functions must be in pure Julia to use the model in [`NonLinMPC`](@ref),
    [`ExtendedKalmanFilter`](@ref), [`MovingHorizonEstimator`](@ref) and [`linearize`](@ref).

See also [`LinModel`](@ref).

# Examples
```jldoctest
julia> model1 = NonLinModel((x,u,_)->0.1x+u, (x,_)->2x, 10.0, 1, 1, 1)
Discrete-time nonlinear model with a sample time Ts = 10.0 s and:
 1 manipulated inputs u
 1 states x
 1 outputs y
 0 measured disturbances d

julia> f!(ẋ, x, u, _ ) = (ẋ .= -0.1x .+ u; nothing);

julia> h!(y, x, _ ) = (y .= 2x; nothing);

julia> model2 = NonLinModel(f!, h!, 5.0, 1, 1, 1, solver=RungeKutta())
Discrete-time nonlinear model with a sample time Ts = 5.0 s and:
 1 manipulated inputs u
 1 states x
 1 outputs y
 0 measured disturbances d
```

# Extended Help
!!! details "Extended Help"
    State-space equations are similar for continuous-time models (replace ``\mathbf{x}(k+1)``
    with ``\mathbf{ẋ}(t)`` and ``k`` with ``t`` on the LHS), with also two possible 
    implementations (second one to reduce the allocations):

    - Non-mutating functions (out-of-place): they must be defined as `f(x, u, d) -> ẋ` and
      `h(x, d) -> y`.
    - Mutating functions (in-place): they must be defined as `f!(ẋ, x, u, d) -> nothing` and
      `h!(y, x, d) -> nothing`. 
"""
function NonLinModel{NT}(
    f::Function, h::Function, Ts::Real, nu::Int, nx::Int, ny::Int, nd::Int=0; solver=nothing
) where {NT<:Real}
    isnothing(solver) && (solver=EmptySolver())
    ismutating_f = validate_f(NT, f)
    ismutating_h = validate_h(NT, h)
    f! = let f = f
        ismutating_f ? f : (xnext, x, u, d) -> xnext .= f(x, u, d)
    end
    h! = let h = h
        ismutating_h ? h : (y, x, d) -> y .= h(x, d)
    end
    f!, h! = get_solver_functions(NT, solver, f!, h!, Ts, nu, nx, ny, nd)
    F, H, DS = get_types(f!, h!, solver)
    return NonLinModel{NT, F, H, DS}(f!, h!, solver, Ts, nu, nx, ny, nd)
end

function NonLinModel(
    f::Function, h::Function, Ts::Real, nu::Int, nx::Int, ny::Int, nd::Int=0; solver=nothing
)
    return NonLinModel{Float64}(f, h, Ts, nu, nx, ny, nd; solver)
end

get_types(f!::F, h!::H, solver::DS) where {F<:Function, H<:Function, DS<:DiffSolver} = F, H, DS

"""
    validate_f(NT, f) -> ismutating

Validate `f` function argument signature and return `true` if it is mutating.
"""
function validate_f(NT, f)
    ismutating = hasmethod(f, Tuple{Vector{NT}, Vector{NT}, Vector{NT}, Vector{NT}})
    if !(ismutating || hasmethod(f, Tuple{Vector{NT}, Vector{NT}, Vector{NT}}))
        error(
            "the state function has no method with type signature "*
            "f(x::Vector{$(NT)}, u::Vector{$(NT)}, d::Vector{$(NT)}) or mutating form "*
            "f!(xnext::Vector{$(NT)}, x::Vector{$(NT)}, u::Vector{$(NT)}, d::Vector{$(NT)})"
        )
    end
    return ismutating
end

"""
    validate_h(NT, h) -> ismutating

Validate `h` function argument signature and return `true` if it is mutating.
"""
function validate_h(NT, h)
    ismutating = hasmethod(h, Tuple{Vector{NT}, Vector{NT}, Vector{NT}})
    if !(ismutating || hasmethod(h, Tuple{Vector{NT}, Vector{NT}}))
        error(
            "the output function has no method with type signature "*
            "h(x::Vector{$(NT)}, d::Vector{$(NT)}) or mutating form "*
            "h!(y::Vector{$(NT)}, x::Vector{$(NT)}, d::Vector{$(NT)})"
        )
    end
    return ismutating
end

"Do nothing if `model` is a [`NonLinModel`](@ref)."
steadystate!(::SimModel, _ , _ ) = nothing


"Call `f!(xnext, x, u, d)` with `model.f!` method for [`NonLinModel`](@ref)."
f!(xnext, model::NonLinModel, x, u, d) = model.f!(xnext, x, u, d)

"Call `h!(y, x, d)` with `model.h` method for [`NonLinModel`](@ref)."
h!(y, model::NonLinModel, x, d) = model.h!(y, x, d)

typestr(model::NonLinModel) = "nonlinear"