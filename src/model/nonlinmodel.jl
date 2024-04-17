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
    xop::Vector{NT}
    function NonLinModel{NT, F, H, DS}(
        f!::F, h!::H, solver::DS, Ts, nu, nx, ny, nd
    ) where {NT<:Real, F<:Function, H<:Function, DS<:DiffSolver}
        Ts > 0 || error("Sampling time Ts must be positive")
        uop = zeros(NT, nu)
        yop = zeros(NT, ny)
        dop = zeros(NT, nd)
        xop = zeros(NT, nx)
        x = zeros(NT, nx)
        return new{NT, F, H, DS}(x, f!, h!, solver, Ts, nu, nx, ny, nd, uop, yop, dop, xop)
    end
end

@doc raw"""
    NonLinModel{NT}(f::Function,  h::Function,  Ts, nu, nx, ny, nd=0; solver=RungeKutta(4))
    NonLinModel{NT}(f!::Function, h!::Function, Ts, nu, nx, ny, nd=0; solver=RungeKutta(4))

Construct a nonlinear model from state-space functions `f`/`f!` and `h`/`h!`.

Both continuous and discrete-time models are supported. The default arguments assume 
continuous dynamics. Use `solver=nothing` for the discrete case (see Extended Help). The
functions are defined as:
```math
\begin{aligned}
    \mathbf{ẋ}(t) &= \mathbf{f}\Big( \mathbf{x}(t), \mathbf{u}(t), \mathbf{d}(t) \Big) \\
    \mathbf{y}(t) &= \mathbf{h}\Big( \mathbf{x}(t), \mathbf{d}(t) \Big)
\end{aligned}
```
They can be implemented in two possible ways:

1. **Non-mutating functions** (out-of-place): define them as `f(x, u, d) -> ẋ` and
   `h(x, d) -> y`. This syntax is simple and intuitive but it allocates more memory.
2. **Mutating functions** (in-place): define them as `f!(ẋ, x, u, d) -> nothing` and
   `h!(y, x, d) -> nothing`. This syntax reduces the allocations and potentially the 
   computational burden as well.

`Ts` is the sampling time in second. `nu`, `nx`, `ny` and `nd` are the respective number of 
manipulated inputs, states, outputs and measured disturbances. 

!!! tip
    Replace the `d` argument with `_` if `nd = 0` (see Examples below).
    
A 4th order [`RungeKutta`](@ref) solver discretizes the differential equations by default. 
The rest of the documentation assumes discrete dynamics since all models end up in this 
form. The optional parameter `NT` explicitly set the number type of vectors (default to 
`Float64`).

!!! warning
    The two functions must be in pure Julia to use the model in [`NonLinMPC`](@ref),
    [`ExtendedKalmanFilter`](@ref), [`MovingHorizonEstimator`](@ref) and [`linearize`](@ref).

See also [`LinModel`](@ref).

# Examples
```jldoctest
julia> f!(ẋ, x, u, _ ) = (ẋ .= -0.2x .+ u; nothing);

julia> h!(y, x, _ ) = (y .= 0.1x; nothing);

julia> model1 = NonLinModel(f!, h!, 5.0, 1, 1, 1)               # continuous dynamics
NonLinModel with a sample time Ts = 5.0 s, RungeKutta solver and:
 1 manipulated inputs u
 1 states x
 1 outputs y
 0 measured disturbances d

julia> f(x, u, _ ) = 0.1x + u;

julia> h(x, _ ) = 2x;

julia> model2 = NonLinModel(f, h, 2.0, 1, 1, 1, solver=nothing) # discrete dynamics
NonLinModel with a sample time Ts = 2.0 s, empty solver and:
 1 manipulated inputs u
 1 states x
 1 outputs y
 0 measured disturbances d
```

# Extended Help
!!! details "Extended Help"
    State-space functions are similar for discrete dynamics:
    ```math
    \begin{aligned}
        \mathbf{x}(k+1) &= \mathbf{f}\Big( \mathbf{x}(k), \mathbf{u}(k), \mathbf{d}(k) \Big) \\
        \mathbf{y}(k)   &= \mathbf{h}\Big( \mathbf{x}(k), \mathbf{d}(k) \Big)
    \end{aligned}
    ```
    with two possible implementations as well:

    1. **Non-mutating functions**: define them as `f(x, u, d) -> xnext` and `h(x, d) -> y`.
    2. **Mutating functions**: define them as `f!(xnext, x, u, d) -> nothing` and
       `h!(y, x, d) -> nothing`.
"""
function NonLinModel{NT}(
    f::Function, h::Function, Ts::Real, nu::Int, nx::Int, ny::Int, nd::Int=0; 
    solver=RungeKutta(4)
) where {NT<:Real}
    isnothing(solver) && (solver=EmptySolver())
    ismutating_f = validate_f(NT, f)
    ismutating_h = validate_h(NT, h)
    f! = ismutating_f ? f : (xnext, x, u, d) -> xnext .= f(x, u, d)
    h! = ismutating_h ? h : (y, x, d) -> y .= h(x, d)
    f!, h! = get_solver_functions(NT, solver, f!, h!, Ts, nu, nx, ny, nd)
    F, H, DS = get_types(f!, h!, solver)
    return NonLinModel{NT, F, H, DS}(f!, h!, solver, Ts, nu, nx, ny, nd)
end

function NonLinModel(
    f::Function, h::Function, Ts::Real, nu::Int, nx::Int, ny::Int, nd::Int=0; 
    solver=RungeKutta(4)
)
    return NonLinModel{Float64}(f, h, Ts, nu, nx, ny, nd; solver)
end

"Get the types of `f!`, `h!` and `solver` to construct a `NonLinModel`."
get_types(::F, ::H, ::DS) where {F<:Function, H<:Function, DS<:DiffSolver} = F, H, DS

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

detailstr(model::NonLinModel) = ", $(typeof(model.solver).name.name) solver"
detailstr(::NonLinModel{<:Real, <:Function, <:Function, <:EmptySolver}) = ", empty solver"