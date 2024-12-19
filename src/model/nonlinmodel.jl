struct NonLinModel{
    NT<:Real, F<:Function, H<:Function, P<:Any, DS<:DiffSolver, SMB<:SimModelBuffer
} <: SimModel{NT}
    x0::Vector{NT}
    f!::F
    h!::H
    p::P
    solver::DS
    Ts::NT
    t::Vector{NT}
    nu::Int
    nx::Int
    ny::Int
    nd::Int
    uop::Vector{NT}
    yop::Vector{NT}
    dop::Vector{NT}
    xop::Vector{NT}
    fop::Vector{NT}
    uname::Vector{String}
    yname::Vector{String}
    dname::Vector{String}
    xname::Vector{String}
    buffer::SMB
    function NonLinModel{NT, F, H, P, DS}(
        f!::F, h!::H, Ts, nu, nx, ny, nd, p::P, solver::DS, buffer::SMB
    ) where {NT<:Real, F<:Function, H<:Function, P<:Any, DS<:DiffSolver, SMB<:SimModelBuffer}
        Ts > 0 || error("Sampling time Ts must be positive")
        uop = zeros(NT, nu)
        yop = zeros(NT, ny)
        dop = zeros(NT, nd)
        xop = zeros(NT, nx)
        fop = zeros(NT, nx)
        uname = ["\$u_{$i}\$" for i in 1:nu]
        yname = ["\$y_{$i}\$" for i in 1:ny]
        dname = ["\$d_{$i}\$" for i in 1:nd]
        xname = ["\$x_{$i}\$" for i in 1:nx]
        x0 = zeros(NT, nx)
        t  = zeros(NT, 1)
        return new{NT, F, H, P, DS, SMB}(
            x0, 
            f!, h!,
            p,
            solver, 
            Ts, t,
            nu, nx, ny, nd, 
            uop, yop, dop, xop, fop,
            uname, yname, dname, xname,
            buffer
        )
    end
end

@doc raw"""
    NonLinModel{NT}(f::Function,  h::Function,  Ts, nu, nx, ny, nd=0; p=[], solver=RungeKutta(4))
    NonLinModel{NT}(f!::Function, h!::Function, Ts, nu, nx, ny, nd=0; p=[], solver=RungeKutta(4))

Construct a nonlinear model from state-space functions `f`/`f!` and `h`/`h!`.

Both continuous and discrete-time models are supported. The default arguments assume 
continuous dynamics. Use `solver=nothing` for the discrete case (see Extended Help). The
functions are defined as:
```math
\begin{aligned}
    \mathbf{ẋ}(t) &= \mathbf{f}\Big( \mathbf{x}(t), \mathbf{u}(t), \mathbf{d}(t), \mathbf{p} \Big) \\
    \mathbf{y}(t) &= \mathbf{h}\Big( \mathbf{x}(t), \mathbf{d}(t), \mathbf{p} \Big)
\end{aligned}
```
where ``\mathbf{x}``, ``\mathbf{y}``, ``\mathbf{u}``, ``\mathbf{d}`` and ``\mathbf{p}`` are
respectively the state, output, manipulated input, measured disturbance and parameter
vectors. If the dynamics is a function of the time, simply add a measured disturbance
defined as ``d(t) = t``. The functions can be implemented in two possible ways:

1. **Non-mutating functions** (out-of-place): define them as `f(x, u, d, p) -> ẋ` and
   `h(x, d, p) -> y`. This syntax is simple and intuitive but it allocates more memory.
2. **Mutating functions** (in-place): define them as `f!(ẋ, x, u, d, p) -> nothing` and
   `h!(y, x, d, p) -> nothing`. This syntax reduces the allocations and potentially the 
   computational burden as well.

`Ts` is the sampling time in second. `nu`, `nx`, `ny` and `nd` are the respective number of 
manipulated inputs, states, outputs and measured disturbances. The keyword argument `p`
is the parameters of the model passed to the two functions. It can be any Julia objects but
use a mutable type if you want to change them later e.g.: a vector.

!!! tip
    Replace the `d` or `p` argument with `_` in your functions if not needed (see Examples below).
    
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
julia> f!(ẋ, x, u, _ , p) = (ẋ .= p*x .+ u; nothing);

julia> h!(y, x, _ , _ ) = (y .= 0.1x; nothing);

julia> model1 = NonLinModel(f!, h!, 5.0, 1, 1, 1, p=-0.2)       # continuous dynamics
NonLinModel with a sample time Ts = 5.0 s, RungeKutta solver and:
 1 manipulated inputs u
 1 states x
 1 outputs y
 0 measured disturbances d

julia> f(x, u, _ , _ ) = 0.1x + u;

julia> h(x, _ , _ ) = 2x;

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
        \mathbf{x}(k+1) &= \mathbf{f}\Big( \mathbf{x}(k), \mathbf{u}(k), \mathbf{d}(k), \mathbf{p} \Big) \\
        \mathbf{y}(k)   &= \mathbf{h}\Big( \mathbf{x}(k), \mathbf{d}(k), \mathbf{p} \Big)
    \end{aligned}
    ```
    with two possible implementations as well:

    1. **Non-mutating functions**: define them as `f(x, u, d, p) -> xnext` and 
       `h(x, d, p) -> y`.
    2. **Mutating functions**: define them as `f!(xnext, x, u, d, p) -> nothing` and
       `h!(y, x, d, p) -> nothing`.
"""
function NonLinModel{NT}(
    f::Function, h::Function, Ts::Real, nu::Int, nx::Int, ny::Int, nd::Int=0;
    p=NT[], solver=RungeKutta(4)
) where {NT<:Real}
    isnothing(solver) && (solver=EmptySolver())
    f!, h! = get_mutating_functions(NT, f, h)
    f!, h! = get_solver_functions(NT, solver, f!, h!, Ts, nu, nx, ny, nd)
    F, H, P, DS = get_types(f!, h!, p, solver)
    jacobian = JacobianBuffer{NT}(f!, h!, nu, nx, ny, nd)
    buffer = SimModelBuffer{NT}(nu, nx, ny, nd, jacobian)
    return NonLinModel{NT, F, H, P, DS}(f!, h!, Ts, nu, nx, ny, nd, p, solver, buffer)
end

function NonLinModel(
    f::Function, h::Function, Ts::Real, nu::Int, nx::Int, ny::Int, nd::Int=0;
    p=Float64[], solver=RungeKutta(4)
)
    return NonLinModel{Float64}(f, h, Ts, nu, nx, ny, nd; p, solver)
end

"Get the mutating functions `f!` and `h!` from the provided functions in argument."
function get_mutating_functions(NT, f, h)
    ismutating_f = validate_f(NT, f)
    f! = if ismutating_f
        f
    else
        function f!(xnext, x, u, d, p)
            xnext .= f(x, u, d, p)
            return nothing
        end
    end
    ismutating_h = validate_h(NT, h)
    h! = if ismutating_h
        h
    else
        function h!(y, x, d, p)
            y .= h(x, d, p)
            return nothing
        end
    end
    return f!, h!
end

"""
    validate_f(NT, f) -> ismutating

Validate `f` function argument signature and return `true` if it is mutating.
"""
function validate_f(NT, f)
    ismutating = hasmethod(
        f, 
        #       ẋ or xnext, x,          u,          d,          p    
        Tuple{  Vector{NT}, Vector{NT}, Vector{NT}, Vector{NT}, Any}
    )
    #                                     x,          u,          d,          p
    if !(ismutating || hasmethod(f, Tuple{Vector{NT}, Vector{NT}, Vector{NT}, Any}))
        error(
            "the state function has no method with type signature "*
            "f(x::Vector{$(NT)}, u::Vector{$(NT)}, d::Vector{$(NT)}, p::Any) or "*
            "mutating form f!(xnext::Vector{$(NT)}, x::Vector{$(NT)}, u::Vector{$(NT)}, "*
                                                   "d::Vector{$(NT)}, p::Any)"
        )
    end
    return ismutating
end

"""
    validate_h(NT, h) -> ismutating

Validate `h` function argument signature and return `true` if it is mutating.
"""
function validate_h(NT, h)
    ismutating = hasmethod(
        h, 
        #     y,          x,          d,          p
        Tuple{Vector{NT}, Vector{NT}, Vector{NT}, Any}
    )
    #                                     x,          d,          p
    if !(ismutating || hasmethod(h, Tuple{Vector{NT}, Vector{NT}, Any}))
        error(
            "the output function has no method with type signature "*
            "h(x::Vector{$(NT)}, d::Vector{$(NT)}, p::Any) or mutating form "*
            "h!(y::Vector{$(NT)}, x::Vector{$(NT)}, d::Vector{$(NT)}, p::Any)"
        )
    end
    return ismutating
end

"Get the types of `f!`, `h!` and `solver` to construct a `NonLinModel`."
function get_types(
    ::F, ::H, ::P, ::DS
) where {F<:Function, H<:Function, P<:Any, DS<:DiffSolver} 
    return F, H, P, DS
end

"Do nothing if `model` is a [`NonLinModel`](@ref)."
steadystate!(::SimModel, _ , _ ) = nothing

"Call `model.f!(xnext0, x0, u0, d0, p)` for [`NonLinModel`](@ref)."
f!(xnext0, model::NonLinModel, x0, u0, d0, p) = model.f!(xnext0, x0, u0, d0, p)

"Call `model.h!(y0, x0, d0, p)` for [`NonLinModel`](@ref)."
h!(y0, model::NonLinModel, x0, d0, p) = model.h!(y0, x0, d0, p)

detailstr(model::NonLinModel) = ", $(typeof(model.solver).name.name) solver"
detailstr(::NonLinModel{<:Real, <:Function, <:Function, <:Any, <:EmptySolver}) = ", empty solver"