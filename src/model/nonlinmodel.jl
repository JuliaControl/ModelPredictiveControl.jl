struct NonLinModel{NT<:Real, F<:Function, H<:Function} <: SimModel{NT}
    x::Vector{NT}
    f!::F
    h!::H
    Ts::NT
    nu::Int
    nx::Int
    ny::Int
    nd::Int
    uop::Vector{NT}
    yop::Vector{NT}
    dop::Vector{NT}
    function NonLinModel{NT, F, H}(
        f!::F, h!::H, Ts, nu, nx, ny, nd
    ) where {NT<:Real, F<:Function, H<:Function}
        Ts > 0 || error("Sampling time Ts must be positive")
        uop = zeros(NT, nu)
        yop = zeros(NT, ny)
        dop = zeros(NT, nd)
        x = zeros(NT, nx)
        return new{NT, F, H}(x, f!, h!, Ts, nu, nx, ny, nd, uop, yop, dop)
    end
end

@doc raw"""
    NonLinModel{NT}(f::Function,  h::Function,  Ts, nu, nx, ny, nd=0)
    NonLinModel{NT}(f!::Function, h!::Function, Ts, nu, nx, ny, nd=0)

Construct a nonlinear model from discrete-time state-space functions `f` and `h`.

The state update ``\mathbf{f}`` and output ``\mathbf{h}`` functions are defined as :
```math
    \begin{aligned}
    \mathbf{x}(k+1) &= \mathbf{f}\Big( \mathbf{x}(k), \mathbf{u}(k), \mathbf{d}(k) \Big) \\
    \mathbf{y}(k)   &= \mathbf{h}\Big( \mathbf{x}(k), \mathbf{d}(k) \Big)
    \end{aligned}
```
Denoting ``\mathbf{x}(k+1)`` as `xnext`, they can be implemented in two possible ways:

- non-mutating functions (out-of-place): they must be defined as `f(x, u, d) -> xnext` and
  `h(x, d) -> y`. This syntax is simple and intuitive but it allocates more memory.
- mutating functions (in-place): they must be defined as `f!(xnext, x, u, d) -> nothing` and
  `h!(y, x, d) -> nothing`. This syntax reduces the allocations and potentially the 
  computational burden as well.

`Ts` is the sampling time in second. `nu`, `nx`, `ny` and `nd` are the respective number of 
manipulated inputs, states, outputs and measured disturbances. The optional parameter `NT`
explicitly specifies the number type of vectors (default to `Float64`).

!!! tip
    Replace the `d` argument with `_` if `nd = 0` (see Examples below).

Nonlinear continuous-time state-space functions are not supported for now. In such a case, 
manually call a differential equation solver in `f` / `f!` (see [Manual](@ref man_nonlin)).

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

julia> f!(xnext,x,u,_) = (xnext .= 0.1x .+ u; nothing);

julia> h!(y,x,_) = (y .= 2x; nothing);

julia> model2 = NonLinModel(f!, h!, 10.0, 1, 1, 1) 
Discrete-time nonlinear model with a sample time Ts = 10.0 s and:
 1 manipulated inputs u
 1 states x
 1 outputs y
 0 measured disturbances d
```
"""
function NonLinModel{NT}(
    f::Function, h::Function, Ts::Real, nu::Int, nx::Int, ny::Int, nd::Int=0
) where {NT<:Real}
    ismutating_f = validate_f(NT, f)
    ismutating_h = validate_h(NT, h)
    f! = let f = f
        ismutating_f ? f : (xnext, x, u, d) -> xnext .= f(x, u, d)
    end
    h! = let h = h
        ismutating_h ? h : (y, x, d) -> y .= h(x, d)
    end
    F, H = getFuncTypes(f!, h!)
    return NonLinModel{NT, F, H}(f!, h!, Ts, nu, nx, ny, nd)
end

function NonLinModel(
    f::Function, h::Function, Ts::Real, nu::Int, nx::Int, ny::Int, nd::Int=0
)
    return NonLinModel{Float64}(f, h, Ts, nu, nx, ny, nd)
end

getFuncTypes(f::F, h::H) where {F<:Function, H<:Function} = F, H


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

function rk4!(x, u, d)
    xterm .= x
    fc!(ẋ, xterm, u, d)
    k1 .= ẋ
    xterm .= @. x + k1 * Ts/2
    fc!(ẋ, xterm, u, d)
    k2 .= ẋ 
    xterm .= @. x + k2 * Ts/2
    fc!(ẋ, xterm, u, d)
    k3 .= ẋ
    xterm .= @. x + k3 * Ts
    fc!(ẋ, xterm, u, d)
    k4 .= ẋ
    x .+= @. (k1 + 2k2 + 2k3 + k4)*Ts/6
    return x
end