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
    NonLinModel{NT}(f!::Function, h!::Function, Ts, nu, nx, ny, nd=0)

Construct a nonlinear model from discrete-time state-space functions `f!` and `h!`.

The state update ``\mathbf{f}`` and output ``\mathbf{h}`` functions are defined as :
```math
    \begin{aligned}
    \mathbf{x}(k+1) &= \mathbf{f}\Big( \mathbf{x}(k), \mathbf{u}(k), \mathbf{d}(k) \Big) \\
    \mathbf{y}(k)   &= \mathbf{h}\Big( \mathbf{x}(k), \mathbf{d}(k) \Big)
    \end{aligned}
```
`Ts` is the sampling time in second. `nu`, `nx`, `ny` and `nd` are the respective number of 
manipulated inputs, states, outputs and measured disturbances. The optional parameter `NT`
explicitly specifies the number type of vectors (default to `Float64`).

!!! tip
    Replace the `d` argument with `_` if `nd = 0` (see Examples below).

Nonlinear continuous-time state-space functions are not supported for now. In such a case, 
manually call a differential equation solver in `f` (see [Manual](@ref man_nonlin)).

!!! warning
    `f!` and `h!` must be pure Julia functions to use the model in [`NonLinMPC`](@ref),
    [`ExtendedKalmanFilter`](@ref), [`MovingHorizonEstimator`](@ref) and [`linearize`](@ref).

See also [`LinModel`](@ref).

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_)->0.1x+u, (x,_)->2x, 10.0, 1, 1, 1)
Discrete-time nonlinear model with a sample time Ts = 10.0 s and:
 1 manipulated inputs u
 1 states x
 1 outputs y
 0 measured disturbances d
```
"""
function NonLinModel{NT}(
    f!::Function, h!::Function, Ts::Real, nu::Int, nx::Int, ny::Int, nd::Int=0
) where {NT<:Real}
    iscontinous = validate_fcts(NT, f!, h!)
    if iscontinous
        fc! = f!
        f! = let nx=nx, Ts=Ts, NT=NT
            ẋ = zeros(NT, nx)
            xterm = zeros(NT, nx)
            k1, k2, k3, k4 = zeros(NT, nx), zeros(NT, nx), zeros(NT, nx), zeros(NT, nx)
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
            rk4!
        end
    end
    F, H = getFuncTypes(f!, h!)
    return NonLinModel{NT, F, H}(f!, h!, Ts, nu, nx, ny, nd)
end

function NonLinModel(
    f!::Function, h!::Function, Ts::Real, nu::Int, nx::Int, ny::Int, nd::Int=0
)
    return NonLinModel{Float64}(f!, h!, Ts, nu, nx, ny, nd)
end

getFuncTypes(f!::F, h!::H) where {F<:Function, H<:Function} = F, H


"""
    validate_fcts(NT, f!, h!) -> iscontinuous

Validate `f!` and `h!` function argument signatures and return `true` if `f!` is continuous.
"""
function validate_fcts(NT, f!, h!)
    isdiscrete = hasmethod(f!,
        Tuple{Vector{NT}, Vector{NT}, Vector{NT}}
    )
    iscontinuous = hasmethod(f!,
        Tuple{Vector{NT}, Vector{NT}, Vector{NT}, Vector{NT}}
    )
    if !isdiscrete && !iscontinuous
        println(isdiscrete)
        println(iscontinuous)
        error("state function has no method with type signature "*
              "f!(x::Vector{$(NT)}, u::Vector{$(NT)}, d::Vector{$(NT)}) or "*
              "f!(ẋ::Vector{$(NT)}, x::Vector{$(NT)}, u::Vector{$(NT)}, d::Vector{$(NT)})")
    end
    hargsvalid = hasmethod(h!, Tuple{Vector{NT}, Vector{NT}, Vector{NT}})
    if !hargsvalid
        error("output function has no method with type signature "*
              "h!(y::Vector{$(NT)}, x::Vector{$(NT)}, d::Vector{$(NT)})")
    end
    return iscontinuous
end

"Do nothing if `model` is a [`NonLinModel`](@ref)."
steadystate!(::SimModel, _ , _ ) = nothing


"Call ``\\mathbf{f!(x, u, d)}`` with `model.f!` function for [`NonLinModel`](@ref)."
f!(x, model::NonLinModel, u, d) = model.f!(x, u, d)

"Call ``\\mathbf{h!(y, x, d)}`` with `model.h` function for [`NonLinModel`](@ref)."
h!(y, model::NonLinModel, x, d) = model.h!(y, x, d)

typestr(model::NonLinModel) = "nonlinear"