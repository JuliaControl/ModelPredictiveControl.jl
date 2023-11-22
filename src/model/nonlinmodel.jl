struct NonLinModel{NT<:Real, F<:Function, H<:Function} <: SimModel{NT}
    x::Vector{NT}
    f::F
    h::H
    Ts::NT
    nu::Int
    nx::Int
    ny::Int
    nd::Int
    uop::Vector{NT}
    yop::Vector{NT}
    dop::Vector{NT}
    function NonLinModel{NT, F, H}(
        f::F, h::H, Ts, nu, nx, ny, nd
    ) where {NT<:Real, F<:Function, H<:Function}
        Ts > 0 || error("Sampling time Ts must be positive")
        validate_fcts(f, h)
        uop = zeros(NT, nu)
        yop = zeros(NT, ny)
        dop = zeros(NT, nd)
        x = zeros(NT, nx)
        return new{NT, F, H}(x, f, h, Ts, nu, nx, ny, nd, uop, yop, dop)
    end
end

@doc raw"""
    NonLinModel{NT}(f::Function, h::Function, Ts, nu, nx, ny, nd=0)

Construct a nonlinear model from discrete-time state-space functions `f` and `h`.

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
    `f` and `h` must be pure Julia functions to use the model in [`NonLinMPC`](@ref),
    [`ExtendedKalmanFilter`](@ref) and [`linearize`](@ref).

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
    f::F, h::H, Ts::Real, nu::Int, nx::Int, ny::Int, nd::Int=0
) where {NT<:Real, F<:Function, H<:Function}
    return NonLinModel{NT, F, H}(f, h, Ts, nu, nx, ny, nd)
end

function NonLinModel(
    f::F, h::H, Ts::Real, nu::Int, nx::Int, ny::Int, nd::Int=0
) where {F<:Function, H<:Function}
    return NonLinModel{Float64, F, H}(f, h, Ts, nu, nx, ny, nd)
end

"Validate `f` and `h` function argument signatures."
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

"Do nothing if `model` is a [`NonLinModel`](@ref)."
steadystate!(::SimModel, _ , _ ) = nothing


"Call ``\\mathbf{f(x, u, d)}`` with `model.f` function for [`NonLinModel`](@ref)."
f(model::NonLinModel, x, u, d) = model.f(x, u, d)

"Call ``\\mathbf{h(x, d)}`` with `model.h` function for [`NonLinModel`](@ref)."
h(model::NonLinModel, x, d) = model.h(x, d)

typestr(model::NonLinModel) = "nonlinear"