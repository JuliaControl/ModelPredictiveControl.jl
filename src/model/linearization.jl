
"""
    LinModel(model::NonLinModel; x=model.x, u=model.uop, d=model.dop)

Call [`linearize(model; x, u, d)`](@ref) and return the resulting linear model.
"""
LinModel(model::NonLinModel; kwargs...) = linearize(model; kwargs...)

@doc raw"""
    linearize(model::NonLinModel; x=model.x, u=model.uop, d=model.dop) -> linmodel

Linearize `model` at the operating points `x`, `u`, `d` and return the [`LinModel`](@ref).

The arguments `x`, `u` and `d` are the linearization points for the state ``\mathbf{x}``,
manipulated input ``\mathbf{u}`` and measured disturbance ``\mathbf{d}``, respectively. The
Jacobians of ``\mathbf{f}`` and ``\mathbf{h}`` functions are automatically computed with
[`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl).

!!! warning
    See Extended Help if you get an error like:    
    `MethodError: no method matching (::var"##")(::Vector{ForwardDiff.Dual})`.

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_)->x.^3 + u, (x,_)->x, 0.1, 1, 1, 1, solver=nothing);

julia> linmodel = linearize(model, x=[10.0], u=[0.0]); 

julia> linmodel.A
1×1 Matrix{Float64}:
 300.0
```

# Extended Help
!!! details "Extended Help"
    Automatic differentiation (AD) allows exact Jacobians. The [`NonLinModel`](@ref) `f` and
    `h` functions must be compatible with this feature though. See [Automatic differentiation](https://jump.dev/JuMP.jl/stable/manual/nlp/#Automatic-differentiation)
    for common mistakes when writing these functions.
"""
function linearize(model::NonLinModel; x=model.x, u=model.uop, d=model.dop)
    u0, d0 = u - model.uop, d - model.dop
    xnext, y = similar(x), similar(model.yop)
    model.h!(y, x, d0)
    y .+= model.yop
    A  = ForwardDiff.jacobian((xnext, x)  -> model.f!(xnext, x, u0, d0), xnext, x)
    Bu = ForwardDiff.jacobian((xnext, u0) -> model.f!(xnext, x, u0, d0), xnext, u0)
    Bd = ForwardDiff.jacobian((xnext, d0) -> model.f!(xnext, x, u0, d0), xnext, d0)
    C  = ForwardDiff.jacobian((y, x)  -> model.h!(y, x, d0), y, x)
    Dd = ForwardDiff.jacobian((y, d0) -> model.h!(y, x, d0), y, d0)
    linmodel = LinModel(A, Bu, C, Bd, Dd, model.Ts)
    setop!(linmodel, uop=u, yop=y, dop=d)
    return linmodel
end