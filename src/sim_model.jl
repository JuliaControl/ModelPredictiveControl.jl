const IntRangeOrVector = Union{UnitRange{Int}, Vector{Int}}

@doc raw"""
Abstract supertype of [`LinModel`](@ref) and [`NonLinModel`](@ref) types.

---

    (model::SimModel)(d=[]) -> y

Functor allowing callable `SimModel` object as an alias for [`evaloutput`](@ref).

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_)->-x + u, (x,_)->x .+ 20, 10.0, 1, 1, 1);

julia> y = model()
1-element Vector{Float64}:
 20.0
```
"""
abstract type SimModel{NT<:Real} end

@doc raw"""
    setop!(model::SimModel; uop=nothing, yop=nothing, dop=nothing) -> model

Set `model` inputs `uop`, outputs `yop` and measured disturbances `dop` operating points.

The state-space model with operating points (a.k.a. nominal values) is:
```math
\begin{aligned}
    \mathbf{x}(k+1) &=  \mathbf{A x}(k) + \mathbf{B_u u_0}(k) + \mathbf{B_d d_0}(k) \\
    \mathbf{y_0}(k) &=  \mathbf{C x}(k) + \mathbf{D_d d_0}(k)
\end{aligned}
```
in which the `uop`, `yop` and `dop` vectors evaluate:
```math
\begin{aligned}
    \mathbf{u_0}(k) &= \mathbf{u}(k) - \mathbf{u_{op}} \\
    \mathbf{y_0}(k) &= \mathbf{y}(k) - \mathbf{y_{op}} \\
    \mathbf{d_0}(k) &= \mathbf{d}(k) - \mathbf{d_{op}} 
\end{aligned}
```
The structure is similar if `model` is a `NonLinModel`:
```math
\begin{aligned}
    \mathbf{x}(k+1) &= \mathbf{f}\Big(\mathbf{x}(k), \mathbf{u_0}(k), \mathbf{d_0}(k)\Big)\\
    \mathbf{y_0}(k) &= \mathbf{h}\Big(\mathbf{x}(k), \mathbf{d_0}(k)\Big)
\end{aligned}
```

# Examples
```jldoctest
julia> model = setop!(LinModel(tf(3, [10, 1]), 2.0), uop=[50], yop=[20])
Discrete-time linear model with a sample time Ts = 2.0 s and:
 1 manipulated inputs u
 1 states x
 1 outputs y
 0 measured disturbances d
```

"""
function setop!(model::SimModel; uop = nothing, yop = nothing, dop = nothing)
    if !isnothing(uop) 
        size(uop) == (model.nu,) || error("uop size must be $((model.nu,))")
        model.uop[:] = uop
    end
    if !isnothing(yop)
        size(yop) == (model.ny,) || error("yop size must be $((model.ny,))")
        model.yop[:] = yop
    end
    if !isnothing(dop)
        size(dop) == (model.nd,) || error("dop size must be $((model.nd,))")
        model.dop[:] = dop
    end
    return model
end

"""
    setstate!(model::SimModel, x)

Set `model.x` states to values specified by `x`. 
"""
function setstate!(model::SimModel, x)
    size(x) == (model.nx,) || error("x size must be $((model.nx,))")
    model.x[:] = x
    return model
end

function Base.show(io::IO, model::SimModel)
    nu, nd = model.nu, model.nd
    nx, ny = model.nx, model.ny
    n = maximum(ndigits.((nu, nx, ny, nd))) + 1
    println(io, "Discrete-time $(typestr(model)) model with "*
                "a sample time Ts = $(model.Ts) s$(detailstr(model)) and:")
    println(io, "$(lpad(nu, n)) manipulated inputs u")
    println(io, "$(lpad(nx, n)) states x")
    println(io, "$(lpad(ny, n)) outputs y")
    print(io,   "$(lpad(nd, n)) measured disturbances d")
end

typestr(model::SimModel) = "SimModel"
detailstr(model::SimModel) = ""

@doc raw"""
    initstate!(model::SimModel, u, d=[]) -> x

Init `model.x` with manipulated inputs `u` and measured disturbances `d` steady-state.

It calls [`steadystate!(model, u, d)`](@ref):

- If `model` is a [`LinModel`](@ref), the method computes the steady-state of current
  inputs `u` and measured disturbances `d`.
- Else, `model.x` is left unchanged. Use [`setstate!`](@ref) to manually modify it.

# Examples
```jldoctest
julia> model = LinModel(tf(6, [10, 1]), 2.0);

julia> u = [1]; x = initstate!(model, u); y = round.(evaloutput(model), digits=3)
1-element Vector{Float64}:
 6.0
 
julia> x ≈ updatestate!(model, u)
true
```

"""
function initstate!(model::SimModel, u, d=empty(model.x))
    validate_args(model::SimModel, d, u)
    steadystate!(model, u, d)
    return model.x
end

"""
    updatestate!(model::SimModel, u, d=[]) -> x

Update `model.x` states with current inputs `u` and measured disturbances `d`.

# Examples
```jldoctest
julia> model = LinModel(ss(1.0, 1.0, 1.0, 0, 1.0));

julia> x = updatestate!(model, [1])
1-element Vector{Float64}:
 1.0
```
"""
function updatestate!(model::SimModel{NT}, u, d=empty(model.x)) where NT <: Real
    validate_args(model::SimModel, d, u)
    xnext = Vector{NT}(undef, model.nx)
    f!(xnext, model, model.x, u - model.uop, d - model.dop)
    model.x .= xnext
    return model.x
end

"""
    evaloutput(model::SimModel, d=[]) -> y

Evaluate `SimModel` outputs `y` from `model.x` states and measured disturbances `d`.

Calling a [`SimModel`](@ref) object calls this `evaloutput` method.

# Examples
```jldoctest
julia> model = setop!(LinModel(tf(2, [10, 1]), 5.0), yop=[20]);

julia> y = evaloutput(model)
1-element Vector{Float64}:
 20.0
```
"""
function evaloutput(model::SimModel{NT}, d=empty(model.x)) where NT <: Real
    validate_args(model, d)
    y = Vector{NT}(undef, model.ny)
    h!(y, model, model.x, d - model.dop)
    y .+= model.yop
    return y
end

"""
    validate_args(model::SimModel, d, u=nothing)

Check `d` and `u` (if provided) sizes against `model` dimensions.
"""
function validate_args(model::SimModel, d, u=nothing)
    nu, nd = model.nu, model.nd
    size(d) ≠ (nd,) && throw(DimensionMismatch("d size $(size(d)) ≠ meas. dist. size ($nd,)"))
    if !isnothing(u)
        size(u) ≠ (nu,) && throw(DimensionMismatch("u size $(size(u)) ≠ manip. input size ($nu,)"))
    end
end

"Convert vectors to single column matrices when necessary."
to_mat(A::AbstractVector) = reshape(A, length(A), 1)
to_mat(A::AbstractMatrix) = A

"Functor allowing callable `SimModel` object as an alias for `evaloutput`."
(model::SimModel)(d=empty(model.x)) = evaloutput(model::SimModel, d)

include("model/linmodel.jl")
include("model/solver.jl")
include("model/nonlinmodel.jl")
include("model/linearization.jl")