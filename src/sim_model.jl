const IntRangeOrVector = Union{UnitRange{Int}, Vector{Int}}

@doc raw"""
Abstract supertype of [`LinModel`](@ref) and [`NonLinModel`](@ref) types.

---

    (model::SimModel)(d=Float64[])

Functor allowing callable `SimModel` object as an alias for [`evaloutput`](@ref).

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_)->-x + u, (x,_)->x .+ 20, 10.0, 1, 1, 1);

julia> y = model()
1-element Vector{Float64}:
 20.0
```
"""
abstract type SimModel end

@doc raw"""
    setop!(model::SimModel; uop=nothing, yop=nothing, dop=nothing)

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
julia> model = setop!(LinModel(tf(3, [10, 1]), 2), uop=[50], yop=[20])
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
    println(io, "Discrete-time $(typestr(model)) model with "*
                "a sample time Ts = $(model.Ts) s and:")
    println(io, " $(model.nu) manipulated inputs u")
    println(io, " $(model.nx) states x")
    println(io, " $(model.ny) outputs y")
    print(io,   " $(model.nd) measured disturbances d")
end

"""
    updatestate!(model::SimModel, u, d=Float64[])

Update `model.x` states with current inputs `u` and measured disturbances `d`.

# Examples
```jldoctest
julia> model = LinModel(ss(1, 1, 1, 0, 1));

julia> x = updatestate!(model, [1])
1-element Vector{Float64}:
 1.0
```
"""
function updatestate!(model::SimModel, u, d=Float64[])
    model.x[:] = f(model, model.x, u - model.uop, d - model.dop)
    return model.x
end

"""
    evaloutput(model::SimModel, d=Float64[])

Evaluate `SimModel` outputs `y` from `model.x` states and measured disturbances `d`.

Calling a [`SimModel`](@ref) object calls this `evaloutput` method.

# Examples
```jldoctest
julia> model = setop!(LinModel(tf(2, [10, 1]), 5), yop=[20]);

julia> y = evaloutput(model)
1-element Vector{Float64}:
 20.0
```
"""
evaloutput(model::SimModel, d=Float64[]) = h(model, model.x, d - model.dop) + model.yop


"Functor allowing callable `SimModel` object as an alias for `evaloutput`."
(model::SimModel)(d=Float64[]) = evaloutput(model::SimModel, d)

include("model/linmodel.jl")
include("model/nonlinmodel.jl")