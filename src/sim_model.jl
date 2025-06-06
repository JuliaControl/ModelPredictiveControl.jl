@doc raw"""
Abstract supertype of [`LinModel`](@ref) and [`NonLinModel`](@ref) types.

---

    (model::SimModel)(d=[]) -> y

Functor allowing callable `SimModel` object as an alias for [`evaloutput`](@ref).

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_,_)->-x + u, (x,_,_)->x .+ 20, 4, 1, 1, 1, solver=nothing);

julia> y = model()
1-element Vector{Float64}:
 20.0
```
"""
abstract type SimModel{NT<:Real} end

struct SimModelBuffer{NT<:Real}
    u::Vector{NT}
    x::Vector{NT}
    y::Vector{NT}
    d::Vector{NT}
    k::Vector{NT}
    empty::Vector{NT}
end

@doc raw"""
    SimModelBuffer{NT}(nu::Int, nx::Int, ny::Int, nd::Int, ni::Int=0)

Create a buffer for `SimModel` objects for inputs, states, outputs, and disturbances.

The buffer is used to store temporary results during simulation without allocating. The
argument `ni` is the number of intermediate stage of the [`DiffSolver`](@ref), when 
applicable.
"""
function SimModelBuffer{NT}(nu::Int, nx::Int, ny::Int, nd::Int, ni::Int=0) where {NT<:Real}
    u = Vector{NT}(undef, nu)
    x = Vector{NT}(undef, nx)
    y = Vector{NT}(undef, ny)
    d = Vector{NT}(undef, nd)
    k = Vector{NT}(undef, nx*(ni+1)) # the "+1" is necessary because of super-sampling
    empty = Vector{NT}(undef, 0)
    return SimModelBuffer{NT}(u, x, y, d, k, empty)
end


@doc raw"""
    setop!(model; uop=nothing, yop=nothing, dop=nothing, xop=nothing, fop=nothing) -> model

Set the operating points of `model` (both [`LinModel`](@ref) and [`NonLinModel`](@ref)).

Introducing deviations vectors around manipulated input `uop`, model output `yop`, measured
disturbance `dop`, and model state `xop` operating points (a.k.a. nominal values):
```math
\begin{align*}
    \mathbf{u_0}(k) &= \mathbf{u}(k) - \mathbf{u_{op}} \\
    \mathbf{d_0}(k) &= \mathbf{d}(k) - \mathbf{d_{op}} \\
    \mathbf{y_0}(k) &= \mathbf{y}(k) - \mathbf{y_{op}} \\
    \mathbf{x_0}(k) &= \mathbf{x}(k) - \mathbf{x_{op}} \\
\end{align*}
```
The state-space description of [`LinModel`](@ref) around the operating points is:
```math
\begin{aligned}
    \mathbf{x_0}(k+1) &= \mathbf{A x_0}(k) + \mathbf{B_u u_0}(k) + \mathbf{B_d d_0}(k) 
                         + \mathbf{f_{op}}   - \mathbf{x_{op}}                                              \\
    \mathbf{y_0}(k)   &= \mathbf{C x_0}(k) + \mathbf{D_d d_0}(k) 
\end{aligned}
```
and, for [`NonLinModel`](@ref):
```math
\begin{aligned}
    \mathbf{x_0}(k+1) &= \mathbf{f}\Big(\mathbf{x_0}(k), \mathbf{u_0}(k), \mathbf{d_0}(k), \mathbf{p}\Big) 
                            + \mathbf{f_{op}} - \mathbf{x_{op}}                                             \\
    \mathbf{y_0}(k)   &= \mathbf{h}\Big(\mathbf{x_0}(k), \mathbf{d_0}(k), \mathbf{p}\Big)
\end{aligned}
```
The state `xop` and the additional `fop` operating points are frequently zero e.g.: when 
`model` comes from system identification. The two vectors are internally used by
[`linearize`](@ref) for non-equilibrium points.

# Examples
```jldoctest
julia> model = setop!(LinModel(tf(3, [10, 1]), 2.0), uop=[50], yop=[20])
LinModel with a sample time Ts = 2.0 s and:
 1 manipulated inputs u
 1 states x
 1 outputs y
 0 measured disturbances d

julia> y = model()
1-element Vector{Float64}:
 20.0
```

"""
function setop!(
    model::SimModel; uop=nothing, yop=nothing, dop=nothing, xop=nothing, fop=nothing
)
    if !isnothing(uop) 
        size(uop) == (model.nu,) || error("uop size must be $((model.nu,))")
        model.uop .= uop
    end
    if !isnothing(yop)
        size(yop) == (model.ny,) || error("yop size must be $((model.ny,))")
        model.yop .= yop
    end
    if !isnothing(dop)
        size(dop) == (model.nd,) || error("dop size must be $((model.nd,))")
        model.dop .= dop
    end
    if !isnothing(xop)
        size(xop) == (model.nx,) || error("xop size must be $((model.nx,))")
        model.xop .= xop
    end
    if !isnothing(fop)
        size(fop) == (model.nx,) || error("fop size must be $((model.nx,))")
        model.fop .= fop
    end
    return model
end

@doc raw"""
    setname!(model::SimModel; u=nothing, y=nothing, d=nothing, x=nothing) -> model

Set the names of `model` inputs `u`, outputs `y`, disturbances `d`, and states `x`.

The keyword arguments `u`, `y`, `d`, and `x` must be vectors of strings. The strings are
used in the plotting functions.

# Examples
```jldoctest
julia> model = setname!(LinModel(tf(3, [10, 1]), 2.0), u=["\$A\$ (%)"], y=["\$T\$ (∘C)"])
LinModel with a sample time Ts = 2.0 s and:
 1 manipulated inputs u
 1 states x
 1 outputs y
 0 measured disturbances d
```
"""
function setname!(model::SimModel; u=nothing, y=nothing, d=nothing, x=nothing)
    if !isnothing(u)
        size(u) == (model.nu,) || throw(DimensionMismatch("u size must be $((model.nu,))"))
        model.uname .= u
    end
    if !isnothing(y)
        size(y) == (model.ny,) || throw(DimensionMismatch("y size must be $((model.ny,))"))
        model.yname .= y
    end
    if !isnothing(d)
        size(d) == (model.nd,) || throw(DimensionMismatch("d size must be $((model.nd,))"))
        model.dname .= d
    end
    if !isnothing(x)
        size(x) == (model.nx,) || throw(DimensionMismatch("x size must be $((model.nx,))"))
        model.xname .= x
    end
    return model
end

"""
    setstate!(model::SimModel, x) -> model

Set `model.x0` to `x - model.xop` from the argument `x`. 
"""
function setstate!(model::SimModel, x)
    size(x) == (model.nx,) || error("x size must be $((model.nx,))")
    model.x0 .= x .- model.xop
    return model
end

detailstr(model::SimModel) = ""

@doc raw"""
    initstate!(model::SimModel, u, d=[]) -> x

Init `model.x0` with manipulated inputs `u` and meas. dist. `d` steady-state.

The method tries to initialize the model state ``\mathbf{x}`` at steady-state. It removes
the operating points on `u` and `d` and calls [`steadystate!`](@ref):

- If `model` is a [`LinModel`](@ref), the method computes the steady-state of current
  inputs `u` and measured disturbances `d`.
- Else, `model.x0` is left unchanged. Use [`setstate!`](@ref) to manually modify it.

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
function initstate!(model::SimModel, u, d=model.buffer.empty)
    validate_args(model::SimModel, d, u)
    u0, d0 = model.buffer.u, model.buffer.d
    u0 .= u .- model.uop
    d0 .= d .- model.dop
    steadystate!(model, u0, d0)
    x  = model.buffer.x
    x .= model.x0 .+ model.xop
    return x
end

@doc raw"""
    preparestate!(model::SimModel) -> x

Do nothing for [`SimModel`](@ref) and return the current model state ``\mathbf{x}(k)``. 
"""
function preparestate!(model::SimModel)
    x  = model.buffer.x
    x .= model.x0 .+ model.xop
    return x 
end

function preparestate!(model::SimModel, ::Any , ::Any=model.buffer.empty)
    Base.depwarn(
        "The method preparestate!(model::SimModel, u, d=[]) is deprecated. Use "*
        " preparestate!(model::SimModel) instead.",
        :preparestate!,
    )
    return preparestate!(model)
end

@doc raw"""
    updatestate!(model::SimModel, u, d=[]) -> xnext

Update `model.x0` states with current inputs `u` and meas. dist. `d` for the next time step.

The method computes and returns the model state for the next time step ``\mathbf{x}(k+1)``.

# Examples
```jldoctest
julia> model = LinModel(ss(1.0, 1.0, 1.0, 0, 1.0));

julia> x = updatestate!(model, [1])
1-element Vector{Float64}:
 1.0
```
"""
function updatestate!(model::SimModel{NT}, u, d=model.buffer.empty) where NT <: Real
    validate_args(model::SimModel, d, u)
    u0, d0, x0next, k0 = model.buffer.u, model.buffer.d, model.buffer.x, model.buffer.k
    u0 .= u .- model.uop
    d0 .= d .- model.dop
    f!(x0next, k0, model, model.x0, u0, d0, model.p)
    x0next  .+= model.fop .- model.xop
    model.x0 .= x0next
    xnext   = x0next
    xnext .+= model.xop
    return xnext
end

@doc raw"""
    evaloutput(model::SimModel, d=[]) -> y

Evaluate `SimModel` outputs `y` from `model.x0` states and measured disturbances `d`.

It returns `model` output at the current time step ``\mathbf{y}(k)``. Calling a 
[`SimModel`](@ref) object calls this `evaloutput` method.

# Examples
```jldoctest
julia> model = setop!(LinModel(tf(2, [10, 1]), 5.0), yop=[20]);

julia> y = evaloutput(model)
1-element Vector{Float64}:
 20.0
```
"""
function evaloutput(model::SimModel{NT}, d=model.buffer.empty) where NT <: Real
    validate_args(model, d)
    d0, y0  = model.buffer.d, model.buffer.y
    d0 .= d .- model.dop
    h!(y0, model, model.x0, d0, model.p)
    y   = y0
    y .+= model.yop
    return y
end

"""
    savetime!(model::SimModel) -> t

Set `model.t` to `time()`  and return the value.

Used in conjunction with [`periodsleep`](@ref) for simple soft real-time simulations. Call
this function before any other in the simulation loop.
"""
function savetime!(model::SimModel)
    model.t[] = time()
    return model.t[]
end

"""
    periodsleep(model::SimModel, busywait=false) -> nothing

Sleep for `model.Ts` s minus the time elapsed since the last call to [`savetime!`](@ref).

It calls [`sleep`](@extref Julia Base.sleep) if `busywait` is `false`. Else, a simple 
`while` loop implements busy-waiting. As a rule-of-thumb, busy-waiting should be used if 
`model.Ts < 0.1` s, since the accuracy of `sleep` is around 1 ms. Can be used to implement
simple soft real-time simulations, see the example below.

!!! warning
    The allocations in Julia are garbage-collected (GC) automatically. This can affect the 
    timing. In such cases, you can temporarily stop the GC with `GC.enable(false)`, and
    restart it at a convenient time e.g.: just before calling `periodsleep`.

# Examples
```jldoctest
julia> model = LinModel(tf(2, [0.3, 1]), 0.25);

julia> function sim_realtime!(model)
           t_0 = time()
           for i=1:3
               t = savetime!(model)      # first function called
               println(round(t - t_0, digits=3))
               updatestate!(model, [1])
               periodsleep(model, true)  # last function called
           end
       end;

julia> sim_realtime!(model)
0.0
0.25
0.5
```
"""
function periodsleep(model::SimModel, busywait=false)
    if !busywait
        model.Ts < 0.1 && @warn "busy-waiting is recommended for Ts < 0.1 s"
        computing_time = time() - model.t[]
        sleep_time = model.Ts - computing_time
        sleep_time > 0 && sleep(sleep_time)
    else
        while time() - model.t[] < model.Ts
        end
    end
    return nothing
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
to_mat(A::AbstractVector, _ ...) = reshape(A, length(A), 1)
to_mat(A::AbstractMatrix, _ ...) = A
to_mat(A::Real, dims...) = fill(A, dims)

include("model/linmodel.jl")
include("model/solver.jl")
include("model/linearization.jl")
include("model/nonlinmodel.jl")

function Base.show(io::IO, model::SimModel)
    nu, nd = model.nu, model.nd
    nx, ny = model.nx, model.ny
    n = maximum(ndigits.((nu, nx, ny, nd))) + 1
    println(io, "$(typeof(model).name.name) with a sample time Ts = $(model.Ts) s"*
                "$(detailstr(model)) and:")
    println(io, "$(lpad(nu, n)) manipulated inputs u")
    println(io, "$(lpad(nx, n)) states x")
    println(io, "$(lpad(ny, n)) outputs y")
    print(io,   "$(lpad(nd, n)) measured disturbances d")
end

"Functor allowing callable `SimModel` object as an alias for `evaloutput`."
(model::SimModel)(d=model.buffer.empty) = evaloutput(model::SimModel, d)