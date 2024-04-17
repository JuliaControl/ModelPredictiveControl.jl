
"""
    LinModel(model::NonLinModel; x=model.x, u=model.uop, d=model.dop)

Call [`linearize(model; x, u, d)`](@ref) and return the resulting linear model.
"""
LinModel(model::NonLinModel; kwargs...) = linearize(model; kwargs...)

@doc raw"""
    linearize(model::NonLinModel; x=model.x, u=model.uop, d=model.dop) -> linmodel

Linearize `model` at the operating points `x`, `u`, `d` and return the [`LinModel`](@ref).

The arguments `x`, `u` and `d` are the linearization points for the state ``\mathbf{x}``,
manipulated input ``\mathbf{u}`` and measured disturbance ``\mathbf{d}``, respectively (not
necessarily an equilibrium, details in Extended Help). The Jacobians of ``\mathbf{f}`` and 
``\mathbf{h}`` functions are automatically computed with [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl).

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
    With the nonlinear state-space model:
    ```math
    \begin{align*}
        \mathbf{x}(k+1) &= \mathbf{f}\Big(\mathbf{x}(k), \mathbf{u}(k), \mathbf{d}(k)\Big) \\
        \mathbf{y}(k)   &= \mathbf{h}\Big(\mathbf{x}(k), \mathbf{d}(k)\Big)
    \end{align*}
    ```
    its linearization at the points ``\mathbf{x̄, ū}`` and ``\mathbf{d̄}`` is:
    ```math
    \begin{align*}
        \mathbf{x_0}(k+1) &≈ \mathbf{A}(k) \mathbf{x_0}(k) 
                                + \mathbf{B_u}(k) \big(\mathbf{u}(k) - \mathbf{ū}\big) 
                                + \mathbf{B_d}(k) \big(\mathbf{d}(k) - \mathbf{d̄}\big)  
                                + \mathbf{f\big(x̄, ū, d̄\big)} - \mathbf{x̄}               \\
        \mathbf{y}(k)     &≈ \mathbf{C}(k) \mathbf{x_0}(k) 
                                + \mathbf{D_d} \big(\mathbf{d}(k) - \mathbf{d̄}\big)
                                + \mathbf{h\big(x̄, d̄\big)}
    \end{align*}
    ```
    in which the Jacobians are:
    ```math
    \begin{align*}
        \mathbf{A}(k)   &= \left. \frac{∂\mathbf{f(x, u, d)}}{∂\mathbf{x}} \right|_{\mathbf{x = x̄,\, u = ū,\, d = x̄}} \\
        \mathbf{B_u}(k) &= \left. \frac{∂\mathbf{f(x, u, d)}}{∂\mathbf{u}} \right|_{\mathbf{x = x̄,\, u = ū,\, d = x̄}} \\
        \mathbf{B_d}(k) &= \left. \frac{∂\mathbf{f(x, u, d)}}{∂\mathbf{d}} \right|_{\mathbf{x = x̄,\, u = ū,\, d = x̄}} \\
        \mathbf{C}(k)   &= \left. \frac{∂\mathbf{h(x, d)}}{∂\mathbf{x}}    \right|_{\mathbf{x = x̄,\, d = x̄}}          \\
        \mathbf{D_d}(k) &= \left. \frac{∂\mathbf{h(x, d)}}{∂\mathbf{d}}    \right|_{\mathbf{x = x̄,\, d = x̄}}
    \end{align*}
    ```
    Following [`setop!`](@ref) notation, we find:
    ```math
    \begin{align*}
        \mathbf{u_{op}} &= \mathbf{ū}                         \\
        \mathbf{d_{op}} &= \mathbf{d̄}                         \\
        \mathbf{y_{op}} &= \mathbf{h\big(x̄, d̄\big)}           \\
        \mathbf{x_{op}} &= \mathbf{f\big(x̄, ū, d̄\big)} - \mathbf{x̄}
    \end{align*}
    ```
    Notice that ``\mathbf{x_{op} = 0}`` if the point is an equilibrium. The equations are
    similar if the nonlinear model has nonzero operating points.

    Automatic differentiation (AD) allows exact Jacobians. The [`NonLinModel`](@ref) `f` and
    `h` functions must be compatible with this feature though. See [Automatic differentiation](https://jump.dev/JuMP.jl/stable/manual/nlp/#Automatic-differentiation)
    for common mistakes when writing these functions.
"""
function linearize(model::NonLinModel; x=model.x, u=model.uop, d=model.dop)
    nonlinmodel = model
    u0, d0 = u - nonlinmodel.uop, d - nonlinmodel.dop
    xnext, y = similar(x), similar(nonlinmodel.yop)
    # --- compute the nonlinear model output at linearization points ---
    h!(y, nonlinmodel, x, d0)
    y .= y .+ nonlinmodel.yop
    # --- compute the Jacobians at linearization points ---
    A  = ForwardDiff.jacobian((xnext, x)  -> nonlinmodel.f!(xnext, x, u0, d0), xnext, x)
    Bu = ForwardDiff.jacobian((xnext, u0) -> nonlinmodel.f!(xnext, x, u0, d0), xnext, u0)
    Bd = ForwardDiff.jacobian((xnext, d0) -> nonlinmodel.f!(xnext, x, u0, d0), xnext, d0)
    C  = ForwardDiff.jacobian((y, x)  -> nonlinmodel.h!(y, x, d0), y, x)
    Dd = ForwardDiff.jacobian((y, d0) -> nonlinmodel.h!(y, x, d0), y, d0)
    # --- construct the linear model ---
    linmodel = LinModel(A, Bu, C, Bd, Dd, nonlinmodel.Ts)
    # --- compute the nonlinear model next state at linearization points ---
    f!(xnext, nonlinmodel, x, u, d)
    xnext .= xnext .+ nonlinmodel.xop
    # --- set the operating points of the linear model ---
    uop, dop, yop = u, d, y
    xop   = xnext
    xop .-= x
    setop!(linmodel; uop, yop, dop, xop)    
    return linmodel
end