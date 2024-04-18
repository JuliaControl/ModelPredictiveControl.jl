"""
    LinModel(model::NonLinModel; x=model.x0+model.xop, u=model.uop, d=model.dop)

Call [`linearize(model; x, u, d)`](@ref) and return the resulting linear model.
"""
LinModel(model::NonLinModel; kwargs...) = linearize(model; kwargs...)

@doc raw"""
    linearize(model::SimModel; x=model.x0+model.xop, u=model.uop, d=model.dop) -> linmodel

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
        \mathbf{x}(k+1) &= \mathbf{f}\Big(\mathbf{x}(k), \mathbf{u}(k), \mathbf{d}(k)\Big)  \\
        \mathbf{y}(k)   &= \mathbf{h}\Big(\mathbf{x}(k), \mathbf{d}(k)\Big)
    \end{align*}
    ```
    its linearization at the operating point ``\mathbf{x_{op}, u_{op}, d_{op}}`` is:
    ```math
    \begin{align*}
        \mathbf{x_0}(k+1) &≈ \mathbf{A x_0}(k) + \mathbf{B_u u_0}(k) + \mathbf{B_d d_0}(k)  
                            + \mathbf{f(x_{op}, u_{op}, d_{op})} - \mathbf{x_{op}}         \\
        \mathbf{y_0}(k)   &≈ \mathbf{C x_0}(k) + \mathbf{D_d d_0}(k) 
    \end{align*}
    ```
    based on the deviation vectors ``\mathbf{x_0, u_0, d_0, y_0}`` introduced in [`setop!`](@ref)
    documentation, and the Jacobian matrices:
    ```math
    \begin{align*}
        \mathbf{A}   &= \left. \frac{∂\mathbf{f(x, u, d)}}{∂\mathbf{x}} \right|_{\mathbf{x=x_{op},\, u=u_{op},\, d=d_{op}}} \\
        \mathbf{B_u} &= \left. \frac{∂\mathbf{f(x, u, d)}}{∂\mathbf{u}} \right|_{\mathbf{x=x_{op},\, u=u_{op},\, d=d_{op}}} \\
        \mathbf{B_d} &= \left. \frac{∂\mathbf{f(x, u, d)}}{∂\mathbf{d}} \right|_{\mathbf{x=x_{op},\, u=u_{op},\, d=d_{op}}} \\
        \mathbf{C}   &= \left. \frac{∂\mathbf{h(x, d)}}{∂\mathbf{x}}    \right|_{\mathbf{x=x_{op},\, d=d_{op}}}             \\
        \mathbf{D_d} &= \left. \frac{∂\mathbf{h(x, d)}}{∂\mathbf{d}}    \right|_{\mathbf{x=x_{op},\, d=d_{op}}}
    \end{align*}
    ```
    Following [`setop!`](@ref) notation, we find:
    ```math
        \mathbf{f_{op}} &= \mathbf{f(x_{op}, u_{op}, d_{op})} \\
        \mathbf{y_{op}} &= \mathbf{h(x_{op}, d_{op})}
    \end{align*}
    ```
    Notice that ``\mathbf{f_{op} - x_{op} = 0}`` if the point is an equilibrium. The 
    equations are similar if the nonlinear model has nonzero operating points.

    Automatic differentiation (AD) allows exact Jacobians. The [`NonLinModel`](@ref) `f` and
    `h` functions must be compatible with this feature though. See [Automatic differentiation](https://jump.dev/JuMP.jl/stable/manual/nlp/#Automatic-differentiation)
    for common mistakes when writing these functions.
"""
function linearize(model::SimModel{NT}; kwargs...) where NT<:Real
    nu, nx, ny, nd = model.nu, model.nx, model.ny, model.nd
    A  = Matrix{NT}(undef, nx, nx)
    Bu = Matrix{NT}(undef, nx, nu) 
    C  = Matrix{NT}(undef, ny, nx)
    Bd = Matrix{NT}(undef, nx, nd)
    Dd = Matrix{NT}(undef, ny, nd)
    linmodel = LinModel(A, Bu, C, Bd, Dd, model.Ts)
    return linearize!(linmodel, model; kwargs...)
end

"""
    linearize!(linmodel::LinModel, model::SimModel; <keyword arguments>) -> linmodel

Linearize `model` and store the result in `linmodel`.

The keyword arguments are identical to [`linearize`](@ref).

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_)->x.^3 + u, (x,_)->x, 0.1, 1, 1, 1, solver=nothing);

julia> linmodel = linearize(model, x=[10.0], u=[0.0]); linmodel.A
1×1 Matrix{Float64}:
 300.0

julia> linearize!(linmodel, model, x=[20.0], u=[0.0]); linmodel.A
1×1 Matrix{Float64}:
 1200.0
"""
function linearize!(
    linmodel::LinModel, model::SimModel; x=model.x0+model.xop, u=model.uop, d=model.dop
)
    nonlinmodel = model
    u0, d0 = u - nonlinmodel.uop, d - nonlinmodel.dop
    xnext, y = similar(x), similar(nonlinmodel.yop)
    # --- compute the nonlinear model output at operating points ---
    h!(y, nonlinmodel, x, d0)
    y .= y .+ nonlinmodel.yop
    # --- compute the Jacobians at linearization points ---
    A, Bu, Bd, C, Dd = linmodel.A, linmodel.Bu, linmodel.Bd, linmodel.C, linmodel.Dd
    ForwardDiff.jacobian!(A,  (xnext, x)  -> f!(xnext, nonlinmodel, x, u0, d0), xnext, x)
    ForwardDiff.jacobian!(Bu, (xnext, u0) -> f!(xnext, nonlinmodel, x, u0, d0), xnext, u0)
    ForwardDiff.jacobian!(Bd, (xnext, d0) -> f!(xnext, nonlinmodel, x, u0, d0), xnext, d0)
    ForwardDiff.jacobian!(C,  (y, x)  -> h!(y, nonlinmodel, x, d0), y, x)
    ForwardDiff.jacobian!(Dd, (y, d0) -> h!(y, nonlinmodel, x, d0), y, d0)
    # --- compute the nonlinear model next state at operating points ---
    f!(xnext, nonlinmodel, x, u, d)
    xnext .+= nonlinmodel.fop .- nonlinmodel.xop
    # --- modify the linear model operating points ---
    linmodel.uop .= u
    linmodel.yop .= y
    linmodel.dop .= d
    linmodel.xop .= x
    linmodel.fop .= xnext
    # --- reset the state of the linear model ---
    linmodel.x0 .= 0 # state deviation vector is always x0=0 after a linearization
    return linmodel
end