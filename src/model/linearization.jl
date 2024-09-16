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
    \begin{aligned}
        \mathbf{x}(k+1) &= \mathbf{f}\Big(\mathbf{x}(k), \mathbf{u}(k), \mathbf{d}(k)\Big)  \\
        \mathbf{y}(k)   &= \mathbf{h}\Big(\mathbf{x}(k), \mathbf{d}(k)\Big)
    \end{aligned}
    ```
    its linearization at the operating point ``\mathbf{x_{op}, u_{op}, d_{op}}`` is:
    ```math
    \begin{aligned}
        \mathbf{x_0}(k+1) &≈ \mathbf{A x_0}(k) + \mathbf{B_u u_0}(k) + \mathbf{B_d d_0}(k)  
                            + \mathbf{f(x_{op}, u_{op}, d_{op})} - \mathbf{x_{op}}         \\
        \mathbf{y_0}(k)   &≈ \mathbf{C x_0}(k) + \mathbf{D_d d_0}(k) 
    \end{aligned}
    ```
    based on the deviation vectors ``\mathbf{x_0, u_0, d_0, y_0}`` introduced in [`setop!`](@ref)
    documentation, and the Jacobian matrices:
    ```math
    \begin{aligned}
        \mathbf{A}   &= \left. \frac{∂\mathbf{f(x, u, d)}}{∂\mathbf{x}} \right|_{\mathbf{x=x_{op},\, u=u_{op},\, d=d_{op}}} \\
        \mathbf{B_u} &= \left. \frac{∂\mathbf{f(x, u, d)}}{∂\mathbf{u}} \right|_{\mathbf{x=x_{op},\, u=u_{op},\, d=d_{op}}} \\
        \mathbf{B_d} &= \left. \frac{∂\mathbf{f(x, u, d)}}{∂\mathbf{d}} \right|_{\mathbf{x=x_{op},\, u=u_{op},\, d=d_{op}}} \\
        \mathbf{C}   &= \left. \frac{∂\mathbf{h(x, d)}}{∂\mathbf{x}}    \right|_{\mathbf{x=x_{op},\, d=d_{op}}}             \\
        \mathbf{D_d} &= \left. \frac{∂\mathbf{h(x, d)}}{∂\mathbf{d}}    \right|_{\mathbf{x=x_{op},\, d=d_{op}}}
    \end{aligned}
    ```
    Following [`setop!`](@ref) notation, we find:
    ```math
    \begin{aligned}
        \mathbf{f_{op}} &= \mathbf{f(x_{op}, u_{op}, d_{op})} \\
        \mathbf{y_{op}} &= \mathbf{h(x_{op}, d_{op})}
    \end{aligned}
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
    linmodel = LinModel{NT}(A, Bu, C, Bd, Dd, model.Ts)
    linmodel.uname .= model.uname
    linmodel.xname .= model.xname
    linmodel.yname .= model.yname
    linmodel.dname .= model.dname
    return linearize!(linmodel, model; kwargs...)
end

"""
    linearize!(linmodel::LinModel, model::SimModel; <keyword arguments>) -> linmodel

Linearize `model` and store the result in `linmodel` (in-place).

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
```
"""
function linearize!(
    linmodel::LinModel{NT}, model::SM; x=model.x0+model.xop, u=model.uop, d=model.dop
) where {NT<:Real, SM<:SimModel{NT}}
    nonlinmodel = model
    buffer = nonlinmodel.buffer
    # --- remove the operating points of the nonlinear model (typically zeros) ---
    x0::Vector{NT}, u0::Vector{NT}, d0::Vector{NT} = buffer.x, buffer.u, buffer.d
    u0 .= u .- nonlinmodel.uop
    d0 .= d .- nonlinmodel.dop
    x0 .= x .- nonlinmodel.xop
    # --- compute the Jacobians at linearization points ---
    A, Bu, Bd, C, Dd = linmodel.A, linmodel.Bu, linmodel.Bd, linmodel.C, linmodel.Dd
    xnext0::Vector{NT}, y0::Vector{NT} = linmodel.buffer.x, linmodel.buffer.y
    myf_x0! = (xnext0, x0) -> f!(xnext0, nonlinmodel, x0, u0, d0)
    myf_u0! = (xnext0, u0) -> f!(xnext0, nonlinmodel, x0, u0, d0)
    myf_d0! = (xnext0, d0) -> f!(xnext0, nonlinmodel, x0, u0, d0)
    myh_x0! = (y0, x0) -> h!(y0, nonlinmodel, x0, d0)
    myh_d0! = (y0, d0) -> h!(y0, nonlinmodel, x0, d0)
    ForwardDiff.jacobian!(A,  myf_x0!, xnext0, x0)
    ForwardDiff.jacobian!(Bu, myf_u0!, xnext0, u0)
    ForwardDiff.jacobian!(Bd, myf_d0!, xnext0, d0)
    ForwardDiff.jacobian!(C,  myh_x0!, y0, x0)
    ForwardDiff.jacobian!(Dd, myh_d0!, y0, d0)
    # --- compute the nonlinear model output at operating points ---
    h!(y0, nonlinmodel, x0, d0)
    y  = y0
    y .= y0 .+ nonlinmodel.yop
    # --- compute the nonlinear model next state at operating points ---
    f!(xnext0, nonlinmodel, x0, u0, d0)
    xnext  = xnext0
    xnext .= xnext0 .+ nonlinmodel.fop .- nonlinmodel.xop
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