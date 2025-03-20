"""
    get_linearization_func(NT, f!, h!, nu, nx, ny, nd, p, backend) -> linfunc!

Return the `linfunc!` function that computes the Jacobians of `f!` and `h!` functions.

The function has the following signature: 
```
    linfunc!(xnext, y, A, Bu, C, Bd, Dd, backend, x, u, d, cst_x, cst_u, cst_d) -> nothing
```
and it should modifies in-place all the arguments before `backend`. The `backend` argument
is an `AbstractADType` object from `DifferentiationInterface`. The `cst_x`, `cst_u` and 
`cst_d` are `DifferentiationInterface.Constant` objects with the linearization points.
"""
function get_linearization_func(NT, f!, h!, nu, nx, ny, nd, p, backend)
    f_x!(xnext, x, u, d) = f!(xnext, x, u, d, p)
    f_u!(xnext, u, x, d) = f!(xnext, x, u, d, p)
    f_d!(xnext, d, x, u) = f!(xnext, x, u, d, p)
    h_x!(y, x, d) = h!(y, x, d, p)
    h_d!(y, d, x) = h!(y, x, d, p)
    strict  = Val(true)
    xnext = zeros(NT, nx)
    y = zeros(NT, ny)
    x = zeros(NT, nx)
    u = zeros(NT, nu)
    d = zeros(NT, nd)
    cst_x = Constant(x)
    cst_u = Constant(u)
    cst_d = Constant(d)
    A_prep  = prepare_jacobian(f_x!, xnext, backend, x, cst_u, cst_d; strict)
    Bu_prep = prepare_jacobian(f_u!, xnext, backend, u, cst_x, cst_d; strict)
    Bd_prep = prepare_jacobian(f_d!, xnext, backend, d, cst_x, cst_u; strict)
    C_prep  = prepare_jacobian(h_x!, y,     backend, x, cst_d       ; strict)
    Dd_prep = prepare_jacobian(h_d!, y,     backend, d, cst_x       ; strict)
    function linfunc!(xnext, y, A, Bu, C, Bd, Dd, backend, x, u, d, cst_x, cst_u, cst_d)
        # all the arguments before `backend` are mutated in this function
        jacobian!(f_x!, xnext, A,  A_prep,  backend, x, cst_u, cst_d)
        jacobian!(f_u!, xnext, Bu, Bu_prep, backend, u, cst_x, cst_d)
        jacobian!(f_d!, xnext, Bd, Bd_prep, backend, d, cst_x, cst_u)
        jacobian!(h_x!, y,     C,  C_prep,  backend, x, cst_d)
        jacobian!(h_d!, y,     Dd, Dd_prep, backend, d, cst_x)
        return nothing
    end
    return linfunc!
end


@doc raw"""
    linearize(model::SimModel; x=model.x0+model.xop, u=model.uop, d=model.dop) -> linmodel

Linearize `model` at the operating points `x`, `u`, `d` and return the [`LinModel`](@ref).

The arguments `x`, `u` and `d` are the linearization points for the state ``\mathbf{x}``,
manipulated input ``\mathbf{u}`` and measured disturbance ``\mathbf{d}``, respectively (not
necessarily an equilibrium, details in Extended Help). By default, [`ForwardDiff`](@extref ForwardDiff)
automatically computes the Jacobians of ``\mathbf{f}`` and ``\mathbf{h}`` functions. Modify
the `jacobian` keyword argument at the construction of `model` to swap the backend.

!!! warning
    See Extended Help if you get an error like:    
    `MethodError: no method matching (::var"##")(::Vector{ForwardDiff.Dual})`.

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_,_)->x.^3 + u, (x,_,_)->x, 0.1, 1, 1, 1, solver=nothing);

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
        \mathbf{x}(k+1) &= \mathbf{f}\Big(\mathbf{x}(k), \mathbf{u}(k), \mathbf{d}(k), \mathbf{p}\Big)  \\
        \mathbf{y}(k)   &= \mathbf{h}\Big(\mathbf{x}(k), \mathbf{d}(k), \mathbf{p}\Big)
    \end{aligned}
    ```
    its linearization at the operating point ``\mathbf{x_{op}, u_{op}, d_{op}}`` is:
    ```math
    \begin{aligned}
        \mathbf{x_0}(k+1) &≈ \mathbf{A x_0}(k) + \mathbf{B_u u_0}(k) + \mathbf{B_d d_0}(k)  
                            + \mathbf{f(x_{op}, u_{op}, d_{op}, p)} - \mathbf{x_{op}}         \\
        \mathbf{y_0}(k)   &≈ \mathbf{C x_0}(k) + \mathbf{D_d d_0}(k) 
    \end{aligned}
    ```
    based on the deviation vectors ``\mathbf{x_0, u_0, d_0, y_0}`` introduced in [`setop!`](@ref)
    documentation, and the Jacobian matrices:
    ```math
    \begin{aligned}
        \mathbf{A}   &= \left. \frac{∂\mathbf{f(x, u, d, p)}}{∂\mathbf{x}} \right|_{\mathbf{x=x_{op},\, u=u_{op},\, d=d_{op}}} \\
        \mathbf{B_u} &= \left. \frac{∂\mathbf{f(x, u, d, p)}}{∂\mathbf{u}} \right|_{\mathbf{x=x_{op},\, u=u_{op},\, d=d_{op}}} \\
        \mathbf{B_d} &= \left. \frac{∂\mathbf{f(x, u, d, p)}}{∂\mathbf{d}} \right|_{\mathbf{x=x_{op},\, u=u_{op},\, d=d_{op}}} \\
        \mathbf{C}   &= \left. \frac{∂\mathbf{h(x, d, p)}}{∂\mathbf{x}}    \right|_{\mathbf{x=x_{op},\, d=d_{op}}}             \\
        \mathbf{D_d} &= \left. \frac{∂\mathbf{h(x, d, p)}}{∂\mathbf{d}}    \right|_{\mathbf{x=x_{op},\, d=d_{op}}}
    \end{aligned}
    ```
    Following [`setop!`](@ref) notation, we find:
    ```math
    \begin{aligned}
        \mathbf{f_{op}} &= \mathbf{f(x_{op}, u_{op}, d_{op}, p)} \\
        \mathbf{y_{op}} &= \mathbf{h(x_{op}, d_{op}, p)}
    \end{aligned}
    ```
    Notice that ``\mathbf{f_{op} - x_{op} = 0}`` if the point is an equilibrium. The 
    equations are similar if the nonlinear model has nonzero operating points.

    Automatic differentiation (AD) allows exact Jacobians. The [`NonLinModel`](@ref) `f` and
    `h` functions must be compatible with this feature though. See [`JuMP` documentation](@extref JuMP Common-mistakes-when-writing-a-user-defined-operator)
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

The keyword arguments are identical to [`linearize`](@ref). The code is allocation-free if
`model` simulations does not allocate.

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_,_)->x.^3 + u, (x,_,_)->x, 0.1, 1, 1, 1, solver=nothing);

julia> linmodel = linearize(model, x=[10.0], u=[0.0]); linmodel.A
1×1 Matrix{Float64}:
 300.0

julia> linearize!(linmodel, model, x=[20.0], u=[0.0]); linmodel.A
1×1 Matrix{Float64}:
 1200.0
```
"""
function linearize!(
    linmodel::LinModel{NT}, model::SimModel; x=model.x0+model.xop, u=model.uop, d=model.dop
) where NT<:Real
    nonlinmodel = model
    buffer = nonlinmodel.buffer
    # --- remove the operating points of the nonlinear model (typically zeros) ---
    x0, u0, d0 = buffer.x, buffer.u, buffer.d
    x0 .= x .- nonlinmodel.xop
    u0 .= u .- nonlinmodel.uop
    d0 .= d .- nonlinmodel.dop
    # --- compute the Jacobians at linearization points ---
    linearize_core!(linmodel, nonlinmodel, x0, u0, d0)
    # --- compute the nonlinear model output at operating points ---
    xnext0, y0 = linmodel.buffer.x, linmodel.buffer.y
    h!(y0, nonlinmodel, x0, d0, model.p)
    y  = y0
    y .= y0 .+ nonlinmodel.yop
    # --- compute the nonlinear model next state at operating points ---
    f!(xnext0, nonlinmodel, x0, u0, d0, model.p)
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

"Call `linfunc!` function to compute the Jacobians of `model` at the linearization point."
function linearize_core!(linmodel::LinModel, model::SimModel, x0, u0, d0)
    xnext0, y0 = linmodel.buffer.x, linmodel.buffer.y
    A, Bu, C, Bd, Dd = linmodel.A, linmodel.Bu, linmodel.C, linmodel.Bd, linmodel.Dd
    cst_x = Constant(x0)
    cst_u = Constant(u0)
    cst_d = Constant(d0)
    backend = model.jacobian
    model.linfunc!(xnext0, y0, A, Bu, C, Bd, Dd, backend, x0, u0, d0, cst_x, cst_u, cst_d)
    return nothing
end

"Copy the state-space matrices of `model` to `linmodel` if `model` is already linear."
function linearize_core!(linmodel::LinModel, model::LinModel, _ , _ , _ )
    linmodel.A  .= model.A
    linmodel.Bu .= model.Bu
    linmodel.C  .= model.C
    linmodel.Bd .= model.Bd
    linmodel.Dd .= model.Dd
    return nothing
end