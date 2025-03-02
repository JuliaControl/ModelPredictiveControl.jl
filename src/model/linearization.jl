"A linearization buffer for the [`linearize`](@ref) function."
struct LinearizationBuffer{
    NT <:Real,
    JB_FUD <:JacobianBuffer,
    JB_FXD <:JacobianBuffer,
    JB_FXU <:JacobianBuffer,
    JB_HD  <:JacobianBuffer,
    JB_HX  <:JacobianBuffer
}
    x::Vector{NT}
    u::Vector{NT}
    d::Vector{NT}
    buffer_f_at_u_d::JB_FUD
    buffer_f_at_x_d::JB_FXD
    buffer_f_at_x_u::JB_FXU
    buffer_h_at_d  ::JB_HD
    buffer_h_at_x  ::JB_HX
    function LinearizationBuffer(
        x::Vector{NT}, 
        u::Vector{NT}, 
        d::Vector{NT},
        buffer_f_at_u_d::JB_FUD, 
        buffer_f_at_x_d::JB_FXD, 
        buffer_f_at_x_u::JB_FXU, 
        buffer_h_at_d::JB_HD, 
        buffer_h_at_x::JB_HX
    ) where {NT<:Real, JB_FUD, JB_FXD, JB_FXU, JB_HD, JB_HX}
        return new{NT, JB_FUD, JB_FXD, JB_FXU, JB_HD, JB_HX}(
            x, u, d, 
            buffer_f_at_u_d, 
            buffer_f_at_x_d, 
            buffer_f_at_x_u, 
            buffer_h_at_d, 
            buffer_h_at_x
        )
    end
    
end

Base.show(io::IO, buffer::LinearizationBuffer) = print(io, "LinearizationBuffer object")

function LinearizationBuffer(NT, f!, h!, nu, nx, ny, nd, p)
    x, u, d, f_at_u_d!, f_at_x_d!, f_at_x_u!, h_at_d!, h_at_x! = get_linearize_funcs(
        NT, f!, h!, nu, nx, ny, nd, p
    )
    xnext, y = Vector{NT}(undef, nx), Vector{NT}(undef, ny) # TODO: to replace ?
    return LinearizationBuffer(
        x, u, d, 
        JacobianBuffer(f_at_u_d!, xnext, x),
        JacobianBuffer(f_at_x_d!, xnext, u),
        JacobianBuffer(f_at_x_u!, xnext, d),
        JacobianBuffer(h_at_d!, y, x),
        JacobianBuffer(h_at_x!, y, d)
    )
end

"Get the linearization functions for [`NonLinModel`](@ref) `f!` and `h!` functions."
function get_linearize_funcs(NT, f!, h!, nu, nx, ny, nd, p)
    x = Vector{NT}(undef, nx)
    u = Vector{NT}(undef, nu)
    d = Vector{NT}(undef, nd)
    f_at_u_d!(xnext, x) = f!(xnext, x, u, d, p)
    f_at_x_d!(xnext, u) = f!(xnext, x, u, d, p)
    f_at_x_u!(xnext, d) = f!(xnext, x, u, d, p)
    h_at_d!(y, x)       = h!(y, x, d, p)
    h_at_x!(y, d)       = h!(y, x, d, p)
    return x, u, d, f_at_u_d!, f_at_x_d!, f_at_x_u!, h_at_d!, h_at_x!
end

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
    xnext0::Vector{NT}, y0::Vector{NT} = linmodel.buffer.x, linmodel.buffer.y
    get_jacobians!(linmodel, xnext0, y0, nonlinmodel, x0, u0, d0)
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

"Compute the 5 Jacobians of `model` at the linearization point and write them in `linmodel`."
function get_jacobians!(linmodel::LinModel, xnext0, y0, model::SimModel, x0, u0, d0)
    linbuffer = model.linbuffer # a LinearizationBuffer object
    linbuffer.x .= x0
    linbuffer.u .= u0
    linbuffer.d .= d0
    jacobian!(linmodel.A,  linbuffer.buffer_f_at_u_d, xnext0, x0)
    jacobian!(linmodel.Bu, linbuffer.buffer_f_at_x_d, xnext0, u0)
    jacobian!(linmodel.Bd, linbuffer.buffer_f_at_x_u, xnext0, d0)
    jacobian!(linmodel.C,  linbuffer.buffer_h_at_d, y0, x0)
    jacobian!(linmodel.Dd, linbuffer.buffer_h_at_x, y0, d0)
    return nothing
end

"Copy the state-space matrices of `model` to `linmodel` if `model` is already linear."
function get_jacobians!(linmodel::LinModel, _ , _ , model::LinModel, _ , _ , _)
    linmodel.A  .= model.A
    linmodel.Bu .= model.Bu
    linmodel.C  .= model.C
    linmodel.Bd .= model.Bd
    linmodel.Dd .= model.Dd
    return nothing
end