
@doc raw"""
    LinModel(model::NonLinModel; x=model.x, u=model.uop, d=model.dop)

Linearize `model` around the operating points `x`, `u` and `d`.

The arguments `x`, `u` and `d` are the linearization points for the state ``\mathbf{x}``,
manipulated input ``\mathbf{u}`` and measured disturbance ``\mathbf{d}``, respectively. The
Jacobians of ``\mathbf{f}`` and ``\mathbf{h}`` functions are automatically computed with
[`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl).

## Examples
```jldoctest
julia> model = NonLinModel((x,u,_)->x.^3+u, (x,_)->x, 0.1, 1, 1, 1);

julia> linmodel = LinModel(model, x=[1.0]); linmodel.A
1×1 Matrix{Float64}:
 3.0
```
"""
function LinModel(model::NonLinModel; x=model.x, u=model.uop, d=model.dop)
    nu, nx, ny, nd = model.nu, model.nx, model.ny, model.nd
    u0, d0 = u - model.uop, d - model.dop
    y  = model.h(x, d0) + model.yop
    A  = ForwardDiff.jacobian(x  -> model.f(x, u0, d0), x)
    Bu = ForwardDiff.jacobian(u0 -> model.f(x, u0, d0), u0)
    Bd = ForwardDiff.jacobian(d0 -> model.f(x, u0, d0), d0)
    C  = ForwardDiff.jacobian(x  -> model.h(x, d0), x)
    Dd = ForwardDiff.jacobian(d0 -> model.h(x, d0), d0)
    linmodel = LinModel(A, Bu, C, Bd, Dd, model.Ts, nu, nx, ny, nd)
    setop!(linmodel, uop=u, yop=y, dop=d)
    return linmodel
end