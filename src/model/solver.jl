"Abstract supertype of all differential equation solvers."
abstract type DiffSolver end

"Empty solver for nonlinear discrete-time models."
struct EmptySolver <: DiffSolver
    ns::Int             # number of stages
    EmptySolver() = new(0)
end

"""
    get_solver_functions(NT::DataType, solver::EmptySolver, f!, h!, Ts, nu, nx, ny, nd)

Get `solver_f!` and `solver_h!` functions for the `EmptySolver` (discrete models).

The functions should have the following signature:
```
    solver_f!(xnext, K, x, u, d, p) -> nothing
    solver_h!(y, x, d, p) -> nothing
```
in which `xnext`, `K` and `y` arguments should be mutated in-place. The `K` argument is 
a vector of `nx*(solver.ns+1)` elements to store the solver intermediary stage values,
and also the current state value when `supersample ≠ 1`.
"""
function get_solver_functions(::DataType, ::EmptySolver, f!, h!, _ , _ , _ , _ )
    solver_f!(xnext, _ , x, u, d, p) = f!(xnext, x, u, d, p)
    solver_h! = h!
    return solver_f!, solver_h!
end

function Base.show(io::IO, solver::EmptySolver)
    print(io, "Empty differential equation solver.")
end

struct RungeKutta <: DiffSolver
    ns::Int             # number of stages
    order::Int          # order of the method
    supersample::Int    # number of internal steps
    function RungeKutta(order::Int, supersample::Int)
        if order ≠ 4 && order ≠ 1
            throw(ArgumentError("only 1st and 4th order Runge-Kutta is supported."))
        end
        if order < 1
            throw(ArgumentError("order must be greater than 0"))
        end
        if supersample < 1
            throw(ArgumentError("supersample must be greater than 0"))
        end
        ns = order # only true for order ≤ 4 with RungeKutta
        return new(ns, order, supersample)
    end
end

"""
    RungeKutta(order=4; supersample=1)

Create an explicit Runge-Kutta solver with optional super-sampling.

The argument `order` specifies the order of the Runge-Kutta solver, which must be 1 or 4.
The keyword argument `supersample` provides the number of internal steps (default to 1 step).
This solver is allocation-free if the `f!` and `h!` functions do not allocate.
"""
RungeKutta(order::Int=4; supersample::Int=1) = RungeKutta(order, supersample)

"Get `solve_f!` and `solver_h!` functions for the explicit Runge-Kutta solvers."
function get_solver_functions(NT::DataType, solver::RungeKutta, f!, h!, Ts, _ , nx, _ , _ )
    Nc = nx + 2
    solver_f! = if solver.order==4
        get_rk4_function(NT, solver, f!, Ts, nx, Nc)
    elseif solver.order==1
        get_euler_function(NT, solver, f!, Ts, nx, Nc)
    else
        throw(ArgumentError("only 1st and 4th order Runge-Kutta is supported."))
    end
    solver_h! = h!
    return solver_f!, solver_h!
end

"Get the f! function for the 4th order explicit Runge-Kutta solver."
function get_rk4_function(NT, solver, f!, Ts, nx, Nc)
    Ts_inner = Ts/solver.supersample
    function rk4_solver_f!(xnext, K, x, u, d, p)
        xcurr = @views K[1:nx]
        k1 = @views K[(1nx + 1):(2nx)]
        k2 = @views K[(2nx + 1):(3nx)]
        k3 = @views K[(3nx + 1):(4nx)]
        k4 = @views K[(4nx + 1):(5nx)]   
        @. xcurr = x
        for i=1:solver.supersample
            f!(k1, xcurr, u, d, p)
            @. xnext = xcurr + k1 * Ts_inner/2
            f!(k2, xnext, u, d, p)
            @. xnext = xcurr + k2 * Ts_inner/2
            f!(k3, xnext, u, d, p)
            @. xnext = xcurr + k3 * Ts_inner
            f!(k4, xnext, u, d, p)
            @. xcurr = xcurr + (k1 + 2k2 + 2k3 + k4)*Ts_inner/6
        end
        @. xnext = xcurr
        return nothing
    end
    return rk4_solver_f!
end

"Get the f! function for the explicit Euler solver."
function get_euler_function(NT, solver, fc!, Ts, nx, Nc)
    Ts_inner = Ts/solver.supersample
    function euler_solver_f!(xnext, x, u, d, p)
        CT = promote_type(eltype(x), eltype(u), eltype(d))
        xcur = get_tmp(xcur_cache, CT)
        k    = get_tmp(k_cache, CT)
        xterm = xnext
        @. xcur = x
        for i=1:solver.supersample
            fc!(k, xcur, u, d, p)
            @. xcur = xcur + k * Ts_inner
        end
        @. xnext = xcur
        return nothing
    end
    return euler_solver_f!
end

"""
    ForwardEuler(; supersample=1)

Create a Forward Euler solver with optional super-sampling.

This is an alias for `RungeKutta(1; supersample)`, see [`RungeKutta`](@ref).
"""
const ForwardEuler(;supersample=1) = RungeKutta(1; supersample)

function Base.show(io::IO, solver::RungeKutta)
    N, n = solver.order, solver.supersample
    print(io, "$(N)th order Runge-Kutta differential equation solver with $n supersamples.")
end