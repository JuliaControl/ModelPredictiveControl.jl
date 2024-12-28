"Abstract supertype of all differential equation solvers."
abstract type DiffSolver end

"Empty solver for nonlinear discrete-time models."
struct EmptySolver <: DiffSolver end
get_solver_functions(::DataType, ::EmptySolver, f!, h!, _ ... ) = f!, h!

function Base.show(io::IO, solver::EmptySolver)
    print(io, "Empty differential equation solver.")
end

struct RungeKutta <: DiffSolver
    order::Int
    supersample::Int
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
        return new(order, supersample)
    end
end

"""
    RungeKutta(order=4; supersample=1)

Create an explicit Runge-Kutta solver with optional super-sampling.

Only the 1st and 4th order Runge-Kutta is supported. The keyword argument
`supersample` provides the number of internal steps (default to 1 step). This solver is
allocation-free if the `f!` and `h!` functions do not allocate.
"""
RungeKutta(order::Int=4; supersample::Int=1) = RungeKutta(order, supersample)

"Get the `f!` and `h!` functions for the explicit Runge-Kutta solvers."
function get_solver_functions(NT::DataType, solver::RungeKutta, fc!, hc!, Ts, _ , nx, _ , _ )
    order = solver.order
    Ts_inner = Ts/solver.supersample
    Nc = nx + 2
    xcur_cache::DiffCache{Vector{NT}, Vector{NT}} = DiffCache(zeros(NT, nx), Nc)
    k1_cache::DiffCache{Vector{NT}, Vector{NT}}   = DiffCache(zeros(NT, nx), Nc)
    k2_cache::DiffCache{Vector{NT}, Vector{NT}}   = DiffCache(zeros(NT, nx), Nc)
    k3_cache::DiffCache{Vector{NT}, Vector{NT}}   = DiffCache(zeros(NT, nx), Nc)
    k4_cache::DiffCache{Vector{NT}, Vector{NT}}   = DiffCache(zeros(NT, nx), Nc)
    if order==1
        f! = function euler_solver!(xnext, x, u, d, p)
            CT = promote_type(eltype(x), eltype(u), eltype(d))
            xcur = get_tmp(xcur_cache, CT)
            k1   = get_tmp(k1_cache, CT)
            xterm = xnext
            @. xcur = x
            for i=1:solver.supersample
                fc!(k1, xcur, u, d, p)
                @. xcur = xcur + k1 * Ts_inner
            end
            @. xnext = xcur
            return nothing
        end
    elseif order==4
        f! = function rk4_solver!(xnext, x, u, d, p)
            CT = promote_type(eltype(x), eltype(u), eltype(d))
            xcur = get_tmp(xcur_cache, CT)
            k1   = get_tmp(k1_cache, CT)
            k2   = get_tmp(k2_cache, CT)
            k3   = get_tmp(k3_cache, CT)
            k4   = get_tmp(k4_cache, CT)
            xterm = xnext
            @. xcur = x
            for i=1:solver.supersample
                fc!(k1, xcur, u, d, p)
                @. xterm = xcur + k1 * Ts_inner/2
                fc!(k2, xterm, u, d, p)
                @. xterm = xcur + k2 * Ts_inner/2
                fc!(k3, xterm, u, d, p)
                @. xterm = xcur + k3 * Ts_inner
                fc!(k4, xterm, u, d, p)
                @. xcur = xcur + (k1 + 2k2 + 2k3 + k4)*Ts_inner/6
            end
            @. xnext = xcur
            return nothing
        end
    end
    h! = hc!
    return f!, h!
end

"""
    ForwardEuler(; supersample=1)

Create a Forward Euler solver with optional super-sampling.

This is an alias for `RungeKutta(1; supersample)`.
"""
const ForwardEuler(;supersample=1) = RungeKutta(1; supersample)

function Base.show(io::IO, solver::RungeKutta)
    N, n = solver.order, solver.supersample
    print(io, "$(N)th order Runge-Kutta differential equation solver with $n supersamples.")
end