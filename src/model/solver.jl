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
        if order ≠ 4
            throw(ArgumentError("only 4th order Runge-Kutta is supported."))
        end
        if supersample < 1
            throw(ArgumentError("supersample must be greater than 0"))
        end
        return new(order, supersample)
    end
end

"""
    RungeKutta(order::Int=4; supersample::Int=1)

Create a Runge-Kutta solver with optional super-sampling.

Only the fourth order Runge-Kutta is supported for now. The keyword argument `supersample`
provides the number of internal steps (default to 1 step).
"""
function RungeKutta(order::Int=4; supersample::Int=1)
    if order < 1
        throw(ArgumentError("order must be greater than 0"))
    end
    if supersample < 1
        throw(ArgumentError("supersample must be greater than 0"))
    end
    return RungeKutta(order, supersample)
end

"Get the `f!` and `h!` functions for Runge-Kutta solver."
function get_solver_functions(NT::DataType, solver::RungeKutta, fc!, hc!, Ts,_ , nx, _, _)
    Ts_inner = Ts/solver.supersample
    Nc = nx + 1
    xcur_cache::DiffCache{Vector{NT}, Vector{NT}} = DiffCache(zeros(NT, nx), Nc)
    k1_cache::DiffCache{Vector{NT}, Vector{NT}}   = DiffCache(zeros(NT, nx), Nc)
    k2_cache::DiffCache{Vector{NT}, Vector{NT}}   = DiffCache(zeros(NT, nx), Nc)
    k3_cache::DiffCache{Vector{NT}, Vector{NT}}   = DiffCache(zeros(NT, nx), Nc)
    k4_cache::DiffCache{Vector{NT}, Vector{NT}}   = DiffCache(zeros(NT, nx), Nc)
    f! = function inner_solver(xnext, x, u, d)
        T = promote_type(eltype(x), eltype(u), eltype(d))
        xcur = get_tmp(xcur_cache, T)
        k1   = get_tmp(k1_cache, T)
        k2   = get_tmp(k2_cache, T)
        k3   = get_tmp(k3_cache, T)
        k4   = get_tmp(k4_cache, T)
        @. xcur = x
        for i=1:solver.supersample
            xterm = xnext
            @. xterm = xcur
            fc!(k1, xterm, u, d)
            @. xterm = xcur + k1 * Ts_inner/2
            fc!(k2, xterm, u, d)
            @. xterm = xcur + k2 * Ts_inner/2
            fc!(k3, xterm, u, d)
            @. xterm = xcur + k3 * Ts_inner
            fc!(k4, xterm, u, d)
            @. xcur = xcur + (k1 + 2k2 + 2k3 + k4)*Ts_inner/6
        end
        @. xnext = xcur
        return nothing
    end
    h! = hc!
    return f!, h!
end

function Base.show(io::IO, solver::RungeKutta)
    N, n = solver.order, solver.supersample
    print(io, "$(N)th order Runge-Kutta differential equation solver with $n supersamples.")
end