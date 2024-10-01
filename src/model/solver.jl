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

Create a Runge-Kutta solver with optional super-sampling.

Only the 4th order Runge-Kutta is supported for now. The keyword argument `supersample`
provides the number of internal steps (default to 1 step).
"""
RungeKutta(order::Int=4; supersample::Int=1) = RungeKutta(order, supersample)

"Get the `f!` and `h!` functions for Runge-Kutta solver."
function get_solver_functions(NT::DataType, solver::RungeKutta, fc!, hc!, Ts, _ , nx, _ , _ )
    Ts_inner = Ts/solver.supersample
    Nc = nx + 2
    xcur_cache::DiffCache{Vector{NT}, Vector{NT}} = DiffCache(zeros(NT, nx), Nc)
    k1_cache::DiffCache{Vector{NT}, Vector{NT}}   = DiffCache(zeros(NT, nx), Nc)
    k2_cache::DiffCache{Vector{NT}, Vector{NT}}   = DiffCache(zeros(NT, nx), Nc)
    k3_cache::DiffCache{Vector{NT}, Vector{NT}}   = DiffCache(zeros(NT, nx), Nc)
    k4_cache::DiffCache{Vector{NT}, Vector{NT}}   = DiffCache(zeros(NT, nx), Nc)
    f! = function inner_solver_f!(xnext, x, u, d, p)
        CT = promote_type(eltype(x), eltype(u), eltype(d))
        # dummy variable for get_tmp, necessary for PreallocationTools + Julia 1.6 :
        var::CT = 0
        xcur = get_tmp(xcur_cache, var)
        k1   = get_tmp(k1_cache, var)
        k2   = get_tmp(k2_cache, var)
        k3   = get_tmp(k3_cache, var)
        k4   = get_tmp(k4_cache, var)
        xterm = xnext
        @. xcur = x
        for i=1:solver.supersample
            @. xterm = xcur
            fc!(k1, xterm, u, d, p)
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
    h! = hc!
    return f!, h!
end

function Base.show(io::IO, solver::RungeKutta)
    N, n = solver.order, solver.supersample
    print(io, "$(N)th order Runge-Kutta differential equation solver with $n supersamples.")
end