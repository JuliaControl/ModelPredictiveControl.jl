"Abstract supertype of all differential equation solvers."
abstract type DiffSolver end

"Empty solver for nonlinear discrete-time models."
struct EmptySolver <: DiffSolver end
get_solver_functions(::DataType, ::EmptySolver, f!, h!, _ ... ) = f!, h!


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

function get_solver_functions(NT::DataType, ::RungeKutta, f!, h!, Ts, _ , nx, _ , _ )
    f! = let fc! = f!, Ts=Ts, nx=nx
        # k1::DiffCache{Vector{NT}, Vector{NT}} = DiffCache(zeros(NT, nx), Nc)
        k1 = zeros(NT, nx)
        k2 = zeros(NT, nx)
        k3 = zeros(NT, nx)
        k4 = zeros(NT, nx)
        f! = (xnext, x, u, d) -> begin
            xterm = xnext
            @. xterm = x
            fc!(k1, xterm, u, d)
            @. xterm = x + k1 * Ts/2
            fc!(k2, xterm, u, d)
            @. xterm = x + k2 * Ts/2
            fc!(k3, xterm, u, d)
            @. xterm = x + k3 * Ts
            fc!(k4, xterm, u, d)
            @. xnext = x + (k1 + 2k2 + 2k3 + k4)*Ts/6
            return nothing
        end
    end
    return f!, h!
end