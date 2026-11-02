struct RungeKutta{N} <: DiffSolver
    ni::Int             # number of intermediate stages
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
        ni = order # only true for order ≤ 4 with RungeKutta
        return new{order}(ni, supersample)
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

"Solve the differential equation with the 4th order Runge-Kutta method."
function solver_f!(
    xnext, k̄, ui, di, f!::F, Ts, solver::RungeKutta{4}, x, u, d, unext, dnext, p
) where F<:Function
    supersample = solver.supersample
    Ts_inner = Ts/supersample
    isFOH_u = isdifferent(u, unext)
    isFOH_d = isdifferent(d, dnext)
    nx = length(x)
    xcurr = @views k̄[1:nx]
    k1 = @views k̄[(1nx + 1):(2nx)]
    k2 = @views k̄[(2nx + 1):(3nx)]
    k3 = @views k̄[(3nx + 1):(4nx)]
    k4 = @views k̄[(4nx + 1):(5nx)]
    @. xcurr = x
    for i=1:supersample
        τ = (i-1.0)/supersample 
        u1 = isFOH_u ? @.(ui = (1-τ)*u + (τ)*unext) : u
        d1 = isFOH_d ? @.(di = (1-τ)*d + (τ)*dnext) : d
        f!(k1, xcurr, u1, d1, p)
        @. xnext = xcurr + k1 * Ts_inner/2
        τ = (i-0.5)/supersample 
        u2and3 = isFOH_u ? @.(ui = (1-τ)*u + (τ)*unext) : u
        d2and3 = isFOH_d ? @.(di = (1-τ)*d + (τ)*dnext) : d
        f!(k2, xnext, u2and3, d2and3, p)
        @. xnext = xcurr + k2 * Ts_inner/2
        f!(k3, xnext, u2and3, d2and3, p)
        @. xnext = xcurr + k3 * Ts_inner
        τ = i/supersample
        u4 = isFOH_u ? @.(ui = (1-τ)*u + (τ)*unext) : u
        d4 = isFOH_d ? @.(di = (1-τ)*d + (τ)*dnext) : d
        f!(k4, xnext, u4, d4, p)
        @. xcurr = xcurr + (k1 + 2k2 + 2k3 + k4)*Ts_inner/6
    end
    @. xnext = xcurr
    return nothing
end

"""
    ForwardEuler(; supersample=1)

Create a Forward Euler solver with optional super-sampling.

This is an alias for `RungeKutta(1; supersample)`, see [`RungeKutta`](@ref).
"""
const ForwardEuler(;supersample=1) = RungeKutta(1; supersample)


"Solve the differential equation with the forward Euler method."
function solver_f!(
    xnext, k̄, ui, di, f!::F, Ts, solver::RungeKutta{1}, x, u, d, unext, dnext, p
) where F<:Function
    supersample = solver.supersample
    Ts_inner = Ts/supersample
    isFOH_u = isdifferent(u, unext)
    isFOH_d = isdifferent(d, dnext)
    nx = length(x)
    xcurr = @views k̄[1:nx]
    k1 = @views k̄[(1nx + 1):(2nx)]
    @. xcurr = x
    for i=1:supersample
        τ = (i-1)/supersample 
        u1 = isFOH_u ? @.(ui = (1-τ)*u + (τ)*unext) : u
        d1 = isFOH_d ? @.(di = (1-τ)*d + (τ)*dnext) : d
        f!(k1, xcurr, u1, d1, p)
        @. xcurr = xcurr + k1 * Ts_inner
    end
    @. xnext = xcurr
    return nothing
end

function Base.show(io::IO, solver::RungeKutta{N}) where N
    n = solver.supersample
    print(io, "$(N)th order Runge-Kutta differential equation solver with $n supersamples.")
end