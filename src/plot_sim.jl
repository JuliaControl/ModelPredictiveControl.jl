struct SimResult{NT<:Real, O<:Union{SimModel, StateEstimator, PredictiveController}}
    obj::O                 # simulated instance
    T_data ::Vector{NT}    # time in seconds
    Y_data ::Matrix{NT}    # plant outputs (both measured and unmeasured)
    Ry_data::Matrix{NT}    # output setpoints
    Ŷ_data ::Matrix{NT}    # estimated outputs
    U_data ::Matrix{NT}    # manipulated inputs
    Ud_data::Matrix{NT}    # manipulated inputs including load disturbances
    Ru_data::Matrix{NT}    # manipulated input setpoints
    D_data ::Matrix{NT}    # measured disturbances
    X_data ::Matrix{NT}    # plant states
    X̂_data ::Matrix{NT}    # estimated states
end

"""
    SimResult(
        obj::Union{SimModel, StateEstimator, PredictiveController}, 
        U_data, 
        Y_data, 
        D_data  = [];
        X_data  = nothing,
        X̂_data  = nothing, 
        Ry_data = nothing, 
        Ru_data = nothing,
        Ŷ_data  = nothing
    )

Manually construct a `SimResult` to quickly plot `obj` simulations.

Except for `obj`, all the arguments should be matrices of `N` columns, where `N` is the 
number of time steps. [`SimResult`](@ref) objects allow to quickly plot simulation results.
Simply call `plot` from [`Plots.jl`](https://github.com/JuliaPlots/Plots.jl) on them.

# Examples
```julia-repl
julia> plant = LinModel(tf(1, [1, 1]), 1.0); N = 5; U_data = fill(1.0, 1, N);

julia> Y_data = reduce(hcat, (updatestate!(plant, U_data[:, i]); plant()) for i=1:N)
1×5 Matrix{Float64}:
 0.632121  0.864665  0.950213  0.981684  0.993262

julia> res = SimResult(plant, U_data, Y_data)
Simulation results of LinModel with 5 time steps.

julia> using Plots; plot(res)

```
"""
function SimResult(
    obj::O, 
    U_data, 
    Y_data, 
    D_data  = zeros(NT, 0, size(U_data, 2));
    X_data  = nothing,
    X̂_data  = nothing, 
    Ry_data = nothing, 
    Ru_data = nothing,
    Ŷ_data  = nothing
) where {NT<:Real, O<:Union{SimModel{NT}, StateEstimator{NT}, PredictiveController{NT}}}
    model = get_model(obj)
    Ts, nu, ny, nx, nx̂ = model.Ts, model.nu, model.ny, model.nx, get_nx̂(obj)
    N = size(U_data, 2)
    T_data = collect(Ts*(0:N-1))
    isnothing(X_data)  && (X_data  = fill(NaN, nx, N))
    isnothing(X̂_data)  && (X̂_data  = fill(NaN, nx̂, N))
    isnothing(Ry_data) && (Ry_data = fill(NaN, ny, N))
    isnothing(Ru_data) && (Ru_data = fill(NaN, nu, N))
    isnothing(Ŷ_data)  && (Ŷ_data  = fill(NaN, ny, N))
    NU, NY, NX, NX̂ = size(U_data, 2), size(Y_data, 2), size(X_data, 2), size(X̂_data, 2)
    NRy, NRu = size(Ry_data, 2), size(Ru_data, 2)
    if !(NU == NY == NX == NX̂ == NRy == NRu)
        throw(ArgumentError("All arguments must have the same number of columns (time steps)"))
    end
    size(Y_data, 2) == N || error("Y_data must be of size ($ny, $N)")
    return SimResult{NT, O}(obj, T_data, Y_data, Ry_data, Ŷ_data, U_data, U_data, 
                     Ru_data, D_data, X_data, X̂_data)
end

get_model(model::SimModel) = model
get_model(estim::StateEstimator) = estim.model
get_model(mpc::PredictiveController) = mpc.estim.model

get_nx̂(model::SimModel) = model.nx
get_nx̂(estim::StateEstimator) = estim.nx̂
get_nx̂(mpc::PredictiveController) = mpc.estim.nx̂


function Base.show(io::IO, res::SimResult) 
    N = length(res.T_data)
    print(io, "Simulation results of $(typeof(res.obj).name.name) with $N time steps.")
end


@doc raw"""
    sim!(plant::SimModel, N::Int, u=plant.uop.+1, d=plant.dop; x_0=plant.xop) -> res

Open-loop simulation of `plant` for `N` time steps, default to unit bump test on all inputs.

The manipulated inputs ``\mathbf{u}`` and measured disturbances ``\mathbf{d}`` are held
constant at `u` and `d` values, respectively. The plant initial state ``\mathbf{x}(0)`` is
specified by `x_0` keyword arguments. The function returns [`SimResult`](@ref) instances 
that can be visualized by calling `plot` from [`Plots.jl`](https://github.com/JuliaPlots/Plots.jl) 
on them (see Examples below). Note that the method mutates `plant` internal states.

# Examples
```julia-repl
julia> plant = NonLinModel((x,u,d)->0.1x+u+d, (x,_)->2x, 10.0, 1, 1, 1, 1, solver=nothing);

julia> res = sim!(plant, 15, [0], [0], x_0=[1])
Simulation results of NonLinModel with 15 time steps.

julia> using Plots; plot(res, plotu=false, plotd=false, plotx=true)

```
"""
function sim!(
    plant::SimModel{NT},
    N::Int,
    u::Vector = plant.uop.+1,
    d::Vector = plant.dop;
    x_0 = plant.xop
) where {NT<:Real}
    T_data  = collect(plant.Ts*(0:(N-1)))
    Y_data  = Matrix{NT}(undef, plant.ny, N)
    U_data  = Matrix{NT}(undef, plant.nu, N)
    D_data  = Matrix{NT}(undef, plant.nd, N)
    X_data  = Matrix{NT}(undef, plant.nx, N)
    setstate!(plant, x_0)
    for i=1:N
        y = evaloutput(plant, d) 
        Y_data[:, i] .= y
        U_data[:, i] .= u
        D_data[:, i] .= d
        X_data[:, i] .= plant.x0 .+ plant.xop
        updatestate!(plant, u, d)
    end
    return SimResult(plant, T_data, Y_data, U_data, Y_data, 
                     U_data, U_data, U_data, D_data, X_data, X_data)
end

@doc raw"""
    sim!(
        estim::StateEstimator,
        N::Int,
        u = estim.model.uop .+ 1,
        d = estim.model.dop;
        <keyword arguments>
    ) -> res

Closed-loop simulation of `estim` estimator for `N` steps, default to input bumps.

See Arguments for the available options. The noises are provided as standard deviations σ
vectors. The simulated sensor and process noises of `plant` are specified by `y_noise` and
`x_noise` arguments, respectively.

# Arguments
- `estim::StateEstimator` : state estimator to simulate
- `N::Int` : simulation length in time steps
- `u = estim.model.uop .+ 1` : manipulated input ``\mathbf{u}`` value
- `d = estim.model.dop` : plant measured disturbance ``\mathbf{d}`` value
- `plant::SimModel = estim.model` : simulated plant model
- `u_step  = zeros(plant.nu)` : step load disturbance on plant inputs ``\mathbf{u}``
- `u_noise = zeros(plant.nu)` : gaussian load disturbance on plant inputs ``\mathbf{u}``
- `y_step  = zeros(plant.ny)` : step disturbance on plant outputs ``\mathbf{y}``
- `y_noise = zeros(plant.ny)` : additive gaussian noise on plant outputs ``\mathbf{y}``
- `d_step  = zeros(plant.nd)` : step on measured disturbances ``\mathbf{d}``
- `d_noise = zeros(plant.nd)` : additive gaussian noise on measured dist. ``\mathbf{d}``
- `x_noise = zeros(plant.nx)` : additive gaussian noise on plant states ``\mathbf{x}``
- `x_0 = plant.xop` : plant initial state ``\mathbf{x}(0)``
- `x̂_0 = nothing` : initial estimate ``\mathbf{x̂}(0)``, [`initstate!`](@ref) is used if `nothing`
- `lastu = plant.uop` : last plant input ``\mathbf{u}`` for ``\mathbf{x̂}`` initialization

# Examples
```julia-repl
julia> model = LinModel(tf(3, [30, 1]), 0.5);

julia> estim = KalmanFilter(model, σR=[0.5], σQ=[0.25], σQint_ym=[0.01], σPint_ym_0=[0.1]);

julia> res = sim!(estim, 50, [0], y_noise=[0.5], x_noise=[0.25], x_0=[-10], x̂_0=[0, 0])
Simulation results of KalmanFilter with 50 time steps.

julia> using Plots; plot(res, plotŷ=true, plotu=false, plotxwithx̂=true)

```
"""
function sim!(
    estim::StateEstimator, 
    N::Int,
    u::Vector = estim.model.uop .+ 1,
    d::Vector = estim.model.dop;
    kwargs...
)
    return sim_closedloop!(estim, estim, N, u, d; kwargs...)
end

@doc raw"""
    sim!(
        mpc::PredictiveController, 
        N::Int,
        ry = mpc.estim.model.yop .+ 1, 
        d  = mpc.estim.model.dop,
        ru = mpc.estim.model.uop;
        <keyword arguments>
    ) -> res

Closed-loop simulation of `mpc` controller for `N` steps, default to output setpoint bumps.

The output and manipulated input setpoints are held constant at `ry` and `ru`, respectively.
The keyword arguments are identical to [`sim!(::StateEstimator, ::Int)`](@ref).

# Examples
```julia-repl
julia> model = LinModel([tf(3, [30, 1]); tf(2, [5, 1])], 4);

julia> mpc = setconstraint!(LinMPC(model, Mwt=[0, 1], Nwt=[0.01], Hp=30), ymin=[0, -Inf]);

julia> res = sim!(mpc, 25, [0, 0], y_noise=[0.1], y_step=[-10, 0])
Simulation results of LinMPC with 25 time steps.

julia> using Plots; plot(res, plotry=true, plotŷ=true, plotymin=true, plotu=true)

```
"""
function sim!(
    mpc::PredictiveController, 
    N::Int,
    ry::Vector = mpc.estim.model.yop .+ 1,
    d ::Vector = mpc.estim.model.dop,
    ru::Vector = mpc.estim.model.uop;
    kwargs...
)
    return sim_closedloop!(mpc, mpc.estim, N, ry, d, ru; kwargs...)
end

"Quick simulation function for `StateEstimator` and `PredictiveController` instances."
function sim_closedloop!(
    est_mpc::Union{StateEstimator{NT}, PredictiveController{NT}}, 
    estim::StateEstimator,
    N::Int,
    u_ry::Vector,
    d   ::Vector,
    ru  ::Vector = estim.model.uop;
    plant::SimModel = estim.model,
    u_step ::Vector = zeros(NT, plant.nu),
    u_noise::Vector = zeros(NT, plant.nu),
    y_step ::Vector = zeros(NT, plant.ny),
    y_noise::Vector = zeros(NT, plant.ny),
    d_step ::Vector = zeros(NT, plant.nd),
    d_noise::Vector = zeros(NT, plant.nd),
    x_noise::Vector = zeros(NT, plant.nx),
    x_0 = plant.xop,
    x̂_0 = nothing,
    lastu = plant.uop,
) where {NT<:Real}
    model = estim.model
    model.Ts ≈ plant.Ts || error("Sampling time of controller/estimator ≠ plant.Ts")
    old_x0    = copy(plant.x0)
    T_data    = collect(plant.Ts*(0:(N-1)))
    Y_data    = Matrix{NT}(undef, plant.ny, N)
    Ŷ_data    = Matrix{NT}(undef, model.ny, N)
    U_Ry_data = Matrix{NT}(undef, length(u_ry), N)
    U_data    = Matrix{NT}(undef, plant.nu, N)
    Ud_data   = Matrix{NT}(undef, plant.nu, N)
    Ru_data   = Matrix{NT}(undef, plant.nu, N)
    D_data    = Matrix{NT}(undef, plant.nd, N)
    X_data    = Matrix{NT}(undef, plant.nx, N) 
    X̂_data    = Matrix{NT}(undef, estim.nx̂, N)
    setstate!(plant, x_0)
    lastd, lasty = d, evaloutput(plant, d)
    initstate!(est_mpc, lastu, lasty[estim.i_ym], lastd)
    isnothing(x̂_0) || setstate!(est_mpc, x̂_0)
    for i=1:N
        d = lastd + d_step + d_noise.*randn(plant.nd)
        y = evaloutput(plant, d) + y_step + y_noise.*randn(plant.ny)
        ym = y[estim.i_ym]
        u  = sim_getu!(est_mpc, u_ry, d, ru, ym)
        ud = u + u_step + u_noise.*randn(plant.nu)
        Y_data[:, i] .= y
        Ŷ_data[:, i] .= evalŷ(estim, ym, d)
        U_Ry_data[:, i] .= u_ry
        U_data[:, i]  .= u
        Ud_data[:, i] .= ud
        Ru_data[:, i] .= ru
        D_data[:, i]  .= d
        X_data[:, i]  .= plant.x0 .+ plant.xop
        X̂_data[:, i]  .= estim.x̂0 .+ estim.x̂op
        x = updatestate!(plant, ud, d); 
        x[:] += x_noise.*randn(plant.nx)
        updatestate!(est_mpc, u, ym, d)
    end
    res = SimResult(est_mpc, T_data, Y_data, U_Ry_data, Ŷ_data, 
                    U_data, Ud_data, Ru_data, D_data, X_data, X̂_data)
    plant.x0 .= old_x0
    return res
end

"Compute new `u` for predictive controller simulation."
function sim_getu!(mpc::PredictiveController, ry, d, ru, ym)
    return moveinput!(mpc, ry, d; R̂u=repeat(ru, mpc.Hp), ym)
end
"Keep manipulated input `u` unchanged for state estimator simulation."
sim_getu!(::StateEstimator, u, _ , _ , _ ) = u

"Plots.jl recipe for `SimResult` objects constructed with `SimModel` objects."
@recipe function plot(
    res::SimResult{<:Real, <:SimModel};
    plotu  = true,
    plotd  = true,
    plotx  = false,
)
    t   = res.T_data
    ny = size(res.Y_data, 1)
    nu = size(res.U_data, 1)
    nd = size(res.D_data, 1)
    nx = size(res.X_data, 1)
    layout_mat = [(ny, 1)]
    plotu && (layout_mat = [layout_mat (nu, 1)])
    (plotd && nd ≠ 0) && (layout_mat = [layout_mat (nd, 1)])
    plotx && (layout_mat = [layout_mat (nx, 1)])

    layout := layout_mat
    # --- outputs y ---
    subplot_base = 0
    for i in 1:ny
        @series begin
            i == ny && (xguide --> "Time (s)")
            yguide  --> "\$y_$i\$"
            color   --> 1
            subplot --> subplot_base + i
            label   --> "\$\\mathbf{y}\$"
            legend  --> false
            t, res.Y_data[i, :]
        end
    end
    subplot_base += ny
    # --- manipulated inputs u ---
    if plotu
        for i in 1:nu
            @series begin
                i == nu && (xguide --> "Time (s)")
                yguide     --> "\$u_$i\$"
                color      --> 1
                subplot    --> subplot_base + i
                seriestype --> :steppost
                label      --> "\$\\mathbf{u}\$"
                legend     --> false
                t, res.U_data[i, :]
            end
        end
        subplot_base += nu
    end
    # --- measured disturbances d ---
    if plotd
        for i in 1:nd
            @series begin
                i == nd && (xguide --> "Time (s)")
                yguide  --> "\$d_$i\$"
                color   --> 1
                subplot --> subplot_base + i
                label   --> "\$\\mathbf{d}\$"
                legend  --> false
                t, res.D_data[i, :]
            end
        end
        subplot_base += nd
    end
    # --- plant states x ---
    if plotx
        for i in 1:nx
            @series begin
                i == nx && (xguide --> "Time (s)")
                yguide     --> "\$x_$i\$"
                color      --> 1
                subplot    --> subplot_base + i
                label      --> "\$\\mathbf{x}\$"
                legend     --> false
                t, res.X_data[i, :]
            end
        end
    end
end

"Plots.jl recipe for `SimResult` objects constructed with `StateEstimator` objects."
@recipe function plot(
    res::SimResult{<:Real, <:StateEstimator};
    plotŷ           = true,
    plotu           = true,
    plotd           = true,
    plotx           = false,
    plotx̂           = false,
    plotxwithx̂      = false
)
    t   = res.T_data
    ny = size(res.Y_data, 1)
    nu = size(res.U_data, 1)
    nd = size(res.D_data, 1)
    nx = size(res.X_data, 1)
    nx̂ = size(res.X̂_data, 1)
    layout_mat = [(ny, 1)]
    plotu && (layout_mat = [layout_mat (nu, 1)])
    (plotd && nd ≠ 0) && (layout_mat = [layout_mat (nd, 1)])
    (plotx && !plotxwithx̂) && (layout_mat = [layout_mat (nx, 1)])
    (plotx̂ ||  plotxwithx̂) && (layout_mat = [layout_mat (nx̂, 1)])
    layout := layout_mat
    # --- outputs y ---
    subplot_base = 0
    for i in 1:ny
        @series begin
            i == ny && (xguide --> "Time (s)")
            yguide  --> "\$y_$i\$"
            color   --> 1
            subplot --> subplot_base + i
            label   --> "\$\\mathbf{y}\$"
            legend  --> false
            t, res.Y_data[i, :]
        end
        if plotŷ
            @series begin
                i == ny && (xguide --> "Time (s)")
                yguide    --> "\$y_$i\$"
                color     --> 2
                subplot   --> subplot_base + i
                linestyle --> :dashdot
                linewidth --> 0.75
                label     --> "\$\\mathbf{\\hat{y}}\$"
                legend    --> true
                t, res.Ŷ_data[i, :]
            end
        end
    end
    subplot_base += ny
    # --- manipulated inputs u ---
    if plotu
        for i in 1:nu
            @series begin
                i == nu && (xguide --> "Time (s)")
                yguide     --> "\$u_$i\$"
                color      --> 1
                subplot    --> subplot_base + i
                seriestype --> :steppost
                label      --> "\$\\mathbf{u}\$"
                legend     --> false
                t, res.U_data[i, :]
            end
        end
        subplot_base += nu
    end
    # --- measured disturbances d ---
    if plotd
        for i in 1:nd
            @series begin
                i == nd && (xguide --> "Time (s)")
                yguide  --> "\$d_$i\$"
                color   --> 1
                subplot --> subplot_base + i
                label   --> "\$\\mathbf{d}\$"
                legend  --> false
                t, res.D_data[i, :]
            end
        end
        subplot_base += nd
    end
    # --- plant states x ---
    if plotx || plotxwithx̂
        for i in 1:nx
            @series begin
                i == nx && !plotxwithx̂ && (xguide --> "Time (s)")
                yguide     --> "\$x_$i\$"
                color      --> 1
                subplot    --> subplot_base + i
                label      --> "\$\\mathbf{x}\$"
                legend     --> false
                t, res.X_data[i, :]
            end
        end
        !plotxwithx̂ && (subplot_base += nx)
    end
    # --- estimated states x̂ ---
    if plotx̂ || plotxwithx̂
        for i in 1:nx̂
            @series begin
                i == nx̂ && (xguide --> "Time (s)")
                withPlantState = plotxwithx̂ && i ≤ nx
                yguide     --> (withPlantState ? "\$x_$i\$" : "\$\\hat{x}_$i\$")
                color      --> 2
                subplot    --> subplot_base + i
                linestyle --> :dashdot
                linewidth --> 0.75
                label      --> "\$\\mathbf{\\hat{x}}\$"
                legend     --> (withPlantState ? true : false)
                t, res.X̂_data[i, :]
            end
        end
    end
end

"Plots.jl recipe for `SimResult` objects constructed with `PredictiveController` objects."
@recipe function plot(
    res::SimResult{<:Real, <:PredictiveController}; 
    plotry      = true,
    plotymin    = true,
    plotymax    = true,
    plotŷ       = false,
    plotu       = true,
    plotru      = true,
    plotumin    = true,
    plotumax    = true,
    plotd       = true,
    plotx       = false,
    plotx̂       = false,
    plotxwithx̂  = false
)
    mpc = res.obj
    t  = res.T_data
    ny = size(res.Y_data, 1)
    nu = size(res.U_data, 1)
    nd = size(res.D_data, 1)
    nx = size(res.X_data, 1)
    nx̂ = size(res.X̂_data, 1)
    layout_mat = [(ny, 1)]
    plotu && (layout_mat = [layout_mat (nu, 1)])
    (plotd && nd ≠ 0) && (layout_mat = [layout_mat (nd, 1)])
    (plotx && !plotxwithx̂) && (layout_mat = [layout_mat (nx, 1)])
    (plotx̂ ||  plotxwithx̂) && (layout_mat = [layout_mat (nx̂, 1)])
    layout := layout_mat
    # --- constraints ---
    Umin, Umax = getUcon(mpc, nu)
    Ymin, Ymax = getYcon(mpc, ny)
    # --- outputs y ---
    subplot_base = 0
    for i in 1:ny
        @series begin
            i == ny && (xguide --> "Time (s)")
            yguide  --> "\$y_$i\$"
            color   --> 1
            subplot --> subplot_base + i
            label   --> "\$\\mathbf{y}\$"
            legend  --> false
            t, res.Y_data[i, :]
        end
        if plotŷ
            @series begin
                i == ny && (xguide --> "Time (s)")
                yguide    --> "\$y_$i\$"
                color     --> 2
                subplot   --> subplot_base + i
                linestyle --> :dashdot
                linewidth --> 0.75
                label     --> "\$\\mathbf{\\hat{y}}\$"
                legend    --> true
                t, res.Ŷ_data[i, :]
            end
        end
        if plotry && !iszero(mpc.M_Hp[i, i])
            @series begin
                i == ny && (xguide --> "Time (s)")
                yguide    --> "\$y_$i\$"
                color     --> 3
                subplot   --> subplot_base + i
                linestyle --> :dash
                linewidth --> 0.75
                label     --> "\$\\mathbf{r_y}\$"
                legend    --> true
                t, res.Ry_data[i, :]
            end
        end
        if plotymin && !isinf(Ymin[i])
            @series begin
                i == ny && (xguide --> "Time (s)")
                yguide    --> "\$y_$i\$"
                color     --> 4
                subplot   --> subplot_base + i
                linestyle --> :dot
                linewidth --> 1.5
                label     --> "\$\\mathbf{y_{min}}\$"
                legend    --> true
                t, fill(Ymin[i], length(t))
            end
        end
        if plotymax && !isinf(Ymax[i])
            @series begin
                i == ny && (xguide --> "Time (s)")
                yguide    --> "\$y_$i\$"
                color     --> 5
                subplot   --> subplot_base + i
                linestyle --> :dot
                linewidth --> 1.5
                label     --> "\$\\mathbf{y_{max}}\$"
                legend    --> true
                t, fill(Ymax[i], length(t))
            end
        end
    end
    subplot_base += ny
    # --- manipulated inputs u ---
    if plotu
        for i in 1:nu
            @series begin
                i == nu && (xguide --> "Time (s)")
                yguide     --> "\$u_$i\$"
                color      --> 1
                subplot    --> subplot_base + i
                seriestype --> :steppost
                label      --> "\$\\mathbf{u}\$"
                legend     --> false
                t, res.U_data[i, :]
            end
            if plotru && !iszero(mpc.L_Hp[i, i])
                @series begin
                    i == nu && (xguide --> "Time (s)")
                    yguide    --> "\$u_$i\$"
                    color     --> 3
                    subplot   --> subplot_base + i
                    seriestype --> :steppost
                    linestyle --> :dash
                    label     --> "\$\\mathbf{r_{u}}\$"
                    legend    --> true
                    t, res.Ru_data[i, :]
                end
            end
            if plotumin && !isinf(Umin[i])
                @series begin
                    i == nu && (xguide --> "Time (s)")
                    yguide    --> "\$u_$i\$"
                    color     --> 4
                    subplot   --> subplot_base + i
                    linestyle --> :dot
                    linewidth --> 1.5
                    label     --> "\$\\mathbf{u_{min}}\$"
                    legend    --> true
                    t, fill(Umin[i], length(t))
                end
            end
            if plotumax && !isinf(Umax[i])
                @series begin
                    i == nu && (xguide --> "Time (s)")
                    yguide    --> "\$u_$i\$"
                    color     --> 5
                    subplot   --> subplot_base + i
                    linestyle --> :dot
                    linewidth --> 1.5
                    label     --> "\$\\mathbf{u_{max}}\$"
                    legend    --> true
                    t, fill(Umax[i], length(t))
                end
            end
        end
        subplot_base += nu
    end
    # --- measured disturbances d ---
    if plotd
        for i in 1:nd
            @series begin
                i == nd && (xguide --> "Time (s)")
                yguide  --> "\$d_$i\$"
                color   --> 1
                subplot --> subplot_base + i
                label   --> "\$\\mathbf{d}\$"
                legend  --> false
                t, res.D_data[i, :]
            end
        end
        subplot_base += nd
    end
    # --- plant states x ---
    if plotx || plotxwithx̂
        for i in 1:nx
            @series begin
                i == nx && !plotxwithx̂ && (xguide --> "Time (s)")
                yguide     --> "\$x_$i\$"
                color      --> 1
                subplot    --> subplot_base + i
                label      --> "\$\\mathbf{x}\$"
                legend     --> false
                t, res.X_data[i, :]
            end
        end
        !plotxwithx̂ && (subplot_base += nx)
    end
    # --- estimated states x̂ ---
    if plotx̂ || plotxwithx̂
        for i in 1:nx̂
            @series begin
                i == nx̂ && (xguide --> "Time (s)")
                withPlantState = plotxwithx̂ && i ≤ nx
                yguide     --> (withPlantState ? "\$x_$i\$" : "\$\\hat{x}_$i\$")
                color      --> 2
                subplot    --> subplot_base + i
                linestyle --> :dashdot
                linewidth --> 0.75
                label      --> "\$\\mathbf{\\hat{x}}\$"
                legend     --> (withPlantState ? true : false)
                t, res.X̂_data[i, :]
            end
        end
    end
end

getUcon(mpc::PredictiveController, _ ) = mpc.con.U0min+mpc.Uop, mpc.con.U0max+mpc.Uop
getUcon(mpc::ExplicitMPC, nu) = fill(-Inf, mpc.Hp*nu), fill(+Inf, mpc.Hp*nu)

getYcon(mpc::PredictiveController, _ ) = mpc.con.Y0min+mpc.Yop, mpc.con.Y0max+mpc.Yop
getYcon(mpc::ExplicitMPC, ny) = fill(-Inf, mpc.Hp*ny), fill(+Inf, mpc.Hp*ny)