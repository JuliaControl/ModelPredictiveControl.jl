"Includes all signals of [`sim!`](@ref), view them with `plot` on `SimResult` instances."
struct SimResult{O<:Union{SimModel, StateEstimator, PredictiveController}}
    T_data ::Vector{Float64}    # time in seconds
    Y_data ::Matrix{Float64}    # plant outputs (both measured and unmeasured)
    Ry_data::Matrix{Float64}    # output setpoints
    Ŷ_data ::Matrix{Float64}    # estimated outputs
    U_data ::Matrix{Float64}    # manipulated inputs
    Ud_data::Matrix{Float64}    # manipulated inputs including load disturbances
    D_data ::Matrix{Float64}    # measured disturbances
    X_data ::Matrix{Float64}    # plant states
    X̂_data ::Matrix{Float64}    # estimated states
    plantIsModel::Bool          # true if simulated plant is identical to estimation model
    obj::O                      # simulated instance
end

@doc raw"""
    sim!(plant::SimModel, N::Int, u=plant.uop.+1, d=plant.dop; x0=zeros(plant.nx))

Open-loop simulation of `plant` for `N` time steps, default to unit bump test on all inputs.

The manipulated inputs ``\mathbf{u}`` and measured disturbances ``\mathbf{d}`` are held
constant at `u` and `d` values, respectively. The plant initial state ``\mathbf{x}(0)`` is
specified by `x0` keyword arguments. The function returns `SimResult` instances that can be
visualized by calling `plot` from [`Plots.jl`](https://github.com/JuliaPlots/Plots.jl) on 
them (see Examples below).

# Examples
```julia-repl
julia> plant = NonLinModel((x,u,d)->0.1x+u+d, (x,_)->2x, 10.0, 1, 1, 1, 1);

julia> res = sim!(plant, 15, [0], [0], x0=[1]);

julia> using Plots; plot(res, plotu=false, plotd=false, plotx=true)

```
"""
function sim!(
    plant::SimModel,
    N::Int,
    u::Vector = plant.uop.+1,
    d::Vector = plant.dop;
    x0 = zeros(plant.nx)
)
    T_data  = collect(plant.Ts*(0:(N-1)))
    Y_data  = Matrix{Float64}(undef, plant.ny, N)
    U_data  = Matrix{Float64}(undef, plant.nu, N)
    D_data  = Matrix{Float64}(undef, plant.nd, N)
    X_data  = Matrix{Float64}(undef, plant.nx, N)
    setstate!(plant, x0)
    for i=1:N
        y = evaloutput(plant, d) 
        Y_data[:, i]  = y
        U_data[:, i]  = u
        D_data[:, i]  = d
        X_data[:, i]  = plant.x
        updatestate!(plant, u, d); 
    end
    return SimResult(
        T_data, Y_data, U_data, Y_data, U_data, U_data, D_data, X_data, X_data, true, plant
    )
end

@doc raw"""
    sim!(
        estim::StateEstimator,
        N::Int,
        u = estim.model.uop .+ 1,
        d = estim.model.dop;
        <keyword arguments>
    )

Closed-loop simulation of `estim` estimator for `N` time steps, default to input bumps.

See Arguments for the available options. The noises are provided as standard deviations σ
vectors. The simulated sensor and process noises of `plant` are specified by `y_noise` and
`x_noise` arguments, respectively.

# Arguments
- `estim::StateEstimator` : state estimator to simulate.
- `N::Int` : simulation length in time steps.
- `u = estim.model.uop .+ 1` : manipulated input ``\mathbf{u}`` value.
- `d = estim.model.dop` : plant measured disturbance ``\mathbf{d}`` value.
- `plant::SimModel = estim.model` : simulated plant model.
- `u_step  = zeros(plant.nu)` : step load disturbance on plant inputs ``\mathbf{u}``.
- `u_noise = zeros(plant.nu)` : gaussian load disturbance on plant inputs ``\mathbf{u}``.
- `y_step  = zeros(plant.ny)` : step disturbance on plant outputs ``\mathbf{y}``.
- `y_noise = zeros(plant.ny)` : additive gaussian noise on plant outputs ``\mathbf{y}``.
- `d_step  = zeros(plant.nd)` : step on measured disturbances ``\mathbf{d}``.
- `d_noise = zeros(plant.nd)` : additive gaussian noise on measured dist. ``\mathbf{d}``.
- `x_noise = zeros(plant.nx)` : additive gaussian noise on plant states ``\mathbf{x}``.
- `x0 = zeros(plant.nx)` : plant initial state ``\mathbf{x}(0)``.
- `x̂0 = nothing` : initial estimate ``\mathbf{x̂}(0)``, [`initstate!`](@ref) is used if `nothing`.
- `lastu = plant.uop` : last plant input ``\mathbf{u}`` for ``\mathbf{x̂}`` initialization.

# Examples
```julia-repl
julia> model = LinModel(tf(3, [30, 1]), 0.5);

julia> estim = KalmanFilter(model, σR=[0.5], σQ=[0.25], σQ_int=[0.01], σP0_int=[0.1]);

julia> res = sim!(estim, 50, [0], y_noise=[0.5], x_noise=[0.25], x0=[-10], x̂0=[0, 0]);

julia> using Plots; plot(res, plotŷ=true, plotu=false, plotx=true, plotx̂=true)

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
        d  = mpc.estim.model.dop; 
        <keyword arguments>
    )

Closed-loop simulation of `mpc` controller for `N` time steps, default to setpoint bumps.

The output setpoint ``\mathbf{r_y}`` is held constant at `r_y`. The keyword arguments are
identical to [`sim!(::StateEstimator, ::Int)`](@ref).

# Examples
```julia-repl
julia> model = LinModel([tf(3, [30, 1]); tf(2, [5, 1])], 4);

julia> mpc = setconstraint!(LinMPC(model, Mwt=[0, 1], Nwt=[0.01], Hp=30), ŷmin=[0, -Inf]);

julia> res = sim!(mpc, 25, [0, 0], y_noise=[0.1], y_step=[-10, 0]);

julia> using Plots; plot(res, plotry=true, plotŷ=true, plotŷmin=true, plotu=true)

```
"""
function sim!(
    mpc::PredictiveController, 
    N::Int,
    ry::Vector = mpc.estim.model.yop .+ 1,
    d ::Vector = mpc.estim.model.dop;
    kwargs...
)
    return sim_closedloop!(mpc, mpc.estim, N, ry, d; kwargs...)
end

"Quick simulation function for `StateEstimator` and `PredictiveController` instances."
function sim_closedloop!(
    est_mpc::Union{StateEstimator, PredictiveController}, 
    estim::StateEstimator,
    N::Int,
    u_ry::Vector,
    d::Vector;
    plant::SimModel = estim.model,
    u_step ::Vector = zeros(plant.nu),
    u_noise::Vector = zeros(plant.nu),
    y_step ::Vector = zeros(plant.ny),
    y_noise::Vector = zeros(plant.ny),
    d_step ::Vector = zeros(plant.nd),
    d_noise::Vector = zeros(plant.nd),
    x_noise::Vector = zeros(plant.nx),
    x0 = zeros(plant.nx),
    x̂0 = nothing,
    lastu = plant.uop,
)
    model = estim.model
    plantIsModel = (plant === model)
    model.Ts ≈ plant.Ts || error("Sampling time of controller/estimator ≠ plant.Ts")
    old_x0 = copy(plant.x)
    T_data  = collect(plant.Ts*(0:(N-1)))
    Y_data  = Matrix{Float64}(undef, plant.ny, N)
    Ŷ_data  = Matrix{Float64}(undef, model.ny, N)
    U_Ry_data = Matrix{Float64}(undef, length(u_ry), N)
    U_data  = Matrix{Float64}(undef, plant.nu, N)
    Ud_data = Matrix{Float64}(undef, plant.nu, N)
    D_data  = Matrix{Float64}(undef, plant.nd, N)
    X_data  = Matrix{Float64}(undef, plant.nx, N) 
    X̂_data  = Matrix{Float64}(undef, estim.nx̂, N)
    setstate!(plant, x0)
    lastd, lasty = d, evaloutput(plant, d)
    if isnothing(x̂0)
        initstate!(est_mpc, lastu, lasty[estim.i_ym], lastd)
    else
        setstate!(est_mpc, x̂0)
    end
    for i=1:N
        d = lastd + d_step + d_noise.*randn(plant.nd)
        y = evaloutput(plant, d) + y_step + y_noise.*randn(plant.ny)
        ym = y[estim.i_ym]
        u  = sim_getu!(est_mpc, u_ry, d, ym)
        ud = u + u_step + u_noise.*randn(plant.nu)
        Y_data[:, i]  = y
        Ŷ_data[:, i]  = evalŷ(estim, ym, d)
        U_Ry_data[:, i] = u_ry
        U_data[:, i]  = u
        Ud_data[:, i] = ud
        D_data[:, i]  = d
        X_data[:, i]  = plant.x
        X̂_data[:, i]  = estim.x̂
        x = updatestate!(plant, ud, d); 
        x[:] += x_noise.*randn(plant.nx)
        updatestate!(est_mpc, u, ym, d)
    end
    res = SimResult(
        T_data, Y_data, U_Ry_data, Ŷ_data, 
        U_data, Ud_data, D_data, X_data, X̂_data, 
        plantIsModel, est_mpc
    )
    setstate!(plant, old_x0)
    return res
end

"Keep manipulated input `u` unchanged for state estimator simulation."
sim_getu!(::StateEstimator, u, _ , _ ) = u
"Compute new `u` for predictive controller simulation."
sim_getu!(mpc::PredictiveController, ry, d, ym) = moveinput!(mpc, ry, d; ym)

"Plots.jl recipe for `SimResult` objects constructed with `SimModel` objects."
@recipe function simresultplot(
    res::SimResult{<:SimModel};
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
    xguide    --> "Time (s)"
    # --- outputs y ---
    subplot_base = 0
    for i in 1:ny
        @series begin
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
                xguide  --> "Time (s)"
                yguide  --> "\$d_$i\$"
                color   --> 1
                subplot --> subplot_base + i
                label   --> "\$\\mathbf{d}\$"
                legend  --> false
                t, res.D_data[i, :]
            end
        end
    end
    # --- plant states x ---
    if plotx
        for i in 1:nx
            @series begin
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
@recipe function simresultplot(
    res::SimResult{<:StateEstimator};
    plotŷ           = true,
    plotu           = true,
    plotd           = true,
    plotx           = false,
    plotx̂           = false
)
    t   = res.T_data
    plantIsModel = res.plantIsModel
    ny = size(res.Y_data, 1)
    nu = size(res.U_data, 1)
    nd = size(res.D_data, 1)
    nx = size(res.X_data, 1)
    nx̂ = size(res.X̂_data, 1)
    layout_mat = [(ny, 1)]
    plotu && (layout_mat = [layout_mat (nu, 1)])
    (plotd && nd ≠ 0) && (layout_mat = [layout_mat (nd, 1)])
    if plantIsModel && plotx && !plotx̂
        layout_mat = [layout_mat (nx, 1)]
    elseif plantIsModel && plotx̂
        layout_mat = [layout_mat (nx̂, 1)]
    elseif !plantIsModel
        plotx && (layout_mat = [layout_mat (nx, 1)])
        plotx̂ && (layout_mat = [layout_mat (nx̂, 1)])
    end
    layout := layout_mat
    xguide    --> "Time (s)"
    # --- outputs y ---
    subplot_base = 0
    for i in 1:ny
        @series begin
            yguide  --> "\$y_$i\$"
            color   --> 1
            subplot --> subplot_base + i
            label   --> "\$\\mathbf{y}\$"
            legend  --> false
            t, res.Y_data[i, :]
        end
        if plotŷ
            @series begin
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
                xguide  --> "Time (s)"
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
                yguide     --> "\$x_$i\$"
                color      --> 1
                subplot    --> subplot_base + i
                label      --> "\$\\mathbf{x}\$"
                legend     --> false
                t, res.X_data[i, :]
            end
        end
        !plantIsModel && (subplot_base += nx)
    end
    # --- estimated states x̂ ---
    if plotx̂
        for i in 1:nx̂
            @series begin
                withPlantState = plantIsModel && plotx && i ≤ nx
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
@recipe function simresultplot(
    res::SimResult{<:PredictiveController}; 
    plotry   = true,
    plotŷmin = true,
    plotŷmax = true,
    plotŷ    = false,
    plotu    = true,
    plotru   = true,
    plotumin = true,
    plotumax = true,
    plotd    = true,
    plotx    = false,
    plotx̂    = false
)
    mpc = res.obj
    t  = res.T_data
    plantIsModel = res.plantIsModel
    ny = size(res.Y_data, 1)
    nu = size(res.U_data, 1)
    nd = size(res.D_data, 1)
    nx = size(res.X_data, 1)
    nx̂ = size(res.X̂_data, 1)
    layout_mat = [(ny, 1)]
    plotu && (layout_mat = [layout_mat (nu, 1)])
    (plotd && nd ≠ 0) && (layout_mat = [layout_mat (nd, 1)])
    if plantIsModel && plotx && !plotx̂
        layout_mat = [layout_mat (nx, 1)]
    elseif plantIsModel && plotx̂
        layout_mat = [layout_mat (nx̂, 1)]
    elseif !plantIsModel
        plotx && (layout_mat = [layout_mat (nx, 1)])
        plotx̂ && (layout_mat = [layout_mat (nx̂, 1)])
    end
    layout := layout_mat
    xguide --> "Time (s)"
    # --- outputs y ---
    subplot_base = 0
    for i in 1:ny
        @series begin
            yguide  --> "\$y_$i\$"
            color   --> 1
            subplot --> subplot_base + i
            label   --> "\$\\mathbf{y}\$"
            legend  --> false
            t, res.Y_data[i, :]
        end
        if plotŷ
            @series begin
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
        if plotŷmin && !isinf(mpc.con.Ŷmin[i])
            @series begin
                yguide    --> "\$y_$i\$"
                color     --> 4
                subplot   --> subplot_base + i
                linestyle --> :dot
                linewidth --> 1.5
                label     --> "\$\\mathbf{\\hat{y}_{min}}\$"
                legend    --> true
                t, fill(mpc.con.Ŷmin[i], length(t))
            end
        end
        if plotŷmax && !isinf(mpc.con.Ŷmax[i])
            @series begin
                yguide    --> "\$y_$i\$"
                color     --> 5
                subplot   --> subplot_base + i
                linestyle --> :dot
                linewidth --> 1.5
                label     --> "\$\\mathbf{\\hat{y}_{max}}\$"
                legend    --> true
                t, fill(mpc.con.Ŷmax[i], length(t))
            end
        end
    end
    subplot_base += ny
    # --- manipulated inputs u ---
    if plotu
        for i in 1:nu
            @series begin
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
                    yguide    --> "\$u_$i\$"
                    color     --> 3
                    subplot   --> subplot_base + i
                    seriestype --> :steppost
                    linestyle --> :dash
                    label     --> "\$\\mathbf{r_{u}}\$"
                    legend    --> true
                    t, res.Ry_data[i, :]
                end
            end
            if plotumin && !isinf(mpc.con.Umin[i])
                @series begin
                    yguide    --> "\$u_$i\$"
                    color     --> 4
                    subplot   --> subplot_base + i
                    linestyle --> :dot
                    linewidth --> 1.5
                    label     --> "\$\\mathbf{u_{min}}\$"
                    legend    --> true
                    t, fill(mpc.con.Umin[i], length(t))
                end
            end
            if plotumax && !isinf(mpc.con.Umax[i])
                @series begin
                    yguide    --> "\$u_$i\$"
                    color     --> 5
                    subplot   --> subplot_base + i
                    linestyle --> :dot
                    linewidth --> 1.5
                    label     --> "\$\\mathbf{u_{max}}\$"
                    legend    --> true
                    t, fill(mpc.con.Umax[i], length(t))
                end
            end
        end
        subplot_base += nu
    end
    # --- measured disturbances d ---
    if plotd
        for i in 1:nd
            @series begin
                xguide  --> "Time (s)"
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
                yguide     --> "\$x_$i\$"
                color      --> 1
                subplot    --> subplot_base + i
                label      --> "\$\\mathbf{x}\$"
                legend     --> false
                t, res.X_data[i, :]
            end
        end
        !plantIsModel && (subplot_base += nx)
    end
    # --- estimated states x̂ ---
    if plotx̂
        for i in 1:nx̂
            @series begin
                withPlantState = plantIsModel && plotx && i ≤ nx
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