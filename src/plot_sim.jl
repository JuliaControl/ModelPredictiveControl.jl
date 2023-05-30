"Includes all signals of [`sim`](@ref), view them with `plot` on `SimResult` instances."
struct SimResult{O<:Union{SimModel, StateEstimator, PredictiveController}}
    obj    ::O
    T_data ::Vector{Float64}
    Y_data ::Matrix{Float64}
    Ry_data::Matrix{Float64}
    Ŷ_data ::Matrix{Float64}
    U_data ::Matrix{Float64}
    D_data ::Matrix{Float64}
    X_data ::Matrix{Float64}
    X̂_data ::Matrix{Float64}
end

@doc raw"""
    sim(
        estim::StateEstimator, 
        N::Int, 
        u = estim.model.uop .+ 1, 
        d = estim.model.dop; 
        <keyword arguments>
    )

Closed-loop simulation of `estim` estimator for `N` time steps, default to input bumps.

See Arguments for the option list. The noise arguments are in standard deviations σ. The 
sensor and process noises of the simulated plant are specified by `y_noise` and `x_noise` 
arguments, respectively.

# Arguments

- `estim::StateEstimator` : state estimator to simulate
- `N::Int` : simulation length in time steps
- `u = mpc.estim.model.uop .+ 1` : manipulated input ``\mathbf{u}`` value
- `d = mpc.estim.model.dop` : plant measured disturbance ``\mathbf{d}`` value
- `plant::SimModel = mpc.estim.model` : simulated plant model
- `u_step  = zeros(plant.nu)` : step disturbance on manipulated input ``\mathbf{u}``
- `u_noise = zeros(plant.nu)` : additive gaussian noise on manipulated input ``\mathbf{u}``
- `y_step  = zeros(plant.ny)` : step disturbance on plant outputs ``\mathbf{y}``
- `y_noise = zeros(plant.ny)` : additive gaussian noise on plant outputs ``\mathbf{y}``
- `d_step  = zeros(plant.nd)` : step disturbance on measured dist. ``\mathbf{d}``
- `d_noise = zeros(plant.nd)` : additive gaussian noise on measured dist. ``\mathbf{d}``
- `x_noise = zeros(plant.nx)` : additive gaussian noise on plant states ``\mathbf{x}``
- `x0 = zeros(plant.nx)` : plant initial state ``\mathbf{x}(0)``
- `x̂0 = nothing` : `mpc.estim` state estimator initial state ``\mathbf{x̂}(0)``, if `nothing`
   then ``\mathbf{x̂}`` is initialized with [`initstate!`](@ref)
- `lastu = plant.uop` : last plant input ``\mathbf{u}`` for ``\mathbf{x̂}`` initialization
"""
function sim(
    estim::StateEstimator, 
    N::Int,
    u::Vector{<:Real} = estim.model.uop .+ 1,
    d::Vector{<:Real} = estim.model.dop;
    kwargs...
)
    return sim_all(estim, estim, estim.model, N, u, d; kwargs...)
end



@doc raw"""
    sim(
        mpc::PredictiveController, 
        N::Int, 
        ry = mpc.estim.model.yop .+ 1, 
        d  = mpc.estim.model.dop; 
        <keyword arguments>
    )

Closed-loop simulation of `mpc` controller for `N` time steps, default to setpoint bumps.

`ry` is the output setpoint value applied at ``t = 0`` second. The keyword arguments are
identical to [`sim(::StateEstimator)`](@ref).

"""
function sim(
    mpc::PredictiveController, 
    N::Int,
    ry::Vector{<:Real} = mpc.estim.model.yop .+ 1,
    d ::Vector{<:Real} = mpc.estim.model.dop;
    kwargs...
)
    return sim_all(mpc, mpc.estim, mpc.estim.model, N, ry, d; kwargs...)
end


"Quick simulation function for `SimModel`, `StateEstimator` and `PredictiveController`."
function sim_all(
    estim_mpc::Union{StateEstimator, PredictiveController}, 
    estim::StateEstimator,
    model::SimModel, 
    N::Int,
    u_ry::Vector{<:Real},
    d::Vector{<:Real};
    plant::SimModel = model,
    u_step ::Vector{<:Real} = zeros(plant.nu),
    u_noise::Vector{<:Real} = zeros(plant.nu),
    y_step ::Vector{<:Real} = zeros(plant.ny),
    y_noise::Vector{<:Real} = zeros(plant.ny),
    d_step ::Vector{<:Real} = zeros(plant.nd),
    d_noise::Vector{<:Real} = zeros(plant.nd),
    x_noise::Vector{<:Real} = zeros(plant.nx),
    x0 = zeros(plant.nx),
    x̂0 = nothing,
    lastu = plant.uop,
)
    model.Ts ≈ plant.Ts || error("Sampling time Ts of mpc and plant must be equal")
    old_x0 = copy(plant.x)
    old_x̂0 = copy(estim.x̂)
    T_data  = collect(plant.Ts*(0:(N-1)))
    Y_data  = Matrix{Float64}(undef, plant.ny, N)
    Ŷ_data  = Matrix{Float64}(undef, model.ny, N)
    U_Ry_data = Matrix{Float64}(undef, length(u_ry), N)
    U_data  = Matrix{Float64}(undef, plant.nu, N)
    D_data  = Matrix{Float64}(undef, plant.nd, N)
    X_data  = Matrix{Float64}(undef, plant.nx, N) 
    X̂_data  = Matrix{Float64}(undef, estim.nx̂, N)
    setstate!(plant, x0)
    lastd, lasty = d, evaloutput(plant, d)
    if isnothing(x̂0)
        initstate!(estim_mpc, lastu, lasty[estim.i_ym], lastd)
    else
        setstate!(estim_mpc, x̂0)
    end
    for i=1:N
        d = lastd + d_step + d_noise.*randn(plant.nd)
        y = evaloutput(plant, d) + y_step + y_noise.*randn(plant.ny)
        ym = y[estim.i_ym]
        u  = sim_getu!(estim_mpc, u_ry, d, ym)
        up = u + u_step + u_noise.*randn(plant.nu)
        Y_data[:, i]  = y
        Ŷ_data[:, i]  = evalŷ(estim, ym, d)
        U_Ry_data[:, i] = u_ry
        U_data[:, i]  = u
        D_data[:, i]  = d
        X_data[:, i]  = plant.x
        X̂_data[:, i]  = estim.x̂
        x = updatestate!(plant, up, d); 
        x[:] += x_noise.*randn(plant.nx)
        updatestate!(estim_mpc, u, ym, d)
    end
    res = SimResult(
        estim_mpc, T_data, Y_data, U_Ry_data, Ŷ_data, U_data, D_data, X_data, X̂_data
    )
    setstate!(plant, old_x0) 
    setstate!(estim, old_x̂0)
    return res
end

sim_getu!(::StateEstimator, u, _ , _ ) = u
sim_getu!(mpc::PredictiveController, ry, d, ym) = moveinput!(mpc, ry, d; ym)


@recipe function simresultplot(
    res::SimResult{<:PredictiveController}; 
    plotRy          = true,
    plotŶminŶmax    = true,
    plotŶ           = false,
    plotU           = true,
    plotRu          = true,
    plotUminUmax    = true,
    plotD           = true,
    plotX           = false,
    plotX̂           = false
)
    mpc = res.obj

    t  = res.T_data
    ny = size(res.Y_data, 1)
    nu = size(res.U_data, 1)
    nd = size(res.D_data, 1)
    nx = size(res.X_data, 1)
    nx̂ = size(res.X̂_data, 1)
    layout_mat = [(ny, 1)]
    plotU && (layout_mat = [layout_mat (nu, 1)])
    (plotD && nd ≠ 0) && (layout_mat = [layout_mat (nd, 1)])
    plotX && (layout_mat = [layout_mat (nx, 1)])
    plotX̂ && (layout_mat = [layout_mat (nx̂, 1)])

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
        if plotŶ
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
        if plotRy && !iszero(mpc.M_Hp)
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
        if plotŶminŶmax && !isinf(mpc.con.Ŷmin[i])
            @series begin
                yguide    --> "\$y_$i\$"
                color     --> 4
                subplot   --> subplot_base + i
                linestyle --> :dot
                linewidth --> 2.0
                label     --> "\$\\mathbf{\\hat{y}_{min}}\$"
                legend    --> true
                t, fill(mpc.con.Ŷmin[i], length(t))
            end
        end
        if plotŶminŶmax && !isinf(mpc.con.Ŷmax[i])
            @series begin
                yguide    --> "\$y_$i\$"
                color     --> 5
                subplot   --> subplot_base + i
                linestyle --> :dot
                linewidth --> 2.0
                label     --> "\$\\mathbf{\\hat{y}_{max}}\$"
                legend    --> true
                t, fill(mpc.con.Ŷmax[i], length(t))
            end
        end
    end
    subplot_base += ny
    # --- manipulated inputs u ---
    if plotU
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
            if plotRu && !iszero(mpc.L_Hp)
                @series begin
                    yguide    --> "\$u_$i\$"
                    color     --> 3
                    subplot   --> subplot_base + i
                    linestyle --> :dash
                    label     --> "\$\\mathbf{r_{u}}\$"
                    legend    --> true
                    t, res.Ry_data[i, :]
                end
            end
            if plotUminUmax && !isinf(mpc.con.Umin[i])
                @series begin
                    yguide    --> "\$u_$i\$"
                    color     --> 4
                    subplot   --> subplot_base + i
                    linestyle --> :dot
                    linewidth --> 2.0
                    label     --> "\$\\mathbf{u_{min}}\$"
                    legend    --> true
                    t, fill(mpc.con.Umin[i], length(t))
                end
            end
            if plotUminUmax && !isinf(mpc.con.Umax[i])
                @series begin
                    yguide    --> "\$u_$i\$"
                    color     --> 5
                    subplot   --> subplot_base + i
                    linestyle --> :dot
                    linewidth --> 2.0
                    label     --> "\$\\mathbf{u_{max}}\$"
                    legend    --> true
                    t, fill(mpc.con.Umax[i], length(t))
                end
            end
        end
        subplot_base += nu
    end
    # --- measured disturbances d ---
    if plotD
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
    if plotX
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
        subplot_base += nx
    end
    # --- estimated states x̂ ---
    if plotX̂
        for i in 1:nx̂
            @series begin
                yguide     --> "\$\\hat{x}_$i\$"
                color      --> 2
                subplot    --> subplot_base + i
                linestyle --> :dashdot
                linewidth --> 0.75
                label      --> "\$\\mathbf{\\hat{x}}\$"
                legend     --> false
                t, res.X̂_data[i, :]
            end
        end
    end

end



@recipe function simresultplot(
    res::SimResult{<:StateEstimator};
    plotŶ           = true,
    plotU           = true,
    plotD           = true,
    plotX           = false,
    plotX̂           = false
)
    t   = res.T_data
    ny = size(res.Y_data, 1)
    nu = size(res.U_data, 1)
    nd = size(res.D_data, 1)
    nx = size(res.X_data, 1)
    nx̂ = size(res.X̂_data, 1)
    layout_mat = [(ny, 1)]
    plotU && (layout_mat = [layout_mat (nu, 1)])
    (plotD && nd ≠ 0) && (layout_mat = [layout_mat (nd, 1)])
    plotX && (layout_mat = [layout_mat (nx, 1)])
    plotX̂ && (layout_mat = [layout_mat (nx̂, 1)])

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
        if plotŶ
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
    if plotU
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
    if plotD
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
    if plotX
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
        subplot_base += nx
    end
    # --- estimated states x̂ ---
    if plotX̂
        for i in 1:nx̂
            @series begin
                yguide     --> "\$\\hat{x}_$i\$"
                color      --> 2
                subplot    --> subplot_base + i
                linestyle --> :dashdot
                linewidth --> 0.75
                label      --> "\$\\mathbf{\\hat{x}}\$"
                legend     --> false
                t, res.X̂_data[i, :]
            end
        end
    end
end
