struct SimResult{NT<:Real, O<:Union{SimModel, StateEstimator, PredictiveController}}
    obj::O                  # simulated instance
    xname  ::Vector{String} # plant state names
    T_data ::Vector{NT}     # time in seconds
    Y_data ::Matrix{NT}     # plant outputs (both measured and unmeasured)
    Ry_data::Matrix{NT}     # output setpoints
    Ŷ_data ::Matrix{NT}     # estimated outputs
    U_data ::Matrix{NT}     # manipulated inputs
    Ud_data::Matrix{NT}     # manipulated inputs including load disturbances
    Ru_data::Matrix{NT}     # manipulated input setpoints
    D_data ::Matrix{NT}     # measured disturbances
    X_data ::Matrix{NT}     # plant states
    X̂_data ::Matrix{NT}     # estimated states
end

"""
    SimResult(obj::SimModel,             U_data, Y_data, D_data=[]; <keyword arguments>)
    SimResult(obj::StateEstimator,       U_data, Y_data, D_data=[]; <keyword arguments>)
    SimResult(obj::PredictiveController, U_data, Y_data, D_data=[]; <keyword arguments>)

Manually construct a `SimResult` to quickly plot `obj` simulations.

Except for `obj`, all the arguments should be matrices of `N` columns, where `N` is the 
number of time steps. [`SimResult`](@ref) objects allow to quickly plot simulation results.
Simply call `plot` on them.

# Arguments
!!! info
    Keyword arguments with *`emphasis`* are non-Unicode alternatives.

- `obj` : simulated [`SimModel`](@ref)/[`StateEstimator`](@ref)/[`PredictiveController`](@ref)
- `U_data` : manipulated inputs
- `Y_data` : plant outputs
- `D_data=[]` : measured disturbances
- `X_data=nothing` : plant states
- `X̂_data=nothing` or *`Xhat_data`* : estimated states
- `Ŷ_data=nothing` or *`Yhat_data`* : estimated outputs
- `Ry_data=nothing` : plant output setpoints
- `Ru_data=nothing` : manipulated input setpoints
- `plant=get_model(obj)` : simulated plant model, default to `obj` internal plant model

# Examples
```jldoctest
julia> model = LinModel(tf(1, [1, 1]), 1.0);

julia> N = 5; U_data = fill(1.0, 1, N); Y_data = zeros(1, N);

julia> for i=1:N; updatestate!(model, U_data[:, i]); Y_data[:, i] = model(); end; Y_data
1×5 Matrix{Float64}:
 0.632121  0.864665  0.950213  0.981684  0.993262

julia> res = SimResult(model, U_data, Y_data)
Simulation results of LinModel with 5 time steps.
```
"""
function SimResult(
    obj::O, 
    U_data, 
    Y_data, 
    D_data      = zeros(NT, 0, size(U_data, 2));
    X_data      = nothing,
    Xhat_data   = nothing,
    Yhat_data   = nothing,
    Ry_data     = nothing, 
    Ru_data     = nothing,
    X̂_data = Xhat_data,
    Ŷ_data = Yhat_data,
    plant  = get_model(obj)
) where {NT<:Real, O<:Union{SimModel{NT}, StateEstimator{NT}, PredictiveController{NT}}}
    model = get_model(obj)
    Ts, nu, ny, nx̂ = model.Ts, model.nu, model.ny, get_nx̂(obj)
    nx = plant.nx
    N = size(U_data, 2)
    T_data = collect(Ts*(0:N-1))
    isnothing(X_data)  && (X_data  = fill(NaN, nx, N))
    isnothing(X̂_data)  && (X̂_data  = fill(NaN, nx̂, N))
    isnothing(Ry_data) && (Ry_data = fill(NaN, ny, N))
    isnothing(Ru_data) && (Ru_data = fill(NaN, nu, N))
    isnothing(Ŷ_data)  && (Ŷ_data  = fill(NaN, ny, N))
    NU, NY, NX, NX̂ = size(U_data, 2), size(Y_data, 2), size(X_data, 2), size(X̂_data, 2)
    NRy, NRu, NŶ = size(Ry_data, 2), size(Ru_data, 2), size(Ŷ_data, 2)
    if !(NU == NY == NX == NX̂ == NRy == NRu == NŶ)
        throw(ArgumentError("All arguments must have the same number of columns (time steps)"))
    end
    size(Y_data, 2) == N || error("Y_data must be of size ($ny, $N)")
    return SimResult{NT, O}(
        obj, plant.xname, 
        T_data, Y_data, Ry_data, Ŷ_data, 
        U_data, U_data, Ru_data, D_data, X_data, X̂_data
    )
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
that can be visualized by calling `plot` on them. Note that the method mutates `plant`
internal states.

# Examples
```jldoctest
julia> plant = NonLinModel((x,u,d)->0.1x+u+d, (x,_)->2x, 10.0, 1, 1, 1, 1, solver=nothing);

julia> res = sim!(plant, 15, [0], [0], x_0=[1])
Simulation results of NonLinModel with 15 time steps.
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
    @progress name="$(typeof(plant).name.name) simulation" for i=1:N
        y = evaloutput(plant, d) 
        Y_data[:, i] .= y
        U_data[:, i] .= u
        D_data[:, i] .= d
        X_data[:, i] .= plant.x0 .+ plant.xop
        updatestate!(plant, u, d)
    end
    return SimResult(
        plant, plant.xname, 
        T_data, Y_data, U_data, Y_data, 
        U_data, U_data, U_data, D_data, X_data, X_data
    )
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
!!! info
    Keyword arguments with *`emphasis`* are non-Unicode alternatives.

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
- `x̂_0 = nothing` or *`xhat_0`* : initial estimate ``\mathbf{x̂}(0)``, [`initstate!`](@ref)
   is used if `nothing`
- `lastu = plant.uop` : last plant input ``\mathbf{u}`` for ``\mathbf{x̂}`` initialization

# Examples
```jldoctest
julia> model = LinModel(tf(3, [30, 1]), 0.5);

julia> estim = KalmanFilter(model, σR=[0.5], σQ=[0.25], σQint_ym=[0.01], σPint_ym_0=[0.1]);

julia> res = sim!(estim, 50, [0], y_noise=[0.5], x_noise=[0.25], x_0=[-10], x̂_0=[0, 0])
Simulation results of KalmanFilter with 50 time steps.
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
```jldoctest
julia> model = LinModel([tf(3, [30, 1]); tf(2, [5, 1])], 4);

julia> mpc = setconstraint!(LinMPC(model, Mwt=[0, 1], Nwt=[0.01], Hp=30), ymin=[0, -Inf]);

julia> res = sim!(mpc, 25, [0, 0], y_noise=[0.1], y_step=[-10, 0])
Simulation results of LinMPC with 25 time steps.
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
    x_0    = plant.xop,
    xhat_0 = nothing,
    lastu  = plant.uop,
    x̂_0 = xhat_0
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
    @progress name="$(typeof(est_mpc).name.name) simulation" for i=1:N
        d = lastd + d_step + d_noise.*randn(plant.nd)
        y = evaloutput(plant, d) + y_step + y_noise.*randn(plant.ny)
        ym = y[estim.i_ym]
        preparestate!(est_mpc, ym, d)
        u  = sim_getu!(est_mpc, u_ry, d, ru)
        ud = u + u_step + u_noise.*randn(plant.nu)
        Y_data[:, i]        .= y
        Ŷ_data[:, i]        .= evalŷ(estim, d)
        U_Ry_data[:, i]     .= u_ry
        U_data[:, i]        .= u
        Ud_data[:, i]       .= ud
        Ru_data[:, i]       .= ru
        D_data[:, i]        .= d
        X_data[:, i]        .= plant.x0 .+ plant.xop
        X̂_data[:, i]        .= estim.x̂0 .+ estim.x̂op
        x = updatestate!(plant, ud, d); 
        x[:] += x_noise.*randn(plant.nx)
        updatestate!(est_mpc, u, ym, d)
    end
    res = SimResult(
        est_mpc, plant.xname,
        T_data, Y_data, U_Ry_data, Ŷ_data, 
        U_data, Ud_data, Ru_data, D_data, X_data, X̂_data
    )
    plant.x0 .= old_x0
    return res
end

"Compute new `u` for predictive controller simulation."
function sim_getu!(mpc::PredictiveController, ry, d, ru)
    return moveinput!(mpc, ry, d; R̂u=repeat(ru, mpc.Hp))
end
"Keep manipulated input `u` unchanged for state estimator simulation."
sim_getu!(::StateEstimator, u, _ , _ ) = u


# dummy plot methods to document recipes (both in ?-mode and web documentation)
plot(::Nothing, ::SimResult{<:Real, <:SimModel}) = nothing
plot(::Nothing, ::SimResult{<:Real, <:StateEstimator}) = nothing
plot(::Nothing, ::SimResult{<:Real, <:PredictiveController}) = nothing

@doc raw"""
    plot(res::SimResult{<:Real, <:SimModel}; <keyword arguments>)

Plot the simulation results of a [`SimModel`](@ref).

# Arguments
!!! info
    The keyword arguments can be `Bool`s, index ranges (`2:4`) or vectors (`[1, 3]`), to
    select the variables to plot.

- `res::SimResult{<:Real, <:SimModel}` : simulation results to plot
- `ploty=true` : plot plant outputs ``\mathbf{y}``
- `plotu=true` : plot manipulated inputs ``\mathbf{u}``
- `plotd=true` : plot measured disturbances ``\mathbf{d}`` if applicable
- `plotx=false` : plot plant states ``\mathbf{x}``

# Examples
```julia-repl
julia> res = sim!(LinModel(tf(2, [10, 1]), 2.0), 25);

julia> using Plots; plot(res, plotu=false)
```
![plot_model](../assets/plot_model.svg)
"""
plot(::Nothing, ::SimResult{<:Real, <:SimModel})

function get_indices(arg::IntRangeOrVector, n)
    if length(unique(arg)) ≠ length(arg) || maximum(arg) > n
        error("Plot keyword argument arguments should contains valid and unique indices")
    end
    return arg
end
get_indices(arg::Bool, n) = arg ? (1:n) : Int64[]

@recipe function plot_recipe(
    res::SimResult{<:Real, <:SimModel};
    ploty  = true,
    plotu  = true,
    plotd  = true,
    plotx  = false,
)
    t   = res.T_data

    model = res.obj
    uname = model.uname
    yname = model.yname
    dname = model.dname
    xname = res.xname

    indices_y = get_indices(ploty, size(res.Y_data, 1))
    indices_u = get_indices(plotu, size(res.U_data, 1))
    indices_d = get_indices(plotd, size(res.D_data, 1))
    indices_x = get_indices(plotx, size(res.X_data, 1))

    ny = length(indices_y)
    nu = length(indices_u)
    nd = length(indices_d)
    nx = length(indices_x)

    layout_mat = Matrix{Tuple{Int64, Int64}}(undef, 1, 0)
    ny ≠ 0 && (layout_mat = [layout_mat (ny, 1)])
    nu ≠ 0 && (layout_mat = [layout_mat (nu, 1)])
    nd ≠ 0 && (layout_mat = [layout_mat (nd, 1)])
    nx ≠ 0 && (layout_mat = [layout_mat (nx, 1)])
    layout := layout_mat

    # --- outputs y ---
    subplot_base = 0
    for i in 1:ny
        i_y = indices_y[i]
        @series begin
            i == ny && (xguide --> "Time (s)")
            yguide  --> yname[i_y]
            color   --> 1
            subplot --> subplot_base + i
            label   --> "\$\\mathbf{y}\$"
            legend  --> false
            t, res.Y_data[i_y, :]
        end
    end
    subplot_base += ny
    # --- manipulated inputs u ---
    for i in 1:nu
        i_u = indices_u[i]
        @series begin
            i == nu && (xguide --> "Time (s)")
            yguide     --> uname[i_u]
            color      --> 1
            subplot    --> subplot_base + i
            seriestype --> :steppost
            label      --> "\$\\mathbf{u}\$"
            legend     --> false
            t, res.U_data[i_u, :]
        end
    end
    subplot_base += nu
    # --- measured disturbances d ---
    for i in 1:nd
        i_d = indices_d[i]
        @series begin
            i == nd && (xguide --> "Time (s)")
            yguide  --> dname[i_d]
            color   --> 1
            subplot --> subplot_base + i
            label   --> "\$\\mathbf{d}\$"
            legend  --> false
            t, res.D_data[i_d, :]
        end
    end
    subplot_base += nd
    # --- plant states x ---
    for i in 1:nx
        i_x = indices_x[i]
        @series begin
            i == nx && (xguide --> "Time (s)")
            yguide     --> xname[i_x]
            color      --> 1
            subplot    --> subplot_base + i
            label      --> "\$\\mathbf{x}\$"
            legend     --> false
            t, res.X_data[i_x, :]
        end
    end
end

@doc raw"""
    plot(res::SimResult{<:Real, <:StateEstimator}; <keyword arguments>)

Plot the simulation results of a [`StateEstimator`](@ref).

# Arguments
!!! info
    The keyword arguments can be `Bool`s, index ranges (`2:4`) or vectors (`[1, 3]`), to
    select the variables to plot. Keywords in *`emphasis`* are non-Unicode alternatives.

- `res::SimResult{<:Real, <:StateEstimator}` : simulation results to plot
- `plotŷ=true` or *`plotyhat`* : plot estimated outputs ``\mathbf{ŷ}``
- `plotx̂=false` or *`plotxhat`* : plot estimated states ``\mathbf{x̂}``
- `plotxwithx̂=false` or *`plotxwithxhat`* : plot plant states ``\mathbf{x}`` and estimated 
   states ``\mathbf{x̂}`` together
- `plotx̂min=true` or *`plotxhatmin`* : plot estimated state lower bounds ``\mathbf{x̂_{min}}``
   if applicable
- `plotx̂max=true` or *`plotxhatmax`* : plot estimated state upper bounds ``\mathbf{x̂_{max}}``
   if applicable
- `<keyword arguments>` of [`plot(::SimResult{<:Real, <:SimModel})`](@ref)

# Examples
```julia-repl
julia> res = sim!(KalmanFilter(LinModel(tf(3, [2.0, 1]), 1.0)), 25, [0], y_step=[1]);

julia> using Plots; plot(res, plotu=false, plotŷ=true, plotxwithx̂=true)
```
![plot_estimator](../assets/plot_estimator.svg)
"""
plot(::Nothing, ::SimResult{<:Real, <:StateEstimator})

@recipe function plot_recipe(
    res::SimResult{<:Real, <:StateEstimator};
    ploty           = true,
    plotyhat        = true,
    plotu           = true,
    plotd           = true,
    plotx           = false,
    plotxhat        = false,
    plotxwithxhat   = false,
    plotxhatmin     = true,
    plotxhatmax     = true,
    plotŷ      = plotyhat,
    plotx̂      = plotxhat,
    plotxwithx̂ = plotxwithxhat,
    plotx̂min   = plotxhatmin,
    plotx̂max   = plotxhatmax
)
    t     = res.T_data
    estim = res.obj
    model = estim.model

    X̂min, X̂max = getX̂con(estim, estim.nx̂)

    uname = model.uname
    yname = model.yname
    dname = model.dname
    xname = res.xname
    x̂name = [model.xname; ["\$\\hat{x}_{$i}\$" for i in (length(xname)+1):(estim.nx̂)]]
    xx̂name = size(res.X̂_data, 1) ≥ size(res.X_data, 1) ? x̂name : xname
   
    indices_y    = get_indices(ploty, size(res.Y_data, 1))
    indices_u    = get_indices(plotu, size(res.U_data, 1))
    indices_d    = get_indices(plotd, size(res.D_data, 1))
    indices_x    = get_indices(plotx, size(res.X_data, 1))
    indices_ŷ    = get_indices(plotŷ, size(res.Ŷ_data, 1))
    indices_x̂    = get_indices(plotx̂, size(res.X̂_data, 1))
    indices_xx̂   = get_indices(plotxwithx̂, max(size(res.X_data, 1), size(res.X̂_data, 1)))
    indices_x̂min = get_indices(plotx̂min, size(res.X̂_data, 1))
    indices_x̂max = get_indices(plotx̂max, size(res.X̂_data, 1))

    ny  = length(indices_y)
    nu  = length(indices_u)
    nd  = length(indices_d)
    nx  = length(indices_x)
    nx̂  = length(indices_x̂)
    nxx̂ = length(indices_xx̂)
    
    layout_mat = Matrix{Tuple{Int64, Int64}}(undef, 1, 0)
    ny  ≠ 0 && (layout_mat = [layout_mat (ny, 1)])
    nu  ≠ 0 && (layout_mat = [layout_mat (nu, 1)])
    nd  ≠ 0 && (layout_mat = [layout_mat (nd, 1)])
    nx  ≠ 0 && (layout_mat = [layout_mat (nx, 1)])
    nx̂  ≠ 0 && (layout_mat = [layout_mat (nx̂, 1)])
    nxx̂ ≠ 0 && (layout_mat = [layout_mat (nxx̂, 1)])
    layout := layout_mat

    # --- outputs y ---
    subplot_base = 0
    for i in 1:ny
        i_y = indices_y[i]
        @series begin
            i == ny && (xguide --> "Time (s)")
            yguide  --> yname[i_y]
            color   --> 1
            subplot --> subplot_base + i
            label   --> "\$\\mathbf{y}\$"
            legend  --> false
            t, res.Y_data[i_y, :]
        end
        if i_y in indices_ŷ
            @series begin
                i == ny && (xguide --> "Time (s)")
                yguide  --> yname[i_y]
                color     --> 2
                subplot   --> subplot_base + i
                linestyle --> :dashdot
                linewidth --> 0.75
                label     --> "\$\\mathbf{\\hat{y}}\$"
                legend    --> true
                t, res.Ŷ_data[i_y, :]
            end
        end
    end
    subplot_base += ny
    # --- manipulated inputs u ---
    for i in 1:nu
        i_u = indices_u[i]
        @series begin
            i == nu && (xguide --> "Time (s)")
            yguide     --> uname[i_u]
            color      --> 1
            subplot    --> subplot_base + i
            seriestype --> :steppost
            label      --> "\$\\mathbf{u}\$"
            legend     --> false
            t, res.U_data[i_u, :]
        end
    end
    subplot_base += nu
    # --- measured disturbances d ---
    for i in 1:nd
        i_d = indices_d[i]
        @series begin
            i == nd && (xguide --> "Time (s)")
            yguide  --> dname[i_d]
            color   --> 1
            subplot --> subplot_base + i
            label   --> "\$\\mathbf{d}\$"
            legend  --> false
            t, res.D_data[i_d, :]
        end
    end
    subplot_base += nd
    # --- plant states x ---
    for i in 1:nx
        i_x = indices_x[i]
        @series begin
            i == nx && (xguide --> "Time (s)")
            yguide     --> xname[i_x]
            color      --> 1
            subplot    --> subplot_base + i
            label      --> "\$\\mathbf{x}\$"
            legend     --> false
            t, res.X_data[i_x, :]
        end
    end
    subplot_base += nx
    # --- estimated states x̂ ---
    for i in 1:nx̂
        i_x̂ = indices_x̂[i]
        @series begin
            i == nx̂ && (xguide --> "Time (s)")
            yguide     --> x̂name[i_x̂]
            color      --> 2
            subplot    --> subplot_base + i
            linestyle  --> :dashdot
            linewidth  --> 0.75
            label      --> "\$\\mathbf{\\hat{x}}\$"
            legend     --> false
            t, res.X̂_data[i_x̂, :]
        end
        x̂min_i, x̂max_i = X̂min[end-2*estim.nx̂+i_x̂], X̂max[end-2*estim.nx̂+i_x̂]
        if i_x̂ in indices_x̂min && !isinf(x̂min_i)
            @series begin
                i == nx̂ && (xguide --> "Time (s)")
                yguide     --> x̂name[i_x̂]
                color      --> 4
                subplot    --> subplot_base + i
                linestyle  --> :dot
                linewidth  --> 1.5
                label      --> "\$\\mathbf{\\hat{x}_{min}}\$"
                legend     --> true
                t, fill(x̂min_i, length(t))
            end
        end
        if i_x̂ in indices_x̂max && !isinf(x̂max_i)
            @series begin
                i == nx̂ && (xguide --> "Time (s)")
                yguide     --> x̂name[i_x̂]
                color      --> 5
                subplot    --> subplot_base + i
                linestyle  --> :dot
                linewidth  --> 1.5
                label      --> "\$\\mathbf{\\hat{x}_{max}}\$"
                legend     --> true
                t, fill(x̂max_i, length(t))
            end
        end
    end
    subplot_base += nx̂
    # --- plant states x and estimated states x̂ together ---
    for i in 1:nxx̂
        i_xx̂ = indices_xx̂[i]
        isplotted_x = i_xx̂ ≤ size(res.X_data, 1)
        isplotted_x̂ = i_xx̂ ≤ size(res.X̂_data, 1)
        if isplotted_x
            @series begin
                i == nxx̂ && (xguide --> "Time (s)")
                yguide     --> xx̂name[i_xx̂]
                color      --> 1
                subplot    --> subplot_base + i
                label      --> "\$\\mathbf{x}\$"
                legend     --> (isplotted_x && isplotted_x̂)
                t, res.X_data[i_xx̂, :]
            end
        end
        if isplotted_x̂
            @series begin
                i == nxx̂ && (xguide --> "Time (s)")
                yguide     --> xx̂name[i_xx̂]
                color      --> 2
                subplot    --> subplot_base + i
                linestyle  --> :dashdot
                linewidth  --> 0.75
                label      --> "\$\\mathbf{\\hat{x}}\$"
                legend     --> (isplotted_x && isplotted_x̂)
                t, res.X̂_data[i_xx̂, :]
            end
            x̂min_i, x̂max_i = X̂min[end-2*estim.nx̂+i_xx̂], X̂max[end-2*estim.nx̂+i_xx̂]
            if i_xx̂ in indices_x̂min && !isinf(x̂min_i)
                @series begin
                    i == nxx̂ && (xguide --> "Time (s)")
                    yguide     --> xx̂name[i_xx̂]
                    color      --> 4
                    subplot    --> subplot_base + i
                    linestyle  --> :dot
                    linewidth  --> 1.5
                    label      --> "\$\\mathbf{\\hat{x}_{min}}\$"
                    legend     --> true
                    t, fill(x̂min_i, length(t))
                end
            end
            if i_xx̂ in indices_x̂max && !isinf(x̂max_i)
                @series begin
                    i == nxx̂ && (xguide --> "Time (s)")
                    yguide     --> xx̂name[i_xx̂]
                    color      --> 5
                    subplot    --> subplot_base + i
                    linestyle  --> :dot
                    linewidth  --> 1.5
                    label      --> "\$\\mathbf{\\hat{x}_{max}}\$"
                    legend     --> true
                    t, fill(x̂max_i, length(t))
                end
            end
        end
    end
end

@doc raw"""
    plot(res::SimResult{<:Real, <:PredictiveController}; <keyword arguments>)

Plot the simulation results of a [`PredictiveController`](@ref).

# Arguments
!!! info
    The keyword arguments can be `Bool`s, index ranges (`2:4`) or vectors (`[1, 3]`), to
    select the variables to plot.

- `res::SimResult{<:Real, <:PredictiveController}` : simulation results to plot
- `plotry=true` : plot plant output setpoints ``\mathbf{r_y}`` if applicable
- `plotymin=true` : plot predicted output lower bounds ``\mathbf{y_{min}}`` if applicable
- `plotymax=true` : plot predicted output upper bounds ``\mathbf{y_{max}}`` if applicable
- `plotru=true` : plot manipulated input setpoints ``\mathbf{r_u}`` if applicable
- `plotumin=true` : plot manipulated input lower bounds ``\mathbf{u_{min}}`` if applicable
- `plotumax=true` : plot manipulated input upper bounds ``\mathbf{u_{max}}`` if applicable
- `<keyword arguments>` of [`plot(::SimResult{<:Real, <:SimModel})`](@ref)
- `<keyword arguments>` of [`plot(::SimResult{<:Real, <:StateEstimator})`](@ref)

# Examples
```julia-repl
julia> model = LinModel(tf(2, [5.0, 1]), 1.0);

julia> res = sim!(setconstraint!(LinMPC(model), umax=[1.0]), 25, [0], u_step=[-1]);

julia> using Plots; plot(res, plotŷ=true, plotry=true, plotumax=true, plotx̂=[2])
```
![plot_controller](../assets/plot_controller.svg)
"""
plot(::Nothing, ::SimResult{<:Real, <:PredictiveController})

@recipe function plot_recipe(
    res::SimResult{<:Real, <:PredictiveController};
    ploty           = true,
    plotry          = true,
    plotymin        = true,
    plotymax        = true,
    plotyhat        = false,
    plotu           = true,
    plotru          = true,
    plotumin        = true,
    plotumax        = true,
    plotd           = true,
    plotx           = false,
    plotxhat        = false,
    plotxwithxhat   = false,
    plotxhatmin     = true,
    plotxhatmax     = true,
    plotŷ       = plotyhat,
    plotx̂       = plotxhat,
    plotxwithx̂  = plotxwithxhat,
    plotx̂min    = plotxhatmin,
    plotx̂max    = plotxhatmax
)
    t     = res.T_data
    mpc   = res.obj
    estim = mpc.estim
    model = mpc.estim.model

    Umin, Umax = getUcon(mpc, model.nu)
    Ymin, Ymax = getYcon(mpc, model.ny)
    X̂min, X̂max = getX̂con(mpc.estim, estim.nx̂)

    uname = model.uname
    yname = model.yname
    dname = model.dname
    xname = res.xname
    x̂name = [model.xname; ["\$\\hat{x}_{$i}\$" for i in (length(xname)+1):(estim.nx̂)]]
    xx̂name = size(res.X̂_data, 1) ≥ size(res.X_data, 1) ? x̂name : xname

    ny = size(res.Y_data, 1)
    nu = size(res.U_data, 1)
    nd = size(res.D_data, 1)
    nx = size(res.X_data, 1)
    nx̂ = size(res.X̂_data, 1)

    indices_y    = get_indices(ploty, size(res.Y_data, 1))
    indices_ry   = get_indices(plotry, size(res.Ry_data, 1))
    indices_ymin = get_indices(plotymin, size(res.Y_data, 1))
    indices_ymax = get_indices(plotymax, size(res.Y_data, 1))
    indices_u    = get_indices(plotu, size(res.U_data, 1))
    indices_ru   = get_indices(plotru, size(res.Ru_data, 1))
    indices_umin = get_indices(plotumin, size(res.U_data, 1))
    indices_umax = get_indices(plotumax, size(res.U_data, 1))
    indices_d    = get_indices(plotd, size(res.D_data, 1))
    indices_x    = get_indices(plotx, size(res.X_data, 1))
    indices_ŷ    = get_indices(plotŷ, size(res.Ŷ_data, 1))
    indices_x̂    = get_indices(plotx̂, size(res.X̂_data, 1))
    indices_xx̂   = get_indices(plotxwithx̂, max(size(res.X_data, 1), size(res.X̂_data, 1)))
    indices_x̂min = get_indices(plotx̂min, size(res.X̂_data, 1))
    indices_x̂max = get_indices(plotx̂max, size(res.X̂_data, 1))
    
    ny  = length(indices_y)
    nu  = length(indices_u)
    nd  = length(indices_d)
    nx  = length(indices_x)
    nx̂  = length(indices_x̂)
    nxx̂ = length(indices_xx̂)

    layout_mat = Matrix{Tuple{Int64, Int64}}(undef, 1, 0)
    ny  ≠ 0 && (layout_mat = [layout_mat (ny, 1)])
    nu  ≠ 0 && (layout_mat = [layout_mat (nu, 1)])
    nd  ≠ 0 && (layout_mat = [layout_mat (nd, 1)])
    nx  ≠ 0 && (layout_mat = [layout_mat (nx, 1)])
    nx̂  ≠ 0 && (layout_mat = [layout_mat (nx̂, 1)])
    nxx̂ ≠ 0 && (layout_mat = [layout_mat (nxx̂, 1)])
    layout := layout_mat
    
    # --- outputs y ---
    subplot_base = 0
    for i in 1:ny
        i_y = indices_y[i]
        @series begin
            i == ny && (xguide --> "Time (s)")
            yguide  --> yname[i_y]
            color   --> 1
            subplot --> subplot_base + i
            label   --> "\$\\mathbf{y}\$"
            legend  --> false
            t, res.Y_data[i_y, :]
        end
        if i_y in indices_ŷ
            @series begin
                i == ny && (xguide --> "Time (s)")
                yguide  --> yname[i_y]
                color     --> 2
                subplot   --> subplot_base + i
                linestyle --> :dashdot
                linewidth --> 0.75
                label     --> "\$\\mathbf{\\hat{y}}\$"
                legend    --> true
                t, res.Ŷ_data[i_y, :]
            end
        end
        M_Hp_i = mpc.M_Hp[i_y, i_y]
        if i_y in indices_ry && !iszero(M_Hp_i)
            @series begin
                i == ny && (xguide --> "Time (s)")
                yguide    --> yname[i_y]
                color     --> 3
                subplot   --> subplot_base + i
                linestyle --> :dash
                linewidth --> 0.75
                label     --> "\$\\mathbf{r_y}\$"
                legend    --> true
                t, res.Ry_data[i_y, :]
            end
        end
        ymin_i, ymax_i = Ymin[i_y], Ymax[i_y]
        if i_y in indices_ymin && !isinf(ymin_i)
            @series begin
                i == ny && (xguide --> "Time (s)")
                yguide    --> yname[i_y]
                color     --> 4
                subplot   --> subplot_base + i
                linestyle --> :dot
                linewidth --> 1.5
                label     --> "\$\\mathbf{y_{min}}\$"
                legend    --> true
                t, fill(ymin_i, length(t))
            end
        end
        if i_y in indices_ymax && !isinf(ymax_i)
            @series begin
                i == ny && (xguide --> "Time (s)")
                yguide    --> yname[i_y]
                color     --> 5
                subplot   --> subplot_base + i
                linestyle --> :dot
                linewidth --> 1.5
                label     --> "\$\\mathbf{y_{max}}\$"
                legend    --> true
                t, fill(ymax_i, length(t))
            end
        end
    end
    subplot_base += ny
    # --- manipulated inputs u ---
    for i in 1:nu
        i_u = indices_u[i]
        @series begin
            i == nu && (xguide --> "Time (s)")
            yguide     --> uname[i_u]
            color      --> 1
            subplot    --> subplot_base + i
            seriestype --> :steppost
            label      --> "\$\\mathbf{u}\$"
            legend     --> false
            t, res.U_data[i_u, :]
        end
        L_Hp_i = mpc.L_Hp[i_u, i_u]
        if i_u in indices_ru && !iszero(L_Hp_i)
            @series begin
                i == nu && (xguide --> "Time (s)")
                yguide    --> uname[i_u]
                color     --> 3
                subplot   --> subplot_base + i
                linestyle --> :dash
                linewidth --> 0.75
                label     --> "\$\\mathbf{r_{u}}\$"
                legend    --> true
                t, res.Ru_data[i_u, :]
            end
        end
        umin_i, umax_i = Umin[i_u], Umax[i_u]
        if i_u in indices_umin && !isinf(umin_i)
            @series begin
                i == nu && (xguide --> "Time (s)")
                yguide    --> uname[i_u]
                color     --> 4
                subplot   --> subplot_base + i
                linestyle --> :dot
                linewidth --> 1.5
                label     --> "\$\\mathbf{u_{min}}\$"
                legend    --> true
                t, fill(umin_i, length(t))
            end
        end
        if i_u in indices_umax && !isinf(umax_i)
            @series begin
                i == nu && (xguide --> "Time (s)")
                yguide    --> uname[i_u]
                color     --> 5
                subplot   --> subplot_base + i
                linestyle --> :dot
                linewidth --> 1.5
                label     --> "\$\\mathbf{u_{max}}\$"
                legend    --> true
                t, fill(umax_i, length(t))
            end
        end
    end
    subplot_base += nu
    # --- measured disturbances d ---
    for i in 1:nd
        i_d = indices_d[i]
        @series begin
            i == nd && (xguide --> "Time (s)")
            yguide  --> dname[i_d]
            color   --> 1
            subplot --> subplot_base + i
            label   --> "\$\\mathbf{d}\$"
            legend  --> false
            t, res.D_data[i_d, :]
        end
    end
    subplot_base += nd
    # --- plant states x ---
    for i in 1:nx
        i_x = indices_x[i]
        @series begin
            i == nx && (xguide --> "Time (s)")
            yguide     --> xname[i_x]
            color      --> 1
            subplot    --> subplot_base + i
            label      --> "\$\\mathbf{x}\$"
            legend     --> false
            t, res.X_data[i_x, :]
        end
    end
    subplot_base += nx
    # --- estimated states x̂ ---
    for i in 1:nx̂
        i_x̂ = indices_x̂[i]
        @series begin
            i == nx̂ && (xguide --> "Time (s)")
            yguide     --> x̂name[i_x̂]
            color      --> 2
            subplot    --> subplot_base + i
            linestyle  --> :dashdot
            linewidth  --> 0.75
            label      --> "\$\\mathbf{\\hat{x}}\$"
            legend     --> false
            t, res.X̂_data[i_x̂, :]
        end
        x̂min_i, x̂max_i = X̂min[end-2*estim.nx̂+i_x̂], X̂max[end-2*estim.nx̂+i_x̂]
        if i_x̂ in indices_x̂min && !isinf(x̂min_i)
            @series begin
                i == nx̂ && (xguide --> "Time (s)")
                yguide     --> x̂name[i_x̂]
                color      --> 4
                subplot    --> subplot_base + i
                linestyle  --> :dot
                linewidth  --> 1.5
                label      --> "\$\\mathbf{\\hat{x}_{min}}\$"
                legend     --> true
                t, fill(x̂min_i, length(t))
            end
        end
        if i_x̂ in indices_x̂max && !isinf(x̂max_i)
            @series begin
                i == nx̂ && (xguide --> "Time (s)")
                yguide     --> x̂name[i_x̂]
                color      --> 5
                subplot    --> subplot_base + i
                linestyle  --> :dot
                linewidth  --> 1.5
                label      --> "\$\\mathbf{\\hat{x}_{max}}\$"
                legend     --> true
                t, fill(x̂max_i, length(t))
            end
        end
    end
    subplot_base += nx̂
    # --- plant states x and estimated states x̂ together ---
    for i in 1:nxx̂
        i_xx̂ = indices_xx̂[i]
        isplotted_x = i_xx̂ ≤ size(res.X_data, 1)
        isplotted_x̂ = i_xx̂ ≤ size(res.X̂_data, 1)
        if isplotted_x
            @series begin
                i == nxx̂ && (xguide --> "Time (s)")
                yguide     --> xx̂name[i_xx̂]
                color      --> 1
                subplot    --> subplot_base + i
                label      --> "\$\\mathbf{x}\$"
                legend     --> (isplotted_x && isplotted_x̂)
                t, res.X_data[i_xx̂, :]
            end
        end
        if isplotted_x̂
            @series begin
                i == nxx̂ && (xguide --> "Time (s)")
                yguide     --> xx̂name[i_xx̂]
                color      --> 2
                subplot    --> subplot_base + i
                linestyle  --> :dashdot
                linewidth  --> 0.75
                label      --> "\$\\mathbf{\\hat{x}}\$"
                legend     --> (isplotted_x && isplotted_x̂)
                t, res.X̂_data[i_xx̂, :]
            end
            x̂min_i, x̂max_i = X̂min[end-2*estim.nx̂+i_xx̂], X̂max[end-2*estim.nx̂+i_xx̂]
            if i_xx̂ in indices_x̂min && !isinf(x̂min_i)
                @series begin
                    i == nxx̂ && (xguide --> "Time (s)")
                    yguide     --> xx̂name[i_xx̂]
                    color      --> 4
                    subplot    --> subplot_base + i
                    linestyle  --> :dot
                    linewidth  --> 1.5
                    label      --> "\$\\mathbf{\\hat{x}_{min}}\$"
                    legend     --> true
                    t, fill(x̂min_i, length(t))
                end
            end
            if i_xx̂ in indices_x̂max && !isinf(x̂max_i)
                @series begin
                    i == nxx̂ && (xguide --> "Time (s)")
                    yguide     --> xx̂name[i_xx̂]
                    color      --> 5
                    subplot    --> subplot_base + i
                    linestyle  --> :dot
                    linewidth  --> 1.5
                    label      --> "\$\\mathbf{\\hat{x}_{max}}\$"
                    legend     --> true
                    t, fill(x̂max_i, length(t))
                end
            end
        end
    end
end

getUcon(mpc::PredictiveController, _ ) = mpc.con.U0min+mpc.Uop, mpc.con.U0max+mpc.Uop
getYcon(mpc::PredictiveController, _ ) = mpc.con.Y0min+mpc.Yop, mpc.con.Y0max+mpc.Yop
getX̂con(estim::StateEstimator, nx̂) = fill(-Inf, 2nx̂), fill(+Inf, 2nx̂)
