@doc raw"""
    initstate!(mpc::PredictiveController, u, ym, d=[]) -> x̂

Init the states of `mpc.estim` [`StateEstimator`](@ref) and warm start `mpc.ΔŨ` at zero.
"""
function initstate!(mpc::PredictiveController, u, ym, d=empty(mpc.estim.x̂))
    mpc.ΔŨ .= 0
    return initstate!(mpc.estim, u, ym, d)
end

@doc raw"""
    moveinput!(mpc::PredictiveController, ry=mpc.estim.model.yop, d=[]; <keyword args>) -> u

Compute the optimal manipulated input value `u` for the current control period.

Solve the optimization problem of `mpc` [`PredictiveController`](@ref) and return the
results ``\mathbf{u}(k)``. Following the receding horizon principle, the algorithm discards
the optimal future manipulated inputs ``\mathbf{u}(k+1), \mathbf{u}(k+2), ...`` Note that
the method mutates `mpc` internal data but it does not modifies `mpc.estim` states. Call
[`updatestate!(mpc, u, ym, d)`](@ref) to update `mpc` state estimates.

Calling a [`PredictiveController`](@ref) object calls this method.

See also [`LinMPC`](@ref), [`ExplicitMPC`](@ref), [`NonLinMPC`](@ref).

# Arguments
- `mpc::PredictiveController` : solve optimization problem of `mpc`.
- `ry=mpc.estim.model.yop` : current output setpoints ``\mathbf{r_y}(k)``.
- `d=[]` : current measured disturbances ``\mathbf{d}(k)``.
- `D̂=repeat(d, mpc.Hp)` : predicted measured disturbances ``\mathbf{D̂}``, constant in the
  future by default or ``\mathbf{d̂}(k+j)=\mathbf{d}(k)`` for ``j=1`` to ``H_p``.
- `R̂y=repeat(ry, mpc.Hp)` : predicted output setpoints ``\mathbf{R̂_y}``, constant in the
  future by default or ``\mathbf{r̂_y}(k+j)=\mathbf{r_y}(k)`` for ``j=1`` to ``H_p``.
- `R̂u=repeat(mpc.estim.model.uop, mpc.Hp)` : predicted manipulated input setpoints, constant
  in the future by default or ``\mathbf{r̂_u}(k+j)=\mathbf{u_{op}}`` for ``j=0`` to ``H_p-1``.
- `ym=nothing` : current measured outputs ``\mathbf{y^m}(k)``, only required if `mpc.estim` 
   is an [`InternalModel`](@ref).

# Examples
```jldoctest
julia> mpc = LinMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1000, Hc=1);

julia> ry = [5]; u = moveinput!(mpc, ry); round.(u, digits=3)
1-element Vector{Float64}:
 1.0
```
"""
function moveinput!(
    mpc::PredictiveController, 
    ry::Vector = mpc.estim.model.yop, 
    d ::Vector = empty(mpc.estim.x̂);
    D̂ ::Vector = repeat(d,  mpc.Hp),
    R̂y::Vector = repeat(ry, mpc.Hp),
    R̂u::Vector = mpc.noR̂u ? empty(mpc.estim.x̂) : repeat(mpc.estim.model.uop, mpc.Hp),
    ym::Union{Vector, Nothing} = nothing
)
    validate_args(mpc, ry, d, D̂, R̂y, R̂u)
    initpred!(mpc, mpc.estim.model, d, ym, D̂, R̂y, R̂u)
    linconstraint!(mpc, mpc.estim.model)
    ΔŨ = optim_objective!(mpc)
    Δu = ΔŨ[1:mpc.estim.model.nu] # receding horizon principle: only Δu(k) is used (1st one)
    u = mpc.estim.lastu0 + mpc.estim.model.uop + Δu
    return u
end



@doc raw"""
    getinfo(mpc::PredictiveController) -> info

Get additional info about `mpc` controller optimum for troubleshooting.

The function should be called after calling [`moveinput!`](@ref). It returns the dictionary
`info` with the following fields:

- `:ΔU`  : optimal manipulated input increments over ``H_c``, ``\mathbf{ΔU}``.
- `:ϵ`   : optimal slack variable, ``ϵ``.
- `:J`   : objective value optimum, ``J``.
- `:U`   : optimal manipulated inputs over ``H_p``, ``\mathbf{U}``.
- `:u`   : current optimal manipulated input, ``\mathbf{u}(k)``.
- `:d`   : current measured disturbance, ``\mathbf{d}(k)``.
- `:D̂`   : predicted measured disturbances over ``H_p``, ``\mathbf{D̂}``.
- `:ŷ`   : current estimated output, ``\mathbf{ŷ}(k)``.
- `:Ŷ`   : optimal predicted outputs over ``H_p``, ``\mathbf{Ŷ}``.
- `:x̂end`: optimal terminal states, ``\mathbf{x̂}_{k-1}(k+H_p)``.
- `:Ŷs`  : predicted stochastic output over ``H_p`` of [`InternalModel`](@ref), ``\mathbf{Ŷ_s}``.
- `:R̂y`  : predicted output setpoint over ``H_p``, ``\mathbf{R̂_y}``.
- `:R̂u`  : predicted manipulated input setpoint over ``H_p``, ``\mathbf{R̂_u}``.

For [`LinMPC`](@ref) and [`NonLinMPC`](@ref), the field `:sol` also contains the optimizer
solution summary that can be printed. Lastly, the optimal economic cost `:JE` is also
available for [`NonLinMPC`](@ref).

# Examples
```jldoctest
julia> mpc = LinMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1, Hc=1);

julia> u = moveinput!(mpc, [10]);

julia> round.(getinfo(mpc)[:Ŷ], digits=3)
1-element Vector{Float64}:
 10.0
```
"""
function getinfo(mpc::PredictiveController{NT}) where NT<:Real
    info = Dict{Symbol, Union{JuMP._SolutionSummary, Vector{NT}, NT}}()
    Ŷ, x̂    = similar(mpc.Ŷop), similar(mpc.estim.x̂)
    Ŷ, x̂end = predict!(Ŷ, x̂, mpc, mpc.estim.model, mpc.ΔŨ)
    info[:ΔU]   = mpc.ΔŨ[1:mpc.Hc*mpc.estim.model.nu]
    info[:ϵ]    = isinf(mpc.C) ? NaN : mpc.ΔŨ[end]
    info[:J]    = obj_nonlinprog(mpc, mpc.estim.model, Ŷ, mpc.ΔŨ)
    info[:U]    = mpc.S̃*mpc.ΔŨ + mpc.T*(mpc.estim.lastu0 + mpc.estim.model.uop)
    info[:u]    = info[:U][1:mpc.estim.model.nu]
    info[:d]    = mpc.d0 + mpc.estim.model.dop
    info[:D̂]    = mpc.D̂0 + mpc.Dop
    info[:ŷ]    = mpc.ŷ
    info[:Ŷ]    = Ŷ
    info[:x̂end] = x̂end
    info[:Ŷs]  = mpc.Ŷop - repeat(mpc.estim.model.yop, mpc.Hp) # Ŷop = Ŷs + Yop
    info[:R̂y]  = mpc.R̂y
    info[:R̂u]  = mpc.R̂u
    info = addinfo!(info, mpc)
    return info
end

"""
    addinfo!(info, mpc::PredictiveController) -> info

By default, add the solution summary `:sol` that can be printed to `info`.
"""
function addinfo!(info, mpc::PredictiveController)
    info[:sol] = solution_summary(mpc.optim, verbose=true)
    return info
end


@doc raw"""
    initpred!(mpc, model::LinModel, d, ym, D̂, R̂y, R̂u)

Init linear model prediction matrices `F, q̃, p` and current estimated output `ŷ`.

See [`init_predmat`](@ref) and [`init_quadprog`](@ref) for the definition of the matrices.
"""
function initpred!(mpc::PredictiveController, model::LinModel, d, ym, D̂, R̂y, R̂u)
    mpc.ŷ[:] = evalŷ(mpc.estim, ym, d)
    predictstoch!(mpc, mpc.estim, d, ym) # init mpc.Ŷop for InternalModel
    mpc.F[:]  = mpc.K  * mpc.estim.x̂  + mpc.V  * mpc.estim.lastu0 + mpc.Ŷop
    if model.nd ≠ 0
        mpc.d0[:], mpc.D̂0[:] = d - model.dop, D̂ - mpc.Dop
        mpc.D̂E[:] = [d; D̂]
        mpc.F[:]  = mpc.F  + mpc.G  * mpc.d0 + mpc.J  * mpc.D̂0
    end
    mpc.R̂y[:] = R̂y
    Ẑ = mpc.F - mpc.R̂y
    mpc.q̃[:] = 2(mpc.M_Hp*mpc.Ẽ)'*Ẑ
    mpc.p[]  = dot(Ẑ, mpc.M_Hp, Ẑ)
    if ~mpc.noR̂u
        mpc.R̂u[:] = R̂u
        lastu = mpc.estim.lastu0 + model.uop
        Ẑ = mpc.T*lastu - mpc.R̂u
        mpc.q̃[:] = mpc.q̃ + 2(mpc.L_Hp*mpc.S̃)'*Ẑ
        mpc.p[]  = mpc.p[] + dot(Ẑ, mpc.L_Hp, Ẑ)
    end
    return nothing
end

@doc raw"""
    initpred!(mpc::PredictiveController, model::SimModel, d, ym, D̂, R̂y, R̂u)

Init `ŷ, Ŷop, d0, D̂0` matrices when model is not a [`LinModel`](@ref).

`d0` and `D̂0` are the measured disturbances and its predictions without the operating points
``\mathbf{d_{op}}``. The vector `Ŷop` is kept unchanged if `mpc.estim` is not an
[`InternalModel`](@ref).
"""
function initpred!(mpc::PredictiveController, model::SimModel, d, ym, D̂, R̂y, R̂u)
    mpc.ŷ[:] = evalŷ(mpc.estim, ym, d)
    predictstoch!(mpc, mpc.estim, d, ym) # init mpc.Ŷop for InternalModel
    if model.nd ≠ 0
        mpc.d0[:], mpc.D̂0[:] = d - model.dop, D̂ - mpc.Dop
        mpc.D̂E[:] = [d; D̂]
    end
    mpc.R̂y[:] = R̂y
    if ~mpc.noR̂u
        mpc.R̂u[:] = R̂u
    end
    return nothing
end

@doc raw"""
    predictstoch!(mpc::PredictiveController, estim::InternalModel, x̂s, d, ym)

Init `Ŷop` vector when if `estim` is an [`InternalModel`](@ref).

The vector combines the output operating points and the stochastic predictions:
``\mathbf{Ŷ_{op} = Ŷ_{s} + Y_{op}}`` (both values are constant between the nonlinear 
programming iterations).
"""
function predictstoch!(
    mpc::PredictiveController{NT}, estim::InternalModel, d, ym
) where {NT<:Real}
    isnothing(ym) && error("Predictive controllers with InternalModel need the measured "*
                           "outputs ym in keyword argument to compute control actions u")
    ŷd = h(estim.model, estim.x̂d, d - estim.model.dop) + estim.model.yop 
    ŷs = zeros(NT, estim.model.ny)
    ŷs[estim.i_ym] = ym - ŷd[estim.i_ym]  # ŷs=0 for unmeasured outputs
    Ŷs = mpc.Ks*mpc.estim.x̂s + mpc.Ps*ŷs
    mpc.Ŷop[:] = Ŷs + repeat(estim.model.yop, mpc.Hp)
    return nothing
end
"Separate stochastic predictions are not needed if `estim` is not [`InternalModel`](@ref)."
predictstoch!(::PredictiveController, ::StateEstimator, _ , _ ) = nothing

@doc raw"""
    predict!(Ŷ, x̂, mpc::PredictiveController, model::LinModel, ΔŨ) -> Ŷ, x̂end

Compute the predictions `Ŷ` and terminal states `x̂end` if model is a [`LinModel`](@ref).

The method mutates `Ŷ` and `x̂` vector arguments. The `x̂end` vector is used for
the terminal constraints applied on ``\mathbf{x̂}_{k-1}(k+H_p)``.
"""
function predict!(Ŷ, x̂, mpc::PredictiveController, ::LinModel, ΔŨ::Vector{NT}) where {NT<:Real}
     # in-place operations to reduce allocations :
    Ŷ[:] = mul!(Ŷ, mpc.Ẽ, ΔŨ) + mpc.F
    x̂[:] = mul!(x̂, mpc.con.ẽx̂, ΔŨ) + mpc.con.fx̂
    x̂end = x̂
    return Ŷ, x̂end
end

@doc raw"""
    predict!(Ŷ, x̂, mpc::PredictiveController, model::SimModel, ΔŨ) -> Ŷ, x̂end

Compute both vectors if `model` is not a [`LinModel`](@ref).
"""
function predict!(Ŷ, x̂, mpc::PredictiveController, model::SimModel, ΔŨ::Vector{NT}) where {NT<:Real}
    nu, ny, nd, Hp, Hc = model.nu, model.ny, model.nd, mpc.Hp, mpc.Hc
    x̂[:] = mpc.estim.x̂
    u0::Vector{NT} = copy(mpc.estim.lastu0)
    d0 = @views mpc.d0[1:end]
    for j=1:Hp
        if j ≤ Hc
            u0[:] = @views u0 + ΔŨ[(1 + nu*(j-1)):(nu*j)]
        end
        x̂[:]  = f̂(mpc.estim, model, x̂, u0, d0)
        d0    = @views mpc.D̂0[(1 + nd*(j-1)):(nd*j)]
        Ŷ[(1 + ny*(j-1)):(ny*j)] = ĥ(mpc.estim, model, x̂, d0)
    end
    Ŷ[:] = Ŷ + mpc.Ŷop # Ŷop = Ŷs + Yop, and Ŷs=0 if mpc.estim is not an InternalModel
    x̂end = x̂
    return Ŷ, x̂end
end

@doc raw"""
    linconstraint!(mpc::PredictiveController, model::LinModel)

Set `b` vector for the linear model inequality constraints (``\mathbf{A ΔŨ ≤ b}``).

Also init ``\mathbf{f_x̂}`` vector for the terminal constraints, see [`init_predmat`](@ref).
"""
function linconstraint!(mpc::PredictiveController, model::LinModel)
    mpc.con.fx̂[:] = mpc.con.kx̂ * mpc.estim.x̂  + mpc.con.vx̂ * mpc.estim.lastu0
    if model.nd ≠ 0
        mpc.con.fx̂[:] = mpc.con.fx̂ + mpc.con.gx̂ * mpc.d0 + mpc.con.jx̂ * mpc.D̂0
    end
    lastu = mpc.estim.lastu0 + model.uop
    mpc.con.b[:] = [
        -mpc.con.Umin + mpc.T*lastu
        +mpc.con.Umax - mpc.T*lastu
        -mpc.con.ΔŨmin
        +mpc.con.ΔŨmax 
        -mpc.con.Ymin + mpc.F
        +mpc.con.Ymax - mpc.F
        -mpc.con.x̂min + mpc.con.fx̂
        +mpc.con.x̂max - mpc.con.fx̂
    ]
    lincon = mpc.optim[:linconstraint]
    set_normalized_rhs.(lincon, mpc.con.b[mpc.con.i_b])
end

"Set `b` excluding predicted output constraints when `model` is not a [`LinModel`](@ref)."
function linconstraint!(mpc::PredictiveController, model::SimModel)
    lastu = mpc.estim.lastu0 + model.uop
    mpc.con.b[:] = [
        -mpc.con.Umin + mpc.T*lastu
        +mpc.con.Umax - mpc.T*lastu
        -mpc.con.ΔŨmin
        +mpc.con.ΔŨmax 
    ]
    lincon = mpc.optim[:linconstraint]
    set_normalized_rhs.(lincon, mpc.con.b[mpc.con.i_b])
end

"""
    optim_objective!(mpc::PredictiveController)

Optimize the objective function ``J`` of `mpc` controller and return the solution `ΔŨ`.
"""
function optim_objective!(mpc::PredictiveController{NT}) where {NT<:Real}
    optim = mpc.optim
    model = mpc.estim.model
    ΔŨvar::Vector{VariableRef} = optim[:ΔŨvar]
    lastΔŨ = mpc.ΔŨ
    # initial ΔŨ (warm-start): [Δu_{k-1}(k); Δu_{k-1}(k+1); ... ; 0_{nu × 1}; ϵ_{k-1}]
    ϵ0  = !isinf(mpc.C) ? [lastΔŨ[end]] : empty(mpc.ΔŨ)
    ΔŨ0 = [lastΔŨ[(model.nu+1):(mpc.Hc*model.nu)]; zeros(NT, model.nu); ϵ0]
    set_start_value.(ΔŨvar, ΔŨ0)
    set_objective_linear_coef!(mpc, ΔŨvar)
    try
        optimize!(optim)
    catch err
        if isa(err, MOI.UnsupportedAttribute{MOI.VariablePrimalStart})
            # reset_optimizer to unset warm-start, set_start_value.(nothing) seems buggy
            MOIU.reset_optimizer(optim)
            optimize!(optim)
        else
            rethrow(err)
        end
    end
    ΔŨcurr, ΔŨlast = value.(ΔŨvar), ΔŨ0
    if !issolved(optim)
        status = termination_status(optim)
        if iserror(optim)
            @error("MPC terminated without solution: returning last solution shifted", 
                   status)
        else
            @warn("MPC termination status not OPTIMAL or LOCALLY_SOLVED: keeping "*
                  "solution anyway", status)
        end
        @debug solution_summary(optim, verbose=true)
    end
    mpc.ΔŨ[:] = iserror(optim) ? ΔŨlast : ΔŨcurr
    return mpc.ΔŨ
end

"By default, no need to modify the objective function."
set_objective_linear_coef!(::PredictiveController, _ ) = nothing

"""
    updatestate!(mpc::PredictiveController, u, ym, d=[]) -> x̂

Call [`updatestate!`](@ref) on `mpc.estim` [`StateEstimator`](@ref).
"""
updatestate!(mpc::PredictiveController, u, ym, d=empty(mpc.estim.x̂)) = updatestate!(mpc.estim,u,ym,d)
updatestate!(::PredictiveController, _ ) = throw(ArgumentError("missing measured outputs ym"))
