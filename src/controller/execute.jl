@doc raw"""
    initstate!(mpc::PredictiveController, u, ym, d=[]) -> x̂

Init the states of `mpc.estim` [`StateEstimator`](@ref) and warm start `mpc.ΔŨ` at zero.
"""
function initstate!(mpc::PredictiveController, u, ym, d=empty(mpc.estim.x̂0))
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
!!! info    
    Keyword arguments with *`emphasis`* are non-Unicode alternatives.

- `mpc::PredictiveController` : solve optimization problem of `mpc`.
- `ry=mpc.estim.model.yop` : current output setpoints ``\mathbf{r_y}(k)``.
- `d=[]` : current measured disturbances ``\mathbf{d}(k)``.
- `D̂=repeat(d, mpc.Hp)` or *`Dhat`* : predicted measured disturbances ``\mathbf{D̂}``, constant
   in the future by default or ``\mathbf{d̂}(k+j)=\mathbf{d}(k)`` for ``j=1`` to ``H_p``.
- `R̂y=repeat(ry, mpc.Hp)` or *`Rhaty`* : predicted output setpoints ``\mathbf{R̂_y}``, constant
   in the future by default or ``\mathbf{r̂_y}(k+j)=\mathbf{r_y}(k)`` for ``j=1`` to ``H_p``.
- `R̂u=repeat(mpc.estim.model.uop, mpc.Hp)` or *`Rhatu`* : predicted manipulated input
   setpoints, constant in the future by default or ``\mathbf{r̂_u}(k+j)=\mathbf{u_{op}}`` for
   ``j=0`` to ``H_p-1``. 
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
    d ::Vector = empty(mpc.estim.x̂0);
    Dhat ::Vector = repeat(d,  mpc.Hp),
    Rhaty::Vector = repeat(ry, mpc.Hp),
    Rhatu::Vector = mpc.noR̂u ? empty(mpc.estim.x̂0) : repeat(mpc.estim.model.uop, mpc.Hp),
    ym::Union{Vector, Nothing} = nothing,
    D̂  = Dhat,
    R̂y = Rhaty,
    R̂u = Rhatu
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

Get additional info about `mpc` [`PredictiveController`](@ref) optimum for troubleshooting.

The function should be called after calling [`moveinput!`](@ref). It returns the dictionary
`info` with the following fields:

!!! info
    Fields with *`emphasis`* are non-Unicode alternatives.

- `:ΔU` or *`:DeltaU`* : optimal manipulated input increments over ``H_c``, ``\mathbf{ΔU}``
- `:ϵ` or *`:epsilon`* : optimal slack variable, ``ϵ``
- `:D̂` or *`:Dhat`* : predicted measured disturbances over ``H_p``, ``\mathbf{D̂}``
- `:ŷ` or *`:yhat`* : current estimated output, ``\mathbf{ŷ}(k)``
- `:Ŷ` or *`:Yhat`* : optimal predicted outputs over ``H_p``, ``\mathbf{Ŷ}``
- `:Ŷs` or *`:Yhats`* : predicted stochastic output over ``H_p`` of [`InternalModel`](@ref), ``\mathbf{Ŷ_s}``
- `:R̂y` or *`:Rhaty`* : predicted output setpoint over ``H_p``, ``\mathbf{R̂_y}``
- `:R̂u` or *`:Rhatu`* : predicted manipulated input setpoint over ``H_p``, ``\mathbf{R̂_u}``
- `:x̂end` or *`:xhatend`* : optimal terminal states, ``\mathbf{x̂}_{k-1}(k+H_p)``
- `:J`   : objective value optimum, ``J``
- `:U`   : optimal manipulated inputs over ``H_p``, ``\mathbf{U}``
- `:u`   : current optimal manipulated input, ``\mathbf{u}(k)``
- `:d`   : current measured disturbance, ``\mathbf{d}(k)``

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
    model = mpc.estim.model
    info = Dict{Symbol, Union{JuMP._SolutionSummary, Vector{NT}, NT}}()
    Ŷ0, u0, û0  = similar(mpc.Yop), similar(model.uop), similar(model.uop)
    Ŷs          = similar(mpc.Yop)
    x̂0, x̂0next  = similar(mpc.estim.x̂0), similar(mpc.estim.x̂0)
    Ȳ, Ū        = similar(mpc.Yop), similar(mpc.Uop)
    Ŷ0, x̂0end = predict!(Ŷ0, x̂0, x̂0next, u0, û0, mpc, model, mpc.ΔŨ)
    U0 = mpc.S̃*mpc.ΔŨ + mpc.T_lastu0
    J  = obj_nonlinprog!(U0, Ȳ, Ū, mpc, model, Ŷ0, mpc.ΔŨ)
    oldF = copy(mpc.F)
    predictstoch!(mpc, mpc.estim, mpc.d0 + model.dop, mpc.ŷ[mpc.estim.i_ym]) 
    Ŷs .= mpc.F # predictstoch! init mpc.F with Ŷs value if estim is an InternalModel
    mpc.F .= oldF  # restore old F value
    info[:ΔU]   = mpc.ΔŨ[1:mpc.Hc*model.nu]
    info[:ϵ]    = mpc.nϵ == 1 ? mpc.ΔŨ[end] : NaN
    info[:J]    = J
    info[:U]    = U0 + mpc.Uop
    info[:u]    = info[:U][1:model.nu]
    info[:d]    = mpc.d0 + model.dop
    info[:D̂]    = mpc.D̂0 + mpc.Dop
    info[:ŷ]    = mpc.ŷ
    info[:Ŷ]    = Ŷ0 + mpc.Yop
    info[:x̂end] = x̂0end + mpc.estim.x̂op
    info[:Ŷs]   = Ŷs
    info[:R̂y]   = mpc.R̂y0 + mpc.Yop
    info[:R̂u]   = mpc.R̂u0 + mpc.Uop
    # --- non-Unicode fields ---
    info[:DeltaU] = info[:ΔU]
    info[:epsilon] = info[:ϵ]
    info[:Dhat] = info[:D̂]
    info[:yhat] = info[:ŷ]
    info[:Yhat] = info[:Ŷ]
    info[:xhatend] = info[:x̂end]
    info[:Yhats] = info[:Ŷs]
    info[:Rhaty] = info[:R̂y]
    info[:Rhatu] = info[:R̂u]
    info = addinfo!(info, mpc)
    return info
end

"""
    addinfo!(info, mpc::PredictiveController) -> info

By default, add the solution summary `:sol` that can be printed to `info`.
"""
function addinfo!(info, mpc::PredictiveController)
    info[:sol] = JuMP.solution_summary(mpc.optim, verbose=true)
    return info
end


@doc raw"""
    initpred!(mpc::PredictiveController, model::LinModel, d, ym, D̂, R̂y, R̂u) -> nothing

Init linear model prediction matrices `F, q̃, p` and current estimated output `ŷ`.

See [`init_predmat`](@ref) and [`init_quadprog`](@ref) for the definition of the matrices.
They are computed with these equations using in-place operations:
```math
\begin{aligned}
    \mathbf{F}       &= \mathbf{G d_0}(k) + \mathbf{J D̂_0} + \mathbf{K x̂_0}(k) 
                            + \mathbf{V u_0}(k-1) + \mathbf{B} + \mathbf{Ŷ_s}           \\
    \mathbf{C_y}     &= \mathbf{F}                 - (\mathbf{R̂_y - Y_{op}})            \\
    \mathbf{C_u}     &= \mathbf{T} \mathbf{u_0}(k-1) - (\mathbf{R̂_u - U_{op}})          \\
    \mathbf{q̃}       &= 2[(\mathbf{M}_{H_p} \mathbf{Ẽ})' \mathbf{C_y} 
                            + (\mathbf{L}_{H_p} \mathbf{S̃})' \mathbf{C_u}]              \\
    p                &= \mathbf{C_y}' \mathbf{M}_{H_p} \mathbf{C_y} 
                            + \mathbf{C_u}' \mathbf{L}_{H_p} \mathbf{C_u}
\end{aligned}
```
"""
function initpred!(mpc::PredictiveController, model::LinModel, d, ym, D̂, R̂y, R̂u)
    mul!(mpc.T_lastu0, mpc.T, mpc.estim.lastu0)
    ŷ, F, q̃, p = mpc.ŷ, mpc.F, mpc.q̃, mpc.p
    ŷ .= evalŷ(mpc.estim, ym, d)
    predictstoch!(mpc, mpc.estim, d, ym) # init mpc.F with Ŷs for InternalModel
    F .+= mpc.B
    mul!(F, mpc.K, mpc.estim.x̂0, 1, 1) 
    mul!(F, mpc.V, mpc.estim.lastu0, 1, 1)
    if model.nd ≠ 0
        mpc.d0 .= d .- model.dop
        mpc.D̂0 .= D̂ .- mpc.Dop
        mpc.D̂E[1:model.nd]     .= d
        mpc.D̂E[model.nd+1:end] .= D̂
        mul!(F, mpc.G, mpc.d0, 1, 1)
        mul!(F, mpc.J, mpc.D̂0, 1, 1)
    end
    mpc.R̂y0 .= R̂y .- mpc.Yop
    Cy = F .- mpc.R̂y0
    M_Hp_Ẽ = mpc.M_Hp*mpc.Ẽ
    mul!(q̃, M_Hp_Ẽ', Cy)
    p .= dot(Cy, mpc.M_Hp, Cy)
    if ~mpc.noR̂u
        mpc.R̂u0 .= R̂u .- mpc.Uop
        Cu = mpc.T_lastu0 .- mpc.R̂u0
        L_Hp_S̃ = mpc.L_Hp*mpc.S̃
        mul!(q̃, L_Hp_S̃', Cu, 1, 1)
        p .+= dot(Cu, mpc.L_Hp, Cu)
    end
    lmul!(2, q̃)
    return nothing
end

@doc raw"""
    initpred!(mpc::PredictiveController, model::SimModel, d, ym, D̂, R̂y, R̂u)

Init `ŷ, F, d0, D̂0, D̂E, R̂y0, R̂u0` vectors when model is not a [`LinModel`](@ref).
"""
function initpred!(mpc::PredictiveController, model::SimModel, d, ym, D̂, R̂y, R̂u)
    mul!(mpc.T_lastu0, mpc.T, mpc.estim.lastu0)
    mpc.ŷ .= evalŷ(mpc.estim, ym, d)
    predictstoch!(mpc, mpc.estim, d, ym) # init mpc.F with Ŷs for InternalModel
    if model.nd ≠ 0
        mpc.d0 .= d .- model.dop
        mpc.D̂0 .= D̂ .- mpc.Dop
        mpc.D̂E[1:model.nd]     .= d
        mpc.D̂E[model.nd+1:end] .= D̂
    end
    mpc.R̂y0 .= (R̂y .- mpc.Yop)
    if ~mpc.noR̂u
        mpc.R̂u0 .= (R̂u .- mpc.Uop)
    end
    return nothing
end

@doc raw"""
    predictstoch!(mpc::PredictiveController, estim::InternalModel, x̂s, d, ym)

Init `mpc.F` vector with ``\mathbf{F = Ŷ_s}`` when `estim` is an [`InternalModel`](@ref).
"""
function predictstoch!(
    mpc::PredictiveController{NT}, estim::InternalModel, d, ym
) where {NT<:Real}
    isnothing(ym) && error("Predictive controllers with InternalModel need the measured "*
                           "outputs ym in keyword argument to compute control actions u")
    F, ny = mpc.F, estim.model.ny
    ŷd = similar(estim.model.yop)
    h!(ŷd, estim.model, estim.x̂d, d - estim.model.dop)
    ŷd .+= estim.model.yop 
    ŷs = zeros(NT, estim.model.ny)
    ŷs[estim.i_ym] .= @views ym .- ŷd[estim.i_ym]  # ŷs=0 for unmeasured outputs
    Ŷs = F
    mul!(Ŷs, mpc.Ks, estim.x̂s)
    mul!(Ŷs, mpc.Ps, ŷs, 1, 1)
    return nothing
end
"Separate stochastic predictions are not needed if `estim` is not [`InternalModel`](@ref)."
predictstoch!(mpc::PredictiveController, ::StateEstimator, _ , _ ) = (mpc.F .= 0; nothing)

@doc raw"""
    linconstraint!(mpc::PredictiveController, model::LinModel)

Set `b` vector for the linear model inequality constraints (``\mathbf{A ΔŨ ≤ b}``).

Also init ``\mathbf{f_x̂} = \mathbf{g_x̂ d}(k) + \mathbf{j_x̂ D̂} + \mathbf{k_x̂ x̂}_{k-1}(k) + \mathbf{v_x̂ u}(k-1) + \mathbf{b_x̂}``
vector for the terminal constraints, see [`init_predmat`](@ref).
"""
function linconstraint!(mpc::PredictiveController, model::LinModel)
    nU, nΔŨ, nY = length(mpc.con.U0min), length(mpc.con.ΔŨmin), length(mpc.con.Y0min)
    nx̂, fx̂ = mpc.estim.nx̂, mpc.con.fx̂
    fx̂ .= mpc.con.bx̂
    mul!(fx̂, mpc.con.kx̂, mpc.estim.x̂0, 1, 1)
    mul!(fx̂, mpc.con.vx̂, mpc.estim.lastu0, 1, 1)
    if model.nd ≠ 0
        mul!(fx̂, mpc.con.gx̂, mpc.d0, 1, 1)
        mul!(fx̂, mpc.con.jx̂, mpc.D̂0, 1, 1)
    end
    n = 0
    mpc.con.b[(n+1):(n+nU)]  .= @. -mpc.con.U0min + mpc.T_lastu0
    n += nU
    mpc.con.b[(n+1):(n+nU)]  .= @. +mpc.con.U0max - mpc.T_lastu0
    n += nU
    mpc.con.b[(n+1):(n+nΔŨ)] .= @. -mpc.con.ΔŨmin
    n += nΔŨ
    mpc.con.b[(n+1):(n+nΔŨ)] .= @. +mpc.con.ΔŨmax
    n += nΔŨ
    mpc.con.b[(n+1):(n+nY)]  .= @. -mpc.con.Y0min + mpc.F
    n += nY
    mpc.con.b[(n+1):(n+nY)]  .= @. +mpc.con.Y0max - mpc.F
    n += nY
    mpc.con.b[(n+1):(n+nx̂)]  .= @. -mpc.con.x̂0min + fx̂
    n += nx̂
    mpc.con.b[(n+1):(n+nx̂)]  .= @. +mpc.con.x̂0max - fx̂
    if any(mpc.con.i_b) 
        lincon = mpc.optim[:linconstraint]
        JuMP.set_normalized_rhs(lincon, mpc.con.b[mpc.con.i_b])
    end
    return nothing
end

"Set `b` excluding predicted output constraints when `model` is not a [`LinModel`](@ref)."
function linconstraint!(mpc::PredictiveController, ::SimModel)
    nU, nΔŨ = length(mpc.con.U0min), length(mpc.con.ΔŨmin)
    n = 0
    mpc.con.b[(n+1):(n+nU)]  .= @. -mpc.con.U0min + mpc.T_lastu0
    n += nU
    mpc.con.b[(n+1):(n+nU)]  .= @. +mpc.con.U0max - mpc.T_lastu0
    n += nU
    mpc.con.b[(n+1):(n+nΔŨ)] .= @. -mpc.con.ΔŨmin
    n += nΔŨ
    mpc.con.b[(n+1):(n+nΔŨ)] .= @. +mpc.con.ΔŨmax
    if any(mpc.con.i_b) 
        lincon = mpc.optim[:linconstraint]
        JuMP.set_normalized_rhs(lincon, mpc.con.b[mpc.con.i_b])
    end
    return nothing
end

@doc raw"""
    predict!(Ŷ0, x̂0, _, _, _, mpc::PredictiveController, model::LinModel, ΔŨ) -> Ŷ0, x̂0end

Compute the predictions `Ŷ0` and terminal states `x̂0end` if model is a [`LinModel`](@ref).

The method mutates `Ŷ0` and `x̂0` vector arguments. The `x̂end` vector is used for
the terminal constraints applied on ``\mathbf{x̂}_{k-1}(k+H_p)``.
"""
function predict!(Ŷ0, x̂0, _ , _ , _ , mpc::PredictiveController, ::LinModel, ΔŨ)
    # in-place operations to reduce allocations :
    Ŷ0 .= mul!(Ŷ0, mpc.Ẽ, ΔŨ) .+ mpc.F
    x̂0 .= mul!(x̂0, mpc.con.ẽx̂, ΔŨ) .+ mpc.con.fx̂
    x̂0end = x̂0
    return Ŷ0, x̂0end
end

@doc raw"""
    predict!(
        Ŷ0, x̂0, x̂0next, u0, û0, mpc::PredictiveController, model::SimModel, ΔŨ
    ) -> Ŷ0, x̂end

Compute both vectors if `model` is not a [`LinModel`](@ref). 
    
The method mutates `Ŷ0`, `x̂0`, `x̂0next`, `u0` and `û0` arguments.
"""
function predict!(Ŷ0, x̂0, x̂0next, u0, û0, mpc::PredictiveController, model::SimModel, ΔŨ)
    nu, ny, nd, Hp, Hc = model.nu, model.ny, model.nd, mpc.Hp, mpc.Hc
    x̂0 .= mpc.estim.x̂0
    u0 .= mpc.estim.lastu0
    d0  = @views mpc.d0[1:end]
    for j=1:Hp
        if j ≤ Hc
            u0 .+= @views ΔŨ[(1 + nu*(j-1)):(nu*j)]
        end
        f̂!(x̂0next, û0, mpc.estim, model, x̂0, u0, d0)
        x̂0next .+= mpc.estim.f̂op .- mpc.estim.x̂op
        x̂0 .= x̂0next
        d0 = @views mpc.D̂0[(1 + nd*(j-1)):(nd*j)]
        ŷ0 = @views Ŷ0[(1 + ny*(j-1)):(ny*j)]
        ĥ!(ŷ0, mpc.estim, model, x̂0, d0)
    end
    Ŷ0 .+= mpc.F # F = Ŷs if mpc.estim is an InternalModel, else F = 0.
    x̂end = x̂0
    return Ŷ0, x̂end
end

"""
    obj_nonlinprog!(U0 , Ȳ, _ , mpc::PredictiveController, model::LinModel, Ŷ0, ΔŨ)

Nonlinear programming objective function when `model` is a [`LinModel`](@ref).

The function is called by the nonlinear optimizer of [`NonLinMPC`](@ref) controllers. It can
also be called on any [`PredictiveController`](@ref)s to evaluate the objective function `J`
at specific input increments `ΔŨ` and predictions `Ŷ0` values. It mutates the `U0` and
`Ȳ` arguments.
"""
function obj_nonlinprog!(
    U0, Ȳ, _ , mpc::PredictiveController, model::LinModel, Ŷ0, ΔŨ::AbstractVector{NT}
) where NT <: Real
    J = obj_quadprog(ΔŨ, mpc.H̃, mpc.q̃) + mpc.p[]
    if !iszero(mpc.E)
        ny, Hp, ŷ, D̂E = model.ny, mpc.Hp, mpc.ŷ, mpc.D̂E
        U = U0
        U  .+= mpc.Uop
        uend = @views U[(end-model.nu+1):end]
        Ŷ  = Ȳ
        Ŷ .= Ŷ0 .+ mpc.Yop
        UE = [U; uend]
        ŶE = [ŷ; Ŷ]
        E_JE = mpc.E*mpc.JE(UE, ŶE, D̂E)
    else
        E_JE = 0.0
    end
    return J + E_JE
end

"""
    obj_nonlinprog!(U0, Ȳ, Ū, mpc::PredictiveController, model::SimModel, Ŷ0, ΔŨ)

Nonlinear programming objective function when `model` is not a [`LinModel`](@ref). The
function `dot(x, A, x)` is a performant way of calculating `x'*A*x`. This method mutates
`U0`, `Ȳ` and `Ū` arguments (input over `Hp`, and output and input setpoint tracking error, 
respectively).
"""
function obj_nonlinprog!(
    U0, Ȳ, Ū, mpc::PredictiveController, model::SimModel, Ŷ0, ΔŨ::AbstractVector{NT}
) where NT<:Real
    # --- output setpoint tracking term ---
    Ȳ  .= mpc.R̂y0 .- Ŷ0
    JR̂y = dot(Ȳ, mpc.M_Hp, Ȳ)
    # --- move suppression and slack variable term ---
    JΔŨ = dot(ΔŨ, mpc.Ñ_Hc, ΔŨ)
    # --- input over prediction horizon ---
    if !mpc.noR̂u || !iszero(mpc.E)
        U0 .= mul!(U0, mpc.S̃, ΔŨ) .+ mpc.T_lastu0
    end
    # --- input setpoint tracking term ---
    if !mpc.noR̂u
        Ū  .= mpc.R̂u0 .- U0
        JR̂u = dot(Ū, mpc.L_Hp, Ū)
    else
        JR̂u = 0.0
    end
    # --- economic term ---
    if !iszero(mpc.E)
        ny, Hp, ŷ, D̂E = model.ny, mpc.Hp, mpc.ŷ, mpc.D̂E
        U = U0
        U  .+= mpc.Uop
        uend = @views U[(end-model.nu+1):end]
        Ŷ  = Ȳ
        Ŷ .= Ŷ0 .+ mpc.Yop
        UE = [U; uend]
        ŶE = [ŷ; Ŷ]
        E_JE = mpc.E*mpc.JE(UE, ŶE, D̂E)
    else
        E_JE = 0.0
    end
    return JR̂y + JΔŨ + JR̂u + E_JE
end

@doc raw"""
    optim_objective!(mpc::PredictiveController) -> ΔŨ

Optimize the objective function of `mpc` [`PredictiveController`](@ref) and return the solution `ΔŨ`.

If supported by `mpc.optim`, it warm-starts the solver at:
```math
\mathbf{ΔŨ} = 
\begin{bmatrix}
    \mathbf{Δu}_{k-1}(k+0)      \\ 
    \mathbf{Δu}_{k-1}(k+1)      \\ 
    \vdots                      \\
    \mathbf{Δu}_{k-1}(k+H_c-2)  \\
    \mathbf{0}                  \\
    ϵ_{k-1}
\end{bmatrix}
```
where ``\mathbf{Δu}_{k-1}(k+j)`` is the input increment for time ``k+j`` computed at the 
last control period ``k-1``. It then calls `JuMP.optimize!(mpc.optim)` and extract the
solution. A failed optimization prints an `@error` log in the REPL and returns the 
warm-start value.
"""
function optim_objective!(mpc::PredictiveController{NT}) where {NT<:Real}
    optim = mpc.optim
    model = mpc.estim.model
    ΔŨvar::Vector{JuMP.VariableRef} = optim[:ΔŨvar]
    # initial ΔŨ (warm-start): [Δu_{k-1}(k); Δu_{k-1}(k+1); ... ; 0_{nu × 1}; ϵ_{k-1}]
    ϵ0  = (mpc.nϵ == 1) ? mpc.ΔŨ[end] : empty(mpc.ΔŨ)
    ΔŨ0 = [mpc.ΔŨ[(model.nu+1):(mpc.Hc*model.nu)]; zeros(NT, model.nu); ϵ0]
    JuMP.set_start_value.(ΔŨvar, ΔŨ0)
    set_objective_linear_coef!(mpc, ΔŨvar)
    try
        JuMP.optimize!(optim)
    catch err
        if isa(err, MOI.UnsupportedAttribute{MOI.VariablePrimalStart})
            # reset_optimizer to unset warm-start, set_start_value.(nothing) seems buggy
            MOIU.reset_optimizer(optim)
            JuMP.optimize!(optim)
        else
            rethrow(err)
        end
    end
    if !issolved(optim)
        status = JuMP.termination_status(optim)
        if iserror(optim)
            @error("MPC terminated without solution: returning last solution shifted", 
                   status)
        else
            @warn("MPC termination status not OPTIMAL or LOCALLY_SOLVED: keeping "*
                  "solution anyway", status)
        end
        @debug JuMP.solution_summary(optim, verbose=true)
    end
    if iserror(optim)
        mpc.ΔŨ .= ΔŨ0
    else
        mpc.ΔŨ .= JuMP.value.(ΔŨvar)
    end
    return mpc.ΔŨ
end

"By default, no need to modify the objective function."
set_objective_linear_coef!(::PredictiveController, _ ) = nothing

"""
    updatestate!(mpc::PredictiveController, u, ym, d=[]) -> x̂

Call [`updatestate!`](@ref) on `mpc.estim` [`StateEstimator`](@ref).
"""
function updatestate!(mpc::PredictiveController, u, ym, d=empty(mpc.estim.x̂0))
    return updatestate!(mpc.estim, u, ym, d)
end
updatestate!(::PredictiveController, _ ) = throw(ArgumentError("missing measured outputs ym"))

"""
    setstate!(mpc::PredictiveController, x̂) -> mpc

Set `mpc.estim.x̂0` to `x̂ - estim.x̂op` from the argument `x̂`.
"""
setstate!(mpc::PredictiveController, x̂) = (setstate!(mpc.estim, x̂); return mpc)


@doc raw"""
    setmodel!(mpc::PredictiveController, model=mpc.estim.model, <keyword arguments>) -> mpc

Set `model` and objective function weights of `mpc` [`PredictiveController`](@ref).

Allows model adaptation of controllers based on [`LinModel`](@ref) at runtime. Modification
of [`NonLinModel`](@ref) is not supported. New weight matrices in the objective function can
be specified with the keyword arguments (see [`LinMPC`](@ref) for the nomenclature). If 
`Cwt ≠ Inf`, the augmented move suppression weight is ``\mathbf{Ñ}_{H_c} = \mathrm{diag}(
\mathbf{N}_{H_c}, C)``, else ``\mathbf{Ñ}_{H_c} = \mathbf{N}_{H_c}``. The [`StateEstimator`](@ref)
`mpc.estim` cannot be a [`Luenberger`](@ref) observer or a [`SteadyKalmanFilter`](@ref) (the
default estimator). Construct the `mpc` object with a time-varying [`KalmanFilter`](@ref)
instead. Note that the model is constant over the prediction horizon ``H_p``.

# Arguments

- `mpc::PredictiveController` : controller to set model and weights.
- `model=mpc.estim.model` : new plant model ([`NonLinModel`](@ref) not supported).
- `M_Hp=mpc.M_Hp` : new ``\mathbf{M_{H_p}}`` weight matrix.
- `Ñ_Hc=mpc.Ñ_Hc` : new ``\mathbf{Ñ_{H_c}}`` weight matrix (see definition above).
- `L_Hp=mpc.L_Hp` : new ``\mathbf{L_{H_p}}`` weight matrix.
- additional keyword arguments are passed to `setmodel!(::StateEstimator)`.

# Examples
```jldoctest
julia> mpc = LinMPC(KalmanFilter(LinModel(ss(0.1, 0.5, 1, 0, 4.0)), σR=[√25]), Hp=1, Hc=1);

julia> mpc.estim.model.A[], mpc.estim.R̂[], mpc.M_Hp[]
(0.1, 25.0)

julia> setmodel!(mpc, LinModel(ss(0.42, 0.5, 1, 0, 4.0)); R̂=[9], M_Hp=[0]); 

julia> mpc.estim.model.A[], mpc.estim.R̂[], mpc.M_Hp[]
(0.42, 9.0)
```
"""
function setmodel!(
        mpc::PredictiveController, 
        model = mpc.estim.model;
        M_Hp = mpc.M_Hp,
        Ñ_Hc = mpc.Ñ_Hc,
        L_Hp = mpc.L_Hp,
        kwargs...
    )
    x̂op_old = copy(mpc.estim.x̂op)
    nu, ny, Hp, Hc, nϵ = model.nu, model.ny, mpc.Hp, mpc.Hc, mpc.nϵ
    setmodel!(mpc.estim, model; kwargs...)
    mpc.M_Hp .= to_hermitian(M_Hp)
    mpc.Ñ_Hc .= to_hermitian(Ñ_Hc)
    mpc.L_Hp .= to_hermitian(L_Hp)
    setmodel_controller!(mpc, x̂op_old, M_Hp, Ñ_Hc, L_Hp)
    return mpc
end

"Update the prediction matrices, linear constraints and JuMP optimization."
function setmodel_controller!(mpc::PredictiveController, x̂op_old, M_Hp, Ñ_Hc, L_Hp)
    estim, model = mpc.estim, mpc.estim.model
    nu, ny, nd, Hp, Hc = model.nu, model.ny, model.nd, mpc.Hp, mpc.Hc
    optim, con = mpc.optim, mpc.con
    # --- predictions matrices ---
    E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂ = init_predmat(estim, model, Hp, Hc)
    A_Ymin, A_Ymax, Ẽ = relaxŶ(model, mpc.nϵ, con.C_ymin, con.C_ymax, E)
    A_x̂min, A_x̂max, ẽx̂ = relaxterminal(model, mpc.nϵ, con.c_x̂min, con.c_x̂max, ex̂)
    mpc.Ẽ .= Ẽ
    mpc.G .= G
    mpc.J .= J
    mpc.K .= K
    mpc.V .= V
    mpc.B .= B
    # --- linear inequality constraints ---
    con.ẽx̂ .= ẽx̂ 
    con.gx̂ .= gx̂
    con.jx̂ .= jx̂
    con.kx̂ .= kx̂
    con.vx̂ .= vx̂
    con.bx̂ .= bx̂
    con.U0min .+= mpc.Uop # convert U0 to U with the old operating point
    con.U0max .+= mpc.Uop # convert U0 to U with the old operating point
    con.Y0min .+= mpc.Yop # convert Y0 to Y with the old operating point
    con.Y0max .+= mpc.Yop # convert Y0 to Y with the old operating point
    con.x̂0min .+= x̂op_old # convert x̂0 to x̂ with the old operating point
    con.x̂0max .+= x̂op_old # convert x̂0 to x̂ with the old operating point
    # --- operating points ---
    for i in 0:Hp-1
        mpc.Uop[(1+nu*i):(nu+nu*i)] .= model.uop
        mpc.Yop[(1+ny*i):(ny+ny*i)] .= model.yop
        mpc.Dop[(1+nd*i):(nd+nd*i)] .= model.dop
    end
    con.U0min .-= mpc.Uop # convert U to U0 with the new operating point
    con.U0max .-= mpc.Uop # convert U to U0 with the new operating point
    con.Y0min .-= mpc.Yop # convert Y to Y0 with the new operating point
    con.Y0max .-= mpc.Yop # convert Y to Y0 with the new operating point
    con.x̂0min .-= estim.x̂op # convert x̂ to x̂0 with the new operating point
    con.x̂0max .-= estim.x̂op # convert x̂ to x̂0 with the new operating point
    con.A_Ymin .= A_Ymin
    con.A_Ymax .= A_Ymax
    con.A_x̂min .= A_x̂min
    con.A_x̂max .= A_x̂max
    nUandΔŨ = length(con.U0min) + length(con.U0max) + length(con.ΔŨmin) + length(con.ΔŨmax)
    con.A[nUandΔŨ+1:end, :] = [con.A_Ymin; con.A_Ymax; con.A_x̂min; con.A_x̂max]
    A = con.A[con.i_b, :]
    b = con.b[con.i_b]
    ΔŨvar::Vector{JuMP.VariableRef} = optim[:ΔŨvar]
    # deletion is required for sparse solvers like OSQP, when the sparsity pattern changes
    JuMP.delete(optim, optim[:linconstraint])
    JuMP.unregister(optim, :linconstraint)
    @constraint(optim, linconstraint, A*ΔŨvar .≤ b)
    # --- quadratic programming Hessian matrix ---
    H̃ = init_quadprog(model, mpc.Ẽ, mpc.S̃, mpc.M_Hp, mpc.Ñ_Hc, mpc.L_Hp)
    mpc.H̃ .= H̃
    set_objective_hessian!(mpc, ΔŨvar)
    return nothing
end

"No need to set the objective Hessian by default (only needed for quadratic optimization)."
set_objective_hessian!(::PredictiveController, _ ) = nothing