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

Get additional info about `mpc` [`PredictiveController`](@ref) optimum for troubleshooting.

The function should be called after calling [`moveinput!`](@ref). It returns the dictionary
`info` with the following fields:

- `:ΔU`  : optimal manipulated input increments over ``H_c``, ``\mathbf{ΔU}``
- `:ϵ`   : optimal slack variable, ``ϵ``
- `:J`   : objective value optimum, ``J``
- `:U`   : optimal manipulated inputs over ``H_p``, ``\mathbf{U}``
- `:u`   : current optimal manipulated input, ``\mathbf{u}(k)``
- `:d`   : current measured disturbance, ``\mathbf{d}(k)``
- `:D̂`   : predicted measured disturbances over ``H_p``, ``\mathbf{D̂}``
- `:ŷ`   : current estimated output, ``\mathbf{ŷ}(k)``
- `:Ŷ`   : optimal predicted outputs over ``H_p``, ``\mathbf{Ŷ}``
- `:x̂end`: optimal terminal states, ``\mathbf{x̂}_{k-1}(k+H_p)``
- `:Ŷs`  : predicted stochastic output over ``H_p`` of [`InternalModel`](@ref), ``\mathbf{Ŷ_s}``
- `:R̂y`  : predicted output setpoint over ``H_p``, ``\mathbf{R̂_y}``
- `:R̂u`  : predicted manipulated input setpoint over ``H_p``, ``\mathbf{R̂_u}``

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
    Ŷ, u, û     = similar(mpc.Ŷop), similar(model.uop), similar(model.uop)
    x̂, x̂next    = similar(mpc.estim.x̂), similar(mpc.estim.x̂)
    Ŷ, x̂end     = predict!(Ŷ, x̂, x̂next, u, û, mpc, model, mpc.ΔŨ)
    U           = mpc.S̃*mpc.ΔŨ + mpc.T_lastu
    Ȳ, Ū        = similar(Ŷ), similar(U)
    J           = obj_nonlinprog!(U, Ȳ, Ū, mpc, model, Ŷ, mpc.ΔŨ)
    info[:ΔU]   = mpc.ΔŨ[1:mpc.Hc*model.nu]
    info[:ϵ]    = isinf(mpc.C) ? NaN : mpc.ΔŨ[end]
    info[:J]    = J
    info[:U]    = U
    info[:u]    = info[:U][1:model.nu]
    info[:d]    = mpc.d0 + model.dop
    info[:D̂]    = mpc.D̂0 + mpc.Dop
    info[:ŷ]    = mpc.ŷ
    info[:Ŷ]    = Ŷ
    info[:x̂end] = x̂end
    info[:Ŷs]  = mpc.Ŷop - repeat(model.yop, mpc.Hp) # Ŷop = Ŷs + Yop
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
    initpred!(mpc, model::LinModel, d, ym, D̂, R̂y, R̂u) -> nothing

Init linear model prediction matrices `F, q̃, p` and current estimated output `ŷ`.

See [`init_predmat`](@ref) and [`init_quadprog`](@ref) for the definition of the matrices.
They are computed with these equations using in-place operations:
```math
\begin{aligned}
    \mathbf{F}   &= \mathbf{G d}(k) + \mathbf{J D̂} + \mathbf{K x̂}_{k-1}(k) + \mathbf{V u}(k-1) \\
    \mathbf{C_y} &= \mathbf{F}                 - \mathbf{R̂_y} \\
    \mathbf{C_u} &= \mathbf{T} \mathbf{u}(k-1) - \mathbf{R̂_u} \\
    \mathbf{q̃}   &= 2[(\mathbf{M}_{H_p} \mathbf{Ẽ})' \mathbf{C_y} + (\mathbf{L}_{H_p} \mathbf{S̃})' \mathbf{C_u}] \\
            p    &= \mathbf{C_y}' \mathbf{M}_{H_p} \mathbf{C_y} + \mathbf{C_u}' \mathbf{L}_{H_p} \mathbf{C_u}
\end{aligned}
```
"""
function initpred!(mpc::PredictiveController, model::LinModel, d, ym, D̂, R̂y, R̂u)
    mul!(mpc.T_lastu, mpc.T, mpc.estim.lastu0 .+ model.uop)
    ŷ, F, q̃, p = mpc.ŷ, mpc.F, mpc.q̃, mpc.p
    ŷ .= evalŷ(mpc.estim, ym, d)
    predictstoch!(mpc, mpc.estim, d, ym) # init mpc.Ŷop for InternalModel
    F .= mpc.Ŷop
    mul!(F, mpc.K, mpc.estim.x̂, 1, 1) 
    mul!(F, mpc.V, mpc.estim.lastu0, 1, 1)
    if model.nd ≠ 0
        mpc.d0 .= d .- model.dop
        mpc.D̂0 .= D̂ .- mpc.Dop
        mpc.D̂E[1:model.nd]     .= mpc.d0
        mpc.D̂E[model.nd+1:end] .= mpc.D̂0
        mul!(F, mpc.G, mpc.d0, 1, 1)
        mul!(F, mpc.J, mpc.D̂0, 1, 1)
    end
    mpc.R̂y .= R̂y
    Cy = F .- mpc.R̂y
    M_Hp_Ẽ = mpc.M_Hp*mpc.Ẽ
    mul!(q̃, M_Hp_Ẽ', Cy)
    p .= dot(Cy, mpc.M_Hp, Cy)
    if ~mpc.noR̂u
        mpc.R̂u .= R̂u
        Cu = mpc.T_lastu .- mpc.R̂u
        L_Hp_S̃ = mpc.L_Hp*mpc.S̃
        mul!(q̃, L_Hp_S̃', Cu, 1, 1)
        p .+= dot(Cu, mpc.L_Hp, Cu)
    end
    lmul!(2, q̃)
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
    mul!(mpc.T_lastu, mpc.T, mpc.estim.lastu0 .+ model.uop)
    mpc.ŷ .= evalŷ(mpc.estim, ym, d)
    predictstoch!(mpc, mpc.estim, d, ym) # init mpc.Ŷop for InternalModel
    if model.nd ≠ 0
        mpc.d0 .= d .- model.dop
        mpc.D̂0 .= D̂ .- mpc.Dop
        mpc.D̂E[1:model.nd]     .= mpc.d0
        mpc.D̂E[model.nd+1:end] .= mpc.D̂0
    end
    mpc.R̂y .= R̂y
    if ~mpc.noR̂u
        mpc.R̂u .= R̂u
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
    Ŷop, ny, yop = mpc.Ŷop, estim.model.ny, estim.model.yop
    ŷd = similar(estim.model.yop)
    h!(ŷd, estim.model, estim.x̂d, d - estim.model.dop)
    ŷd .+= estim.model.yop 
    ŷs = zeros(NT, estim.model.ny)
    ŷs[estim.i_ym] .= @views ym .- ŷd[estim.i_ym]  # ŷs=0 for unmeasured outputs
    for j=1:mpc.Hp
        Ŷop[(1 + ny*(j-1)):(ny*j)] .= yop
    end
    mul!(Ŷop, mpc.Ks, estim.x̂s, 1, 1)
    mul!(Ŷop, mpc.Ps, ŷs, 1, 1)
    return nothing
end
"Separate stochastic predictions are not needed if `estim` is not [`InternalModel`](@ref)."
predictstoch!(::PredictiveController, ::StateEstimator, _ , _ ) = nothing

@doc raw"""
    linconstraint!(mpc::PredictiveController, model::LinModel)

Set `b` vector for the linear model inequality constraints (``\mathbf{A ΔŨ ≤ b}``).

Also init ``\mathbf{f_x̂} = \mathbf{g_x̂ d}(k) + \mathbf{j_x̂ D̂} + \mathbf{k_x̂ x̂}_{k-1}(k) + \mathbf{v_x̂ u}(k-1)``
vector for the terminal constraints, see [`init_predmat`](@ref).
"""
function linconstraint!(mpc::PredictiveController, model::LinModel)
    nU, nΔŨ, nY = length(mpc.con.Umin), length(mpc.con.ΔŨmin), length(mpc.con.Ymin)
    nx̂ = mpc.estim.nx̂
    fx̂ = mpc.con.fx̂
    mul!(fx̂, mpc.con.kx̂, mpc.estim.x̂)
    mul!(fx̂, mpc.con.vx̂, mpc.estim.lastu0, 1, 1)
    if model.nd ≠ 0
        mul!(fx̂, mpc.con.gx̂, mpc.d0, 1, 1)
        mul!(fx̂, mpc.con.jx̂, mpc.D̂0, 1, 1)
    end
    n = 0
    mpc.con.b[(n+1):(n+nU)]  .= @. -mpc.con.Umin + mpc.T_lastu
    n += nU
    mpc.con.b[(n+1):(n+nU)]  .= @. +mpc.con.Umax - mpc.T_lastu
    n += nU
    mpc.con.b[(n+1):(n+nΔŨ)] .= @. -mpc.con.ΔŨmin
    n += nΔŨ
    mpc.con.b[(n+1):(n+nΔŨ)] .= @. +mpc.con.ΔŨmax
    n += nΔŨ
    mpc.con.b[(n+1):(n+nY)]  .= @. -mpc.con.Ymin + mpc.F
    n += nY
    mpc.con.b[(n+1):(n+nY)]  .= @. +mpc.con.Ymax - mpc.F
    n += nY
    mpc.con.b[(n+1):(n+nx̂)]  .= @. -mpc.con.x̂min + fx̂
    n += nx̂
    mpc.con.b[(n+1):(n+nx̂)]  .= @. +mpc.con.x̂max - fx̂
    lincon = mpc.optim[:linconstraint]
    set_normalized_rhs(lincon, mpc.con.b[mpc.con.i_b])
end

"Set `b` excluding predicted output constraints when `model` is not a [`LinModel`](@ref)."
function linconstraint!(mpc::PredictiveController, ::SimModel)
    nU, nΔŨ = length(mpc.con.Umin), length(mpc.con.ΔŨmin)
    n = 0
    mpc.con.b[(n+1):(n+nU)]  .= @. -mpc.con.Umin + mpc.T_lastu
    n += nU
    mpc.con.b[(n+1):(n+nU)]  .= @. +mpc.con.Umax - mpc.T_lastu
    n += nU
    mpc.con.b[(n+1):(n+nΔŨ)] .= @. -mpc.con.ΔŨmin
    n += nΔŨ
    mpc.con.b[(n+1):(n+nΔŨ)] .= @. +mpc.con.ΔŨmax
    lincon = mpc.optim[:linconstraint]
    set_normalized_rhs.(lincon, mpc.con.b[mpc.con.i_b])
end

@doc raw"""
    predict!(Ŷ, x̂, _ , _ , _ , mpc::PredictiveController, model::LinModel, ΔŨ) -> Ŷ, x̂end

Compute the predictions `Ŷ` and terminal states `x̂end` if model is a [`LinModel`](@ref).

The method mutates `Ŷ` and `x̂` vector arguments. The `x̂end` vector is used for
the terminal constraints applied on ``\mathbf{x̂}_{k-1}(k+H_p)``.
"""
function predict!(Ŷ, x̂, _ , _ , _ , mpc::PredictiveController, ::LinModel, ΔŨ)
    # in-place operations to reduce allocations :
    Ŷ .= mul!(Ŷ, mpc.Ẽ, ΔŨ) .+ mpc.F
    x̂ .= mul!(x̂, mpc.con.ẽx̂, ΔŨ) .+ mpc.con.fx̂
    x̂end = x̂
    return Ŷ, x̂end
end

@doc raw"""
    predict!(Ŷ, x̂, x̂next, u, û, mpc::PredictiveController, model::SimModel, ΔŨ) -> Ŷ, x̂end

Compute both vectors if `model` is not a [`LinModel`](@ref). 
    
The method mutates `Ŷ`, `x̂`, `x̂next`, `u` and `û` arguments.
"""
function predict!(Ŷ, x̂, x̂next, u, û, mpc::PredictiveController, model::SimModel, ΔŨ)
    nu, ny, nd, Hp, Hc = model.nu, model.ny, model.nd, mpc.Hp, mpc.Hc
    u0 = u
    x̂  .= mpc.estim.x̂
    u0 .= mpc.estim.lastu0
    d0  = @views mpc.d0[1:end]
    for j=1:Hp
        if j ≤ Hc
            u0 .+= @views ΔŨ[(1 + nu*(j-1)):(nu*j)]
        end
        f̂!(x̂next, û, mpc.estim, model, x̂, u0, d0)
        x̂ .= x̂next
        d0 = @views mpc.D̂0[(1 + nd*(j-1)):(nd*j)]
        ŷ  = @views Ŷ[(1 + ny*(j-1)):(ny*j)]
        ĥ!(ŷ, mpc.estim, model, x̂, d0)
    end
    Ŷ .= Ŷ .+ mpc.Ŷop # Ŷop = Ŷs + Yop, and Ŷs=0 if mpc.estim is not an InternalModel
    x̂end = x̂
    return Ŷ, x̂end
end

"""
    obj_nonlinprog!(U , _ , _ , mpc::PredictiveController, model::LinModel, Ŷ, ΔŨ)

Nonlinear programming objective function when `model` is a [`LinModel`](@ref).

The function is called by the nonlinear optimizer of [`NonLinMPC`](@ref) controllers. It can
also be called on any [`PredictiveController`](@ref)s to evaluate the objective function `J`
at specific input increments `ΔŨ` and predictions `Ŷ` values. It mutates the `U` argument.
"""
function obj_nonlinprog!(U, _ , _ , mpc::PredictiveController, model::LinModel, Ŷ, ΔŨ)
    J = obj_quadprog(ΔŨ, mpc.H̃, mpc.q̃) + mpc.p[]
    if !iszero(mpc.E)
        U .= mul!(U, mpc.S̃, ΔŨ) .+ mpc.T_lastu
        UE = [U; U[(end - model.nu + 1):end]]
        ŶE = [mpc.ŷ; Ŷ]
        J += mpc.E*mpc.JE(UE, ŶE, mpc.D̂E)
    end
    return J
end

"""
    obj_nonlinprog!(U, Ȳ, Ū, mpc::PredictiveController, model::SimModel, Ŷ, ΔŨ)

Nonlinear programming objective function when `model` is not a [`LinModel`](@ref). The
function `dot(x, A, x)` is a performant way of calculating `x'*A*x`. This method mutates
`U`, `Ȳ` and `Ū` arguments (input over `Hp`, and output and input setpoint tracking error, 
respectively).
"""
function obj_nonlinprog!(U, Ȳ, Ū, mpc::PredictiveController, model::SimModel, Ŷ, ΔŨ)
    # --- output setpoint tracking term ---
    Ȳ  .= mpc.R̂y .- Ŷ
    JR̂y = dot(Ȳ, mpc.M_Hp, Ȳ)
    # --- move suppression and slack variable term ---
    JΔŨ = dot(ΔŨ, mpc.Ñ_Hc, ΔŨ)
    # --- input over prediction horizon ---
    if !mpc.noR̂u || !iszero(mpc.E)
        U .= mul!(U, mpc.S̃, ΔŨ) .+ mpc.T_lastu
    end
    # --- input setpoint tracking term ---
    if !mpc.noR̂u
        Ū  .= mpc.R̂u .- U
        JR̂u = dot(Ū, mpc.L_Hp, Ū)
    else
        JR̂u = 0.0
    end
    # --- economic term ---
    if !iszero(mpc.E)
        UE = [U; U[(end - model.nu + 1):end]]
        ŶE = [mpc.ŷ; Ŷ]
        E_JE = mpc.E*mpc.JE(UE, ŶE, mpc.D̂E)
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
    ΔŨvar::Vector{VariableRef} = optim[:ΔŨvar]
    # initial ΔŨ (warm-start): [Δu_{k-1}(k); Δu_{k-1}(k+1); ... ; 0_{nu × 1}; ϵ_{k-1}]
    ϵ0  = !isinf(mpc.C) ? mpc.ΔŨ[end] : empty(mpc.ΔŨ)
    ΔŨ0 = [mpc.ΔŨ[(model.nu+1):(mpc.Hc*model.nu)]; zeros(NT, model.nu); ϵ0]
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
    mpc.ΔŨ .= iserror(optim) ? ΔŨlast : ΔŨcurr
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
