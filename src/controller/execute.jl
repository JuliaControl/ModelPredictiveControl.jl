@doc raw"""
    initstate!(mpc::PredictiveController, u, ym, d=[]) -> x̂

Init the states of `mpc.estim` [`StateEstimator`](@ref) and warm start `mpc.Z̃` at zero.

It also stores `u - mpc.estim.model.uop` at `mpc.lastu0` for converting the input increments
``\mathbf{ΔU}`` to inputs ``\mathbf{U}``.
"""
function initstate!(mpc::PredictiveController, u, ym, d=mpc.estim.buffer.empty)
    mpc.Z̃ .= 0
    mpc.lastu0 .= u .- mpc.estim.model.uop
    return initstate!(mpc.estim, u, ym, d)
end

@doc raw"""
    moveinput!(mpc::PredictiveController, ry=mpc.estim.model.yop, d=[]; <keyword args>) -> u

Compute the optimal manipulated input value `u` for the current control period.

Solve the optimization problem of `mpc` [`PredictiveController`](@ref) and return the
results ``\mathbf{u}(k)``. Following the receding horizon principle, the algorithm discards
the optimal future manipulated inputs ``\mathbf{u}(k+1), \mathbf{u}(k+2), ...`` Note that
the method mutates `mpc` internal data (it stores `u - mpc.estim.model.uop` at `mpc.lastu0`
for instance) but it does not modifies `mpc.estim` states. Call [`preparestate!(mpc, ym, d)`](@ref)
before `moveinput!`, and [`updatestate!(mpc, u, ym, d)`](@ref) after, to update `mpc` state
estimates. Setpoint and measured disturbance previews can be implemented with the `R̂y`, `R̂u`
and `D̂` keyword arguments. 

Calling a [`PredictiveController`](@ref) object calls this method.

See also [`LinMPC`](@ref), [`ExplicitMPC`](@ref), [`NonLinMPC`](@ref).

# Arguments
!!! info    
    Keyword arguments with *`emphasis`* are non-Unicode alternatives.

- `mpc::PredictiveController` : solve optimization problem of `mpc`.
- `ry=mpc.estim.model.yop` : current output setpoints ``\mathbf{r_y}(k)``.
- `d=[]` : current measured disturbances ``\mathbf{d}(k)``.
- `lastu=mpc.lastu0+mpc.estim.model.uop`: last manipulated input ``\mathbf{u}(k-1)``.
- `D̂=repeat(d, mpc.Hp)` or *`Dhat`* : predicted measured disturbances ``\mathbf{D̂}``, constant
   in the future by default or ``\mathbf{d̂}(k+j)=\mathbf{d}(k)`` for ``j=1`` to ``H_p``.
- `R̂y=repeat(ry, mpc.Hp)` or *`Rhaty`* : predicted output setpoints ``\mathbf{R̂_y}``, constant
   in the future by default or ``\mathbf{r̂_y}(k+j)=\mathbf{r_y}(k)`` for ``j=1`` to ``H_p``.
- `R̂u=mpc.Uop` or *`Rhatu`* : predicted manipulated input setpoints ``\mathbf{R̂_u}``, constant
   in the future by default or ``\mathbf{r̂_u}(k+j)=\mathbf{u_{op}}`` for ``j=0`` to ``H_p-1``. 

# Examples
```jldoctest
julia> mpc = LinMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1000, Hc=1);

julia> preparestate!(mpc, [0]); ry = [5];

julia> u = moveinput!(mpc, ry); round.(u, digits=3)
1-element Vector{Float64}:
 1.0
```
"""
function moveinput!(
    mpc::PredictiveController, 
    ry::AbstractVector = mpc.estim.model.yop, 
    d ::AbstractVector = mpc.buffer.empty;
    lastu::AbstractVector = (mpc.buffer.u .= mpc.lastu0 .+ mpc.estim.model.uop),
    Dhat ::AbstractVector = repeat!(mpc.buffer.D̂, d,  mpc.Hp),
    Rhaty::AbstractVector = repeat!(mpc.buffer.Ŷ, ry, mpc.Hp),
    Rhatu::AbstractVector = mpc.Uop,
    D̂  = Dhat,
    R̂y = Rhaty,
    R̂u = Rhatu
)
    if mpc.estim.direct && !mpc.estim.corrected[]
        @warn "preparestate! should be called before moveinput! with current estimators"
    end
    validate_args(mpc, ry, d, lastu, D̂, R̂y, R̂u)
    initpred!(mpc, mpc.estim.model, ry, d, lastu, D̂, R̂y, R̂u)
    linconstraint!(mpc, mpc.estim.model, mpc.transcription)
    linconstrainteq!(mpc, mpc.estim.model, mpc.estim, mpc.transcription)
    Z̃ = optim_objective!(mpc)
    return getinput!(mpc, Z̃)
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
- `:x̂end` or *`:xhatend`* : optimal terminal states, ``\mathbf{x̂}_i(k+H_p)``
- `:J`   : objective value optimum, ``J``
- `:U`   : optimal manipulated inputs over ``H_p``, ``\mathbf{U}``
- `:u`   : current optimal manipulated input, ``\mathbf{u}(k)``
- `:d`   : current measured disturbance, ``\mathbf{d}(k)``

For [`LinMPC`](@ref) and [`NonLinMPC`](@ref), the following fields are also available:

- `:W` : custom linear inequality constraint vector at the optimum, ``\mathbf{W}``.
- `:sol` : a [solution summary](@extref JuMP solution_summary) that can be printed. 

Lastly, the following fields are also available for [`NonLinMPC`](@ref) only:

- `:JE`: economic cost value at the optimum, ``J_E``
- `:gc`: custom nonlinear constraints values at the optimum, ``\mathbf{g_c}``
- `:∇J` or *`:nablaJ`* : optimal gradient of the objective function, ``\mathbf{\nabla} J``
- `:∇²J` or *`:nabla2J`* : optimal Hessian of the objective function, ``\mathbf{\nabla^2}J``
- `:g` : optimal nonlinear inequality constraint values, ``\mathbf{g}``
- `:∇g` or *`:nablag`* : optimal Jacobian of the inequality constraint, ``\mathbf{\nabla g}``
- `:∇²ℓg` or *`:nabla2lg`* : optimal Hessian of the inequality Lagrangian, ``\mathbf{\nabla^2}\ell_{\mathbf{g}}``
- `:geq` : optimal nonlinear equality constraint values, ``\mathbf{g_{eq}}``
- `:∇geq` or *`:nablageq`* : optimal Jacobian of the equality constraint, ``\mathbf{\nabla g_{eq}}``
- `:∇²ℓgeq` or *`:nabla2lgeq`* : optimal Hessian of the equality Lagrangian, ``\mathbf{\nabla^2}\ell_{\mathbf{g_{eq}}}``

# Examples
```jldoctest
julia> mpc = LinMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1, Hc=1);

julia> preparestate!(mpc, [0]); u = moveinput!(mpc, [10]);

julia> round.(getinfo(mpc)[:Ŷ], digits=3)
1-element Vector{Float64}:
 10.0
```
"""
function getinfo(mpc::PredictiveController{NT}) where NT<:Real
    model, buffer, transcription = mpc.estim.model, mpc.buffer, mpc.transcription
    nΔŨ, nK = mpc.Hc*model.nu + mpc.nϵ, mpc.Hp*model.nk
    nŶe, nUe = (mpc.Hp+1)*model.ny, (mpc.Hp+1)*model.nu
    nX̂0, nÛ0 = mpc.estim.nx̂*mpc.Hp, model.nu*mpc.Hp 
    Z̃ = mpc.Z̃
    info = Dict{Symbol, Any}()
    ΔŨ     = Vector{NT}(undef, nΔŨ)
    x̂0end  = similar(mpc.estim.x̂0)
    K      = Vector{NT}(undef, nK) 
    Ue, Ŷe = Vector{NT}(undef, nUe), Vector{NT}(undef, nŶe)
    U0, Ŷ0 = similar(mpc.Uop), similar(mpc.Yop)
    Û0, X̂0 = Vector{NT}(undef, nÛ0), Vector{NT}(undef, nX̂0)
    U,  Ŷ  = buffer.U, buffer.Ŷ
    D̂      = buffer.D̂
    U0 = getU0!(U0, mpc, Z̃)
    ΔŨ = getΔŨ!(ΔŨ, mpc, transcription, Z̃)
    Ŷ0, x̂0end  = predict!(Ŷ0, x̂0end, X̂0, Û0, K, mpc, model, transcription, U0, Z̃)
    Ue, Ŷe = extended_vectors!(Ue, Ŷe, mpc, U0, Ŷ0)
    U .= U0 .+ mpc.Uop
    Ŷ .= Ŷ0 .+ mpc.Yop
    D̂ .= mpc.D̂0 + mpc.Dop
    J = obj_nonlinprog!(Ŷ0, U0, mpc, Ue, Ŷe, ΔŨ)
    Ŷs = similar(mpc.Yop)
    predictstoch!(Ŷs, mpc, mpc.estim)
    info[:ΔU]   = Z̃[1:mpc.Hc*model.nu]
    info[:ϵ]    = getϵ(mpc, Z̃)
    info[:J]    = J
    info[:U]    = U
    info[:u]    = info[:U][1:model.nu]
    info[:d]    = mpc.d0 + model.dop
    info[:D̂]    = D̂
    info[:ŷ]    = mpc.ŷ
    info[:Ŷ]    = Ŷ
    info[:x̂end] = x̂0end + mpc.estim.x̂op
    info[:Ŷs]   = Ŷs
    info[:R̂y]   = mpc.R̂y
    info[:R̂u]   = mpc.R̂u
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

@doc raw"""
    getϵ(mpc::PredictiveController, Z̃orΔŨ) -> ϵ

Get the slack `ϵ` from `Z̃orΔŨ` if present, otherwise return 0.

The argument `Z̃orΔŨ` can be the augmented decision vector ``\mathbf{Z̃}`` or the augmented
input increment vector ``\mathbf{ΔŨ}``, it works with both.
"""
function getϵ(mpc::PredictiveController, Z̃orΔŨ::AbstractVector{NT}) where NT<:Real
    return mpc.nϵ ≠ 0 ? Z̃orΔŨ[end] : zero(NT)
end

"""
    addinfo!(info, mpc::PredictiveController) -> info

By default, add the custom linear inrquality vector `W` and the solution summary `:sol` that
can be printed to `info`.
"""
function addinfo!(info, mpc::PredictiveController{NT}) where NT<:Real
    nW = mpc.con.nw*(mpc.Hp+1)
    W = Vector{NT}(undef, nW)
    W .= mul!(W, mpc.con.Ẽw, mpc.Z̃) .+ mpc.con.Fw
    info[:W] = W
    info[:sol] = JuMP.solution_summary(mpc.optim, verbose=true)
    return info
end

@doc raw"""
    initpred!(mpc::PredictiveController, model::LinModel, ry, d, lastu, D̂, R̂y, R̂u) -> nothing

Init linear model prediction matrices `F, q̃, r` and current estimated output `ŷ`.

See [`init_predmat`](@ref) and [`init_quadprog`](@ref) for the definition of the matrices.
They are computed with these equations using in-place operations:
```math
\begin{aligned}
    \mathbf{F}       &= \mathbf{G d_0}(k) + \mathbf{J D̂_0} + \mathbf{K x̂_0}(k) 
                            + \mathbf{V u_0}(k-1) + \mathbf{B} + \mathbf{Ŷ_s}           \\
    \mathbf{C_y}     &= \mathbf{F} + \mathbf{Y_{op}} - \mathbf{R̂_y}                     \\
    \mathbf{C_u}     &= \mathbf{T_u}\mathbf{u}(k-1)  - \mathbf{R̂_u}                     \\
    \mathbf{q̃}       &= 2[    (\mathbf{M}_{H_p} \mathbf{Ẽ})'   \mathbf{C_y} 
                            + (\mathbf{L}_{H_p} \mathbf{P̃_U})' \mathbf{C_u}   ]         \\
    r                &=     \mathbf{C_y'}  \mathbf{M}_{H_p} \mathbf{C_y} 
                          + \mathbf{C_u'}  \mathbf{L}_{H_p} \mathbf{C_u}
\end{aligned}
```
"""
function initpred!(mpc::PredictiveController, model::LinModel, ry, d, lastu, D̂, R̂y, R̂u)
    F   = initpred_common!(mpc, model, ry, d, lastu, D̂, R̂y, R̂u)
    F .+= mpc.B                                 # F = F + B
    mul!(F, mpc.K, mpc.estim.x̂0, 1, 1)          # F = F + K*x̂0
    mul!(F, mpc.V, mpc.lastu0, 1, 1)            # F = F + V*lastu0
    if model.nd > 0
        mul!(F, mpc.G, mpc.d0, 1, 1)            # F = F + G*d0
        mul!(F, mpc.J, mpc.D̂0, 1, 1)            # F = F + J*D̂0
    end
    Cy, Cu, M_Hp_Ẽ, L_Hp_P̃u = mpc.buffer.Ŷ, mpc.buffer.U, mpc.buffer.Ẽ, mpc.buffer.P̃u
    q̃, r = mpc.q̃, mpc.r
    q̃ .= 0
    r .= 0
    # --- output setpoint tracking term ---
    if !mpc.weights.iszero_M_Hp[]
        Cy .= F .+ mpc.Yop .- R̂y
        mul!(M_Hp_Ẽ, mpc.weights.M_Hp, mpc.Ẽ)
        mul!(q̃, M_Hp_Ẽ', Cy, 1, 1)              # q̃ = q̃ + M_Hp*Ẽ'*Cy
        r .+= dot(Cy, mpc.weights.M_Hp, Cy)     # r = r + Cy'*M_Hp*Cy
    end
    # --- input setpoint tracking term ---
    if !mpc.weights.iszero_L_Hp[]
        Cu .= mpc.Tu_lastu0 .+ mpc.Uop .- R̂u 
        mul!(L_Hp_P̃u, mpc.weights.L_Hp, mpc.P̃u)
        mul!(q̃, L_Hp_P̃u', Cu, 1, 1)             # q̃ = q̃ + L_Hp*P̃u'*Cu
        r .+= dot(Cu, mpc.weights.L_Hp, Cu)     # r = r + Cu'*L_Hp*Cu
    end
    # --- finalize ---
    lmul!(2, q̃)                                 # q̃ = 2*q̃
    return nothing
end

@doc raw"""
    initpred!(mpc::PredictiveController, model::SimModel, ry, d, lastu, D̂, R̂y, R̂u) -> nothing

Init `lastu0, ŷ, F, d0, D̂0, D̂e, R̂y, R̂u` vectors when model is not a [`LinModel`](@ref).
"""
function initpred!(mpc::PredictiveController, model::SimModel, ry, d, lastu, D̂, R̂y, R̂u)
    initpred_common!(mpc, model, ry, d, lastu, D̂, R̂y, R̂u)
    return nothing
end

"""
    initpred_common!(mpc::PredictiveController, model::SimModel, ry, d, lastu, D̂, R̂y, R̂u) -> F

Common computations of `initpred!` for all types of [`SimModel`](@ref).

Will also init `mpc.F` with 0 values, or with the stochastic predictions `Ŷs` if `mpc.estim`
is an [`InternalModel`](@ref). The function returns `mpc.F`.
"""
function initpred_common!(
    mpc::PredictiveController, model::SimModel, ry, d, lastu, D̂, R̂y, R̂u
)
    mpc.lastu0 .= lastu .- model.uop
    mul!(mpc.Tu_lastu0, mpc.Tu, mpc.lastu0)
    mpc.ŷ .= evaloutput(mpc.estim, d)
    if model.nd > 0
        mpc.d0 .= d .- model.dop
        mpc.D̂0 .= D̂ .- mpc.Dop
        mpc.D̂e[1:model.nd]     .= d
        mpc.D̂e[model.nd+1:end] .= D̂
    end
    mpc.ry .= ry
    mpc.R̂y .= R̂y
    mpc.R̂u .= R̂u
    predictstoch!(mpc.F, mpc, mpc.estim)
    return mpc.F
end

@doc raw"""
    predictstoch!(Ŷs, mpc::PredictiveController, estim::InternalModel) -> nothing

Fill `Ŷs` in-place with stochastic predictions if `estim` is an [`InternalModel`](@ref).
"""
function predictstoch!(Ŷs, mpc::PredictiveController, estim::InternalModel)
    mul!(Ŷs, mpc.Ks, estim.x̂s)
    mul!(Ŷs, mpc.Ps, estim.ŷs, 1, 1)
    return nothing
end
"Fill `Ŷs` vector with 0 values when `estim` is not an [`InternalModel`](@ref)."
predictstoch!(Ŷs, ::PredictiveController, ::StateEstimator) = (Ŷs .= 0; nothing)

"""
    disturbedinput!(Û0, mpc::PredictiveController, estim::StateEstimator, U0, X̂0) -> nothing

Fill disturbed inputs of the augmented model `Û0` in-place with stochastic states in `X̂0`

Both `Û0` and `U0` variables include deviation vectors from ``k+0`` to ``k+H_p-1``. The
predicted states `X̂0` include deviation vectors from ``k+1`` to ``k+H_p-1`` (the current one
is stored in `estim.x̂0`).

This function is used for the collocation methods that directly call the state derivative 
function `estim.model.f!` with the manipulated inputs augmented with the estimated 
disturbances at model input (see [`init_estimstoch`](@ref)). It's also necessary to prefill
the `Û0` vector before anything since both `û0` and `û0next` are needed at each stage with
hold order `h>0`, thus potential race conditions with multi-threading.
"""
function disturbedinput!(Û0, mpc::PredictiveController, estim::StateEstimator, U0, X̂0)
    nu, nx, nx̂ = estim.model.nu, estim.model.nx, estim.nx̂
    Cs_u = estim.Cs_u
    Û0 .= U0                            
    for j=0:mpc.Hp-1
        xs = @views j < 1 ? estim.x̂0[(nx+1):(nx̂)] : X̂0[(nx+1+nx̂*(j-1)):(nx̂*j)] 
        û0 = @views Û0[(1+nu*j):(nu*(j+1))]
        mul!(û0, Cs_u, xs, 1, 1)                    # û0 = u0 + Cs_u*xs               
    end
    return nothing
end
"No input disturbances for [`InternalModel`](@ref), hence do `Û0 .= U0`."
disturbedinput!(Û0, ::PredictiveController, ::InternalModel, U0, _) = (Û0 .= U0; nothing)

@doc raw"""
    linconstraint_custom!(mpc::PredictiveController, model::SimModel)

Init the ``\mathbf{F_w}`` vector for the custom linear inequality constraints.

See [`relaxW`](@ref) for the definition of the vector. The function does nothing if
`mpc.con.nw < 1`.
"""
function linconstraint_custom!(mpc::PredictiveController, model::SimModel)
    mpc.con.nw < 1 && return nothing
    ny, nu, nd, buffer = model.ny, model.nu, model.nd, mpc.buffer
    Fw = mpc.con.Fw
    Ue_term, D̂e_term, R̂e_term = buffer.Ue, buffer.D̂e, buffer.Ŷe
    Fw .= 0
    Ue_term[1:end-nu]     .= mpc.Tu_lastu0 .+ mpc.Uop
    Ue_term[end-nu+1:end] .= mpc.lastu0    .+ model.uop
    mul!(Fw, mpc.con.W̄u, Ue_term, 1, 1)
    if model.nd > 0
        D̂e_term[1:nd]     .= mpc.d0 .+ model.dop
        D̂e_term[nd+1:end] .= mpc.D̂0 .+ mpc.Dop
        mul!(Fw, mpc.con.W̄d, D̂e_term, 1, 1)
    end
    R̂e_term[1:ny]     .= mpc.ry
    R̂e_term[ny+1:end] .= mpc.R̂y
    mul!(Fw, mpc.con.W̄r, R̂e_term, 1, 1)
    return linconstraint_custom_outputs!(mpc, model)
end

"Also include the `W̄y` term in the custom linear constraints for [`LinModel`](@ref)."
function linconstraint_custom_outputs!(mpc::PredictiveController, model::LinModel)
    Ŷe_term, Fw, ny = mpc.buffer.Ŷe, mpc.con.Fw, model.ny
    Ŷe_term[1:ny]     .= mpc.ŷ
    Ŷe_term[ny+1:end] .= mpc.F .+ mpc.Yop
    mul!(Fw, mpc.con.W̄y, Ŷe_term, 1, 1)
    return nothing
end
"Do nothing for other model types."
linconstraint_custom_outputs!(::PredictiveController, ::SimModel) = nothing

"""
    extended_vectors!(Ue, Ŷe, mpc::PredictiveController, U0, Ŷ0) -> Ue, Ŷe

Compute the extended `Ue` and `Ŷe` vectors for nonlinear programming using `U0` and `Ŷ0`.

See [`NonLinMPC`](@ref) for the definition of the vectors. The function mutates `Ue` and
and `Ŷe` in arguments, without assuming any initial values for them. Using 
`nocustomfcts = mpc.weights.iszero_E && mpc.con.nc == 0`, there are three special cases in
which `Ue` and `Ŷe` are not mutated:

- If `mpc.weights.iszero_M_Hp[] && nocustomfcts`, the `Ŷe` vector is not computed for the
  same reason as above.
- If `mpc.weights.iszero_L_Hp[] && nocustomfcts`, the `Ue` vector is not computed for the
  same reason as above.
"""
function extended_vectors!(Ue, Ŷe, mpc::PredictiveController, U0, Ŷ0)
    model = mpc.estim.model
    ny, nu = model.ny, model.nu
    nocustomfcts = (mpc.weights.iszero_E && iszero_nc(mpc))
    # --- extended manipulated inputs Ue = [U; u(k+Hp-1)] ---
    if !(mpc.weights.iszero_L_Hp[] && nocustomfcts)
        Ue[1:end-nu] .= U0 .+ mpc.Uop
        # u(k + Hp) = u(k + Hp - 1) since Δu(k+Hp) = 0 (because Hc ≤ Hp):
        Ue[end-nu+1:end] .= @views Ue[end-2*nu+1:end-nu]
    end
    # --- extended output predictions Ŷe = [ŷ(k); Ŷ] ---
    if !(mpc.weights.iszero_M_Hp[] && nocustomfcts)
        Ŷe[1:ny] .= mpc.ŷ
        Ŷe[ny+1:end] .= Ŷ0 .+ mpc.Yop
    end
    return Ue, Ŷe 
end

"Verify if the custom nonlinear constraint has zero elements."
iszero_nc(mpc::PredictiveController) = (mpc.con.nc == 0)

"""
    obj_nonlinprog!(Ȳ, Ū, mpc::PredictiveController, Ue, Ŷe, ΔŨ)

Nonlinear programming objective method when `model` is not a [`LinModel`](@ref). The
function `dot(x, A, x)` is a performant way of calculating `x'*A*x`. This method mutates
`Ȳ` and `Ū` arguments, without assuming any initial values (it recuperates the values in
`Ŷe` and `Ue` arguments).

Note that a specialized version on [`LinModel`](@ref) that uses the Hessian matrix `mpc.H̃`
is actually slower in the [`MultipleShooting`](@ref) case, so only one method is defined.
"""
function obj_nonlinprog!(
    Ȳ, Ū, mpc::PredictiveController, Ue, Ŷe, ΔŨ::AbstractVector{NT}
) where NT<:Real
    model = mpc.estim.model
    nu, ny = model.nu, model.ny
    # --- output setpoint tracking term ---
    if mpc.weights.iszero_M_Hp[]
        JR̂y = zero(NT)
    else
        Ȳ  .= @views Ŷe[ny+1:end]
        Ȳ  .= Ȳ .- mpc.R̂y  
        JR̂y = dot(Ȳ, mpc.weights.M_Hp, Ȳ)
    end
    # --- move suppression and slack variable term ---
    if mpc.weights.iszero_Ñ_Hc[]
        JΔŨ = zero(NT)
    else
        JΔŨ = dot(ΔŨ, mpc.weights.Ñ_Hc, ΔŨ)
    end
    # --- input setpoint tracking term ---
    if mpc.weights.iszero_L_Hp[]
        JR̂u = zero(NT)
    else
        Ū  .= @views Ue[1:end-nu]
        Ū  .= Ū .- mpc.R̂u
        JR̂u = dot(Ū, mpc.weights.L_Hp, Ū)
    end
    # --- economic term ---
    ϵ = getϵ(mpc, ΔŨ)
    E_JE = obj_econ(mpc, model, Ue, Ŷe, ϵ)
    return JR̂y + JΔŨ + JR̂u + E_JE
end

"No custom nonlinear constraints `gc` by default, return `gc` unchanged." 
con_custom!(gc, ::PredictiveController, _ , _, _ ) = gc

"By default, the economic term is zero."
function obj_econ(::PredictiveController, ::SimModel, _ , ::AbstractVector{NT}, _ ) where NT
    return zero(NT)
end

@doc raw"""
    optim_objective!(mpc::PredictiveController) -> Z̃

Optimize the objective function of `mpc` [`PredictiveController`](@ref) and return the solution `Z̃`.

If first warm-starts the solver with [`set_warmstart!`](@ref). It then calls 
`JuMP.optimize!(mpc.optim)` and extract the solution. A failed optimization prints an 
`@error` log in the REPL and returns the warm-start value. A failed optimization also prints
[`getinfo`](@ref) results in the debug log [if activated](https://docs.julialang.org/en/v1/stdlib/Logging/#Example:-Enable-debug-level-messages).
"""
function optim_objective!(mpc::PredictiveController{NT}) where {NT<:Real}
    model, optim = mpc.estim.model, mpc.optim
    nu, Hc = model.nu, mpc.Hc 
    Z̃var::Vector{JuMP.VariableRef} = optim[:Z̃var]
    Z̃s = set_warmstart!(mpc, mpc.transcription, Z̃var)
    set_objective_linear_coef!(mpc, Z̃var)
    try
        JuMP.optimize!(optim)
    catch err
        if isa(err, MOI.UnsupportedAttribute{MOI.VariablePrimalStart})
            # reset_optimizer to unset warm-start, set_start_value.(nothing) seems buggy
            MOIU.reset_optimizer(optim)
            JuMP.optimize!(optim)
        else
            rethrow()
        end
    end
    if !issolved(optim)
        status = JuMP.termination_status(optim)
        if iserror(optim)
            @error(
                "MPC terminated without solution: returning last solution shifted "*
                "(more info in debug log)",
                status
            )
        else
            @warn(
                "MPC termination status not OPTIMAL or LOCALLY_SOLVED: keeping solution "*
                "anyway (more info in debug log)", 
                status
            )
        end
        @debug info2debugstr(getinfo(mpc))
    end
    if iserror(optim)
        mpc.Z̃ .= Z̃s
    else
        mpc.Z̃ .= JuMP.value.(Z̃var)
    end
    return mpc.Z̃
end

"By default, no need to modify the objective function."
set_objective_linear_coef!(::PredictiveController, _ ) = nothing

"""
    preparestate!(mpc::PredictiveController, ym, d=[]) -> x̂

Call [`preparestate!`](@ref) on `mpc.estim` [`StateEstimator`](@ref).
"""
function preparestate!(mpc::PredictiveController, ym, d=mpc.estim.buffer.empty)
    return preparestate!(mpc.estim, ym, d)
end

@doc raw"""
    getinput!(mpc::PredictiveController, Z̃) -> u

Get current manipulated input `u` from the solution `Z̃`, store it and return it.

The first manipulated input ``\mathbf{u}(k)`` is extracted from the decision vector
``\mathbf{Z̃}`` and applied on the plant (from the receding horizon principle). It also
stores `u - mpc.estim.model.uop` at `mpc.lastu0`.
"""
function getinput!(mpc, Z̃)
    model = mpc.estim.model
    Δu  = mpc.buffer.u
    for i in 1:model.nu
        Δu[i] = Z̃[i]
    end
    u   = Δu
    u .+= mpc.lastu0 .+ model.uop
    mpc.lastu0 .=  u .- model.uop
    return u
end

"""
    updatestate!(mpc::PredictiveController, u, ym, d=[]) -> x̂next

Call [`updatestate!`](@ref) on `mpc.estim` [`StateEstimator`](@ref).
"""
function updatestate!(mpc::PredictiveController, u, ym, d=mpc.estim.buffer.empty)
    return updatestate!(mpc.estim, u, ym, d)
end
updatestate!(::PredictiveController, _ ) = throw(ArgumentError("missing measured outputs ym"))

"""
    savetime!(mpc::PredictiveController) -> t

Call `savetime!(mpc.estim.model)` and return the time `t`.
"""
savetime!(mpc::PredictiveController) = savetime!(mpc.estim.model)

"""
    periodsleep(mpc::PredictiveController, busywait=false) -> nothing

Call `periodsleep(mpc.estim.model)`.
"""
periodsleep(mpc::PredictiveController, busywait=false) = periodsleep(mpc.estim.model, busywait)

"""
    setstate!(mpc::PredictiveController, x̂[, P̂]) -> mpc

Call [`setstate!`](@ref) on `mpc.estim` [`StateEstimator`](@ref).
"""
setstate!(mpc::PredictiveController, x̂, P̂=nothing) = (setstate!(mpc.estim, x̂, P̂); return mpc)

@doc raw"""
    setmodel!(mpc::PredictiveController, model=mpc.estim.model; <keyword arguments>) -> mpc

Set `model` and objective function weights of `mpc` [`PredictiveController`](@ref).

Allows model adaptation of controllers based on [`LinModel`](@ref) at runtime. Modification
of [`NonLinModel`](@ref) state-space functions is not supported. New weight matrices in the
objective function can be specified with the keyword arguments (see [`LinMPC`](@ref) for the
nomenclature). If `Cwt ≠ Inf`, the augmented move suppression weight is ``\mathbf{Ñ}_{H_c} =
\mathrm{diag}(\mathbf{N}_{H_c}, C)``, else ``\mathbf{Ñ}_{H_c} = \mathbf{N}_{H_c}``. The
[`StateEstimator`](@ref) `mpc.estim` cannot be a [`Luenberger`](@ref) observer or a
[`SteadyKalmanFilter`](@ref) (the default estimator). Construct the `mpc` object with a
time-varying [`KalmanFilter`](@ref) instead. Note that the model is constant over the
prediction horizon ``H_p``.

# Arguments
!!! info
    Keyword arguments with *`emphasis`* are non-Unicode alternatives.

- `mpc::PredictiveController` : controller to set model and weights.
- `model=mpc.estim.model` : new plant model (not supported by [`NonLinModel`](@ref)).
- `Mwt=nothing` : new main diagonal in ``\mathbf{M}`` weight matrix (vector).
- `Nwt=nothing` : new main diagonal in ``\mathbf{N}`` weight matrix (vector).
- `Lwt=nothing` : new main diagonal in ``\mathbf{L}`` weight matrix (vector).
- `M_Hp=nothing` : new ``\mathbf{M}_{H_p}`` weight matrix.
- `Ñ_Hc=nothing` or *`Ntilde_Hc`* : new ``\mathbf{Ñ}_{H_c}`` weight matrix (see def. above).
- `L_Hp=nothing` : new ``\mathbf{L}_{H_p}`` weight matrix.
- additional keyword arguments are passed to `setmodel!(mpc.estim)`.

# Examples
```jldoctest
julia> mpc = LinMPC(KalmanFilter(LinModel(ss(0.1, 0.5, 1, 0, 4.0)), σR=[√25]), Hp=1, Hc=1);

julia> mpc.estim.model.A[1], mpc.estim.cov.R̂[1], mpc.weights.M_Hp[1], mpc.weights.Ñ_Hc[1]
(0.1, 25.0, 1.0, 0.1)

julia> setmodel!(mpc, LinModel(ss(0.42, 0.5, 1, 0, 4.0)); R̂=[9], M_Hp=[10], Nwt=[0.666]);

julia> mpc.estim.model.A[1], mpc.estim.cov.R̂[1], mpc.weights.M_Hp[1], mpc.weights.Ñ_Hc[1]
(0.42, 9.0, 10.0, 0.666)
```
"""
function setmodel!(
        mpc::PredictiveController, 
        model = mpc.estim.model;
        Mwt       = nothing,
        Nwt       = nothing,
        Lwt       = nothing,
        M_Hp      = nothing,
        Ntilde_Hc = nothing,
        L_Hp      = nothing,
        Ñ_Hc      = Ntilde_Hc,
        kwargs...
    )
    uop_old = copy(mpc.estim.model.uop)
    x̂op_old = copy(mpc.estim.x̂op)
    nu, ny, Hp, Hc, nϵ = model.nu, model.ny, mpc.Hp, mpc.Hc, mpc.nϵ
    setmodel!(mpc.estim, model; kwargs...)
    if isnothing(M_Hp) && !isnothing(Mwt)
        size(Mwt) == (ny,) || throw(ArgumentError("Mwt should be a vector of length $ny"))
        any(x -> x < 0, Mwt) && throw(ArgumentError("Mwt values should be nonnegative"))
        for i=1:ny*Hp
            mpc.weights.M_Hp[i, i] = Mwt[(i-1) % ny + 1]
        end
        mpc.weights.iszero_M_Hp[] = iszero(mpc.weights.M_Hp)
    elseif !isnothing(M_Hp)
        M_Hp = to_hermitian(M_Hp)
        nŶ = ny*Hp
        size(M_Hp) == (nŶ, nŶ) || throw(ArgumentError("M_Hp size should be ($nŶ, $nŶ)"))
        mpc.weights.M_Hp .= M_Hp
        mpc.weights.iszero_M_Hp[] = iszero(mpc.weights.M_Hp)
    end
    if isnothing(Ñ_Hc) && !isnothing(Nwt)
        size(Nwt) == (nu,) || throw(ArgumentError("Nwt should be a vector of length $nu"))
        any(x -> x < 0, Nwt) && throw(ArgumentError("Nwt values should be nonnegative"))
        for i=1:nu*Hc
            mpc.weights.Ñ_Hc[i, i] = Nwt[(i-1) % nu + 1]
        end
        mpc.weights.iszero_Ñ_Hc[] = iszero(mpc.weights.Ñ_Hc)
    elseif !isnothing(Ñ_Hc)
        Ñ_Hc = to_hermitian(Ñ_Hc)
        nΔŨ = nu*Hc+nϵ
        size(Ñ_Hc) == (nΔŨ, nΔŨ) || throw(ArgumentError("Ñ_Hc size should be ($nΔŨ, $nΔŨ)"))
        mpc.weights.Ñ_Hc .= Ñ_Hc
        mpc.weights.iszero_Ñ_Hc[] = iszero(mpc.weights.Ñ_Hc)
    end
    if isnothing(L_Hp) && !isnothing(Lwt)
        size(Lwt) == (nu,) || throw(ArgumentError("Lwt should be a vector of length $nu"))
        any(x -> x < 0, Lwt) && throw(ArgumentError("Lwt values should be nonnegative"))
        for i=1:nu*Hp
            mpc.weights.L_Hp[i, i] = Lwt[(i-1) % nu + 1]
        end
        mpc.weights.iszero_L_Hp[] = iszero(mpc.weights.L_Hp)
    elseif !isnothing(L_Hp)
        L_Hp = to_hermitian(L_Hp)
        nU = nu*Hp
        size(L_Hp) == (nU, nU) || throw(ArgumentError("L_Hp size should be ($nU, $nU)"))
        mpc.weights.L_Hp .= L_Hp
        mpc.weights.iszero_L_Hp[] = iszero(mpc.weights.L_Hp)
    end
    setmodel_controller!(mpc, uop_old, x̂op_old)
    return mpc
end

"Update the prediction matrices, linear constraints and JuMP optimization."
function setmodel_controller!(mpc::PredictiveController, uop_old, x̂op_old)
    model, estim, transcription = mpc.estim.model, mpc.estim, mpc.transcription
    weights = mpc.weights
    nu, ny, nd, Hp, Hc, nb = model.nu, model.ny, model.nd, mpc.Hp, mpc.Hc, mpc.nb
    optim, con = mpc.optim, mpc.con
    nZ = get_nZ(estim, transcription, Hp, Hc)
    Pu = mpc.P̃u[:, 1:nZ]
    # --- prediction matrices ---
    E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂ = init_predmat(
        model, estim, transcription, Hp, Hc, nb
    )
    A_Ymin, A_Ymax, Ẽ = relaxŶ(E, con.C_ymin, con.C_ymax, mpc.nϵ)
    A_Wmin, A_Wmax, Ẽw = relaxW(E, Pu, Hp, con.W̄y, con.W̄u, con.C_wmin, con.C_wmax, mpc.nϵ)
    A_x̂min, A_x̂max, ẽx̂ = relaxterminal(ex̂, con.c_x̂min, con.c_x̂max, mpc.nϵ)
    mpc.Ẽ .= Ẽ
    mpc.G .= G
    mpc.J .= J
    mpc.K .= K
    mpc.V .= V
    mpc.B .= B
    # --- terminal constraints ---
    con.ẽx̂ .= ẽx̂ 
    con.gx̂ .= gx̂
    con.jx̂ .= jx̂
    con.kx̂ .= kx̂
    con.vx̂ .= vx̂
    con.bx̂ .= bx̂
    # --- defect matrices ---
    Eŝ, Gŝ, Jŝ, Kŝ, Vŝ, Bŝ = init_defectmat(model, estim, transcription, Hp, Hc, nb)
    Aeq, Ẽŝ = augmentdefect(Eŝ, mpc.nϵ)
    con.Ẽŝ .= Ẽŝ
    con.Gŝ .= Gŝ
    con.Jŝ .= Jŝ
    con.Kŝ .= Kŝ
    con.Vŝ .= Vŝ
    con.Bŝ .= Bŝ
    # --- custom linear constraints ---
    con.Ẽw .= Ẽw
    # --- linear inequality constraints ---
    con.A_Ymin .= A_Ymin
    con.A_Ymax .= A_Ymax
    con.A_Wmin .= A_Wmin
    con.A_Wmax .= A_Wmax
    con.A_x̂min .= A_x̂min
    con.A_x̂max .= A_x̂max
    con.A .= [
        con.A_Umin
        con.A_Umax 
        con.A_ΔŨmin 
        con.A_ΔŨmax 
        con.A_Ymin  
        con.A_Ymax 
        con.A_x̂min  
        con.A_x̂max
    ]
    # --- linear equality constraints ---
    con.Aeq .= Aeq
    # --- operating points ---
    con.U0min .+= mpc.Uop # convert U0 to U with the old operating point
    con.U0max .+= mpc.Uop # convert U0 to U with the old operating point
    con.Y0min .+= mpc.Yop # convert Y0 to Y with the old operating point
    con.Y0max .+= mpc.Yop # convert Y0 to Y with the old operating point
    con.x̂0min .+= x̂op_old # convert x̂0 to x̂ with the old operating point
    con.x̂0max .+= x̂op_old # convert x̂0 to x̂ with the old operating point
    mpc.lastu0 .+= uop_old .- model.uop
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
    # --- quadratic programming Hessian matrix ---
    # do not verify the condition number of the Hessian here:
    H̃ = init_quadprog(model, transcription, weights, mpc.Ẽ, mpc.P̃Δu, mpc.P̃u; warn_cond=Inf)
    mpc.H̃ .= H̃
    # --- JuMP optimization ---
    Z̃var::Vector{JuMP.VariableRef} = optim[:Z̃var]
    A = con.A[con.i_b, :]
    b = con.b[con.i_b]
    # deletion is required for sparse solvers like OSQP, when the sparsity pattern changes
    JuMP.delete(optim, optim[:linconstraint])
    JuMP.unregister(optim, :linconstraint)
    @constraint(optim, linconstraint, A*Z̃var .≤ b)
    JuMP.delete(optim, optim[:linconstrainteq])
    JuMP.unregister(optim, :linconstrainteq)
    @constraint(optim, linconstrainteq, con.Aeq*Z̃var .== con.beq)
    set_objective_hessian!(mpc, Z̃var)
    return nothing
end

"No need to set the objective Hessian by default (only needed for quadratic optimization)."
set_objective_hessian!(::PredictiveController, _ ) = nothing