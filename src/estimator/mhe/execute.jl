"Reset the data windows and time-varying variables for the moving horizon estimator."
function init_estimate_cov!(estim::MovingHorizonEstimator, _ , _ , _ ) 
    estim.invP̄      .= inv(estim.P̂_0)
    estim.P̂arr_old  .= estim.P̂_0
    estim.x̂0arr_old .= 0
    estim.Z̃         .= 0
    estim.X̂0        .= 0
    estim.Y0m       .= 0
    estim.U0        .= 0
    estim.D0        .= 0
    estim.Ŵ         .= 0
    estim.Nk        .= 0
    estim.H̃         .= 0
    estim.q̃         .= 0
    estim.p         .= 0
    return nothing
end

@doc raw"""
    update_estimate!(estim::MovingHorizonEstimator, u, ym, d=[])
    
Update [`MovingHorizonEstimator`](@ref) state `estim.x̂`.

The optimization problem of [`MovingHorizonEstimator`](@ref) documentation is solved at
each discrete time ``k``. Once solved, the next estimate ``\mathbf{x̂}_k(k+1)`` is computed
by inserting the optimal values of ``\mathbf{x̂}_k(k-N_k+1)`` and ``\mathbf{Ŵ}`` in the
augmented model from ``j = N_k-1`` to ``0`` inclusively. Afterward, if ``k ≥ H_e``, the
arrival covariance for the next time step ``\mathbf{P̂}_{k-N_k+1}(k-N_k+2)`` is estimated
with the equations of [`update_estimate!(::ExtendedKalmanFilter)`](@ref), or `KalmanFilter`,
for `LinModel`.
"""
function update_estimate!(
    estim::MovingHorizonEstimator{NT}, u, ym, d=empty(estim.x̂)
) where NT<:Real
    model, optim, x̂0 = estim.model, estim.optim, estim.x̂0
    add_data_windows!(estim::MovingHorizonEstimator, u, d, ym)
    initpred!(estim, model)
    linconstraint!(estim, model)
    nu, ny, nx̂, nym, nŵ, Nk = model.nu, model.ny, estim.nx̂, estim.nym, estim.nx̂, estim.Nk[]
    nx̃ = !isinf(estim.C) + nx̂
    Z̃var::Vector{JuMP.VariableRef} = optim[:Z̃var]
    V̂  = Vector{NT}(undef, nym*Nk)
    X̂0  = Vector{NT}(undef, nx̂*Nk)
    û0  = Vector{NT}(undef, nu)
    ŷ0  = Vector{NT}(undef, ny)
    x̄  = Vector{NT}(undef, nx̂)
    ϵ_0 = isinf(estim.C) ? empty(estim.Z̃) : estim.Z̃[begin]
    Z̃_0 = [ϵ_0; estim.x̂0arr_old; estim.Ŵ]
    V̂, X̂0 = predict!(V̂, X̂0, û0, ŷ0, estim, model, Z̃_0)
    J_0 = obj_nonlinprog!(x̄, estim, model, V̂, Z̃_0)
    # initial Z̃0 with Ŵ=0 if objective or constraint function not finite :
    isfinite(J_0) || (Z̃_0 = [ϵ_0; estim.x̂0arr_old; zeros(NT, nŵ*estim.He)])
    JuMP.set_start_value.(Z̃var, Z̃_0)
    # ------- solve optimization problem --------------
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
    # -------- error handling -------------------------
    if !issolved(optim)
        status = JuMP.termination_status(optim)
        if iserror(optim)
            @error("MHE terminated without solution: estimation in open-loop", 
                   status)
        else
            @warn("MHE termination status not OPTIMAL or LOCALLY_SOLVED: keeping "*
                  "solution anyway", status)
        end
        @debug JuMP.solution_summary(optim, verbose=true)
    end
    if iserror(optim)
        estim.Z̃ .= Z̃_0
    else
        estim.Z̃ .= JuMP.value.(Z̃var)
    end
    # --------- update estimate -----------------------
    estim.Ŵ[1:nŵ*Nk] .= @views estim.Z̃[nx̃+1:nx̃+nŵ*Nk] # update Ŵ with optimum for warm-start
    V̂, X̂0 = predict!(V̂, X̂0, û0, ŷ0, estim, model, estim.Z̃)
    # remove constant (f̂op - x̂op), since it is added in updatestate! method:
    estim.x̂0 .= @views X̂0[end-nx̂+1:end] .- (estim.f̂op .- estim.x̂op)
    Nk == estim.He && update_cov!(estim::MovingHorizonEstimator)
    return nothing
end


@doc raw"""
    getinfo(estim::MovingHorizonEstimator) -> info

Get additional info on `estim` [`MovingHorizonEstimator`](@ref) optimum for troubleshooting.

The function should be called after calling [`updatestate!`](@ref). It returns the
dictionary `info` with the following fields:

- `:Ŵ`   : optimal estimated process noise over ``N_k``, ``\mathbf{Ŵ}``
- `:x̂arr`: optimal estimated state at arrival, ``\mathbf{x̂}_k(k-N_k+1)``
- `:ϵ`   : optimal slack variable, ``ϵ``
- `:J`   : objective value optimum, ``J``
- `:X̂`   : optimal estimated states over ``N_k+1``, ``\mathbf{X̂}``
- `:x̂`   : optimal estimated state for the next time step, ``\mathbf{x̂}_k(k+1)``
- `:V̂`   : optimal estimated sensor noise over ``N_k``, ``\mathbf{V̂}``
- `:P̄`   : estimation error covariance at arrival, ``\mathbf{P̄}``
- `:x̄`   : optimal estimation error at arrival, ``\mathbf{x̄}``
- `:Ŷ`   : optimal estimated outputs over ``N_k``, ``\mathbf{Ŷ}``
- `:Ŷm`  : optimal estimated measured outputs over ``N_k``, ``\mathbf{Ŷ^m}``
- `:Ym`  : measured outputs over ``N_k``, ``\mathbf{Y^m}``
- `:U`   : manipulated inputs over ``N_k``, ``\mathbf{U}``
- `:D`   : measured disturbances over ``N_k``, ``\mathbf{D}``
- `:sol` : solution summary of the optimizer for printing

# Examples
```jldoctest
julia> estim = MovingHorizonEstimator(LinModel(ss(1.0, 1.0, 1.0, 0, 1)), He=1, nint_ym=0);

julia> updatestate!(estim, [0], [1]);

julia> round.(getinfo(estim)[:Ŷ], digits=3)
1-element Vector{Float64}:
 0.5
```
"""
function getinfo(estim::MovingHorizonEstimator{NT}) where NT<:Real
    model, Nk = estim.model, estim.Nk[]
    nu, ny, nd = model.nu, model.ny, model.nd
    nx̂, nym, nŵ = estim.nx̂, estim.nym, estim.nx̂
    nx̃ = !isinf(estim.C) + nx̂
    MyTypes = Union{JuMP._SolutionSummary, Hermitian{NT, Matrix{NT}}, Vector{NT}, NT}
    info = Dict{Symbol, MyTypes}()
    V̂, X̂0  = similar(estim.Y0m[1:nym*Nk]), similar(estim.X̂0[1:nx̂*Nk])
    û0, ŷ0 = similar(model.uop), similar(model.yop)
    V̂, X̂0  = predict!(V̂, X̂0, û0, ŷ0, estim, model, estim.Z̃)
    x̂0arr  = @views estim.Z̃[nx̃-nx̂+1:nx̃]
    x̄ = estim.x̂0arr_old - x̂0arr
    X̂0 = [x̂0arr; X̂0]
    Ym0, U0, D0 = estim.Y0m[1:nym*Nk], estim.U0[1:nu*Nk], estim.D0[1:nd*Nk]
    Ŷ0m, Ŷ0 = Vector{NT}(undef, nym*Nk), Vector{NT}(undef, ny*Nk)
    for i=1:Nk
        d0 = @views D0[(1 + nd*(i-1)):(nd*i)]
        x̂0 = @views X̂0[(1 + nx̂*(i-1)):(nx̂*i)]
        @views ĥ!(Ŷ0[(1 + ny*(i-1)):(ny*i)], estim, model, x̂0, d0)
        Ŷ0m[(1 + nym*(i-1)):(nym*i)] .= @views Ŷ0[(1 + ny*(i-1)):(ny*i)][estim.i_ym]
    end
    Ym, U, D, Ŷm, Ŷ = Ym0, U0, D0, Ŷ0m, Ŷ0
    for i=1:Nk
        Ŷ[(1 + ny*(i-1)):(ny*i)]    .+= model.yop
        Ŷm[(1 + nym*(i-1)):(nym*i)] .+= @views model.yop[estim.i_ym]
        Ym[(1 + nym*(i-1)):(nym*i)] .+= @views model.yop[estim.i_ym]
        U[(1 + nu*(i-1)):(nu*i)]    .+= model.uop
        D[(1 + nd*(i-1)):(nd*i)]    .+= model.dop
    end
    info[:Ŵ] = estim.Ŵ[1:Nk*nŵ]
    info[:x̂arr] = x̂0arr + estim.x̂op
    info[:ϵ]  = isinf(estim.C) ? NaN : estim.Z̃[begin]
    info[:J]  = obj_nonlinprog!(x̄, estim, estim.model, V̂, estim.Z̃)
    info[:X̂]  = X̂0       + [estim.x̂op; estim.X̂op]
    info[:x̂]  = estim.x̂0 +  estim.x̂op
    info[:V̂]  = V̂
    info[:P̄]  = estim.P̂arr_old
    info[:x̄]  = x̄
    info[:Ŷ]  = Ŷ
    info[:Ŷm] = Ŷm
    info[:Ym] = Ym
    info[:U]  = U 
    info[:D]  = D
    info[:sol] = JuMP.solution_summary(estim.optim, verbose=true)
    return info
end

"Add data to the observation windows of the moving horizon estimator."
function add_data_windows!(estim::MovingHorizonEstimator, u0, d0, y0m)
    model = estim.model
    nx̂, nym, nu, nd, nŵ = estim.nx̂, estim.nym, model.nu, model.nd, estim.nx̂
    x̂0, ŵ = estim.x̂0, zeros(nŵ) # ŵ(k) = 0 for warm-starting
    estim.Nk .+= 1
    Nk = estim.Nk[]
    if Nk > estim.He
        estim.X̂0[1:end-nx̂]       .= @views estim.X̂0[nx̂+1:end]
        estim.X̂0[end-nx̂+1:end]   .= x̂0
        estim.Y0m[1:end-nym]     .= @views estim.Y0m[nym+1:end]
        estim.Y0m[end-nym+1:end] .= y0m
        estim.U0[1:end-nu]       .= @views estim.U0[nu+1:end]
        estim.U0[end-nu+1:end]   .= u0
        estim.D0[1:end-nd]       .= @views estim.D0[nd+1:end]
        estim.D0[end-nd+1:end]   .= d0
        estim.Ŵ[1:end-nŵ]        .= @views estim.Ŵ[nŵ+1:end]
        estim.Ŵ[end-nŵ+1:end]    .= ŵ
        estim.Nk .= estim.He
    else
        estim.X̂0[(1 + nx̂*(Nk-1)):(nx̂*Nk)]    .= x̂0
        estim.Y0m[(1 + nym*(Nk-1)):(nym*Nk)] .= y0m
        estim.U0[(1 + nu*(Nk-1)):(nu*Nk)]    .= u0
        estim.D0[(1 + nd*(Nk-1)):(nd*Nk)]    .= d0
        estim.Ŵ[(1 + nŵ*(Nk-1)):(nŵ*Nk)]     .= ŵ
    end
    estim.x̂0arr_old .= @views estim.X̂0[1:nx̂]
    return nothing
end

@doc raw"""
    initpred!(estim::MovingHorizonEstimator, model::LinModel) -> nothing

Init quadratic optimization matrices `F, fx̄, H̃, q̃, p` for [`MovingHorizonEstimator`](@ref).

See [`init_predmat_mhe`](@ref) for the definition of the vectors ``\mathbf{F, f_x̄}``. It
also inits `estim.optim` objective function, expressed as the quadratic general form:
```math
    J = \min_{\mathbf{Z̃}} \frac{1}{2}\mathbf{Z̃' H̃ Z̃} + \mathbf{q̃' Z̃} + p 
```
in which ``\mathbf{Z̃} = [\begin{smallmatrix} ϵ \\ \mathbf{Z} \end{smallmatrix}]``. Note that
``p`` is useless at optimization but required to evaluate the objective minima ``J``. The 
Hessian ``\mathbf{H̃}`` matrix of the quadratic general form is not constant here because
of the time-varying ``\mathbf{P̄}`` covariance . The computed variables are:
```math
\begin{aligned}
    \mathbf{F}       &= \mathbf{G U_0} + \mathbf{J D_0} + \mathbf{Y_0^m} + \mathbf{B}       \\
    \mathbf{f_x̄}     &= \mathbf{x̂_0^†}(k-N_k+1)                                             \\
    \mathbf{F_Z̃}     &= [\begin{smallmatrix}\mathbf{f_x̄} \\ \mathbf{F} \end{smallmatrix}]   \\
    \mathbf{Ẽ_Z̃}     &= [\begin{smallmatrix}\mathbf{ẽ_x̄} \\ \mathbf{Ẽ} \end{smallmatrix}]   \\
    \mathbf{M}_{N_k} &= \mathrm{diag}(\mathbf{P̄}^{-1}, \mathbf{R̂}_{N_k}^{-1})               \\
    \mathbf{Ñ}_{N_k} &= \mathrm{diag}(C,  \mathbf{0},  \mathbf{Q̂}_{N_k}^{-1})               \\
    \mathbf{H̃}       &= 2(\mathbf{Ẽ_Z̃}' \mathbf{M}_{N_k} \mathbf{Ẽ_Z̃} + \mathbf{Ñ}_{N_k})   \\
    \mathbf{q̃}       &= 2(\mathbf{M}_{N_k} \mathbf{Ẽ_Z̃})' \mathbf{F_Z̃}                      \\
            p        &= \mathbf{F_Z̃}' \mathbf{M}_{N_k} \mathbf{F_Z̃}
\end{aligned}
```
"""
function initpred!(estim::MovingHorizonEstimator, model::LinModel)
    F, C, optim = estim.F, estim.C, estim.optim
    nϵ = isinf(C) ? 0 : 1
    nx̂, nŵ, nym, Nk = estim.nx̂, estim.nx̂, estim.nym, estim.Nk[]
    nYm, nŴ = nym*Nk, nŵ*Nk
    nZ̃ = nϵ + nx̂ + nŴ
    # --- update F and fx̄ vectors for MHE predictions ---
    F .= estim.Y0m .+ estim.B
    mul!(F, estim.G, estim.U0, 1, 1)
    if model.nd ≠ 0
        mul!(F, estim.J, estim.D0, 1, 1)
    end
    estim.fx̄ .= estim.x̂0arr_old
    # --- update H̃, q̃ and p vectors for quadratic optimization ---
    ẼZ̃ = @views [estim.ẽx̄[:, 1:nZ̃]; estim.Ẽ[1:nYm, 1:nZ̃]]
    FZ̃ = @views [estim.fx̄; estim.F[1:nYm]]
    invQ̂_Nk, invR̂_Nk = @views estim.invQ̂_He[1:nŴ, 1:nŴ], estim.invR̂_He[1:nYm, 1:nYm]
    M_Nk = [estim.invP̄ zeros(nx̂, nYm); zeros(nYm, nx̂) invR̂_Nk]
    Ñ_Nk = [fill(C, nϵ, nϵ) zeros(nϵ, nx̂+nŴ); zeros(nx̂, nϵ+nx̂+nŴ); zeros(nŴ, nϵ+nx̂) invQ̂_Nk]
    M_Nk_ẼZ̃ = M_Nk*ẼZ̃
    @views mul!(estim.q̃[1:nZ̃], M_Nk_ẼZ̃', FZ̃)
    @views lmul!(2, estim.q̃[1:nZ̃])
    estim.p .= dot(FZ̃, M_Nk, FZ̃)
    estim.H̃.data[1:nZ̃, 1:nZ̃] .= Ñ_Nk
    @views mul!(estim.H̃.data[1:nZ̃, 1:nZ̃], ẼZ̃', M_Nk_ẼZ̃, 1, 1) 
    @views lmul!(2, estim.H̃.data[1:nZ̃, 1:nZ̃])
    Z̃var_Nk::Vector{JuMP.VariableRef} = @views optim[:Z̃var][1:nZ̃]
    H̃_Nk = @views estim.H̃[1:nZ̃,1:nZ̃]
    q̃_Nk = @views estim.q̃[1:nZ̃]
    JuMP.set_objective_function(optim, obj_quadprog(Z̃var_Nk, H̃_Nk, q̃_Nk))
    return nothing
end
"Does nothing if `model` is not a [`LinModel`](@ref)."
initpred!(::MovingHorizonEstimator, ::SimModel) = nothing

@doc raw"""
    linconstraint!(estim::MovingHorizonEstimator, model::LinModel)

Set `b` vector for the linear model inequality constraints (``\mathbf{A Z̃ ≤ b}``) of MHE.

Also init ``\mathbf{F_x̂ = G_x̂ U_0 + J_x̂ D_0 + B_x̂}`` vector for the state constraints, see 
[`init_predmat_mhe`](@ref).
"""
function linconstraint!(estim::MovingHorizonEstimator, model::LinModel)
    Fx̂  = estim.con.Fx̂
    Fx̂ .= estim.con.Bx̂
    mul!(Fx̂, estim.con.Gx̂, estim.U0, 1, 1)
    if model.nd ≠ 0
        mul!(Fx̂, estim.con.Jx̂, estim.D0, 1, 1)
    end
    X̂0min, X̂0max = trunc_bounds(estim, estim.con.X̂0min, estim.con.X̂0max, estim.nx̂)
    Ŵmin, Ŵmax   = trunc_bounds(estim, estim.con.Ŵmin,  estim.con.Ŵmax,  estim.nx̂)
    V̂min, V̂max   = trunc_bounds(estim, estim.con.V̂min,  estim.con.V̂max,  estim.nym)
    nX̂, nŴ, nV̂ = length(X̂0min), length(Ŵmin), length(V̂min)
    nx̃ = length(estim.con.x̃0min)
    n = 0
    estim.con.b[(n+1):(n+nx̃)] .= @. -estim.con.x̃0min
    n += nx̃
    estim.con.b[(n+1):(n+nx̃)] .= @. +estim.con.x̃0max
    n += nx̃
    estim.con.b[(n+1):(n+nX̂)] .= @. -X̂0min + Fx̂
    n += nX̂
    estim.con.b[(n+1):(n+nX̂)] .= @. +X̂0max - Fx̂
    n += nX̂
    estim.con.b[(n+1):(n+nŴ)] .= @. -Ŵmin
    n += nŴ
    estim.con.b[(n+1):(n+nŴ)] .= @. +Ŵmax
    n += nŴ
    estim.con.b[(n+1):(n+nV̂)] .= @. -V̂min + estim.F
    n += nV̂
    estim.con.b[(n+1):(n+nV̂)] .= @. +V̂max - estim.F
    if any(estim.con.i_b) 
        lincon = estim.optim[:linconstraint]
        JuMP.set_normalized_rhs(lincon, estim.con.b[estim.con.i_b])
    end
    return nothing
end

"Set `b` excluding state and sensor noise bounds if `model` is not a [`LinModel`](@ref)."
function linconstraint!(estim::MovingHorizonEstimator, ::SimModel)
    Ŵmin, Ŵmax = trunc_bounds(estim, estim.con.Ŵmin, estim.con.Ŵmax, estim.nx̂)
    nx̃, nŴ = length(estim.con.x̃0min), length(Ŵmin)
    n = 0
    estim.con.b[(n+1):(n+nx̃)] .= @. -estim.con.x̃0min
    n += nx̃
    estim.con.b[(n+1):(n+nx̃)] .= @. +estim.con.x̃0max
    n += nx̃
    estim.con.b[(n+1):(n+nŴ)] .= @. -Ŵmin
    n += nŴ
    estim.con.b[(n+1):(n+nŴ)] .= @. +Ŵmax
    if any(estim.con.i_b) 
        lincon = estim.optim[:linconstraint]
        JuMP.set_normalized_rhs(lincon, estim.con.b[estim.con.i_b])
    end
    return nothing
end

"Truncate the bounds `Bmin` and `Bmax` to the window size `Nk` if `Nk < He`."
function trunc_bounds(estim::MovingHorizonEstimator{NT}, Bmin, Bmax, n) where NT<:Real
    He, Nk = estim.He, estim.Nk[]
    if Nk < He
        nB = n*Nk
        Bmin_t, Bmax_t = similar(Bmin), similar(Bmax)
        Bmin_t[1:nB]     .= @views Bmin[end-nB+1:end]
        Bmin_t[nB+1:end] .= -Inf
        Bmax_t[1:nB]     .= @views Bmax[end-nB+1:end]
        Bmax_t[nB+1:end] .= +Inf
    else
        Bmin_t = Bmin
        Bmax_t = Bmax
    end
    return Bmin_t, Bmax_t
end

"Update the covariance estimate at arrival using `covestim` [`StateEstimator`](@ref)."
function update_cov!(estim::MovingHorizonEstimator)
    nu, nd, nym = estim.model.nu, estim.model.nd, estim.nym
    u0arr, y0marr, d0arr = @views estim.U0[1:nu], estim.Y0m[1:nym], estim.D0[1:nd]
    estim.covestim.x̂0 .= estim.x̂0arr_old
    estim.covestim.P̂  .= estim.P̂arr_old
    update_estimate!(estim.covestim, u0arr, y0marr, d0arr)
    estim.P̂arr_old    .= estim.covestim.P̂
    estim.invP̄        .= inv(estim.P̂arr_old)
    return nothing
end

"""
    obj_nonlinprog!( _ , estim::MovingHorizonEstimator, ::LinModel, _ , Z̃) 

Objective function of [`MovingHorizonEstimator`](@ref) when `model` is a [`LinModel`](@ref).

It can be called on a [`MovingHorizonEstimator`](@ref) object to evaluate the objective 
function at specific `Z̃` and `V̂` values.
"""
function obj_nonlinprog!( _ , estim::MovingHorizonEstimator, ::LinModel, _ , Z̃)
    return obj_quadprog(Z̃, estim.H̃, estim.q̃) + estim.p[]
end

"""
    obj_nonlinprog!(x̄, estim::MovingHorizonEstimator, model::SimModel, V̂, Z̃)

Objective function of the MHE when `model` is not a [`LinModel`](@ref).

The function `dot(x, A, x)` is a performant way of calculating `x'*A*x`. This method mutates
`x̄` vector arguments.
"""
function obj_nonlinprog!(x̄, estim::MovingHorizonEstimator, ::SimModel, V̂, Z̃) 
    Nk, nϵ = estim.Nk[], !isinf(estim.C)
    nYm, nŴ, nx̂, invP̄ = Nk*estim.nym, Nk*estim.nx̂, estim.nx̂, estim.invP̄
    nx̃ = nϵ + nx̂
    invQ̂_Nk, invR̂_Nk = @views estim.invQ̂_He[1:nŴ, 1:nŴ], estim.invR̂_He[1:nYm, 1:nYm]
    x̂0arr, Ŵ, V̂ = @views Z̃[nx̃-nx̂+1:nx̃], Z̃[nx̃+1:nx̃+nŴ], V̂[1:nYm]
    x̄ .= estim.x̂0arr_old .- x̂0arr
    Jϵ = nϵ ? estim.C*Z̃[begin]^2 : 0
    return dot(x̄, invP̄, x̄) + dot(Ŵ, invQ̂_Nk, Ŵ) + dot(V̂, invR̂_Nk, V̂) + Jϵ
end

"""
    predict!(V̂, X̂0, û0, ŷ0, estim::MovingHorizonEstimator, model::LinModel, Z̃) -> V̂, X̂0

Compute the `V̂` vector and `X̂0` vectors for the `MovingHorizonEstimator` and `LinModel`.

The function mutates `V̂`, `X̂0`, `û0` and `ŷ0` vector arguments. The vector `V̂` is the
estimated sensor noises from ``k-N_k+1`` to ``k``. The `X̂0` vector is estimated states from 
``k-N_k+2`` to ``k+1``.
"""
function predict!(V̂, X̂0, _ , _ , estim::MovingHorizonEstimator, ::LinModel, Z̃) 
    Nk, nϵ = estim.Nk[], !isinf(estim.C)
    nX̂, nŴ, nYm = estim.nx̂*Nk, estim.nx̂*Nk, estim.nym*Nk
    nZ̃ = nϵ + estim.nx̂ + nŴ
    V̂[1:nYm] .= @views estim.Ẽ[1:nYm, 1:nZ̃]*Z̃[1:nZ̃]     + estim.F[1:nYm]
    X̂0[1:nX̂] .= @views estim.con.Ẽx̂[1:nX̂, 1:nZ̃]*Z̃[1:nZ̃] + estim.con.Fx̂[1:nX̂]
    return V̂, X̂0
end

"Compute the two vectors when `model` is not a `LinModel`."
function predict!(V̂, X̂0, û0, ŷ0, estim::MovingHorizonEstimator, model::SimModel, Z̃)
    Nk = estim.Nk[]
    nu, nd, nx̂, nŵ, nym = model.nu, model.nd, estim.nx̂, estim.nx̂, estim.nym
    nx̃ = !isinf(estim.C) + nx̂
    x̂0 = @views Z̃[nx̃-nx̂+1:nx̃]
    for j=1:Nk
        u0  = @views estim.U0[ (1 + nu  * (j-1)):(nu*j)]
        y0m = @views estim.Y0m[(1 + nym * (j-1)):(nym*j)]
        d0  = @views estim.D0[ (1 + nd  * (j-1)):(nd*j)]
        ŵ   = @views Z̃[(1 + nx̃ + nŵ*(j-1)):(nx̃ + nŵ*j)]
        ĥ!(ŷ0, estim, model, x̂0, d0)
        ŷ0m = @views ŷ0[estim.i_ym]
        V̂[(1 + nym*(j-1)):(nym*j)] .= y0m .- ŷ0m
        x̂0next = @views X̂0[(1 + nx̂ *(j-1)):(nx̂ *j)]
        f̂!(x̂0next, û0, estim, model, x̂0, u0, d0)
        x̂0next .+= ŵ .+ estim.f̂op .- estim.x̂op
        x̂0 = x̂0next
    end
    return V̂, X̂0
end

"""
    con_nonlinprog!(g, estim::MovingHorizonEstimator, model::SimModel, X̂0, V̂, Z̃)

Nonlinear constrains for [`MovingHorizonEstimator`](@ref).
"""
function con_nonlinprog!(g, estim::MovingHorizonEstimator, ::SimModel, X̂0, V̂, Z̃)
    nX̂con, nX̂ = length(estim.con.X̂0min), estim.nx̂ *estim.Nk[]
    nV̂con, nV̂ = length(estim.con.V̂min),  estim.nym*estim.Nk[]
    ϵ = isinf(estim.C) ? 0 : Z̃[begin] # ϵ = 0 if Cwt=Inf (meaning: no relaxation)
    for i in eachindex(g)
        estim.con.i_g[i] || continue
        if i ≤ nX̂con
            j = i
            jcon = nX̂con-nX̂+j
            g[i] = j > nX̂ ? 0 : estim.con.X̂0min[jcon] - X̂0[j] - ϵ*estim.con.C_x̂min[jcon]
        elseif i ≤ 2nX̂con
            j = i - nX̂con
            jcon = nX̂con-nX̂+j
            g[i] = j > nX̂ ? 0 : X̂0[j] - estim.con.X̂0max[jcon] - ϵ*estim.con.C_x̂max[jcon]
        elseif i ≤ 2nX̂con + nV̂con
            j = i - 2nX̂con
            jcon = nV̂con-nV̂+j
            g[i] = j > nV̂ ? 0 : estim.con.V̂min[jcon] - V̂[j] - ϵ*estim.con.C_v̂min[jcon]
        else
            j = i - 2nX̂con - nV̂con
            jcon = nV̂con-nV̂+j
            g[i] = j > nV̂ ? 0 : V̂[j] - estim.con.V̂max[jcon] - ϵ*estim.con.C_v̂max[jcon]
        end
    end
    return g
end


"Update the augmented model matrices of `estim` by default."
function setmodel_estimator!(estim::MovingHorizonEstimator, model::LinModel)
    As, Cs_u, Cs_y = estim.As, estim.Cs_u, estim.   Cs_y
    Â, B̂u, Ĉ, B̂d, D̂d = augment_model(model, As, Cs_u, Cs_y, verify_obsv=false)
    estim.Â  .= Â
    estim.B̂u .= B̂u
    estim.Ĉ  .= Ĉ
    estim.B̂d .= B̂d
    estim.D̂d .= D̂d
    # TODO: re-construct the MHE prediction matrices here:
    error("setmodel! for MovingHorizonEstimator is not implemented yet.")











    # LINMPC:
    # --- predictions matrices ---
    E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂ = init_predmat(estim, model, Hp, Hc)
    A_Ymin, A_Ymax, Ẽ = relaxŶ(model, mpc.C, con.C_ymin, con.C_ymax, E)
    A_x̂min, A_x̂max, ẽx̂ = relaxterminal(model, mpc.C, con.c_x̂min, con.c_x̂max, ex̂)
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
    con.U0min .-= mpc.Uop # convert U0 to U with the new operating point
    con.U0max .-= mpc.Uop # convert U0 to U with the new operating point
    con.Y0min .-= mpc.Yop # convert Y0 to Y with the new operating point
    con.Y0max .-= mpc.Yop # convert Y0 to Y with the new operating point
    con.x̂0min .-= estim.x̂op # convert x̂0 to x̂ with the new operating point
    con.x̂0max .-= estim.x̂op # convert x̂0 to x̂ with the new operating point
    con.A_Ymin .= A_Ymin
    con.A_Ymax .= A_Ymax
    con.A_x̂min .= A_x̂min
    con.A_x̂max .= A_x̂max
    nUandΔŨ = length(con.U0min) + length(con.U0max) + length(con.ΔŨmin) + length(con.ΔŨmax)
    con.A[nUandΔŨ+1:end, :] = [con.A_Ymin; con.A_Ymax; con.A_x̂min; con.A_x̂max]
    A = con.A[con.i_b, :]
    b = con.b[con.i_b]
    ΔŨvar::Vector{JuMP.VariableRef} = optim[:ΔŨvar]
    JuMP.delete(optim, optim[:linconstraint])
    JuMP.unregister(optim, :linconstraint)
    @constraint(optim, linconstraint, A*ΔŨvar .≤ b)
    # --- quadratic programming Hessian matrix ---
    H̃ = init_quadprog(model, mpc.Ẽ, mpc.S̃, mpc.M_Hp, mpc.Ñ_Hc, mpc.L_Hp)
    mpc.H̃ .= H̃
    set_objective_hessian!(mpc, ΔŨvar)


    return nothing
end