"Reset the data windows and time-varying variables for the moving horizon estimator."
function init_estimate_cov!(estim::MovingHorizonEstimator, _ , _ , _ ) 
    estim.invP̄.data          .= inv(estim.P̂0)
    estim.P̂arr_old.data      .= estim.P̂0
    estim.x̂arr_old           .= 0
    estim.Z̃                  .= 0
    estim.X̂                  .= 0
    estim.Ym                 .= 0
    estim.U                  .= 0
    estim.D                  .= 0
    estim.Ŵ                  .= 0
    estim.Nk                 .= 0
    estim.H̃.data             .= 0
    estim.q̃                  .= 0
    estim.p                  .= 0
    return nothing
end

@doc raw"""
    update_estimate!(estim::MovingHorizonEstimator, u, ym, d)
    
Update [`MovingHorizonEstimator`](@ref) state `estim.x̂`.

The optimization problem of [`MovingHorizonEstimator`](@ref) documentation is solved at
each discrete time ``k``. Once solved, the next estimate ``\mathbf{x̂}_k(k+1)`` is computed
by inserting the optimal values of ``\mathbf{x̂}_k(k-N_k+1)`` and ``\mathbf{Ŵ}`` in the
augmented model from ``j = N_k-1`` to ``0`` inclusively. Afterward, if ``k ≥ H_e``, the
arrival covariance for the next time step ``\mathbf{P̂}_{k-N_k+1}(k-N_k+2)`` is estimated
with the equations of [`update_estimate!(::ExtendedKalmanFilter)`](@ref), or `KalmanFilter`,
for `LinModel`.
"""
function update_estimate!(estim::MovingHorizonEstimator{NT}, u, ym, d) where NT<:Real
    model, optim, x̂ = estim.model, estim.optim, estim.x̂
    add_data_windows!(estim::MovingHorizonEstimator, u, d, ym)
    initpred!(estim, model)
    linconstraint!(estim, model)
    nu, ny, nx̂, nym, nŵ, Nk = model.nu, model.ny, estim.nx̂, estim.nym, estim.nx̂, estim.Nk[]
    nx̃ = !isinf(estim.C) + nx̂
    Z̃var::Vector{VariableRef} = optim[:Z̃var]
    V̂  = Vector{NT}(undef, nym*Nk)
    X̂  = Vector{NT}(undef, nx̂*Nk)
    û  = Vector{NT}(undef, nu)
    ŷ  = Vector{NT}(undef, ny)
    x̄  = Vector{NT}(undef, nx̂)
    ϵ0 = isinf(estim.C) ? empty(estim.Z̃) : estim.Z̃[begin]
    Z̃0 = [ϵ0; estim.x̂arr_old; estim.Ŵ]
    V̂, X̂ = predict!(V̂, X̂, û, ŷ, estim, model, Z̃0)
    J0 = obj_nonlinprog!(x̄, estim, model, V̂, Z̃0)
    # initial Z̃0 with Ŵ=0 if objective or constraint function not finite :
    isfinite(J0) || (Z̃0 = [ϵ0; estim.x̂arr_old; zeros(NT, nŵ*estim.He)])
    set_start_value.(Z̃var, Z̃0)
    # ------- solve optimization problem --------------
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
    # -------- error handling -------------------------
    Z̃curr, Z̃last = value.(Z̃var), Z̃0
    if !issolved(optim)
        status = termination_status(optim)
        if iserror(optim)
            @error("MHE terminated without solution: estimation in open-loop", 
                   status)
        else
            @warn("MHE termination status not OPTIMAL or LOCALLY_SOLVED: keeping "*
                  "solution anyway", status)
        end
        @debug solution_summary(optim, verbose=true)
    end
    estim.Z̃ .= iserror(optim) ? Z̃last : Z̃curr
    # --------- update estimate -----------------------
    estim.Ŵ[1:nŵ*Nk] .= @views estim.Z̃[nx̃+1:nx̃+nŵ*Nk] # update Ŵ with optimum for warm-start
    V̂, X̂ = predict!(V̂, X̂, û, ŷ, estim, model, estim.Z̃)
    x̂ .= X̂[end-nx̂+1:end]
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
    V̂, X̂ = similar(estim.Ym[1:nym*Nk]), similar(estim.X̂[1:nx̂*Nk])
    û, ŷ = similar(model.uop), similar(model.yop)
    V̂, X̂ = predict!(V̂, X̂, û, ŷ, estim, model, estim.Z̃)
    x̂arr = estim.Z̃[nx̃-nx̂+1:nx̃]
    x̄ = estim.x̂arr_old - x̂arr
    X̂ = [x̂arr; X̂]
    Ym, U, D = estim.Ym[1:nym*Nk], estim.U[1:nu*Nk], estim.D[1:nd*Nk]
    Ŷ = Vector{NT}(undef, ny*Nk)
    for i=1:Nk
        d0 = @views D[(1 + nd*(i-1)):(nd*i)] # operating point already removed in estim.D
        x̂  = @views X̂[(1 + nx̂*(i-1)):(nx̂*i)]
        @views ĥ!(Ŷ[(1 + ny*(i-1)):(ny*i)], estim, model, x̂, d0)
        Ŷ[(1 + ny*(i-1)):(ny*i)] .+= model.yop
    end
    Ŷm = Ym - V̂
    info[:Ŵ] = estim.Ŵ[1:Nk*nŵ]
    info[:x̂arr] = x̂arr
    info[:ϵ] = isinf(estim.C) ? NaN : estim.Z̃[begin]
    info[:J] = obj_nonlinprog!(x̄, estim, estim.model, V̂, estim.Z̃)
    info[:X̂] = X̂
    info[:x̂] = estim.x̂
    info[:V̂] = V̂
    info[:P̄] = estim.P̂arr_old
    info[:x̄] = x̄
    info[:Ŷ] = Ŷ
    info[:Ŷm] = Ŷm
    info[:Ym] = Ym
    info[:U] = U
    info[:D] = D
    info[:sol] = solution_summary(estim.optim, verbose=true)
    return info
end

"Add data to the observation windows of the moving horizon estimator."
function add_data_windows!(estim::MovingHorizonEstimator, u, d, ym)
    model = estim.model
    nx̂, nym, nu, nd, nŵ = estim.nx̂, estim.nym, model.nu, model.nd, estim.nx̂
    x̂, ŵ = estim.x̂, zeros(nŵ) # ŵ(k) = 0 for warm-starting
    estim.Nk .+= 1
    Nk = estim.Nk[]
    if Nk > estim.He
        estim.X̂[1:end-nx̂]       .= @views estim.X̂[nx̂+1:end]
        estim.X̂[end-nx̂+1:end]   .= x̂
        estim.Ym[1:end-nym]     .= @views estim.Ym[nym+1:end]
        estim.Ym[end-nym+1:end] .= ym
        estim.U[1:end-nu]       .= @views estim.U[nu+1:end]
        estim.U[end-nu+1:end]   .= u
        estim.D[1:end-nd]       .= @views estim.D[nd+1:end]
        estim.D[end-nd+1:end]   .= d
        estim.Ŵ[1:end-nŵ]       .= @views estim.Ŵ[nŵ+1:end]
        estim.Ŵ[end-nŵ+1:end]   .= ŵ
        estim.Nk .= estim.He
    else
        estim.X̂[(1 + nx̂*(Nk-1)):(nx̂*Nk)]    .= x̂
        estim.Ym[(1 + nym*(Nk-1)):(nym*Nk)] .= ym
        estim.U[(1 + nu*(Nk-1)):(nu*Nk)]    .= u
        estim.D[(1 + nd*(Nk-1)):(nd*Nk)]    .= d
        estim.Ŵ[(1 + nŵ*(Nk-1)):(nŵ*Nk)]    .= ŵ
    end
    estim.x̂arr_old .= @views estim.X̂[1:nx̂]
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
    \mathbf{F}       &= \mathbf{G U} + \mathbf{J D} + \mathbf{Y^m} \\
    \mathbf{f_x̄}     &= \mathbf{x̂}_{k-N_k}(k-N_k+1) \\
    \mathbf{F_Z̃}     &= [\begin{smallmatrix}\mathbf{f_x̄} \\ \mathbf{F} \end{smallmatrix}] \\
    \mathbf{Ẽ_Z̃}     &= [\begin{smallmatrix}\mathbf{ẽ_x̄} \\ \mathbf{Ẽ} \end{smallmatrix}] \\
    \mathbf{M}_{N_k} &= \mathrm{diag}(\mathbf{P̄}^{-1}, \mathbf{R̂}_{N_k}^{-1}) \\
    \mathbf{Ñ}_{N_k} &= \mathrm{diag}(C,  \mathbf{0},  \mathbf{Q̂}_{N_k}^{-1}) \\
    \mathbf{H̃}       &= 2(\mathbf{Ẽ_Z̃}' \mathbf{M}_{N_k} \mathbf{Ẽ_Z̃} + \mathbf{Ñ}_{N_k}) \\
    \mathbf{q̃}       &= 2(\mathbf{M}_{N_k} \mathbf{Ẽ_Z̃})' \mathbf{F_Z̃} \\
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
    F .= estim.Ym
    mul!(F, estim.G, estim.U, 1, 1)
    if model.nd ≠ 0
        mul!(F, estim.J, estim.D, 1, 1)
    end
    estim.fx̄ .= estim.x̂arr_old
    # --- update H̃, q̃ and p vectors for quadratic optimization ---
    ẼZ̃ = @views [estim.ẽx̄[:, 1:nZ̃]; estim.Ẽ[1:nYm, 1:nZ̃]]
    FZ̃ = @views [estim.fx̄; estim.F[1:nYm]]
    invQ̂_Nk, invR̂_Nk = @views estim.invQ̂_He[1:nŴ, 1:nŴ], estim.invR̂_He[1:nYm, 1:nYm]
    M_Nk = [estim.invP̄ zeros(nx̂, nYm); zeros(nYm, nx̂) invR̂_Nk]
    Ñ_Nk = [fill(C, nϵ, nϵ) zeros(nϵ, nx̂+nŴ); zeros(nx̂, nϵ+nx̂+nŴ); zeros(nŴ, nϵ+nx̂) invQ̂_Nk]
    estim.q̃[1:nZ̃] .= lmul!(2, (M_Nk*ẼZ̃)'*FZ̃)
    estim.p       .= dot(FZ̃, M_Nk, FZ̃)
    estim.H̃.data[1:nZ̃, 1:nZ̃] .= lmul!(2, (ẼZ̃'*M_Nk*ẼZ̃ .+ Ñ_Nk))
    Z̃var_Nk::Vector{VariableRef} = @views optim[:Z̃var][1:nZ̃]
    H̃_Nk = @views estim.H̃[1:nZ̃,1:nZ̃]
    q̃_Nk = @views estim.q̃[1:nZ̃]
    set_objective_function(optim, obj_quadprog(Z̃var_Nk, H̃_Nk, q̃_Nk))
    return nothing
end
"Does nothing if `model` is not a [`LinModel`](@ref)."
initpred!(::MovingHorizonEstimator, ::SimModel) = nothing

@doc raw"""
    linconstraint!(estim::MovingHorizonEstimator, model::LinModel)

Set `b` vector for the linear model inequality constraints (``\mathbf{A Z̃ ≤ b}``) of MHE.

Also init ``\mathbf{F_x̂ = G_x̂ U + J_x̂ D}`` vector for the state constraints, see 
[`init_predmat_mhe`](@ref).
"""
function linconstraint!(estim::MovingHorizonEstimator, model::LinModel)
    Fx̂ = estim.con.Fx̂
    mul!(Fx̂, estim.con.Gx̂, estim.U)
    if model.nd ≠ 0
        mul!(Fx̂, estim.con.Jx̂, estim.D, 1, 1)
    end
    X̂min, X̂max = trunc_bounds(estim, estim.con.X̂min, estim.con.X̂max, estim.nx̂)
    Ŵmin, Ŵmax = trunc_bounds(estim, estim.con.Ŵmin, estim.con.Ŵmax, estim.nx̂)
    V̂min, V̂max = trunc_bounds(estim, estim.con.V̂min, estim.con.V̂max, estim.nym)
    nX̂, nŴ, nV̂ = length(X̂min), length(Ŵmin), length(V̂min)
    nx̃ = length(estim.con.x̃min)
    n = 0
    estim.con.b[(n+1):(n+nx̃)] .= @. -estim.con.x̃min
    n += nx̃
    estim.con.b[(n+1):(n+nx̃)] .= @. +estim.con.x̃max
    n += nx̃
    estim.con.b[(n+1):(n+nX̂)] .= @. -X̂min + Fx̂
    n += nX̂
    estim.con.b[(n+1):(n+nX̂)] .= @. +X̂max - Fx̂
    n += nX̂
    estim.con.b[(n+1):(n+nŴ)] .= @. -Ŵmin
    n += nŴ
    estim.con.b[(n+1):(n+nŴ)] .= @. +Ŵmax
    n += nŴ
    estim.con.b[(n+1):(n+nV̂)] .= @. -V̂min + estim.F
    n += nV̂
    estim.con.b[(n+1):(n+nV̂)] .= @. +V̂max - estim.F
    lincon = estim.optim[:linconstraint]
    set_normalized_rhs.(lincon, estim.con.b[estim.con.i_b])
end

"Set `b` excluding state and sensor noise bounds if `model` is not a [`LinModel`](@ref)."
function linconstraint!(estim::MovingHorizonEstimator, ::SimModel)
    Ŵmin, Ŵmax = trunc_bounds(estim, estim.con.Ŵmin, estim.con.Ŵmax, estim.nx̂)
    nx̃, nŴ = length(estim.con.x̃min), length(Ŵmin)
    n = 0
    estim.con.b[(n+1):(n+nx̃)] .= @. -estim.con.x̃min
    n += nx̃
    estim.con.b[(n+1):(n+nx̃)] .= @. +estim.con.x̃max
    n += nx̃
    estim.con.b[(n+1):(n+nŴ)] .= @. -Ŵmin
    n += nŴ
    estim.con.b[(n+1):(n+nŴ)] .= @. +Ŵmax
    lincon = estim.optim[:linconstraint]
    set_normalized_rhs.(lincon, estim.con.b[estim.con.i_b])
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
    uarr, ymarr, darr = @views estim.U[1:nu], estim.Ym[1:nym], estim.D[1:nd]
    estim.covestim.x̂      .= estim.x̂arr_old
    estim.covestim.P̂.data .= estim.P̂arr_old # .data is necessary for Hermitian
    update_estimate!(estim.covestim, uarr, ymarr, darr)
    estim.P̂arr_old.data   .= estim.covestim.P̂
    estim.invP̄.data       .= Hermitian(inv(estim.P̂arr_old), :L)
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
    x̂arr, Ŵ, V̂ = @views Z̃[nx̃-nx̂+1:nx̃], Z̃[nx̃+1:nx̃+nŴ], V̂[1:nYm]
    x̄ .= estim.x̂arr_old .- x̂arr
    Jϵ = nϵ ? estim.C*Z̃[begin]^2 : 0
    return dot(x̄, invP̄, x̄) + dot(Ŵ, invQ̂_Nk, Ŵ) + dot(V̂, invR̂_Nk, V̂) + Jϵ
end

"""
    predict!(V̂, X̂, û, ŷ, estim::MovingHorizonEstimator, model::LinModel, Z̃) -> V̂, X̂

Compute the `V̂` vector and `X̂` vectors for the `MovingHorizonEstimator` and `LinModel`.

The function mutates `V̂`, `X̂`, `û` and `ŷ` vector arguments. The vector `V̂` is the estimated
sensor noises `V̂` from ``k-N_k+1`` to ``k``. The `X̂` vector is estimated states from 
``k-N_k+2`` to ``k+1``.
"""
function predict!(V̂, X̂, _ , _ , estim::MovingHorizonEstimator, ::LinModel, Z̃) 
    Nk, nϵ = estim.Nk[], !isinf(estim.C)
    nX̂, nŴ, nYm = estim.nx̂*Nk, estim.nx̂*Nk, estim.nym*Nk
    nZ̃ = nϵ + estim.nx̂ + nŴ
    V̂[1:nYm] .= @views estim.Ẽ[1:nYm, 1:nZ̃]*Z̃[1:nZ̃] + estim.F[1:nYm]
    X̂[1:nX̂]  .= @views estim.con.Ẽx̂[1:nX̂, 1:nZ̃]*Z̃[1:nZ̃] + estim.con.Fx̂[1:nX̂]
    return V̂, X̂
end

"Compute the two vectors when `model` is not a `LinModel`."
function predict!(V̂, X̂, û, ŷ, estim::MovingHorizonEstimator, model::SimModel, Z̃)
    Nk = estim.Nk[]
    nu, nd, ny, nx̂, nŵ, nym = model.nu, model.nd, model.ny, estim.nx̂, estim.nx̂, estim.nym
    nx̃ = !isinf(estim.C) + nx̂
    x̂ = @views Z̃[nx̃-nx̂+1:nx̃]
    for j=1:Nk
        u  = @views estim.U[ (1 + nu  * (j-1)):(nu*j)]
        ym = @views estim.Ym[(1 + nym * (j-1)):(nym*j)]
        d  = @views estim.D[ (1 + nd  * (j-1)):(nd*j)]
        ŵ  = @views Z̃[(1 + nx̃ + nŵ*(j-1)):(nx̃ + nŵ*j)]
        ĥ!(ŷ, estim, model, x̂, d)
        ŷm = @views ŷ[estim.i_ym]
        V̂[(1 + nym*(j-1)):(nym*j)] .= ym .- ŷm
        x̂next = @views X̂[(1 + nx̂ *(j-1)):(nx̂ *j)]
        f̂!(x̂next, û, estim, model, x̂, u, d)
        x̂next .+= ŵ
        x̂ = x̂next
    end
    return V̂, X̂
end

"""
    con_nonlinprog!(g, estim::MovingHorizonEstimator, model::SimModel, X̂, V̂, Z̃)

Nonlinear constrains for [`MovingHorizonEstimator`](@ref).
"""
function con_nonlinprog!(g, estim::MovingHorizonEstimator, ::SimModel, X̂, V̂, Z̃)
    nX̂con, nX̂ = length(estim.con.X̂min), estim.nx̂ *estim.Nk[]
    nV̂con, nV̂ = length(estim.con.V̂min), estim.nym*estim.Nk[]
    ϵ = isinf(estim.C) ? 0 : Z̃[begin] # ϵ = 0 if Cwt=Inf (meaning: no relaxation)
    for i in eachindex(g)
        estim.con.i_g[i] || continue
        if i ≤ nX̂con
            j = i
            jcon = nX̂con-nX̂+j
            g[i] = j > nX̂ ? 0 : estim.con.X̂min[jcon] - X̂[j] - ϵ*estim.con.C_x̂min[jcon]
        elseif i ≤ 2nX̂con
            j = i - nX̂con
            jcon = nX̂con-nX̂+j
            g[i] = j > nX̂ ? 0 : X̂[j] - estim.con.X̂max[jcon] - ϵ*estim.con.C_x̂max[jcon]
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