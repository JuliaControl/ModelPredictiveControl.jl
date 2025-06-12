"Reset the data windows and time-varying variables for the moving horizon estimator."
function init_estimate_cov!(estim::MovingHorizonEstimator, _ , d0, u0) 
    estim.Z̃         .= 0
    estim.X̂0        .= 0
    estim.Y0m       .= 0
    estim.U0        .= 0
    estim.D0        .= 0
    estim.Ŵ         .= 0
    estim.Nk        .= 0
    estim.H̃         .= 0
    estim.q̃         .= 0
    estim.r         .= 0
    if estim.direct
        # add u0(-1) and d0(-1) to the data windows:
        estim.U0[1:estim.model.nu] .= u0
        estim.D0[1:estim.model.nd] .= d0
    end
    estim.lastu0 .= u0
    # estim.P̂_0 is in fact P̂(-1|-1) is estim.direct==false, else P̂(-1|0)
    invert_cov!(estim, estim.P̂_0)
    estim.P̂arr_old  .= estim.P̂_0
    estim.x̂0arr_old .= 0
    return nothing
end

"""
    correct_estimate!(estim::MovingHorizonEstimator, y0m, d0)

Do the same but for [`MovingHorizonEstimator`](@ref) objects.
"""
function correct_estimate!(estim::MovingHorizonEstimator, y0m, d0)
    if estim.direct
        ismoving = add_data_windows!(estim, y0m, d0)
        ismoving && correct_cov!(estim)
        initpred!(estim, estim.model)
        linconstraint!(estim, estim.model)
        optim_objective!(estim)
    end
    return nothing
end

@doc raw"""
    update_estimate!(estim::MovingHorizonEstimator, y0m, d0, u0)
    
Update [`MovingHorizonEstimator`](@ref) state `estim.x̂0`.

The optimization problem of [`MovingHorizonEstimator`](@ref) documentation is solved if
`estim.direct` is `false` (otherwise solved in [`correct_estimate!`](@ref)). The prediction
matrices are provided at [`init_predmat_mhe`](@ref) documentation. Once solved, the optimal
estimate ``\mathbf{x̂}_k(k+p)`` is computed by inserting the optimal values of 
``\mathbf{x̂}_k(k-N_k+p)`` and ``\mathbf{Ŵ}`` in the augmented model from ``j = N_k-1`` to
``0`` inclusively. Afterward, if ``N_k = H_e``, the arrival covariance for the next time
step ``\mathbf{P̂}_{k-N_k}(k-N_k+1)`` is estimated using `estim.covestim` object. It
also stores `u0` at `estim.lastu0`, so it can be added to the data window at the next time
step in [`correct_estimate!`](@ref).
"""
function update_estimate!(estim::MovingHorizonEstimator, y0m, d0, u0)
    if !estim.direct
        add_data_windows!(estim, y0m, d0, u0)
        initpred!(estim, estim.model)
        linconstraint!(estim, estim.model)
        optim_objective!(estim)
    end
    (estim.Nk[] == estim.He) && update_cov!(estim)
    estim.lastu0 .= u0
    return nothing
end

@doc raw"""
    getinfo(estim::MovingHorizonEstimator) -> info

Get additional info on `estim` [`MovingHorizonEstimator`](@ref) optimum for troubleshooting.

If `estim.direct==true`, the function should be called after calling [`preparestate!`](@ref).
Otherwise, call it after [`updatestate!`](@ref). It returns the dictionary `info` with the
following fields:

!!! info
    Fields with *`emphasis`* are non-Unicode alternatives.

- `:Ŵ` or *`:What`* : optimal estimated process noise over ``N_k``, ``\mathbf{Ŵ}``
- `:ϵ` or *`:epsilon`* : optimal slack variable, ``ϵ``
- `:X̂` or *`:Xhat`* : optimal estimated states over ``N_k+1``, ``\mathbf{X̂}``
- `:x̂` or *`:xhat`* : optimal estimated state, ``\mathbf{x̂}_k(k+p)``
- `:V̂` or *`:Vhat`* : optimal estimated sensor noise over ``N_k``, ``\mathbf{V̂}``
- `:P̄` or *`:Pbar`* : estimation error covariance at arrival, ``\mathbf{P̄}``
- `:x̄` or *`:xbar`* : optimal estimation error at arrival, ``\mathbf{x̄}``
- `:Ŷ` or *`:Yhat`* : optimal estimated outputs over ``N_k``, ``\mathbf{Ŷ}``
- `:Ŷm` or *`:Yhatm`* : optimal estimated measured outputs over ``N_k``, ``\mathbf{Ŷ^m}``
- `:x̂arr` or *`:xhatarr`* : optimal estimated state at arrival, ``\mathbf{x̂}_k(k-N_k+p)``
- `:J`   : objective value optimum, ``J``
- `:Ym`  : measured outputs over ``N_k``, ``\mathbf{Y^m}``
- `:U`   : manipulated inputs over ``N_k``, ``\mathbf{U}``
- `:D`   : measured disturbances over ``N_k``, ``\mathbf{D}``
- `:sol` : solution summary of the optimizer for printing

# Examples
```jldoctest
julia> model = LinModel(ss(1.0, 1.0, 1.0, 0, 5.0));

julia> estim = MovingHorizonEstimator(model, He=1, nint_ym=0, direct=false);

julia> updatestate!(estim, [0], [1]);

julia> round.(getinfo(estim)[:Ŷ], digits=3)
1-element Vector{Float64}:
 0.5
```
"""
function getinfo(estim::MovingHorizonEstimator{NT}) where NT<:Real
    model, buffer, Nk = estim.model, estim.buffer, estim.Nk[]
    nu, ny, nd = model.nu, model.ny, model.nd
    nx̂, nym, nŵ, nϵ = estim.nx̂, estim.nym, estim.nx̂, estim.nϵ
    nx̃ = nϵ + nx̂
    info = Dict{Symbol, Any}()
    V̂,  X̂0 = similar(estim.Y0m[1:nym*Nk]), similar(estim.X̂0[1:nx̂*Nk])
    û0, k0, ŷ0 = buffer.û, buffer.k, buffer.ŷ
    V̂,  X̂0 = predict!(V̂, X̂0, û0, k0, ŷ0, estim, model, estim.Z̃)
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
    info[:ϵ]  = nϵ ≠ 0 ? estim.Z̃[begin] : zero(NT)
    info[:J]  = obj_nonlinprog!(x̄, estim, estim.model, V̂, estim.Z̃)
    info[:X̂]  = X̂0       .+ @views [estim.x̂op; estim.X̂op[1:nx̂*Nk]]
    info[:x̂]  = estim.x̂0 .+ estim.x̂op
    info[:V̂]  = V̂
    info[:P̄]  = estim.P̂arr_old
    info[:x̄]  = x̄
    info[:Ŷ]  = Ŷ
    info[:Ŷm] = Ŷm
    info[:Ym] = Ym
    info[:U]  = U 
    info[:D]  = D
    info[:sol] = JuMP.solution_summary(estim.optim, verbose=true)
    # --- non-Unicode fields ---
    info[:What] = info[:Ŵ]
    info[:xhatarr] = info[:x̂arr]
    info[:epsilon] = info[:ϵ]
    info[:Xhat] = info[:X̂]
    info[:xhat] = info[:x̂]
    info[:Vhat] = info[:V̂]
    info[:Pbar] = info[:P̄]
    info[:xbar] = info[:x̄]
    info[:Yhat] = info[:Ŷ]
    info[:Yhatm] = info[:Ŷm]
    return info
end

"""
    getϵ(estim::MovingHorizonEstimator, Z̃) -> ϵ

Get the slack `ϵ` from the decision vector `Z̃` if present, otherwise return 0.
"""
function getϵ(estim::MovingHorizonEstimator, Z̃::AbstractVector{NT}) where NT<:Real
    return estim.nϵ ≠ 0 ? Z̃[begin] : zero(NT)
end

"""
    add_data_windows!(estim::MovingHorizonEstimator, y0m, d0, u0=estim.lastu0) -> ismoving

Add data to the observation windows of the moving horizon estimator and clamp `estim.Nk`.

If ``k ≥ H_e``, the observation windows are moving in time and `estim.Nk` is clamped to
`estim.He`. It returns `true` if the observation windows are moving, `false` otherwise.
If no `u0` argument is provided, the manipulated input of the last time step is added to its
window (the correct value if `estim.direct == true`).
"""
function add_data_windows!(estim::MovingHorizonEstimator, y0m, d0, u0=estim.lastu0)
    model = estim.model
    nx̂, nym, nd, nu, nŵ = estim.nx̂, estim.nym, model.nd, model.nu, estim.nx̂
    Nk = estim.Nk[]
    p = estim.direct ? 0 : 1
    x̂0, ŵ = estim.x̂0, 0 # ŵ(k-1+p) = 0 for warm-start
    estim.Nk .+= 1
    Nk = estim.Nk[]
    ismoving = (Nk > estim.He)
    if ismoving
        estim.Y0m[1:end-nym]     .= @views estim.Y0m[nym+1:end]
        estim.Y0m[end-nym+1:end] .= y0m
        if nd > 0
            estim.D0[1:end-nd]       .= @views estim.D0[nd+1:end]
            estim.D0[end-nd+1:end]   .= d0
        end
        estim.U0[1:end-nu]       .= @views estim.U0[nu+1:end]
        estim.U0[end-nu+1:end]   .= u0
        estim.X̂0[1:end-nx̂]       .= @views estim.X̂0[nx̂+1:end]
        estim.X̂0[end-nx̂+1:end]   .= x̂0
        estim.Ŵ[1:end-nŵ]        .= @views estim.Ŵ[nŵ+1:end]
        estim.Ŵ[end-nŵ+1:end]    .= ŵ
        estim.Nk .= estim.He
    else
        estim.Y0m[(1 + nym*(Nk-1)):(nym*Nk)]  .= y0m
        if nd > 0
            # D0 include 1 additional measured disturbance if direct==true (p==0):
            estim.D0[(1 + nd*(Nk-p)):(nd*Nk+1-p)] .= d0 
        end  
        estim.U0[(1 + nu*(Nk-1)):(nu*Nk)]     .= u0
        estim.X̂0[(1 + nx̂*(Nk-1)):(nx̂*Nk)]     .= x̂0
        estim.Ŵ[(1 + nŵ*(Nk-1)):(nŵ*Nk)]      .= ŵ
    end
    estim.x̂0arr_old .= @views estim.X̂0[1:nx̂]
    return ismoving
end
    
@doc raw"""
    initpred!(estim::MovingHorizonEstimator, model::LinModel) -> nothing

Init quadratic optimization matrices `F, fx̄, H̃, q̃, r` for [`MovingHorizonEstimator`](@ref).

See [`init_predmat_mhe`](@ref) for the definition of the vectors ``\mathbf{F, f_x̄}``. It
also inits `estim.optim` objective function, expressed as the quadratic general form:
```math
    J = \min_{\mathbf{Z̃}} \frac{1}{2}\mathbf{Z̃' H̃ Z̃} + \mathbf{q̃' Z̃} + r 
```
in which ``\mathbf{Z̃} = [\begin{smallmatrix} ϵ \\ \mathbf{Z} \end{smallmatrix}]``. Note that
``r`` is useless at optimization but required to evaluate the objective minima ``J``. The 
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
            r        &= \mathbf{F_Z̃}' \mathbf{M}_{N_k} \mathbf{F_Z̃}
\end{aligned}
```
"""
function initpred!(estim::MovingHorizonEstimator, model::LinModel)
    F, C, optim = estim.F, estim.C, estim.optim
    nx̂, nŵ, nym, nϵ, Nk = estim.nx̂, estim.nx̂, estim.nym, estim.nϵ, estim.Nk[]
    nYm, nŴ = nym*Nk, nŵ*Nk
    nZ̃ = nϵ + nx̂ + nŴ
    # --- update F and fx̄ vectors for MHE predictions ---
    F .= estim.Y0m .+ estim.B
    mul!(F, estim.G, estim.U0, 1, 1)
    if model.nd > 0
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
    estim.r .= dot(FZ̃, M_Nk, FZ̃)
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
    if model.nd > 0
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

@doc raw"""
    optim_objective!(estim::MovingHorizonEstimator) -> Z̃

Optimize objective of `estim` [`MovingHorizonEstimator`](@ref) and return the solution `Z̃`.

If first warm-starts the solver with [`set_warmstart_mhe!`](@ref). It then calls 
`JuMP.optimize!(estim.optim)` and extract the solution. A failed optimization prints an 
`@error` log in the REPL and returns the warm-start value. A failed optimization also prints
[`getinfo`](@ref) results in the debug log [if activated](@extref Julia Example:-Enable-debug-level-messages).
"""
function optim_objective!(estim::MovingHorizonEstimator{NT}) where NT<:Real
    model, optim, buffer = estim.model, estim.optim, estim.buffer
    nym, nx̂, nŵ, nϵ, Nk = estim.nym, estim.nx̂, estim.nx̂, estim.nϵ, estim.Nk[]
    nx̃ = nϵ + nx̂
    Z̃var::Vector{JuMP.VariableRef} = optim[:Z̃var]
    V̂   = Vector{NT}(undef, nym*Nk)     # TODO: remove this allocation
    X̂0  = Vector{NT}(undef, nx̂*Nk)      # TODO: remove this allocation
    Z̃s = set_warmstart_mhe!(V̂, X̂0, estim, Z̃var)
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
            @error(
                "MHE terminated without solution: estimation in open-loop "*
                "(more info in debug log)",
                status
            )
        else
            @warn(
                "MHE termination status not OPTIMAL or LOCALLY_SOLVED: keeping solution "*
                "anyway (more info in debug log)", 
                status
            )
        end
        @debug info2debugstr(getinfo(estim))
    end
    if iserror(optim)
        estim.Z̃ .= Z̃s
    else
        estim.Z̃ .= JuMP.value.(Z̃var)
    end
    # --------- update estimate -----------------------
    û0, ŷ0, k0 = buffer.û, buffer.ŷ, buffer.k
    estim.Ŵ[1:nŵ*Nk] .= @views estim.Z̃[nx̃+1:nx̃+nŵ*Nk] # update Ŵ with optimum for warm-start
    V̂, X̂0 = predict!(V̂, X̂0, û0, k0, ŷ0, estim, model, estim.Z̃)
    x̂0next    = @views X̂0[end-nx̂+1:end] 
    estim.x̂0 .= x̂0next
    return estim.Z̃
end

@doc raw"""
    set_warmstart_mhe!(V̂, X̂0, estim::MovingHorizonEstimator, Z̃var) -> Z̃s

Set and return the warm-start value of `Z̃var` for [`MovingHorizonEstimator`](@ref).

If supported by `estim.optim`, it warm-starts the solver at:
```math
\mathbf{Z̃_s} = 
\begin{bmatrix}
    ϵ_{k-1}                         \\
    \mathbf{x̂}_{k-1}(k-N_k+p)       \\ 
    \mathbf{ŵ}_{k-1}(k-N_k+p+0)     \\ 
    \mathbf{ŵ}_{k-1}(k-N_k+p+1)     \\ 
    \vdots                          \\
    \mathbf{ŵ}_{k-1}(k-p-2)         \\
    \mathbf{0}                      \\
\end{bmatrix}
```
where ``ϵ(k-1)``, ``\mathbf{x̂}_{k-1}(k-N_k+p)`` and ``\mathbf{ŵ}_{k-1}(k-j)`` are
respectively the slack variable, the arrival state estimate and the process noise estimates
computed at the last time step ``k-1``. If the objective function is not finite at this
point, all the process noises ``\mathbf{ŵ}_{k-1}(k-j)`` are warm-started at zeros. The
method mutates all the arguments.
"""
function set_warmstart_mhe!(V̂, X̂0, estim::MovingHorizonEstimator{NT}, Z̃var) where NT<:Real
    model, buffer = estim.model, estim.buffer
    nϵ, nx̂, nŵ, nZ̃, Nk = estim.nϵ, estim.nx̂, estim.nx̂, length(estim.Z̃), estim.Nk[]
    nx̃ = nϵ + nx̂
    Z̃s  = Vector{NT}(undef, nZ̃)  # TODO: remove this allocation
    û0, ŷ0, x̄, k0 = buffer.û, buffer.ŷ, buffer.x̂, buffer.k
    # --- slack variable ϵ ---
    estim.nϵ == 1 && (Z̃s[begin] = estim.Z̃[begin])
    # --- arrival state estimate x̂0arr ---
    Z̃s[nϵ+1:nx̃] = estim.x̂0arr_old
    # --- process noise estimates Ŵ ---
    Z̃s[nx̃+1:end] = estim.Ŵ
    # verify definiteness of objective function:
    V̂, X̂0 = predict!(V̂, X̂0, û0, k0, ŷ0, estim, model, Z̃s)
    Js = obj_nonlinprog!(x̄, estim, model, V̂, Z̃s)
    if !isfinite(Js)
        Z̃s[nx̃+1:end] = 0
    end
    # --- unused variable in Z̃ (applied only when Nk ≠ He) ---
    # We force the update of the NLP gradient and jacobian by warm-starting the unused 
    # variable in Z̃ at 1. Since estim.Ŵ is initialized with 0s, at least 1 variable in Z̃s
    # will be inevitably different at the following time step.
    Z̃s[nx̃+Nk*nŵ+1:end] .= 1
    JuMP.set_start_value.(Z̃var, Z̃s)
end

"Correct the covariance estimate at arrival using `covestim` [`StateEstimator`](@ref)."
function correct_cov!(estim::MovingHorizonEstimator)
    nym, nd = estim.nym, estim.model.nd
    buffer = estim.covestim.buffer
    y0marr, d0arr = buffer.ym, buffer.d
    y0marr .= @views estim.Y0m[1:nym]
    d0arr  .= @views estim.D0[1:nd]
    estim.covestim.x̂0 .= estim.x̂0arr_old
    estim.covestim.P̂  .= estim.P̂arr_old
    try
        correct_estimate!(estim.covestim, y0marr, d0arr)
        all(isfinite, estim.covestim.P̂) || error("Arrival covariance P̄ is not finite")
        estim.P̂arr_old .= estim.covestim.P̂
        invert_cov!(estim, estim.P̂arr_old)
    catch err
        if err isa PosDefException
            @error("Arrival covariance P̄ is not positive definite: keeping the old one")
        elseif err isa ErrorException
            @error("Arrival covariance P̄ is not finite: keeping the old one")
        else
            rethrow()
        end
    end
    return nothing
end

"Update the covariance estimate at arrival using `covestim` [`StateEstimator`](@ref)."
function update_cov!(estim::MovingHorizonEstimator)
    nu, nd, nym = estim.model.nu, estim.model.nd, estim.nym
    buffer = estim.covestim.buffer
    u0arr, y0marr, d0arr = buffer.u, buffer.ym, buffer.d
    u0arr  .= @views estim.U0[1:nu]
    y0marr .= @views estim.Y0m[1:nym]
    d0arr  .= @views estim.D0[1:nd]
    estim.covestim.x̂0 .= estim.x̂0arr_old
    estim.covestim.P̂  .= estim.P̂arr_old
    try
        update_estimate!(estim.covestim, y0marr, d0arr, u0arr)
        all(isfinite, estim.covestim.P̂) || error("Arrival covariance P̄ is not finite")
        estim.P̂arr_old .= estim.covestim.P̂
        invert_cov!(estim, estim.P̂arr_old)
    catch err
        if err isa PosDefException
            @error("Arrival covariance P̄ is not positive definite: keeping the old one")
        elseif err isa ErrorException
            @error("Arrival covariance P̄ is not finite: keeping the old one")
        else
            rethrow()
        end
    end
    return nothing
end

"Invert the covariance estimate at arrival `P̄`."
function invert_cov!(estim::MovingHorizonEstimator, P̄)
    invP̄  = Hermitian(estim.buffer.P̂, :L)
    invP̄ .= P̄
    try
        inv!(invP̄)
    catch err
        if err isa PosDefException
            @error("Arrival covariance P̄ is not invertible: keeping the old one")
        else
            rethrow()
        end
    end
    return nothing
end

"""
    obj_nonlinprog!( _ , estim::MovingHorizonEstimator, ::LinModel, _ , Z̃) 

Objective function of [`MovingHorizonEstimator`](@ref) when `model` is a [`LinModel`](@ref).

It can be called on a [`MovingHorizonEstimator`](@ref) object to evaluate the objective 
function at specific `Z̃` and `V̂` values.
"""
function obj_nonlinprog!(
    _ , estim::MovingHorizonEstimator, ::LinModel, _ , Z̃::AbstractVector{NT}
) where NT<:Real
    return obj_quadprog(Z̃, estim.H̃, estim.q̃) + estim.r[]
end

"""
    obj_nonlinprog!(x̄, estim::MovingHorizonEstimator, model::SimModel, V̂, Z̃)

Objective function of the MHE when `model` is not a [`LinModel`](@ref).

The function `dot(x, A, x)` is a performant way of calculating `x'*A*x`. This method mutates
`x̄` vector arguments.
"""
function obj_nonlinprog!(
    x̄, estim::MovingHorizonEstimator, ::SimModel, V̂, Z̃::AbstractVector{NT}
) where NT<:Real
    nϵ, Nk = estim.nϵ, estim.Nk[] 
    nYm, nŴ, nx̂, invP̄ = Nk*estim.nym, Nk*estim.nx̂, estim.nx̂, estim.invP̄
    nx̃ = nϵ + nx̂
    invQ̂_Nk, invR̂_Nk = @views estim.invQ̂_He[1:nŴ, 1:nŴ], estim.invR̂_He[1:nYm, 1:nYm]
    x̂0arr, Ŵ, V̂ = @views Z̃[nx̃-nx̂+1:nx̃], Z̃[nx̃+1:nx̃+nŴ], V̂[1:nYm]
    x̄ .= estim.x̂0arr_old .- x̂0arr
    Jϵ = nϵ ≠ 0 ? estim.C*Z̃[begin]^2 : zero(NT)
    return dot(x̄, invP̄, x̄) + dot(Ŵ, invQ̂_Nk, Ŵ) + dot(V̂, invR̂_Nk, V̂) + Jϵ
end

"""
    predict!(V̂, X̂0, û0, k0, ŷ0, estim::MovingHorizonEstimator, model::LinModel, Z̃) -> V̂, X̂0

Compute the `V̂` vector and `X̂0` vectors for the `MovingHorizonEstimator` and `LinModel`.

The function mutates `V̂`, `X̂0`, `û0` and `ŷ0` vector arguments. The vector `V̂` is the
estimated sensor noises from ``k-N_k+1`` to ``k``. The `X̂0` vector is estimated states from 
``k-N_k+2`` to ``k+1``.
"""
function predict!(V̂, X̂0, _ , _ , _ , estim::MovingHorizonEstimator, ::LinModel, Z̃) 
    nϵ, Nk = estim.nϵ, estim.Nk[]
    nX̂, nŴ, nYm = estim.nx̂*Nk, estim.nx̂*Nk, estim.nym*Nk
    nZ̃ = nϵ + estim.nx̂ + nŴ
    V̂[1:nYm] .= @views estim.Ẽ[1:nYm, 1:nZ̃]*Z̃[1:nZ̃]     + estim.F[1:nYm]
    X̂0[1:nX̂] .= @views estim.con.Ẽx̂[1:nX̂, 1:nZ̃]*Z̃[1:nZ̃] + estim.con.Fx̂[1:nX̂]
    return V̂, X̂0
end

"Compute the two vectors when `model` is not a `LinModel`."
function predict!(V̂, X̂0, û0, k0, ŷ0, estim::MovingHorizonEstimator, model::SimModel, Z̃)
    nϵ, Nk = estim.nϵ, estim.Nk[]
    nu, nd, nx̂, nŵ, nym = model.nu, model.nd, estim.nx̂, estim.nx̂, estim.nym
    nx̃ = nϵ + nx̂
    x̂0 = @views Z̃[nx̃-nx̂+1:nx̃]
    if estim.direct
        ŷ0next = ŷ0
        d0 = @views estim.D0[1:nd]
        for j=1:Nk
            u0  = @views estim.U0[ (1 + nu  * (j-1)):(nu*j)]
            ŵ   = @views Z̃[(1 + nx̃ + nŵ*(j-1)):(nx̃ + nŵ*j)]
            x̂0next = @views X̂0[(1 + nx̂ *(j-1)):(nx̂ *j)]
            f̂!(x̂0next, û0, k0, estim, model, x̂0, u0, d0)
            x̂0next .+= ŵ .+ estim.f̂op .- estim.x̂op
            y0nextm = @views estim.Y0m[(1 + nym * (j-1)):(nym*j)]
            d0next  = @views estim.D0[(1 + nd*j):(nd*(j+1))]
            ĥ!(ŷ0next, estim, model, x̂0next, d0next)
            ŷ0nextm = @views ŷ0next[estim.i_ym]
            V̂[(1 + nym*(j-1)):(nym*j)] .= y0nextm .- ŷ0nextm
            x̂0, d0 = x̂0next, d0next
        end        
    else
        for j=1:Nk
            y0m = @views estim.Y0m[(1 + nym * (j-1)):(nym*j)]
            d0  = @views estim.D0[ (1 + nd  * (j-1)):(nd*j)]
            u0  = @views estim.U0[ (1 + nu  * (j-1)):(nu*j)]
            ŵ   = @views Z̃[(1 + nx̃ + nŵ*(j-1)):(nx̃ + nŵ*j)]
            ĥ!(ŷ0, estim, model, x̂0, d0)
            ŷ0m = @views ŷ0[estim.i_ym]
            V̂[(1 + nym*(j-1)):(nym*j)] .= y0m .- ŷ0m
            x̂0next = @views X̂0[(1 + nx̂ *(j-1)):(nx̂ *j)]
            f̂!(x̂0next, û0, k0, estim, model, x̂0, u0, d0)
            x̂0next .+= ŵ .+ estim.f̂op .- estim.x̂op
            x̂0 = x̂0next
        end
    end
    return V̂, X̂0
end


"""
    update_predictions!(V̂, X̂0, û0, k0, ŷ0, g, estim::MovingHorizonEstimator, Z̃)

Update in-place the vectors for the predictions of `estim` estimator at decision vector `Z̃`.

The method mutates all the arguments before `estim` argument.
"""
function update_prediction!(V̂, X̂0, û0, k0, ŷ0, g, estim::MovingHorizonEstimator, Z̃)
    model = estim.model
    V̂, X̂0  = predict!(V̂, X̂0, û0, k0, ŷ0, estim, model, Z̃)
    ϵ = getϵ(estim, Z̃)
    g = con_nonlinprog!(g, estim, model, X̂0, V̂, ϵ)
    return nothing
end

"""
    con_nonlinprog!(g, estim::MovingHorizonEstimator, model::SimModel, X̂0, V̂, ϵ)

Nonlinear constrains for [`MovingHorizonEstimator`](@ref).
"""
function con_nonlinprog!(g, estim::MovingHorizonEstimator, ::SimModel, X̂0, V̂, ϵ)
    nX̂con, nX̂ = length(estim.con.X̂0min), estim.nx̂ *estim.Nk[]
    nV̂con, nV̂ = length(estim.con.V̂min),  estim.nym*estim.Nk[]
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

"No nonlinear constraints if `model` is a [`LinModel`](@ref), return `g` unchanged."
con_nonlinprog!(g, ::MovingHorizonEstimator, ::LinModel, _ , _ , _ ) = g

"Throw an error if P̂ != nothing."
function setstate_cov!(estim::MovingHorizonEstimator, P̂)
    P̂ == nothing || error("MovingHorizonEstimator does not compute an estimation covariance matrix P̂.")
    return nothing
end

"Update the augmented model, prediction matrices, constrains and data windows for MHE."
function setmodel_estimator!(
    estim::MovingHorizonEstimator, model, uop_old, yop_old, dop_old, Q̂, R̂
)
    con = estim.con
    nx̂, nym, nu, nd, He, nϵ = estim.nx̂, estim.nym, model.nu, model.nd, estim.He, estim.nϵ
    As, Cs_u, Cs_y = estim.As, estim.Cs_u, estim.Cs_y
    Â, B̂u, Ĉ, B̂d, D̂d, x̂op, f̂op = augment_model(model, As, Cs_u, Cs_y, verify_obsv=false)
    # --- update augmented state-space matrices ---
    estim.Â  .= Â
    estim.B̂u .= B̂u
    estim.Ĉ  .= Ĉ
    estim.B̂d .= B̂d
    estim.D̂d .= D̂d
    estim.Ĉm  .= @views Ĉ[estim.i_ym, :]
    estim.D̂dm .= @views D̂d[estim.i_ym, :]
    # --- update state estimate and its operating points ---
    x̂op_old = copy(estim.x̂op)
    estim.x̂0 .+= estim.x̂op # convert x̂0 to x̂ with the old operating point
    estim.x̂op .= x̂op
    estim.f̂op .= f̂op
    estim.x̂0 .-= estim.x̂op # convert x̂ to x̂0 with the new operating point
    # --- predictions matrices ---
    p = estim.direct ? 0 : 1
    E, G, J, B, _ , Ex̂, Gx̂, Jx̂, Bx̂ = init_predmat_mhe(
        model, He, estim.i_ym, 
        estim.Â, estim.B̂u, estim.Ĉm, estim.B̂d, estim.D̂dm, 
        estim.x̂op, estim.f̂op, p
    )
    A_X̂min, A_X̂max, Ẽx̂ = relaxX̂(model, nϵ, con.C_x̂min, con.C_x̂max, Ex̂)   
    A_V̂min, A_V̂max, Ẽ  = relaxV̂(model, nϵ, con.C_v̂min, con.C_v̂max, E) 
    estim.Ẽ .= Ẽ
    estim.G .= G
    estim.J .= J
    estim.B .= B
    # --- linear inequality constraints ---
    con.Ẽx̂ .= Ẽx̂
    con.Gx̂ .= Gx̂
    con.Jx̂ .= Jx̂
    con.Bx̂ .= Bx̂
    # convert x̃0 to x̃ with the old operating point:
    con.x̃0min[end-nx̂+1:end] .+= x̂op_old 
    con.x̃0max[end-nx̂+1:end] .+= x̂op_old
    # convert X̂0 to X̂ with the old operating point:
    con.X̂0min .+= estim.X̂op
    con.X̂0max .+= estim.X̂op
    for i in 0:He-1
        estim.X̂op[(1+nx̂*i):(nx̂+nx̂*i)] .= estim.x̂op
    end
    # convert x̃ to x̃0 with the new operating point:
    con.x̃0min[end-nx̂+1:end] .-= estim.x̂op 
    con.x̃0max[end-nx̂+1:end] .-= estim.x̂op 
    # convert X̂ to X̂0 with the new operating point:
    con.X̂0min .-= estim.X̂op
    con.X̂0max .-= estim.X̂op
    con.A_X̂min .= A_X̂min
    con.A_X̂max .= A_X̂max
    con.A_V̂min .= A_V̂min
    con.A_V̂max .= A_V̂max
    con.A .= [
        con.A_x̃min
        con.A_x̃max
        con.A_X̂min
        con.A_X̂max
        con.A_Ŵmin
        con.A_Ŵmax
        con.A_V̂min
        con.A_V̂max
    ]
    A = con.A[con.i_b, :]
    b = zeros(count(con.i_b)) # dummy value, updated before optimization (avoid ±Inf)
    Z̃var::Vector{JuMP.VariableRef} = estim.optim[:Z̃var]
    JuMP.delete(estim.optim, estim.optim[:linconstraint])
    JuMP.unregister(estim.optim, :linconstraint)
    @constraint(estim.optim, linconstraint, A*Z̃var .≤ b)
    # --- data windows ---
    for i in 1:He
        # convert x̂0 to x̂ with the old operating point:
        estim.X̂0[(1+nx̂*(i-1)):(nx̂*i)]    .+= x̂op_old 
        # convert y0m to ym with the old operating point:
        estim.Y0m[(1+nym*(i-1)):(nym*i)] .+= @views yop_old[estim.i_ym]
        # convert u0 to u with the old operating point:
        estim.U0[(1+nu*(i-1)):(nu*i)]    .+= uop_old
        # convert d0 to d with the old operating point:
        estim.D0[(1+nd*(i-1)):(nd*i)]    .+= dop_old
        # convert x̂ to x̂0 with the new operating point:
        estim.X̂0[(1+nx̂*(i-1)):(nx̂*i)]    .-= x̂op
        # convert ym to y0m with the new operating point:
        estim.Y0m[(1+nym*(i-1)):(nym*i)] .-= @views model.yop[estim.i_ym]
        # convert u to u0 with the new operating point:
        estim.U0[(1+nu*(i-1)):(nu*i)]    .-= model.uop
        # convert d to d0 with the new operating point:
        estim.D0[(1+nd*(i-1)):(nd*i)]    .-= model.dop
    end
    estim.lastu0        .+= uop_old
    estim.Z̃[nϵ+1:nϵ+nx̂] .+= x̂op_old
    estim.x̂0arr_old     .+= x̂op_old
    estim.lastu0        .-= model.uop
    estim.Z̃[nϵ+1:nϵ+nx̂] .-= x̂op
    estim.x̂0arr_old     .-= x̂op
    # --- covariance matrices ---
    if !isnothing(Q̂)
        estim.Q̂ .= to_hermitian(Q̂)
        invQ̂  = Hermitian(estim.buffer.Q̂, :L)
        invQ̂ .= estim.Q̂
        inv!(invQ̂)
        estim.invQ̂_He .= Hermitian(repeatdiag(invQ̂, He), :L)
    end
    if !isnothing(R̂) 
        estim.R̂ .= to_hermitian(R̂)
        invR̂  = Hermitian(estim.buffer.R̂, :L)
        invR̂ .= estim.R̂
        inv!(invR̂)
        estim.invR̂_He .= Hermitian(repeatdiag(invR̂, He), :L)
    end
    return nothing
end

"Called by plots recipes for the estimated states constraints."
getX̂con(estim::MovingHorizonEstimator, _ ) = estim.con.X̂0min+estim.X̂op, estim.con.X̂0max+estim.X̂op