"Reset the data windows and time-varying variables for the moving horizon estimator."
function init_estimate_cov!(estim::MovingHorizonEstimator, _ , _ , _ ) 
    estim.invP̄.data[:]        = inv(estim.P̂0)
    estim.P̂arr_old.data[:]    = estim.P̂0
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
    nx̂, nym, nŵ, Nk = estim.nx̂, estim.nym, estim.nx̂, estim.Nk[]
    nx̃ = !isinf(estim.C) + nx̂
    Z̃var::Vector{VariableRef} = optim[:Z̃var]
    V̂  = Vector{NT}(undef, nym*Nk)
    X̂  = Vector{NT}(undef, nx̂*Nk)
    ϵ0 = isinf(estim.C) ? empty(estim.Z̃) : estim.Z̃[begin]
    Z̃0 = [ϵ0; estim.x̂arr_old; estim.Ŵ]
    V̂, X̂ = predict!(V̂, X̂, estim, model, Z̃0)
    J0 = obj_nonlinprog(estim, model, V̂, Z̃0)
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
    estim.Z̃[:] = iserror(optim) ? Z̃last : Z̃curr
    # --------- update estimate -----------------------
    estim.Ŵ[1:nŵ*Nk] = estim.Z̃[nx̃+1:nx̃+nŵ*Nk] # update Ŵ with optimum for warm-start
    V̂, X̂ = predict!(V̂, X̂, estim, model, estim.Z̃)
    x̂[:] = X̂[end-nx̂+1:end]
    if Nk == estim.He
        uarr, ymarr, darr = estim.U[1:model.nu], estim.Ym[1:nym], estim.D[1:model.nd]
        update_cov!(estim.P̂arr_old, estim, model, uarr, ymarr, darr)
        estim.invP̄.data[:] = Hermitian(inv(estim.P̂arr_old), :L)
    end
    return nothing
end


@doc raw"""
    getinfo(estim::MovingHorizonEstimator) -> info

Get additional info on `estim` [`MovingHorizonEstimator`](@ref) optimum for troubleshooting.

The function should be called after calling [`updatestate!`](@ref). It returns the
dictionary `info` with the following fields:

- `:Ŵ`   : optimal estimated process noise over ``N_k``, ``\mathbf{Ŵ}``.
- `:x̂arr`: optimal estimated state at arrival, ``\mathbf{x̂}_k(k-N_k+1)``.
- `:ϵ`   : optimal slack variable, ``ϵ``.
- `:J`   : objective value optimum, ``J``.
- `:X̂`   : optimal estimated states over ``N_k+1``, ``\mathbf{X̂}``.
- `:x̂`   : optimal estimated state for the next time step, ``\mathbf{x̂}_k(k+1)``.
- `:V̂`   : optimal estimated sensor noise over ``N_k``, ``\mathbf{V̂}``.
- `:P̄`   : estimation error covariance at arrival, ``\mathbf{P̄}``.
- `:x̄`   : optimal estimation error at arrival, ``\mathbf{x̄}``.
- `:Ŷ`   : optimal estimated outputs over ``N_k``, ``\mathbf{Ŷ}``.
- `:Ŷm`  : optimal estimated measured outputs over ``N_k``, ``\mathbf{Ŷ^m}``.
- `:Ym`  : measured outputs over ``N_k``, ``\mathbf{Y^m}``.
- `:U`   : manipulated inputs over ``N_k``, ``\mathbf{U}``.
- `:D`   : measured disturbances over ``N_k``, ``\mathbf{D}``.
- `:sol` : solution summary of the optimizer for printing.

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
    V̂, X̂ = predict!(V̂, X̂, estim, model, estim.Z̃)
    x̂arr = estim.Z̃[nx̃-nx̂+1:nx̃]
    X̂ = [x̂arr; X̂]
    Ym, U, D = estim.Ym[1:nym*Nk], estim.U[1:nu*Nk], estim.D[1:nd*Nk]
    Ŷ = Vector{NT}(undef, ny*Nk)
    for i=1:Nk
        d = @views D[(1 + nd*(i-1)):(nd*i)] # operating point already removed in estim.D
        x̂ = @views X̂[(1 + nx̂*(i-1)):(nx̂*i)]
        Ŷ[(1 + ny*(i-1)):(ny*i)] = ĥ(estim, model, x̂, d) + model.yop
    end
    Ŷm = Ym - V̂
    info[:Ŵ] = estim.Ŵ[1:Nk*nŵ]
    info[:x̂arr] = x̂arr
    info[:ϵ] = isinf(estim.C) ? NaN : estim.Z̃[begin]
    info[:J] = obj_nonlinprog(estim, estim.model, V̂, estim.Z̃)
    info[:X̂] = X̂
    info[:x̂] = estim.x̂
    info[:V̂] = V̂
    info[:P̄] = estim.P̂arr_old
    info[:x̄] = estim.x̂arr_old - x̂arr
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
        estim.X̂[:]  = [estim.X̂[nx̂+1:end]  ; x̂]
        estim.Ym[:] = [estim.Ym[nym+1:end]; ym]
        estim.U[:]  = [estim.U[nu+1:end]  ; u]
        estim.D[:]  = [estim.D[nd+1:end]  ; d]
        estim.Ŵ[:]  = [estim.Ŵ[nŵ+1:end]  ; ŵ]
        estim.Nk[:] = [estim.He]
    else
        estim.X̂[(1 + nx̂*(Nk-1)):(nx̂*Nk)]    = x̂
        estim.Ym[(1 + nym*(Nk-1)):(nym*Nk)] = ym
        estim.U[(1 + nu*(Nk-1)):(nu*Nk)]    = u
        estim.D[(1 + nd*(Nk-1)):(nd*Nk)]    = d
        estim.Ŵ[(1 + nŵ*(Nk-1)):(nŵ*Nk)]    = ŵ
    end
    estim.x̂arr_old[:] = estim.X̂[1:nx̂]
    return nothing
end

@doc raw"""
    initpred!(estim::MovingHorizonEstimator, model::LinModel)

Init linear model prediction matrices `F, fx̄, H̃, q̃, p` for [`MovingHorizonEstimator`](@ref).

Also init `estim.optim` objective function. See [`init_predmat_mhe`](@ref) for the 
definition of the matrices. The Hessian ``H̃`` matrix of the quadratic general form is not
constant here because of the time-varying ``\mathbf{P̄}`` weight (the estimation error 
covariance at arrival).
"""
function initpred!(estim::MovingHorizonEstimator, model::LinModel)
    C, optim = estim.C, estim.optim
    nϵ = isinf(C) ? 0 : 1
    nx̂, nŵ, nym, Nk = estim.nx̂, estim.nx̂, estim.nym, estim.Nk[]
    nYm, nŴ = nym*Nk, nŵ*Nk
    nZ̃ = nϵ + nx̂ + nŴ
    # --- update F and fx̄ vectors for MHE predictions ---
    estim.F[:] = estim.G*estim.U + estim.Ym
    if model.nd ≠ 0
        estim.F[:] = estim.F + estim.J*estim.D
    end
    estim.fx̄[:] = estim.x̂arr_old
    # --- update H̃, q̃ and p vectors for quadratic optimization ---
    Ẽ = @views [estim.ẽx̄[:, 1:nZ̃]; estim.Ẽ[1:nYm, 1:nZ̃]]
    F = @views [estim.fx̄; estim.F[1:nYm]]
    invQ̂_Nk, invR̂_Nk = @views estim.invQ̂_He[1:nŴ, 1:nŴ], estim.invR̂_He[1:nYm, 1:nYm]
    M = [estim.invP̄ zeros(nx̂, nYm); zeros(nYm, nx̂) invR̂_Nk]
    Ñ = [fill(C, nϵ, nϵ) zeros(nϵ, nx̂+nŴ); zeros(nx̂, nϵ+nx̂+nŴ); zeros(nŴ, nϵ+nx̂) invQ̂_Nk]
    estim.q̃[1:nZ̃] = 2(M*Ẽ)'*F
    estim.p[] = dot(F, M, F)
    estim.H̃.data[1:nZ̃, 1:nZ̃] = 2*(Ẽ'*M*Ẽ + Ñ)
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

Set `b` vector for the linear model inequality constraints (``\mathbf{A ΔŨ ≤ b}``) of MHE.

Also init ``\mathbf{F_x̂}`` vector for the state constraints, see [`init_predmat_mhe`](@ref).
"""
function linconstraint!(estim::MovingHorizonEstimator, model::LinModel)
    estim.con.Fx̂[:] = estim.con.Gx̂*estim.U
    if model.nd ≠ 0
        estim.con.Fx̂[:] = estim.con.Fx̂ + estim.con.Jx̂*estim.D
    end
    X̂min, X̂max = trunc_bounds(estim, estim.con.X̂min, estim.con.X̂max, estim.nx̂)
    Ŵmin, Ŵmax = trunc_bounds(estim, estim.con.Ŵmin, estim.con.Ŵmax, estim.nx̂)
    V̂min, V̂max = trunc_bounds(estim, estim.con.V̂min, estim.con.V̂max, estim.nym)
    estim.con.b[:] = [
        -estim.con.x̃min
        +estim.con.x̃max
        -X̂min + estim.con.Fx̂
        +X̂max - estim.con.Fx̂
        -Ŵmin
        +Ŵmax
        -V̂min + estim.F
        +V̂max - estim.F
    ]
    lincon = estim.optim[:linconstraint]
    set_normalized_rhs.(lincon, estim.con.b[estim.con.i_b])
end

"Set `b` excluding state and sensor noise bounds if `model` is not a [`LinModel`](@ref)."
function linconstraint!(estim::MovingHorizonEstimator, ::SimModel)
    Ŵmax, Ŵmax = trunc_bounds(estim, estim.con.Ŵmax, estim.con.Ŵmax, estim.nx̂)
    estim.con.b[:] = [
        -estim.con.x̃min
        +estim.con.x̃max
        -Ŵmax
        +Ŵmax
    ]
    lincon = estim.optim[:linconstraint]
    set_normalized_rhs.(lincon, estim.con.b[estim.con.i_b])
end

"Truncate the bounds `Bmin` and `Bmax` to the window size `Nk` if `Nk < He`."
function trunc_bounds(estim::MovingHorizonEstimator{NT}, Bmin, Bmax, n) where NT<:Real
    He, Nk = estim.He, estim.Nk[]
    if Nk < He
        nB = n*Nk
        Bmin_t = @views [Bmin[end-nB+1:end]; fill(-Inf, He*n-nB)]
        Bmax_t = @views [Bmax[end-nB+1:end]; fill(+Inf, He*n-nB)]
    else
        Bmin_t = Bmin
        Bmax_t = Bmax
    end
    return Bmin_t, Bmax_t
end

"Update the covariance `P̂` with the `KalmanFilter` if `model` is a `LinModel`."
function update_cov!(P̂, estim::MovingHorizonEstimator, ::LinModel, u, ym, d) 
    return update_estimate_kf!(estim, u, ym, d, estim.Â, estim.Ĉ[estim.i_ym, :], P̂)
end
"Update it with the `ExtendedKalmanFilter` if model is not a `LinModel`."
function update_cov!(P̂, estim::MovingHorizonEstimator, ::SimModel, u, ym, d) 
    # TODO: also support UnscentedKalmanFilter
    F̂ = ForwardDiff.jacobian(x̂ -> f̂(estim, estim.model, x̂, u, d), estim.x̂)
    Ĥ = ForwardDiff.jacobian(x̂ -> ĥ(estim, estim.model, x̂, d), estim.x̂)
    return update_estimate_kf!(estim, u, ym, d, F̂, Ĥ[estim.i_ym, :],  P̂)
end


"""
    obj_nonlinprog(estim::MovingHorizonEstimator, ::LinModel, _ , Z̃) 

Nonlinear programming objective function of MHE when `model` is a [`LinModel`](@ref).

It can be called on a [`MovingHorizonEstimator`](@ref) object to evaluate the objective 
function at specific `Z̃` and `V̂` values.
"""
function obj_nonlinprog(
    estim::MovingHorizonEstimator, ::LinModel, _ , Z̃::Vector{T}
) where {T<:Real}
    return obj_quadprog(Z̃, estim.H̃, estim.q̃) + estim.p[]
end

"""
    obj_nonlinprog(estim::MovingHorizonEstimator, model::SimModel, V̂, Z̃)

Objective function for the [`MovingHorizonEstimator`](@ref).

The function `dot(x, A, x)` is a performant way of calculating `x'*A*x`.
"""
function obj_nonlinprog(
    estim::MovingHorizonEstimator, ::SimModel, V̂, Z̃::Vector{T}
) where {T<:Real}
    Nk = estim.Nk[]
    nYm, nŴ, nx̂, invP̄ = Nk*estim.nym, Nk*estim.nx̂, estim.nx̂, estim.invP̄
    nx̃ = !isinf(estim.C) + nx̂
    invQ̂_Nk, invR̂_Nk = @views estim.invQ̂_He[1:nŴ, 1:nŴ], estim.invR̂_He[1:nYm, 1:nYm]
    x̂arr, Ŵ, V̂ = @views Z̃[nx̃-nx̂+1:nx̃], Z̃[nx̃+1:nx̃+nŴ], V̂[1:nYm]
    x̄ = estim.x̂arr_old - x̂arr
    Jϵ = isinf(estim.C) ? 0 : estim.C*Z̃[begin]^2
    return dot(x̄, invP̄, x̄) + dot(Ŵ, invQ̂_Nk, Ŵ) + dot(V̂, invR̂_Nk, V̂) + Jϵ
end

"""
    predict!(V̂, X̂, estim::MovingHorizonEstimator, model::LinModel, Z̃) -> V̂, X̂

Compute the `V̂` vector and `X̂` vectors for the `MovingHorizonEstimator` and `LinModel`.

The method mutates `V̂` and `X̂` vector arguments. The vector `V̂` is the estimated sensor
noises `V̂` from ``k-N_k+1`` to ``k``. The `X̂` vector is estimated states from ``k-N_k+2`` to
``k+1``.
"""
function predict!(
    V̂, X̂, estim::MovingHorizonEstimator, ::LinModel, Z̃::Vector{T}
) where {T<:Real}
    Nk = estim.Nk[]
    nX̂, nŴ, nYm = estim.nx̂*Nk, estim.nx̂*Nk, estim.nym*Nk
    nZ̃ = !isinf(estim.C) + estim.nx̂ + nŴ
    V̂[1:nYm] = @views estim.Ẽ[1:nYm, 1:nZ̃]*Z̃[1:nZ̃] + estim.F[1:nYm]
    X̂[1:nX̂]  = @views estim.con.Ẽx̂[1:nX̂, 1:nZ̃]*Z̃[1:nZ̃] + estim.con.Fx̂[1:nX̂]
    return V̂, X̂
end

"Compute the two vectors when `model` is not a `LinModel`."
function predict!(
    V̂, X̂, estim::MovingHorizonEstimator, model::SimModel, Z̃::Vector{T}
) where {T<:Real}
    Nk = estim.Nk[]
    nu, nd, nx̂, nŵ, nym = model.nu, model.nd, estim.nx̂, estim.nx̂, estim.nym
    nx̃ = !isinf(estim.C) + nx̂
    x̂ = @views Z̃[nx̃-nx̂+1:nx̃]
    for j=1:Nk
        u  = @views estim.U[ (1 + nu  * (j-1)):(nu*j)]
        ym = @views estim.Ym[(1 + nym * (j-1)):(nym*j)]
        d  = @views estim.D[ (1 + nd  * (j-1)):(nd*j)]
        ŵ  = @views Z̃[(1 + nx̃ + nŵ*(j-1)):(nx̃ + nŵ*j)]
        V̂[(1 + nym*(j-1)):(nym*j)] = ym - ĥ(estim, model, x̂, d)[estim.i_ym]
        X̂[(1 + nx̂ *(j-1)):(nx̂ *j)] = f̂(estim, model, x̂, u, d) + ŵ
        x̂ = @views X̂[(1 + nx̂*(j-1)):(nx̂*j)]
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