"Reset the data windows and time-varying variables for the moving horizon estimator."
function init_estimate_cov!(estim::MovingHorizonEstimator, y0m, d0, u0) 
    model = estim.model
    estim.Z̃         .= 0
    estim.Y0m       .= NaN
    estim.Yem       .= NaN
    estim.U0        .= NaN
    estim.Ue        .= NaN
    estim.D0        .= NaN
    estim.De        .= NaN
    estim.Ŵ         .= NaN
    estim.X̂0_old    .= NaN
    estim.Nk        .= 0
    estim.F         .= 0
    estim.H̃         .= 0
    estim.q̃         .= 0
    estim.r         .= 0
    estim.con.Fx̂    .= 0
    if estim.direct
        # add y0m(-1) to the extended data window (custom NL constraints):
        estim.Yem[1:ny] .= y0m .+ @views yop[estim.i_ym]
        # add u0(-1) to the two data windows:
        estim.U0[1:nu] .= u0
        estim.Ue[1:nu] .= u0 .+ uop
        # add d0(-1) to the extended data window (custom NL constraints):
        nd > 0 && (estim.De[1:nd] .= d0 .+ dop)
    end 
    nd > 0 && (estim.D0[1:nd] .= d0) # add d0(-1) to the data window
    estim.lastu0 .= u0
    # estim.cov.P̂_0 is P̂(-1|-1) if estim.direct==false, else P̂(-1|0)
    invert_cov!(estim, estim.cov.P̂_0)
    estim.P̂arr_old  .= estim.cov.P̂_0
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
- `:ε` or *`:epsilon`* : optimal slack variable, ``ε``
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

For [`NonLinModel`](@ref), it also includes the following fields:

- `:∇J` or *`:nablaJ`* : optimal gradient of the objective function, ``\mathbf{\nabla} J``
- `:∇²J` or *`:nabla2J`* : optimal Hessian of the objective function, ``\mathbf{\nabla^2}J``
- `:∇²J_ncolors` or *`:nabla2J_ncolors`* : number of colors in `:∇²J` sparsity pattern
- `:g` : optimal nonlinear inequality constraint values, ``\mathbf{g}``
- `:∇g` or *`:nablag`* : optimal Jacobian of the inequality constraint, ``\mathbf{\nabla g}``
- `:∇g_ncolors` or *`:nablag_ncolors`* : number of colors in `:∇g` sparsity pattern
- `:∇²ℓg` or *`:nabla2lg`* : optimal Hessian of the inequality Lagrangian, ``\mathbf{\nabla^2}\ell_{\mathbf{g}}``
- `:∇²ℓg_ncolors` or *`:nabla2lg_ncolors`* : number of colors in `:∇²ℓg` sparsity pattern

Note that the inequality constraint vectors and matrices only include the non-`Inf` values.

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
    nx̂, nym, nŵ, nε = estim.nx̂, estim.nym, estim.nx̂, estim.nε
    nx̃ = nε + nx̂
    info = Dict{Symbol, Any}()
    V̂,  X̂0 = buffer.V̂, buffer.X̂
    x̂0arr, û0, k, ŷ0 = buffer.x̂, buffer.û, buffer.k, buffer.ŷ
    x̂0arr  = getarrival!(x̂0arr, estim, estim.Z̃)
    x̄      = estim.x̂0arr_old - x̂0arr
    V̂,  X̂0 = predict_mhe!(V̂, X̂0, û0, k, ŷ0, estim, model, x̂0arr, estim.Ŵ, estim.Z̃)
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
    info[:ε]  = nε ≠ 0 ? estim.Z̃[begin] : zero(NT)
    info[:J]  = obj_nonlinprog(estim, estim.model, x̄, V̂, estim.Ŵ, estim.Z̃)
    info[:X̂]  = (X̂0       .+ @views [estim.x̂op; estim.X̂op])[1:nx̂*(Nk+1)]
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
    info[:epsilon] = info[:ε]
    info[:Xhat] = info[:X̂]
    info[:xhat] = info[:x̂]
    info[:Vhat] = info[:V̂]
    info[:Pbar] = info[:P̄]
    info[:xbar] = info[:x̄]
    info[:Yhat] = info[:Ŷ]
    info[:Yhatm] = info[:Ŷm]
    # --- deprecated fields ---
    info[:ϵ] = info[:ε]
    info = addinfo!(info, estim, model)
    return info
end


"""
    addinfo!(info, estim::MovingHorizonEstimator, model::NonLinModel)

For [`NonLinModel`](@ref), add the various derivatives.
"""
function addinfo!(
    info, estim::MovingHorizonEstimator{NT}, model::NonLinModel
) where NT <:Real
    # --- objective derivatives ---
    optim, con = estim.optim, estim.con
    hess = estim.hessian
    nx̂, nym, nŷ, nu, nk, nc = estim.nx̂, estim.nym, model.ny, model.nu, model.nk, con.nc
    He = estim.He
    i_g = findall(con.i_g) # convert to non-logical indices for non-allocating @views
    ng, ngi = length(con.i_g), sum(con.i_g)
    nV̂, nX̂, nŴ = He*nym, He*nx̂, He*nx̂
    nŴe, nX̂e, nV̂e = (He+1)*nx̂, (He+1)*nx̂, (He+1)*nym
    x̂0arr, x̄  = zeros(NT, nx̂), zeros(NT, nx̂)
    Ŵ         = zeros(NT, nŴ)
    V̂, X̂0     = zeros(NT, nV̂),  zeros(NT, nX̂)
    Ŵe        = zeros(NT, nŴe)
    V̂e, X̂e    = zeros(NT, nV̂e), zeros(NT, nX̂e)
    k         = zeros(NT, nk)
    û0, ŷ0    = zeros(NT, nu), zeros(NT, nŷ)
    gc, g     = zeros(NT, nc), zeros(NT, ng) 
    gi        = zeros(NT, ngi)
    J_cache = (
        Cache(x̂0arr), Cache(x̄), 
        Cache(Ŵ), Cache(V̂), Cache(X̂0), 
        Cache(Ŵe), Cache(V̂e), Cache(X̂e),
        Cache(û0), Cache(k), Cache(ŷ0), Cache(gc), Cache(g),
    )
    function J!(Z̃, x̂0arr, x̄, Ŵ, V̂, X̂0, Ŵe, V̂e, X̂e, û0, k, ŷ0, gc, g)
        update_prediction!(x̂0arr, x̄, Ŵ, V̂, X̂0, Ŵe, V̂e, X̂e, û0, k, ŷ0, gc, g, estim, Z̃)
        return obj_nonlinprog(estim, model, x̄, V̂, Ŵ, Z̃)
    end
    if !isnothing(hess)
        prep_∇²J = prepare_hessian(J!, hess, estim.Z̃, J_cache...)
        _, ∇J_opt, ∇²J_opt = value_gradient_and_hessian(J!, prep_∇²J, hess, estim.Z̃, J_cache...)
        ∇²J_ncolors = get_ncolors(prep_∇²J)
    else
        prep_∇J = prepare_gradient(J!, estim.gradient, estim.Z̃, J_cache...)
        ∇J_opt = gradient(J!, prep_∇J, estim.gradient, estim.Z̃, J_cache...)
        ∇²J_opt, ∇²J_ncolors = nothing, nothing
    end
    # --- inequality constraint derivatives ---
    ∇g_cache = (
        Cache(x̂0arr), Cache(x̄), 
        Cache(Ŵ), Cache(V̂), Cache(X̂0), 
        Cache(Ŵe), Cache(V̂e), Cache(X̂e),
        Cache(û0), Cache(k), Cache(ŷ0), Cache(gc), Cache(g),
    )
    function gi!(gi, Z̃, x̂0arr, x̄, Ŵ, V̂, X̂0, Ŵe, V̂e, X̂e, û0, k, ŷ0, gc, g)
        update_prediction!(x̂0arr, x̄, Ŵ, V̂, X̂0, Ŵe, V̂e, X̂e, û0, k, ŷ0, gc, g, estim, Z̃)
        gi .= @views g[i_g]
        return nothing
    end
    prep_∇g = prepare_jacobian(gi!, gi, estim.jacobian, estim.Z̃, ∇g_cache...)
    g_opt, ∇g_opt = value_and_jacobian(gi!, gi, prep_∇g, estim.jacobian, estim.Z̃, ∇g_cache...)
    ∇g_ncolors = get_ncolors(prep_∇g)
    if !isnothing(hess) && ngi > 0
        nonlincon = optim[:nonlinconstraint]
        λi = try
            JuMP.get_attribute(nonlincon, MOI.LagrangeMultiplier())
        catch err
            if err isa MOI.GetAttributeNotAllowed{MOI.LagrangeMultiplier}
                @warn(
                    "The optimizer does not support retrieving optimal Hessian of the Lagrangian.\n"*
                    "Its nonzero coefficients will be random values.", maxlog=1
                )
                rand(ngi)
            else
                rethrow()
            end
        end
        ∇²g_cache = (
            Cache(x̂0arr), Cache(x̄), 
            Cache(Ŵ), Cache(V̂), Cache(X̂0), 
            Cache(Ŵe), Cache(V̂e), Cache(X̂e),
            Cache(û0), Cache(k), Cache(ŷ0), Cache(gc), Cache(g), Cache(gi)
        )
        function ℓ_gi(Z̃, λi, x̂0arr, x̄, Ŵ, V̂, X̂0, Ŵe, V̂e, X̂e, û0, k, ŷ0, gc, g, gi)
            update_prediction!(x̂0arr, x̄, Ŵ, V̂, X̂0, Ŵe, V̂e, X̂e, û0, k, ŷ0, gc, g, estim, Z̃)
            gi .= @views g[i_g]
            return dot(λi, gi)
        end
        prep_∇²ℓg = prepare_hessian(ℓ_gi, hess, estim.Z̃, Constant(λi), ∇²g_cache...)
        ∇²ℓg_opt = hessian(ℓ_gi, prep_∇²ℓg, hess, estim.Z̃, Constant(λi), ∇²g_cache...)
        ∇²ℓg_ncolors = get_ncolors(prep_∇²ℓg)
    else
        ∇²ℓg_opt, ∇²ℓg_ncolors = nothing, nothing
    end
    info[:∇J] = ∇J_opt
    info[:∇²J] = ∇²J_opt
    info[:∇²J_ncolors] = ∇²J_ncolors
    info[:g] = g_opt
    info[:∇g] = ∇g_opt
    info[:∇g_ncolors] = ∇g_ncolors
    info[:∇²ℓg] = ∇²ℓg_opt
    info[:∇²ℓg_ncolors] = ∇²ℓg_ncolors
    # --- non-Unicode fields ---
    info[:nablaJ] = ∇J_opt
    info[:nabla2J] = ∇²J_opt
    info[:nabla2J_ncolors] = ∇²J_ncolors
    info[:nablag] = ∇g_opt
    info[:nablag_ncolors] = ∇g_ncolors
    info[:nabla2lg] = ∇²ℓg_opt
    info[:nabla2lg_ncolors] = ∇²ℓg_ncolors
    return info
end

"Nothing to add in the `info` dict for [`LinModel`](@ref)."
addinfo!(info, ::MovingHorizonEstimator, ::LinModel) = info

"Get the estimated state at arrival from the decision vector `Z̃`."
function getarrival!(x̂0arr, estim::MovingHorizonEstimator, Z̃) 
    nx̃ = estim.nε + estim.nx̂
    return x̂0arr .= @views Z̃[nx̃-estim.nx̂+1:nx̃]
end

"Get the estimated process noise over the horizon from the decision vector `Z̃`."
function getŴ!(Ŵ, estim::MovingHorizonEstimator, Z̃)
    nx̃ = estim.nε + estim.nx̂
    return Ŵ .= @views Z̃[(nx̃ + 1):(nx̃ + estim.nx̂*estim.He)]
end

"""
    getε(estim::MovingHorizonEstimator, Z̃) -> ε

Get the slack `ε` from the decision vector `Z̃` if present, otherwise return 0.
"""
function getε(estim::MovingHorizonEstimator, Z̃::AbstractVector{NT}) where NT<:Real
    return estim.nε > 0 ? Z̃[begin] : zero(NT)
end

"""
    add_data_windows!(estim::MovingHorizonEstimator, y0m, d0, u0=estim.lastu0) -> ismoving

Add data to the observation windows of the moving horizon estimator and clamp `estim.Nk`.

If ``k ≥ H_e``, the observation windows are moving in time and `estim.Nk` is clamped to
`estim.He`. It returns `true` if the observation windows are moving, `false` otherwise.
If no `u0` argument is provided, the manipulated input of the last time step is added to its
window (the correct value if `estim.direct`).
"""
function add_data_windows!(estim::MovingHorizonEstimator, y0m, d0, u0=estim.lastu0)
    model = estim.model
    nx̂, nym, nd, nu, nŵ = estim.nx̂, estim.nym, model.nd, model.nu, estim.nx̂
    yopm = @views model.yop[estim.i_ym]
    Nk = estim.Nk[]
    p = estim.direct ? 0 : 1 # u0 argument is u0(k-1) if estim.direct, else u0(k)
    x̂0_old = estim.x̂0        # x̂0_old is x̂0(k-1|k-1) if estim.direct, else x̂0(k|k-1)
    ŵ = 0                    # ŵ(k-1+p) = 0 for warm-start
    estim.Nk .+= 1
    Nk = estim.Nk[]
    ismoving = (Nk > estim.He)
    # --- data windows for the predictions ---
    # see MovingHorzionEstimator extended help for the exact time steps in each data window
    if ismoving
        estim.Y0m[1:end-nym]        .= @views estim.Y0m[nym+1:end]
        estim.Yem[1:end-nym]        .= @views estim.Yem[nym+1:end]
        estim.Y0m[end-nym+1:end]                        .= y0m
        estim.Yem[(end-nym+1 - p*nym):(end - p*nym)]    .= y0m .+ yopm
        if nd > 0
            estim.D0[1:end-nd]       .= @views estim.D0[nd+1:end]
            estim.De[1:end-nd]       .= @views estim.De[nd+1:end]
            estim.D0[end-nd+1:end]                      .= d0
            estim.De[(end-nd+1 - p*nd):(end - p*nd)]    .= d0 .+ model.dop
        end
        estim.U0[1:end-nu]          .= @views estim.U0[nu+1:end]
        estim.Ue[1:end-nu]          .= @views estim.Ue[nu+1:end]
        estim.U0[end-nu+1:end]                          .= u0
        estim.Ue[(end-nu+1 - nu):(end - nu)]            .= u0 .+ model.uop
        estim.Ŵ[1:end-nŵ]           .= @views estim.Ŵ[nŵ+1:end]
        estim.Ŵ[end-nŵ+1:end]       .= ŵ
        estim.X̂0_old[1:end-nx̂]      .= @views estim.X̂0_old[nx̂+1:end]
        estim.X̂0_old[end-nx̂+1:end]  .= x̂0_old
        estim.Nk .= estim.He
    else
        estim.Y0m[(1 + nym*(Nk-1)):(nym*Nk)]            .= y0m
        estim.Yem[(1 + nym*(Nk-p)):(nym*(Nk-p+1))]      .= y0m .+ yopm
        if nd > 0 
            estim.D0[(1 + nd*Nk):(nd*(Nk+1))]           .= d0
            estim.De[(1 + nd*(Nk-p)):(nd*(Nk-p+1))]     .= d0 .+ model.dop
        end
        estim.U0[(1 + nu*(Nk-1)):(nu*Nk)]               .= u0
        estim.Ue[(1 + nu*(Nk-1)):(nu*Nk)]               .= u0 .+ model.uop
        estim.Ŵ[(1 + nŵ*(Nk-1)):(nŵ*Nk)]                .= ŵ
        estim.X̂0_old[(1 + nx̂*(Nk-1)):(nx̂*Nk)]           .= x̂0_old
    end
    estim.x̂0arr_old .= @views estim.X̂0_old[1:nx̂]
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
in which ``\mathbf{Z̃} = [\begin{smallmatrix} ε \\ \mathbf{Z} \end{smallmatrix}]``. Note that
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
    invP̄, invQ̂_He, invR̂_He = estim.cov.invP̄, estim.cov.invQ̂_He, estim.cov.invR̂_He
    F, C, optim = estim.F, estim.C, estim.optim
    nx̂, nŵ, nym, nε, Nk = estim.nx̂, estim.nx̂, estim.nym, estim.nε, estim.Nk[]
    nU, nYm, nŴ, nD = model.nu*Nk, estim.nym*Nk, nŵ*Nk, model.nd*(Nk+1)
    nZ̃ = nε + nx̂ + nŴ
    # --- truncate vector and matrices if necessary ---
    if Nk < estim.He
        # avoid views since allocations only when Nk < He and we want fast mul!:
        Y0m, B = estim.Y0m[1:nYm],     estim.B[1:nYm]
        G, U0  = estim.G[1:nYm, 1:nU], estim.U0[1:nU]
        J, D0  = estim.J[1:nYm, 1:nD], estim.D0[1:nD]
        Ẽ, ẽx̄  = estim.Ẽ[1:nYm, 1:nZ̃], estim.ẽx̄[:, 1:nZ̃]
        F, q̃   = @views estim.F[1:nYm], estim.q̃[1:nZ̃]
        H̃_data = @views estim.H̃.data[1:nZ̃, 1:nZ̃]
        H̃      = @views estim.H̃[1:nZ̃, 1:nZ̃]
        Z̃var   = @views optim[:Z̃var][1:nZ̃]
    else
        Y0m, B = estim.Y0m, estim.B
        G, U0  = estim.G, estim.U0
        J, D0  = estim.J, estim.D0
        Ẽ, ẽx̄  = estim.Ẽ, estim.ẽx̄
        F, q̃   = estim.F, estim.q̃
        H̃_data = estim.H̃.data
        H̃      = estim.H̃
        Z̃var   = optim[:Z̃var]
    end
    invQ̂_Nk = trunc_cov(invQ̂_He, nx̂, Nk, estim.He)
    invR̂_Nk = trunc_cov(invR̂_He, nym, Nk, estim.He)
    fx̄ = estim.fx̄
    r = estim.r
    # --- update F and fx̄ vectors for MHE predictions ---
    F .= Y0m .+ B
    mul!(F, G, U0, 1, 1)
    (model.nd > 0) && mul!(F, J, D0, 1, 1)
    fx̄ .= estim.x̂0arr_old
    # --- update H̃, q̃ and p vectors for quadratic optimization ---
    ẼZ̃ = [ẽx̄; Ẽ]
    FZ̃ = [fx̄; F]
    M_Nk = [invP̄ zeros(nx̂, nYm); zeros(nYm, nx̂) invR̂_Nk]
    Ñ_Nk = [fill(C, nε, nε) zeros(nε, nx̂+nŴ); zeros(nx̂, nε+nx̂+nŴ); zeros(nŴ, nε+nx̂) invQ̂_Nk]
    M_Nk_ẼZ̃ = M_Nk*ẼZ̃
    mul!(q̃, M_Nk_ẼZ̃', FZ̃)
    lmul!(2, q̃)
    r .= dot(FZ̃, M_Nk, FZ̃)
    H̃_data .= Ñ_Nk
    mul!(H̃_data, ẼZ̃', M_Nk_ẼZ̃, 1, 1) 
    lmul!(2, H̃_data)
    println(q̃)
    JuMP.set_objective_function(optim, obj_quadprog(Z̃var, H̃, q̃))
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
    nx̂, nŵ, nym, Nk = estim.nx̂, estim.nx̂, estim.nym, estim.Nk[]
    nU, nX̂, nD = model.nu*Nk, estim.nx̂*Nk, model.nd*Nk
    # --- truncate vector and matrices if necessary ---
    if Nk < estim.He
        # avoid views since allocations only when Nk < He and we want fast mul!:
        Bx̂     = estim.con.Bx̂[1:nX̂]
        Gx̂, U0 = estim.con.Gx̂[1:nX̂, 1:nU], estim.U0[1:nU]
        Jx̂, D0 = estim.con.Jx̂[1:nX̂, 1:nD], estim.D0[1:nD]
        Fx̂     = @views estim.con.Fx̂[1:nX̂]
    else
        Bx̂     = estim.con.Bx̂
        Gx̂, U0 = estim.con.Gx̂, estim.U0
        Jx̂, D0 = estim.con.Jx̂, estim.D0
        Fx̂     = estim.con.Fx̂
    end
    X̂0min, X̂0max = trunc_bounds(estim, estim.con.X̂0min, estim.con.X̂0max, nx̂)
    Ŵmin, Ŵmax   = trunc_bounds(estim, estim.con.Ŵmin,  estim.con.Ŵmax,  nŵ)
    V̂min, V̂max   = trunc_bounds(estim, estim.con.V̂min,  estim.con.V̂max,  nym)
    # --- update Fx̂ vectors for MHE state constraints ---
    Fx̂ .= Bx̂
    mul!(Fx̂, Gx̂, U0, 1, 1)
    model.nd > 0 && mul!(Fx̂, Jx̂, D0, 1, 1)
    # --- update b vector for linear inequality constraints ---
    nX̂_He, nŴ_He, nV̂_He = length(X̂0min), length(Ŵmin), length(V̂min)
    nx̃ = length(estim.con.x̃0min)
    n = 0
    estim.con.b[(n+1):(n+nx̃)] .= @. -estim.con.x̃0min
    n += nx̃
    estim.con.b[(n+1):(n+nx̃)] .= @. +estim.con.x̃0max
    n += nx̃
    estim.con.b[(n+1):(n+nX̂_He)] .= @. -X̂0min + estim.con.Fx̂
    n += nX̂_He
    estim.con.b[(n+1):(n+nX̂_He)] .= @. +X̂0max - estim.con.Fx̂
    n += nX̂_He
    estim.con.b[(n+1):(n+nŴ_He)] .= @. -Ŵmin
    n += nŴ_He
    estim.con.b[(n+1):(n+nŴ_He)] .= @. +Ŵmax
    n += nŴ_He
    estim.con.b[(n+1):(n+nV̂_He)] .= @. -V̂min + estim.F
    n += nV̂_He
    estim.con.b[(n+1):(n+nV̂_He)] .= @. +V̂max - estim.F
    if any(estim.con.i_b) 
        lincon = estim.optim[:linconstraint]
        JuMP.set_normalized_rhs(lincon, estim.con.b[estim.con.i_b])
    end
    return nothing
end

"Set `b` excluding state and sensor noise bounds if `model` is not a [`LinModel`](@ref)."
function linconstraint!(estim::MovingHorizonEstimator, ::SimModel)
    # --- truncate vector and matrices if necessary ---
    Ŵmin, Ŵmax = trunc_bounds(estim, estim.con.Ŵmin, estim.con.Ŵmax, estim.nx̂)
    # --- update b vector for linear inequality constraints ---
    nx̃, nŴ_He = length(estim.con.x̃0min), length(Ŵmin)
    n = 0
    estim.con.b[(n+1):(n+nx̃)] .= @. -estim.con.x̃0min
    n += nx̃
    estim.con.b[(n+1):(n+nx̃)] .= @. +estim.con.x̃0max
    n += nx̃
    estim.con.b[(n+1):(n+nŴ_He)] .= @. -Ŵmin
    n += nŴ_He
    estim.con.b[(n+1):(n+nŴ_He)] .= @. +Ŵmax
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
    nŵ, nx̂, Nk =  estim.nx̂, estim.nx̂, estim.Nk[]
    nx̃ = estim.nε + nx̂
    Z̃var::Vector{JuMP.VariableRef} = optim[:Z̃var]
    Z̃s = set_warmstart_mhe!(estim, Z̃var)
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
    x̂0arr, û0, ŷ0, k = buffer.x̂, buffer.û, buffer.ŷ, buffer.k
    V̂, X̂0  = buffer.V̂, buffer.X̂
    estim.Ŵ[1:nŵ*Nk] .= @views estim.Z̃[nx̃+1:nx̃+nŵ*Nk] # update Ŵ with optimum for warm-start
    getarrival!(x̂0arr, estim, estim.Z̃)
    predict_mhe!(V̂, X̂0, û0, k, ŷ0, estim, model, x̂0arr, estim.Ŵ, estim.Z̃)
    x̂0corrORnext = @views X̂0[Nk*nx̂-nx̂+1:Nk*nx̂]
    estim.x̂0 .= x̂0corrORnext
    return estim.Z̃
end

@doc raw"""
    set_warmstart_mhe!(estim::MovingHorizonEstimator, Z̃var) -> Z̃s

Set and return the warm-start value of `Z̃var` for [`MovingHorizonEstimator`](@ref).

If supported by `estim.optim`, it warm-starts the solver at:
```math
\mathbf{Z̃_s} = 
\begin{bmatrix}
    ε_{k-1}                         \\
    \mathbf{x̂}_{k-1}(k-N_k+p)       \\ 
    \mathbf{ŵ}_{k-1}(k-N_k+p+0)     \\ 
    \mathbf{ŵ}_{k-1}(k-N_k+p+1)     \\ 
    \vdots                          \\
    \mathbf{ŵ}_{k-1}(k-p-2)         \\
    \mathbf{0}                      \\
\end{bmatrix}
```
where ``ε(k-1)``, ``\mathbf{x̂}_{k-1}(k-N_k+p)`` and ``\mathbf{ŵ}_{k-1}(k-j)`` are
respectively the slack variable, the arrival state estimate and the process noise estimates
computed at the last time step ``k-1``. If the objective function is not finite at this
point, all the process noises ``\mathbf{ŵ}_{k-1}(k-j)`` are warm-started at zeros. The
method mutates all the arguments.
"""
function set_warmstart_mhe!(estim::MovingHorizonEstimator{NT}, Z̃var) where NT<:Real
    model, buffer = estim.model, estim.buffer
    nε, nx̂, nŵ, Nk = estim.nε, estim.nx̂, estim.nx̂, estim.Nk[]
    nx̃ = nε + nx̂
    Z̃s = estim.buffer.Z̃
    û0, ŷ0, x̄, k = buffer.û, buffer.ŷ, buffer.x̂, buffer.k
    # --- slack variable ε ---
    estim.nε == 1 && (Z̃s[begin] = estim.Z̃[begin])
    # --- arrival state estimate x̂0arr ---
    Z̃s[nε+1:nx̃] = estim.x̂0arr_old
    # --- process noise estimates Ŵ ---
    Z̃s[nx̃+1:end] = estim.Ŵ
    # verify definiteness of objective function:
    V̂, X̂0 = estim.buffer.V̂, estim.buffer.X̂
    x̄ .= 0 # x̂0arr == x̂arr_old implies the error at arrival x̄ is zero
    predict_mhe!(V̂, X̂0, û0, k, ŷ0, estim, model, estim.x̂0arr_old, estim.Ŵ, Z̃s)
    Js = obj_nonlinprog(estim, model, x̄, V̂, estim.Ŵ, Z̃s)
    if !isfinite(Js)
        Z̃s[nx̃+1:end] .= 0
    end
    # --- unused variable in Z̃ (applied only when Nk ≠ He) ---
    # We force the update of the NLP gradient and jacobian by warm-starting the unused 
    # variable in Z̃ at 1. Since estim.Ŵ is initialized with 0s, at least 1 variable in Z̃s
    # will be inevitably different at the following time step.
    Z̃s[nx̃+Nk*nŵ+1:end] .= 1
    JuMP.set_start_value.(Z̃var, Z̃s)
    return Z̃s
end

"Truncate the inverse covariance `invA_He` to the window size `Nk` if `Nk < He`."
function trunc_cov(invA_He::Hermitian{<:Real, <:AbstractMatrix}, n, Nk, He) 
    if Nk < He
        nA = Nk*n
        # avoid views since allocations only when Nk < He and we want type-stability:
        return Hermitian(invA_He[1:nA, 1:nA], :L)
    else
        return invA_He
    end
end
function trunc_cov(
    invA_He::Hermitian{NT, Diagonal{NT, Vector{NT}}}, n, Nk, He
) where NT <:Real
    if Nk < He
        nA = Nk*n
        # avoid views since allocations only when Nk < He and we want type-stability:
        return Hermitian(Diagonal(invA_He.data.diag[1:nA]), :L)
    else
        return invA_He
    end
end

"Correct the covariance estimate at arrival using `covestim` [`StateEstimator`](@ref)."
function correct_cov!(estim::MovingHorizonEstimator)
    nym, nd = estim.nym, estim.model.nd
    buffer = estim.covestim.buffer
    y0marr, d0arr = buffer.ym, buffer.d
    y0marr .= @views estim.Y0m[1:nym]
    d0arr  .= @views estim.D0[1:nd]
    estim.covestim.x̂0     .= estim.x̂0arr_old
    estim.covestim.cov.P̂  .= estim.P̂arr_old
    try
        correct_estimate!(estim.covestim, y0marr, d0arr)
        all(isfinite, estim.covestim.cov.P̂) || error("Arrival covariance P̄ is not finite")
        estim.P̂arr_old .= estim.covestim.cov.P̂
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
    estim.covestim.x̂0     .= estim.x̂0arr_old
    estim.covestim.cov.P̂  .= estim.P̂arr_old
    try
        update_estimate!(estim.covestim, y0marr, d0arr, u0arr)
        all(isfinite, estim.covestim.cov.P̂) || error("Arrival covariance P̄ is not finite")
        estim.P̂arr_old .= estim.covestim.cov.P̂
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
    estim.cov.invP̄ .= P̄
    try
        inv!(estim.cov.invP̄)
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
    obj_nonlinprog(estim::MovingHorizonEstimator, ::LinModel, _ , _ , _ , Z̃) 

Objective function of [`MovingHorizonEstimator`](@ref) when `model` is a [`LinModel`](@ref).

It can be called on a [`MovingHorizonEstimator`](@ref) object to evaluate the objective 
function at specific `Z̃`.
"""
function obj_nonlinprog(estim::MovingHorizonEstimator, ::LinModel, _ , _ , _ , Z̃)
    return obj_quadprog(Z̃, estim.H̃, estim.q̃) + estim.r[]
end

"""
    obj_nonlinprog(estim::MovingHorizonEstimator, model::SimModel, x̄, V̂, Ŵ, Z̃)

Objective function of the MHE when `model` is not a [`LinModel`](@ref).

The function `dot(x, A, x)` is a performant way of calculating `x'*A*x`.
"""
function obj_nonlinprog(estim::MovingHorizonEstimator, ::SimModel, x̄, V̂, Ŵ, Z̃) 
    Nk = estim.Nk[] 
    invP̄ = estim.cov.invP̄
    invQ̂_Nk = trunc_cov(estim.cov.invQ̂_He, estim.nx̂, Nk, estim.He)
    invR̂_Nk = trunc_cov(estim.cov.invR̂_He, estim.nym, Nk, estim.He)
    if Nk < estim.He
        nŴ, nYm = Nk*estim.nx̂, Nk*estim.nym
        Ŵ, V̂ = Ŵ[1:nŴ], V̂[1:nYm]
    end
    Jε = estim.nε > 0 ? estim.C*Z̃[begin]^2 : 0
    return dot(x̄, invP̄, x̄) + dot(Ŵ, invQ̂_Nk, Ŵ) + dot(V̂, invR̂_Nk, V̂) + Jε
end

@doc raw"""
    predict_mhe!(
        V̂, X̂0, _, _, _, estim::MovingHorizonEstimator, model::LinModel, _ , _ , Z̃
    ) -> V̂, X̂0

Compute the `V̂` vector and `X̂0` vectors for the `MovingHorizonEstimator` and `LinModel`.

The function mutates `V̂` and `X̂0` vector arguments. The vector `V̂` is the estimated sensor
noises from ``k-N_k+1`` to ``k``. The `X̂0` vector is estimated states from ``k-N_k+2`` to 
``k+1``. The computations are (by truncating the matrices when `N_k < H_e`):
```math
\begin{aligned}
\mathbf{V̂}   &= \mathbf{Ẽ Z̃}   + \mathbf{F}     \\
\mathbf{X̂_0} &= \mathbf{Ẽ_x̂ Z̃} + \mathbf{F_x̂}
\end{aligned}
```
"""
function predict_mhe!(
    V̂, X̂0, _ , _ , _ , estim::MovingHorizonEstimator, ::LinModel, _ , _ , Z̃
)
    nε, Nk = estim.nε, estim.Nk[]
    if Nk < estim.He
        # avoid views since allocations only when Nk < He and we want fast mul!:
        nX̂, nŴ, nYm = estim.nx̂*Nk, estim.nx̂*Nk, estim.nym*Nk
        nZ̃ = nε + estim.nx̂ + nŴ
        Ẽ,  F  = estim.Ẽ[1:nYm, 1:nZ̃],     estim.F[1:nYm]
        Ẽx̂, Fx̂ = estim.con.Ẽx̂[1:nX̂, 1:nZ̃], estim.con.Fx̂[1:nX̂]
        Z̃ = Z̃[1:nZ̃]
        V̂_res, X̂0_res = @views V̂[1:nYm], X̂0[1:nX̂]
    else
        Ẽ, F = estim.Ẽ, estim.F
        Ẽx̂, Fx̂ = estim.con.Ẽx̂, estim.con.Fx̂
        V̂_res, X̂0_res = V̂, X̂0
    end
    V̂_res  .= mul!(V̂_res, Ẽ, Z̃) .+ F
    X̂0_res .= mul!(X̂0_res, Ẽx̂, Z̃) .+ Fx̂
    return V̂, X̂0
end

@doc raw"""
    predict_mhe!(
        V̂, X̂0, û0, k, ŷ0, estim::MovingHorizonEstimator, model::SimModel, x̂0arr, Ŵ, _ 
    ) -> V̂, X̂0

Compute the vectors when `model` is *not* a [`LinModel`](@ref).

The function mutates `V̂`, `X̂0`, `û0` and `ŷ0` vector arguments. The augmented model of
[`f̂!`](@ref) and [`ĥ!`](@ref) is called recursively in a `for` loop from ``j=1`` to ``N_k``,
and by adding the estimated process noise ``\mathbf{ŵ}``.
"""
function predict_mhe!(
    V̂, X̂0, û0, k, ŷ0, estim::MovingHorizonEstimator, model::SimModel, x̂0arr, Ŵ, _ 
)
    nu, nd, nx̂, nŵ, nym, Nk = model.nu, model.nd, estim.nx̂, estim.nx̂, estim.nym, estim.Nk[]
    x̂0 = x̂0arr
    if estim.direct     # p = 0
        ŷ0next = ŷ0
        d0 = @views estim.D0[1:nd]
        for j=1:Nk
            u0  = @views estim.U0[ (1 + nu  * (j-1)):(nu*j)]
            ŵ   = @views Ŵ[(1 + nŵ*(j-1)):(nŵ*j)]
            x̂0next = @views X̂0[(1 + nx̂ *(j-1)):(nx̂ *j)]
            f̂!(x̂0next, û0, k, estim, model, x̂0, u0, d0)
            x̂0next .+= ŵ
            y0nextm = @views estim.Y0m[(1 + nym * (j-1)):(nym*j)]
            d0next  = @views estim.D0[(1 + nd*j):(nd*(j+1))]
            ĥ!(ŷ0next, estim, model, x̂0next, d0next)
            ŷ0nextm = @views ŷ0next[estim.i_ym]
            V̂[(1 + nym*(j-1)):(nym*j)] .= y0nextm .- ŷ0nextm
            x̂0, d0 = x̂0next, d0next
        end        
    else                # p = 1
        for j=1:Nk
            y0m = @views estim.Y0m[(1 + nym * (j-1)):(nym*j)]
            u0  = @views estim.U0[ (1 + nu  * (j-1)):(nu*j)]
            d0  = @views estim.D0[ (1 + nd*j):(nd*(j+1))] # 1st one is d(k-Nk), not used
            ŵ   = @views Ŵ[(1 + nŵ*(j-1)):(nŵ*j)]
            ĥ!(ŷ0, estim, model, x̂0, d0)
            ŷ0m = @views ŷ0[estim.i_ym]
            V̂[(1 + nym*(j-1)):(nym*j)] .= y0m .- ŷ0m
            x̂0next = @views X̂0[(1 + nx̂ *(j-1)):(nx̂ *j)]
            f̂!(x̂0next, û0, k, estim, model, x̂0, u0, d0)
            x̂0next .+= ŵ
            x̂0 = x̂0next
        end
    end
    return V̂, X̂0
end


"""
    update_predictions!(
        x̂0arr, x̄, Ŵ, V̂, X̂0, Ŵe, V̂e, X̂e, û0, k, ŷ0, gc, g, 
        estim::MovingHorizonEstimator, Z̃
    ) -> nothing

Update in-place the vectors for the predictions of `estim` estimator at decision vector `Z̃`.

The method mutates all the arguments before `estim` argument.
"""
function update_prediction!(
    x̂0arr, x̄, Ŵ, V̂, X̂0, Ŵe, V̂e, X̂e, û0, k, ŷ0, gc, g, estim::MovingHorizonEstimator, Z̃
)
    x̂0arr      = getarrival!(x̂0arr, estim, Z̃)
    x̄         .= estim.x̂0arr_old .- x̂0arr
    Ŵ          = getŴ!(Ŵ, estim, Z̃)
    V̂, X̂0      = predict_mhe!(V̂, X̂0, û0, k, ŷ0, estim, estim.model, x̂0arr, Ŵ, Z̃)
    Ŵe, V̂e, X̂e = extended_vectors!(Ŵe, V̂e, X̂e, estim, Ŵ, V̂, X̂0, x̂0arr)
    ε          = getε(estim, Z̃)
    gc         = con_custom_mhe!(gc, estim, X̂e, V̂e, Ŵe, x̄, ε) 
    g          = con_nonlinprog_mhe!(g, estim, estim.model, X̂0, V̂, gc, ε)
    return nothing
end

"""
    extended_vectors!(
        Ŵe, V̂e, X̂e, estim::MovingHorizonEstimator, Ŵ, V̂, X̂0, x̂0arr
    ) -> Ŵe, V̂e, X̂e

Compute the extended `Ŵe, V̂e` and `X̂e` vectors for NLP using the `Ŵ, V̂` and `X̂0` vectors.

See [`MovingHorizonEstimator`](@ref) for the definition of the vectors, the exact time
steps of the samples in them and the missing values with `NaN`s. The method mutates all
the arguments before `estim` argument.
"""
function extended_vectors!(Ŵe, V̂e, X̂e, estim::MovingHorizonEstimator, Ŵ, V̂, X̂0, x̂0arr)
    nym, nŵ, nx̂ = estim.nym, estim.nx̂, estim.nx̂
    Ŵe[1:end-nŵ]            .= Ŵ
    Ŵe[end-nŵ+1:end]        .= NaN
    X̂e[1:nx̂]                .= x̂0arr .+ estim.x̂op
    X̂e[nx̂+1:end]            .= X̂0 .+ estim.X̂op
    if estim.direct
        V̂e[1:nym]           .= NaN
        V̂e[1+nym:end]       .= V̂
    else
        V̂e[1:end-nym]       .= V̂
        V̂e[end-nym+1:end]   .= NaN
    end
    return Ŵe, V̂e, X̂e
end


"""
    con_custom_mhe!(gc, estim::MovingHorizonEstimator, X̂e, V̂e, Ŵe, x̄, ε) -> gc

Evaluate the custom inequality constraint `gc` in-place for [`MovingHorizonEstimator`](@ref).
"""
function con_custom_mhe!(gc, estim::MovingHorizonEstimator, X̂e, V̂e, Ŵe, x̄, ε) 
    if estim.con.nc > 0
        P̄ = estim.P̂arr_old
        Nk = estim.Nk[]
        Ue, Yem, De = estim.Ue, estim.Yem, estim.De
        if Nk < estim.He
            # avoid views since allocations only when Nk < He and we want fast mul!:
            nX̂e, nŴe, nYem = (Nk+1)*estim.nx̂, (Nk+1)*estim.nx̂, (Nk+1)*estim.nym
            nUe, nDe       = (Nk+1)*estim.model.nu, (Nk+1)*estim.model.nd
            Ue, Yem, De = estim.Ue[1:nUe], estim.Yem[1:nYem], estim.De[1:nDe]
            X̂e, V̂e, Ŵe  = X̂e[1:nX̂e], V̂e[1:nYem], Ŵe[1:nŴe]
        else
            Ue, Yem, De = estim.Ue, estim.Yem, estim.De
        end
        estim.con.gc!(gc, X̂e, V̂e, Ŵe, Ue, Yem, De, P̄, x̄, estim.p, ε)
    end 
    return gc
end

"""
    con_nonlinprog_mhe!(
        g, estim::MovingHorizonEstimator, model::SimModel, X̂0, V̂, gc, ε
    ) -> g

Compute nonlinear constrains `g` in-place for [`MovingHorizonEstimator`](@ref).
"""
function con_nonlinprog_mhe!(g, estim::MovingHorizonEstimator, ::SimModel, X̂0, V̂, gc, ε)
    nX̂con, nX̂ = length(estim.con.X̂0min), estim.nx̂ *estim.Nk[]
    nV̂con, nV̂ = length(estim.con.V̂min),  estim.nym*estim.Nk[]
    for i in eachindex(g)
        estim.con.i_g[i] || continue
        if i ≤ nX̂con
            j = i
            jcon = nX̂con-nX̂+j
            g[i] = j > nX̂ ? 0 : estim.con.X̂0min[jcon] - X̂0[j] - ε*estim.con.C_x̂min[jcon]
        elseif i ≤ 2nX̂con
            j = i - nX̂con
            jcon = nX̂con-nX̂+j
            g[i] = j > nX̂ ? 0 : X̂0[j] - estim.con.X̂0max[jcon] - ε*estim.con.C_x̂max[jcon]
        elseif i ≤ 2nX̂con + nV̂con
            j = i - 2nX̂con
            jcon = nV̂con-nV̂+j
            g[i] = j > nV̂ ? 0 : estim.con.V̂min[jcon] - V̂[j] - ε*estim.con.C_v̂min[jcon]
        elseif i ≤ 2nX̂con + 2nV̂con
            j = i - 2nX̂con - nV̂con
            jcon = nV̂con-nV̂+j
            g[i] = j > nV̂ ? 0 : V̂[j] - estim.con.V̂max[jcon] - ε*estim.con.C_v̂max[jcon]
        else
            j = i - 2nX̂con - 2nV̂con
            g[i] = gc[j]
        end
    end
    return g
end

"""
    con_nonlinprog_mhe!(g, ::MovingHorizonEstimator, ::LinModel, _ , _ , gc, _ )

Compute the same but for [`LinModel`](@ref). 

The nonlinear custom inequality constraints in `gc` are the only nonlinear constraints
for this case. 
"""
function con_nonlinprog_mhe!(g, ::MovingHorizonEstimator, ::LinModel, _ , _ , gc , _ )
    for i in eachindex(g)
        g[i] = gc[i]
    end
    return g
end

"Throw an error if P̂ != nothing."
function setstate_cov!(::MovingHorizonEstimator, P̂)
    isnothing(P̂) || error("MovingHorizonEstimator does not compute an estimation covariance matrix P̂.")
    return nothing
end

"Update the augmented model, prediction matrices, constrains and data windows for MHE."
function setmodel_estimator!(
    estim::MovingHorizonEstimator, model, uop_old, yop_old, dop_old, Q̂, R̂
)
    con = estim.con
    nx̂, nym, nu, nd, He, nε = estim.nx̂, estim.nym, model.nu, model.nd, estim.He, estim.nε
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
    E, G, J, B, _ , Ex̂, Gx̂, Jx̂, Bx̂ = init_predmat_mhe(
        model, He, estim.i_ym, 
        estim.Â, estim.B̂u, estim.Ĉm, estim.B̂d, estim.D̂dm, 
        estim.x̂op, estim.f̂op, estim.direct
    )
    A_X̂min, A_X̂max, Ẽx̂ = relaxX̂(model, nε, con.C_x̂min, con.C_x̂max, Ex̂)   
    A_V̂min, A_V̂max, Ẽ  = relaxV̂(model, nε, con.C_v̂min, con.C_v̂max, E) 
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
        # convert y0m to ym with the old operating point:
        estim.Y0m[(1+nym*(i-1)):(nym*i)]  .+= @views yop_old[estim.i_ym]
        # convert u0 to u with the old operating point:
        estim.U0[(1+nu*(i-1)):(nu*i)]     .+= uop_old
        # convert d0 to d with the old operating point:
        estim.D0[(1+nd*(i-1)):(nd*i)]     .+= dop_old
         # convert x̂0 to x̂ with the old operating point:
        estim.X̂0_old[(1+nx̂*(i-1)):(nx̂*i)] .+= x̂op_old
        # convert ym to y0m with the new operating point:
        estim.Y0m[(1+nym*(i-1)):(nym*i)]  .-= @views model.yop[estim.i_ym]
        # convert u to u0 with the new operating point:
        estim.U0[(1+nu*(i-1)):(nu*i)]     .-= model.uop
        # convert d to d0 with the new operating point:
        estim.D0[(1+nd*(i-1)):(nd*i)]     .-= model.dop
        # convert x̂ to x̂0 with the new operating point:
        estim.X̂0_old[(1+nx̂*(i-1)):(nx̂*i)] .-= x̂op
    end
    estim.lastu0        .+= uop_old
    estim.Z̃[nε+1:nε+nx̂] .+= x̂op_old
    estim.x̂0arr_old     .+= x̂op_old
    estim.lastu0        .-= model.uop
    estim.Z̃[nε+1:nε+nx̂] .-= x̂op
    estim.x̂0arr_old     .-= x̂op
    # --- covariance matrices ---
    if !isnothing(Q̂)
        estim.cov.Q̂ .= to_hermitian(Q̂)
        invQ̂  = Hermitian(estim.buffer.Q̂, :L)
        invQ̂ .= estim.cov.Q̂
        try
            inv!(invQ̂)
        catch err
            if err isa PosDefException
                error("Q̂ is not positive definite")
            else
                rethrow()
            end
        end
        estim.cov.invQ̂_He .= Hermitian(repeatdiag(invQ̂, He), :L)
    end
    if !isnothing(R̂) 
        estim.cov.R̂ .= to_hermitian(R̂)
        invR̂  = Hermitian(estim.buffer.R̂, :L)
        invR̂ .= estim.cov.R̂
        try
            inv!(invR̂)
        catch err
            if err isa PosDefException
                error("R̂ is not positive definite")
            else
                rethrow()
            end
        end
        estim.cov.invR̂_He .= Hermitian(repeatdiag(invR̂, He), :L)
    end
    return nothing
end

"Called by plots recipes for the estimated states constraints."
getX̂con(estim::MovingHorizonEstimator, _ ) = estim.con.X̂0min+estim.X̂op, estim.con.X̂0max+estim.X̂op