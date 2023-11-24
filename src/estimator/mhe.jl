const DEFAULT_MHE_OPTIMIZER = optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes")

struct MovingHorizonEstimator{
    NT<:Real, 
    SM<:SimModel, 
    JM<:JuMP.GenericModel
} <: StateEstimator{NT}
    model::SM
    # note: `NT` and the number type `JNT` in `JuMP.GenericModel{JNT}` can be
    # different since solvers that support non-Float64 are scarce.
    optim::JM
    lastu0::Vector{NT}
    x̂::Vector{NT}
    P̂::Hermitian{NT, Matrix{NT}}
    He::Int
    i_ym::Vector{Int}
    nx̂ ::Int
    nym::Int
    nyu::Int
    nxs::Int
    As  ::Matrix{NT}
    Cs_u::Matrix{NT}
    Cs_y::Matrix{NT}
    nint_u ::Vector{Int}
    nint_ym::Vector{Int}
    Â ::Matrix{NT}
    B̂u::Matrix{NT}
    Ĉ ::Matrix{NT}
    B̂d::Matrix{NT}
    D̂d::Matrix{NT}
    P̂0::Hermitian{NT, Matrix{NT}}
    Q̂::Hermitian{NT, Matrix{NT}}
    R̂::Hermitian{NT, Matrix{NT}}
    function MovingHorizonEstimator{NT, SM, JM}(
        model::SM, He, i_ym, nint_u, nint_ym, P̂0, Q̂, R̂, optim::JM
    ) where {NT<:Real, SM<:SimModel{NT}, JM<:JuMP.GenericModel}
        nym, nyu = validate_ym(model, i_ym)
        As, Cs_u, Cs_y, nint_u, nint_ym = init_estimstoch(model, i_ym, nint_u, nint_ym)
        nxs = size(As, 1)
        nx̂  = model.nx + nxs
        Â, B̂u, Ĉ, B̂d, D̂d = augment_model(model, As, Cs_u, Cs_y)
        validate_kfcov(nym, nx̂, Q̂, R̂, P̂0)
        lastu0 = zeros(NT, model.nu)
        x̂ = [zeros(NT, model.nx); zeros(NT, nxs)]
        Q̂, R̂ = Hermitian(Q̂, :L),  Hermitian(R̂, :L)
        P̂0 = Hermitian(P̂0, :L)
        P̂ = copy(P̂0)
        estim = new{NT, SM, JM}(
            model, optim,
            lastu0, x̂, P̂, He,
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs_u, Cs_y, nint_u, nint_ym,
            Â, B̂u, Ĉ, B̂d, D̂d,
            P̂0, Q̂, R̂,
        )
        init_optimization!(estim, optim)
        return mhe
    end
end

function MovingHorizonEstimator(
    model::SM;
    He::Int=nothing,
    i_ym::IntRangeOrVector = 1:model.ny,
    σP0::Vector = fill(1/model.nx, model.nx),
    σQ::Vector  = fill(1/model.nx, model.nx),
    σR::Vector  = fill(1, length(i_ym)),
    nint_u   ::IntVectorOrInt = 0,
    σQint_u  ::Vector = fill(1, max(sum(nint_u), 0)),
    σP0int_u ::Vector = fill(1, max(sum(nint_u), 0)),
    nint_ym  ::IntVectorOrInt = default_nint(model, i_ym, nint_u),
    σQint_ym ::Vector = fill(1, max(sum(nint_ym), 0)),
    σP0int_ym::Vector = fill(1, max(sum(nint_ym), 0)),
    optim::JM = JuMP.Model(DEFAULT_MHE_OPTIMIZER, add_bridges=false),
) where {NT<:Real, SM<:SimModel{NT}, JM<:JuMP.GenericModel}
    # estimated covariances matrices (variance = σ²) :
    P̂0 = Diagonal{NT}([σP0; σP0int_u; σP0int_ym].^2)
    Q̂  = Diagonal{NT}([σQ;  σQint_u;  σQint_ym].^2)
    R̂  = Diagonal{NT}(σR.^2)
    return MovingHorizonEstimator{NT, SM, JM}(
        model, He, i_ym, nint_u, nint_ym, P̂0, Q̂, R̂, optim
    )
end

"""
    init_optimization!(estim::MovingHorizonEstimator, optim::JuMP.GenericModel)

Init the nonlinear optimization of [`MovingHorizonEstimator`](@ref).
"""
function init_optimization!(
    estim::MovingHorizonEstimator, optim::JuMP.GenericModel{JNT}
) where JNT<:Real
    # --- variables and linear constraints ---
    nvar = length(estim.ΔŨ)
    set_silent(optim)
    limit_solve_time(estim)
    @variable(optim, ΔŨvar[1:nvar])
    ΔŨvar = optim[:ΔŨvar]
    A = con.A[con.i_b, :]
    b = con.b[con.i_b]
    @constraint(optim, linconstraint, A*ΔŨvar .≤ b)
    # --- nonlinear optimization init ---
    model = estim.estim.model
    ny, nx̂, Hp, ng = model.ny, estim.estim.nx̂, estim.Hp, length(con.i_g)
    # inspired from https://jump.dev/JuMP.jl/stable/tutorials/nonlinear/tips_and_tricks/#User-defined-operators-with-vector-outputs
    Jfunc, gfunc = let mpc=estim, model=model, ng=ng, nvar=nvar , nŶ=Hp*ny, nx̂=nx̂
        last_ΔŨtup_float, last_ΔŨtup_dual = nothing, nothing
        Ŷ_cache::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, nŶ), nvar + 3)
        g_cache::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, ng), nvar + 3)
        x̂_cache::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, nx̂), nvar + 3)
        function Jfunc(ΔŨtup::JNT...)
            Ŷ = get_tmp(Ŷ_cache, ΔŨtup[1])
            ΔŨ = collect(ΔŨtup)
            if ΔŨtup != last_ΔŨtup_float
                x̂ = get_tmp(x̂_cache, ΔŨtup[1])
                g = get_tmp(g_cache, ΔŨtup[1])
                Ŷ, x̂end = predict!(Ŷ, x̂, mpc, model, ΔŨ)
                con_nonlinprog!(g, mpc, model, x̂end, Ŷ, ΔŨ)
                last_ΔŨtup_float = ΔŨtup
            end
            return obj_nonlinprog(mpc, model, Ŷ, ΔŨ)
        end
        function Jfunc(ΔŨtup::ForwardDiff.Dual...)
            Ŷ = get_tmp(Ŷ_cache, ΔŨtup[1])
            ΔŨ = collect(ΔŨtup)
            if ΔŨtup != last_ΔŨtup_dual
                x̂ = get_tmp(x̂_cache, ΔŨtup[1])
                g = get_tmp(g_cache, ΔŨtup[1])
                Ŷ, x̂end = predict!(Ŷ, x̂, mpc, model, ΔŨ)
                con_nonlinprog!(g, mpc, model, x̂end, Ŷ, ΔŨ)
                last_ΔŨtup_dual = ΔŨtup
            end
            return obj_nonlinprog(mpc, model, Ŷ, ΔŨ)
        end
        function gfunc_i(i, ΔŨtup::NTuple{N, JNT}) where N
            g = get_tmp(g_cache, ΔŨtup[1])
            if ΔŨtup != last_ΔŨtup_float
                x̂ = get_tmp(x̂_cache, ΔŨtup[1])
                Ŷ = get_tmp(Ŷ_cache, ΔŨtup[1])
                ΔŨ = collect(ΔŨtup)
                Ŷ, x̂end = predict!(Ŷ, x̂, mpc, model, ΔŨ)
                g = con_nonlinprog!(g, mpc, model, x̂end, Ŷ, ΔŨ)
                last_ΔŨtup_float = ΔŨtup
            end
            return g[i]
        end 
        function gfunc_i(i, ΔŨtup::NTuple{N, ForwardDiff.Dual}) where N
            g = get_tmp(g_cache, ΔŨtup[1])
            if ΔŨtup != last_ΔŨtup_dual
                x̂ = get_tmp(x̂_cache, ΔŨtup[1])
                Ŷ = get_tmp(Ŷ_cache, ΔŨtup[1])
                ΔŨ = collect(ΔŨtup)
                Ŷ, x̂end = predict!(Ŷ, x̂, mpc, model, ΔŨ)
                g = con_nonlinprog!(g, mpc, model, x̂end, Ŷ, ΔŨ)
                last_ΔŨtup_dual = ΔŨtup
            end
            return g[i]
        end
        gfunc = [(ΔŨ...) -> gfunc_i(i, ΔŨ) for i in 1:ng]
        (Jfunc, gfunc)
    end
    register(optim, :Jfunc, nvar, Jfunc, autodiff=true)
    @NLobjective(optim, Min, Jfunc(ΔŨvar...))
    if ng ≠ 0
        i_end_Ymin, i_end_Ymax, i_end_x̂min = 1Hp*ny, 2Hp*ny, 2Hp*ny + nx̂
        for i in eachindex(con.Ymin)
            sym = Symbol("g_Ymin_$i")
            register(optim, sym, nvar, gfunc[i], autodiff=true)
        end
        for i in eachindex(con.Ymax)
            sym = Symbol("g_Ymax_$i")
            register(optim, sym, nvar, gfunc[i_end_Ymin+i], autodiff=true)
        end
        for i in eachindex(con.x̂min)
            sym = Symbol("g_x̂min_$i")
            register(optim, sym, nvar, gfunc[i_end_Ymax+i], autodiff=true)
        end
        for i in eachindex(con.x̂max)
            sym = Symbol("g_x̂max_$i")
            register(optim, sym, nvar, gfunc[i_end_x̂min+i], autodiff=true)
        end
    end
    return nothing
end


@doc raw"""
    update_estimate!(estim::UnscentedKalmanFilter, u, ym, d)
    
Update [`UnscentedKalmanFilter`](@ref) state `estim.x̂` and covariance estimate `estim.P̂`.

A ref[^4]:

```math
\begin{aligned}
    \mathbf{Ŷ^m}(k) &= \bigg[\begin{matrix} \mathbf{ĥ^m}\Big( \mathbf{X̂}_{k-1}^{1}(k) \Big) & \mathbf{ĥ^m}\Big( \mathbf{X̂}_{k-1}^{2}(k) \Big) & \cdots & \mathbf{ĥ^m}\Big( \mathbf{X̂}_{k-1}^{n_σ}(k) \Big) \end{matrix}\bigg] \\
    \mathbf{ŷ^m}(k) &= \mathbf{Ŷ^m}(k) \mathbf{m̂} 
\end{aligned} 
```

[^4]: TODO
"""
function update_estimate!(estim::MovingHorizonEstimator, u, ym, d) where NT<:Real
    x̂, P̂, Q̂, R̂, K̂ = estim.x̂, estim.P̂, estim.Q̂, estim.R̂, estim.K̂
    nym, nx̂, nσ = estim.nym, estim.nx̂, estim.nσ
    γ, m̂, Ŝ = estim.γ, estim.m̂, estim.Ŝ
    # --- correction step ---
    sqrt_P̂ = cholesky(P̂).L
    X̂ = repeat(x̂, 1, nσ) + [zeros(NT, nx̂) +γ*sqrt_P̂ -γ*sqrt_P̂]
    Ŷm = Matrix{NT}(undef, nym, nσ)
    for j in axes(Ŷm, 2)
        Ŷm[:, j] = ĥ(estim, estim.model, X̂[:, j], d)[estim.i_ym]
    end
    ŷm = Ŷm * m̂
    X̄ = X̂ .- x̂
    Ȳm = Ŷm .- ŷm
    M̂ = Hermitian(Ȳm * Ŝ * Ȳm' + R̂, :L)
    mul!(K̂, X̄, lmul!(Ŝ, Ȳm'))
    rdiv!(K̂, cholesky(M̂))
    x̂_cor = x̂ + K̂ * (ym - ŷm)
    P̂_cor = P̂ - Hermitian(K̂ * M̂ * K̂', :L)
    # --- prediction step ---
    sqrt_P̂_cor = cholesky(P̂_cor).L
    X̂_cor = repeat(x̂_cor, 1, nσ) + [zeros(NT, nx̂) +γ*sqrt_P̂_cor -γ*sqrt_P̂_cor]
    X̂_next = Matrix{NT}(undef, nx̂, nσ)
    for j in axes(X̂_next, 2)
        X̂_next[:, j] = f̂(estim, estim.model, X̂_cor[:, j], u, d)
    end
    x̂[:] = X̂_next * m̂
    X̄_next = X̂_next .- x̂
    P̂.data[:] = X̄_next * Ŝ * X̄_next' + Q̂ # .data is necessary for Hermitians
    return x̂, P̂
end