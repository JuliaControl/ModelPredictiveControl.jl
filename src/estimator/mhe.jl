struct NonLinMHE{M<:SimModel} <: StateEstimator
    model::M
    optim::JuMP.Model
    W̃::Vector{Float64}
    lastu0::Vector{Float64}
    x̂::Vector{Float64}
    P̂::Hermitian{Float64, Matrix{Float64}}
    i_ym::Vector{Int}
    nx̂::Int
    nym::Int
    nyu::Int
    nxs::Int
    As::Matrix{Float64}
    Cs::Matrix{Float64}
    nint_ym::Vector{Int}
    He::Int
    P̂0::Hermitian{Float64, Matrix{Float64}}
    Q̂_He::Hermitian{Float64, Matrix{Float64}}
    R̂_He::Hermitian{Float64, Matrix{Float64}}
    X̂min::Vector{Float64}
    X̂max::Vector{Float64}
    X̂ ::Vector{Float64}
    Ym::Vector{Float64}
    U ::Vector{Float64}
    D ::Vector{Float64}
    Ŵ ::Vector{Float64}
    x̂0_past::Vector{Float64}
    function NonLinMHE{M}(model::M, i_ym, nint_ym, He, P̂0, Q̂, R̂, optim) where {M<:SimModel}
        nu, nd, nx, ny = model.nu, model.nd, model.nx, model.ny
        He < 1  && error("Estimation horizon He should be ≥ 1")
        nym, nyu = length(i_ym), ny - length(i_ym)
        Asm, Csm, nint_ym = init_estimstoch(i_ym, nint_ym)
        nxs = size(Asm,1)
        nx̂ = nx + nxs
        validate_kfcov(nym, nx̂, Q̂, R̂, P̂0)
        As, _ , Cs, _  = stoch_ym2y(model, i_ym, Asm, [], Csm, [])
        i_ym = collect(i_ym)
        lastu0 = zeros(nu)
        x̂ = [zeros(model.nx); zeros(nxs)]
        P̂0 = Hermitian(P̂0, :L)
        Q̂_He = Hermitian(repeatdiag(Q̂, He), :L)
        R̂_He = Hermitian(repeatdiag(R̂, He), :L)
        P̂ = copy(P̂0)
        X̂min, X̂max = fill(-Inf, nx̂*He), fill(+Inf, nx̂*He)
        nvar = nx̂*(He + 1) 
        W̃ = zeros(nvar)
        X̂, Ym, U, D, Ŵ = zeros(nx̂*He), zeros(nym*He), zeros(nu*He), zeros(nd*He), zeros(nx̂*He)
        x̂0_past = zeros(nx̂)
        estim = new(
            model, optim, W̃,
            lastu0, x̂, P̂, 
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs, nint_ym,
            He, P̂0, Q̂_He, R̂_He,
            X̂min, X̂max,
            X̂, Ym, U, D, Ŵ,
            x̂0_past
        )
        init_optimization!(estim)
        return estim
    end
end

@doc raw"""
    NonLinMHE(model::SimModel; <keyword arguments>)

Construct a nonlinear moving horizon estimator with the [`SimModel`](@ref) `model`.

Both [`LinModel`](@ref) and [`NonLinModel`](@ref) are supported. The process model is 
identical to [`UnscentedKalmanFilter`](@ref).

# Arguments
- `model::SimModel` : (deterministic) model for the estimations.
- `He::Int=10` : estimation horizon.
- `<keyword arguments>` of [`SteadyKalmanFilter`](@ref) constructor.
- `<keyword arguments>` of [`KalmanFilter`](@ref) constructor.

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_)->0.1x+u, (x,_)->2x, 10.0, 1, 1, 1);

```
"""
function NonLinMHE(
    model::M;
    i_ym::IntRangeOrVector = 1:model.ny,
    He::Int = 10,
    σP0::Vector = fill(1/model.nx, model.nx),
    σQ::Vector  = fill(1/model.nx, model.nx),
    σR::Vector  = fill(1, length(i_ym)),
    nint_ym::IntVectorOrInt = fill(1, length(i_ym)),
    σP0_int::Vector = fill(1, max(sum(nint_ym), 0)),
    σQ_int::Vector  = fill(1, max(sum(nint_ym), 0)),
    optim::JuMP.Model = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"))
) where {M<:SimModel}
    # estimated covariances matrices (variance = σ²) :
    P̂0 = Diagonal{Float64}([σP0  ; σP0_int   ].^2);
    Q̂  = Diagonal{Float64}([σQ   ; σQ_int    ].^2);
    R̂  = Diagonal{Float64}(σR.^2);
    return NonLinMHE{M}(model, i_ym, nint_ym, He, P̂0, Q̂, R̂, optim)
end

@doc raw"""
    NonLinMHE{M<:SimModel}(model::M, i_ym, nint_ym, P̂0, Q̂, R̂, He)

Construct the estimator from the augmented covariance matrices `P̂0`, `Q̂` and `R̂`.

This syntax allows nonzero off-diagonal elements in ``\mathbf{P̂}_{-1}(0), \mathbf{Q̂, R̂}``.
"""
NonLinMHE{M}(model::M, i_ym, nint_ym, P̂0, Q̂, R̂, He) where {M<:SimModel}

@doc raw"""
    init_stochpred(As, Cs, He)

Construct stochastic model prediction matrices for nonlinear observers.
   
It allows separate simulations of the stochastic model (integrators). Stochastic model 
states and outputs are simulated using :
```math
\begin{aligned}
    \mathbf{X̂_s} &= \mathbf{M_s x̂_s}(k) + \mathbf{N_s Ŵ_s} 
    \mathbf{Ŷ_s} &= \mathbf{P_s X̂_s}
\end{aligned}
```
where ``\mathbf{X̂_s}`` and ``\mathbf{Ŷ_s}`` are integrator states and outputs from ``k + 1`` 
to ``k + H_e`` inclusively, and ``\mathbf{Ŵ_s}`` are integrator process noises from ``k`` to
``k + H_e - 1``. Stochastic simulations are combined with ``\mathbf{f, h}`` results to 
simulate the augmented model:
```math
\begin{aligned}
    \mathbf{X̂} &= \mathbf{M_s x̂_s}(k) + \mathbf{N_s Ŵ_s} 
    \mathbf{Ŷ_s} &= \mathbf{P_s X̂_s}
\end{aligned}
```
       Xhat = [XhatD; XhatS]
       Yhat = YhatD + YhatS
where XhatD and YhatD are SimulFunc states and outputs from ``k + 1`` to ``k + H_e``.
"""
function init_stochpred(As, Cs, He)
    nxs = size(As,1);
    Ms = zeros(He*nxs, nxs);
    for i = 1:Ho
        iRow = (1:nxs) + nxs*(i-1);
        Ms[iRow, :] = As^i;
    end
    Ns = zeros(Ho*nxs, He*nxs);
    for i = 1:Ho
        iCol = (1:nxs) + nxs*(i-1);
        Ns[nxs*(i-1)+1:end, iCol] = [eye(nxs); Ms(1:nxs*(Ho-i),:)];
    end
    Ps = kron(eye(Ho),Cs);
    return (Ms, Ns, Ps)
end
    
    

"""
    init_optimization!(estim::NonLinMHE)

Init the nonlinear optimization for [`NonLinMHE`](@ref).
"""
function init_optimization!(estim::NonLinMHE)
       # --- variables and linear constraints ---
       optim = estim.optim
       nvar = length(estim.W̃)
       set_silent(optim)
       @variable(optim, W̃var[1:nvar])
       W̃var = optim[:W̃var]
       # --- nonlinear optimization init ---
       model = estim.model
       nym, nx̂, He = estim.nym, estim.nx̂, estim.He
       # inspired from https://jump.dev/JuMP.jl/stable/tutorials/nonlinear/tips_and_tricks/#User-defined-functions-with-vector-outputs
       Jfunc = let estim=estim, model=model, nvar=nvar , nŶm=He*nym, nX̂=He*nx̂
            last_W̃tup_float, last_W̃tup_dual = nothing, nothing
            Ŷm_cache::DiffCacheType = DiffCache(zeros(nŶm), nvar + 3)
            X̂_cache ::DiffCacheType = DiffCache(zeros(nX̂) , nvar + 3)
            function Jfunc(W̃tup::Float64...)
                Ŷm, X̂ = get_tmp(Ŷm_cache, W̃tup[1]), get_tmp(X̂_cache, W̃tup[1])
                W̃ = collect(W̃tup)
                if W̃tup != last_W̃tup_float
                    Ŷm[:], X̂[:] = predict(estim, model, W̃)
                    last_W̃tup_float = W̃tup
                end
                return obj_nonlinprog(estim, model, Ŷm, W̃)
            end
            function Jfunc(W̃tup::Real...)
                Ŷm, X̂ = get_tmp(Ŷm_cache, W̃tup[1]), get_tmp(X̂_cache, W̃tup[1])
                W̃ = collect(W̃tup)
                if W̃tup != last_W̃tup_dual
                    Ŷm[:], X̂[:] = predict(estim, model, W̃)
                    last_W̃tup_dual = W̃tup
                end
                return obj_nonlinprog(estim, model, Ŷm, W̃)
            end
            Jfunc
       end
       register(optim, :Jfunc, nvar, Jfunc, autodiff=true)
       @NLobjective(optim, Min, Jfunc(W̃var...))
       return nothing
end

"""
    obj_nonlinprog(estim::NonLinMHE, model::SimModel, ΔŨ::Vector{Real})

Objective function for [`NonLinMHE`] when `model` is not a [`LinModel`](@ref).
"""
function obj_nonlinprog(estim::NonLinMHE, ::SimModel, Ŷm, W̃::Vector{T}) where {T<:Real}
    # `@views` macro avoid copies with matrix slice operator e.g. [a:b]
    @views x̄0 = W̃[1:estim.nx̂] - estim.x̂0_past
    V̂  = estim.win_Ym - Ŷm
    @views Ŵ = W̃[estim.nx̂+1:end]
    return x̄0'/estim.P̂0*x̄0 + Ŵ'/Q̂_He*Ŵ + V̂'/R̂_He*V̂
end

function predict(estim::NonLinMHE, model::SimModel, W̃::Vector{T}) where {T<:Real}
    nu, nd, nym, nx̂, He = model.nu, model.nd, estim.nym, estim.nx̂, estim.He
    Ŷm::Vector{T} = Vector{T}(undef, nym*(He+1))
    X̂ ::Vector{T} = Vector{T}(undef, nx̂*(He+1))
    Ŵ ::Vector{T} = @views W̃[nx̂+1:end]
    u ::Vector{T} = Vector{T}(undef, nu)
    d ::Vector{T} = Vector{T}(undef, nu)
    ŵ ::Vector{T} = Vector{T}(undef, nx̂)
    x̂ ::Vector{T} = W̃[1:nx̂]
    for j=1:He
        u[:] = @views estim.U[(1 + nu*(j-1)):(nu*j)]
        d[:] = @views estim.D[(1 + nd*(j-1)):(nd*j)]
        ŵ[:] = @views Ŵ[(1 + nx̂*(j-1)):(nx̂*j)]
        X̂[(1 + nx̂*(j-1)):(nx̂*j)] = x̂
        Ŷm[(1 + ny*(j-1)):(ny*j)] = @views ĥ(estim, x̂, d)[estim.i_ym]
        x̂[:] = f̂(estim, x̂, u, d) + ŵ
    end
    X̂[end-nx̂+1:end] = x̂
    Ŷm[end-nx̂+1:end] = @views ĥ(estim, x̂, estin.D[end-nd+1:end])[estim.i_ym]
    return Ŷm, X̂
end

function update_estimate!(estim::NonLinMHE, u, ym, d)
    model, x̂ = estim.model, estim.x̂
    nx̂, nym, nu, nd, nŵ = estim.nx̂, estim.nym, model.nu, model.nd, estim.nx̂
    ŵ = zeros(nŵ) # ŵ(k) = 0 for warm-starting
    # --- adding new data in time windows ---
    estim.X̂[:]  = @views [estim.X̂[nx̂+1:end]  ; x̂]
    estim.Ym[:] = @views [estim.Ym[nym+1:end]; ym]
    estim.U[:]  = @views [estim.U[nu+1:end]  ; u]
    estim.D[:]  = @views [estim.D[nd+1:end]  ; d]
    estim.Ŵ[:]  = @views [estim.Ŵ[nŵ+1:end]  ; ŵ]
    
    #estim.x̂0_past[:] =
end