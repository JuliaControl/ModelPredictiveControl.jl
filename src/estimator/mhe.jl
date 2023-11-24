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
    W̃::Vector{NT}
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
    P̂0  ::Hermitian{NT, Matrix{NT}}
    Q̂_He::Hermitian{NT, Matrix{NT}}
    R̂_He::Hermitian{NT, Matrix{NT}}
    X̂min::Vector{NT}
    X̂max::Vector{NT}
    X̂ ::Vector{NT}
    Ym::Vector{NT}
    U ::Vector{NT}
    D ::Vector{NT}
    Ŵ ::Vector{NT}
    x̂0_past::Vector{NT}
    Nk::Vector{Int}
    function MovingHorizonEstimator{NT, SM, JM}(
        model::SM, He, i_ym, nint_u, nint_ym, P̂0, Q̂, R̂, optim::JM
    ) where {NT<:Real, SM<:SimModel{NT}, JM<:JuMP.GenericModel}
        nu, nd, nx, ny = model.nu, model.nd, model.nx, model.ny
        He < 1  && throw(ArgumentError("Estimation horizon He should be ≥ 1"))
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
        Q̂_He = Hermitian(repeatdiag(Q̂, He), :L)
        R̂_He = Hermitian(repeatdiag(R̂, He), :L)
        P̂ = copy(P̂0)
        X̂min, X̂max = fill(-Inf, nx̂*He), fill(+Inf, nx̂*He)
        nvar = nx̂*(He + 1) 
        W̃ = zeros(nvar)
        X̂, Ym, U, D, Ŵ = zeros(nx̂*He), zeros(nym*He), zeros(nu*He), zeros(nd*He), zeros(nx̂*He)
        x̂0_past = zeros(nx̂)
        Nk = [1]
        estim = new{NT, SM, JM}(
            model, optim, W̃,
            lastu0, x̂, P̂, He,
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs_u, Cs_y, nint_u, nint_ym,
            Â, B̂u, Ĉ, B̂d, D̂d,
            P̂0, Q̂_He, R̂_He,
            X̂min, X̂max, 
            X̂, Ym, U, D, Ŵ, 
            x̂0_past, Nk
        )
        init_optimization!(estim, optim)
        return estim
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

#=
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
=#

"""
    init_optimization!(estim::MovingHorizonEstimator, optim::JuMP.GenericModel)

Init the nonlinear optimization of [`MovingHorizonEstimator`](@ref).
"""
function init_optimization!(
    estim::MovingHorizonEstimator, optim::JuMP.GenericModel{JNT}
) where JNT<:Real
    # --- variables and linear constraints ---
    nvar = length(estim.W̃)
    set_silent(optim)
    #limit_solve_time(estim)
    @variable(optim, W̃var[1:nvar])
    # --- nonlinear optimization init ---
    nym, nx̂, He = estim.nym, estim.nx̂, estim.He #, length(i_g)
    # inspired from https://jump.dev/JuMP.jl/stable/tutorials/nonlinear/tips_and_tricks/#User-defined-operators-with-vector-outputs
    Jfunc = let estim=estim, model=estim.model, nvar=nvar , nŶm=He*nym, nX̂=He*nx̂
        last_W̃tup_float, last_W̃tup_dual = nothing, nothing
        Ŷm_cache::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(nŶm), nvar + 3)
        X̂_cache ::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(nX̂) , nvar + 3)
        function Jfunc(W̃tup::Float64...)
            Ŷm, X̂ = get_tmp(Ŷm_cache, W̃tup[1]), get_tmp(X̂_cache, W̃tup[1])
            W̃ = collect(W̃tup)
            if W̃tup != last_W̃tup_float
                Ŷm, _ = predict!(Ŷm, X̂, estim, model, W̃)
                last_W̃tup_float = W̃tup
            end
            return obj_nonlinprog(estim, model, Ŷm, W̃)
        end
        function Jfunc(W̃tup::Real...)
            Ŷm, X̂ = get_tmp(Ŷm_cache, W̃tup[1]), get_tmp(X̂_cache, W̃tup[1])
            W̃ = collect(W̃tup)
            if W̃tup != last_W̃tup_dual
                Ŷm, _ = predict!(Ŷm, X̂, estim, model, W̃)
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
function update_estimate!(estim::MovingHorizonEstimator, u, ym, d)
    return esti.x̂, estim.P̂
end

"""
    obj_nonlinprog(estim::MovingHorizonEstimator, model::SimModel, ΔŨ::Vector{Real})

Objective function for [`NonLinMHE`] when `model` is not a [`LinModel`](@ref).
"""
function obj_nonlinprog(estim::MovingHorizonEstimator, ::SimModel, Ŷm, W̃::Vector{T}) where {T<:Real}
    # `@views` macro avoid copies with matrix slice operator e.g. [a:b]
    @views x̄0 = W̃[1:estim.nx̂] - estim.x̂0_past
    V̂  = estim.Ym - Ŷm
    @views Ŵ = W̃[estim.nx̂+1:end]
    return x̄0'/estim.P̂0*x̄0 + Ŵ'/Q̂_He*Ŵ + V̂'/R̂_He*V̂
end

function predict!(
    Ŷm, X̂, estim::MovingHorizonEstimator, model::SimModel, W̃::Vector{T}
) where {T<:Real}
    nu, nd, nx̂, He = model.nu, model.nd, estim.nx̂, estim.He
    u::Vector{T} = Vector{T}(undef, nu)
    d::Vector{T} = Vector{T}(undef, nu)
    ŵ::Vector{T} = Vector{T}(undef, nx̂)
    x̂::Vector{T} = W̃[1:nx̂]
    for j=1:He
        u[:] = estim.U[(1 + nu*(j-1)):(nu*j)]
        d[:] = estim.D[(1 + nd*(j-1)):(nd*j)]
        i = j+1
        ŵ[:] = W̃[(1 + nx̂*(i-1)):(nx̂*i)]
        X̂[(1 + nx̂*(j-1)):(nx̂*j)] = x̂
        Ŷm[(1 + ny*(j-1)):(ny*j)] = ĥ(estim, model, x̂, d)[estim.i_ym]
        x̂[:] = f̂(estim, model, x̂, u, d) + ŵ
    end
    return Ŷm, X̂
end

function update_estimate!(estim::MovingHorizonEstimator, u, ym, d)
    model, x̂, P̂ = estim.model, estim.x̂, estim.P̂
    nx̂, nym, nu, nd, nŵ = estim.nx̂, estim.nym, model.nu, model.nd, estim.nx̂
    ŵ = zeros(nŵ) # ŵ(k) = 0 for warm-starting
    # --- adding new data in the windows ---
    estim.X̂[:]  = [estim.X̂[nx̂+1:end]  ; x̂]
    estim.Ym[:] = [estim.Ym[nym+1:end]; ym]
    estim.U[:]  = [estim.U[nu+1:end]  ; u]
    estim.D[:]  = [estim.D[nd+1:end]  ; d]
    estim.Ŵ[:]  = [estim.Ŵ[nŵ+1:end]  ; ŵ]






    Nk, He = estim.Nk, estim.He
    Nk[] = Nk[] < He ? Nk[] + 1 : He
    return x̂, P̂
    #estim.x̂0_past[:] =
end