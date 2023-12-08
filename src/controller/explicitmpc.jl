struct ExplicitMPC{NT<:Real, SE<:StateEstimator} <: PredictiveController{NT}
    estim::SE
    ΔŨ::Vector{NT}
    ŷ ::Vector{NT}
    Hp::Int
    Hc::Int
    M_Hp::Diagonal{NT, Vector{NT}}
    Ñ_Hc::Diagonal{NT, Vector{NT}}
    L_Hp::Diagonal{NT, Vector{NT}}
    C::NT
    E::NT
    R̂u::Vector{NT}
    R̂y::Vector{NT}
    noR̂u::Bool
    S̃::Matrix{NT} 
    T::Matrix{NT}
    Ẽ::Matrix{NT}
    F::Vector{NT}
    G::Matrix{NT}
    J::Matrix{NT}
    K::Matrix{NT}
    V::Matrix{NT}
    P̃::Hermitian{NT, Matrix{NT}}
    q̃::Vector{NT}
    p::Vector{NT}
    P̃_chol::Cholesky{NT, Matrix{NT}}
    Ks::Matrix{NT}
    Ps::Matrix{NT}
    d0::Vector{NT}
    D̂0::Vector{NT}
    D̂E::Vector{NT}
    Ŷop::Vector{NT}
    Dop::Vector{NT}
    function ExplicitMPC{NT, SE}(
        estim::SE, Hp, Hc, M_Hp, N_Hc, L_Hp
    ) where {NT<:Real, SE<:StateEstimator}
        model = estim.model
        nu, ny, nd = model.nu, model.ny, model.nd
        ŷ = copy(model.yop) # dummy vals (updated just before optimization)
        Cwt = Inf # no slack variable ϵ for ExplicitMPC
        Ewt = 0   # economic costs not supported for ExplicitMPC
        validate_weights(model, Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt)
        M_Hp, N_Hc, L_Hp = float(M_Hp), float(N_Hc), float(L_Hp) # debug julia 1.6
        # dummy vals (updated just before optimization):
        R̂y, R̂u = zeros(NT, ny*Hp), zeros(NT, nu*Hp)
        noR̂u = iszero(L_Hp)
        S, T = init_ΔUtoU(model, Hp, Hc)
        E, F, G, J, K, V = init_predmat(estim, model, Hp, Hc)
        S̃, Ñ_Hc, Ẽ  = S, N_Hc, E # no slack variable ϵ for ExplicitMPC
        P̃, q̃, p = init_quadprog(model, Ẽ, S̃, M_Hp, Ñ_Hc, L_Hp)
        P̃_chol = cholesky(P̃)
        Ks, Ps = init_stochpred(estim, Hp)
        # dummy vals (updated just before optimization):
        d0, D̂0, D̂E = zeros(NT, nd), zeros(NT, nd*Hp), zeros(NT, nd + nd*Hp)
        Ŷop, Dop = repeat(model.yop, Hp), repeat(model.dop, Hp)
        nvar = size(Ẽ, 2)
        ΔŨ = zeros(NT, nvar)
        mpc = new{NT, SE}(
            estim,
            ΔŨ, ŷ,
            Hp, Hc, 
            M_Hp, Ñ_Hc, L_Hp, Cwt, Ewt, 
            R̂u, R̂y, noR̂u,
            S̃, T, 
            Ẽ, F, G, J, K, V, P̃, q̃, p,
            P̃_chol,
            Ks, Ps,
            d0, D̂0, D̂E,
            Ŷop, Dop,
        )
        return mpc
    end
end

@doc raw"""
    ExplicitMPC(model::LinModel; <keyword arguments>)

Construct an explicit linear predictive controller based on [`LinModel`](@ref) `model`.

The controller minimizes the following objective function at each discrete time ``k``:
```math
\begin{aligned}
\min_{\mathbf{ΔU}}   \mathbf{(R̂_y - Ŷ)}' \mathbf{M}_{H_p} \mathbf{(R̂_y - Ŷ)}     
                   + \mathbf{(ΔU)}'      \mathbf{N}_{H_c} \mathbf{(ΔU)}        \\
                   + \mathbf{(R̂_u - U)}' \mathbf{L}_{H_p} \mathbf{(R̂_u - U)} 
\end{aligned}
```

See [`LinMPC`](@ref) for the variable definitions. This controller does not support
constraints but the computational costs are extremely low (array division), therefore 
suitable for applications that require small sample times. This method uses the default
state estimator, a [`SteadyKalmanFilter`](@ref) with default arguments.

# Arguments
- `model::LinModel` : model used for controller predictions and state estimations.
- `Hp=10+nk`: prediction horizon ``H_p``, `nk` is the number of delays in `model`.
- `Hc=2` : control horizon ``H_c``.
- `Mwt=fill(1.0,model.ny)` : main diagonal of ``\mathbf{M}`` weight matrix (vector).
- `Nwt=fill(0.1,model.nu)` : main diagonal of ``\mathbf{N}`` weight matrix (vector).
- `Lwt=fill(0.0,model.nu)` : main diagonal of ``\mathbf{L}`` weight matrix (vector).
- `M_Hp` / `N_Hc` / `L_Hp` : diagonal matrices ``\mathbf{M}_{H_p}, \mathbf{N}_{H_c},
  \mathbf{L}_{H_p}``, for time-varying weights (generated from `Mwt/Nwt/Lwt` args if omitted).
- additional keyword arguments are passed to [`SteadyKalmanFilter`](@ref) constructor.

# Examples
```jldoctest
julia> model = LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 4);

julia> mpc = ExplicitMPC(model, Mwt=[0, 1], Nwt=[0.5], Hp=30, Hc=1)
ExplicitMPC controller with a sample time Ts = 4.0 s, SteadyKalmanFilter estimator and:
 30 prediction steps Hp
  1 control steps Hc
  1 manipulated inputs u (0 integrating states)
  4 states x̂
  2 measured outputs ym (2 integrating states)
  0 unmeasured outputs yu
  0 measured disturbances d
```

"""
function ExplicitMPC(
    model::LinModel; 
    Hp::Union{Int, Nothing} = nothing,
    Hc::Int = DEFAULT_HC,
    Mwt = fill(DEFAULT_MWT, model.ny),
    Nwt = fill(DEFAULT_NWT, model.nu),
    Lwt = fill(DEFAULT_LWT, model.nu),
    M_Hp = nothing,
    N_Hc = nothing,
    L_Hp = nothing,
    kwargs...
) 
    estim = SteadyKalmanFilter(model; kwargs...)
    return ExplicitMPC(estim; Hp, Hc, Mwt, Nwt, Lwt, M_Hp, N_Hc, L_Hp)
end

"""
    ExplicitMPC(estim::StateEstimator; <keyword arguments>)

Use custom state estimator `estim` to construct `ExplicitMPC`.

`estim.model` must be a [`LinModel`](@ref). Else, a [`NonLinMPC`](@ref) is required. 

# Examples
```jldoctest
julia> estim = KalmanFilter(LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 4), i_ym=[2]);

julia> mpc = ExplicitMPC(estim, Mwt=[0, 1], Nwt=[0.5], Hp=30, Hc=1)
ExplicitMPC controller with a sample time Ts = 4.0 s, KalmanFilter estimator and:
 30 prediction steps Hp
  1 control steps Hc
  1 manipulated inputs u (0 integrating states)
  3 states x̂
  1 measured outputs ym (1 integrating states)
  1 unmeasured outputs yu
  0 measured disturbances d
```
"""
function ExplicitMPC(
    estim::SE;
    Hp::Union{Int, Nothing} = nothing,
    Hc::Int = DEFAULT_HC,
    Mwt = fill(DEFAULT_MWT, estim.model.ny),
    Nwt = fill(DEFAULT_NWT, estim.model.nu),
    Lwt = fill(DEFAULT_LWT, estim.model.nu),
    M_Hp = nothing,
    N_Hc = nothing,
    L_Hp = nothing
) where {NT<:Real, SE<:StateEstimator{NT}}
    isa(estim.model, LinModel) || error("estim.model type must be LinModel") 
    Hp = default_Hp(estim.model, Hp)
    isnothing(M_Hp) && (M_Hp = Diagonal{NT}(repeat(Mwt, Hp)))
    isnothing(N_Hc) && (N_Hc = Diagonal{NT}(repeat(Nwt, Hc)))
    isnothing(L_Hp) && (L_Hp = Diagonal{NT}(repeat(Lwt, Hp)))
    return ExplicitMPC{NT, SE}(estim, Hp, Hc, M_Hp, N_Hc, L_Hp)
end

setconstraint!(::ExplicitMPC,kwargs...) = error("ExplicitMPC does not support constraints.")

function Base.show(io::IO, mpc::ExplicitMPC)
    Hp, Hc = mpc.Hp, mpc.Hc
    nu, nd = mpc.estim.model.nu, mpc.estim.model.nd
    nx̂, nym, nyu = mpc.estim.nx̂, mpc.estim.nym, mpc.estim.nyu
    n = maximum(ndigits.((Hp, Hc, nu, nx̂, nym, nyu, nd))) + 1
    println(io, "$(typeof(mpc).name.name) controller with a sample time Ts = "*
                "$(mpc.estim.model.Ts) s, "*
                "$(typeof(mpc.estim).name.name) estimator and:")
    println(io, "$(lpad(Hp, n)) prediction steps Hp")
    println(io, "$(lpad(Hc, n)) control steps Hc")
    print_estim_dim(io, mpc.estim, n)
end

linconstraint!(::ExplicitMPC, ::LinModel) = nothing

@doc raw"""
Analytically solve the optimization problem for [`ExplicitMPC`](@ref).

The solution is ``\mathbf{ΔŨ = - P̃^{-1} q̃}``, see [`init_quadprog`](@ref).
"""
function optim_objective!(mpc::ExplicitMPC)
    return lmul!(-1, ldiv!(mpc.ΔŨ, mpc.P̃_chol, mpc.q̃))
end

"Compute the predictions but not the terminal states if `mpc` is an [`ExplicitMPC`](@ref)."
function predict!(Ŷ, x̂, mpc::ExplicitMPC, ::LinModel, ΔŨ::Vector{NT}) where {NT<:Real}
    # in-place operations to reduce allocations :
    mul!(Ŷ, mpc.Ẽ, ΔŨ) + mpc.F
    x̂[:] .= NaN
    return Ŷ, x̂
end


"""
    addinfo!(info, mpc::ExplicitMPC) -> info

For [`ExplicitMPC`](@ref), add nothing to `info`.
"""
addinfo!(info, mpc::ExplicitMPC) = info


