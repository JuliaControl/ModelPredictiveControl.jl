struct ExplicitMPC{S<:StateEstimator} <: PredictiveController
    estim::S
    ΔŨ::Vector{Float64}
    ŷ ::Vector{Float64}
    Hp::Int
    Hc::Int
    M_Hp::Diagonal{Float64, Vector{Float64}}
    Ñ_Hc::Diagonal{Float64, Vector{Float64}}
    L_Hp::Diagonal{Float64, Vector{Float64}}
    C::Float64
    E::Float64
    R̂u::Vector{Float64}
    R̂y::Vector{Float64}
    S̃_Hp::Matrix{Bool}
    T_Hp::Matrix{Bool}
    T_Hc::Matrix{Bool}
    Ẽ::Matrix{Float64}
    F::Vector{Float64}
    G::Matrix{Float64}
    J::Matrix{Float64}
    K::Matrix{Float64}
    Q::Matrix{Float64}
    P̃::Hermitian{Float64, Matrix{Float64}}
    q̃::Vector{Float64}
    p::Vector{Float64}
    P̃_chol::Cholesky{Float64, Matrix{Float64}}
    Ks::Matrix{Float64}
    Ps::Matrix{Float64}
    d::Vector{Float64}
    D̂::Vector{Float64}
    Ŷop::Vector{Float64}
    Dop::Vector{Float64}
    function ExplicitMPC{S}(estim::S, Hp, Hc, Mwt, Nwt, Lwt, ru) where {S<:StateEstimator}
        model = estim.model
        nu, ny, nd = model.nu, model.ny, model.nd
        ŷ = zeros(ny)
        Cwt = Inf # no slack variable ϵ for ExplicitMPC
        Ewt = 0   # economic costs not supported for ExplicitMPC
        validate_weights(model, Hp, Hc, Mwt, Nwt, Lwt, Cwt, ru)
        M_Hp = Diagonal{Float64}(repeat(Mwt, Hp))
        N_Hc = Diagonal{Float64}(repeat(Nwt, Hc)) 
        L_Hp = Diagonal{Float64}(repeat(Lwt, Hp))
        C = Cwt
        # manipulated input setpoint predictions are constant over Hp :
        R̂u = ~iszero(Lwt) ? repeat(ru, Hp) : R̂u = empty(estim.x̂)
        R̂y = zeros(ny* Hp) # dummy R̂y (updated just before optimization)
        S_Hp, T_Hp, S_Hc, T_Hc = init_ΔUtoU(nu, Hp, Hc)
        E, F, G, J, K, Q = init_predmat(estim, model, Hp, Hc)
        _ , S̃_Hp, Ñ_Hc, Ẽ = init_defaultcon(model, Hp, Hc, C, S_Hp, S_Hc, N_Hc, E)
        P̃, q̃, p = init_quadprog(model, Ẽ, S̃_Hp, M_Hp, Ñ_Hc, L_Hp)
        P̃_chol = cholesky(P̃)
        Ks, Ps = init_stochpred(estim, Hp)
        d, D̂ = zeros(nd), zeros(nd*Hp)
        Ŷop, Dop = repeat(model.yop, Hp), repeat(model.dop, Hp)
        nvar = size(Ẽ, 2)
        ΔŨ = zeros(nvar)
        mpc = new(
            estim,
            ΔŨ, ŷ,
            Hp, Hc, 
            M_Hp, Ñ_Hc, L_Hp, Cwt, Ewt, R̂u, R̂y,
            S̃_Hp, T_Hp, T_Hc, 
            Ẽ, F, G, J, K, Q, P̃, q̃, p,
            P̃_chol,
            Ks, Ps,
            d, D̂,
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
\min_{\mathbf{ΔU}}       \mathbf{(R̂_y - Ŷ)}' \mathbf{M}_{H_p} \mathbf{(R̂_y - Ŷ)}   
                       + \mathbf{(ΔU)}'      \mathbf{N}_{H_c} \mathbf{(ΔU)}  
                       + \mathbf{(R̂_u - U)}' \mathbf{L}_{H_p} \mathbf{(R̂_u - U)} 
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
- `ru=model.uop` : manipulated input setpoints ``\mathbf{r_u}`` (vector).

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
ExplicitMPC(model::LinModel; kwargs...) = ExplicitMPC(SteadyKalmanFilter(model); kwargs...)

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
    estim::S;
    Hp::Union{Int, Nothing} = nothing,
    Hc::Int = 2,
    Mwt = fill(1.0, estim.model.ny),
    Nwt = fill(0.1, estim.model.nu),
    Lwt = fill(0.0, estim.model.nu),
    ru  = estim.model.uop,
) where {S<:StateEstimator}
    isa(estim.model, LinModel) || error("estim.model type must be LinModel") 
    poles = eigvals(estim.model.A)
    nk = sum(poles .≈ 0)
    if isnothing(Hp)
        Hp = 10 + nk
    end
    if Hp ≤ nk
        @warn("prediction horizon Hp ($Hp) ≤ number of delays in model "*
              "($nk), the closed-loop system may be zero-gain (unresponsive) or unstable")
    end
    return ExplicitMPC{S}(estim, Hp, Hc, Mwt, Nwt, Lwt, ru)
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

"""
Analytically solve the optimization problem for [`ExplicitMPC`](@ref).
"""
function optim_objective!(mpc::ExplicitMPC)
    return lmul!(-1, ldiv!(mpc.ΔŨ, mpc.P̃_chol, mpc.q̃))
end

"For [`ExplicitMPC`](@ref), return an empty summary."
get_summary(::ExplicitMPC) = solution_summary(JuMP.Model(), verbose=true)
