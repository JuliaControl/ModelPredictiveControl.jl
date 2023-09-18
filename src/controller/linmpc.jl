struct LinMPC{S<:StateEstimator} <: PredictiveController
    estim::S
    optim::JuMP.Model
    con::ControllerConstraint
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
    Ks::Matrix{Float64}
    Ps::Matrix{Float64}
    d::Vector{Float64}
    D̂::Vector{Float64}
    Ŷop::Vector{Float64}
    Dop::Vector{Float64}
    function LinMPC{S}(estim::S, Hp, Hc, Mwt, Nwt, Lwt, Cwt, ru, optim) where {S<:StateEstimator}
        model = estim.model
        nu, ny, nd = model.nu, model.ny, model.nd
        ŷ = zeros(ny)
        Ewt = 0   # economic costs not supported for LinMPC
        validate_weights(model, Hp, Hc, Mwt, Nwt, Lwt, Cwt, ru)
        M_Hp = Diagonal{Float64}(repeat(Mwt, Hp))
        N_Hc = Diagonal{Float64}(repeat(Nwt, Hc)) 
        L_Hp = Diagonal{Float64}(repeat(Lwt, Hp))
        C = Cwt
        # manipulated input setpoint predictions are constant over Hp :
        R̂u = ~iszero(Lwt) ? repeat(ru, Hp) : R̂u = Float64[]
        R̂y = zeros(ny* Hp) # dummy R̂y (updated just before optimization)
        S_Hp, T_Hp, S_Hc, T_Hc = init_ΔUtoU(nu, Hp, Hc)
        E, F, G, J, K, Q = init_predmat(estim, model, Hp, Hc)
        con, S̃_Hp, Ñ_Hc, Ẽ = init_defaultcon(model, Hp, Hc, C, S_Hp, S_Hc, N_Hc, E)
        P̃, q̃, p = init_quadprog(model, Ẽ, S̃_Hp, M_Hp, Ñ_Hc, L_Hp)
        Ks, Ps = init_stochpred(estim, Hp)
        d, D̂ = zeros(nd), zeros(nd*Hp)
        Ŷop, Dop = repeat(model.yop, Hp), repeat(model.dop, Hp)
        nvar = size(Ẽ, 2)
        ΔŨ = zeros(nvar)
        mpc = new(
            estim, optim, con,
            ΔŨ, ŷ,
            Hp, Hc, 
            M_Hp, Ñ_Hc, L_Hp, Cwt, Ewt, R̂u, R̂y,
            S̃_Hp, T_Hp, T_Hc, 
            Ẽ, F, G, J, K, Q, P̃, q̃, p,
            Ks, Ps,
            d, D̂,
            Ŷop, Dop,
        )
        init_optimization!(mpc)
        return mpc
    end
end

@doc raw"""
    LinMPC(model::LinModel; <keyword arguments>)

Construct a linear predictive controller based on [`LinModel`](@ref) `model`.

The controller minimizes the following objective function at each discrete time ``k``:
```math
\min_{\mathbf{ΔU}, ϵ}    \mathbf{(R̂_y - Ŷ)}' \mathbf{M}_{H_p} \mathbf{(R̂_y - Ŷ)}   
                       + \mathbf{(ΔU)}'      \mathbf{N}_{H_c} \mathbf{(ΔU)}  
                       + \mathbf{(R̂_u - U)}' \mathbf{L}_{H_p} \mathbf{(R̂_u - U)} 
                       + C ϵ^2
```
in which the weight matrices are repeated ``H_p`` or ``H_c`` times:
```math
\begin{aligned}
    \mathbf{M}_{H_p} &= \text{diag}\mathbf{(M,M,...,M)}     \\
    \mathbf{N}_{H_c} &= \text{diag}\mathbf{(N,N,...,N)}     \\
    \mathbf{L}_{H_p} &= \text{diag}\mathbf{(L,L,...,L)}     
\end{aligned}
```
The ``\mathbf{ΔU}`` vector includes the manipulated input increments ``\mathbf{Δu}(k+j) = 
\mathbf{u}(k+j) - \mathbf{u}(k+j-1)`` from ``j=0`` to ``H_c-1``, the ``\mathbf{Ŷ}`` vector, 
the output predictions ``\mathbf{ŷ}(k+j)`` from ``j=1`` to ``H_p``, and the ``\mathbf{U}`` 
vector, the manipulated inputs ``\mathbf{u}(k+j)`` from ``j=0`` to ``H_p-1``. The 
manipulated input setpoint predictions ``\mathbf{R̂_u}`` are constant at ``\mathbf{r_u}``.
See Extended Help for a detailed nomenclature.

This method uses the default state estimator, a [`SteadyKalmanFilter`](@ref) with default
arguments.

# Arguments
- `model::LinModel` : model used for controller predictions and state estimations.
- `Hp=10+nk`: prediction horizon ``H_p``, `nk` is the number of delays in `model`.
- `Hc=2` : control horizon ``H_c``.
- `Mwt=fill(1.0,model.ny)` : main diagonal of ``\mathbf{M}`` weight matrix (vector).
- `Nwt=fill(0.1,model.nu)` : main diagonal of ``\mathbf{N}`` weight matrix (vector).
- `Lwt=fill(0.0,model.nu)` : main diagonal of ``\mathbf{L}`` weight matrix (vector).
- `Cwt=1e5` : slack variable weight ``C`` (scalar), use `Cwt=Inf` for hard constraints only.
- `ru=model.uop` : manipulated input setpoints ``\mathbf{r_u}`` (vector).
- `optim=JuMP.Model(OSQP.MathOptInterfaceOSQP.Optimizer)` : quadratic optimizer used in
  the predictive controller, provided as a [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/reference/models/#JuMP.Model)
  (default to [`OSQP.jl`](https://osqp.org/docs/parsers/jump.html) optimizer).

# Examples
```jldoctest
julia> model = LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 4);

julia> mpc = LinMPC(model, Mwt=[0, 1], Nwt=[0.5], Hp=30, Hc=1)
LinMPC controller with a sample time Ts = 4.0 s, OSQP optimizer, SteadyKalmanFilter estimator and:
 30 prediction steps Hp
  1 control steps Hc
  1 manipulated inputs u (0 integrators)
  4 states x̂
  2 measured outputs ym (2 integrators)
  0 unmeasured outputs yu
  0 measured disturbances d
```

# Extended Help
Manipulated inputs setpoints ``\mathbf{r_u}`` are not common but they can be interesting
for over-actuated systems, when `nu > ny` (e.g. prioritize solutions with lower economical 
costs). The default `Lwt` value implies that this feature is disabled by default.

The objective function follows this nomenclature:

| VARIABLE         | DESCRIPTION                                        | SIZE             |
| :--------------- | :------------------------------------------------- | :--------------- |
| ``H_p``          | prediction horizon (integer)                       | `()`             |
| ``H_c``          | control horizon (integer)                          | `()`             |
| ``\mathbf{ΔU}``  | manipulated input increments over ``H_c``          | `(nu*Hc,)`       |
| ``\mathbf{Ŷ}``   | predicted outputs over ``H_p``                     | `(ny*Hp,)`       |
| ``\mathbf{U}``   | manipulated inputs over ``H_p``                    | `(nu*Hp,)`       |
| ``\mathbf{R̂_y}`` | predicted output setpoints over ``H_p``            | `(ny*Hp,)`       |
| ``\mathbf{R̂_u}`` | predicted manipulated input setpoints over ``H_p`` | `(nu*Hp,)`       |
| ``\mathbf{M}``   | output setpoint tracking weights                   | `(ny*Hp, ny*Hp)` |
| ``\mathbf{N}``   | manipulated input increment weights                | `(nu*Hc, nu*Hc)` |
| ``\mathbf{L}``   | manipulated input setpoint tracking weights        | `(nu*Hp, nu*Hp)` |
| ``C``            | slack variable weight                              | `()`             |
| ``ϵ``            | slack variable for constraint softening            | `()`             |
"""
LinMPC(model::LinModel; kwargs...) = LinMPC(SteadyKalmanFilter(model); kwargs...)


"""
    LinMPC(estim::StateEstimator; <keyword arguments>)

Use custom state estimator `estim` to construct `LinMPC`.

`estim.model` must be a [`LinModel`](@ref). Else, a [`NonLinMPC`](@ref) is required. 

# Examples
```jldoctest
julia> estim = KalmanFilter(LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 4), i_ym=[2]);

julia> mpc = LinMPC(estim, Mwt=[0, 1], Nwt=[0.5], Hp=30, Hc=1)
LinMPC controller with a sample time Ts = 4.0 s, OSQP optimizer, KalmanFilter estimator and:
 30 prediction steps Hp
  1 control steps Hc
  1 manipulated inputs u (0 integrators)
  3 states x̂
  1 measured outputs ym (1 integrators)
  1 unmeasured outputs yu
  0 measured disturbances d
```
"""
function LinMPC(
    estim::S;
    Hp::Union{Int, Nothing} = nothing,
    Hc::Int = 2,
    Mwt = fill(1.0, estim.model.ny),
    Nwt = fill(0.1, estim.model.nu),
    Lwt = fill(0.0, estim.model.nu),
    Cwt = 1e5,
    ru  = estim.model.uop,
    optim::JuMP.Model = JuMP.Model(OSQP.MathOptInterfaceOSQP.Optimizer)
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
    return LinMPC{S}(estim, Hp, Hc, Mwt, Nwt, Lwt, Cwt, ru, optim)
end

"""
    init_optimization!(mpc::LinMPC)

Init the quadratic optimization for [`LinMPC`](@ref) controllers.
"""
function init_optimization!(mpc::LinMPC)
    # --- variables and linear constraints ---
    optim, con = mpc.optim, mpc.con
    nvar = length(mpc.ΔŨ)
    set_silent(optim)
    @variable(optim, ΔŨvar[1:nvar])
    A = con.A[con.i_b, :]
    b = con.b[con.i_b]
    @constraint(optim, linconstraint, A*ΔŨvar .≤ b)
    # --- quadratic optimization init ---
    @objective(mpc.optim, Min, obj_quadprog(ΔŨvar, mpc.P̃, mpc.q̃))
    return nothing
end

"For [`LinMPC`](@ref), set the QP linear coefficient `q̃` just before optimization."
function set_objective_linear_coef!(mpc::LinMPC, ΔŨvar)
    set_objective_coefficient.(mpc.optim, ΔŨvar, mpc.q̃)
    return nothing
end
