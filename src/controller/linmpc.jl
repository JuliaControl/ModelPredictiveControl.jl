struct LinMPC{S<:StateEstimator} <: PredictiveController
    estim::S
    optim::JuMP.Model
    info::OptimInfo
    con::ControllerConstraint
    Hp::Int
    Hc::Int
    M_Hp::Diagonal{Float64, Vector{Float64}}
    Ñ_Hc::Diagonal{Float64, Vector{Float64}}
    L_Hp::Diagonal{Float64, Vector{Float64}}
    C::Float64
    R̂u::Vector{Float64}
    S̃_Hp::Matrix{Bool}
    T_Hp::Matrix{Bool}
    S̃_Hc::Matrix{Bool}
    T_Hc::Matrix{Bool}
    Ẽ ::Matrix{Float64}
    F ::Vector{Float64}
    G ::Matrix{Float64}
    J ::Matrix{Float64}
    Kd::Matrix{Float64}
    Q ::Matrix{Float64}
    P̃ ::Hermitian{Float64, Matrix{Float64}}
    q̃ ::Vector{Float64}
    Ks::Matrix{Float64}
    Ps::Matrix{Float64}
    Yop::Vector{Float64}
    Dop::Vector{Float64}
    function LinMPC{S}(estim::S, Hp, Hc, Mwt, Nwt, Lwt, Cwt, ru, optim) where {S<:StateEstimator}
        model = estim.model
        nu, ny = model.nu, model.ny
        validate_weights(model, Hp, Hc, Mwt, Nwt, Lwt, Cwt, ru)
        M_Hp = Diagonal{Float64}(repeat(Mwt, Hp))
        N_Hc = Diagonal{Float64}(repeat(Nwt, Hc)) 
        L_Hp = Diagonal{Float64}(repeat(Lwt, Hp))
        C = Cwt
        # manipulated input setpoint predictions are constant over Hp :
        R̂u = ~iszero(Lwt) ? repeat(ru, Hp) : R̂u = Float64[] 
        umin,  umax      = fill(-Inf, nu), fill(+Inf, nu)
        Δumin, Δumax     = fill(-Inf, nu), fill(+Inf, nu)
        ŷmin,  ŷmax      = fill(-Inf, ny), fill(+Inf, ny)
        c_umin, c_umax   = fill(0.0, nu),  fill(0.0, nu)
        c_Δumin, c_Δumax = fill(0.0, nu),  fill(0.0, nu)
        c_ŷmin, c_ŷmax   = fill(1.0, ny),  fill(1.0, ny)
        Umin, Umax, ΔUmin, ΔUmax, Ŷmin, Ŷmax = 
            repeat_constraints(Hp, Hc, umin, umax, Δumin, Δumax, ŷmin, ŷmax)
        c_Umin, c_Umax, c_ΔUmin, c_ΔUmax, c_Ŷmin, c_Ŷmax = 
            repeat_constraints(Hp, Hc, c_umin, c_umax, c_Δumin, c_Δumax, c_ŷmin, c_ŷmax)
        S_Hp, T_Hp, S_Hc, T_Hc = init_ΔUtoU(nu, Hp, Hc)
        E, F, G, J, Kd, Q = init_deterpred(model, Hp, Hc)
        A_Umin, A_Umax, S̃_Hp, S̃_Hc = relaxU(C, c_Umin, c_Umax, S_Hp, S_Hc)
        A_ΔŨmin, A_ΔŨmax, ΔŨmin, ΔŨmax, Ñ_Hc = relaxΔU(C,c_ΔUmin,c_ΔUmax,ΔUmin,ΔUmax,N_Hc)
        A_Ŷmin, A_Ŷmax, Ẽ = relaxŶ(model, C, c_Ŷmin, c_Ŷmax, E)
        i_Umin,  i_Umax  = .!isinf.(Umin),  .!isinf.(Umax)
        i_ΔŨmin, i_ΔŨmax = .!isinf.(ΔŨmin), .!isinf.(ΔŨmax)
        i_Ŷmin,  i_Ŷmax  = .!isinf.(Ŷmin),  .!isinf.(Ŷmax)
        A, b, i_b = init_linconstraint(model, 
            A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, A_Ŷmin, A_Ŷmax,
            i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ŷmin, i_Ŷmax
        )
        con = ControllerConstraint(
            Umin    , Umax  , ΔŨmin  , ΔŨmax    , Ŷmin  , Ŷmax,
            c_Umin  , c_Umax, c_ΔUmin, c_ΔUmax  , c_Ŷmin, c_Ŷmax,
            A       , b     , i_b  , i_Ŷmin, i_Ŷmax
        )
        nvar = size(Ẽ, 2)
        P̃, q̃ = init_quadprog(model, Ẽ, S̃_Hp, M_Hp, Ñ_Hc, L_Hp)
        Ks, Ps = init_stochpred(estim, Hp)
        Yop, Dop = repeat(model.yop, Hp), repeat(model.dop, Hp)
        set_silent(optim)
        @variable(optim, ΔŨ[1:nvar])
        A = con.A[con.i_b, :]
        b = con.b[con.i_b]
        @constraint(optim, linconstraint, A*ΔŨ .≤ b)
        ΔŨ0 = zeros(nvar)
        ϵ = isinf(C) ? NaN : 0.0 # C = Inf means hard constraints only
        u, U = copy(model.uop), repeat(model.uop, Hp)
        ŷ, Ŷ = copy(model.yop), repeat(model.yop, Hp)
        ŷs, Ŷs = zeros(ny), zeros(ny*Hp)
        info = OptimInfo(ΔŨ0, ϵ, 0, u, U, ŷ, Ŷ, ŷs, Ŷs)
        mpc = new(
            estim, optim, info, con,
            Hp, Hc, 
            M_Hp, Ñ_Hc, L_Hp, Cwt, R̂u,
            S̃_Hp, T_Hp, S̃_Hc, T_Hc, 
            Ẽ, F, G, J, Kd, Q, P̃, q̃,
            Ks, Ps,
            Yop, Dop,
        )
        @objective(optim, Min, obj_quadprog(ΔŨ, mpc.P̃, mpc.q̃))
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
and with the following nomenclature:

| VAR.              | DESCRIPTION                                        |
| :---------------- | :------------------------------------------------- |
| ``H_p``           | prediction horizon                                 |
| ``H_c``           | control horizon                                    |
| ``\mathbf{ΔU}``   | manipulated input increments over ``H_c``          |
| ``\mathbf{Ŷ}``    | predicted outputs over ``H_p``                     |
| ``\mathbf{U}``    | manipulated inputs over ``H_p``                    |
| ``\mathbf{R̂_y}``  | predicted output setpoints over ``H_p``            |
| ``\mathbf{R̂_u}``  | predicted manipulated input setpoints over ``H_p`` |
| ``\mathbf{M}``    | output setpoint tracking weights                   |
| ``\mathbf{N}``    | manipulated input increment weights                |
| ``\mathbf{L}``    | manipulated input setpoint tracking weights        |
| ``C``             | slack variable weight                              |
| ``ϵ``             | slack variable for constraint softening            |

The ``\mathbf{ΔU}`` vector includes the manipulated input increments ``\mathbf{Δu}(k+j) = 
\mathbf{u}(k+j) - \mathbf{u}(k+j-1)`` from ``j=0`` to ``H_c-1``, the ``\mathbf{Ŷ}`` vector, 
the output predictions ``\mathbf{ŷ(k+j)}`` from ``j=1`` to ``H_p``, and the ``\mathbf{U}`` 
vector, the manipulated inputs ``\mathbf{u}(k+j)`` from ``j=0`` to ``H_p-1``. The 
manipulated input setpoint predictions ``\mathbf{R̂_u}`` are constant at ``\mathbf{r_u}``.

This method uses the default state estimator, a [`SteadyKalmanFilter`](@ref) with default
arguments.

# Arguments
- `model::LinModel` : model used for controller predictions and state estimations.
- `Hp=10+nk`: prediction horizon ``H_p``, `nk` is the number of delays in `model`.
- `Hc=2` : control horizon ``H_c``.
- `Mwt=fill(1.0,model.ny)` : main diagonal of ``\mathbf{M}`` weight matrix (vector)
- `Nwt=fill(0.1,model.nu)` : main diagonal of ``\mathbf{N}`` weight matrix (vector)
- `Lwt=fill(0.0,model.nu)` : main diagonal of ``\mathbf{L}`` weight matrix (vector)
- `Cwt=1e5` : slack variable weight ``C`` (scalar), use `Cwt=Inf` for hard constraints only
- `ru=model.uop` : manipulated input setpoints ``\mathbf{r_u}`` (vector)
- `optim=JuMP.Model(OSQP.MathOptInterfaceOSQP.Optimizer)` : quadratic optimizer used in
  the predictive controller, provided as a [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/reference/models/#JuMP.Model)
  (default to [`OSQP.jl`](https://osqp.org/docs/parsers/jump.html) optimizer)

# Examples
```jldoctest
julia> model = LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 4);

julia> mpc = LinMPC(model, Mwt=[0, 1], Nwt=[0.5], Hp=30, Hc=1)
LinMPC controller with a sample time Ts = 4.0 s, SteadyKalmanFilter estimator and:
 1 manipulated inputs u
 4 states x̂
 2 measured outputs ym
 0 unmeasured outputs yu
 0 measured disturbances d
```

# Extended Help
Manipulated inputs setpoints ``\mathbf{r_u}`` are not common but they can be interesting
for over-actuated systems, when `nu > ny` (e.g. prioritize solutions with lower economical 
costs). The default `Lwt` value implies that this feature is disabled by default.
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
LinMPC controller with a sample time Ts = 4.0 s, KalmanFilter estimator and:
 1 manipulated inputs u
 3 states x̂
 1 measured outputs ym
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

function init_objective(mpc::LinMPC, ΔŨ)
    set_objective_function(mpc.optim, obj_quadprog(ΔŨ, mpc.P̃, mpc.q̃))
end

"""
    write_info!(mpc::LinMPC, ΔŨ, ϵ, J, info, ŷs, Ŷs, lastu, F, ym, d)

Write `mpc.info` with the [`LinMPC`](@ref) optimization results.
"""
function write_info!(mpc::LinMPC, ΔŨ, J, ŷs, Ŷs, lastu, ym, d)
    mpc.info.ΔŨ = ΔŨ
    mpc.info.ϵ = isinf(mpc.C) ? NaN : ΔŨ[end]
    mpc.info.J = J
    mpc.info.U = mpc.S̃_Hp*ΔŨ + mpc.T_Hp*lastu
    mpc.info.u = mpc.info.U[1:mpc.estim.model.nu]
    mpc.info.ŷ = eval_ŷ(mpc.estim, ym, d)
    mpc.info.Ŷ = mpc.Ẽ*ΔŨ + mpc.F
    mpc.info.ŷs, mpc.info.Ŷs = ŷs, Ŷs
end