const DEFAULT_LINMPC_OPTIMIZER = OSQP.MathOptInterfaceOSQP.Optimizer

struct LinMPC{
    NT<:Real, 
    SE<:StateEstimator, 
    JM<:JuMP.GenericModel
} <: PredictiveController{NT}
    estim::SE
    # note: `NT` and the number type `JNT` in `JuMP.GenericModel{JNT}` can be
    # different since solvers that support non-Float64 are scarce.
    optim::JM
    con::ControllerConstraint{NT}
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
    T_lastu::Vector{NT}
    Ẽ::Matrix{NT}
    F::Vector{NT}
    G::Matrix{NT}
    J::Matrix{NT}
    K::Matrix{NT}
    V::Matrix{NT}
    H̃::Hermitian{NT, Matrix{NT}}
    q̃::Vector{NT}
    p::Vector{NT}
    Ks::Matrix{NT}
    Ps::Matrix{NT}
    d0::Vector{NT}
    D̂0::Vector{NT}
    D̂E::Vector{NT}
    Ŷop::Vector{NT}
    Dop::Vector{NT}
    function LinMPC{NT, SE, JM}(
        estim::SE, Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt, optim::JM
    ) where {NT<:Real, SE<:StateEstimator, JM<:JuMP.GenericModel}
        model = estim.model
        nu, ny, nd = model.nu, model.ny, model.nd
        ŷ = copy(model.yop) # dummy vals (updated just before optimization)
        Ewt = 0   # economic costs not supported for LinMPC
        validate_weights(model, Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt)
        M_Hp, N_Hc, L_Hp = Diagonal{NT}(M_Hp), Diagonal{NT}(N_Hc), Diagonal{NT}(L_Hp) # debug julia 1.6
        # dummy vals (updated just before optimization):
        R̂y, R̂u, T_lastu = zeros(NT, ny*Hp), zeros(NT, nu*Hp), zeros(NT, nu*Hp)
        noR̂u = iszero(L_Hp)
        S, T = init_ΔUtoU(model, Hp, Hc)
        E, F, G, J, K, V, ex̂, fx̂, gx̂, jx̂, kx̂, vx̂ = init_predmat(estim, model, Hp, Hc)
        con, S̃, Ñ_Hc, Ẽ = init_defaultcon_mpc(estim, Hp, Hc, Cwt, S, N_Hc, E, ex̂, fx̂, gx̂, jx̂, kx̂, vx̂)
        H̃, q̃, p = init_quadprog(model, Ẽ, S̃, M_Hp, Ñ_Hc, L_Hp)
        Ks, Ps = init_stochpred(estim, Hp)
        # dummy vals (updated just before optimization):
        d0, D̂0, D̂E = zeros(NT, nd), zeros(NT, nd*Hp), zeros(NT, nd + nd*Hp)
        Ŷop, Dop = repeat(model.yop, Hp), repeat(model.dop, Hp)
        nΔŨ = size(Ẽ, 2)
        ΔŨ = zeros(NT, nΔŨ)
        mpc = new{NT, SE, JM}(
            estim, optim, con,
            ΔŨ, ŷ,
            Hp, Hc, 
            M_Hp, Ñ_Hc, L_Hp, Cwt, Ewt, 
            R̂u, R̂y, noR̂u,
            S̃, T, T_lastu,
            Ẽ, F, G, J, K, V, H̃, q̃, p,
            Ks, Ps,
            d0, D̂0, D̂E,
            Ŷop, Dop,
        )
        init_optimization!(mpc, optim)
        return mpc
    end
end

@doc raw"""
    LinMPC(model::LinModel; <keyword arguments>)

Construct a linear predictive controller based on [`LinModel`](@ref) `model`.

The controller minimizes the following objective function at each discrete time ``k``:
```math
\begin{aligned}
\min_{\mathbf{ΔU}, ϵ}   \mathbf{(R̂_y - Ŷ)}' \mathbf{M}_{H_p} \mathbf{(R̂_y - Ŷ)}
                      + \mathbf{(ΔU)}'      \mathbf{N}_{H_c} \mathbf{(ΔU)}        \\
                      + \mathbf{(R̂_u - U)}' \mathbf{L}_{H_p} \mathbf{(R̂_u - U)} 
                      + C ϵ^2
\end{aligned}
```
in which the weight matrices are repeated ``H_p`` or ``H_c`` times by default:
```math
\begin{aligned}
    \mathbf{M}_{H_p} &= \text{diag}\mathbf{(M,M,...,M)}     \\
    \mathbf{N}_{H_c} &= \text{diag}\mathbf{(N,N,...,N)}     \\
    \mathbf{L}_{H_p} &= \text{diag}\mathbf{(L,L,...,L)}     
\end{aligned}
```
Time-varying weights over the horizons are also supported. The ``\mathbf{ΔU}`` includes the 
input increments ``\mathbf{Δu}(k+j) = \mathbf{u}(k+j) - \mathbf{u}(k+j-1)`` from ``j=0`` to
``H_c-1``, the ``\mathbf{Ŷ}`` vector, the output predictions ``\mathbf{ŷ}(k+j)`` from
``j=1`` to ``H_p``, and the ``\mathbf{U}`` vector, the manipulated inputs ``\mathbf{u}(k+j)``
from ``j=0`` to ``H_p-1``. The slack variable ``ϵ`` relaxes the constraints, as described
in [`setconstraint!`](@ref) documentation. See Extended Help for a detailed nomenclature. 

This method uses the default state estimator, a [`SteadyKalmanFilter`](@ref) with default
arguments.

# Arguments
- `model::LinModel` : model used for controller predictions and state estimations.
- `Hp=10+nk`: prediction horizon ``H_p``, `nk` is the number of delays in `model`.
- `Hc=2` : control horizon ``H_c``.
- `Mwt=fill(1.0,model.ny)` : main diagonal of ``\mathbf{M}`` weight matrix (vector).
- `Nwt=fill(0.1,model.nu)` : main diagonal of ``\mathbf{N}`` weight matrix (vector).
- `Lwt=fill(0.0,model.nu)` : main diagonal of ``\mathbf{L}`` weight matrix (vector).
- `M_Hp=Diagonal(repeat(Mwt),Hp)` : diagonal weight matrix ``\mathbf{M}_{H_p}``.
- `N_Hc=Diagonal(repeat(Nwt),Hc)` : diagonal weight matrix ``\mathbf{N}_{H_c}``.
- `L_Hp=Diagonal(repeat(Lwt),Hp)` : diagonal weight matrix ``\mathbf{L}_{H_p}``.
- `Cwt=1e5` : slack variable weight ``C`` (scalar), use `Cwt=Inf` for hard constraints only.
- `optim=JuMP.Model(OSQP.MathOptInterfaceOSQP.Optimizer)` : quadratic optimizer used in
  the predictive controller, provided as a [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.Model)
  (default to [`OSQP`](https://osqp.org/docs/parsers/jump.html) optimizer).

- additional keyword arguments are passed to [`SteadyKalmanFilter`](@ref) constructor.

# Examples
```jldoctest
julia> model = LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 4);

julia> mpc = LinMPC(model, Mwt=[0, 1], Nwt=[0.5], Hp=30, Hc=1)
LinMPC controller with a sample time Ts = 4.0 s, OSQP optimizer, SteadyKalmanFilter estimator and:
 30 prediction steps Hp
  1 control steps Hc
  1 manipulated inputs u (0 integrating states)
  4 estimated states x̂
  2 measured outputs ym (2 integrating states)
  0 unmeasured outputs yu
  0 measured disturbances d
```

# Extended Help
!!! details "Extended Help"
    Manipulated inputs setpoints ``\mathbf{r_u}`` are not common but they can be interesting
    for over-actuated systems, when `nu > ny` (e.g. prioritize solutions with lower 
    economical costs). The default `Lwt` value implies that this feature is disabled by default.

    The objective function follows this nomenclature:

    | VARIABLE             | DESCRIPTION                                              | SIZE             |
    | :------------------- | :------------------------------------------------------- | :--------------- |
    | ``H_p``              | prediction horizon (integer)                             | `()`             |
    | ``H_c``              | control horizon (integer)                                | `()`             |
    | ``\mathbf{ΔU}``      | manipulated input increments over ``H_c``                | `(nu*Hc,)`       |
    | ``\mathbf{Ŷ}``       | predicted outputs over ``H_p``                           | `(ny*Hp,)`       |
    | ``\mathbf{U}``       | manipulated inputs over ``H_p``                          | `(nu*Hp,)`       |
    | ``\mathbf{R̂_y}``     | predicted output setpoints over ``H_p``                  | `(ny*Hp,)`       |
    | ``\mathbf{R̂_u}``     | predicted manipulated input setpoints over ``H_p``       | `(nu*Hp,)`       |
    | ``\mathbf{M}_{H_p}`` | output setpoint tracking weights over ``H_p``            | `(ny*Hp, ny*Hp)` |
    | ``\mathbf{N}_{H_c}`` | manipulated input increment weights over ``H_c``         | `(nu*Hc, nu*Hc)` |
    | ``\mathbf{L}_{H_p}`` | manipulated input setpoint tracking weights over ``H_p`` | `(nu*Hp, nu*Hp)` |
    | ``C``                | slack variable weight                                    | `()`             |
    | ``ϵ``                | slack variable for constraint softening                  | `()`             |
"""
function LinMPC(
    model::LinModel;
    Hp::Int = default_Hp(model),
    Hc::Int = DEFAULT_HC,
    Mwt  = fill(DEFAULT_MWT, model.ny),
    Nwt  = fill(DEFAULT_NWT, model.nu),
    Lwt  = fill(DEFAULT_LWT, model.nu),
    M_Hp = Diagonal(repeat(Mwt, Hp)),
    N_Hc = Diagonal(repeat(Nwt, Hc)),
    L_Hp = Diagonal(repeat(Lwt, Hp)),
    Cwt = DEFAULT_CWT,
    optim::JuMP.GenericModel = JuMP.Model(DEFAULT_LINMPC_OPTIMIZER, add_bridges=false),
    kwargs...
)
    estim = SteadyKalmanFilter(model; kwargs...)
    return LinMPC(estim; Hp, Hc, Mwt, Nwt, Lwt, Cwt, M_Hp, N_Hc, L_Hp, optim)
end


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
  1 manipulated inputs u (0 integrating states)
  3 estimated states x̂
  1 measured outputs ym (1 integrating states)
  1 unmeasured outputs yu
  0 measured disturbances d
```
"""
function LinMPC(
    estim::SE;
    Hp::Int = default_Hp(estim.model),
    Hc::Int = DEFAULT_HC,
    Mwt  = fill(DEFAULT_MWT, estim.model.ny),
    Nwt  = fill(DEFAULT_NWT, estim.model.nu),
    Lwt  = fill(DEFAULT_LWT, estim.model.nu),
    M_Hp = Diagonal(repeat(Mwt, Hp)),
    N_Hc = Diagonal(repeat(Nwt, Hc)),
    L_Hp = Diagonal(repeat(Lwt, Hp)),
    Cwt  = DEFAULT_CWT,
    optim::JM = JuMP.Model(DEFAULT_LINMPC_OPTIMIZER, add_bridges=false),
) where {NT<:Real, SE<:StateEstimator{NT}, JM<:JuMP.GenericModel}
    isa(estim.model, LinModel) || error("estim.model type must be a LinModel") 
    nk = estimate_delays(estim.model)
    if Hp ≤ nk
        @warn("prediction horizon Hp ($Hp) ≤ estimated number of delays in model "*
              "($nk), the closed-loop system may be unstable or zero-gain (unresponsive)")
    end
    return LinMPC{NT, SE, JM}(estim, Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt, optim)
end

"""
    init_optimization!(mpc::LinMPC, optim::JuMP.GenericModel)

Init the quadratic optimization for [`LinMPC`](@ref) controllers.
"""
function init_optimization!(mpc::LinMPC, optim::JuMP.GenericModel)
    # --- variables and linear constraints ---
    con = mpc.con
    nΔŨ = length(mpc.ΔŨ)
    set_silent(optim)
    limit_solve_time(mpc.optim, mpc.estim.model.Ts)
    @variable(optim, ΔŨvar[1:nΔŨ])
    A = con.A[con.i_b, :]
    b = con.b[con.i_b]
    @constraint(optim, linconstraint, A*ΔŨvar .≤ b)
    # --- quadratic optimization init ---
    @objective(mpc.optim, Min, obj_quadprog(ΔŨvar, mpc.H̃, mpc.q̃))
    return nothing
end

"For [`LinMPC`](@ref), set the QP linear coefficient `q̃` just before optimization."
function set_objective_linear_coef!(mpc::LinMPC, ΔŨvar)
    set_objective_coefficient.(mpc.optim, ΔŨvar, mpc.q̃)
    return nothing
end
