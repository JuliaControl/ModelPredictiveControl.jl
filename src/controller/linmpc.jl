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
    nϵ::Int
    M_Hp::Hermitian{NT, Matrix{NT}}
    Ñ_Hc::Hermitian{NT, Matrix{NT}}
    L_Hp::Hermitian{NT, Matrix{NT}}
    E::NT
    R̂u0::Vector{NT}
    R̂y0::Vector{NT}
    noR̂u::Bool
    S̃::Matrix{NT} 
    T::Matrix{NT}
    T_lastu0::Vector{NT}
    Ẽ::Matrix{NT}
    F::Vector{NT}
    G::Matrix{NT}
    J::Matrix{NT}
    K::Matrix{NT}
    V::Matrix{NT}
    B::Vector{NT}
    H̃::Hermitian{NT, Matrix{NT}}
    q̃::Vector{NT}
    r::Vector{NT}
    Ks::Matrix{NT}
    Ps::Matrix{NT}
    d0::Vector{NT}
    D̂0::Vector{NT}
    D̂E::Vector{NT}
    Uop::Vector{NT}
    Yop::Vector{NT}
    Dop::Vector{NT}
    buffer::PredictiveControllerBuffer{NT}
    function LinMPC{NT, SE, JM}(
        estim::SE, Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt, optim::JM
    ) where {NT<:Real, SE<:StateEstimator, JM<:JuMP.GenericModel}
        model = estim.model
        nu, ny, nd, nx̂ = model.nu, model.ny, model.nd, estim.nx̂
        ŷ = copy(model.yop) # dummy vals (updated just before optimization)
        Ewt = 0   # economic costs not supported for LinMPC
        validate_weights(model, Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt)
        # convert `Diagonal` to normal `Matrix` if required:
        M_Hp = Hermitian(convert(Matrix{NT}, M_Hp), :L) 
        N_Hc = Hermitian(convert(Matrix{NT}, N_Hc), :L)
        L_Hp = Hermitian(convert(Matrix{NT}, L_Hp), :L)
        # dummy vals (updated just before optimization):
        R̂y0, R̂u0, T_lastu0 = zeros(NT, ny*Hp), zeros(NT, nu*Hp), zeros(NT, nu*Hp)
        noR̂u = iszero(L_Hp)
        S, T = init_ΔUtoU(model, Hp, Hc)
        E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂ = init_predmat(estim, model, Hp, Hc)
        # dummy vals (updated just before optimization):
        F, fx̂  = zeros(NT, ny*Hp), zeros(NT, nx̂)
        con, nϵ, S̃, Ñ_Hc, Ẽ = init_defaultcon_mpc(
            estim, Hp, Hc, Cwt, S, N_Hc, E, ex̂, fx̂, gx̂, jx̂, kx̂, vx̂, bx̂
        )
        H̃ = init_quadprog(model, Ẽ, S̃, M_Hp, Ñ_Hc, L_Hp)
        # dummy vals (updated just before optimization):
        q̃, r = zeros(NT, size(H̃, 1)), zeros(NT, 1)
        Ks, Ps = init_stochpred(estim, Hp)
        # dummy vals (updated just before optimization):
        d0, D̂0, D̂E = zeros(NT, nd), zeros(NT, nd*Hp), zeros(NT, nd + nd*Hp)
        Uop, Yop, Dop = repeat(model.uop, Hp), repeat(model.yop, Hp), repeat(model.dop, Hp)
        nΔŨ = size(Ẽ, 2)
        ΔŨ = zeros(NT, nΔŨ)
        buffer = PredictiveControllerBuffer{NT}(nu, ny, nd, Hp)
        mpc = new{NT, SE, JM}(
            estim, optim, con,
            ΔŨ, ŷ,
            Hp, Hc, nϵ,
            M_Hp, Ñ_Hc, L_Hp, Ewt, 
            R̂u0, R̂y0, noR̂u,
            S̃, T, T_lastu0,
            Ẽ, F, G, J, K, V, B, 
            H̃, q̃, r,
            Ks, Ps,
            d0, D̂0, D̂E,
            Uop, Yop, Dop,
            buffer
        )
        init_optimization!(mpc, model, optim)
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
Time-varying and non-diagonal weights are also supported. Modify the last block in 
``\mathbf{M}_{H_p}`` to specify a terminal weight. The ``\mathbf{ΔU}`` includes the input 
increments ``\mathbf{Δu}(k+j) = \mathbf{u}(k+j) - \mathbf{u}(k+j-1)`` from ``j=0`` to
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
- `M_Hp=diagm(repeat(Mwt,Hp))` : positive semidefinite symmetric matrix ``\mathbf{M}_{H_p}``.
- `N_Hc=diagm(repeat(Nwt,Hc))` : positive semidefinite symmetric matrix ``\mathbf{N}_{H_c}``.
- `L_Hp=diagm(repeat(Lwt,Hp))` : positive semidefinite symmetric matrix ``\mathbf{L}_{H_p}``.
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
  1 slack variable ϵ (control constraints)
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
    M_Hp = diagm(repeat(Mwt, Hp)),
    N_Hc = diagm(repeat(Nwt, Hc)),
    L_Hp = diagm(repeat(Lwt, Hp)),
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
  1 slack variable ϵ (control constraints)
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
    M_Hp = diagm(repeat(Mwt, Hp)),
    N_Hc = diagm(repeat(Nwt, Hc)),
    L_Hp = diagm(repeat(Lwt, Hp)),
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
    init_optimization!(mpc::LinMPC, model::LinModel, optim)

Init the quadratic optimization for [`LinMPC`](@ref) controllers.
"""
function init_optimization!(mpc::LinMPC, model::LinModel, optim)
    # --- variables and linear constraints ---
    con = mpc.con
    nΔŨ = length(mpc.ΔŨ)
    JuMP.num_variables(optim) == 0 || JuMP.empty!(optim)
    JuMP.set_silent(optim)
    limit_solve_time(mpc.optim, model.Ts)
    @variable(optim, ΔŨvar[1:nΔŨ])
    A = con.A[con.i_b, :]
    b = con.b[con.i_b]
    @constraint(optim, linconstraint, A*ΔŨvar .≤ b)
    set_objective_hessian!(mpc, ΔŨvar)
    return nothing
end

"For [`LinMPC`](@ref), set the QP linear coefficient `q̃` just before optimization."
function set_objective_linear_coef!(mpc::LinMPC, ΔŨvar)
    JuMP.set_objective_coefficient(mpc.optim, ΔŨvar, mpc.q̃)
    return nothing
end

"Update the quadratic objective function for [`LinMPC`](@ref) controllers."
function set_objective_hessian!(mpc::LinMPC, ΔŨvar)
    @objective(mpc.optim, Min, obj_quadprog(ΔŨvar, mpc.H̃, mpc.q̃))
    return nothing
end