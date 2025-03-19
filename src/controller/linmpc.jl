const DEFAULT_LINMPC_OPTIMIZER = OSQP.MathOptInterfaceOSQP.Optimizer
const DEFAULT_LINMPC_TRANSCRIPTION = SingleShooting()

struct LinMPC{
    NT<:Real, 
    SE<:StateEstimator, 
    TM<:TranscriptionMethod,
    JM<:JuMP.GenericModel
} <: PredictiveController{NT}
    estim::SE
    transcription::TM
    # note: `NT` and the number type `JNT` in `JuMP.GenericModel{JNT}` can be
    # different since solvers that support non-Float64 are scarce.
    optim::JM
    con::ControllerConstraint{NT, Nothing}
    Z̃::Vector{NT}
    ŷ::Vector{NT}
    Hp::Int
    Hc::Int
    nϵ::Int
    weights::ControllerWeights{NT}
    R̂u::Vector{NT}
    R̂y::Vector{NT}
    P̃Δu::Matrix{NT}
    P̃u ::Matrix{NT} 
    Tu ::Matrix{NT}
    Tu_lastu0::Vector{NT}
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
    D̂e::Vector{NT}
    Uop::Vector{NT}
    Yop::Vector{NT}
    Dop::Vector{NT}
    buffer::PredictiveControllerBuffer{NT}
    function LinMPC{NT}(
        estim::SE, Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt, 
        transcription::TM, optim::JM
    ) where {NT<:Real, SE<:StateEstimator, TM<:TranscriptionMethod, JM<:JuMP.GenericModel}
        model = estim.model
        nu, ny, nd, nx̂ = model.nu, model.ny, model.nd, estim.nx̂
        ŷ = copy(model.yop) # dummy vals (updated just before optimization)
        weights = ControllerWeights{NT}(model, Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt)
        # dummy vals (updated just before optimization):
        R̂y, R̂u, Tu_lastu0 = zeros(NT, ny*Hp), zeros(NT, nu*Hp), zeros(NT, nu*Hp)
        PΔu = init_ZtoΔU(estim, transcription, Hp, Hc)
        Pu, Tu = init_ZtoU(estim, transcription, Hp, Hc)
        E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂ = init_predmat(
            model, estim, transcription, Hp, Hc
        )
        Eŝ, Gŝ, Jŝ, Kŝ, Vŝ, Bŝ = init_defectmat(model, estim, transcription, Hp, Hc)
        # dummy vals (updated just before optimization):
        F, fx̂, Fŝ  = zeros(NT, ny*Hp), zeros(NT, nx̂), zeros(NT, nx̂*Hp)
        con, nϵ, P̃Δu, P̃u, Ẽ, Ẽŝ = init_defaultcon_mpc(
            estim, transcription,
            Hp, Hc, Cwt, PΔu, Pu, E, 
            ex̂, fx̂, gx̂, jx̂, kx̂, vx̂, bx̂, 
            Eŝ, Fŝ, Gŝ, Jŝ, Kŝ, Vŝ, Bŝ
        )
        H̃ = init_quadprog(model, weights, Ẽ, P̃Δu, P̃u)
        # dummy vals (updated just before optimization):
        q̃, r = zeros(NT, size(H̃, 1)), zeros(NT, 1)
        Ks, Ps = init_stochpred(estim, Hp)
        # dummy vals (updated just before optimization):
        d0, D̂0, D̂e = zeros(NT, nd), zeros(NT, nd*Hp), zeros(NT, nd + nd*Hp)
        Uop, Yop, Dop = repeat(model.uop, Hp), repeat(model.yop, Hp), repeat(model.dop, Hp)
        nZ̃ = get_nZ(estim, transcription, Hp, Hc) + nϵ
        Z̃ = zeros(NT, nZ̃)
        buffer = PredictiveControllerBuffer(estim, transcription, Hp, Hc, nϵ)
        mpc = new{NT, SE, TM, JM}(
            estim, transcription, optim, con,
            Z̃, ŷ,
            Hp, Hc, nϵ,
            weights,
            R̂u, R̂y,
            P̃Δu, P̃u, Tu, Tu_lastu0,
            Ẽ, F, G, J, K, V, B, 
            H̃, q̃, r,
            Ks, Ps,
            d0, D̂0, D̂e,
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
\min_{\mathbf{Z}, ϵ}    \mathbf{(R̂_y - Ŷ)}' \mathbf{M}_{H_p} \mathbf{(R̂_y - Ŷ)}
                      + \mathbf{(ΔU)}'      \mathbf{N}_{H_c} \mathbf{(ΔU)}        \\
                      + \mathbf{(R̂_u - U)}' \mathbf{L}_{H_p} \mathbf{(R̂_u - U)} 
                      + C ϵ^2
\end{aligned}
```
subject to [`setconstraint!`](@ref) bounds, and in which the weight matrices are repeated 
``H_p`` or ``H_c`` times by default:
```math
\begin{aligned}
    \mathbf{M}_{H_p} &= \text{diag}\mathbf{(M,M,...,M)}     \\
    \mathbf{N}_{H_c} &= \text{diag}\mathbf{(N,N,...,N)}     \\
    \mathbf{L}_{H_p} &= \text{diag}\mathbf{(L,L,...,L)}     
\end{aligned}
```
Time-varying and non-diagonal weights are also supported. Modify the last block in 
``\mathbf{M}_{H_p}`` to specify a terminal weight. The content of the decision vector
``\mathbf{Z}`` depends on the chosen [`TranscriptionMethod`](@ref) (default to
[`SingleShooting`](@ref), hence ``\mathbf{Z = ΔU}``). The ``\mathbf{ΔU}`` includes the input
increments ``\mathbf{Δu}(k+j) = \mathbf{u}(k+j) - \mathbf{u}(k+j-1)`` from ``j=0`` to
``H_c-1``, the ``\mathbf{Ŷ}`` vector, the output predictions ``\mathbf{ŷ}(k+j)`` from
``j=1`` to ``H_p``, and the ``\mathbf{U}`` vector, the manipulated inputs ``\mathbf{u}(k+j)``
from ``j=0`` to ``H_p-1``. The slack variable ``ϵ`` relaxes the constraints, as described
in [`setconstraint!`](@ref) documentation. See Extended Help for a detailed nomenclature. 

This method uses the default state estimator, a [`SteadyKalmanFilter`](@ref) with default
arguments. This controller allocates memory at each time step for the optimization.

# Arguments
- `model::LinModel` : model used for controller predictions and state estimations.
- `Hp=10+nk` : prediction horizon ``H_p``, `nk` is the number of delays in `model`.
- `Hc=2` : control horizon ``H_c``.
- `Mwt=fill(1.0,model.ny)` : main diagonal of ``\mathbf{M}`` weight matrix (vector).
- `Nwt=fill(0.1,model.nu)` : main diagonal of ``\mathbf{N}`` weight matrix (vector).
- `Lwt=fill(0.0,model.nu)` : main diagonal of ``\mathbf{L}`` weight matrix (vector).
- `M_Hp=diagm(repeat(Mwt,Hp))` : positive semidefinite symmetric matrix ``\mathbf{M}_{H_p}``.
- `N_Hc=diagm(repeat(Nwt,Hc))` : positive semidefinite symmetric matrix ``\mathbf{N}_{H_c}``.
- `L_Hp=diagm(repeat(Lwt,Hp))` : positive semidefinite symmetric matrix ``\mathbf{L}_{H_p}``.
- `Cwt=1e5` : slack variable weight ``C`` (scalar), use `Cwt=Inf` for hard constraints only.
- `transcription=SingleShooting()` : a [`TranscriptionMethod`](@ref) for the optimization.
- `optim=JuMP.Model(OSQP.MathOptInterfaceOSQP.Optimizer)` : quadratic optimizer used in
  the predictive controller, provided as a [`JuMP.Model`](@extref) object (default to 
  [`OSQP`](https://osqp.org/docs/parsers/jump.html) optimizer).
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
    | ``\mathbf{Z}``       | decision variable vector (excluding ``ϵ``)               |  var.            |
    | ``\mathbf{ΔU}``      | manipulated input increments over ``H_c``                | `(nu*Hc,)`       |
    | ``\mathbf{D̂}``       | predicted measured disturbances over ``H_p``             | `(nd*Hp,)`       |
    | ``\mathbf{Ŷ}``       | predicted outputs over ``H_p``                           | `(ny*Hp,)`       |
    | ``\mathbf{X̂}``       | predicted states over ``H_p``                            | `(nx̂*Hp,)`       |
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
    transcription::TranscriptionMethod = DEFAULT_LINMPC_TRANSCRIPTION,
    optim::JuMP.GenericModel = JuMP.Model(DEFAULT_LINMPC_OPTIMIZER, add_bridges=false),
    kwargs...
)
    estim = SteadyKalmanFilter(model; kwargs...)
    return LinMPC(estim; Hp, Hc, Mwt, Nwt, Lwt, Cwt, M_Hp, N_Hc, L_Hp, transcription, optim)
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
    transcription::TranscriptionMethod = DEFAULT_LINMPC_TRANSCRIPTION,
    optim::JM = JuMP.Model(DEFAULT_LINMPC_OPTIMIZER, add_bridges=false),
) where {NT<:Real, SE<:StateEstimator{NT}, JM<:JuMP.GenericModel}
    isa(estim.model, LinModel) || error("estim.model type must be a LinModel") 
    nk = estimate_delays(estim.model)
    if Hp ≤ nk
        @warn("prediction horizon Hp ($Hp) ≤ estimated number of delays in model "*
              "($nk), the closed-loop system may be unstable or zero-gain (unresponsive)")
    end
    return LinMPC{NT}(estim, Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt, transcription, optim)
end

"""
    init_optimization!(mpc::LinMPC, model::LinModel, optim::JuMP.GenericModel) -> nothing

Init the quadratic optimization for [`LinMPC`](@ref) controllers.
"""
function init_optimization!(mpc::LinMPC, model::LinModel, optim::JuMP.GenericModel)
    # --- variables and linear constraints ---
    con = mpc.con
    nZ̃ = length(mpc.Z̃)
    JuMP.num_variables(optim) == 0 || JuMP.empty!(optim)
    JuMP.set_silent(optim)
    limit_solve_time(mpc.optim, model.Ts)
    @variable(optim, Z̃var[1:nZ̃])
    A = con.A[con.i_b, :]
    b = con.b[con.i_b]
    @constraint(optim, linconstraint, A*Z̃var .≤ b)
    Aeq = con.Aeq
    beq = con.beq
    @constraint(optim, linconstrainteq, Aeq*Z̃var .== beq)
    set_objective_hessian!(mpc, Z̃var)
    return nothing
end

"For [`LinMPC`](@ref), set the QP linear coefficient `q̃` just before optimization."
function set_objective_linear_coef!(mpc::LinMPC, Z̃var)
    JuMP.set_objective_coefficient(mpc.optim, Z̃var, mpc.q̃)
    return nothing
end

"Update the quadratic objective function for [`LinMPC`](@ref) controllers."
function set_objective_hessian!(mpc::LinMPC, Z̃var)
    @objective(mpc.optim, Min, obj_quadprog(Z̃var, mpc.H̃, mpc.q̃))
    return nothing
end