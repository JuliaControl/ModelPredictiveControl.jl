const DEFAULT_LINMHE_OPTIMIZER    = OSQP.MathOptInterfaceOSQP.Optimizer
const DEFAULT_NONLINMHE_OPTIMIZER = optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes")

@doc raw"""
Include all the data for the constraints of [`MovingHorizonEstimator`](@ref).

The bounds on the estimated state at arrival ``\mathbf{x̂}_k(k-N_k+1)`` is separated from
the other state constraints ``\mathbf{x̂}_k(k-N_k+2), \mathbf{x̂}_k(k-N_k+3), ...`` since
the former is always a linear inequality constraint (it's a decision variable). The fields
`x̃min` and `x̃max` refer to the bounds at the arrival (augmented with the slack variable
ϵ), and `X̂min` and `X̂max`, the others.
"""
struct EstimatorConstraint{NT<:Real}
    Ẽx̂      ::Matrix{NT}
    Fx̂      ::Vector{NT}
    Gx̂      ::Matrix{NT}
    Jx̂      ::Matrix{NT}
    Bx̂      ::Vector{NT}
    x̃0min   ::Vector{NT}
    x̃0max   ::Vector{NT}
    X̂0min   ::Vector{NT}
    X̂0max   ::Vector{NT}
    Ŵmin    ::Vector{NT}
    Ŵmax    ::Vector{NT}
    V̂min    ::Vector{NT}
    V̂max    ::Vector{NT}
    A_x̃min  ::Matrix{NT}
    A_x̃max  ::Matrix{NT}
    A_X̂min  ::Matrix{NT}
    A_X̂max  ::Matrix{NT}
    A_Ŵmin  ::Matrix{NT}
    A_Ŵmax  ::Matrix{NT}
    A_V̂min  ::Matrix{NT}
    A_V̂max  ::Matrix{NT}
    A       ::Matrix{NT}
    b       ::Vector{NT}
    C_x̂min  ::Vector{NT}
    C_x̂max  ::Vector{NT}
    C_v̂min  ::Vector{NT}
    C_v̂max  ::Vector{NT}
    i_b     ::BitVector
    i_g     ::BitVector
end

struct MovingHorizonEstimator{
    NT<:Real, 
    SM<:SimModel,
    JM<:JuMP.GenericModel,
    CE<:StateEstimator,
} <: StateEstimator{NT}
    model::SM
    # note: `NT` and the number type `JNT` in `JuMP.GenericModel{JNT}` can be
    # different since solvers that support non-Float64 are scarce.
    optim::JM
    con::EstimatorConstraint{NT}
    covestim::CE
    Z̃::Vector{NT}
    lastu0::Vector{NT}
    x̂op::Vector{NT}
    f̂op::Vector{NT}
    x̂0 ::Vector{NT}
    He::Int
    nϵ::Int
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
    Â   ::Matrix{NT}
    B̂u  ::Matrix{NT}
    Ĉ   ::Matrix{NT}
    B̂d  ::Matrix{NT}
    D̂d  ::Matrix{NT}
    Ĉm  ::Matrix{NT}
    D̂dm ::Matrix{NT}
    Ẽ ::Matrix{NT}
    F ::Vector{NT}
    G ::Matrix{NT}
    J ::Matrix{NT}
    B ::Vector{NT}
    ẽx̄::Matrix{NT}
    fx̄::Vector{NT}
    H̃::Hermitian{NT, Matrix{NT}}
    q̃::Vector{NT}
    r::Vector{NT}
    P̂_0::Hermitian{NT, Matrix{NT}}
    Q̂::Hermitian{NT, Matrix{NT}}
    R̂::Hermitian{NT, Matrix{NT}}
    invP̄::Hermitian{NT, Matrix{NT}}
    invQ̂_He::Hermitian{NT, Matrix{NT}}
    invR̂_He::Hermitian{NT, Matrix{NT}}
    C::NT
    X̂op::Vector{NT}
    X̂0 ::Vector{NT}
    Y0m::Vector{NT}
    U0 ::Vector{NT}
    D0 ::Vector{NT}
    Ŵ  ::Vector{NT}
    x̂0arr_old::Vector{NT}
    P̂arr_old ::Hermitian{NT, Matrix{NT}}
    Nk::Vector{Int}
    direct::Bool
    corrected::Vector{Bool}
    buffer::StateEstimatorBuffer{NT}
    function MovingHorizonEstimator{NT}(
        model::SM, He, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂, Cwt, optim::JM, covestim::CE;
        direct=true
    ) where {NT<:Real, SM<:SimModel{NT}, JM<:JuMP.GenericModel, CE<:StateEstimator{NT}}
        nu, ny, nd = model.nu, model.ny, model.nd
        He < 1  && throw(ArgumentError("Estimation horizon He should be ≥ 1"))
        Cwt < 0 && throw(ArgumentError("Cwt weight should be ≥ 0"))
        nym, nyu = validate_ym(model, i_ym)
        As, Cs_u, Cs_y, nint_u, nint_ym = init_estimstoch(model, i_ym, nint_u, nint_ym)
        nxs = size(As, 1)
        nx̂  = model.nx + nxs
        Â, B̂u, Ĉ, B̂d, D̂d, x̂op, f̂op = augment_model(model, As, Cs_u, Cs_y)
        Ĉm, D̂dm = Ĉ[i_ym, :], D̂d[i_ym, :]
        validate_kfcov(nym, nx̂, Q̂, R̂, P̂_0)
        lastu0 = zeros(NT, nu)
        x̂0 = [zeros(NT, model.nx); zeros(NT, nxs)]
        r = direct ? 0 : 1
        E, G, J, B, ex̄, Ex̂, Gx̂, Jx̂, Bx̂ = init_predmat_mhe(
            model, He, i_ym, Â, B̂u, Ĉm, B̂d, D̂dm, x̂op, f̂op, r
        )
        # dummy values (updated just before optimization):
        F, fx̄, Fx̂ = zeros(NT, nym*He), zeros(NT, nx̂), zeros(NT, nx̂*He)
        con, nϵ, Ẽ, ẽx̄ = init_defaultcon_mhe(
            model, He, Cwt, nx̂, nym, E, ex̄, Ex̂, Fx̂, Gx̂, Jx̂, Bx̂
        )
        nZ̃ = size(Ẽ, 2)
        # dummy values, updated before optimization:
        H̃, q̃, r = Hermitian(zeros(NT, nZ̃, nZ̃), :L), zeros(NT, nZ̃), zeros(NT, 1)
        Z̃ = zeros(NT, nZ̃)
        X̂op = repeat(x̂op, He)
        X̂0, Y0m = zeros(NT, nx̂*He), zeros(NT, nym*He)
        nD0 = direct ? nd*(He+1) : nd*He
        U0, D0  = zeros(NT, nu*He), zeros(NT, nD0) 
        Ŵ = zeros(NT, nx̂*He)
        buffer = StateEstimatorBuffer{NT}(nu, nx̂, nym, ny, nd)
        P̂_0 = Hermitian(P̂_0, :L)
        Q̂, R̂ = Hermitian(Q̂, :L),  Hermitian(R̂, :L)
        P̂_0 = Hermitian(P̂_0, :L)
        invP̄ = inv_cholesky!(buffer.P̂, P̂_0)
        invQ̂ = inv_cholesky!(buffer.Q̂, Q̂)
        invR̂ = inv_cholesky!(buffer.R̂, R̂)
        invQ̂_He = Hermitian(repeatdiag(invQ̂, He), :L)
        invR̂_He = Hermitian(repeatdiag(invR̂, He), :L)
        x̂0arr_old = zeros(NT, nx̂)
        P̂arr_old = copy(P̂_0)
        Nk = [0]
        corrected = [false]
        estim = new{NT, SM, JM, CE}(
            model, optim, con, covestim,  
            Z̃, lastu0, x̂op, f̂op, x̂0, 
            He, nϵ,
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs_u, Cs_y, nint_u, nint_ym,
            Â, B̂u, Ĉ, B̂d, D̂d, Ĉm, D̂dm,
            Ẽ, F, G, J, B, ẽx̄, fx̄,
            H̃, q̃, r,
            P̂_0, Q̂, R̂, invP̄, invQ̂_He, invR̂_He, Cwt,
            X̂op, X̂0, Y0m, U0, D0, Ŵ, 
            x̂0arr_old, P̂arr_old, Nk,
            direct, corrected,
            buffer
        )
        init_optimization!(estim, model, optim)
        return estim
    end
end

@doc raw"""
    MovingHorizonEstimator(model::SimModel; <keyword arguments>)

Construct a moving horizon estimator (MHE) based on `model` ([`LinModel`](@ref) or [`NonLinModel`](@ref)).

It can handle constraints on the estimates, see [`setconstraint!`](@ref). Additionally, 
`model` is not linearized like the [`ExtendedKalmanFilter`](@ref), and the probability 
distribution is not approximated like the [`UnscentedKalmanFilter`](@ref). The computational
costs are drastically higher, however, since it minimizes the following objective function
at each discrete time ``k``:
```math
\min_{\mathbf{x̂}_k(k-N_k+p), \mathbf{Ŵ}, ϵ}   \mathbf{x̄}' \mathbf{P̄}^{-1}       \mathbf{x̄} 
                                            + \mathbf{Ŵ}' \mathbf{Q̂}_{N_k}^{-1} \mathbf{Ŵ}  
                                            + \mathbf{V̂}' \mathbf{R̂}_{N_k}^{-1} \mathbf{V̂}
                                            + C ϵ^2
```
in which the arrival costs are evaluated from the states estimated at time ``k-N_k``:
```math
\begin{aligned}
    \mathbf{x̄} &= \mathbf{x̂}_{k-N_k}(k-N_k+p) - \mathbf{x̂}_k(k-N_k+p) \\
    \mathbf{P̄} &= \mathbf{P̂}_{k-N_k}(k-N_k+p)
\end{aligned}
```
and the covariances are repeated ``N_k`` times:
```math
\begin{aligned}
    \mathbf{Q̂}_{N_k} &= \text{diag}\mathbf{(Q̂,Q̂,...,Q̂)}  \\
    \mathbf{R̂}_{N_k} &= \text{diag}\mathbf{(R̂,R̂,...,R̂)} 
\end{aligned}
```
The estimation horizon ``H_e`` limits the window length:
```math
N_k =                     \begin{cases}
    k + 1   &  k < H_e    \\
    H_e     &  k ≥ H_e    \end{cases}
```
The vectors ``\mathbf{Ŵ}`` and ``\mathbf{V̂}`` respectively encompass the estimated process
noises ``\mathbf{ŵ}(k-j+p)`` from ``j=N_k`` to ``1`` and sensor noises ``\mathbf{v̂}(k-j+1)``
from ``j=N_k`` to ``1``. The Extended Help defines the two vectors, the slack variable
``ϵ``, and the estimation of the covariance at arrival ``\mathbf{P̂}_{k-N_k}(k-N_k+p)``. If
the keyword argument `direct=true` (default value), the constant ``p=0`` in the equations
above, and the MHE is in the current form. Else ``p=1``, leading to the prediction form.

See [`UnscentedKalmanFilter`](@ref) for details on the augmented process model and 
``\mathbf{R̂}, \mathbf{Q̂}`` covariances. This estimator allocates a fair amount of memory 
at each time step for the optimization, which is hard-coded as a single shooting
transcription for now.

!!! warning
    See the Extended Help if you get an error like:    
    `MethodError: no method matching (::var"##")(::Vector{ForwardDiff.Dual})`.

# Arguments
!!! info
    Keyword arguments with *`emphasis`* are non-Unicode alternatives.

- `model::SimModel` : (deterministic) model for the estimations.
- `He=nothing` : estimation horizon ``H_e``, must be specified.
- `i_ym=1:model.ny` : `model` output indices that are measured ``\mathbf{y^m}``, the rest 
    are unmeasured ``\mathbf{y^u}``.
- `σP_0=fill(1/model.nx,model.nx)` or *`sigmaP_0`* : main diagonal of the initial estimate
    covariance ``\mathbf{P}(0)``, specified as a standard deviation vector.
- `σQ=fill(1/model.nx,model.nx)` or *`sigmaQ`* : main diagonal of the process noise
    covariance ``\mathbf{Q}`` of `model`, specified as a standard deviation vector.
- `σR=fill(1,length(i_ym))` or *`sigmaR`* : main diagonal of the sensor noise covariance
    ``\mathbf{R}`` of `model` measured outputs, specified as a standard deviation vector.
- `nint_u=0`: integrator quantity for the stochastic model of the unmeasured disturbances at
    the manipulated inputs (vector), use `nint_u=0` for no integrator (see Extended Help).
- `nint_ym=default_nint(model,i_ym,nint_u)` : same than `nint_u` but for the unmeasured 
    disturbances at the measured outputs, use `nint_ym=0` for no integrator (see Extended Help).
- `σQint_u=fill(1,sum(nint_u))` or *`sigmaQint_u`* : same than `σQ` but for the unmeasured
    disturbances at manipulated inputs ``\mathbf{Q_{int_u}}`` (composed of integrators).
- `σPint_u_0=fill(1,sum(nint_u))` or *`sigmaPint_u_0`* : same than `σP_0` but for the unmeasured
    disturbances at manipulated inputs ``\mathbf{P_{int_u}}(0)`` (composed of integrators).
- `σQint_ym=fill(1,sum(nint_ym))` or *`sigmaQint_u`* : same than `σQ` for the unmeasured
    disturbances at measured outputs ``\mathbf{Q_{int_{ym}}}`` (composed of integrators).
- `σPint_ym_0=fill(1,sum(nint_ym))` or *`sigmaPint_ym_0`* : same than `σP_0` but for the unmeasured
    disturbances at measured outputs ``\mathbf{P_{int_{ym}}}(0)`` (composed of integrators).
- `Cwt=Inf` : slack variable weight ``C``, default to `Inf` meaning hard constraints only.
- `optim=default_optim_mhe(model)` : a [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.Model)
   with a quadratic/nonlinear optimizer for solving (default to [`Ipopt`](https://github.com/jump-dev/Ipopt.jl),
   or [`OSQP`](https://osqp.org/docs/parsers/jump.html) if `model` is a [`LinModel`](@ref)).
- `direct=true`: construct with a direct transmission from ``\mathbf{y^m}`` (a.k.a. current
   estimator, in opposition to the delayed/predictor form).

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_,_)->0.1x+u, (x,_,_)->2x, 10.0, 1, 1, 1, solver=nothing);

julia> estim = MovingHorizonEstimator(model, He=5, σR=[1], σP_0=[0.01])
MovingHorizonEstimator estimator with a sample time Ts = 10.0 s, Ipopt optimizer, NonLinModel and:
 5 estimation steps He
 0 slack variable ϵ (estimation constraints)
 1 manipulated inputs u (0 integrating states)
 2 estimated states x̂
 1 measured outputs ym (1 integrating states)
 0 unmeasured outputs yu
 0 measured disturbances d
```

# Extended Help
!!! details "Extended Help"
    The estimated process and sensor noises are defined as:
    ```math
    \mathbf{Ŵ} = 
    \begin{bmatrix}
        \mathbf{ŵ}(k-N_k+p+0)     \\
        \mathbf{ŵ}(k-N_k+p+1)     \\
        \vdots                  \\
        \mathbf{ŵ}(k+p-1)
    \end{bmatrix} , \quad
    \mathbf{V̂} =
    \begin{bmatrix}
        \mathbf{v̂}(k-N_k+1)     \\
        \mathbf{v̂}(k-N_k+2)     \\
        \vdots                  \\
        \mathbf{v̂}(k)
    \end{bmatrix}
    ```
    based on the augmented model functions ``\mathbf{f̂, ĥ^m}``:
    ```math
    \begin{aligned}
        \mathbf{v̂}(k-j)     &= \mathbf{y^m}(k-j) - \mathbf{ĥ^m}\Big(\mathbf{x̂}_k(k-j), \mathbf{d}(k-j)\Big) \\
        \mathbf{x̂}_k(k-j+1) &= \mathbf{f̂}\Big(\mathbf{x̂}_k(k-j), \mathbf{u}(k-j), \mathbf{d}(k-j)\Big) + \mathbf{ŵ}(k-j)
    \end{aligned}
    ```
    The constant ``p`` equals to `!direct`. In other words, ``\mathbf{Ŵ}`` and ``\mathbf{V̂}``
    are shifted by one time step if `direct==true`. The non-default prediction form
    with ``p=1`` is particularly useful for the MHE since it moves its expensive
    computations after the MPC optimization. That is, [`preparestate!`](@ref) will solve the
    optimization by default, but it can be postponed to [`updatestate!`](@ref) with
    `direct=false`.

    The Extended Help of [`SteadyKalmanFilter`](@ref) details the tuning of the covariances
    and the augmentation with `nint_ym` and `nint_u` arguments. The default augmentation
    scheme is identical, that is `nint_u=0` and `nint_ym` computed by [`default_nint`](@ref).
    Note that the constructor does not validate the observability of the resulting augmented
    [`NonLinModel`](@ref). In such cases, it is the user's responsibility to ensure that it
    is still observable.

    The estimation covariance at arrival ``\mathbf{P̂}_{k-N_k}(k-N_k+p)`` gives an uncertainty
    on the state estimate at the beginning of the window ``k-N_k+p``, that is, in the past.
    It is not the same as the current estimate covariance ``\mathbf{P̂}_k(k)``, a value not
    computed by the MHE (contrarily to e.g. the [`KalmanFilter`](@ref)). Three keyword
    arguments specify its initial value with ``\mathbf{P̂_i} =  \mathrm{diag}\{ \mathbf{P}(0),
    \mathbf{P_{int_{u}}}(0), \mathbf{P_{int_{ym}}}(0) \}``. The initial state estimate
    ``\mathbf{x̂_i}`` can be manually specified with [`setstate!`](@ref), or automatically 
    with [`initstate!`](@ref) for [`LinModel`](@ref). Note the MHE with ``p=0`` is slightly
    inconsistent with all the other estimators here. It interprets the initial values as
    ``\mathbf{x̂_i} = \mathbf{x̂}_{-1}(-1)`` and  ``\mathbf{P̂_i} = \mathbf{P̂}_{-1}(-1)``, an 
    *a posteriori* estimate[^2] from the last time step. The MHE with ``p=1`` is consistent,
    interpreting them as  ``\mathbf{x̂_i} = \mathbf{x̂}_{-1}(0)`` and
    ``\mathbf{P̂_i} = \mathbf{P̂}_{-1}(0)``.

    [^2]: M. Hovd (2012), "A Note On The Smoothing Formulation Of Moving Horizon Estimation",
          *Facta Universitatis*, Vol. 11 №2.

    The optimization and the update of the arrival covariance depend on `model`:

    - If `model` is a [`LinModel`](@ref), the optimization is treated as a quadratic program
      with a time-varying Hessian, which is generally cheaper than nonlinear programming. By
      default, a [`KalmanFilter`](@ref) estimates the arrival covariance (customizable).
    - Else, a nonlinear program with automatic differentiation (AD) solves the optimization.
      Optimizers generally benefit from exact derivatives like AD. However, the `f` and `h` 
      functions must be compatible with this feature. See [Automatic differentiation](https://jump.dev/JuMP.jl/stable/manual/nlp/#Automatic-differentiation)
      for common mistakes when writing these functions. An [`UnscentedKalmanFilter`](@ref)
      estimates the arrival covariance by default.
    
    The slack variable ``ϵ`` relaxes the constraints if enabled, see [`setconstraint!`](@ref). 
    It is disabled by default for the MHE (from `Cwt=Inf`) but it should be activated for
    problems with two or more types of bounds, to ensure feasibility (e.g. on the estimated
    state ``\mathbf{x̂}`` and sensor noise ``\mathbf{v̂}``). Note that if `Cwt≠Inf`, the
    attribute `nlp_scaling_max_gradient` of `Ipopt` is set to  `10/Cwt` (if not already set), 
    to scale the small values of ``ϵ``. Use the second constructor to specify the arrival
    covariance estimation method.
"""
function MovingHorizonEstimator(
    model::SM;
    He::Union{Int, Nothing} = nothing,
    i_ym::IntRangeOrVector = 1:model.ny,
    sigmaP_0 = fill(1/model.nx, model.nx),
    sigmaQ   = fill(1/model.nx, model.nx),
    sigmaR   = fill(1, length(i_ym)),
    nint_u ::IntVectorOrInt = 0,
    nint_ym::IntVectorOrInt = default_nint(model, i_ym, nint_u),
    sigmaPint_u_0  = fill(1, max(sum(nint_u),  0)),
    sigmaQint_u    = fill(1, max(sum(nint_u),  0)),
    sigmaPint_ym_0 = fill(1, max(sum(nint_ym), 0)),
    sigmaQint_ym   = fill(1, max(sum(nint_ym), 0)),
    Cwt::Real = Inf,
    optim::JM = default_optim_mhe(model),
    direct = true,
    σP_0       = sigmaP_0,
    σQ         = sigmaQ,
    σR         = sigmaR,
    σPint_u_0  = sigmaPint_u_0,
    σQint_u    = sigmaQint_u,
    σPint_ym_0 = sigmaPint_ym_0,
    σQint_ym   = sigmaQint_ym,
) where {NT<:Real, SM<:SimModel{NT}, JM<:JuMP.GenericModel}
    # estimated covariances matrices (variance = σ²) :
    P̂_0 = Hermitian(diagm(NT[σP_0; σPint_u_0; σPint_ym_0].^2), :L)
    Q̂  = Hermitian(diagm(NT[σQ;  σQint_u;  σQint_ym ].^2), :L)
    R̂  = Hermitian(diagm(NT[σR;].^2), :L)
    isnothing(He) && throw(ArgumentError("Estimation horizon He must be explicitly specified")) 
    return MovingHorizonEstimator(
        model, He, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂, Cwt; direct, optim
    )
end

default_optim_mhe(::LinModel) = JuMP.Model(DEFAULT_LINMHE_OPTIMIZER, add_bridges=false)
default_optim_mhe(::SimModel) = JuMP.Model(DEFAULT_NONLINMHE_OPTIMIZER, add_bridges=false)

@doc raw"""
    MovingHorizonEstimator(
        model, He, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂, Cwt=Inf;
        optim=default_optim_mhe(model), 
        direct=true,
        covestim=default_covestim_mhe(model, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂; direct)
    )

Construct the estimator from the augmented covariance matrices `P̂_0`, `Q̂` and `R̂`.

This syntax allows nonzero off-diagonal elements in ``\mathbf{P̂_i}, \mathbf{Q̂, R̂}``,
where ``\mathbf{P̂_i}`` is the initial estimation covariance, provided by `P̂_0` argument. The
keyword argument `covestim` also allows specifying a custom [`StateEstimator`](@ref) object
for the estimation of covariance at the arrival ``\mathbf{P̂}_{k-N_k}(k-N_k+p)``. The
supported types are [`KalmanFilter`](@ref), [`UnscentedKalmanFilter`](@ref) and 
[`ExtendedKalmanFilter`](@ref).
"""
function MovingHorizonEstimator(
    model::SM, He, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂, Cwt=Inf;
    optim::JM = default_optim_mhe(model),
    direct = true,
    covestim::CE = default_covestim_mhe(model, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂; direct)
) where {NT<:Real, SM<:SimModel{NT}, JM<:JuMP.GenericModel, CE<:StateEstimator{NT}}
    P̂_0, Q̂, R̂ = to_mat(P̂_0), to_mat(Q̂), to_mat(R̂)
    return MovingHorizonEstimator{NT}(
        model, He, i_ym, nint_u, nint_ym, P̂_0, Q̂ , R̂, Cwt, optim, covestim; direct
    )
end

function default_covestim_mhe(model::LinModel, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂; direct)
    return KalmanFilter(model, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂; direct)
end
function default_covestim_mhe(model::SimModel, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂; direct)
    return UnscentedKalmanFilter(model,  i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂; direct)
end

@doc raw"""
    setconstraint!(estim::MovingHorizonEstimator; <keyword arguments>) -> estim

Set the bound constraint parameters of the [`MovingHorizonEstimator`](@ref) `estim`.
   
It supports both soft and hard constraints on the estimated state ``\mathbf{x̂}``, process 
noise ``\mathbf{ŵ}`` and sensor noise ``\mathbf{v̂}``:
```math 
\begin{alignat*}{3}
    \mathbf{x̂_{min} - c_{x̂_{min}}} ϵ ≤&&\   \mathbf{x̂}_k(k-j+p) &≤ \mathbf{x̂_{max} + c_{x̂_{max}}} ϵ &&\qquad  j = N_k, N_k - 1, ... , 0    \\
    \mathbf{ŵ_{min} - c_{ŵ_{min}}} ϵ ≤&&\     \mathbf{ŵ}(k-j+p) &≤ \mathbf{ŵ_{max} + c_{ŵ_{max}}} ϵ &&\qquad  j = N_k, N_k - 1, ... , 1    \\
    \mathbf{v̂_{min} - c_{v̂_{min}}} ϵ ≤&&\     \mathbf{v̂}(k-j+1) &≤ \mathbf{v̂_{max} + c_{v̂_{max}}} ϵ &&\qquad  j = N_k, N_k - 1, ... , 1
\end{alignat*}
```
and also ``ϵ ≥ 0``. All the constraint parameters are vector. Use `±Inf` values when there
is no bound. The constraint softness parameters ``\mathbf{c}``, also called equal concern
for relaxation, are non-negative values that specify the softness of the associated bound.
Use `0.0` values for hard constraints (default for all of them). Notice that constraining
the estimated sensor noises is equivalent to bounding the innovation term, since 
``\mathbf{v̂}(k) = \mathbf{y^m}(k) - \mathbf{ŷ^m}(k)``. See Extended Help for details on
the constant ``p``, on model augmentation and on time-varying constraints.

# Arguments
!!! info
    All the keyword arguments have non-Unicode alternatives e.g. *`xhatmin`* or *`Vhatmax`*. 

    The default constraints are mentioned here for clarity but omitting a keyword argument 
    will not re-assign to its default value (defaults are set at construction only).

- `estim::MovingHorizonEstimator` : moving horizon estimator to set constraints
- `x̂min=fill(-Inf,nx̂)` / `x̂max=fill(+Inf,nx̂)` : estimated state bound ``\mathbf{x̂_{min/max}}``
- `ŵmin=fill(-Inf,nx̂)` / `ŵmax=fill(+Inf,nx̂)` : estimated process noise bound ``\mathbf{ŵ_{min/max}}``
- `v̂min=fill(-Inf,nym)` / `v̂max=fill(+Inf,nym)` : estimated sensor noise bound ``\mathbf{v̂_{min/max}}``
- `c_x̂min=fill(0.0,nx̂)` / `c_x̂max=fill(0.0,nx̂)` : `x̂min` / `x̂max` softness weight ``\mathbf{c_{x̂_{min/max}}}``
- `c_ŵmin=fill(0.0,nx̂)` / `c_ŵmax=fill(0.0,nx̂)` : `ŵmin` / `ŵmax` softness weight ``\mathbf{c_{ŵ_{min/max}}}``
- `c_v̂min=fill(0.0,nym)` / `c_v̂max=fill(0.0,nym)` : `v̂min` / `v̂max` softness weight ``\mathbf{c_{v̂_{min/max}}}``
-  all the keyword arguments above but with a first capital letter, e.g. `X̂max` or `C_ŵmax`:
   for time-varying constraints (see Extended Help)

# Examples
```jldoctest
julia> estim = MovingHorizonEstimator(LinModel(ss(0.5,1,1,0,1)), He=3);

julia> estim = setconstraint!(estim, x̂min=[-50, -50], x̂max=[50, 50])
MovingHorizonEstimator estimator with a sample time Ts = 1.0 s, OSQP optimizer, LinModel and:
 3 estimation steps He
 0 slack variable ϵ (estimation constraints)
 1 manipulated inputs u (0 integrating states)
 2 estimated states x̂
 1 measured outputs ym (1 integrating states)
 0 unmeasured outputs yu
 0 measured disturbances d
```

# Extended Help
!!! details "Extended Help"
    The constant ``p=0`` if `estim.direct==true` (current form), else ``p=1`` (prediction
    form). Note that the state ``\mathbf{x̂}`` and process noise ``\mathbf{ŵ}`` constraints
    are applied on the augmented model, detailed in [`SteadyKalmanFilter`](@ref) Extended
    Help. For variable constraints, the bounds can be modified after calling [`updatestate!`](@ref),
    that is, at runtime, except for `±Inf` bounds. Time-varying constraints over the
    estimation horizon ``H_e`` are also possible, mathematically defined as:
    ```math 
    \begin{alignat*}{3}
        \mathbf{X̂_{min} - C_{x̂_{min}}} ϵ ≤&&\ \mathbf{X̂} &≤ \mathbf{X̂_{max} + C_{x̂_{max}}} ϵ \\
        \mathbf{Ŵ_{min} - C_{ŵ_{min}}} ϵ ≤&&\ \mathbf{Ŵ} &≤ \mathbf{Ŵ_{max} + C_{ŵ_{max}}} ϵ \\
        \mathbf{V̂_{min} - C_{v̂_{min}}} ϵ ≤&&\ \mathbf{V̂} &≤ \mathbf{V̂_{max} + C_{v̂_{max}}} ϵ
    \end{alignat*}
    ```
    For this, use the same keyword arguments as above but with a first capital letter:
    - `X̂min` / `X̂max` / `C_x̂min` / `C_x̂max` : ``\mathbf{X̂}`` constraints `(nx̂*(He+1),)`.
    - `Ŵmin` / `Ŵmax` / `C_ŵmin` / `C_ŵmax` : ``\mathbf{Ŵ}`` constraints `(nx̂*He,)`.
    - `V̂min` / `V̂max` / `C_v̂min` / `C_v̂max` : ``\mathbf{V̂}`` constraints `(nym*He,)`.
"""
function setconstraint!(
    estim::MovingHorizonEstimator; 
    xhatmin   = nothing, xhatmax   = nothing,
    whatmin   = nothing, whatmax   = nothing,
    vhatmin   = nothing, vhatmax   = nothing,
    c_xhatmin = nothing, c_xhatmax = nothing,
    c_whatmin = nothing, c_whatmax = nothing,
    c_vhatmin = nothing, c_vhatmax = nothing,
    Xhatmin   = nothing, Xhatmax   = nothing,
    Whatmin   = nothing, Whatmax   = nothing,
    Vhatmin   = nothing, Vhatmax   = nothing,
    C_xhatmin = nothing, C_xhatmax = nothing,
    C_whatmin = nothing, C_whatmax = nothing,
    C_vhatmin = nothing, C_vhatmax = nothing,
    x̂min   = xhatmin,   x̂max   = xhatmax,
    ŵmin   = whatmin,   ŵmax   = whatmax,
    v̂min   = vhatmin,   v̂max   = vhatmax,
    c_x̂min = c_xhatmin, c_x̂max = c_xhatmax,
    c_ŵmin = c_whatmin, c_ŵmax = c_whatmax,
    c_v̂min = c_vhatmin, c_v̂max = c_vhatmax,
    X̂min   = Xhatmin,   X̂max   = Xhatmax,
    Ŵmin   = Whatmin,   Ŵmax   = Whatmax,
    V̂min   = Vhatmin,   V̂max   = Vhatmax,
    C_x̂min = C_xhatmin, C_x̂max = C_xhatmax,
    C_ŵmin = C_whatmin, C_ŵmax = C_whatmax,
    C_v̂min = C_vhatmin, C_v̂max = C_vhatmax,
)
    model, optim, con = estim.model, estim.optim, estim.con
    nx̂, nŵ, nym, He = estim.nx̂, estim.nx̂, estim.nym, estim.He
    nX̂con = nx̂*(He+1)
    notSolvedYet = (JuMP.termination_status(optim) == JuMP.OPTIMIZE_NOT_CALLED)
    C = estim.C
    if isnothing(X̂min) && !isnothing(x̂min)
        size(x̂min) == (nx̂,) || throw(ArgumentError("x̂min size must be $((nx̂,))"))
        con.x̃0min[end-nx̂+1:end] .= x̂min .- estim.x̂op # if C is finite : x̃ = [ϵ; x̂]
        for i in 1:nx̂*He
            con.X̂0min[i] = x̂min[(i-1) % nx̂ + 1] - estim.X̂op[i]
        end
    elseif !isnothing(X̂min)
        size(X̂min) == (nX̂con,) || throw(ArgumentError("X̂min size must be $((nX̂con,))"))
        con.x̃0min[end-nx̂+1:end] .= X̂min[1:nx̂] .- estim.x̂op
        con.X̂0min .= @views X̂min[nx̂+1:end] .- estim.X̂op
    end
    if isnothing(X̂max) && !isnothing(x̂max)
        size(x̂max) == (nx̂,) || throw(ArgumentError("x̂max size must be $((nx̂,))"))
        con.x̃0max[end-nx̂+1:end] .= x̂max .- estim.x̂op # if C is finite : x̃ = [ϵ; x̂]
        for i in 1:nx̂*He
            con.X̂0max[i] = x̂max[(i-1) % nx̂ + 1] - estim.X̂op[i]
        end
    elseif !isnothing(X̂max)
        size(X̂max) == (nX̂con,) || throw(ArgumentError("X̂max size must be $((nX̂con,))"))
        con.x̃0max[end-nx̂+1:end] .= X̂max[1:nx̂] .- estim.x̂op
        con.X̂0max .= @views X̂max[nx̂+1:end] .- estim.X̂op
    end
    if isnothing(Ŵmin) && !isnothing(ŵmin)
        size(ŵmin) == (nŵ,) || throw(ArgumentError("ŵmin size must be $((nŵ,))"))
        for i in 1:nŵ*He
            con.Ŵmin[i] = ŵmin[(i-1) % nŵ + 1]
        end
    elseif !isnothing(Ŵmin)
        size(Ŵmin) == (nŵ*He,) || throw(ArgumentError("Ŵmin size must be $((nŵ*He,))"))
        con.Ŵmin .= Ŵmin
    end
    if isnothing(Ŵmax) && !isnothing(ŵmax)
        size(ŵmax) == (nŵ,) || throw(ArgumentError("ŵmax size must be $((nŵ,))"))
        for i in 1:nŵ*He
            con.Ŵmax[i] = ŵmax[(i-1) % nŵ + 1]
        end
    elseif !isnothing(Ŵmax)
        size(Ŵmax) == (nŵ*He,) || throw(ArgumentError("Ŵmax size must be $((nŵ*He,))"))
        con.Ŵmax .= Ŵmax
    end
    if isnothing(V̂min) && !isnothing(v̂min)
        size(v̂min) == (nym,) || throw(ArgumentError("v̂min size must be $((nym,))"))
        for i in 1:nym*He
            con.V̂min[i] = v̂min[(i-1) % nym + 1]
        end
    elseif !isnothing(V̂min)
        size(V̂min) == (nym*He,) || throw(ArgumentError("V̂min size must be $((nym*He,))"))
        con.V̂min .= V̂min
    end
    if isnothing(V̂max) && !isnothing(v̂max)
        size(v̂max) == (nym,) || throw(ArgumentError("v̂max size must be $((nym,))"))
        for i in 1:nym*He
            con.V̂max[i] = v̂max[(i-1) % nym + 1]
        end
    elseif !isnothing(V̂max)
        size(V̂max) == (nym*He,) || throw(ArgumentError("V̂max size must be $((nym*He,))"))
        con.V̂max .= V̂max
    end
    allECRs = (
        c_x̂min, c_x̂max, c_ŵmin, c_ŵmax, c_v̂min, c_v̂max,
        C_x̂min, C_x̂max, C_ŵmin, C_ŵmax, C_v̂min, C_v̂max,
    )
    if any(ECR -> !isnothing(ECR), allECRs)
        !isinf(C) || throw(ArgumentError("Slack variable weight Cwt must be finite to set softness parameters"))
        notSolvedYet || error("Cannot set softness parameters after calling updatestate!")
    end
    if notSolvedYet
        isnothing(C_x̂min) && !isnothing(c_x̂min) && (C_x̂min = repeat(c_x̂min, He+1))
        isnothing(C_x̂max) && !isnothing(c_x̂max) && (C_x̂max = repeat(c_x̂max, He+1))
        isnothing(C_ŵmin) && !isnothing(c_ŵmin) && (C_ŵmin = repeat(c_ŵmin, He))
        isnothing(C_ŵmax) && !isnothing(c_ŵmax) && (C_ŵmax = repeat(c_ŵmax, He))
        isnothing(C_v̂min) && !isnothing(c_v̂min) && (C_v̂min = repeat(c_v̂min, He))
        isnothing(C_v̂max) && !isnothing(c_v̂max) && (C_v̂max = repeat(c_v̂max, He))
        if !isnothing(C_x̂min)
            size(C_x̂min) == (nX̂con,) || throw(ArgumentError("C_x̂min size must be $((nX̂con,))"))
            any(C_x̂min .< 0) && error("C_x̂min weights should be non-negative")
            # if C is finite : x̃ = [ϵ; x̂] 
            con.A_x̃min[end-nx̂+1:end, end] .= @views -C_x̂min[1:nx̂] 
            con.C_x̂min .= @views C_x̂min[nx̂+1:end]
            size(con.A_X̂min, 1) ≠ 0 && (con.A_X̂min[:, end] = -con.C_x̂min) # for LinModel
        end
        if !isnothing(C_x̂max)
            size(C_x̂max) == (nX̂con,) || throw(ArgumentError("C_x̂max size must be $((nX̂con,))"))
            any(C_x̂max .< 0) && error("C_x̂max weights should be non-negative")
            # if C is finite : x̃ = [ϵ; x̂] :
            con.A_x̃max[end-nx̂+1:end, end] .= @views -C_x̂max[1:nx̂]
            con.C_x̂max .= @views C_x̂max[nx̂+1:end]
            size(con.A_X̂max, 1) ≠ 0 && (con.A_X̂max[:, end] = -con.C_x̂max) # for LinModel
        end
        if !isnothing(C_ŵmin)
            size(C_ŵmin) == (nŵ*He,) || throw(ArgumentError("C_ŵmin size must be $((nŵ*He,))"))
            any(C_ŵmin .< 0) && error("C_ŵmin weights should be non-negative")
            con.A_Ŵmin[:, end] .= -C_ŵmin
        end
        if !isnothing(C_ŵmax)
            size(C_ŵmax) == (nŵ*He,) || throw(ArgumentError("C_ŵmax size must be $((nŵ*He,))"))
            any(C_ŵmax .< 0) && error("C_ŵmax weights should be non-negative")
            con.A_Ŵmax[:, end] .= -C_ŵmax
        end
        if !isnothing(C_v̂min)
            size(C_v̂min) == (nym*He,) || throw(ArgumentError("C_v̂min size must be $((nym*He,))"))
            any(C_v̂min .< 0) && error("C_v̂min weights should be non-negative")
            con.C_v̂min .= C_v̂min
            size(con.A_V̂min, 1) ≠ 0 && (con.A_V̂min[:, end] = -con.C_v̂min) # for LinModel
        end
        if !isnothing(C_v̂max)
            size(C_v̂max) == (nym*He,) || throw(ArgumentError("C_v̂max size must be $((nym*He,))"))
            any(C_v̂max .< 0) && error("C_v̂max weights should be non-negative")
            con.C_v̂max .= C_v̂max
            size(con.A_V̂max, 1) ≠ 0 && (con.A_V̂max[:, end] = -con.C_v̂max) # for LinModel
        end
    end
    i_x̃min, i_x̃max  = .!isinf.(con.x̃0min), .!isinf.(con.x̃0max)
    i_X̂min, i_X̂max  = .!isinf.(con.X̂0min), .!isinf.(con.X̂0max)
    i_Ŵmin, i_Ŵmax  = .!isinf.(con.Ŵmin),  .!isinf.(con.Ŵmax)
    i_V̂min, i_V̂max  = .!isinf.(con.V̂min),  .!isinf.(con.V̂max)
    if notSolvedYet
        con.i_b[:], con.i_g[:], con.A[:] = init_matconstraint_mhe(model, 
            i_x̃min, i_x̃max, i_X̂min, i_X̂max, i_Ŵmin, i_Ŵmax, i_V̂min, i_V̂max,
            con.A_x̃min, con.A_x̃max, con.A_X̂min, con.A_X̂max, 
            con.A_Ŵmin, con.A_Ŵmax, con.A_V̂min, con.A_V̂max
        )
        A = con.A[con.i_b, :]
        b = con.b[con.i_b]
        Z̃var = optim[:Z̃var]
        JuMP.delete(optim, optim[:linconstraint])
        JuMP.unregister(optim, :linconstraint)
        @constraint(optim, linconstraint, A*Z̃var .≤ b)
        set_nonlincon!(estim, model, optim)
    else
        i_b, i_g = init_matconstraint_mhe(model, 
            i_x̃min, i_x̃max, i_X̂min, i_X̂max, i_Ŵmin, i_Ŵmax, i_V̂min, i_V̂max
        )
        if i_b ≠ con.i_b || i_g ≠ con.i_g
            error("Cannot modify ±Inf constraints after calling updatestate!")
        end
    end
    return estim
end

@doc raw"""
    init_matconstraint_mhe(model::LinModel, 
        i_x̃min, i_x̃max, i_X̂min, i_X̂max, i_Ŵmin, i_Ŵmax, i_V̂min, i_V̂max, args...
    ) -> i_b, i_g, A

Init `i_b`, `i_g` and `A` matrices for the MHE linear inequality constraints.

The linear and nonlinear inequality constraints are respectively defined as:
```math
\begin{aligned} 
    \mathbf{A Z̃ } &≤ \mathbf{b} \\ 
    \mathbf{g(Z̃)} &≤ \mathbf{0}
\end{aligned}
```
`i_b` is a `BitVector` including the indices of ``\mathbf{b}`` that are finite numbers. 
`i_g` is a similar vector but for the indices of ``\mathbf{g}`` (empty if `model` is a 
[`LinModel`](@ref)). The method also returns the ``\mathbf{A}`` matrix if `args` is
provided. In such a case, `args`  needs to contain all the inequality constraint matrices: 
`A_x̃min, A_x̃max, A_X̂min, A_X̂max, A_Ŵmin, A_Ŵmax, A_V̂min, A_V̂max`.
"""
function init_matconstraint_mhe(::LinModel{NT}, 
    i_x̃min, i_x̃max, i_X̂min, i_X̂max, i_Ŵmin, i_Ŵmax, i_V̂min, i_V̂max, args...
) where {NT<:Real}
    i_b = [i_x̃min; i_x̃max; i_X̂min; i_X̂max; i_Ŵmin; i_Ŵmax; i_V̂min; i_V̂max]
    i_g = BitVector()
    if isempty(args)
        A = zeros(NT, length(i_b), 0)
    else
        A_x̃min, A_x̃max, A_X̂min, A_X̂max, A_Ŵmin, A_Ŵmax, A_V̂min, A_V̂max = args
        A = [A_x̃min; A_x̃max; A_X̂min; A_X̂max; A_Ŵmin; A_Ŵmax; A_V̂min; A_V̂max]
    end
    return i_b, i_g, A
end

"Init `i_b, A` without state and sensor noise constraints if `model` is not a [`LinModel`](@ref)."
function init_matconstraint_mhe(::SimModel{NT}, 
    i_x̃min, i_x̃max, i_X̂min, i_X̂max, i_Ŵmin, i_Ŵmax, i_V̂min, i_V̂max, args...
) where {NT<:Real}
    i_b = [i_x̃min; i_x̃max; i_Ŵmin; i_Ŵmax]
    i_g = [i_X̂min; i_X̂max; i_V̂min; i_V̂max]
    if isempty(args)
        A = zeros(NT, length(i_b), 0)
    else
        A_x̃min, A_x̃max, _ , _ , A_Ŵmin, A_Ŵmax, _ , _ = args
        A = [A_x̃min; A_x̃max; A_Ŵmin; A_Ŵmax]
    end
    return i_b, i_g, A
end

"By default, no nonlinear constraints in the MHE, thus return nothing."
set_nonlincon!(::MovingHorizonEstimator, ::SimModel, ::JuMP.GenericModel) = nothing

"Set the nonlinear constraints on the output predictions `Ŷ` and terminal states `x̂end`."
function set_nonlincon!(
    estim::MovingHorizonEstimator, ::NonLinModel, optim::JuMP.GenericModel{JNT}
) where JNT<:Real
    optim, con = estim.optim, estim.con
    Z̃var = optim[:Z̃var]
    nonlin_constraints = JuMP.all_constraints(optim, JuMP.NonlinearExpr, MOI.LessThan{JNT})
    map(con_ref -> JuMP.delete(optim, con_ref), nonlin_constraints)
    for i in findall(.!isinf.(con.X̂0min))
        gfunc_i = optim[Symbol("g_X̂0min_$(i)")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    for i in findall(.!isinf.(con.X̂0max))
        gfunc_i = optim[Symbol("g_X̂0max_$(i)")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    for i in findall(.!isinf.(con.V̂min))
        gfunc_i = optim[Symbol("g_V̂min_$(i)")]
        JuMP.@constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    for i in findall(.!isinf.(con.V̂max))
        gfunc_i = optim[Symbol("g_V̂max_$(i)")]
        JuMP.@constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    return nothing
end

"""
    init_defaultcon_mhe(
        model::SimModel, He, C, nx̂, nym, E, ex̄, Ex̂, Fx̂, Gx̂, Jx̂, Bx̂
    ) -> con, Ẽ, ẽx̄

    Init `EstimatatorConstraint` struct with default parameters based on model `model`.

Also return `Ẽ` and `ẽx̄` matrices for the the augmented decision vector `Z̃`.
"""
function init_defaultcon_mhe(
    model::SimModel{NT}, He, C, nx̂, nym, E, ex̄, Ex̂, Fx̂, Gx̂, Jx̂, Bx̂
) where {NT<:Real}
    nŵ = nx̂
    nZ̃, nX̂, nŴ, nYm = nx̂+nŵ*He, nx̂*He, nŵ*He, nym*He
    nϵ = isinf(C) ? 0 : 1
    x̂min, x̂max = fill(convert(NT,-Inf), nx̂),  fill(convert(NT,+Inf), nx̂)
    X̂min, X̂max = fill(convert(NT,-Inf), nX̂),  fill(convert(NT,+Inf), nX̂)
    Ŵmin, Ŵmax = fill(convert(NT,-Inf), nŴ),  fill(convert(NT,+Inf), nŴ)
    V̂min, V̂max = fill(convert(NT,-Inf), nYm), fill(convert(NT,+Inf), nYm)
    c_x̂min, c_x̂max = fill(0.0, nx̂),  fill(0.0, nx̂)
    C_x̂min, C_x̂max = fill(0.0, nX̂),  fill(0.0, nX̂)
    C_ŵmin, C_ŵmax = fill(0.0, nŴ),  fill(0.0, nŴ)
    C_v̂min, C_v̂max = fill(0.0, nYm), fill(0.0, nYm)
    A_x̃min, A_x̃max, x̃min, x̃max, ẽx̄ = relaxarrival(model, nϵ, c_x̂min, c_x̂max, x̂min, x̂max, ex̄)
    A_X̂min, A_X̂max, Ẽx̂ = relaxX̂(model, nϵ, C_x̂min, C_x̂max, Ex̂)
    A_Ŵmin, A_Ŵmax = relaxŴ(model, nϵ, C_ŵmin, C_ŵmax, nx̂)
    A_V̂min, A_V̂max, Ẽ = relaxV̂(model, nϵ, C_v̂min, C_v̂max, E)
    i_x̃min, i_x̃max = .!isinf.(x̃min), .!isinf.(x̃max)
    i_X̂min, i_X̂max = .!isinf.(X̂min), .!isinf.(X̂max)
    i_Ŵmin, i_Ŵmax = .!isinf.(Ŵmin), .!isinf.(Ŵmax)
    i_V̂min, i_V̂max = .!isinf.(V̂min), .!isinf.(V̂max)
    i_b, i_g, A = init_matconstraint_mhe(model, 
        i_x̃min, i_x̃max, i_X̂min, i_X̂max, i_Ŵmin, i_Ŵmax, i_V̂min, i_V̂max,
        A_x̃min, A_x̃max, A_X̂min, A_X̂max, A_Ŵmin, A_Ŵmax, A_V̂min, A_V̂max
    )
    b = zeros(NT, size(A, 1)) # dummy b vector (updated just before optimization)
    con = EstimatorConstraint{NT}(
        Ẽx̂, Fx̂, Gx̂, Jx̂, Bx̂,
        x̃min, x̃max, X̂min, X̂max, Ŵmin, Ŵmax, V̂min, V̂max,
        A_x̃min, A_x̃max, A_X̂min, A_X̂max, A_Ŵmin, A_Ŵmax, A_V̂min, A_V̂max,
        A, b,
        C_x̂min, C_x̂max, C_v̂min, C_v̂max,
        i_b, i_g
    )
    return con, nϵ, Ẽ, ẽx̄
end

@doc raw"""
    relaxarrival(
        model::SimModel, nϵ, c_x̂min, c_x̂max, x̂min, x̂max, ex̄
    ) -> A_x̃min, A_x̃max, x̃min, x̃max, ẽx̄

Augment arrival state constraints with slack variable ϵ for softening the MHE.

Denoting the MHE decision variable augmented with the slack variable ``\mathbf{Z̃} = 
[\begin{smallmatrix} ϵ \\ \mathbf{Z} \end{smallmatrix}]``, it returns the ``\mathbf{ẽ_x̄}``
matrix that appears in the estimation error at arrival equation ``\mathbf{x̄} =
\mathbf{ẽ_x̄ Z̃ + f_x̄}``. It also returns the augmented constraints ``\mathbf{x̃_{min}}`` and
``\mathbf{x̃_{max}}``, and the ``\mathbf{A}`` matrices for the inequality constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{x̃_{min}}} \\ 
    \mathbf{A_{x̃_{max}}}
\end{bmatrix} \mathbf{Z̃} ≤
\begin{bmatrix}
    - \mathbf{(x̃_{min} - x̃_{op})} \\
    + \mathbf{(x̃_{max} - x̃_{op})}
\end{bmatrix}
```
in which
``\mathbf{x̃_{min}} = [\begin{smallmatrix} 0 \\ \mathbf{x̂_{min}} \end{smallmatrix}]``, 
``\mathbf{x̃_{max}} = [\begin{smallmatrix} ∞ \\ \mathbf{x̂_{max}} \end{smallmatrix}]`` and
``\mathbf{x̃_{op}}  = [\begin{smallmatrix} 0 \\ \mathbf{x̂_{op}}  \end{smallmatrix}]``
"""
function relaxarrival(::SimModel{NT}, nϵ, c_x̂min, c_x̂max, x̂min, x̂max, ex̄) where {NT<:Real}
    ex̂ = -ex̄
    if nϵ ≠ 0 # Z̃ = [ϵ; Z]
        x̃min, x̃max = [NT[0.0]; x̂min], [NT[Inf]; x̂max]
        A_ϵ = [NT[1.0] zeros(NT, 1, size(ex̂, 2))]
        # ϵ impacts arrival state constraint calculations:
        A_x̃min, A_x̃max = -[A_ϵ; c_x̂min ex̂], [A_ϵ; -c_x̂max ex̂]
        # ϵ has no impact on estimation error at arrival:
        ẽx̄ = [zeros(NT, size(ex̄, 1), 1) ex̄] 
    else # Z̃ = Z (only hard constraints)
        x̃min, x̃max = x̂min, x̂max
        ẽx̄ = ex̄
        A_x̃min, A_x̃max = -ex̂, ex̂
    end
    return A_x̃min, A_x̃max, x̃min, x̃max, ẽx̄
end

@doc raw"""
    relaxX̂(model::SimModel, nϵ, C_x̂min, C_x̂max, Ex̂) -> A_X̂min, A_X̂max, Ẽx̂

Augment estimated state constraints with slack variable ϵ for softening the MHE.

Denoting the MHE decision variable augmented with the slack variable ``\mathbf{Z̃} = 
[\begin{smallmatrix} ϵ \\ \mathbf{Z} \end{smallmatrix}]``, it returns the ``\mathbf{Ẽ_x̂}``
matrix that appears in estimated states equation ``\mathbf{X̂} = \mathbf{Ẽ_x̂ Z̃ + F_x̂}``. It
also returns the ``\mathbf{A}`` matrices for the inequality constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{X̂_{min}}} \\ 
    \mathbf{A_{X̂_{max}}}
\end{bmatrix} \mathbf{Z̃} ≤
\begin{bmatrix}
    - \mathbf{(X̂_{min} - X̂_{op}) + F_x̂} \\
    + \mathbf{(X̂_{max} - X̂_{op}) - F_x̂}
\end{bmatrix}
```
in which ``\mathbf{X̂_{min}, X̂_{max}}`` and ``\mathbf{X̂_{op}}`` vectors respectively contains
``\mathbf{x̂_{min}, x̂_{max}}`` and ``\mathbf{x̂_{op}}`` repeated ``H_e`` times.
"""
function relaxX̂(::LinModel{NT}, nϵ, C_x̂min, C_x̂max, Ex̂) where {NT<:Real}
    if nϵ ≠ 0 # Z̃ = [ϵ; Z]
        # ϵ impacts estimated process noise constraint calculations:
        A_X̂min, A_X̂max = -[C_x̂min Ex̂], [-C_x̂max Ex̂]
        # ϵ has no impact on estimated process noises:
        Ẽx̂ = [zeros(NT, size(Ex̂, 1), 1) Ex̂] 
    else # Z̃ = Z (only hard constraints)
        Ẽx̂ = Ex̂
        A_X̂min, A_X̂max = -Ex̂, Ex̂
    end
    return A_X̂min, A_X̂max, Ẽx̂
end

"Return empty matrices if model is not a [`LinModel`](@ref)"
function relaxX̂(::SimModel{NT}, nϵ, C_x̂min, C_x̂max, Ex̂) where {NT<:Real}
    Ẽx̂ = [zeros(NT, 0, nϵ) Ex̂]
    A_X̂min, A_X̂max = -Ẽx̂,  Ẽx̂
    return A_X̂min, A_X̂max, Ẽx̂
end

@doc raw"""
    relaxŴ(model::SimModel, nϵ, C_ŵmin, C_ŵmax, nx̂) -> A_Ŵmin, A_Ŵmax

Augment estimated process noise constraints with slack variable ϵ for softening the MHE.

Denoting the MHE decision variable augmented with the slack variable ``\mathbf{Z̃} = 
[\begin{smallmatrix} ϵ \\ \mathbf{Z} \end{smallmatrix}]``, it returns the ``\mathbf{A}`` 
matrices for the inequality constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{Ŵ_{min}}} \\ 
    \mathbf{A_{Ŵ_{max}}}
\end{bmatrix} \mathbf{Z̃} ≤
\begin{bmatrix}
    - \mathbf{Ŵ_{min}} \\
    + \mathbf{Ŵ_{max}}
\end{bmatrix}
```
"""
function relaxŴ(::SimModel{NT}, nϵ, C_ŵmin, C_ŵmax, nx̂) where {NT<:Real}
    A = [zeros(NT, length(C_ŵmin), nx̂) I]
    if nϵ ≠ 0 # Z̃ = [ϵ; Z]
        A_Ŵmin, A_Ŵmax = -[C_ŵmin A], [-C_ŵmax A]
    else # Z̃ = Z (only hard constraints)
        A_Ŵmin, A_Ŵmax = -A, A
    end
    return A_Ŵmin, A_Ŵmax
end

@doc raw"""
    relaxV̂(model::SimModel, nϵ, C_v̂min, C_v̂max, E) -> A_V̂min, A_V̂max, Ẽ

Augment estimated sensor noise constraints with slack variable ϵ for softening the MHE.

Denoting the MHE decision variable augmented with the slack variable ``\mathbf{Z̃} = 
[\begin{smallmatrix} ϵ \\ \mathbf{Z} \end{smallmatrix}]``, it returns the ``\mathbf{Ẽ}``
matrix that appears in estimated sensor noise equation ``\mathbf{V̂} = \mathbf{Ẽ Z̃ + F}``. It
also returns the ``\mathbf{A}`` matrices for the inequality constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{V̂_{min}}} \\ 
    \mathbf{A_{V̂_{max}}}
\end{bmatrix} \mathbf{Z̃} ≤
\begin{bmatrix}
    - \mathbf{V̂_{min} + F} \\
    + \mathbf{V̂_{max} - F}
\end{bmatrix}
```
"""
function relaxV̂(::LinModel{NT}, nϵ, C_v̂min, C_v̂max, E) where {NT<:Real}
    if nϵ ≠ 0 # Z̃ = [ϵ; Z]
        # ϵ impacts estimated sensor noise constraint calculations:
        A_V̂min, A_V̂max = -[C_v̂min E], [-C_v̂max E]
        # ϵ has no impact on estimated sensor noises:
        Ẽ = [zeros(NT, size(E, 1), 1) E] 
    else # Z̃ = Z (only hard constraints)
        Ẽ = E
        A_V̂min, A_V̂max = -Ẽ, Ẽ
    end
    return A_V̂min, A_V̂max, Ẽ
end

"Return empty matrices if model is not a [`LinModel`](@ref)"
function relaxV̂(::SimModel{NT}, nϵ, C_v̂min, C_v̂max, E) where {NT<:Real}
    Ẽ = [zeros(NT, 0, nϵ) E]
    A_V̂min, A_V̂max = -Ẽ, Ẽ
    return A_V̂min, A_V̂max, Ẽ
end

@doc raw"""
    init_predmat_mhe(
        model::LinModel, He, i_ym, Â, B̂u, Ĉm, B̂d, D̂dm, x̂op, f̂op, p
    ) -> E, G, J, B, ex̄, Ex̂, Gx̂, Jx̂, Bx̂

Construct the [`MovingHorizonEstimator`](@ref) prediction matrices for [`LinModel`](@ref) `model`.

We first introduce the deviation vector of the estimated state at arrival 
``\mathbf{x̂_0}(k-N_k+p) = \mathbf{x̂}_k(k-N_k+p) - \mathbf{x̂_{op}}`` (see [`setop!`](@ref)),
and the vector ``\mathbf{Z} = [\begin{smallmatrix} \mathbf{x̂_0}(k-N_k+p)
\\ \mathbf{Ŵ} \end{smallmatrix}]`` with the decision variables. Setting the constant ``p=0``
produces an estimator in the current form, while the prediction form is obtained with
``p=1``. The estimated sensor noises from time ``k-N_k+1`` to ``k`` are computed by:
```math
\begin{aligned}
    \mathbf{V̂} = \mathbf{Y_0^m - Ŷ_0^m} &= \mathbf{E Z + G U_0 + J D_0 + Y_0^m + B}     \\
                                        &= \mathbf{E Z + F}
\end{aligned}
```
in which ``\mathbf{U_0}`` and ``\mathbf{Y_0^m}`` respectively include the deviation values of
the manipulated inputs ``\mathbf{u_0}(k-j+p)`` from ``j=N_k`` to ``1`` and measured outputs
``\mathbf{y_0^m}(k-j+1)`` from ``j=N_k`` to ``1``. The vector ``\mathbf{D_0}`` comprises one 
additional measured disturbance if ``p=0``, that is, it includes the deviation vectors
``\mathbf{d_0}(k-j+1)`` from ``j=N_k+1-p`` to ``1``. The constant ``\mathbf{B}`` is the
contribution for non-zero state ``\mathbf{x̂_{op}}`` and state update ``\mathbf{f̂_{op}}``
operating points (for linearization, see [`augment_model`](@ref) and [`linearize`](@ref)).
The method also returns the matrices for the estimation error at arrival:
```math
    \mathbf{x̄} = \mathbf{x̂_0^†}(k-N_k+p) - \mathbf{x̂_0}(k-N_k+p) = \mathbf{e_x̄ Z + f_x̄}
```
in which ``\mathbf{e_x̄} = [\begin{smallmatrix} -\mathbf{I} & \mathbf{0} & \cdots & \mathbf{0} \end{smallmatrix}]``,
and ``\mathbf{f_x̄} = \mathbf{x̂_0^†}(k-N_k+p)``. The latter is the deviation vector of the
state at arrival, estimated at time ``k-N_k``, i.e. ``\mathbf{x̂_0^†}(k-N_k+p) = 
\mathbf{x̂}_{k-N_k}(k-N_k+p) - \mathbf{x̂_{op}}``. Lastly, the estimates ``\mathbf{x̂_0}(k-j+p)``
from ``j=N_k-1`` to ``0``, also in deviation form, are computed with:
```math
\begin{aligned}
    \mathbf{X̂_0}  &= \mathbf{E_x̂ Z + G_x̂ U_0 + J_x̂ D_0 + B_x̂} \\
                  &= \mathbf{E_x̂ Z + F_x̂}
\end{aligned}
```
The matrices ``\mathbf{E, G, J, B, E_x̂, G_x̂, J_x̂, B_x̂}`` are defined in the Extended Help 
section. The vectors ``\mathbf{F, F_x̂, f_x̄}`` are recalculated at each discrete time step, 
see [`initpred!(::MovingHorizonEstimator, ::LinModel)`](@ref) and [`linconstraint!(::MovingHorizonEstimator, ::LinModel)`](@ref).

# Extended Help
!!! details "Extended Help"
    Using the augmented process model matrices ``\mathbf{Â, B̂_u, Ĉ^m, B̂_d, D̂_d^m}``, and the
    function ``\mathbf{S}(j) = ∑_{i=0}^j \mathbf{Â}^i``, the prediction matrices for the
    sensor noises depend on the constant ``p``. For ``p=0``, the matrices are computed by
    (notice the minus signs after the equalities):
    ```math
    \begin{aligned}
    \mathbf{E} &= - \begin{bmatrix}
        \mathbf{Ĉ^m}\mathbf{Â}^{1}                  & \mathbf{Ĉ^m}\mathbf{Â}^{0}                    & \cdots & \mathbf{0}                               \\ 
        \mathbf{Ĉ^m}\mathbf{Â}^{2}                  & \mathbf{Ĉ^m}\mathbf{Â}^{1}                    & \cdots & \mathbf{0}                               \\ 
        \vdots                                      & \vdots                                        & \ddots & \vdots                                   \\
        \mathbf{Ĉ^m}\mathbf{Â}^{H_e}                & \mathbf{Ĉ^m}\mathbf{Â}^{H_e-1}                & \cdots & \mathbf{Ĉ^m}\mathbf{Â}^{0}               \end{bmatrix} \\
    \mathbf{G} &= - \begin{bmatrix}
        \mathbf{Ĉ^m}\mathbf{Â}^{0}\mathbf{B̂_u}      & \mathbf{0}                                    & \cdots & \mathbf{0}                               \\ 
        \mathbf{Ĉ^m}\mathbf{Â}^{1}\mathbf{B̂_u}      & \mathbf{Ĉ^m}\mathbf{Â}^{0}\mathbf{B̂_u}        & \cdots & \mathbf{0}                               \\ 
        \vdots                                      & \vdots                                        & \ddots & \vdots                                   \\
        \mathbf{Ĉ^m}\mathbf{Â}^{H_e-1}\mathbf{B̂_u}  & \mathbf{Ĉ^m}\mathbf{Â}^{H_e-2}\mathbf{B̂_u}    & \cdots & \mathbf{Ĉ^m}\mathbf{Â}^{0}\mathbf{B̂_u}   \end{bmatrix} \\
    \mathbf{J} &= - \begin{bmatrix}
        \mathbf{Ĉ^m}\mathbf{Â}^{0}\mathbf{B̂_d}      & \mathbf{D̂_d^m}                                & \cdots & \mathbf{0}                               \\ 
        \mathbf{Ĉ^m}\mathbf{Â}^{1}\mathbf{B̂_d}      & \mathbf{Ĉ^m}\mathbf{Â}^{0}\mathbf{B̂_d}        & \cdots & \mathbf{0}                               \\ 
        \vdots                                      & \vdots                                        & \ddots & \vdots                                   \\
        \mathbf{Ĉ^m}\mathbf{Â}^{H_e-1}\mathbf{B̂_d}  & \mathbf{Ĉ^m}\mathbf{Â}^{H_e-2}\mathbf{B̂_d}    & \cdots & \mathbf{D̂_d^m}                           \end{bmatrix} \\
    \mathbf{B} &= - \begin{bmatrix}
        \mathbf{Ĉ^m S}(0)                    \\
        \mathbf{Ĉ^m S}(1)                    \\
        \vdots                               \\
        \mathbf{Ĉ^m S}(H_e-1) \end{bmatrix}  \mathbf{\big(f̂_{op} - x̂_{op}\big)}
    \end{aligned}
    ```
    or, for ``p=1``, the matrices are given by:
    ```math
    \begin{aligned}
    \mathbf{E} &= - \begin{bmatrix}
        \mathbf{Ĉ^m}\mathbf{Â}^{0}                  & \mathbf{0}                                    & \cdots & \mathbf{0}   \\ 
        \mathbf{Ĉ^m}\mathbf{Â}^{1}                  & \mathbf{Ĉ^m}\mathbf{Â}^{0}                    & \cdots & \mathbf{0}   \\ 
        \vdots                                      & \vdots                                        & \ddots & \vdots       \\
        \mathbf{Ĉ^m}\mathbf{Â}^{H_e-1}              & \mathbf{Ĉ^m}\mathbf{Â}^{H_e-2}                & \cdots & \mathbf{0}   \end{bmatrix} \\
    \mathbf{G} &= - \begin{bmatrix}
        \mathbf{0}                                  & \mathbf{0}                                    & \cdots & \mathbf{0}   \\ 
        \mathbf{Ĉ^m}\mathbf{Â}^{0}\mathbf{B̂_u}      & \mathbf{0}                                    & \cdots & \mathbf{0}   \\ 
        \vdots                                      & \vdots                                        & \ddots & \vdots       \\
        \mathbf{Ĉ^m}\mathbf{Â}^{H_e-2}\mathbf{B̂_u}  & \mathbf{Ĉ^m}\mathbf{Â}^{H_e-3}\mathbf{B̂_u}    & \cdots & \mathbf{0}   \end{bmatrix} \\
    \mathbf{J} &= - \begin{bmatrix}
        \mathbf{D̂_d^m}                              & \mathbf{0}                                    & \cdots & \mathbf{0}   \\ 
        \mathbf{Ĉ^m}\mathbf{Â}^{0}\mathbf{B̂_d}      & \mathbf{D̂_d^m}                                & \cdots & \mathbf{0}   \\ 
        \vdots                                      & \vdots                                        & \ddots & \vdots       \\
        \mathbf{Ĉ^m}\mathbf{Â}^{H_e-2}\mathbf{B̂_d}  & \mathbf{Ĉ^m}\mathbf{Â}^{H_e-3}\mathbf{B̂_d}    & \cdots & \mathbf{D̂_d^m} \end{bmatrix} \\
    \mathbf{B} &= - \begin{bmatrix}
        \mathbf{0}                           \\  
        \mathbf{Ĉ^m S}(0)                    \\
        \vdots                               \\
        \mathbf{Ĉ^m S}(H_e-2) \end{bmatrix}  \mathbf{\big(f̂_{op} - x̂_{op}\big)}
    \end{aligned}
    ```
    The matrices for the estimated states are computed by:
    ```math
    \begin{aligned}
    \mathbf{E_x̂} &= \begin{bmatrix}
        \mathbf{Â}^{1}                      & \mathbf{A}^{0}                    & \cdots & \mathbf{0}                   \\
        \mathbf{Â}^{2}                      & \mathbf{Â}^{1}                    & \cdots & \mathbf{0}                   \\ 
        \vdots                              & \vdots                            & \ddots & \vdots                       \\
        \mathbf{Â}^{H_e}                    & \mathbf{Â}^{H_e-1}                & \cdots & \mathbf{Â}^{0}               \end{bmatrix} \\
    \mathbf{G_x̂} &= \begin{bmatrix}
        \mathbf{Â}^{0}\mathbf{B̂_u}          & \mathbf{0}                        & \cdots & \mathbf{0}                   \\ 
        \mathbf{Â}^{1}\mathbf{B̂_u}          & \mathbf{Â}^{0}\mathbf{B̂_u}        & \cdots & \mathbf{0}                   \\ 
        \vdots                              & \vdots                            & \ddots & \vdots                       \\
        \mathbf{Â}^{H_e-1}\mathbf{B̂_u}      & \mathbf{Â}^{H_e-2}\mathbf{B̂_u}    & \cdots & \mathbf{Â}^{0}\mathbf{B̂_u}   \end{bmatrix} \\
    \mathbf{J_x̂^†} &= \begin{bmatrix}
        \mathbf{Â}^{0}\mathbf{B̂_d}          & \mathbf{0}                        & \cdots & \mathbf{0}                   \\ 
        \mathbf{Â}^{1}\mathbf{B̂_d}          & \mathbf{Â}^{0}\mathbf{B̂_d}        & \cdots & \mathbf{0}                   \\ 
        \vdots                              & \vdots                            & \ddots & \vdots                       \\
        \mathbf{Â}^{H_e-1}\mathbf{B̂_d}      & \mathbf{Â}^{H_e-2}\mathbf{B̂_d}    & \cdots & \mathbf{Â}^{0}\mathbf{B̂_d}   \end{bmatrix} \ , \quad
    \mathbf{J_x̂} = \begin{cases}
        [\begin{smallmatrix} \mathbf{J_x̂^†} & \mathbf{0} \end{smallmatrix}]     & p=0                                   \\
                             \mathbf{J_x̂^†}                                     & p=1                                   \end{cases}   \\
    \mathbf{B_x̂} &= \begin{bmatrix}
        \mathbf{S}(0)                    \\
        \mathbf{S}(1)                    \\
        \vdots                           \\
        \mathbf{S}(H_e-1) \end{bmatrix}  \mathbf{\big(f̂_{op} - x̂_{op}\big)}
    \end{aligned}
    ```
    All these matrices are truncated when ``N_k < H_e`` (at the beginning).
"""
function init_predmat_mhe(
    model::LinModel{NT}, He, i_ym, Â, B̂u, Ĉm, B̂d, D̂dm, x̂op, f̂op, p
) where {NT<:Real}
    nu, nd = model.nu, model.nd
    nym, nx̂ = length(i_ym), size(Â, 2)
    nŵ = nx̂
    # --- pre-compute matrix powers ---
    # Apow3D array : Apow[:,:,1] = A^0, Apow[:,:,2] = A^1, ... , Apow[:,:,He+1] = A^He
    Âpow3D = Array{NT}(undef, nx̂, nx̂, He+1)
    Âpow3D[:,:,1] = I(nx̂)
    for j=2:He+1
        Âpow3D[:,:,j] = @views Âpow3D[:,:,j-1]*Â
    end
    # nĈm_Âpow3D array : similar indices as Apow3D
    nĈm_Âpow3D = Array{NT}(undef, nym, nx̂, He+1)
    nĈm_Âpow3D[:,:,1] = -Ĉm
    for j=2:He+1
        nĈm_Âpow3D[:,:,j] = @views -Ĉm*Âpow3D[:,:,j]
    end
    # helper function to improve code clarity and be similar to eqs. in docstring:
    getpower(array3D, power) = @views array3D[:,:, power+1]
    # --- decision variables Z ---
    nĈm_Âpow = reduce(vcat, getpower(nĈm_Âpow3D, i) for i=0:He)
    E = zeros(NT, nym*He, nx̂ + nŵ*He)
    col_begin = iszero(p) ? 1    : 0
    col_end   = iszero(p) ? He : He-1
    i = 0
    for j=col_begin:col_end
        iRow = (1 + i*nym):(nym*He)
        iCol = (1:nŵ) .+ j*nŵ
        E[iRow, iCol] = @views nĈm_Âpow[1:length(iRow) ,:]
        i += 1
    end
    iszero(p) && @views (E[:, 1:nx̂] = @views nĈm_Âpow[nym+1:end, :])
    ex̄ = [-I zeros(NT, nx̂, nŵ*He)]
    Âpow_vec = reduce(vcat, getpower(Âpow3D, i) for i=0:He)
    Ex̂ = zeros(NT, nx̂*He, nx̂ + nŵ*He)
    i=0
    for j=1:He
        iRow = (1 + i*nx̂):(nx̂*He)
        iCol = (1:nŵ) .+ j*nŵ
        Ex̂[iRow, iCol] = @views Âpow_vec[1:length(iRow) ,:]
        i+=1
    end
    Ex̂[:, 1:nx̂] = @views Âpow_vec[nx̂+1:end, :] 
    # --- manipulated inputs U ---
    nĈm_Âpow_B̂u = reduce(vcat, getpower(nĈm_Âpow3D, i)*B̂u for i=0:He-1)
    nĈm_Âpow_B̂u = [zeros(nym, nu) ; nĈm_Âpow_B̂u]
    G = zeros(NT, nym*He, nu*He)
    i=0
    col_begin = iszero(p) ? 1    : 0
    col_end   = iszero(p) ? He-1 : He-2
    for j=col_begin:col_end
        iRow = (1 + i*nym):(nym*He)
        iCol = (1:nu) .+ j*nu
        G[iRow, iCol] = @views nĈm_Âpow_B̂u[1:length(iRow) ,:]
        i+=1
    end
    iszero(p) && @views (G[:, 1:nu] = nĈm_Âpow_B̂u[nym+1:end, :])
    Âpow_B̂u = reduce(vcat, getpower(Âpow3D, i)*B̂u for i=0:He-1)
    Gx̂ = zeros(NT, nx̂*He, nu*He)
    for j=0:He-1
        iRow = (1 + j*nx̂):(nx̂*He)
        iCol = (1:nu) .+ j*nu
        Gx̂[iRow, iCol] = @views Âpow_B̂u[1:length(iRow) ,:]
    end
    # --- measured disturbances D ---
    nĈm_Âpow_B̂d = reduce(vcat, getpower(nĈm_Âpow3D, i)*B̂d for i=0:He-1)
    nĈm_Âpow_B̂d = [-D̂dm; nĈm_Âpow_B̂d]
    J = zeros(NT, nym*He, nd*(He+1-p))
    col_begin = iszero(p) ? 1    : 0
    col_end   = iszero(p) ? He+1 : He
    i=0
    for j=col_begin:col_end-1
        iRow = (1 + i*nym):(nym*He)
        iCol = (1:nd) .+ j*nd
        J[iRow, iCol] = nĈm_Âpow_B̂d[1:length(iRow) ,:]
        i+=1
    end
    iszero(p) && @views (J[:, 1:nd] = nĈm_Âpow_B̂d[nym+1:end, :])
    Âpow_B̂d = reduce(vcat, getpower(Âpow3D, i)*B̂d for i=0:He-1)
    Jx̂ = zeros(NT, nx̂*He, nd*(He+1-p))
    for j=0:He-1
        iRow = (1 + j*nx̂):(nx̂*He)
        iCol = (1:nd) .+ j*nd
        Jx̂[iRow, iCol] = Âpow_B̂d[1:length(iRow) ,:]
    end
    # --- state x̂op and state update f̂op operating points ---
    # Apow_csum 3D array : Apow_csum[:,:,1] = A^0, Apow_csum[:,:,2] = A^1 + A^0, ...
    Âpow_csum  = cumsum(Âpow3D, dims=3)
    f̂_op_n_x̂op = (f̂op - x̂op)
    coef_B  = zeros(NT, nym*He, nx̂)
    row_begin = iszero(p) ? 0    : 1
    row_end   = iszero(p) ? He-1 : He-2
    j=0
    for i=row_begin:row_end
        iRow = (1:nym) .+ nym*i
        coef_B[iRow,:] = -Ĉm*getpower(Âpow_csum, j)
        j+=1
    end
    B = coef_B*f̂_op_n_x̂op
    coef_Bx̂ = Matrix{NT}(undef, nx̂*He, nx̂)
    for j=0:He-1
        iRow = (1:nx̂)  .+ nx̂*j
        coef_Bx̂[iRow,:] = getpower(Âpow_csum, j)
    end
    Bx̂ = coef_Bx̂*f̂_op_n_x̂op
    return E, G, J, B, ex̄, Ex̂, Gx̂, Jx̂, Bx̂
end

"Return empty matrices if `model` is not a [`LinModel`](@ref), except for `ex̄`."
function init_predmat_mhe(
    model::SimModel{NT}, He, i_ym, Â, _ , _ , _ , _ , _ , _ , p
) where {NT<:Real}
    nym, nx̂ = length(i_ym), size(Â, 2)
    nŵ = nx̂
    E  = zeros(NT, 0, nx̂ + nŵ*He)
    ex̄ = [-I zeros(NT, nx̂, nŵ*He)]
    Ex̂ = zeros(NT, 0, nx̂ + nŵ*He)
    G  = zeros(NT, 0, model.nu*He)
    Gx̂ = zeros(NT, 0, model.nu*He)
    J  = zeros(NT, 0, model.nd*(He+1-p))
    Jx̂ = zeros(NT, 0, model.nd*He)
    B  = zeros(NT, nym*He)
    Bx̂ = zeros(NT, nx̂*He)
    return E, G, J, B, ex̄, Ex̂, Gx̂, Jx̂, Bx̂
end

"""
    init_optimization!(estim::MovingHorizonEstimator, model::SimModel, optim)

Init the quadratic optimization of [`MovingHorizonEstimator`](@ref).
"""
function init_optimization!(
    estim::MovingHorizonEstimator, ::LinModel, optim::JuMP.GenericModel
)
    nZ̃ = length(estim.Z̃)
    JuMP.num_variables(optim) == 0 || JuMP.empty!(optim)
    JuMP.set_silent(optim)
    limit_solve_time(estim.optim, estim.model.Ts)
    @variable(optim, Z̃var[1:nZ̃])
    A = estim.con.A[estim.con.i_b, :]
    b = estim.con.b[estim.con.i_b]
    @constraint(optim, linconstraint, A*Z̃var .≤ b)
    @objective(optim, Min, obj_quadprog(Z̃var, estim.H̃, estim.q̃))
    return nothing
end

"""
    init_optimization!(estim::MovingHorizonEstimator, model::SimModel, optim)

Init the nonlinear optimization of [`MovingHorizonEstimator`](@ref).
"""
function init_optimization!(
    estim::MovingHorizonEstimator, model::SimModel, optim::JuMP.GenericModel{JNT},
) where JNT<:Real
    C, con = estim.C, estim.con
    nZ̃ = length(estim.Z̃)
    # --- variables and linear constraints ---
    JuMP.num_variables(optim) == 0 || JuMP.empty!(optim)
    JuMP.set_silent(optim)
    limit_solve_time(estim.optim, estim.model.Ts)
    @variable(optim, Z̃var[1:nZ̃])
    A = estim.con.A[con.i_b, :]
    b = estim.con.b[con.i_b]
    @constraint(optim, linconstraint, A*Z̃var .≤ b)
    # --- nonlinear optimization init ---
    if !isinf(C) && JuMP.solver_name(optim) == "Ipopt"
        try
            JuMP.get_attribute(optim, "nlp_scaling_max_gradient")
        catch
            # default "nlp_scaling_max_gradient" to `10.0/C` if not already set:
            JuMP.set_attribute(optim, "nlp_scaling_max_gradient", 10.0/C)
        end
    end
    Jfunc, ∇Jfunc!, gfuncs, ∇gfuncs! = get_optim_functions(estim, optim)
    @operator(optim, J, nZ̃, Jfunc)
    @objective(optim, Min, J(Z̃var...))
    nV̂, nX̂ = estim.He*estim.nym, estim.He*estim.nx̂
    if length(con.i_g) ≠ 0
        for i in eachindex(con.X̂0min)
            name = Symbol("g_X̂0min_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i]; name
            )
        end
        i_end_X̂min = nX̂
        for i in eachindex(con.X̂0max)
            name = Symbol("g_X̂0max_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_end_X̂min + i]; name
            )
        end
        i_end_X̂max = 2*nX̂
        for i in eachindex(con.V̂min)
            name = Symbol("g_V̂min_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_end_X̂max + i]; name
            )
        end
        i_end_V̂min = 2*nX̂ + nV̂
        for i in eachindex(con.V̂max)
            name = Symbol("g_V̂max_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_end_V̂min + i]; name
            )
        end
    end
    return nothing
end


"""
    get_optim_functions(
        estim::MovingHorizonEstimator, optim::JuMP.GenericModel
    ) -> Jfunc, ∇Jfunc!, gfuncs, ∇gfuncs!

Return the functions for the nonlinear optimization of [`MovingHorizonEstimator`](@ref).

Return the nonlinear objective `Jfunc` function, and `∇Jfunc!`, to compute its gradient. 
Also return vectors with the nonlinear inequality constraint functions `gfuncs`, and 
`∇gfuncs!`, for the associated gradients. 

This method is really indicated and I'm not proud of it. That's because of 3 elements:

- These functions are used inside the nonlinear optimization, so they must be type-stable
  and as efficient as possible.
- The `JuMP` NLP syntax forces splatting for the decision variable, which implies use
  of `Vararg{T,N}` (see the [performance tip](https://docs.julialang.org/en/v1/manual/performance-tips/#Be-aware-of-when-Julia-avoids-specializing))
  and memoization to avoid redundant computations. This is already complex, but it's even
  worse knowing that most automatic differentiation tools do not support splatting.
- The signature of gradient and hessian functions is not the same for univariate (`nZ̃ == 1`)
  and multivariate (`nZ̃ > 1`) operators in `JuMP`. Both must be defined.

Inspired from: [User-defined operators with vector outputs](https://jump.dev/JuMP.jl/stable/tutorials/nonlinear/tips_and_tricks/#User-defined-operators-with-vector-outputs)
"""
function get_optim_functions(
    estim::MovingHorizonEstimator, ::JuMP.GenericModel{JNT}
) where {JNT <: Real}
    model, con = estim.model, estim.con
    nx̂, nym, nŷ, nu, nϵ, He = estim.nx̂, estim.nym, model.ny, model.nu, estim.nϵ, estim.He
    nV̂, nX̂, ng, nZ̃ = He*nym, He*nx̂, length(con.i_g), length(estim.Z̃)
    Ncache = nZ̃ + 3
    myNaN = convert(JNT, NaN) # fill Z̃ with NaNs to force update_simulations! at 1st call:
    # ---------------------- differentiation cache ---------------------------------------
    Z̃_cache::DiffCache{Vector{JNT}, Vector{JNT}}  = DiffCache(fill(myNaN, nZ̃), Ncache)
    V̂_cache::DiffCache{Vector{JNT}, Vector{JNT}}  = DiffCache(zeros(JNT, nV̂),  Ncache)
    g_cache::DiffCache{Vector{JNT}, Vector{JNT}}  = DiffCache(zeros(JNT, ng),  Ncache)
    X̂0_cache::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, nX̂),  Ncache)
    x̄_cache::DiffCache{Vector{JNT}, Vector{JNT}}  = DiffCache(zeros(JNT, nx̂),  Ncache)
    û0_cache::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, nu),  Ncache)
    ŷ0_cache::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, nŷ),  Ncache)
    # --------------------- update simulation function ------------------------------------
    function update_simulations!(
        Z̃arg::Union{NTuple{N, T}, AbstractVector{T}}, Z̃cache
    ) where {N, T <:Real}
        if isdifferent(Z̃cache, Z̃arg)
            for i in eachindex(Z̃cache)
                # Z̃cache .= Z̃arg is type unstable with Z̃arg::NTuple{N, FowardDiff.Dual}
                Z̃cache[i] = Z̃arg[i]
            end
            Z̃ = Z̃cache
            ϵ = (nϵ ≠ 0) ? Z̃[begin] : zero(T) # ϵ = 0 if Cwt=Inf (meaning: no relaxation)
            V̂,  X̂0 = get_tmp(V̂_cache, Z̃1),  get_tmp(X̂0_cache, Z̃1)
            û0, ŷ0 = get_tmp(û0_cache, Z̃1), get_tmp(ŷ0_cache, Z̃1)
            g      = get_tmp(g_cache, Z̃1)
            V̂, X̂0  = predict!(V̂, X̂0, û0, ŷ0, estim, model, Z̃)
            g = con_nonlinprog!(g, estim, model, X̂0, V̂, ϵ)
        end
        return nothing
    end
    # --------------------- objective functions -------------------------------------------
    function Jfunc(Z̃arg::Vararg{T, N}) where {N, T<:Real}
        update_simulations!(Z̃arg, get_tmp(Z̃_cache, T))
        x̄, V̂ = get_tmp(x̄_cache, T), get_tmp(V̂_cache, T)
        return obj_nonlinprog!(x̄, estim, model, V̂, Z̃)::T
    end
    function Jfunc_vec(Z̃arg::AbstractVector{T}) where T<:Real
        update_simulations!(Z̃arg, get_tmp(Z̃_cache, T))
        x̄, V̂ = get_tmp(x̄_cache, T), get_tmp(V̂_cache, T)
        return obj_nonlinprog!(x̄, estim, model, V̂, Z̃)::T
    end
    Z̃_∇J      = fill(myNaN, nZ̃) 
    ∇J        = Vector{JNT}(undef, nZ̃)       # gradient of objective J
    ∇J_buffer = GradientBuffer(Jfunc_vec, Z̃_∇J)
    ∇Jfunc! = if nZ̃ == 1
        function (Z̃arg::T) where T<:Real 
            Z̃_∇J .= Z̃arg
            gradient!(∇J, ∇J_buffer, Z̃_∇J)
            return ∇J[begin]    # univariate syntax, see JuMP.@operator doc
        end
    else
        function (∇J::AbstractVector{T}, Z̃arg::Vararg{T, N}) where {N, T<:Real}
            Z̃_∇J .= Z̃arg
            gradient!(∇J, ∇J_buffer, Z̃_∇J)
            return ∇J           # multivariate syntax, see JuMP.@operator doc
        end
    end
    # --------------------- inequality constraint functions -------------------------------
    gfuncs = Vector{Function}(undef, ng)
    for i in eachindex(gfuncs)
        func_i = function (Z̃arg::Vararg{T, N}) where {N, T<:Real}
            update_simulations!(Z̃arg, get_tmp(Z̃_cache, T))
            g = get_tmp(g_cache, T)
            return g[i]::T
        end
        gfuncs[i] = func_i
    end
    function gfunc_vec!(g, Z̃vec::AbstractVector{T}) where T<:Real
        update_simulations!(Z̃vec, get_tmp(Z̃_cache, T))
        g .= get_tmp(g_cache, T)
        return g
    end
    Z̃_∇g      = fill(myNaN, nZ̃)
    g_vec     = Vector{JNT}(undef, ng)
    ∇g        = Matrix{JNT}(undef, ng, nZ̃)   # Jacobian of inequality constraints g
    ∇g_buffer = JacobianBuffer(gfunc_vec!, g_vec, Z̃_∇g)
    ∇gfuncs!  = Vector{Function}(undef, ng)
    for i in eachindex(∇gfuncs!)
        ∇gfuncs![i] = if nZ̃ == 1
            function (Z̃arg::T) where T<:Real
                if isdifferent(Z̃arg, Z̃_∇g)
                    Z̃_∇g .= Z̃arg
                    jacobian!(∇g, ∇g_buffer, g_vec, Z̃_∇g)
                end
                return ∇g[i, begin]            # univariate syntax, see JuMP.@operator doc
            end
        else
            function (∇g_i, Z̃arg::Vararg{T, N}) where {N, T<:Real}
                if isdifferent(Z̃arg, Z̃_∇g)
                    Z̃_∇g .= Z̃arg
                    jacobian!(∇g, ∇g_buffer, g_vec, Z̃_∇g)
                end
                return ∇g_i .= @views ∇g[i, :] # multivariate syntax, see JuMP.@operator doc
            end
        end
    end
    return Jfunc, ∇Jfunc!, gfuncs, ∇gfuncs!
end