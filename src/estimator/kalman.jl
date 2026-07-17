"Abstract supertype of all Kalman-type state estimators."
abstract type KalmanEstimator{NT<:Real} <: StateEstimator{NT} end

struct SteadyKalmanFilter{
    NT<:Real, 
    SM<:LinModel, 
    KC<:KalmanCovariances
} <: KalmanEstimator{NT}
    model::SM
    cov  ::KC
    x̂op ::Vector{NT}
    f̂op ::Vector{NT}
    x̂0  ::Vector{NT}
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
    K̂::Matrix{NT}
    direct::Bool
    corrected::Vector{Bool}
    buffer::StateEstimatorBuffer{NT}
    function SteadyKalmanFilter{NT}(
        model::SM, i_ym, nint_u, nint_ym, cov::KC; direct=true
    ) where {NT<:Real, SM<:LinModel, KC<:KalmanCovariances}
        nu, ny, nd, nk = model.nu, model.ny, model.nd, model.nk
        nym, nyu = validate_ym(model, i_ym)
        As, Cs_u, Cs_y, nint_u, nint_ym = init_estimstoch(model, i_ym, nint_u, nint_ym)
        nxs = size(As, 1)
        nx̂  = model.nx + nxs
        Â, B̂u, Ĉ, B̂d, D̂d, x̂op, f̂op = augment_model(model, As, Cs_u, Cs_y)
        Ĉm, D̂dm = Ĉ[i_ym, :], D̂d[i_ym, :]
        R̂, Q̂ = cov.R̂, cov.Q̂
        K̂, P̂ = init_skf(i_ym, Â, Ĉ, Q̂, R̂; direct)
        cov.P̂ .= P̂
        x̂0 = [zeros(NT, model.nx); zeros(NT, nxs)]
        corrected = [false]
        buffer = StateEstimatorBuffer{NT}(nu, nx̂, nym, ny, nd, nk)
        return new{NT, SM, KC}(
            model,
            cov,
            x̂op, f̂op, x̂0, 
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs_u, Cs_y, nint_u, nint_ym,
            Â, B̂u, Ĉ, B̂d, D̂d, Ĉm, D̂dm,
            K̂,
            direct, corrected,
            buffer
        )
    end
end

@doc raw"""
    SteadyKalmanFilter(model::LinModel; <keyword arguments>)

Construct a steady-state Kalman Filter with the [`LinModel`](@ref) `model`.

The steady-state (or [asymptotic](https://en.wikipedia.org/wiki/Kalman_filter#Asymptotic_form))
Kalman filter is based on the process model:
```math
\begin{aligned}
    \mathbf{x}(k+1) &= 
            \mathbf{Â x}(k) + \mathbf{B̂_u u}(k) + \mathbf{B̂_d d}(k) + \mathbf{w}(k) \\
    \mathbf{y^m}(k) &= \mathbf{Ĉ^m x}(k) + \mathbf{D̂_d^m d}(k) + \mathbf{v}(k) \\
    \mathbf{y^u}(k) &= \mathbf{Ĉ^u x}(k) + \mathbf{D̂_d^u d}(k)
\end{aligned}
```
with sensor ``\mathbf{v}(k)`` and process ``\mathbf{w}(k)`` noises as uncorrelated zero mean 
white noise vectors, with a respective covariance of ``\mathbf{R̂}`` and ``\mathbf{Q̂}``. 
The arguments are in standard deviations σ, i.e. same units than outputs and states. The 
matrices ``\mathbf{Â, B̂_u, B̂_d, Ĉ, D̂_d}`` are `model` matrices augmented with the stochastic
model, which is specified by the numbers of integrator `nint_u` and `nint_ym` (see Extended
Help). Likewise, the covariance matrices are augmented with ``\mathbf{Q̂ = \text{diag}(Q, 
Q_{int_u}, Q_{int_{ym}})}`` and ``\mathbf{R̂ = R}``. The Extended Help provide some guidelines
on the covariance tuning. The matrices ``\mathbf{Ĉ^m, D̂_d^m}`` are the rows of 
``\mathbf{Ĉ, D̂_d}`` that correspond to measured outputs ``\mathbf{y^m}`` (and unmeasured
ones, for ``\mathbf{Ĉ^u, D̂_d^u}``). The Kalman filter will estimate the current state with 
the newest measurements ``\mathbf{x̂}_k(k)`` if `direct` is `true`, else it will predict the
state of the next time step ``\mathbf{x̂}_k(k+1)``. This estimator is allocation-free.

# Arguments
!!! info
    Keyword arguments with *`emphasis`* are non-Unicode alternatives.

- `model::LinModel` : (deterministic) model for the estimations.
- `i_ym=1:model.ny` : `model` output indices that are measured ``\mathbf{y^m}``, the rest 
    are unmeasured ``\mathbf{y^u}``.
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
- `σQint_ym=fill(1,sum(nint_ym))` or *`sigmaQint_u`* : same than `σQ` for the unmeasured
    disturbances at measured outputs ``\mathbf{Q_{int_{ym}}}`` (composed of integrators).
- `direct=true`: construct with a direct transmission from ``\mathbf{y^m}`` (a.k.a. current
   estimator, in opposition to the delayed/predictor form).

# Examples
```jldoctest
julia> model = LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 0.5);

julia> estim = SteadyKalmanFilter(model, i_ym=[2], σR=[1], σQint_ym=[0.01])
SteadyKalmanFilter estimator with a sample time Ts = 0.5 s:
├ model: LinModel
├ direct: true
└ dimensions:
  ├ 1 manipulated inputs u (0 integrating states)
  ├ 3 estimated states x̂
  ├ 1 measured outputs ym (1 integrating states)
  ├ 1 unmeasured outputs yu
  └ 0 measured disturbances d
```

# Extended Help
!!! details "Extended Help"
    The `σR` argument is generally fixed at the estimated standard deviations of the sensor
    noises. The `σQ`, `σQint_u` and `σQint_ym` arguments can be used to tune the filter
    response. Increasing them make the filter more responsive to disturbances but more
    sensitive to measurement noise.

    The model augmentation with `nint_u` vector adds integrators at model manipulated inputs,
    and `nint_ym`, at measured outputs. They create the integral action when the estimator
    is used in a controller as state feedback. By default, the method [`default_nint`](@ref)
    adds one integrator per measured output if feasible. The argument `nint_ym` can also be
    tweaked by following these rules on each measured output:

    - Use 0 integrator if the model output is already integrating (else it will be unobservable)
    - Use 1 integrator if the disturbances on the output are typically "step-like"
    - Use 2 integrators if the disturbances on the output are typically "ramp-like" 

    The function [`init_estimstoch`](@ref) builds the stochastic model for estimation.

    !!! tip
        Increasing `σQint_u` and `σQint_ym` values increases the integral action "gain".

    Custom stochastic model for the unmeasured disturbances (different than integrated white
    gaussian noise) can be specified by constructing a [`LinModel`](@ref) object with the
    augmented state-space matrices directly, and by setting `nint_u=0` and `nint_ym=0`. See
    [Disturbance-gallery](@extref LowLevelParticleFilters) for examples of other
    disturbance models.
    
    The constructor pre-compute the steady-state Kalman gain `K̂` with the [`kalman`](@extref ControlSystemsBase.kalman)
    function. It can sometimes fail, for example when `model` matrices are ill-conditioned.
    In such a case, you can try the alternative time-varying [`KalmanFilter`](@ref).
"""
function SteadyKalmanFilter(
    model::SM;
    i_ym::AbstractVector{Int} = 1:model.ny,
    sigmaQ = fill(1/model.nx, model.nx),
    sigmaR = fill(1, length(i_ym)),
    nint_u ::IntVectorOrInt = 0,
    nint_ym::IntVectorOrInt = default_nint(model, i_ym, nint_u),
    sigmaQint_u  = fill(1, max(sum(nint_u),  0)),
    sigmaQint_ym = fill(1, max(sum(nint_ym), 0)),
    direct   = true,
    σQ       = sigmaQ,
    σR       = sigmaR,
    σQint_u  = sigmaQint_u,
    σQint_ym = sigmaQint_ym,
) where {NT<:Real, SM<:LinModel{NT}}
    # estimated covariances matrices (variance = σ²) :
    Q̂ = Diagonal([σQ;  σQint_u;  σQint_ym].^2)
    R̂ = Diagonal([σR;].^2)
    return SteadyKalmanFilter(model, i_ym, nint_u, nint_ym, Q̂, R̂; direct)
end

@doc raw"""
    SteadyKalmanFilter(model, i_ym, nint_u, nint_ym, Q̂, R̂; direct=true)

Construct the estimator from the augmented covariance matrices `Q̂` and `R̂`.

This syntax allows nonzero off-diagonal elements in ``\mathbf{Q̂, R̂}``.
"""
function SteadyKalmanFilter(
    model::SM, i_ym, nint_u, nint_ym, Q̂, R̂; direct=true
) where {NT<:Real, SM<:LinModel{NT}}
    Q̂, R̂ = to_mat(Q̂), to_mat(R̂)
    cov = KalmanCovariances(model, i_ym, nint_u, nint_ym, Q̂, R̂)
    return SteadyKalmanFilter{NT}(model, i_ym, nint_u, nint_ym, cov; direct)
end

"""
    init_skf(i_ym, Â, Ĉ, Q̂, R̂; direct=true) -> K̂, P̂

Initialize the steady-state Kalman gain `K̂` and estimation error covariance `P̂`.
"""
function init_skf(i_ym, Â, Ĉ, Q̂, R̂; direct=true)
    ny, nym = size(Ĉ, 1), length(i_ym)
    if ny != nym
        R̂_y = zeros(eltype(R̂), ny, ny)
        R̂_y[i_ym, i_ym] = R̂
        R̂ = Hermitian(R̂_y, :L)
    end
    K̂, P̂ = try 
        ControlSystemsBase.kalman(Discrete, Â, Ĉ, Q̂, R̂; direct, extra=Val(true))
    catch my_error
        if isa(my_error, ErrorException)
            error("Cannot compute the optimal Kalman gain K̂ for the "* 
                  "SteadyKalmanFilter. You may try to remove integrators with "*
                  "nint_u/nint_ym parameter or use the time-varying KalmanFilter.")
        else
            rethrow()
        end
    end
    if ny != nym 
        K̂ = K̂[:, i_ym]
    end
    P̂ = Hermitian(P̂, :L)
    return K̂, P̂
end

"Throw an error if `setmodel!` is called on a SteadyKalmanFilter w/o the default values."
function setmodel_estimator!(estim::SteadyKalmanFilter, model, _ , _ , _ , Q̂, R̂)
    if estim.model !== model || !isnothing(Q̂) || !isnothing(R̂)
        error("SteadyKalmanFilter does not support setmodel! (use KalmanFilter instead)")
    end
    return nothing
end

struct KalmanFilter{
    NT<:Real, 
    SM<:LinModel,
    KC<:KalmanCovariances
} <: KalmanEstimator{NT}
    model::SM
    cov  ::KC
    x̂op::Vector{NT}
    f̂op::Vector{NT}
    x̂0 ::Vector{NT}
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
    K̂::Matrix{NT}
    direct::Bool
    corrected::Vector{Bool}
    buffer::StateEstimatorBuffer{NT}
    function KalmanFilter{NT}(
        model::SM, i_ym, nint_u, nint_ym, cov::KC; direct=true
    ) where {NT<:Real, SM<:LinModel, KC<:KalmanCovariances}
        nu, ny, nd, nk = model.nu, model.ny, model.nd, model.nk
        nym, nyu = validate_ym(model, i_ym)
        As, Cs_u, Cs_y, nint_u, nint_ym = init_estimstoch(model, i_ym, nint_u, nint_ym)
        nxs = size(As, 1)
        nx̂  = model.nx + nxs
        Â, B̂u, Ĉ, B̂d, D̂d, x̂op, f̂op = augment_model(model, As, Cs_u, Cs_y)
        Ĉm, D̂dm = Ĉ[i_ym, :], D̂d[i_ym, :]
        x̂0  = [zeros(NT, model.nx); zeros(NT, nxs)]
        K̂ = zeros(NT, nx̂, nym)
        corrected = [false]
        buffer = StateEstimatorBuffer{NT}(nu, nx̂, nym, ny, nd, nk)
        return new{NT, SM, KC}(
            model, 
            cov,
            x̂op, f̂op, x̂0,
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs_u, Cs_y, nint_u, nint_ym,
            Â, B̂u, Ĉ, B̂d, D̂d, Ĉm, D̂dm,
            K̂,
            direct, corrected,
            buffer
        )
    end
end

@doc raw"""
    KalmanFilter(model::LinModel; <keyword arguments>)

Construct a time-varying Kalman Filter with the [`LinModel`](@ref) `model`.

The process model is identical to [`SteadyKalmanFilter`](@ref). The matrix ``\mathbf{P̂}`` is
the estimation error covariance of `model` states augmented with the stochastic ones
(specified by `nint_u` and `nint_ym`). Three keyword arguments specify its initial value with
``\mathbf{P̂}_{-1}(0) = \mathrm{diag}\{ \mathbf{P}(0), \mathbf{P_{int_{u}}}(0), 
\mathbf{P_{int_{ym}}}(0) \}``. The initial state estimate ``\mathbf{x̂}_{-1}(0)`` can be
manually specified with [`setstate!`](@ref), or automatically with [`initstate!`](@ref).
This estimator is allocation-free.

# Arguments
!!! info
    Keyword arguments with *`emphasis`* are non-Unicode alternatives.

- `model::LinModel` : (deterministic) model for the estimations.
- `i_ym=1:model.ny` : `model` output indices that are measured ``\mathbf{y^m}``, the rest 
    are unmeasured ``\mathbf{y^u}``.
- `σP_0=fill(1/model.nx,model.nx)` or *`sigmaP_0`* : main diagonal of the initial estimate
    covariance ``\mathbf{P}(0)``, specified as a standard deviation vector.
- `σQ=fill(1/model.nx,model.nx)` or *`sigmaQ`* : main diagonal of the process noise
    covariance ``\mathbf{Q}`` of `model`, specified as a standard deviation vector.
- `σR=fill(1,length(i_ym))` or *`sigmaR`* : main diagonal of the sensor noise covariance
    ``\mathbf{R}`` of `model` measured outputs, specified as a standard deviation vector.
- `nint_u=0`: integrator quantity for the stochastic model of the unmeasured disturbances at
    the manipulated inputs (vector), use `nint_u=0` for no integrator.
- `nint_ym=default_nint(model,i_ym,nint_u)` : same than `nint_u` but for the unmeasured 
    disturbances at the measured outputs, use `nint_ym=0` for no integrator.
- `σQint_u=fill(1,sum(nint_u))` or *`sigmaQint_u`* : same than `σQ` but for the unmeasured
    disturbances at manipulated inputs ``\mathbf{Q_{int_u}}`` (composed of integrators).
- `σPint_u_0=fill(1,sum(nint_u))` or *`sigmaPint_u_0`* : same than `σP_0` but for the unmeasured
    disturbances at manipulated inputs ``\mathbf{P_{int_u}}(0)`` (composed of integrators).
- `σQint_ym=fill(1,sum(nint_ym))` or *`sigmaQint_u`* : same than `σQ` for the unmeasured
    disturbances at measured outputs ``\mathbf{Q_{int_{ym}}}`` (composed of integrators).
- `σPint_ym_0=fill(1,sum(nint_ym))` or *`sigmaPint_ym_0`* : same than `σP_0` but for the unmeasured
    disturbances at measured outputs ``\mathbf{P_{int_{ym}}}(0)`` (composed of integrators).
- `direct=true`: construct with a direct transmission from ``\mathbf{y^m}`` (a.k.a. current
   estimator, in opposition to the delayed/predictor form).

# Examples
```jldoctest
julia> model = LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 0.5);

julia> estim = KalmanFilter(model, i_ym=[2], σR=[1], σP_0=[100, 100], σQint_ym=[0.01])
KalmanFilter estimator with a sample time Ts = 0.5 s:
├ model: LinModel
├ direct: true
└ dimensions:
  ├ 1 manipulated inputs u (0 integrating states)
  ├ 3 estimated states x̂
  ├ 1 measured outputs ym (1 integrating states)
  ├ 1 unmeasured outputs yu
  └ 0 measured disturbances d
```
"""
function KalmanFilter(
    model::SM;
    i_ym::AbstractVector{Int} = 1:model.ny,
    sigmaP_0 = fill(1/model.nx, model.nx),
    sigmaQ   = fill(1/model.nx, model.nx),
    sigmaR   = fill(1, length(i_ym)),
    nint_u ::IntVectorOrInt = 0,
    nint_ym::IntVectorOrInt = default_nint(model, i_ym, nint_u),
    sigmaPint_u_0  = fill(1, max(sum(nint_u),  0)),
    sigmaQint_u    = fill(1, max(sum(nint_u),  0)),
    sigmaPint_ym_0 = fill(1, max(sum(nint_ym), 0)),
    sigmaQint_ym   = fill(1, max(sum(nint_ym), 0)),
    direct = true,
    σP_0       = sigmaP_0,
    σQ         = sigmaQ,
    σR         = sigmaR,
    σPint_u_0  = sigmaPint_u_0,
    σQint_u    = sigmaQint_u,
    σPint_ym_0 = sigmaPint_ym_0,
    σQint_ym   = sigmaQint_ym,
) where {NT<:Real, SM<:LinModel{NT}}
    # estimated covariances matrices (variance = σ²) :
    P̂_0 = Diagonal([σP_0; σPint_u_0; σPint_ym_0].^2)
    Q̂   = Diagonal([σQ;  σQint_u;  σQint_ym ].^2)
    R̂   = Diagonal([σR;].^2)
    return KalmanFilter(model, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂; direct)
end

@doc raw"""
    KalmanFilter(model, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂; direct=true)

Construct the estimator from the augmented covariance matrices `P̂_0`, `Q̂` and `R̂`.

This syntax allows nonzero off-diagonal elements in ``\mathbf{P̂}_{-1}(0), \mathbf{Q̂, R̂}``.
"""
function KalmanFilter(
    model::SM, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂; direct=true
) where {NT<:Real, SM<:LinModel{NT}}
    P̂_0, Q̂, R̂ = to_mat(P̂_0), to_mat(Q̂), to_mat(R̂)
    cov = KalmanCovariances(model, i_ym, nint_u, nint_ym, Q̂, R̂, P̂_0)
    return KalmanFilter{NT}(model, i_ym, nint_u, nint_ym, cov; direct)
end

#=@doc raw"""
    correct_estimate!(estim::KalmanFilter, y0m, d0)

Correct `estim.x̂0` and `estim.cov.P̂` using the time-varying [`KalmanFilter`](@ref).

It computes the corrected state estimate ``\mathbf{x̂}_{k}(k)`` estimation covariance 
``\mathbf{P̂}_{k}(k)``.
"""=#

@doc raw"""
    correct_estimate!(estim::KalmanFilter, y0m, d0)

Correct [`KalmanFilter`](@ref) state `estim.x̂0` and estimation error covariance `estim.cov.P̂`.

It implements the classical time-varying Kalman Filter based on the process model described
in [`SteadyKalmanFilter`](@ref). The correction and prediction step equations are provided
below, see [^2] for details.

# Correction Step
```math
\begin{aligned}
    \mathbf{M̂}(k)     &= \mathbf{Ĉ^m P̂}_{k-1}(k)\mathbf{Ĉ^m}' + \mathbf{R̂}                        \\
    \mathbf{K̂}(k)     &= \mathbf{P̂}_{k-1}(k)\mathbf{Ĉ^m}'\mathbf{M̂^{-1}}(k)                       \\
    \mathbf{ŷ^m}(k)   &= \mathbf{Ĉ^m x̂}_{k-1}(k) + \mathbf{D̂_d^m d}(k)                            \\
    \mathbf{x̂}_{k}(k) &= \mathbf{x̂}_{k-1}(k) + \mathbf{K̂}(k)[\mathbf{y^m}(k) - \mathbf{ŷ^m}(k)]   \\
    \mathbf{P̂}_{k}(k) &= [\mathbf{I - K̂}(k)\mathbf{Ĉ^m}]\mathbf{P̂}_{k-1}(k)
\end{aligned}
```

# Prediction Step
```math
\begin{aligned}
    \mathbf{x̂}_{k}(k+1) &= \mathbf{Â x̂}_{k}(k) + \mathbf{B̂_u u}(k) + \mathbf{B̂_d d}(k)      \\
    \mathbf{P̂}_{k}(k+1) &= \mathbf{Â P̂}_{k}(k)\mathbf{Â}' + \mathbf{Q̂}
\end{aligned}
```

[^2]: "Kalman Filter", *Wikipedia: The Free Encyclopedia*, 
     <https://en.wikipedia.org/wiki/Kalman_filter>, Accessed 2024-08-08.
"""
correct_estimate!(estim::KalmanFilter, y0m, d0) = correct_kf!(estim, y0m, d0, estim.Ĉm)

"Prediction step of [`KalmanFilter`](@ref), see equations in [`correct_estimate!`](@ref)."
predict_estimate!(estim::KalmanFilter, u0, d0) = predict_kf!(estim, u0, d0, estim.Â)

struct UnscentedKalmanFilter{
    NT<:Real, 
    SM<:SimModel,
    KC<:KalmanCovariances
} <: KalmanEstimator{NT}
    model::SM
    cov  ::KC
    x̂op ::Vector{NT}
    f̂op ::Vector{NT}
    x̂0  ::Vector{NT}
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
    K̂::Matrix{NT}
    M̂::Hermitian{NT, Matrix{NT}}
    X̂0::Matrix{NT}
    X̄0::Matrix{NT}
    Ŷ0m::Matrix{NT}
    Ȳ0m::Matrix{NT}
    nσ::Int 
    γ::NT
    m̂::Vector{NT}
    Ŝ::Diagonal{NT, Vector{NT}}
    direct::Bool
    corrected::Vector{Bool}
    buffer::StateEstimatorBuffer{NT}
    function UnscentedKalmanFilter{NT}(
        model::SM, i_ym, nint_u, nint_ym, cov::KC, α, β, κ; direct=true
    ) where {NT<:Real, SM<:SimModel{NT}, KC<:KalmanCovariances}
        nu, ny, nd, nk = model.nu, model.ny, model.nd, model.nk
        nym, nyu = validate_ym(model, i_ym)
        As, Cs_u, Cs_y, nint_u, nint_ym = init_estimstoch(model, i_ym, nint_u, nint_ym)
        nxs = size(As, 1)
        nx̂  = model.nx + nxs
        Â, B̂u, Ĉ, B̂d, D̂d, x̂op, f̂op = augment_model(model, As, Cs_u, Cs_y)
        Ĉm, D̂dm = Ĉ[i_ym, :], D̂d[i_ym, :]
        nσ, γ, m̂, Ŝ = init_ukf(nx̂, α, β, κ)
        x̂0  = [zeros(NT, model.nx); zeros(NT, nxs)]
        K̂ = zeros(NT, nx̂, nym)
        M̂ = Hermitian(zeros(NT, nym, nym), :L)
        X̂0,  X̄0  = zeros(NT, nx̂, nσ),  zeros(NT, nx̂, nσ)
        Ŷ0m, Ȳ0m = zeros(NT, nym, nσ), zeros(NT, nym, nσ)
        corrected = [false]
        buffer = StateEstimatorBuffer{NT}(nu, nx̂, nym, ny, nd, nk)
        return new{NT, SM, KC}(
            model,
            cov,
            x̂op, f̂op, x̂0, 
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs_u, Cs_y, nint_u, nint_ym,
            Â, B̂u, Ĉ, B̂d, D̂d, Ĉm, D̂dm,
            K̂, 
            M̂, X̂0, X̄0, Ŷ0m, Ȳ0m,
            nσ, γ, m̂, Ŝ,
            direct, corrected,
            buffer
        )
    end
end

@doc raw"""
    UnscentedKalmanFilter(model::SimModel; <keyword arguments>)

Construct an unscented Kalman Filter with the [`SimModel`](@ref) `model`.

Both [`LinModel`](@ref) and [`NonLinModel`](@ref) are supported. The unscented Kalman filter
is based on the process model :
```math
\begin{aligned}
    \mathbf{x}(k+1) &= \mathbf{f̂}\Big(\mathbf{x}(k), \mathbf{u}(k), \mathbf{d}(k)\Big) 
                        + \mathbf{w}(k)                                                   \\
    \mathbf{y^m}(k) &= \mathbf{ĥ^m}\Big(\mathbf{x}(k), \mathbf{d}(k)\Big) + \mathbf{v}(k) \\
    \mathbf{y^u}(k) &= \mathbf{ĥ^u}\Big(\mathbf{x}(k), \mathbf{d}(k)\Big)                 \\
\end{aligned}
```
See [`SteadyKalmanFilter`](@ref) for details on ``\mathbf{v}(k), \mathbf{w}(k)`` noises and
``\mathbf{R̂}, \mathbf{Q̂}`` covariances. The two matrices are constructed from ``\mathbf{Q̂ =
\text{diag}(Q, Q_{int_u}, Q_{int_{ym}})}`` and ``\mathbf{R̂ = R}``. The functions
``\mathbf{f̂, ĥ}`` are `model` state-space functions augmented with the stochastic model of
the unmeasured disturbances, which is specified by the numbers of integrator `nint_u` and
`nint_ym` (see Extended Help). Model parameters ``\mathbf{p}`` are not argument of
``\mathbf{f̂, ĥ}`` functions for conciseness. The ``\mathbf{ĥ^m}`` function represents the
measured outputs of ``\mathbf{ĥ}`` function (and unmeasured ones, for ``\mathbf{ĥ^u}``). The
matrix ``\mathbf{P̂}`` is the estimation error covariance of `model` state augmented with the 
stochastic ones. Three keyword arguments specify its initial value with ``\mathbf{P̂}_{-1}(0) = 
\mathrm{diag}\{ \mathbf{P}(0), \mathbf{P_{int_{u}}}(0), \mathbf{P_{int_{ym}}}(0) \}``. The 
initial state estimate ``\mathbf{x̂}_{-1}(0)`` can be manually specified with [`setstate!`](@ref).
This estimator is allocation-free if `model` simulations do not allocate.

# Arguments
!!! info
    Keyword arguments with *`emphasis`* are non-Unicode alternatives.

- `model::SimModel` : (deterministic) model for the estimations.
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
- `α=1e-3` or *`alpha`* : alpha parameter, spread of the state distribution ``(0 < α ≤ 1)``.
- `β=2` or *`beta`* : beta parameter, skewness and kurtosis of the states distribution ``(β ≥ 0)``.
- `κ=0` or *`kappa`* : kappa parameter, another spread parameter ``(0 ≤ κ ≤ 3)``.
- `direct=true`: construct with a direct transmission from ``\mathbf{y^m}`` (a.k.a. current
   estimator, in opposition to the delayed/predictor form).

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_,_)->0.1x+u, (x,_,_)->2x, 10.0, 1, 1, 1, solver=nothing);

julia> estim = UnscentedKalmanFilter(model, σR=[1], nint_ym=[2], σPint_ym_0=[1, 1])
UnscentedKalmanFilter estimator with a sample time Ts = 10.0 s:
├ model: NonLinModel
├ direct: true
└ dimensions:
  ├ 1 manipulated inputs u (0 integrating states)
  ├ 3 estimated states x̂
  ├ 1 measured outputs ym (2 integrating states)
  ├ 0 unmeasured outputs yu
  └ 0 measured disturbances d
```

# Extended Help
!!! details "Extended Help"
    The Extended Help of [`SteadyKalmanFilter`](@ref) details the tuning of the covariances
    and the augmentation with `nint_ym` and `nint_u` arguments. The default augmentation
    scheme is identical, that is `nint_u=0` and `nint_ym` computed by [`default_nint`](@ref).
    Note that the constructor does not validate the observability of the resulting augmented
    [`NonLinModel`](@ref). In such cases, it is the user's responsibility to ensure that it
    is still observable.
"""
function UnscentedKalmanFilter(
    model::SM;
    i_ym::AbstractVector{Int} = 1:model.ny,
    sigmaP_0 = fill(1/model.nx, model.nx),
    sigmaQ   = fill(1/model.nx, model.nx),
    sigmaR   = fill(1, length(i_ym)),
    nint_u ::IntVectorOrInt = 0,
    nint_ym::IntVectorOrInt = default_nint(model, i_ym, nint_u),
    sigmaPint_u_0  = fill(1, max(sum(nint_u),  0)),
    sigmaQint_u    = fill(1, max(sum(nint_u),  0)),
    sigmaPint_ym_0 = fill(1, max(sum(nint_ym), 0)),
    sigmaQint_ym   = fill(1, max(sum(nint_ym), 0)),
    alpha::Real = 1e-3,
    beta ::Real = 2,
    kappa::Real = 0,
    direct = true,
    σP_0       = sigmaP_0,
    σQ         = sigmaQ,
    σR         = sigmaR,
    σPint_u_0  = sigmaPint_u_0,
    σQint_u    = sigmaQint_u,
    σPint_ym_0 = sigmaPint_ym_0,
    σQint_ym   = sigmaQint_ym,
    α = alpha,
    β = beta,
    κ = kappa,
) where {NT<:Real, SM<:SimModel{NT}}
    # estimated covariances matrices (variance = σ²) :
    P̂_0 = Diagonal([σP_0; σPint_u_0; σPint_ym_0].^2)
    Q̂   = Diagonal([σQ;  σQint_u;  σQint_ym ].^2)
    R̂   = Diagonal([σR;].^2)
    return UnscentedKalmanFilter(model, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂, α, β, κ; direct)
end

@doc raw"""
    UnscentedKalmanFilter(
        model, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂, α=1e-3, β=2, κ=0; direct=true
    )

Construct the estimator from the augmented covariance matrices `P̂_0`, `Q̂` and `R̂`.

This syntax allows nonzero off-diagonal elements in ``\mathbf{P̂}_{-1}(0), \mathbf{Q̂, R̂}``.
"""
function UnscentedKalmanFilter(
    model::SM, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂, α=1e-3, β=2, κ=0; direct=true
) where {NT<:Real, SM<:SimModel{NT}}
    P̂_0, Q̂, R̂ = to_mat(P̂_0), to_mat(Q̂), to_mat(R̂)
    cov = KalmanCovariances(model, i_ym, nint_u, nint_ym, Q̂, R̂, P̂_0)
    return UnscentedKalmanFilter{NT}(model, i_ym, nint_u, nint_ym, cov, α, β, κ; direct)
end


@doc raw"""
    init_ukf(nx̂, α, β, κ) -> nσ, γ, m̂, Ŝ

Compute the [`UnscentedKalmanFilter`](@ref) constants from ``α, β`` and ``κ``.

With ``n_\mathbf{x̂}`` elements in the state vector ``\mathbf{x̂}`` and 
``n_σ = 2 n_\mathbf{x̂} + 1`` sigma points, the scaling factor applied on standard deviation 
matrices ``\sqrt{\mathbf{P̂}}`` is:
```math
    γ = α \sqrt{ n_\mathbf{x̂} + κ }
```
The weight vector ``(n_σ × 1)`` for the mean and the weight matrix ``(n_σ × n_σ)`` for the 
covariance are respectively:
```math
\begin{aligned}
    \mathbf{m̂} &= \begin{bmatrix} 1 - \tfrac{n_\mathbf{x̂}}{γ^2} & \tfrac{1}{2γ^2} & \tfrac{1}{2γ^2} & \cdots & \tfrac{1}{2γ^2} \end{bmatrix}' \\
    \mathbf{Ŝ} &= \mathrm{diag}\big( 2 - α^2 + β - \tfrac{n_\mathbf{x̂}}{γ^2} \:,\; \tfrac{1}{2γ^2} \:,\; \tfrac{1}{2γ^2} \:,\; \cdots \:,\; \tfrac{1}{2γ^2} \big)
\end{aligned}
```
See [`correct_estimate!`](@ref) for other details.
"""
function init_ukf(nx̂, α, β, κ)
    α, β, κ = promote(α, β, κ)
    nσ =2nx̂ + 1                                 # number of sigma points
    γ = α * √(nx̂ + κ)                           # constant factor of standard deviation √P
    m̂_0 = 1 - nx̂ / γ^2
    Ŝ_0 = m̂_0 + 1 - α^2 + β
    w = 1 / 2 / γ^2
    m̂ = [m̂_0; fill(w, 2 * nx̂)]                  # weights for the mean
    Ŝ = Diagonal([Ŝ_0; fill(w, 2 * nx̂)])        # weights for the covariance
    return nσ, γ, m̂, Ŝ
end

@doc raw"""
    correct_estimate!(estim::UnscentedKalmanFilter, y0m, d0)
    
Correct [`UnscentedKalmanFilter`](@ref) state `estim.x̂0` and covariance estimate `estim.cov.P̂`.

It implements the unscented Kalman Filter based on the generalized unscented transform[^3].
See [`init_ukf`](@ref) for the definition of the constants ``\mathbf{m̂, Ŝ}`` and ``γ``. The
superscript in e.g. ``\mathbf{X̂}_{k-1}^j(k)`` refers the vector at the ``j``th column of 
``\mathbf{X̂}_{k-1}(k)``. The symbol ``\mathbf{0}`` is a vector with zeros. The number of
sigma points is ``n_σ = 2 n_\mathbf{x̂} + 1``. The matrices ``\sqrt{\mathbf{P̂}_{k-1}(k)}``
and ``\sqrt{\mathbf{P̂}_{k}(k)}`` are the the lower triangular factors of [`cholesky`](@extref Julia LinearAlgebra.cholesky)
results. The correction and prediction step equations are provided below.

# Correction Step
```math
\begin{aligned}
    \mathbf{X̂}_{k-1}(k) &= \bigg[\begin{matrix} \mathbf{x̂}_{k-1}(k) & \mathbf{x̂}_{k-1}(k) & \cdots & \mathbf{x̂}_{k-1}(k)  \end{matrix}\bigg] + \bigg[\begin{matrix} \mathbf{0} & γ \sqrt{\mathbf{P̂}_{k-1}(k)} & -γ \sqrt{\mathbf{P̂}_{k-1}(k)} \end{matrix}\bigg] \\
    \mathbf{Ŷ^m}(k)     &= \bigg[\begin{matrix} \mathbf{ĥ^m}\Big( \mathbf{X̂}_{k-1}^{1}(k) \Big) & \mathbf{ĥ^m}\Big( \mathbf{X̂}_{k-1}^{2}(k) \Big) & \cdots & \mathbf{ĥ^m}\Big( \mathbf{X̂}_{k-1}^{n_σ}(k) \Big) \end{matrix}\bigg] \\
    \mathbf{ŷ^m}(k)     &= \mathbf{Ŷ^m}(k) \mathbf{m̂} \\
    \mathbf{X̄}_{k-1}(k) &= \begin{bmatrix} \mathbf{X̂}_{k-1}^{1}(k) - \mathbf{x̂}_{k-1}(k) & \mathbf{X̂}_{k-1}^{2}(k) - \mathbf{x̂}_{k-1}(k) & \cdots & \mathbf{X̂}_{k-1}^{n_σ}(k) - \mathbf{x̂}_{k-1}(k) \end{bmatrix} \\
    \mathbf{Ȳ^m}(k)     &= \begin{bmatrix} \mathbf{Ŷ^m}^{1}(k)     - \mathbf{ŷ^m}(k)     & \mathbf{Ŷ^m}^{2}(k)     - \mathbf{ŷ^m}(k)     & \cdots & \mathbf{Ŷ^m}^{n_σ}(k)     - \mathbf{ŷ^m}(k)     \end{bmatrix} \\
    \mathbf{M̂}(k)       &= \mathbf{Ȳ^m}(k) \mathbf{Ŝ} \mathbf{Ȳ^m}'(k) + \mathbf{R̂} \\
    \mathbf{K̂}(k)       &= \mathbf{X̄}_{k-1}(k) \mathbf{Ŝ} \mathbf{Ȳ^m}'(k) \mathbf{M̂^{-1}}(k) \\
    \mathbf{x̂}_k(k)     &= \mathbf{x̂}_{k-1}(k) + \mathbf{K̂}(k) \big[ \mathbf{y^m}(k) - \mathbf{ŷ^m}(k) \big] \\
    \mathbf{P̂}_k(k)     &= \mathbf{P̂}_{k-1}(k) - \mathbf{K̂}(k) \mathbf{M̂}(k) \mathbf{K̂}'(k) \\
\end{aligned} 
```

# Prediction Step
```math
\begin{aligned}
    \mathbf{X̂}_k(k)     &= \bigg[\begin{matrix} \mathbf{x̂}_{k}(k) & \mathbf{x̂}_{k}(k) & \cdots & \mathbf{x̂}_{k}(k) \end{matrix}\bigg] + \bigg[\begin{matrix} \mathbf{0} & \gamma \sqrt{\mathbf{P̂}_{k}(k)} & - \gamma \sqrt{\mathbf{P̂}_{k}(k)} \end{matrix}\bigg] \\
    \mathbf{X̂}_{k}(k+1) &= \bigg[\begin{matrix} \mathbf{f̂}\Big( \mathbf{X̂}_{k}^{1}(k), \mathbf{u}(k), \mathbf{d}(k) \Big) & \mathbf{f̂}\Big( \mathbf{X̂}_{k}^{2}(k), \mathbf{u}(k), \mathbf{d}(k) \Big) & \cdots & \mathbf{f̂}\Big( \mathbf{X̂}_{k}^{n_σ}(k), \mathbf{u}(k), \mathbf{d}(k) \Big) \end{matrix}\bigg] \\
    \mathbf{x̂}_{k}(k+1) &= \mathbf{X̂}_{k}(k+1)\mathbf{m̂} \\
    \mathbf{X̄}_k(k+1)   &= \begin{bmatrix} \mathbf{X̂}_{k}^{1}(k+1) - \mathbf{x̂}_{k}(k+1) & \mathbf{X̂}_{k}^{2}(k+1) - \mathbf{x̂}_{k}(k+1) & \cdots &\, \mathbf{X̂}_{k}^{n_σ}(k+1) - \mathbf{x̂}_{k}(k+1) \end{bmatrix} \\
    \mathbf{P̂}_k(k+1)   &= \mathbf{X̄}_k(k+1) \mathbf{Ŝ} \mathbf{X̄}_k'(k+1) + \mathbf{Q̂}
\end{aligned}
```

[^3]: Simon, D. 2006, "Chapter 14: The unscented Kalman filter" in "Optimal State Estimation: 
     Kalman, H∞, and Nonlinear Approaches", John Wiley & Sons, p. 433–459, <https://doi.org/10.1002/0470045345.ch14>, 
     ISBN9780470045343.
"""
function correct_estimate!(estim::UnscentedKalmanFilter, y0m, d0)
    x̂0, P̂, R̂, K̂ = estim.x̂0, estim.cov.P̂, estim.cov.R̂, estim.K̂
    nx̂ = estim.nx̂
    γ, m̂, Ŝ = estim.γ, estim.m̂, estim.Ŝ
    # in-place operations to reduce allocations:
    P̂_temp  = Hermitian(estim.buffer.P̂, :L)
    P̂_temp .= P̂
    P̂_chol  = cholesky!(P̂_temp) # also modifies P̂_temp
    sqrtP̂   = P̂_chol.L
    γ_sqrtP̂ = lmul!(γ, sqrtP̂)
    X̂0, Ŷ0m = estim.X̂0, estim.Ŷ0m
    X̂0 .= x̂0
    X̂0[:, 2:nx̂+1]   .= @views X̂0[:, 2:nx̂+1]   .+ γ_sqrtP̂
    X̂0[:, nx̂+2:end] .= @views X̂0[:, nx̂+2:end] .- γ_sqrtP̂
    ŷ0 = estim.buffer.ŷ
    for j in axes(Ŷ0m, 2)
        @views ĥ!(ŷ0, estim, estim.model, X̂0[:, j], d0)
        @views Ŷ0m[:, j] .= ŷ0[estim.i_ym]
    end
    ŷ0m = @views ŷ0[estim.i_ym]
    mul!(ŷ0m, Ŷ0m, m̂)
    X̄0, Ȳ0m = estim.X̄0, estim.Ȳ0m
    X̄0  .= X̂0  .- x̂0
    Ȳ0m .= Ŷ0m .- ŷ0m
    Ŝ_Ŷ0mᵀ = estim.Ŷ0m'
    mul!(Ŝ_Ŷ0mᵀ, Ŝ, Ȳ0m')
    M̂ = estim.buffer.R̂
    mul!(M̂, Ȳ0m, Ŝ_Ŷ0mᵀ)
    M̂ .+= R̂
    M̂ = Hermitian(M̂, :L)
    estim.M̂ .= M̂
    mul!(K̂, X̄0, Ŝ_Ŷ0mᵀ)
    rdiv!(K̂, cholesky!(M̂)) # also modifies M̂ (estim.M̂ contains unmodified M̂, see line below)
    M̂ = estim.M̂
    v̂  = ŷ0m
    v̂ .= y0m .- ŷ0m
    x̂0corr, P̂corr = estim.x̂0, estim.cov.P̂
    mul!(x̂0corr, K̂, v̂, 1, 1)
    K̂_M̂   = estim.buffer.K̂
    mul!(K̂_M̂, K̂, M̂)
    K̂_M̂_K̂ᵀ = estim.buffer.Q̂
    mul!(K̂_M̂_K̂ᵀ, K̂_M̂, K̂')
    P̂corr  = estim.buffer.P̂
    P̂corr .= P̂ .- Hermitian(K̂_M̂_K̂ᵀ, :L)
    estim.cov.P̂ .= Hermitian(P̂corr, :L)
    return nothing
end

"Prediction step of [`UnscentedKalmanFilter`](@ref), see [`correct_estimate!`](@ref)."
function predict_estimate!(estim::UnscentedKalmanFilter, u0, d0)
    x̂0corr, X̂0corr, P̂corr = estim.x̂0, estim.X̂0, estim.cov.P̂
    Q̂, nx̂ = estim.cov.Q̂, estim.nx̂
    γ, m̂, Ŝ = estim.γ, estim.m̂, estim.Ŝ
    x̂0next, û0, k = estim.buffer.x̂, estim.buffer.û, estim.buffer.k
    # in-place operations to reduce allocations:
    P̂corr_temp  = Hermitian(estim.buffer.P̂, :L)
    P̂corr_temp .= P̂corr
    P̂corr_chol  = cholesky!(P̂corr_temp) # also modifies P̂corr_temp
    sqrtP̂corr   = P̂corr_chol.L
    γ_sqrtP̂corr = lmul!(γ, sqrtP̂corr)
    X̂0corr .= x̂0corr
    X̂0corr[:, 2:nx̂+1]   .= @views X̂0corr[:, 2:nx̂+1]   .+ γ_sqrtP̂corr
    X̂0corr[:, nx̂+2:end] .= @views X̂0corr[:, nx̂+2:end] .- γ_sqrtP̂corr
    X̂0next = X̂0corr
    for j in axes(X̂0next, 2)
        @views x̂0corr .= X̂0corr[:, j]
        @views f̂!(X̂0next[:, j], û0, k, estim, estim.model, x̂0corr, u0, d0)
    end
    x̂0next .= mul!(x̂0corr, X̂0next, m̂)
    X̄0next  = estim.X̄0
    X̄0next .= X̂0next .- x̂0next
    Ŝ_X̄0nextᵀ = estim.X̂0'
    mul!(Ŝ_X̄0nextᵀ, Ŝ, X̄0next')
    P̂next = estim.buffer.P̂
    mul!(P̂next, X̄0next, Ŝ_X̄0nextᵀ) 
    P̂next   .+= Q̂
    estim.x̂0 .= x̂0next
    estim.cov.P̂  .= Hermitian(P̂next, :L)
    return nothing
end

struct ExtendedKalmanFilter{
        NT<:Real, 
        SM<:SimModel, 
        KC<:KalmanCovariances,
        JB<:AbstractADType, 
        FF<:Function,
        HF<:Function
} <: KalmanEstimator{NT}
    model::SM
    cov  ::KC
    x̂op ::Vector{NT}
    f̂op ::Vector{NT}
    x̂0  ::Vector{NT}
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
    K̂::Matrix{NT}
    F̂_û::Matrix{NT}
    F̂  ::Matrix{NT}
    Ĥ  ::Matrix{NT}
    Ĥm ::Matrix{NT}
    jacobian::JB
    linfuncF̂!::FF
    linfuncĤ!::HF
    direct::Bool
    corrected::Vector{Bool}
    buffer::StateEstimatorBuffer{NT}
    function ExtendedKalmanFilter{NT}(
        model::SM, 
        i_ym, nint_u, nint_ym, cov::KC; 
        jacobian::JB, linfuncF̂!::FF, linfuncĤ!::HF, direct=true
    ) where {
            NT<:Real, 
            SM<:SimModel, 
            KC<:KalmanCovariances, 
            JB<:AbstractADType, 
            FF<:Function,
            HF<:Function
        }
        nu, ny, nd, nk = model.nu, model.ny, model.nd, model.nk
        nym, nyu = validate_ym(model, i_ym)
        As, Cs_u, Cs_y, nint_u, nint_ym = init_estimstoch(model, i_ym, nint_u, nint_ym)
        nxs = size(As, 1)
        nx̂  = model.nx + nxs
        Â, B̂u, Ĉ, B̂d, D̂d, x̂op, f̂op = augment_model(model, As, Cs_u, Cs_y)
        Ĉm, D̂dm = Ĉ[i_ym, :], D̂d[i_ym, :]
        x̂0 = [zeros(NT, model.nx); zeros(NT, nxs)]
        K̂ = zeros(NT, nx̂, nym)
        F̂_û, F̂ = zeros(NT, nx̂+nu, nx̂), zeros(NT, nx̂, nx̂)
        Ĥ,  Ĥm = zeros(NT, ny, nx̂),    zeros(NT, nym, nx̂)
        corrected = [false]
        buffer = StateEstimatorBuffer{NT}(nu, nx̂, nym, ny, nd, nk)
        return new{NT, SM, KC, JB, FF, HF}(
            model,
            cov,
            x̂op, f̂op, x̂0,
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs_u, Cs_y, nint_u, nint_ym,
            Â, B̂u, Ĉ, B̂d, D̂d, Ĉm, D̂dm,
            K̂,
            F̂_û, F̂, Ĥ, Ĥm,
            jacobian, linfuncF̂!, linfuncĤ!,
            direct, corrected,
            buffer
        )
    end
end

@doc raw"""
    ExtendedKalmanFilter(model::SimModel; <keyword arguments>)

Construct an extended Kalman Filter with the [`SimModel`](@ref) `model`.

Both [`LinModel`](@ref) and [`NonLinModel`](@ref) are supported. The process model is
identical to [`UnscentedKalmanFilter`](@ref). By default, the Jacobians of the augmented
model ``\mathbf{f̂, ĥ}`` are computed with [`ForwardDiff`](@extref ForwardDiff) automatic
differentiation. This estimator is allocation-free if `model` simulations do not allocate.
!!! warning
    See the Extended Help of [`linearize`](@ref) function if you get an error like:    
    `MethodError: no method matching (::var"##")(::Vector{ForwardDiff.Dual})`.

# Arguments
!!! info
    Keyword arguments with *`emphasis`* are non-Unicode alternatives.

- `model::SimModel` : (deterministic) model for the estimations.
- `i_ym=1:model.ny` : `model` output indices that are measured ``\mathbf{y^m}``, the rest 
    are unmeasured ``\mathbf{y^u}``.
- `σP_0=fill(1/model.nx,model.nx)` or *`sigmaP_0`* : main diagonal of the initial estimate
    covariance ``\mathbf{P}(0)``, specified as a standard deviation vector.
- `σQ=fill(1/model.nx,model.nx)` or *`sigmaQ`* : main diagonal of the process noise
    covariance ``\mathbf{Q}`` of `model`, specified as a standard deviation vector.
- `σR=fill(1,length(i_ym))` or *`sigmaR`* : main diagonal of the sensor noise covariance
    ``\mathbf{R}`` of `model` measured outputs, specified as a standard deviation vector.
- `nint_u=0`: integrator quantity for the stochastic model of the unmeasured disturbances at
    the manipulated inputs (vector), use `nint_u=0` for no integrator.
- `nint_ym=default_nint(model,i_ym,nint_u)` : same than `nint_u` but for the unmeasured 
    disturbances at the measured outputs, use `nint_ym=0` for no integrator.
- `σQint_u=fill(1,sum(nint_u))` or *`sigmaQint_u`* : same than `σQ` but for the unmeasured
    disturbances at manipulated inputs ``\mathbf{Q_{int_u}}`` (composed of integrators).
- `σPint_u_0=fill(1,sum(nint_u))` or *`sigmaPint_u_0`* : same than `σP_0` but for the unmeasured
    disturbances at manipulated inputs ``\mathbf{P_{int_u}}(0)`` (composed of integrators).
- `σQint_ym=fill(1,sum(nint_ym))` or *`sigmaQint_u`* : same than `σQ` for the unmeasured
    disturbances at measured outputs ``\mathbf{Q_{int_{ym}}}`` (composed of integrators).
- `σPint_ym_0=fill(1,sum(nint_ym))` or *`sigmaPint_ym_0`* : same than `σP_0` but for the unmeasured
    disturbances at measured outputs ``\mathbf{P_{int_{ym}}}(0)`` (composed of integrators).
- `jacobian=AutoForwardDiff()`: an `AbstractADType` backend for the Jacobians of the augmented
    model, see [`DifferentiationInterface` doc](@extref DifferentiationInterface List).
- `direct=true`: construct with a direct transmission from ``\mathbf{y^m}`` (a.k.a. current
   estimator, in opposition to the delayed/predictor form).

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_,_)->0.2x+u, (x,_,_)->-3x, 5.0, 1, 1, 1, solver=nothing);

julia> estim = ExtendedKalmanFilter(model, σQ=[2], σQint_ym=[2], σP_0=[0.1], σPint_ym_0=[0.1])
ExtendedKalmanFilter estimator with a sample time Ts = 5.0 s:
├ model: NonLinModel
├ jacobian: AutoForwardDiff
├ direct: true
└ dimensions:
  ├ 1 manipulated inputs u (0 integrating states)
  ├ 2 estimated states x̂
  ├ 1 measured outputs ym (1 integrating states)
  ├ 0 unmeasured outputs yu
  └ 0 measured disturbances d
```
"""
function ExtendedKalmanFilter(
    model::SM;
    i_ym::AbstractVector{Int} = 1:model.ny,
    sigmaP_0 = fill(1/model.nx, model.nx),
    sigmaQ   = fill(1/model.nx, model.nx),
    sigmaR   = fill(1, length(i_ym)),
    nint_u ::IntVectorOrInt = 0,
    nint_ym::IntVectorOrInt = default_nint(model, i_ym, nint_u),
    sigmaPint_u_0  = fill(1, max(sum(nint_u),  0)),
    sigmaQint_u    = fill(1, max(sum(nint_u),  0)),
    sigmaPint_ym_0 = fill(1, max(sum(nint_ym), 0)),
    sigmaQint_ym   = fill(1, max(sum(nint_ym), 0)),
    jacobian = AutoForwardDiff(),
    direct = true,
    σP_0       = sigmaP_0,
    σQ         = sigmaQ,
    σR         = sigmaR,
    σPint_u_0  = sigmaPint_u_0,
    σQint_u    = sigmaQint_u,
    σPint_ym_0 = sigmaPint_ym_0,
    σQint_ym   = sigmaQint_ym,
) where {NT<:Real, SM<:SimModel{NT}}
    # estimated covariances matrices (variance = σ²) :
    P̂_0 = Diagonal([σP_0; σPint_u_0; σPint_ym_0].^2)
    Q̂   = Diagonal([σQ;  σQint_u;  σQint_ym ].^2)
    R̂   = Diagonal([σR;].^2)
    return ExtendedKalmanFilter(
        model, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂; jacobian, direct
    )
end

@doc raw"""
    ExtendedKalmanFilter(
        model, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂; jacobian=AutoForwardDiff(), direct=true
    )

Construct the estimator from the augmented covariance matrices `P̂_0`, `Q̂` and `R̂`.

This syntax allows nonzero off-diagonal elements in ``\mathbf{P̂}_{-1}(0), \mathbf{Q̂, R̂}``.
"""
function ExtendedKalmanFilter(
    model::SM, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂; jacobian=AutoForwardDiff(), direct=true
) where {NT<:Real, SM<:SimModel{NT}}
    P̂_0, Q̂, R̂ = to_mat(P̂_0), to_mat(Q̂), to_mat(R̂)    
    cov = KalmanCovariances(model, i_ym, nint_u, nint_ym, Q̂, R̂, P̂_0)
    linfuncF̂!, linfuncĤ! = get_ekf_linfuncs(NT, model, i_ym, nint_u, nint_ym, jacobian)
    return ExtendedKalmanFilter{NT}(
        model, i_ym, nint_u, nint_ym, cov; jacobian, linfuncF̂!, linfuncĤ!, direct
    )
end

"""
    get_ekf_linfuncs(NT, model, i_ym, nint_u, nint_ym, jacobian) -> linfuncF̂!, linfuncĤ! 

Return the functions that computes the `F̂` and `Ĥ` Jacobians of the augmented model.

The functions has the following signatures:
```
linfuncF̂!(F̂, x̂0next , backend, x̂0, cst_u0, cst_d0) -> nothing
linfuncĤ!(Ĥ, ŷ0     , backend, x̂0, cst_d0) -> nothing
```
They mutates all the arguments before `backend`, which is an `AbstractADType` object from 
`DifferentiationInterface`. The `cst_u0` and `cst_d0` are `DifferentiationInterface.Constant`
objects with the linearization points.
"""
function get_ekf_linfuncs(NT, model, i_ym, nint_u, nint_ym, jacobian)
    As, Cs_u, Cs_y = init_estimstoch(model, i_ym, nint_u, nint_ym)
    nu, ny, nd, nk = model.nu, model.ny, model.nd, model.nk
    nx̂ = model.nx + size(As, 1)
    x̂op = f̂op = zeros(nx̂) # not important for Jacobian computations
    function f̂_ekf!(x̂0next, x̂0, û0, k, u0, d0)
        return f̂!(x̂0next, û0, k, model, As, Cs_u, f̂op, x̂op, x̂0, u0, d0)
    end
    ĥ_ekf!(ŷ0, x̂0, d0) = ĥ!(ŷ0, model, Cs_y, x̂0, d0)
    strict  = Val(true)
    x̂0next = zeros(NT, nx̂)
    ŷ0 = zeros(NT, ny)
    x̂0 = zeros(NT, nx̂)
    û0 = Cache(zeros(NT, nu))
    k = Cache(zeros(NT, nk))
    cst_u0 = Constant(rand(NT, nu))
    cst_d0 = Constant(rand(NT, nd))
    F̂prep = prepare_jacobian(
        f̂_ekf!, x̂0next, jacobian, x̂0, û0, k, cst_u0, cst_d0; strict
    )
    Ĥprep = prepare_jacobian(ĥ_ekf!, ŷ0, jacobian, x̂0, cst_d0; strict)
    function linfuncF̂!(F̂, x̂0next, backend, x̂0, cst_u0, cst_d0)
        return jacobian!(f̂_ekf!, x̂0next, F̂, F̂prep, backend, x̂0, û0, k, cst_u0, cst_d0)
    end
    function linfuncĤ!(Ĥ, ŷ0, backend, x̂0, cst_d0)
        return jacobian!(ĥ_ekf!, ŷ0, Ĥ, Ĥprep, backend, x̂0, cst_d0)
    end
    return linfuncF̂!, linfuncĤ! 
end

@doc raw"""
    correct_estimate!(estim::ExtendedKalmanFilter, y0m, d0)

Correct [`ExtendedKalmanFilter`](@ref) state `estim.x̂0` and covariance `estim.cov.P̂`.

The equations are similar to [`KalmanFilter`](@ref) but with the substitutions
``\mathbf{Ĉ^m = Ĥ^m}(k)`` and ``\mathbf{Â = F̂}(k)``, the Jacobians of the augmented process
model:
```math
\begin{aligned}
    \mathbf{Ĥ}(k) &= \left. \frac{∂\mathbf{ĥ}(\mathbf{x̂}, \mathbf{d})}{∂\mathbf{x̂}}             \right|_{\mathbf{x̂ = x̂}_{k-1}(k),\, \mathbf{d = d}(k)}   \\
    \mathbf{F̂}(k) &= \left. \frac{∂\mathbf{f̂}(\mathbf{x̂}, \mathbf{u}, \mathbf{d})}{∂\mathbf{x̂}} \right|_{\mathbf{x̂ = x̂}_{k}(k),  \, \mathbf{u = u}(k),\, \mathbf{d = d}(k)}
\end{aligned}
```
The matrix ``\mathbf{Ĥ^m}`` is the rows of ``\mathbf{Ĥ}`` that are measured outputs. The
Jacobians are computed with [`ForwardDiff`](@extref ForwardDiff) by default. The correction
and prediction step equations are provided below.

# Correction Step
```math
\begin{aligned}
    \mathbf{Ŝ}(k)     &= \mathbf{Ĥ^m}(k)\mathbf{P̂}_{k-1}(k)\mathbf{Ĥ^m}'(k) + \mathbf{R̂}             \\
    \mathbf{K̂}(k)     &= \mathbf{P̂}_{k-1}(k)\mathbf{Ĥ^m}'(k)\mathbf{Ŝ^{-1}}(k)                       \\
    \mathbf{ŷ^m}(k)   &= \mathbf{ĥ^m}\Big( \mathbf{x̂}_{k-1}(k), \mathbf{d}(k) \Big)                  \\
    \mathbf{x̂}_{k}(k) &= \mathbf{x̂}_{k-1}(k) + \mathbf{K̂}(k)[\mathbf{y^m}(k) - \mathbf{ŷ^m}(k)]      \\
    \mathbf{P̂}_{k}(k) &= [\mathbf{I - K̂}(k)\mathbf{Ĥ^m}(k)]\mathbf{P̂}_{k-1}(k)
\end{aligned}
```

# Prediction Step
```math
\begin{aligned}
    \mathbf{x̂}_{k}(k+1) &= \mathbf{f̂}\Big( \mathbf{x̂}_{k}(k), \mathbf{u}(k), \mathbf{d}(k) \Big)   \\
    \mathbf{P̂}_{k}(k+1) &= \mathbf{F̂}(k)\mathbf{P̂}_{k}(k)\mathbf{F̂}'(k) + \mathbf{Q̂}
\end{aligned}
```
"""
function correct_estimate!(estim::ExtendedKalmanFilter, y0m, d0)
    x̂0 = estim.x̂0
    cst_d0 = Constant(d0)
    ŷ0, Ĥ, Ĥm = estim.buffer.ŷ, estim.Ĥ, estim.Ĥm
    estim.linfuncĤ!(Ĥ, ŷ0, estim.jacobian, x̂0, cst_d0)
    Ĥm .= @views Ĥ[estim.i_ym, :]
    return correct_kf!(estim, y0m, d0, Ĥm)
end

"Prediction step of [`ExtendedKalmanFilter`](@ref), see [`correct_estimate!`](@ref)."
function predict_estimate!(estim::ExtendedKalmanFilter, u0, d0)
    cst_u0, cst_d0 = Constant(u0), Constant(d0)
    x̂0corr = estim.x̂0
    x̂0next, F̂ = estim.buffer.x̂, estim.F̂
    estim.linfuncF̂!(F̂, x̂0next, estim.jacobian, x̂0corr, cst_u0, cst_d0)
    return predict_kf!(estim, u0, d0, F̂)
end

"Print the `jacobian` backend and `direct` flag for [`ExtendedKalmanFilter`](@ref)."
function print_details(io::IO, estim::ExtendedKalmanFilter)
    println(io, "├ jacobian: $(backend_str(estim.jacobian))")
    println(io, "├ direct: $(estim.direct)")
end

"Set `estim.cov.P̂` to `estim.cov.P̂_0` for the time-varying Kalman Filters."
function init_estimate_cov!(
    estim::Union{KalmanFilter, UnscentedKalmanFilter, ExtendedKalmanFilter}, _ , _ , _
) 
    estim.cov.P̂ .= estim.cov.P̂_0
    return nothing
end

"""
    correct_kf!(estim::Union{KalmanFilter, ExtendedKalmanFilter}, y0m, d0, Ĉm)

Correct time-varying/extended Kalman Filter estimates with augmented `Ĉm` matrices.

Allows code reuse for [`KalmanFilter`](@ref), [`ExtendedKalmanFilterKalmanFilter`](@ref).
See [`update_estimate_kf!`](@ref) for more information.
"""
function correct_kf!(estim::Union{KalmanFilter, ExtendedKalmanFilter}, y0m, d0, Ĉm)
    R̂, K̂ = estim.cov.R̂, estim.K̂
    x̂0, P̂ = estim.x̂0, estim.cov.P̂
    # in-place operations to reduce allocations:
    P̂_Ĉmᵀ = K̂
    mul!(P̂_Ĉmᵀ, P̂, Ĉm')
    M̂ = estim.buffer.R̂
    mul!(M̂, Ĉm, P̂_Ĉmᵀ)
    M̂ .+= R̂
    K̂ = P̂_Ĉmᵀ
    M̂_chol = cholesky!(Hermitian(M̂, :L)) # also modifies M̂
    rdiv!(K̂, M̂_chol)
    ŷ0 = estim.buffer.ŷ
    ĥ!(ŷ0, estim, estim.model, x̂0, d0)
    ŷ0m = @views ŷ0[estim.i_ym]
    v̂  = ŷ0m
    v̂ .= y0m .- ŷ0m
    x̂0corr = x̂0
    mul!(x̂0corr, K̂, v̂, 1, 1) # also modifies estim.x̂0
    I_minus_K̂_Ĉm = estim.buffer.Q̂
    mul!(I_minus_K̂_Ĉm, K̂, Ĉm)
    lmul!(-1, I_minus_K̂_Ĉm)
    for i=diagind(I_minus_K̂_Ĉm)
        I_minus_K̂_Ĉm[i] += 1 # compute I - K̂*Ĉm in-place
    end
    P̂corr = estim.buffer.P̂
    mul!(P̂corr, I_minus_K̂_Ĉm, P̂)
    estim.cov.P̂ .= Hermitian(P̂corr, :L)
    return nothing
end

"""
    predict_kf!(estim::Union{KalmanFilter, ExtendedKalmanFilter}, u0, d0, Â)

Predict time-varying/extended Kalman Filter estimates with augmented `Ĉm` and `Â` matrices.

Allows code reuse for [`KalmanFilter`](@ref), [`ExtendedKalmanFilterKalmanFilter`](@ref).
They predict the state `x̂` and covariance `P̂` with the same equations. See 
[`update_estimate`](@ref) methods for the equations.
"""
function predict_kf!(estim::Union{KalmanFilter, ExtendedKalmanFilter}, u0, d0, Â)
    x̂0corr, P̂corr = estim.x̂0, estim.cov.P̂
    Q̂ = estim.cov.Q̂
    x̂0next, û0, k = estim.buffer.x̂, estim.buffer.û, estim.buffer.k
    # in-place operations to reduce allocations:
    f̂!(x̂0next, û0, k, estim, estim.model, x̂0corr, u0, d0)
    P̂corr_Âᵀ = estim.buffer.P̂
    mul!(P̂corr_Âᵀ, P̂corr, Â')
    Â_P̂corr_Âᵀ = estim.buffer.Q̂
    mul!(Â_P̂corr_Âᵀ, Â, P̂corr_Âᵀ)
    P̂next  = estim.buffer.P̂
    P̂next .= Â_P̂corr_Âᵀ .+ Q̂
    estim.x̂0 .= x̂0next
    estim.cov.P̂  .= Hermitian(P̂next, :L)
    return nothing
end
