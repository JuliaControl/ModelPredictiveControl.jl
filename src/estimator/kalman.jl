struct SteadyKalmanFilter{NT<:Real, SM<:LinModel} <: StateEstimator{NT}
    model::SM
    lastu0::Vector{NT}
    x̂::Vector{NT}
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
    Q̂::Hermitian{NT, Matrix{NT}}
    R̂::Hermitian{NT, Matrix{NT}}
    K̂::Matrix{NT}
    function SteadyKalmanFilter{NT, SM}(
        model::SM, i_ym, nint_u, nint_ym, Q̂, R̂
    ) where {NT<:Real, SM<:LinModel}
        nym, nyu = validate_ym(model, i_ym)
        As, Cs_u, Cs_y, nint_u, nint_ym = init_estimstoch(model, i_ym, nint_u, nint_ym)
        nxs = size(As, 1)
        nx̂  = model.nx + nxs
        Â, B̂u, Ĉ, B̂d, D̂d = augment_model(model, As, Cs_u, Cs_y)
        validate_kfcov(nym, nx̂, Q̂, R̂)
        K̂ = try
            Q̂_kalman = Matrix(Q̂) # Matrix() required for Julia 1.6
            R̂_kalman = zeros(NT, model.ny, model.ny)
            R̂_kalman[i_ym, i_ym] = R̂
            kalman(Discrete, Â, Ĉ, Q̂_kalman, R̂_kalman)[:, i_ym] 
        catch my_error
            if isa(my_error, ErrorException)
                error("Cannot compute the optimal Kalman gain K for the "* 
                      "SteadyKalmanFilter. You may try to remove integrators with "*
                      "nint_u/nint_ym parameter or use the time-varying KalmanFilter.")
            else
                rethrow()
            end
        end
        Ĉm, D̂dm = Ĉ[i_ym, :], D̂d[i_ym, :] # measured outputs ym only
        lastu0 = zeros(NT, model.nu)
        x̂ = [zeros(NT, model.nx); zeros(NT, nxs)]
        Q̂, R̂ = Hermitian(Q̂, :L),  Hermitian(R̂, :L)
        return new{NT, SM}(
            model, 
            lastu0, x̂, 
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs_u, Cs_y, nint_u, nint_ym,
            Â, B̂u, Ĉ, B̂d, D̂d,
            Ĉm, D̂dm,
            Q̂, R̂,
            K̂
        )
    end
end

@doc raw"""
    SteadyKalmanFilter(model::LinModel; <keyword arguments>)

Construct a steady-state Kalman Filter with the [`LinModel`](@ref) `model`.

The steady-state (or [asymptotic](https://en.wikipedia.org/wiki/Kalman_filter#Asymptotic_form))
Kalman filter is based on the process model :
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
Q_{int_u}, Q_{int_{ym}})}`` and ``\mathbf{R̂ = R}``. The matrices ``\mathbf{Ĉ^m, D̂_d^m}`` are
the rows of ``\mathbf{Ĉ, D̂_d}`` that correspond to measured outputs ``\mathbf{y^m}`` (and 
unmeasured ones, for ``\mathbf{Ĉ^u, D̂_d^u}``).

# Arguments
- `model::LinModel` : (deterministic) model for the estimations.
- `i_ym=1:model.ny` : `model` output indices that are measured ``\mathbf{y^m}``, the rest 
    are unmeasured ``\mathbf{y^u}``.
- `σQ=fill(1/model.nx,model.nx)` : main diagonal of the process noise covariance
    ``\mathbf{Q}`` of `model`, specified as a standard deviation vector.
- `σR=fill(1,length(i_ym))` : main diagonal of the sensor noise covariance ``\mathbf{R}``
    of `model` measured outputs, specified as a standard deviation vector.
- `nint_u=0`: integrator quantity for the stochastic model of the unmeasured disturbances at
    the manipulated inputs (vector), use `nint_u=0` for no integrator (see Extended Help).
- `σQint_u=fill(1,sum(nint_u))`: same than `σQ` but for the unmeasured disturbances at 
    manipulated inputs ``\mathbf{Q_{int_u}}`` (composed of integrators).
- `nint_ym=default_nint(model,i_ym,nint_u)` : same than `nint_u` but for the unmeasured 
    disturbances at the measured outputs, use `nint_ym=0` for no integrator (see Extended Help).
- `σQint_ym=fill(1,sum(nint_ym))` : same than `σQ` for the unmeasured disturbances at 
    measured outputs ``\mathbf{Q_{int_{ym}}}`` (composed of integrators).

# Examples
```jldoctest
julia> model = LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 0.5);

julia> estim = SteadyKalmanFilter(model, i_ym=[2], σR=[1], σQint_ym=[0.01])
SteadyKalmanFilter estimator with a sample time Ts = 0.5 s, LinModel and:
 1 manipulated inputs u (0 integrating states)
 3 estimated states x̂
 1 measured outputs ym (1 integrating states)
 1 unmeasured outputs yu
 0 measured disturbances d
```

# Extended Help
!!! details "Extended Help"
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

    The constructor pre-compute the steady-state Kalman gain `K̂` with the [`kalman`](https://juliacontrol.github.io/ControlSystems.jl/stable/lib/synthesis/#ControlSystemsBase.kalman-Tuple{Any,%20Any,%20Any,%20Any,%20Any,%20Vararg{Any}})
    function. It can sometimes fail, for example when `model` matrices are ill-conditioned.
    In such a case, you can try the alternative time-varying [`KalmanFilter`](@ref).
"""
function SteadyKalmanFilter(
    model::SM;
    i_ym::IntRangeOrVector = 1:model.ny,
    σQ::Vector = fill(1/model.nx, model.nx),
    σR::Vector = fill(1, length(i_ym)),
    nint_u ::IntVectorOrInt = 0,
    σQint_u::Vector = fill(1, max(sum(nint_u), 0)),
    nint_ym::IntVectorOrInt = default_nint(model, i_ym, nint_u),
    σQint_ym::Vector = fill(1, max(sum(nint_ym), 0))
) where {NT<:Real, SM<:LinModel{NT}}
    # estimated covariances matrices (variance = σ²) :
    Q̂  = Hermitian(diagm(NT[σQ;  σQint_u;  σQint_ym ].^2), :L)
    R̂  = Hermitian(diagm(NT[σR;].^2), :L)
    return SteadyKalmanFilter{NT, SM}(model, i_ym, nint_u, nint_ym, Q̂ , R̂)
end

@doc raw"""
    SteadyKalmanFilter(model, i_ym, nint_u, nint_ym, Q̂, R̂)

Construct the estimator from the augmented covariance matrices `Q̂` and `R̂`.

This syntax allows nonzero off-diagonal elements in ``\mathbf{Q̂, R̂}``.
"""
function SteadyKalmanFilter(model::SM, i_ym, nint_u, nint_ym, Q̂, R̂) where {NT<:Real, SM<:LinModel{NT}}
    Q̂, R̂ = to_mat(Q̂), to_mat(R̂)
    return SteadyKalmanFilter{NT, SM}(model, i_ym, nint_u, nint_ym, Q̂ , R̂)
end



@doc raw"""
    update_estimate!(estim::SteadyKalmanFilter, u, ym, d)

Update `estim.x̂` estimate with current inputs `u`, measured outputs `ym` and dist. `d`.

The [`SteadyKalmanFilter`](@ref) updates it with the precomputed Kalman gain ``\mathbf{K̂}``:
```math
\mathbf{x̂}_{k}(k+1) = \mathbf{Â x̂}_{k-1}(k) + \mathbf{B̂_u u}(k) + \mathbf{B̂_d d}(k) 
               + \mathbf{K̂}[\mathbf{y^m}(k) - \mathbf{Ĉ^m x̂}_{k-1}(k) - \mathbf{D̂_d^m d}(k)]
```
"""
function update_estimate!(estim::SteadyKalmanFilter, u, ym, d=empty(estim.x̂))
    Â, B̂u, B̂d, Ĉm, D̂dm = estim.Â, estim.B̂u, estim.B̂d, estim.Ĉm, estim.D̂dm
    x̂, K̂ = estim.x̂, estim.K̂
    ŷm, x̂next = similar(ym), similar(x̂)
    # in-place operations to reduce allocations:
    mul!(ŷm, Ĉm, x̂) 
    mul!(ŷm, D̂dm, d, 1, 1)
    v̂  = ŷm
    v̂ .= ym .- ŷm
    mul!(x̂next, Â, x̂)
    mul!(x̂next, B̂u, u, 1, 1)
    mul!(x̂next, B̂d, d, 1, 1)
    mul!(x̂next, K̂, v̂, 1, 1)
    x̂ .= x̂next
    return nothing
end

struct KalmanFilter{NT<:Real, SM<:LinModel} <: StateEstimator{NT}
    model::SM
    lastu0::Vector{NT}
    x̂::Vector{NT}
    P̂::Hermitian{NT, Matrix{NT}}
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
    P̂0::Hermitian{NT, Matrix{NT}}
    Q̂::Hermitian{NT, Matrix{NT}}
    R̂::Hermitian{NT, Matrix{NT}}
    K̂::Matrix{NT}
    M̂::Matrix{NT}
    function KalmanFilter{NT, SM}(
        model::SM, i_ym, nint_u, nint_ym, P̂0, Q̂, R̂
    ) where {NT<:Real, SM<:LinModel}
        nym, nyu = validate_ym(model, i_ym)
        As, Cs_u, Cs_y, nint_u, nint_ym = init_estimstoch(model, i_ym, nint_u, nint_ym)
        nxs = size(As, 1)
        nx̂  = model.nx + nxs
        Â, B̂u, Ĉ, B̂d, D̂d = augment_model(model, As, Cs_u, Cs_y)
        validate_kfcov(nym, nx̂, Q̂, R̂, P̂0)
        Ĉm, D̂dm = Ĉ[i_ym, :], D̂d[i_ym, :] # measured outputs ym only
        lastu0 = zeros(NT, model.nu)
        x̂ = [zeros(NT, model.nx); zeros(NT, nxs)]
        Q̂, R̂ = Hermitian(Q̂, :L),  Hermitian(R̂, :L)
        P̂0 = Hermitian(P̂0, :L)
        P̂ = copy(P̂0)
        K̂, M̂ = zeros(NT, nx̂, nym), zeros(NT, nx̂, nym)
        return new{NT, SM}(
            model, 
            lastu0, x̂, P̂, 
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs_u, Cs_y, nint_u, nint_ym,
            Â, B̂u, Ĉ, B̂d, D̂d, 
            Ĉm, D̂dm,
            P̂0, Q̂, R̂,
            K̂, M̂
        )
    end
end

@doc raw"""
    KalmanFilter(model::LinModel; <keyword arguments>)

Construct a time-varying Kalman Filter with the [`LinModel`](@ref) `model`.

The process model is identical to [`SteadyKalmanFilter`](@ref). The matrix 
``\mathbf{P̂}_k(k+1)`` is the estimation error covariance of `model` states augmented with
the stochastic ones (specified by `nint_u` and `nint_ym`). Three keyword arguments modify
its initial value with ``\mathbf{P̂}_{-1}(0) = 
    \mathrm{diag}\{ \mathbf{P}(0), \mathbf{P_{int_{u}}}(0), \mathbf{P_{int_{ym}}}(0) \}``.

# Arguments
- `model::LinModel` : (deterministic) model for the estimations.
- `σP0=fill(1/model.nx,model.nx)` : main diagonal of the initial estimate covariance
    ``\mathbf{P}(0)``, specified as a standard deviation vector.
- `σP0int_u=fill(1,sum(nint_u))` : same than `σP0` but for the unmeasured disturbances at 
    manipulated inputs ``\mathbf{P_{int_u}}(0)`` (composed of integrators).
- `σP0int_ym=fill(1,sum(nint_ym))` : same than `σP0` but for the unmeasured disturbances at 
    measured outputs ``\mathbf{P_{int_{ym}}}(0)`` (composed of integrators).
- `<keyword arguments>` of [`SteadyKalmanFilter`](@ref) constructor.

# Examples
```jldoctest
julia> model = LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 0.5);

julia> estim = KalmanFilter(model, i_ym=[2], σR=[1], σP0=[100, 100], σQint_ym=[0.01])
KalmanFilter estimator with a sample time Ts = 0.5 s, LinModel and:
 1 manipulated inputs u (0 integrating states)
 3 estimated states x̂
 1 measured outputs ym (1 integrating states)
 1 unmeasured outputs yu
 0 measured disturbances d
```
"""
function KalmanFilter(
    model::SM;
    i_ym::IntRangeOrVector = 1:model.ny,
    σP0::Vector = fill(1/model.nx, model.nx),
    σQ ::Vector = fill(1/model.nx, model.nx),
    σR ::Vector = fill(1, length(i_ym)),
    nint_u   ::IntVectorOrInt = 0,
    σQint_u  ::Vector = fill(1, max(sum(nint_u), 0)),
    σP0int_u ::Vector = fill(1, max(sum(nint_u), 0)),
    nint_ym  ::IntVectorOrInt = default_nint(model, i_ym, nint_u),
    σQint_ym ::Vector = fill(1, max(sum(nint_ym), 0)),
    σP0int_ym::Vector = fill(1, max(sum(nint_ym), 0)),
) where {NT<:Real, SM<:LinModel{NT}}
    # estimated covariances matrices (variance = σ²) :
    P̂0 = Hermitian(diagm(NT[σP0; σP0int_u; σP0int_ym].^2), :L)
    Q̂  = Hermitian(diagm(NT[σQ;  σQint_u;  σQint_ym ].^2), :L)
    R̂  = Hermitian(diagm(NT[σR;].^2), :L)
    return KalmanFilter{NT, SM}(model, i_ym, nint_u, nint_ym, P̂0, Q̂ , R̂)
end

@doc raw"""
    KalmanFilter(model, i_ym, nint_u, nint_ym, P̂0, Q̂, R̂)

Construct the estimator from the augmented covariance matrices `P̂0`, `Q̂` and `R̂`.

This syntax allows nonzero off-diagonal elements in ``\mathbf{P̂}_{-1}(0), \mathbf{Q̂, R̂}``.
"""
function KalmanFilter(model::SM, i_ym, nint_u, nint_ym, P̂0, Q̂, R̂) where {NT<:Real, SM<:LinModel{NT}}
    P̂0, Q̂, R̂ = to_mat(P̂0), to_mat(Q̂), to_mat(R̂)
    return KalmanFilter{NT, SM}(model, i_ym, nint_u, nint_ym, P̂0, Q̂ , R̂)
end

@doc raw"""
    update_estimate!(estim::KalmanFilter, u, ym, d)

Update [`KalmanFilter`](@ref) state `estim.x̂` and estimation error covariance `estim.P̂`.

It implements the time-varying Kalman Filter in its predictor (observer) form :
```math
\begin{aligned}
    \mathbf{M̂}(k)       &= \mathbf{P̂}_{k-1}(k)\mathbf{Ĉ^m}'
                           [\mathbf{Ĉ^m P̂}_{k-1}(k)\mathbf{Ĉ^m}' + \mathbf{R̂}]^{-1}       \\
    \mathbf{K̂}(k)       &= \mathbf{Â M̂}(k)                                                \\
    \mathbf{ŷ^m}(k)     &= \mathbf{Ĉ^m x̂}_{k-1}(k) + \mathbf{D̂_d^m d}(k)                  \\
    \mathbf{x̂}_{k}(k+1) &= \mathbf{Â x̂}_{k-1}(k) + \mathbf{B̂_u u}(k) + \mathbf{B̂_d d}(k)
                           + \mathbf{K̂}(k)[\mathbf{y^m}(k) - \mathbf{ŷ^m}(k)]             \\
    \mathbf{P̂}_{k}(k+1) &= \mathbf{Â}[\mathbf{P̂}_{k-1}(k)
                           - \mathbf{M̂}(k)\mathbf{Ĉ^m P̂}_{k-1}(k)]\mathbf{Â}' + \mathbf{Q̂}
\end{aligned}
```
based on the process model described in [`SteadyKalmanFilter`](@ref). The notation 
``\mathbf{x̂}_{k-1}(k)`` refers to the state for the current time ``k`` estimated at the last 
control period ``k-1``. See [^2] for details.

[^2]: Boyd S., "Lecture 8 : The Kalman Filter" (Winter 2008-09) [course slides], *EE363: 
     Linear Dynamical Systems*, <https://web.stanford.edu/class/ee363/lectures/kf.pdf>.
"""
function update_estimate!(estim::KalmanFilter, u, ym, d)
    return update_estimate_kf!(estim, u, ym, d, estim.Â, estim.Ĉm, estim.P̂, estim.x̂)
end

struct UnscentedKalmanFilter{NT<:Real, SM<:SimModel} <: StateEstimator{NT}
    model::SM
    lastu0::Vector{NT}
    x̂::Vector{NT}
    P̂::Hermitian{NT, Matrix{NT}}
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
    P̂0::Hermitian{NT, Matrix{NT}}
    Q̂::Hermitian{NT, Matrix{NT}}
    R̂::Hermitian{NT, Matrix{NT}}
    K̂::Matrix{NT}
    nσ::Int 
    γ::NT
    m̂::Vector{NT}
    Ŝ::Diagonal{NT, Vector{NT}}
    function UnscentedKalmanFilter{NT, SM}(
        model::SM, i_ym, nint_u, nint_ym, P̂0, Q̂, R̂, α, β, κ
    ) where {NT<:Real, SM<:SimModel{NT}}
        nym, nyu = validate_ym(model, i_ym)
        As, Cs_u, Cs_y, nint_u, nint_ym = init_estimstoch(model, i_ym, nint_u, nint_ym)
        nxs = size(As, 1)
        nx̂  = model.nx + nxs
        Â, B̂u, Ĉ, B̂d, D̂d = augment_model(model, As, Cs_u, Cs_y)
        validate_kfcov(nym, nx̂, Q̂, R̂, P̂0)
        nσ, γ, m̂, Ŝ = init_ukf(model, nx̂, α, β, κ)
        lastu0 = zeros(NT, model.nu)
        x̂ = [zeros(NT, model.nx); zeros(NT, nxs)]
        Q̂, R̂ = Hermitian(Q̂, :L),  Hermitian(R̂, :L)
        P̂0 = Hermitian(P̂0, :L)
        P̂ = copy(P̂0)
        K̂ = zeros(NT, nx̂, nym)
        return new{NT, SM}(
            model,
            lastu0, x̂, P̂, 
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs_u, Cs_y, nint_u, nint_ym,
            Â, B̂u, Ĉ, B̂d, D̂d,
            P̂0, Q̂, R̂,
            K̂,
            nσ, γ, m̂, Ŝ
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
``\mathbf{R̂}, \mathbf{Q̂}`` covariances. The functions ``\mathbf{f̂, ĥ}`` are `model` 
state-space functions augmented with the stochastic model, which is specified by the numbers
of integrator `nint_u` and `nint_ym` (see Extended Help). The ``\mathbf{ĥ^m}`` function 
represents the measured outputs of ``\mathbf{ĥ}`` function (and unmeasured ones, for 
``\mathbf{ĥ^u}``).

# Arguments
- `model::SimModel` : (deterministic) model for the estimations.
- `α=1e-3` : alpha parameter, spread of the state distribution ``(0 ≤ α ≤ 1)``.
- `β=2` : beta parameter, skewness and kurtosis of the states distribution ``(β ≥ 0)``.
- `κ=0` : kappa parameter, another spread parameter ``(0 ≤ κ ≤ 3)``.
- `<keyword arguments>` of [`SteadyKalmanFilter`](@ref) constructor.
- `<keyword arguments>` of [`KalmanFilter`](@ref) constructor.

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_)->0.1x+u, (x,_)->2x, 10.0, 1, 1, 1, solver=nothing);

julia> estim = UnscentedKalmanFilter(model, σR=[1], nint_ym=[2], σP0int_ym=[1, 1])
UnscentedKalmanFilter estimator with a sample time Ts = 10.0 s, NonLinModel and:
 1 manipulated inputs u (0 integrating states)
 3 estimated states x̂
 1 measured outputs ym (2 integrating states)
 0 unmeasured outputs yu
 0 measured disturbances d
```

# Extended Help
!!! details "Extended Help"
    The Extended Help of [`SteadyKalmanFilter`](@ref) details the augmentation with `nint_ym` 
    and `nint_u` arguments. Note that the constructor does not validate the observability of
    the resulting augmented [`NonLinModel`](@ref). In such cases, it is the user's 
    responsibility to ensure that the augmented model is still observable.
"""
function UnscentedKalmanFilter(
    model::SM;
    i_ym::IntRangeOrVector = 1:model.ny,
    σP0::Vector = fill(1/model.nx, model.nx),
    σQ ::Vector = fill(1/model.nx, model.nx),
    σR ::Vector = fill(1, length(i_ym)),
    nint_u   ::IntVectorOrInt = 0,
    σQint_u  ::Vector = fill(1, max(sum(nint_u), 0)),
    σP0int_u ::Vector = fill(1, max(sum(nint_u), 0)),
    nint_ym  ::IntVectorOrInt = default_nint(model, i_ym, nint_u),
    σQint_ym ::Vector = fill(1, max(sum(nint_ym), 0)),
    σP0int_ym::Vector = fill(1, max(sum(nint_ym), 0)),
    α::Real = 1e-3,
    β::Real = 2,
    κ::Real = 0
) where {NT<:Real, SM<:SimModel{NT}}
    # estimated covariances matrices (variance = σ²) :
    P̂0 = Hermitian(diagm(NT[σP0; σP0int_u; σP0int_ym].^2), :L)
    Q̂  = Hermitian(diagm(NT[σQ;  σQint_u;  σQint_ym ].^2), :L)
    R̂  = Hermitian(diagm(NT[σR;].^2), :L)
    return UnscentedKalmanFilter{NT, SM}(model, i_ym, nint_u, nint_ym, P̂0, Q̂, R̂, α, β, κ)
end

@doc raw"""
    UnscentedKalmanFilter(model, i_ym, nint_u, nint_ym, P̂0, Q̂, R̂, α, β, κ)

Construct the estimator from the augmented covariance matrices `P̂0`, `Q̂` and `R̂`.

This syntax allows nonzero off-diagonal elements in ``\mathbf{P̂}_{-1}(0), \mathbf{Q̂, R̂}``.
"""
function UnscentedKalmanFilter(
    model::SM, i_ym, nint_u, nint_ym, P̂0, Q̂, R̂, α, β, κ
) where {NT<:Real, SM<:SimModel{NT}}
    P̂0, Q̂, R̂ = to_mat(P̂0), to_mat(Q̂), to_mat(R̂)
    return UnscentedKalmanFilter{NT, SM}(model, i_ym, nint_u, nint_ym, P̂0, Q̂ , R̂, α, β, κ)
end


@doc raw"""
    init_ukf(model, nx̂, α, β, κ) -> nσ, γ, m̂, Ŝ

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
See [`update_estimate!(::UnscentedKalmanFilter)`](@ref) for other details.
"""
function init_ukf(::SimModel{NT}, nx̂, α, β, κ) where {NT<:Real}
    nσ = 2nx̂ + 1                                  # number of sigma points
    γ = α * √(nx̂ + κ)                             # constant factor of standard deviation √P
    m̂_0 = 1 - nx̂ / γ^2
    Ŝ_0 = m̂_0 + 1 - α^2 + β
    w = 1 / 2 / γ^2
    m̂ = NT[m̂_0; fill(w, 2 * nx̂)]                  # weights for the mean
    Ŝ = Diagonal(NT[Ŝ_0; fill(w, 2 * nx̂)])        # weights for the covariance
    return nσ, γ, m̂, Ŝ
end

@doc raw"""
    update_estimate!(estim::UnscentedKalmanFilter, u, ym, d)
    
Update [`UnscentedKalmanFilter`](@ref) state `estim.x̂` and covariance estimate `estim.P̂`.

It implements the unscented Kalman Filter in its predictor (observer) form, based on the 
generalized unscented transform[^3]. See [`init_ukf`](@ref) for the definition of the 
constants ``\mathbf{m̂, Ŝ}`` and ``γ``. 

Denoting ``\mathbf{x̂}_{k-1}(k)`` as the state for the current time ``k`` estimated at the 
last period ``k-1``, ``\mathbf{0}``, a null vector, ``n_σ = 2 n_\mathbf{x̂} + 1``, the number
of sigma points, and ``\mathbf{X̂}_{k-1}^j(k)``, the vector at the ``j``th column of 
``\mathbf{X̂}_{k-1}(k)``, the estimator updates the states with:
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
    \mathbf{X̂}_k(k)     &= \bigg[\begin{matrix} \mathbf{x̂}_{k}(k) & \mathbf{x̂}_{k}(k) & \cdots & \mathbf{x̂}_{k}(k) \end{matrix}\bigg] + \bigg[\begin{matrix} \mathbf{0} & \gamma \sqrt{\mathbf{P̂}_{k}(k)} & - \gamma \sqrt{\mathbf{P̂}_{k}(k)} \end{matrix}\bigg] \\
    \mathbf{X̂}_{k}(k+1) &= \bigg[\begin{matrix} \mathbf{f̂}\Big( \mathbf{X̂}_{k}^{1}(k), \mathbf{u}(k), \mathbf{d}(k) \Big) & \mathbf{f̂}\Big( \mathbf{X̂}_{k}^{2}(k), \mathbf{u}(k), \mathbf{d}(k) \Big) & \cdots & \mathbf{f̂}\Big( \mathbf{X̂}_{k}^{n_σ}(k), \mathbf{u}(k), \mathbf{d}(k) \Big) \end{matrix}\bigg] \\
    \mathbf{x̂}_{k}(k+1) &= \mathbf{X̂}_{k}(k+1)\mathbf{m̂} \\
    \mathbf{X̄}_k(k+1)   &= \begin{bmatrix} \mathbf{X̂}_{k}^{1}(k+1) - \mathbf{x̂}_{k}(k+1) & \mathbf{X̂}_{k}^{2}(k+1) - \mathbf{x̂}_{k}(k+1) & \cdots &\, \mathbf{X̂}_{k}^{n_σ}(k+1) - \mathbf{x̂}_{k}(k+1) \end{bmatrix} \\
    \mathbf{P̂}_k(k+1)   &= \mathbf{X̄}_k(k+1) \mathbf{Ŝ} \mathbf{X̄}_k'(k+1) + \mathbf{Q̂}
\end{aligned} 
```
by using the lower triangular factor of [`cholesky`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.cholesky)
to compute ``\sqrt{\mathbf{P̂}_{k-1}(k)}`` and ``\sqrt{\mathbf{P̂}_{k}(k)}``.  The matrices 
``\mathbf{P̂, Q̂, R̂}`` are the covariance of the estimation error, process noise and sensor 
noise, respectively.

[^3]: Simon, D. 2006, "Chapter 14: The unscented Kalman filter" in "Optimal State Estimation: 
     Kalman, H∞, and Nonlinear Approaches", John Wiley & Sons, p. 433–459, <https://doi.org/10.1002/0470045345.ch14>, 
     ISBN9780470045343.
"""
function update_estimate!(estim::UnscentedKalmanFilter{NT}, u, ym, d) where NT<:Real
    return update_estimate_ukf!(estim, u, ym, d, estim.P̂, estim.x̂)
end

"""
    update_estimate_ukf!(estim::StateEstimator, u, ym, d, P̂, x̂=nothing)

Update Unscented Kalman Filter estimates and covariance matrices.

Allows code reuse for [`UnscentedKalmanFilter`](@ref) and [`MovingHorizonEstimator`](@ref).
See  [`update_estimate!(::UnscentedKalmanFilter, ::Any, ::Any, ::Any)`(@ref) docstring
for the equations. If `isnothing(x̂)`, only the covariance `P̂` is updated.
"""
function update_estimate_ukf!(
    estim::StateEstimator{NT}, u, ym, d, P̂, x̂=nothing
) where NT<:Real
    Q̂, R̂, K̂ = estim.Q̂, estim.R̂, estim.K̂
    nym, nx̂, nσ = estim.nym, estim.nx̂, estim.nσ
    γ, m̂, Ŝ = estim.γ, estim.m̂, estim.Ŝ
    # --- initialize matrices ---
    X̂, X̂_next = Matrix{NT}(undef, nx̂, nσ), Matrix{NT}(undef, nx̂, nσ)
    ŷm = Vector{NT}(undef, nym)
    ŷ  = Vector{NT}(undef, estim.model.ny)
    Ŷm = Matrix{NT}(undef, nym, nσ)
    sqrt_P̂ = LowerTriangular{NT, Matrix{NT}}(Matrix{NT}(undef, nx̂, nx̂))
    # --- correction step ---
    sqrt_P̂.data .= cholesky(P̂).L
    γ_sqrt_P̂ = lmul!(γ, sqrt_P̂)
    X̂ .= x̂
    X̂[:, 2:nx̂+1]   .+= γ_sqrt_P̂
    X̂[:, nx̂+2:end] .-= γ_sqrt_P̂
    for j in axes(Ŷm, 2)
        @views ĥ!(ŷ, estim, estim.model, X̂[:, j], d)
        @views Ŷm[:, j] .= ŷ[estim.i_ym]
    end
    mul!(ŷm, Ŷm, m̂)
    X̄, Ȳm = X̂, Ŷm
    X̄  .= X̂  .- x̂
    Ȳm .= Ŷm .- ŷm
    M̂ = Hermitian(Ȳm * Ŝ * Ȳm' + R̂, :L)
    mul!(K̂, X̄, lmul!(Ŝ, Ȳm'))
    rdiv!(K̂, cholesky(M̂))
    v̂ = ŷm
    v̂ .= ym .- ŷm
    x̂_cor = x̂ + K̂ * v̂
    P̂_cor = P̂ - Hermitian(K̂ * M̂ * K̂', :L)
    # --- prediction step ---
    X̂_cor, sqrt_P̂_cor = X̂, sqrt_P̂
    sqrt_P̂_cor.data .= cholesky(P̂_cor).L
    γ_sqrt_P̂_cor = lmul!(γ, sqrt_P̂_cor)
    X̂_cor .= x̂_cor
    X̂_cor[:, 2:nx̂+1]   .+= γ_sqrt_P̂_cor
    X̂_cor[:, nx̂+2:end] .-= γ_sqrt_P̂_cor
    X̂_next = similar(X̂_cor)
    for j in axes(X̂_next, 2)
        @views f̂!(X̂_next[:, j], estim, estim.model, X̂_cor[:, j], u, d)
    end
    x̂_next = mul!(x̂, X̂_next, m̂)
    X̄_next = X̂_next
    X̄_next .= X̂_next .- x̂_next
    P̂.data .= X̄_next * Ŝ * X̄_next' .+ Q̂ # .data is necessary for Hermitians
    return nothing
end

struct ExtendedKalmanFilter{NT<:Real, SM<:SimModel} <: StateEstimator{NT}
    model::SM
    lastu0::Vector{NT}
    x̂::Vector{NT}
    P̂::Hermitian{NT, Matrix{NT}}
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
    P̂0::Hermitian{NT, Matrix{NT}}
    Q̂::Hermitian{NT, Matrix{NT}}
    R̂::Hermitian{NT, Matrix{NT}}
    K̂::Matrix{NT}
    M̂::Matrix{NT}
    function ExtendedKalmanFilter{NT, SM}(
        model::SM, i_ym, nint_u, nint_ym, P̂0, Q̂, R̂
    ) where {NT<:Real, SM<:SimModel}
        nym, nyu = validate_ym(model, i_ym)
        As, Cs_u, Cs_y, nint_u, nint_ym = init_estimstoch(model, i_ym, nint_u, nint_ym)
        nxs = size(As, 1)
        nx̂  = model.nx + nxs
        Â, B̂u, Ĉ, B̂d, D̂d = augment_model(model, As, Cs_u, Cs_y)
        validate_kfcov(nym, nx̂, Q̂, R̂, P̂0)
        lastu0 = zeros(NT, model.nu)
        x̂ = [zeros(NT, model.nx); zeros(NT, nxs)]
        P̂0 = Hermitian(P̂0, :L)
        Q̂ = Hermitian(Q̂, :L)
        R̂ = Hermitian(R̂, :L)
        P̂ = copy(P̂0)
        K̂, M̂ = zeros(NT, nx̂, nym), zeros(NT, nx̂, nym)
        return new{NT, SM}(
            model,
            lastu0, x̂, P̂, 
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs_u, Cs_y, nint_u, nint_ym,
            Â, B̂u, Ĉ, B̂d, D̂d,
            P̂0, Q̂, R̂,
            K̂, M̂
        )
    end
end

@doc raw"""
    ExtendedKalmanFilter(model::SimModel; <keyword arguments>)

Construct an extended Kalman Filter with the [`SimModel`](@ref) `model`.

Both [`LinModel`](@ref) and [`NonLinModel`](@ref) are supported. The process model is 
identical to [`UnscentedKalmanFilter`](@ref). The Jacobians of the augmented model 
``\mathbf{f̂, ĥ}`` are computed with [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl)
automatic differentiation.

!!! warning
    See the Extended Help of [`linearize`](@ref) function if you get an error like:    
    `MethodError: no method matching (::var"##")(::Vector{ForwardDiff.Dual})`.

# Arguments
- `model::SimModel` : (deterministic) model for the estimations.
- `<keyword arguments>` of [`SteadyKalmanFilter`](@ref) constructor.
- `<keyword arguments>` of [`KalmanFilter`](@ref) constructor.

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_)->0.2x+u, (x,_)->-3x, 5.0, 1, 1, 1, solver=nothing);

julia> estim = ExtendedKalmanFilter(model, σQ=[2], σQint_ym=[2], σP0=[0.1], σP0int_ym=[0.1])
ExtendedKalmanFilter estimator with a sample time Ts = 5.0 s, NonLinModel and:
 1 manipulated inputs u (0 integrating states)
 2 estimated states x̂
 1 measured outputs ym (1 integrating states)
 0 unmeasured outputs yu
 0 measured disturbances d
```
"""
function ExtendedKalmanFilter(
    model::SM;
    i_ym::IntRangeOrVector = 1:model.ny,
    σP0::Vector = fill(1/model.nx, model.nx),
    σQ ::Vector = fill(1/model.nx, model.nx),
    σR ::Vector = fill(1, length(i_ym)),
    nint_u   ::IntVectorOrInt = 0,
    σQint_u  ::Vector = fill(1, max(sum(nint_u), 0)),
    σP0int_u ::Vector = fill(1, max(sum(nint_u), 0)),
    nint_ym  ::IntVectorOrInt = default_nint(model, i_ym, nint_u),
    σQint_ym ::Vector = fill(1, max(sum(nint_ym), 0)),
    σP0int_ym::Vector = fill(1, max(sum(nint_ym), 0)),
) where {NT<:Real, SM<:SimModel{NT}}
    # estimated covariances matrices (variance = σ²) :
    P̂0 = Hermitian(diagm(NT[σP0; σP0int_u; σP0int_ym].^2), :L)
    Q̂  = Hermitian(diagm(NT[σQ;  σQint_u;  σQint_ym ].^2), :L)
    R̂  = Hermitian(diagm(NT[σR;].^2), :L)
    return ExtendedKalmanFilter{NT, SM}(model, i_ym, nint_u, nint_ym, P̂0, Q̂ , R̂)
end

@doc raw"""
    ExtendedKalmanFilter(model, i_ym, nint_u, nint_ym, P̂0, Q̂, R̂)

Construct the estimator from the augmented covariance matrices `P̂0`, `Q̂` and `R̂`.

This syntax allows nonzero off-diagonal elements in ``\mathbf{P̂}_{-1}(0), \mathbf{Q̂, R̂}``.
"""
function ExtendedKalmanFilter(model::SM, i_ym, nint_u, nint_ym,P̂0, Q̂, R̂) where {NT<:Real, SM<:SimModel{NT}}
    P̂0, Q̂, R̂ = to_mat(P̂0), to_mat(Q̂), to_mat(R̂)
    return ExtendedKalmanFilter{NT, SM}(model, i_ym, nint_u, nint_ym, P̂0, Q̂ , R̂)
end


@doc raw"""
    update_estimate!(estim::ExtendedKalmanFilter, u, ym, d=empty(estim.x̂))

Update [`ExtendedKalmanFilter`](@ref) state `estim.x̂` and covariance `estim.P̂`.

The equations are similar to [`update_estimate!(::KalmanFilter)`](@ref) but with the 
substitutions ``\mathbf{Â = F̂}(k)`` and ``\mathbf{Ĉ^m = Ĥ^m}(k)``:
```math
\begin{aligned}
    \mathbf{M̂}(k)       &= \mathbf{P̂}_{k-1}(k)\mathbf{Ĥ^m}'(k)
                           [\mathbf{Ĥ^m}(k)\mathbf{P̂}_{k-1}(k)\mathbf{Ĥ^m}'(k) + \mathbf{R̂}]^{-1}    \\
    \mathbf{K̂}(k)       &= \mathbf{F̂}(k) \mathbf{M̂}(k)                                    \\
    \mathbf{ŷ^m}(k)     &= \mathbf{ĥ^m}\Big( \mathbf{x̂}_{k-1}(k), \mathbf{d}(k) \Big)     \\
    \mathbf{x̂}_{k}(k+1) &= \mathbf{f̂}\Big( \mathbf{x̂}_{k-1}(k), \mathbf{u}(k), \mathbf{d}(k) \Big)
                           + \mathbf{K̂}(k)[\mathbf{y^m}(k) - \mathbf{ŷ^m}(k)]             \\
    \mathbf{P̂}_{k}(k+1) &= \mathbf{F̂}(k)[\mathbf{P̂}_{k-1}(k)
                           - \mathbf{M̂}(k)\mathbf{Ĥ^m}(k)\mathbf{P̂}_{k-1}(k)]\mathbf{F̂}'(k) 
                           + \mathbf{Q̂}
\end{aligned}
```
[`ForwardDiff.jacobian`](https://juliadiff.org/ForwardDiff.jl/stable/user/api/#ForwardDiff.jacobian)
automatically computes the Jacobians:
```math
\begin{aligned}
    \mathbf{F̂}(k) &= \left. \frac{∂\mathbf{f̂}(\mathbf{x̂}, \mathbf{u}, \mathbf{d})}{∂\mathbf{x̂}} \right|_{\mathbf{x̂ = x̂}_{k-1}(k),\, \mathbf{u = u}(k),\, \mathbf{d = d}(k)}  \\
    \mathbf{Ĥ}(k) &= \left. \frac{∂\mathbf{ĥ}(\mathbf{x̂}, \mathbf{d})}{∂\mathbf{x̂}}             \right|_{\mathbf{x̂ = x̂}_{k-1}(k),\, \mathbf{d = d}(k)}
\end{aligned}
```
The matrix ``\mathbf{Ĥ^m}`` is the rows of ``\mathbf{Ĥ}`` that are measured outputs.
"""
function update_estimate!(
    estim::ExtendedKalmanFilter{NT}, u, ym, d=empty(estim.x̂)
) where NT<:Real
    model = estim.model
    x̂next, ŷ = Vector{NT}(undef, estim.nx̂), Vector{NT}(undef, model.ny)
    F̂ = ForwardDiff.jacobian((x̂next, x̂) -> f̂!(x̂next, estim, model, x̂, u, d), x̂next, estim.x̂)
    Ĥ = ForwardDiff.jacobian((ŷ, x̂)     -> ĥ!(ŷ, estim, model, x̂, d), ŷ, estim.x̂)
    return update_estimate_kf!(estim, u, ym, d, F̂, Ĥ[estim.i_ym, :], estim.P̂, estim.x̂)
end

"Set `estim.P̂` to `estim.P̂0` for the time-varying Kalman Filters."
function init_estimate_cov!(
    estim::Union{KalmanFilter, UnscentedKalmanFilter, ExtendedKalmanFilter}, _ , _ , _
) 
    estim.P̂.data .= estim.P̂0 # .data is necessary for Hermitians
    return nothing
end

"""
    validate_kfcov(nym, nx̂, Q̂, R̂, P̂0=nothing)

Validate sizes and Hermitianity of process `Q̂`` and sensor `R̂` noises covariance matrices.

Also validate initial estimate covariance `P̂0`, if provided.
"""
function validate_kfcov(nym, nx̂, Q̂, R̂, P̂0=nothing)
    size(Q̂)  ≠ (nx̂, nx̂)     && error("Q̂ size $(size(Q̂)) ≠ nx̂, nx̂ $((nx̂, nx̂))")
    !ishermitian(Q̂)         && error("Q̂ is not Hermitian")
    size(R̂)  ≠ (nym, nym)   && error("R̂ size $(size(R̂)) ≠ nym, nym $((nym, nym))")
    !ishermitian(R̂)         && error("R̂ is not Hermitian")
    if ~isnothing(P̂0)
        size(P̂0) ≠ (nx̂, nx̂) && error("P̂0 size $(size(P̂0)) ≠ nx̂, nx̂ $((nx̂, nx̂))")
        !ishermitian(P̂0)    && error("P̂0 is not Hermitian")
    end
end

"""
    update_estimate_kf!(estim::StateEstimator, u, ym, d, Â, Ĉm, P̂, x̂=nothing)

Update time-varying/extended Kalman Filter estimates with augmented `Â` and `Ĉm` matrices.

Allows code reuse for [`KalmanFilter`](@ref), [`ExtendedKalmanFilterKalmanFilter`](@ref).
They update the state `x̂` and covariance `P̂` with the same equations. The extended filter
substitutes the augmented model matrices with its Jacobians (`Â = F̂` and `Ĉm = Ĥm`).
The implementation uses in-place operations and explicit factorization to reduce
allocations. See e.g. [`KalmanFilter`](@ref) docstring for the equations. If `isnothing(x̂)`,
only the covariance `P̂` is updated.
"""
function update_estimate_kf!(
    estim::StateEstimator{NT}, u, ym, d, Â, Ĉm, P̂, x̂=nothing
) where NT<:Real
    Q̂, R̂, M̂ = estim.Q̂, estim.R̂, estim.M̂
    mul!(M̂, P̂, Ĉm')
    rdiv!(M̂, cholesky!(Hermitian(Ĉm * P̂ * Ĉm' .+ R̂)))
    if !isnothing(x̂)
        mul!(estim.K̂, Â, M̂)
        x̂next, ŷ = Vector{NT}(undef, estim.nx̂), Vector{NT}(undef, estim.model.ny)
        ĥ!(ŷ, estim, estim.model, x̂, d)
        ŷm = @views ŷ[estim.i_ym]
        v̂  = ŷm
        v̂ .= ym .- ŷm
        f̂!(x̂next, estim, estim.model, x̂, u, d)
        mul!(x̂next, estim.K̂, v̂, 1, 1)
        estim.x̂ .= x̂next
    end
    P̂.data .= Â * (P̂ .- M̂ * Ĉm * P̂) * Â' .+ Q̂ # .data is necessary for Hermitians
    return nothing
end
