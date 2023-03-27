struct SteadyKalmanFilter <: StateEstimator
    model::LinModel
    x̂::Vector{Float64}
    i_ym::IntRangeOrVector
    nx̂::Int
    nym::Int
    nyu::Int
    nxs::Int
    As::Matrix{Float64}
    Cs::Matrix{Float64}
    nint_ym::Vector{Int}
    Â   ::Matrix{Float64}
    B̂u  ::Matrix{Float64}
    B̂d  ::Matrix{Float64}
    Ĉ   ::Matrix{Float64}
    D̂d  ::Matrix{Float64}
    Ĉm  ::Matrix{Float64}
    D̂dm ::Matrix{Float64}
    f̂::Function
    ĥ::Function
    Q̂::Union{Diagonal{Float64}, Matrix{Float64}}
    R̂::Union{Diagonal{Float64}, Matrix{Float64}}
    K::Matrix{Float64}
    function SteadyKalmanFilter(model, i_ym, nint_ym, Asm, Csm, Q̂, R̂)
        nx, ny = model.nx, model.ny
        nym, nyu = length(i_ym), ny - length(i_ym)
        nxs = size(Asm,1)
        nx̂ = nx + nxs
        validate_kfcov(nym, nx̂, Q̂, R̂)
        As, _ , Cs, _  = stoch_ym2y(model, i_ym, Asm, [], Csm, [])
        f̂, ĥ, Â, B̂u, Ĉ, B̂d, D̂d = augment_model(model, As, Cs)
        Ĉm, D̂dm = Ĉ[i_ym, :], D̂d[i_ym, :] # measured outputs ym only
        K = try
            kalman(Discrete, Â, Ĉm, Matrix(Q̂), Matrix(R̂)) # Matrix() required for Julia 1.6
        catch my_error
            if isa(my_error, ErrorException)
                error("Cannot compute the optimal Kalman gain K for the "* 
                      "SteadyKalmanFilter. You may try to remove integrators with nint_ym "*
                      "parameter or use the time-varying KalmanFilter.")
            end
        end
        x̂ = [copy(model.x); zeros(nxs)]
        return new(
            model, 
            x̂,
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs, nint_ym,
            Â, B̂u, B̂d, Ĉ, D̂d, 
            Ĉm, D̂dm,
            f̂, ĥ,
            Q̂, R̂,
            K
        )
    end
end

@doc raw"""
    SteadyKalmanFilter(model::LinModel; <keyword arguments>)

Construct a steady-state Kalman Filter with the [`LinModel`](@ref) `model`.

The steady-state (or asymptotic) Kalman filter is based on the process model :
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
model, which is specified by the numbers of output integrator `nint_ym` (see Extended Help). 
Likewise, the covariance matrices are augmented with ``\mathbf{Q̂ = \text{diag}(Q, Q_{int})}`` 
and ``\mathbf{R̂ = R}``. The matrices ``\mathbf{Ĉ^m, D̂_d^m}`` are the rows of 
``\mathbf{Ĉ, D̂_d}`` that correspond to measured outputs ``\mathbf{y^m}`` (and unmeasured 
ones, for ``\mathbf{Ĉ^u, D̂_d^u}``).

# Arguments
- `model::LinModel` : (deterministic) model for the estimations.
- `i_ym=1:model.ny` : `model` output indices that are measured ``\mathbf{y^m}``, the rest 
    are unmeasured ``\mathbf{y^u}``.
- `σQ=fill(0.1,model.nx)` : main diagonal of the process noise covariance ``\mathbf{Q}`` of
    `model`, specified as a standard deviation vector.
- `σR=fill(0.1,length(i_ym))` : main diagonal of the sensor noise covariance ``\mathbf{R}``
    of `model` measured outputs, specified as a standard deviation vector.
- `nint_ym=fill(1,length(i_ym))` : integrator quantity per measured outputs (vector) for the 
    stochastic model, use `nint_ym=0` for no integrator at all.
- `σQ_int=fill(0.1,sum(nint_ym))` : same than `σQ` but for the stochastic model covariance
    ``\mathbf{Q_{int}}`` (composed of output integrators).

# Examples
```jldoctest
julia> model = LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 0.5);

julia> estim = SteadyKalmanFilter(model, i_ym=[2], σR=[1], σQ_int=[0.01])
SteadyKalmanFilter estimator with a sample time Ts = 0.5 s and:
 1 manipulated inputs u
 3 states x̂
 1 measured outputs ym
 1 unmeasured outputs yu
 0 measured disturbances d
```

# Extended Help
The model augmentation with `nint_ym` vector produces the integral action when the estimator
is used in a controller as state feedback. The default is 1 integrator per measured outputs,
resulting in an offset-free tracking for "step-like" unmeasured output disturbances. Use 2 
integrators for "ramp-like" disturbances. See [`init_estimstoch`](@ref).

!!! tip
    Increasing `σQ_int` values increases the integral action "gain".

The constructor pre-compute the steady-state Kalman gain `K` with the [`kalman`](https://juliacontrol.github.io/ControlSystems.jl/stable/lib/synthesis/#ControlSystemsBase.kalman-Tuple{Any,%20Any,%20Any,%20Any,%20Any,%20Vararg{Any}})
function. It can sometimes fail, for example when `model` is integrating. In such a case,
you can use 0 integrator on `model` integrating outputs, or the alternative time-varying 
[`KalmanFilter`](@ref).
"""
function SteadyKalmanFilter(
    model::LinModel;
    i_ym::IntRangeOrVector = 1:model.ny,
    σQ::Vector{<:Real} = fill(0.1, model.nx),
    σR::Vector{<:Real} = fill(0.1, length(i_ym)),
    nint_ym::IntVectorOrInt = fill(1, length(i_ym)),
    σQ_int::Vector{<:Real} = fill(0.1, max(sum(nint_ym), 0))
)
    if nint_ym == 0 # alias for no output integrator at all :
        nint_ym = fill(0, length(i_ym));
    end
    Asm, Csm = init_estimstoch(i_ym, nint_ym)
    # estimated covariances matrices (variance = σ²) :
    Q̂  = Diagonal{Float64}([σQ   ; σQ_int    ].^2);
    R̂  = Diagonal{Float64}(σR.^2);
    return SteadyKalmanFilter(model, i_ym, nint_ym, Asm, Csm, Q̂ , R̂)
end

@doc raw"""
    updatestate!(estim::SteadyKalmanFilter, u, ym, d=Float64[])

Update `estim.x̂` estimate with current inputs `u`, measured outputs `ym` and dist. `d`.

The [`SteadyKalmanFilter`](@ref) updates it with the precomputed Kalman gain ``\mathbf{K}``:
```math
\mathbf{x̂}_{k}(k+1) = \mathbf{Â x̂}_{k-1}(k) + \mathbf{B̂_u u}(k) + \mathbf{B̂_d d}(k) 
               + \mathbf{K}[\mathbf{y^m}(k) - \mathbf{Ĉ^m x̂}_{k-1}(k) - \mathbf{D̂_d^m d}(k)]
```

# Examples
```jldoctest
julia> kf = SteadyKalmanFilter(LinModel(ss(1, 1, 1, 0, 1)));

julia> x̂ = updatestate!(kf, [1], [0]) # x̂[2] is the integrator state (nint_ym argument)
2-element Vector{Float64}:
 1.0
 0.0
```
"""
function updatestate!(estim::SteadyKalmanFilter, u, ym, d=Float64[])
    u, d, ym = remove_op(estim, u, d, ym)
    Â, B̂u, B̂d, Ĉm, D̂dm = estim.Â, estim.B̂u, estim.B̂d, estim.Ĉm, estim.D̂dm
    x̂, K = estim.x̂, estim.K
    x̂[:] = Â*x̂ + B̂u*u + B̂d*d + K*(ym - Ĉm*x̂ - D̂dm*d)
    return x̂
end


struct KalmanFilter <: StateEstimator
    model::LinModel
    x̂::Vector{Float64}
    P̂::Hermitian{Float64}
    i_ym::IntRangeOrVector
    nx̂::Int
    nym::Int
    nyu::Int
    nxs::Int
    As::Matrix{Float64}
    Cs::Matrix{Float64}
    nint_ym::Vector{Int}
    Â   ::Matrix{Float64}
    B̂u  ::Matrix{Float64}
    B̂d  ::Matrix{Float64}
    Ĉ   ::Matrix{Float64}
    D̂d  ::Matrix{Float64}
    Ĉm  ::Matrix{Float64}
    D̂dm ::Matrix{Float64}
    f̂::Function
    ĥ::Function
    P̂0::Union{Diagonal{Float64}, Hermitian{Float64}}
    Q̂::Union{Diagonal{Float64}, Matrix{Float64}}
    R̂::Union{Diagonal{Float64}, Matrix{Float64}}
    function KalmanFilter(model, i_ym, nint_ym, Asm, Csm, P̂0, Q̂, R̂)
        nx, ny = model.nx, model.ny
        nym, nyu = length(i_ym), ny - length(i_ym)
        nxs = size(Asm,1)
        nx̂ = nx + nxs
        validate_kfcov(nym, nx̂, Q̂, R̂, P̂0)
        As, _ , Cs, _  = stoch_ym2y(model, i_ym, Asm, [], Csm, [])
        f̂, ĥ, Â, B̂u, Ĉ, B̂d, D̂d = augment_model(model, As, Cs)
        Ĉm, D̂dm = Ĉ[i_ym, :], D̂d[i_ym, :] # measured outputs ym only
        x̂ = [copy(model.x); zeros(nxs)]
        P̂ = Hermitian(Matrix(P̂0), :L)
        return new(
            model, 
            x̂, P̂, 
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs, nint_ym,
            Â, B̂u, B̂d, Ĉ, D̂d, 
            Ĉm, D̂dm,
            f̂, ĥ,
            P̂0, Q̂, R̂
        )
    end
end

@doc raw"""
    KalmanFilter(model::LinModel; <keyword arguments>)

Construct a time-varying Kalman Filter with the [`LinModel`](@ref) `model`.

The process model is identical to [`SteadyKalmanFilter`](@ref). The augmented model 
estimation error ``\mathbf{x}(k+1) - \mathbf{x̂}_k(k+1)`` covariance is denoted 
``\mathbf{P̂}_k(k+1)``. Two keyword arguments modify its initial value through
``\mathbf{P̂}_{-1}(0) = \mathrm{diag}\{ \mathbf{P}(0), \mathbf{P_{int}}(0) \}``.


# Arguments
- `model::LinModel` : (deterministic) model for the estimations.
- `σP0=fill(10,model.nx)` : main diagonal of the initial estimate covariance 
    ``\mathbf{P}(0)``, specified as a standard deviation vector.
- `σP0_int=fill(10,sum(nint_ym))` : same than `σP0` but for the stochastic model
    covariance ``\mathbf{P_{int}}(0)`` (composed of output integrators).
- `<keyword arguments>` of [`SteadyKalmanFilter`](@ref) constructor.

# Examples
```jldoctest
julia> model = LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 0.5);

julia> estim = KalmanFilter(model, i_ym=[2], σR=[1], σP0=[100, 100], σQ_int=[0.01])
KalmanFilter estimator with a sample time Ts = 0.5 s and:
 1 manipulated inputs u
 3 states x̂
 1 measured outputs ym
 1 unmeasured outputs yu
 0 measured disturbances d
```
"""
function KalmanFilter(
    model::LinModel;
    i_ym::IntRangeOrVector = 1:model.ny,
    σP0::Vector{<:Real} = fill(10, model.nx),
    σQ::Vector{<:Real} = fill(0.1, model.nx),
    σR::Vector{<:Real} = fill(0.1, length(i_ym)),
    nint_ym::IntVectorOrInt = fill(1, length(i_ym)),
    σP0_int::Vector{<:Real} = fill(10, max(sum(nint_ym), 0)),
    σQ_int::Vector{<:Real} = fill(0.1, max(sum(nint_ym), 0))
)
    if nint_ym == 0 # alias for no output integrator at all :
        nint_ym = fill(0, length(i_ym));
    end
    Asm, Csm = init_estimstoch(i_ym, nint_ym)
    # estimated covariances matrices (variance = σ²) :
    P̂0 = Diagonal{Float64}([σP0  ; σP0_int   ].^2);
    Q̂  = Diagonal{Float64}([σQ   ; σQ_int    ].^2);
    R̂  = Diagonal{Float64}(σR.^2);
    return KalmanFilter(model, i_ym, nint_ym, Asm, Csm, P̂0, Q̂ , R̂)
end

@doc raw"""
    updatestate!(estim::KalmanFilter, u, ym, d=Float64[])

Update `estim.x̂` \ `P̂` with current inputs `u`, measured outputs `ym` and dist. `d`.

See [`updatestate_kf!`](@ref) for the implementation details.
"""
function updatestate!(estim::KalmanFilter, u, ym, d=Float64[])
    u, d, ym = remove_op(estim, u, d, ym) 
    updatestate_kf!(estim, u, ym, d)
    return estim.x̂
end


@doc raw"""
    updatestate_kf!(estim::KalmanFilter, u, ym, d)

Update [`KalmanFilter`](@ref) state `estim.x̂` and estimation error covariance `estim.P̂`.

It implements the time-varying Kalman Filter in its predictor (observer) form :
```math
\begin{aligned}
    \mathbf{M}(k)       &= \mathbf{P̂}_{k-1}(k)\mathbf{Ĉ^m}'
                           [\mathbf{Ĉ^m P̂}_{k-1}(k)\mathbf{Ĉ^m + R̂}]^{-1}                 \\
    \mathbf{K}(k)       &= \mathbf{Â M(k)}                                                \\
    \mathbf{ŷ^m}(k)     &= \mathbf{Ĉ^m x̂}_{k-1}(k) + \mathbf{D̂_d^m d}(k)                  \\
    \mathbf{x̂}_{k}(k+1) &= \mathbf{Â x̂}_{k-1}(k) + \mathbf{B̂_u u}(k) + \mathbf{B̂_d d}(k) 
                           + \mathbf{K}(k)[\mathbf{y^m}(k) - \mathbf{ŷ^m}(k)]             \\
    \mathbf{P̂}_{k}(k+1) &= \mathbf{Â}[\mathbf{P̂}_{k-1}(k) - 
                           \mathbf{M}(k)\mathbf{Ĉ^m P̂}_{k-1}(k)]\mathbf{Â}' + \mathbf{Q̂}
\end{aligned}
```
based on the process model described in [`SteadyKalmanFilter`](@ref). The notation 
``\mathbf{x̂}_{k-1}(k)`` refers to the state for the current time ``k`` estimated at the last 
control period ``k-1``. See [^1] for details.

[^1]: Boyd S., "Lecture 8 : The Kalman Filter" (Winter 2008-09) [course slides], *EE363: 
     Linear Dynamical Systems*, https://web.stanford.edu/class/ee363/lectures/kf.pdf.
"""
function updatestate_kf!(estim::KalmanFilter, u, ym, d)
    Â, B̂u, B̂d, Ĉm, D̂dm = estim.Â, estim.B̂u, estim.B̂d, estim.Ĉm, estim.D̂dm
    x̂, P̂, Q̂, R̂ = estim.x̂, estim.P̂, estim.Q̂, estim.R̂
    M = (P̂ * Ĉm') / (Ĉm * P̂ * Ĉm' + R̂)
    K = Â * M
    x̂[:] = Â * x̂ + B̂u * u + B̂d * d + K * (ym - Ĉm * x̂ - D̂dm * d)
    P̂.data[:] = Â * (P̂ - M * Ĉm * P̂) * Â' + Q̂ # .data is necessary for Hermitian matrices
    return x̂, P̂
end

struct UnscentedKalmanFilter <: StateEstimator
    model::SimModel
    x̂::Vector{Float64}
    P̂::Hermitian{Float64}
    i_ym::IntRangeOrVector
    nx̂::Int
    nym::Int
    nyu::Int
    nxs::Int
    As::Matrix{Float64}
    Cs::Matrix{Float64}
    nint_ym::Vector{Int}
    f̂::Function
    ĥ::Function
    P̂0::Union{Diagonal{Float64}, Hermitian{Float64}}
    Q̂::Union{Diagonal{Float64}, Matrix{Float64}}
    R̂::Union{Diagonal{Float64}, Matrix{Float64}}
    nσ::Int 
    γ::Float64
    m̂::Vector{Float64}
    Ŝ::Diagonal{Float64}
    function UnscentedKalmanFilter(model, i_ym, nint_ym, Asm, Csm, P̂0, Q̂, R̂, α, β, κ)
        nx, ny = model.nx, model.ny
        nym, nyu = length(i_ym), ny - length(i_ym)
        nxs = size(Asm,1)
        nx̂ = nx + nxs
        validate_kfcov(nym, nx̂, Q̂, R̂, P̂0)
        As, _ , Cs, _  = stoch_ym2y(model, i_ym, Asm, [], Csm, [])
        f̂, ĥ = augment_model(model, As, Cs)
        nσ, γ, m̂, Ŝ = init_ukf(nx̂, α, β, κ)
        x̂ = [copy(model.x); zeros(nxs)]
        P̂ = Hermitian(Matrix(P̂0), :L)
        return new(
            model,
            x̂, P̂, 
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs, nint_ym,
            f̂, ĥ,
            P̂0, Q̂, R̂,
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
    \mathbf{x}(k+1) &= \mathbf{f̂}\Big(\mathbf{x̂}(k), \mathbf{u}(k), \mathbf{d}(k)\Big) 
                        + \mathbf{w}(k)                                                   \\
    \mathbf{y^m}(k) &= \mathbf{ĥ^m}\Big(\mathbf{x̂}(k), \mathbf{d}(k)\Big) + \mathbf{v}(k) \\
    \mathbf{y^u}(k) &= \mathbf{ĥ^u}\Big(\mathbf{x̂}(k), \mathbf{d}(k)\Big)                 \\
\end{aligned}
```
See [`SteadyKalmanFilter`](@ref) for details on ``\mathbf{v}(k), \mathbf{w}(k)`` noises and 
``\mathbf{R̂}, \mathbf{Q̂}`` covariances. The functions ``\mathbf{f̂, ĥ}`` are `model` 
state-space functions augmented with the stochastic model, which is specified by the numbers
of output integrator `nint_ym` (see [`SteadyKalmanFilter`](@ref) for details). The 
``\mathbf{ĥ^m}`` function represents the measured outputs of ``\mathbf{ĥ}`` function (and 
unmeasured ones, for ``\mathbf{ĥ^u}``).

# Arguments
- `model::SimModel` : (deterministic) model for the estimations.
- `α=1e-3` : alpha parameter, spread of the state distribution ``(0 ≤ α ≤ 1)``.
- `β=2` : beta parameter, skewness and kurtosis of the states distribution ``(β ≥ 0)``.
- `κ=0` : kappa parameter, another spread parameter ``(0 ≤ κ ≤ 3)``.
- `<keyword arguments>` of [`SteadyKalmanFilter`](@ref) constructor.
- `<keyword arguments>` of [`KalmanFilter`](@ref) constructor.

# Examples
```jldoctest
julia> a = 1;

```
"""
function UnscentedKalmanFilter(
    model::SimModel;
    i_ym::IntRangeOrVector = 1:model.ny,
    σP0::Vector{<:Real} = fill(10, model.nx),
    σQ::Vector{<:Real} = fill(0.1, model.nx),
    σR::Vector{<:Real} = fill(0.1, length(i_ym)),
    nint_ym::IntVectorOrInt = fill(1, length(i_ym)),
    σP0_int::Vector{<:Real} = fill(10, max(sum(nint_ym), 0)),
    σQ_int::Vector{<:Real} = fill(0.1, max(sum(nint_ym), 0)),
    α::Real = 1e-3,
    β::Real = 2,
    κ::Real = 0
    )
    if nint_ym == 0 # alias for no output integrator at all :
        nint_ym = fill(0, length(i_ym));
    end
    Asm, Csm = init_estimstoch(i_ym, nint_ym)
    # estimated covariances matrices (variance = σ²) :
    P̂0 = Diagonal{Float64}([σP0  ; σP0_int   ].^2);
    Q̂  = Diagonal{Float64}([σQ   ; σQ_int    ].^2);
    R̂  = Diagonal{Float64}(σR.^2);
    return UnscentedKalmanFilter(model, i_ym, nint_ym, Asm, Csm, P̂0, Q̂ , R̂, α, β, κ)
end

@doc raw"""
    init_ukf(nx̂, α, β, κ)

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
See [`updatestate_ukf!`](@ref) for other details.
"""
function init_ukf(nx̂, α, β, κ)
    nσ = 2nx̂ + 1                                  # number of sigma points
    γ = α * √(nx̂ + κ)                             # constant factor of standard deviation √P
    m̂_0 = 1 - nx̂ / α^2 / (nx̂ + κ)
    Ŝ_0 = m̂_0 + 1 - α^2 + β
    w = 1 / 2 / α^2 / (nx̂ + κ)
    m̂ = [m̂_0; fill(w, 2 * nx̂)]                    # weights for means
    Ŝ = Diagonal{Float64}([Ŝ_0; fill(w, 2 * nx̂)]) # weights for covariances
    return nσ, γ, m̂, Ŝ
end
  

@doc raw"""
    updatestate!(estim::UnscentedKalmanFilter, u, ym, d=Float64[])

Same than `KalmanFilter` but using the unscented estimator.

See [`updatestate_ukf!`](@ref) for the implementation details.
"""
function updatestate!(estim::UnscentedKalmanFilter, u, ym, d=Float64[])
    u, d, ym = remove_op(estim, u, d, ym) 
    updatestate_ukf!(estim, u, ym, d)
    return estim.x̂
end

@doc raw"""
    updatestate_ukf!(estim::UnscentedKalmanFilter, u, ym, d)
    
Update [`UnscentedKalmanFilter`](@ref) state `estim.x̂` and covariance estimate `estim.P̂`.

It implements the unscented Kalman Filter in its predictor (observer) form, based on the 
generalized unscented transform[^2]. See [`init_ukf`](@ref) for the definition of the 
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
    \mathbf{M}(k)       &= \mathbf{Ȳ^m}(k) \mathbf{Ŝ} \mathbf{Ȳ^m}'(k) + \mathbf{R̂} \\
    \mathbf{K}(k)       &= \mathbf{X̄}_{k-1}(k) \mathbf{Ŝ} \mathbf{Ȳ^m}'(k) \mathbf{M}(k)^{-1} \\
    \mathbf{x̂}_k(k)     &= \mathbf{x̂}_{k-1}(k) + \mathbf{K}(k) \big[ \mathbf{y^m}(k) - \mathbf{ŷ^m}(k) \big] \\
    \mathbf{P̂}_k(k)     &= \mathbf{P̂}_{k-1}(k) - \mathbf{K}(k) \mathbf{M}(k) \mathbf{K}'(k) \\
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

[^2]: Simon, D. 2006, "Chapter 14: The unscented Kalman filter" in "Optimal State Estimation: 
     Kalman, H∞, and Nonlinear Approaches", John Wiley & Sons, p. 433–459, https://doi.org/10.1002/0470045345.ch14, 
     ISBN9780470045343.
"""
function updatestate_ukf!(estim::UnscentedKalmanFilter, u, ym, d)
    f̂, ĥ = estim.f̂, estim.ĥ
    x̂, P̂, Q̂, R̂ = estim.x̂, estim.P̂, estim.Q̂, estim.R̂
    nym, nx̂, nσ = estim.nym, estim.nx̂, estim.nσ
    γ, m̂, Ŝ = estim.γ, estim.m̂, estim.Ŝ
    # --- correction step ---
    sqrt_P̂ = cholesky(P̂).L
    X̂ = repeat(x̂, 1, nσ) + [zeros(nx̂) +γ*sqrt_P̂ -γ*sqrt_P̂]
    Ŷm = Matrix{Float64}(undef, nym, nσ)
    for j in axes(Ŷm, 2)
        Ŷm[:, j] = ĥ(X̂[:, j], d)[estim.i_ym]
    end
    ŷm = Ŷm * m̂
    X̄ = X̂ .- x̂
    Ȳm = Ŷm .- ŷm
    M = Hermitian(Ȳm * Ŝ * Ȳm' + R̂, :L)
    K = X̄ * Ŝ * Ȳm' / M
    x̂_cor = x̂ + K * (ym - ŷm)
    P̂_cor = P̂ - Hermitian(K * M * K', :L)
    # --- prediction step ---
    sqrt_P̂_cor = cholesky(P̂_cor).L
    X̂_cor = repeat(x̂_cor, 1, nσ) + [zeros(nx̂) +γ*sqrt_P̂_cor -γ*sqrt_P̂_cor]
    X̂_next = Matrix{Float64}(undef, nx̂, nσ)
    for j in axes(X̂_next, 2)
        X̂_next[:, j] = f̂(X̂_cor[:, j], u, d)
    end
    x̂[:] = X̂_next * m̂
    X̄_next = X̂_next .- x̂
    P̂.data[:] = X̄_next * Ŝ * X̄_next' + Q̂ # .data is necessary for Hermitian matrices
    return x̂, P̂
end


"""
    initstate!(estim::{KalmanFilter, UnscentedKalmanFilter}, u, ym, d=Float64[])

Initialize covariance `estim.P̂` and invoke [`initstate!(::StateEstimator)`](@ref).
"""
function initstate!(estim::Union{KalmanFilter, UnscentedKalmanFilter}, u, ym, d=Float64[])
    estim.P̂.data[:] = estim.P̂0 # .data is necessary for Hermitian matrices
    invoke(initstate!, Tuple{StateEstimator, Any, Any, Any}, estim, u, ym, d)
end


"""
    validate_kfcov(nym, nx̂, Q̂, R̂, P̂0=nothing)

Validate sizes of process Q̂ and sensor R̂ noises covariance matrices.

Also validate initial estimate covariance size, if provided.
"""
function validate_kfcov(nym, nx̂, Q̂, R̂, P̂0=nothing)
    size(Q̂)  ≠ (nx̂, nx̂)     && error("Q̂ size $(size(Q̂)) ≠ nx̂, nx̂ $((nx̂, nx̂))")
    size(R̂)  ≠ (nym, nym)   && error("R̂ size $(size(R̂)) ≠ nym, nym $((nym, nym))")
    if ~isnothing(P̂0)
        size(P̂0) ≠ (nx̂, nx̂) && error("P̂0 size $(size(P̂0)) ≠ nx̂, nx̂ $((nx̂, nx̂))")
    end
end



