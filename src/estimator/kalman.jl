const IntVectorOrInt = Union{Int, Vector{Int}}

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
    Â   ::Matrix{Float64}
    B̂u  ::Matrix{Float64}
    B̂d  ::Matrix{Float64}
    Ĉ   ::Matrix{Float64}
    D̂d  ::Matrix{Float64}
    Ĉm  ::Matrix{Float64}
    D̂dm ::Matrix{Float64}
    Q̂   ::Union{Diagonal{Float64}, Matrix{Float64}}
    R̂   ::Union{Diagonal{Float64}, Matrix{Float64}}
    Ko  ::Matrix{Float64}
    function SteadyKalmanFilter(model, i_ym, Asm, Csm, Q̂, R̂)
        nx, ny = model.nx, model.ny
        nym = length(i_ym)
        nyu = ny - nym
        nxs = size(Asm,1)
        nx̂ = nx + nxs
        validate_kfcov(nym, nx̂, Q̂, R̂)
        As, _ , Cs, _  = stoch_ym2y(model, i_ym, Asm, [], Csm, [])
        Â, B̂u, Ĉ, B̂d, D̂d = augment_model(model, As, Cs)
        Ĉm, D̂dm = Ĉ[i_ym, :], D̂d[i_ym, :] # measured outputs ym only
        Ko = kalman(Discrete, Â, Ĉm, Q̂, R̂)
        x̂ = zeros(nx̂)
        return new(
            model, 
            x̂,
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs,
            Â, B̂u, B̂d, Ĉ, D̂d, 
            Ĉm, D̂dm,
            Q̂, R̂,
            Ko
        )
    end
end

@doc raw"""
    SteadyKalmanFilter(model::LinModel; <keyword arguments>)

Construct a `SteadyKalmanFilter` (steady-state or asymptotic) based on `model`.

The process model is the stochastic system :
```math
\begin{aligned}
    \mathbf{x}(k+1) &= 
            \mathbf{Â x}(k) + \mathbf{B̂_u u}(k) + \mathbf{B̂_d d}(k) + \mathbf{w}(k) \\
    \mathbf{y^m}(k) &= \mathbf{Ĉ^m x}(k) + \mathbf{D̂_d^m d}(k) + \mathbf{v}(k) \\
    \mathbf{y^u}(k) &= \mathbf{Ĉ^u x}(k) + \mathbf{D̂_d^u d}(k)
\end{aligned}
```
with sensor ``\mathbf{v}(k)`` and process ``\mathbf{w}(k)`` noises as zero mean white noise 
processes, with a respective covariance of ``\mathbf{R}`` and ``\mathbf{Q}``. The arguments 
are in standard deviations σ, i.e. same units than outputs and states. The matrices 
``\mathbf{Â, B̂_u, B̂_d, Ĉ, D̂_d}`` are `model` matrices augmented with the stochastic model, 
which is specified by the numbers of output integrator `nint_ym`. ``\mathbf{Ĉ^m, D̂_d^m}`` 
are the rows of ``\mathbf{Ĉ, D̂_d}`` that correspond to measured outputs ``\mathbf{y^m}`` 
(or unmeasured ``\mathbf{y^u}``, for ``\mathbf{Ĉ^u, D̂_d^u}``)

See also [`LinModel`](@ref).

# Arguments
- `model::LinModel` : (deterministic) model for the estimations.
- `i_ym=1:model.ny` : `model` output indices that are measured ``\mathbf{y^m}``, the rest 
    are unmeasured ``\mathbf{y^u}``.
- `σQ=fill(0.1,model.nx)` : standard deviation vector for the process noise covariance 
    ``\mathbf{Q}`` of `model`.
- `σR=fill(0.1,length(i_ym))` : standard deviation vector for the sensor noise covariance 
    ``\mathbf{R}`` of `model` measured outputs.
- `nint_ym=fill(1,length(i_ym))` : integrator quantity per measured outputs (vector) for the 
    stochastic model, use `nint_ym=0` for no integrator at all.
- `σQ_int=fill(0.1,sum(nint_ym))` : standard deviation vector for the process noise 
    covariance of the stochastic model (composed of output integrators).

# Extended Help
The model augmentation with `nint_ym` vector produces the integral action when the estimator
is used in a controller as state feedback. More than 1 integrator per measured outputs is 
interesting only when `model` is integrating or unstable, or when the unmeasured 
output disturbances are "ramp-like". See [`augment_model`](@ref).

!!! tip
    Increasing `σQ_int` values increases the integral action "gain".
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
    return SteadyKalmanFilter(model, i_ym, Asm, Csm, Q̂ , R̂)
end

@doc raw"""
    updatestate!(estim::SteadyKalmanFilter, u, ym, d=Float64[])

Update `estim.x̂` with current inputs `u`, measured outputs `ym` and dist. `d`.
"""
function updatestate!(estim::SteadyKalmanFilter, u, ym, d=Float64[])
    u, d, ym = remove_op(estim, u, d, ym)
    A, Bu, Bd, C, Dd = estim.Â, estim.B̂u, estim.B̂d, estim.Ĉm, estim.D̂dm
    x̂, Ko = estim.x̂, estim.Ko
    x̂[:] = A*x̂ + Bu*u + Bd*d + Ko*(ym - C*x̂ - Dd*d)
    return x̂
end


struct KalmanFilter <: StateEstimator
    model::LinModel
    x̂::Vector{Float64}
    P̂::Matrix{Float64}
    i_ym::IntRangeOrVector
    nx̂::Int
    nym::Int
    nyu::Int
    nxs::Int
    As::Matrix{Float64}
    Cs::Matrix{Float64}
    Â   ::Matrix{Float64}
    B̂u  ::Matrix{Float64}
    B̂d  ::Matrix{Float64}
    Ĉ   ::Matrix{Float64}
    D̂d  ::Matrix{Float64}
    Ĉm  ::Matrix{Float64}
    D̂dm ::Matrix{Float64}
    P̂0  ::Union{Diagonal{Float64}, Matrix{Float64}}
    Q̂   ::Union{Diagonal{Float64}, Matrix{Float64}}
    R̂   ::Union{Diagonal{Float64}, Matrix{Float64}}
    function KalmanFilter(model, i_ym, Asm, Csm, P̂0, Q̂, R̂)
        nx, ny = model.nx, model.ny
        nym = length(i_ym);
        nyu = ny - nym;
        nxs = size(Asm,1)
        nx̂ = nx + nxs
        validate_kfcov(nym, nx̂, Q̂, R̂, P̂0)
        As, _ , Cs, _  = stoch_ym2y(model, i_ym, Asm, [], Csm, [])
        Â, B̂u, Ĉ, B̂d, D̂d = augment_model(model, As, Cs)
        Ĉm, D̂dm = Ĉ[i_ym, :], D̂d[i_ym, :] # measured outputs ym only
        x̂ = zeros(nx̂)
        P̂ = P̂0
        return new(
            model, 
            x̂, P̂, 
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs,
            Â, B̂u, B̂d, Ĉ, D̂d, 
            Ĉm, D̂dm,
            P̂0, Q̂, R̂
        )
    end
end

@doc raw"""
    KalmanFilter(model::LinModel; <keyword arguments>)

Construct a time-varying `KalmanFilter` based on `model`.

The process model is identical to [`SteadyKalmanFilter`](@ref).

# Arguments
- `model::LinModel` : (deterministic) model for the estimations.
- `σP0=fill(10,model.nx)` : standard deviation vector for the initial estimate covariance 
    ``\mathbf{P}(0)`` of `model`.
- `σP0_int=fill(10,sum(nint_ym))` : standard deviation vector for the initial estimate 
    covariance of the stochastic model (composed of output integrators).
- `<keyword arguments>` of [`SteadyKalmanFilter`](@ref)
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
    return KalmanFilter(model, i_ym, Asm, Csm, P̂0, Q̂ , R̂)
end

@doc raw"""
    updatestate!(estim::KalmanFilter, u, ym, d=Float64[])

Update `estim.x̂` \ `P̂` with current inputs `u`, measured outputs `ym` and dist. `d`.
"""
function updatestate!(estim::KalmanFilter, u, ym, d=Float64[])
    u, d, ym = remove_op(estim, u, d, ym)
    A, Bu, Bd, C, Dd = estim.Â, estim.B̂u, estim.B̂d, estim.Ĉm, estim.D̂dm
    x̂, P̂, Q̂, R̂ = estim.x̂, estim.P̂, estim.Q̂, estim.R̂ 
    # --- observer gain calculation ---
    M  = (P̂*C')/(C*P̂*C'+R̂)
    Ko = A*M
    # --- next state calculation ---
    x̂[:] = A*x̂ + Bu*u + Bd*d + Ko*(ym - C*x̂ - Dd*d)
    # --- next estimation error covariance calculation ---
    P̂[:] = A*(P̂-M*C*P̂)*A' + Q̂ 
    return x̂
end


@doc raw"""
    evaloutput(estim::Union{SteadyKalmanFilter, KalmanFilter}, d=Float64[])

Evaluate Kalman filter outputs `̂ŷ` from `estim.x̂` states and current disturbances `d`.
"""
function evaloutput(estim::Union{SteadyKalmanFilter, KalmanFilter}, d=Float64[])
    return estim.Ĉ*estim.x̂ + estim.D̂d*(d - estim.model.dop) + estim.model.yop
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



