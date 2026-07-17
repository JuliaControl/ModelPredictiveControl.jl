struct Luenberger{NT<:Real, SM<:LinModel} <: StateEstimator{NT}
    model::SM
    x̂op::Vector{NT}
    f̂op::Vector{NT}
    x̂0 ::Vector{NT}
    i_ym::Vector{Int}
    nx̂::Int
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
    function Luenberger{NT, SM}(
        model, i_ym, nint_u, nint_ym, poles; direct=true
    ) where {NT<:Real, SM<:LinModel}
        nu, ny, nd, nk = model.nu, model.ny, model.nd, model.nk
        nym, nyu = validate_ym(model, i_ym)
        validate_luenberger(model, nint_u, nint_ym, poles)
        As, Cs_u, Cs_y, nint_u, nint_ym = init_estimstoch(model, i_ym, nint_u, nint_ym)
        nxs = size(As, 1)
        nx̂  = model.nx + nxs
        Â, B̂u, Ĉ, B̂d, D̂d, x̂op, f̂op = augment_model(model, As, Cs_u, Cs_y)
        Ĉm, D̂dm = Ĉ[i_ym, :], D̂d[i_ym, :]
        K̂ = try
            ControlSystemsBase.place(Â, Ĉ, poles, :o; direct)[:, i_ym]
        catch
            error("Cannot compute the Luenberger gain K̂ with specified poles.")
        end
        x̂0 = [zeros(NT, model.nx); zeros(NT, nxs)]
        corrected = [false]
        buffer = StateEstimatorBuffer{NT}(nu, nx̂, nym, ny, nd, nk)
        return new{NT, SM}(
            model, 
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
    Luenberger(
        model::LinModel; 
        i_ym = 1:model.ny, 
        nint_u  = 0,
        nint_ym = default_nint(model, i_ym),
        poles = 1e-3*(1:(model.nx + sum(nint_u) + sum(nint_ym))) .+ 0.5,
        direct = true
    )

Construct a Luenberger observer with the [`LinModel`](@ref) `model`.

`i_ym` provides the `model` output indices that are measured ``\mathbf{y^m}``, the rest are
unmeasured ``\mathbf{y^u}``. `model` matrices are augmented with the stochastic model, which
is specified by the numbers of integrator `nint_u` and `nint_ym` (see [`SteadyKalmanFilter`](@ref)
Extended Help). The argument `poles` is a vector of `model.nx + sum(nint_u) + sum(nint_ym)`
elements specifying the observer poles/eigenvalues (near ``z=0.5`` by default). The observer
is constructed with a direct transmission from ``\mathbf{y^m}`` if `direct=true` (a.k.a. 
current observers, in opposition to the delayed/prediction form). The method computes the
observer gain `K̂` with [`place`](@extref ControlSystemsBase.place) function. This estimator
is allocation-free.

# Examples
```jldoctest
julia> model = LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 0.5);

julia> estim = Luenberger(model, nint_ym=[1, 1], poles=[0.61, 0.62, 0.63, 0.64])
Luenberger estimator with a sample time Ts = 0.5 s:
├ model: LinModel
├ direct: true
└ dimensions:
  ├ 1 manipulated inputs u (0 integrating states)
  ├ 4 estimated states x̂
  ├ 2 measured outputs ym (2 integrating states)
  ├ 0 unmeasured outputs yu
  └ 0 measured disturbances d
```
"""
function Luenberger(
    model::SM;
    i_ym::AbstractVector{Int}  = 1:model.ny,
    nint_u ::IntVectorOrInt = 0,
    nint_ym::IntVectorOrInt = default_nint(model, i_ym, nint_u),
    poles = 1e-3*(1:(model.nx + sum(nint_u) + sum(nint_ym))) .+ 0.5,
    direct = true
) where{NT<:Real, SM<:LinModel{NT}}
    return Luenberger{NT, SM}(model, i_ym, nint_u, nint_ym, poles; direct)
end

"Validate the quantity and stability of the Luenberger `poles`."
function validate_luenberger(model, nint_u, nint_ym, poles)
    if length(poles) ≠ model.nx + sum(nint_u) +  sum(nint_ym)
        error("poles length ($(length(poles))) ≠ nx ($(model.nx)) + "*
              "integrator quantity ($(sum(nint_ym)))")
    end
    any(abs.(poles) .≥ 1) && error("Observer poles should be inside the unit circles.")
end

@doc raw"""
    correct_estimate!(estim::Union{SteadyKalmanFilter, Luenberger}, y0m, d0)

Correct `estim.x̂0` estimate with current measured outputs `y0m` and disturbances `d0`.

The computations are identical for both [`SteadyKalmanFilter`](@ref) and [`Luenberger`](@ref)
state estimators. It will corrects the state estimate with the precomputed Kalman/observer
gain ``\mathbf{K̂}``. The correction and prediction step equations are provided below.

# Correction Step
```math
\mathbf{x̂}_k(k) = \mathbf{x̂}_{k-1}(k) + \mathbf{K̂}[\mathbf{y^m}(k) - \mathbf{Ĉ^m x̂}_{k-1}(k)
                                                                   - \mathbf{D̂_d^m d}(k)    ]
```

# Prediction Step
```math
\mathbf{x̂}_{k}(k+1) = \mathbf{Â x̂}_{k}(k) + \mathbf{B̂_u u}(k) + \mathbf{B̂_d d}(k) 
```
"""
function correct_estimate!(estim::Union{SteadyKalmanFilter, Luenberger}, y0m, d0)
    Ĉm, D̂dm, K̂ = estim.Ĉm, estim.D̂dm, estim.K̂
    ŷ0m = @views estim.buffer.ŷ[estim.i_ym]
    # in-place operations to reduce allocations:
    mul!(ŷ0m, Ĉm, estim.x̂0) 
    mul!(ŷ0m, D̂dm, d0, 1, 1)
    v̂  = ŷ0m
    v̂ .= y0m .- ŷ0m
    x̂0corr = estim.x̂0
    mul!(x̂0corr, K̂, v̂, 1, 1)
    return nothing
end

"""
    predict_estimate!(estim::Union{SteadyKalmanFilter, Luenberger}, u0, d0)

Prediction step of [`SteadyKalmanFilter`](@ref) and [`Luenberger`](@ref), see [`correct_estimate!`](@ref).
"""
function predict_estimate!(estim::Union{SteadyKalmanFilter, Luenberger}, u0, d0)
    x̂0corr = estim.x̂0
    Â, B̂u, B̂d = estim.Â, estim.B̂u, estim.B̂d
    x̂0next = estim.buffer.x̂
    # in-place operations to reduce allocations:
    mul!(x̂0next, Â, x̂0corr)
    mul!(x̂0next, B̂u, u0, 1, 1)
    mul!(x̂0next, B̂d, d0, 1, 1)
    x̂0next  .+= estim.f̂op .- estim.x̂op
    estim.x̂0 .= x̂0next
    return nothing
end

"Throw an error if P̂ != nothing."
function setstate_cov!(::Luenberger, P̂)
    isnothing(P̂) || error("Luenberger does not compute an estimation covariance matrix P̂.")
    return nothing
end

"Throw an error if `setmodel!` is called on `Luenberger` observer w/o the default values."
function setmodel_estimator!(estim::Luenberger, model, args...)
    if estim.model !== model
        error("Luenberger does not support setmodel!")
    end
    return nothing
end