struct ManualEstimator{NT<:Real, SM<:SimModel} <: StateEstimator{NT}
    model::SM
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
    direct::Bool
    corrected::Vector{Bool}
    buffer::StateEstimatorBuffer{NT}
    function ManualEstimator{NT}(
        model::SM, i_ym, nint_u, nint_ym
    ) where {NT<:Real, SM<:SimModel{NT}}
        nu, ny, nd, nk = model.nu, model.ny, model.nd, model.nk
        nym, nyu = validate_ym(model, i_ym)
        As, Cs_u, Cs_y, nint_u, nint_ym = init_estimstoch(model, i_ym, nint_u, nint_ym)
        nxs = size(As, 1)
        nx̂  = model.nx + nxs
        Â, B̂u, Ĉ, B̂d, D̂d, x̂op, f̂op = augment_model(model, As, Cs_u, Cs_y)
        Ĉm, D̂dm = Ĉ[i_ym, :], D̂d[i_ym, :]
        x̂0  = [zeros(NT, model.nx); zeros(NT, nxs)]
        direct = false
        corrected = [true]
        buffer = StateEstimatorBuffer{NT}(nu, nx̂, nym, ny, nd, nk)
        return new{NT, SM}(
            model,
            x̂op, f̂op, x̂0,
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs_u, Cs_y, nint_u, nint_ym,
            Â, B̂u, Ĉ, B̂d, D̂d, Ĉm, D̂dm,
            direct, corrected,
            buffer
        )
    end
end

@doc raw"""
    ManualEstimator(model::SimModel; <keyword arguments>)

Construct a manual state estimator for `model` ([`LinModel`](@ref) or [`NonLinModel`](@ref)).

This [`StateEstimator`](@ref) type allows the construction of [`PredictiveController`](@ref)
objects but turns off the built-in state estimation. The user must manually provides the
estimate ``\mathbf{x̂}_{k}(k)`` or ``\mathbf{x̂}_{k-1}(k)`` through [`setstate!`](@ref) at
each time step. Calling [`preparestate!`](@ref) and [`updatestate!`](@ref) will not modify
the estimate. See Extended Help for usage examples.

# Arguments
- `model::SimModel` : (deterministic) model for the estimations.
- `i_ym=1:model.ny` : `model` output indices that are measured ``\mathbf{y^m}``, the rest 
    are unmeasured ``\mathbf{y^u}``.
- `nint_u=0`: integrator quantity for the stochastic model of the unmeasured disturbances at
    the manipulated inputs (vector), use `nint_u=0` for no integrator (see Extended Help).
- `nint_ym=default_nint(model,i_ym,nint_u)` : same than `nint_u` but for the unmeasured 
    disturbances at the measured outputs, use `nint_ym=0` for no integrator (see Extended Help).

# Examples
```jldoctest
julia> model = LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 0.5);

julia> estim = ManualEstimator(model, nint_ym=0) # disable augmentation with integrators
ManualEstimator estimator with a sample time Ts = 0.5 s, LinModel and:
 1 manipulated inputs u (0 integrating states)
 2 estimated states x̂
 2 measured outputs ym (0 integrating states)
 0 unmeasured outputs yu
 0 measured disturbances d
```

# Extended Help
!!! details "Extended Help"
    A first use case is a linear predictive controller based on nonlinear state estimation.
    The [`ManualEstimator`](@ref) serves as a wrapper to provide the minimal required
    information to construct a [`PredictiveController`](@ref). e.g.:

    ```julia
    a=1
    ```
"""
function ManualEstimator(
    model::SM;
    i_ym::AbstractVector{Int} = 1:model.ny,
    nint_u ::IntVectorOrInt = 0,
    nint_ym::IntVectorOrInt = default_nint(model, i_ym, nint_u),
) where {NT<:Real, SM<:SimModel{NT}}
    return ManualEstimator{NT}(model, i_ym, nint_u, nint_ym)
end

"""
    update_estimate!(estim::ManualEstimator, y0m, d0, u0)

Do nothing for [`ManualEstimator`](@ref).
"""
update_estimate!(estim::ManualEstimator, y0m, d0, u0) = nothing