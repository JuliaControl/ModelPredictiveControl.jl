struct InternalModel{NT<:Real, SM<:SimModel} <: StateEstimator{NT}
    model::SM
    lastu0::Vector{NT}
    x̂::Vector{NT}
    x̂d::Vector{NT}
    x̂s::Vector{NT}
    i_ym::Vector{Int}
    nx̂::Int
    nym::Int
    nyu::Int
    nxs::Int
    As::Matrix{NT}
    Bs::Matrix{NT}
    Cs::Matrix{NT}
    Ds::Matrix{NT}
    Â ::Matrix{NT}
    B̂u::Matrix{NT}
    Ĉ ::Matrix{NT}
    B̂d::Matrix{NT}
    D̂d::Matrix{NT}
    Âs::Matrix{NT}
    B̂s::Matrix{NT}
    function InternalModel{NT, SM}(
        model::SM, i_ym, Asm, Bsm, Csm, Dsm
    ) where {NT<:Real, SM<:SimModel}
        nym, nyu = validate_ym(model, i_ym)
        validate_internalmodel(model, nym, Csm, Dsm)
        As, Bs, Cs, Ds = stoch_ym2y(model, i_ym, Asm, Bsm, Csm, Dsm)
        nxs = size(As,1)
        nx̂ = model.nx
        Â, B̂u, Ĉ, B̂d, D̂d = matrices_internalmodel(model)
        Âs, B̂s = init_internalmodel(As, Bs, Cs, Ds)
        lastu0 = zeros(NT, model.nu)
        x̂d = x̂ = zeros(NT, model.nx) # x̂ and x̂d are same object (updating x̂d will update x̂)
        x̂s = zeros(NT, nxs)
        return new{NT, SM}(
            model, 
            lastu0, x̂, x̂d, x̂s, 
            i_ym, nx̂, nym, nyu, nxs, 
            As, Bs, Cs, Ds, 
            Â, B̂u, Ĉ, B̂d, D̂d,
            Âs, B̂s
        )
    end
end

@doc raw"""
    InternalModel(model::SimModel; i_ym=1:model.ny, stoch_ym=ss(I,I,I,I,model.Ts))

Construct an internal model estimator based on `model` ([`LinModel`](@ref) or [`NonLinModel`](@ref)).

`i_ym` provides the `model` output indices that are measured ``\mathbf{y^m}``, the rest are 
unmeasured ``\mathbf{y^u}``. `model` evaluates the deterministic predictions 
``\mathbf{ŷ_d}``, and `stoch_ym`, the stochastic predictions of the measured outputs 
``\mathbf{ŷ_s^m}`` (the unmeasured ones being ``\mathbf{ŷ_s^u=0}``). The predicted outputs
sum both values : ``\mathbf{ŷ = ŷ_d + ŷ_s}``.

!!! warning
    `InternalModel` estimator does not work if `model` is integrating or unstable. The 
    constructor verifies these aspects for `LinModel` but not for `NonLinModel`. Uses any 
    other state estimator in such cases.

# Examples
```jldoctest
julia> estim = InternalModel(LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 0.5), i_ym=[2])
InternalModel estimator with a sample time Ts = 0.5 s, LinModel and:
 1 manipulated inputs u
 2 estimated states x̂
 1 measured outputs ym
 1 unmeasured outputs yu
 0 measured disturbances d
```

# Extended Help
!!! details "Extended Help"
    `stoch_ym` is a `TransferFunction` or `StateSpace` object that models disturbances on
    ``\mathbf{y^m}``. Its input is a hypothetical zero mean white noise vector. `stoch_ym` 
    supposes 1 integrator per measured outputs by default, assuming that the current stochastic
    estimate ``\mathbf{ŷ_s^m}(k) = \mathbf{y^m}(k) - \mathbf{ŷ_d^m}(k)`` is constant in the 
    future. This is the dynamic matrix control (DMC) strategy, which is simple but sometimes
    too aggressive. Additional poles and zeros in `stoch_ym` can mitigate this.
"""
function InternalModel(
    model::SM;
    i_ym::IntRangeOrVector = 1:model.ny,
    stoch_ym::LTISystem = (In = I(length(i_ym)); ss(In, In, In, In, model.Ts))
) where {NT<:Real, SM<:SimModel{NT}}
    stoch_ym = minreal(ss(stoch_ym))
    if iscontinuous(stoch_ym)
        stoch_ym = c2d(stoch_ym, model.Ts, :tustin)
    else
        if !(stoch_ym.Ts ≈ model.Ts) 
            @info "InternalModel: resampling stochastic model from Ts = $(stoch_ym.Ts) to "*
                  "$(model.Ts) s..."
            stoch_ym_c = d2c(stoch_ym, :tustin)
            stoch_ym   = c2d(stoch_ym_c, model.Ts, :tustin)
        end
    end
    return InternalModel{NT, SM}(model, i_ym, stoch_ym.A, stoch_ym.B, stoch_ym.C, stoch_ym.D)
end

"Validate if deterministic `model` and stochastic model `Csm, Dsm` for `InternalModel`s."
function validate_internalmodel(model::SimModel, nym, Csm, Dsm)
    validate_poles(model)
    if size(Csm,1) ≠ nym || size(Dsm,1) ≠ nym
        error("Stochastic model output quantity ($(size(Csm,1))) is different from "*
              "measured output quantity ($nym)")
    end
    if iszero(Dsm)
        error("Stochastic model requires a nonzero direct transmission matrix D")
    end
    return nothing
end

"Validate if `model` is asymptotically stable for `LinModel`s."
function validate_poles(model::LinModel)
    poles = eigvals(model.A)
    if any(abs.(poles) .≥ 1) 
        error("InternalModel does not support integrating or unstable model")
    end
    return nothing
end
validate_poles(::SimModel) = nothing

@doc raw"""
    matrices_internalmodel(model::LinModel)

Get state-space matrices of the [`LinModel`](@ref) `model` for [`InternalModel`](@ref).

The [`InternalModel`](@ref) does not augment the state vector, thus:
```math
    \mathbf{Â = A, B̂_u = B_u, Ĉ = C, B̂_d = B_d, D̂_d = D_d }
```
"""
function matrices_internalmodel(model::LinModel)
    Â, B̂u, Ĉ, B̂d, D̂d = model.A, model.Bu, model.C, model.Bd, model.Dd 
    return Â, B̂u, Ĉ, B̂d, D̂d
end
"Return empty matrices if `model` is not a [`LinModel`](@ref)."
function matrices_internalmodel(model::SimModel{NT}) where NT<:Real
    nu, nx, nd = model.nu, model.nx, model.nd
    Â, B̂u, Ĉ, B̂d, D̂d = zeros(NT,0,nx), zeros(NT,0,nu), zeros(NT,0,nx), zeros(NT,0,nd), zeros(NT,0,nd)
    return Â, B̂u, Ĉ, B̂d, D̂d
end

@doc raw"""
    f̂!(x̂next, ::InternalModel, model::NonLinModel, x̂, u, d)

State function ``\mathbf{f̂}`` of [`InternalModel`](@ref) for [`NonLinModel`](@ref).

It calls `model.f!(x̂next, x̂, u ,d)` since this estimator does not augment the states.
"""
f̂!(x̂next, ::InternalModel, model::NonLinModel, x̂, u, d) = model.f!(x̂next, x̂, u, d)

@doc raw"""
    ĥ!(ŷ, ::InternalModel, model::NonLinModel, x̂, d)

Output function ``\mathbf{ĥ}`` of [`InternalModel`](@ref), it calls `model.h!`.
"""
ĥ!(x̂next, ::InternalModel, model::NonLinModel, x̂, d) = model.h!(x̂next, x̂, d)


@doc raw"""
    init_internalmodel(As, Bs, Cs, Ds) -> Âs, B̂s

Calc stochastic model update matrices `Âs` and `B̂s` for `InternalModel` estimator.

`As`, `Bs`, `Cs` and `Ds` are the stochastic model matrices :
```math
\begin{aligned}
    \mathbf{x_s}(k+1) &= \mathbf{A_s x_s}(k) + \mathbf{B_s e}(k) \\
    \mathbf{y_s}(k)   &= \mathbf{C_s x_s}(k) + \mathbf{D_s e}(k)
\end{aligned}
```
where ``\mathbf{e}(k)`` is conceptual and unknown zero mean white noise. Its optimal
estimation is ``\mathbf{ê=0}``, the expected value. Thus, the `Âs` and `B̂s` matrices that 
optimally update the stochastic estimate ``\mathbf{x̂_s}`` are:
```math
\begin{aligned}
    \mathbf{x̂_s}(k+1) 
        &= \mathbf{(A_s - B_s D_s^{-1} C_s) x̂_s}(k) + \mathbf{(B_s D_s^{-1}) ŷ_s}(k) \\
        &= \mathbf{Â_s x̂_s}(k) + \mathbf{B̂_s ŷ_s}(k)
\end{aligned}
```
with current stochastic outputs estimation ``\mathbf{ŷ_s}(k)``, composed of the measured 
``\mathbf{ŷ_s^m}(k) = \mathbf{y^m}(k) - \mathbf{ŷ_d^m}(k)`` and unmeasured 
``\mathbf{ŷ_s^u = 0}`` outputs. See [^1].

[^1]: Desbiens, A., D. Hodouin & É. Plamondon. 2000, "Global predictive control : a unified
    control structure for decoupling setpoint tracking, feedforward compensation and 
    disturbance rejection dynamics", *IEE Proceedings - Control Theory and Applications*, 
    vol. 147, no 4, <https://doi.org/10.1049/ip-cta:20000443>, p. 465–475, ISSN 1350-2379.
"""
function init_internalmodel(As, Bs, Cs, Ds)
    B̂s = Bs/Ds
    Âs = As - B̂s*Cs
    return Âs, B̂s
end

@doc raw"""
    update_estimate!(estim::InternalModel, u, ym, d=empty(estim.x̂)) -> x̂d

Update `estim.x̂` / `x̂d` / `x̂s` with current inputs `u`, measured outputs `ym` and dist. `d`.

The [`InternalModel`](@ref) updates the deterministic `x̂d` and stochastic `x̂s` estimates with:
```math
\begin{aligned}
    \mathbf{x̂_d}(k+1) &= \mathbf{f}\Big( \mathbf{x̂_d}(k), \mathbf{u}(k), \mathbf{d}(k) \Big) \\
    \mathbf{x̂_s}(k+1) &= \mathbf{Â_s x̂_s}(k) + \mathbf{B̂_s ŷ_s}(k)
\end{aligned}
```
This estimator does not augment the state vector, thus ``\mathbf{x̂ = x̂_d}``. See 
[`init_internalmodel`](@ref) for details. 
"""
function update_estimate!(
    estim::InternalModel{NT, SM}, u, ym, d=empty(estim.x̂)
) where {NT<:Real, SM}
    model = estim.model
    x̂d, x̂s = estim.x̂d, estim.x̂s
    # -------------- deterministic model ---------------------
    ŷd, x̂dnext = Vector{NT}(undef, model.ny), Vector{NT}(undef, model.nx)
    h!(ŷd, model, x̂d, d)
    f!(x̂dnext, model, x̂d, u, d) 
    x̂d .= x̂dnext # this also updates estim.x̂ (they are the same object)
    # --------------- stochastic model -----------------------
    x̂snext = Vector{NT}(undef, estim.nxs)
    ŷs = zeros(NT, model.ny)
    ŷs[estim.i_ym] = ym - ŷd[estim.i_ym]   # ŷs=0 for unmeasured outputs
    mul!(x̂snext, estim.Âs, x̂s)
    mul!(x̂snext, estim.B̂s, ŷs, 1, 1)
    x̂s .= x̂snext
    return nothing
end

@doc raw"""
    init_estimate!(estim::InternalModel, model::LinModel, u, ym, d)

Init `estim.x̂` / `x̂d` / `x̂s` estimate at steady-state for [`InternalModel`](@ref)s.

The deterministic estimates `estim.x̂d` start at steady-state using `u` and `d` arguments:
```math
    \mathbf{x̂_d} = \mathbf{(I - A)^{-1} (B_u u + B_d d)}
```
Based on `ym` argument and current stochastic outputs estimation ``\mathbf{ŷ_s}``, composed
of the measured ``\mathbf{ŷ_s^m} = \mathbf{y^m} - \mathbf{ŷ_d^m}`` and unmeasured 
``\mathbf{ŷ_s^u = 0}`` outputs, the stochastic estimates also start at steady-state:
```math
    \mathbf{x̂_s} = \mathbf{(I - Â_s)^{-1} B̂_s ŷ_s}
```
This estimator does not augment the state vector, thus ``\mathbf{x̂ = x̂_d}``. See
[`init_internalmodel`](@ref) for details.
"""
function init_estimate!(estim::InternalModel, model::LinModel{NT}, u, ym, d) where NT<:Real
    x̂d, x̂s = estim.x̂d, estim.x̂s
    x̂d .= (I - model.A)\(model.Bu*u + model.Bd*d)
    ŷd = Vector{NT}(undef, model.ny)
    h!(ŷd, model, x̂d, d)
    ŷs = zeros(NT, model.ny)
    ŷs[estim.i_ym] = ym - ŷd[estim.i_ym]  # ŷs=0 for unmeasured outputs
    x̂s .= (I-estim.Âs)\estim.B̂s*ŷs
    return nothing
end

@doc raw"""
    evalŷ(estim::InternalModel, ym, d) -> ŷ

Get [`InternalModel`](@ref) output `ŷ` from current measured outputs `ym` and dist. `d`.

[`InternalModel`](@ref) estimator needs current measured outputs ``\mathbf{y^m}(k)`` to 
estimate its outputs ``\mathbf{ŷ}(k)``, since the strategy imposes that 
``\mathbf{ŷ^m}(k) = \mathbf{y^m}(k)`` is always true.
"""
function evalŷ(estim::InternalModel{NT}, ym, d) where NT<:Real
    ŷ = Vector{NT}(undef, estim.model.ny)
    h!(ŷ, estim.model, estim.x̂d, d - estim.model.dop) 
    ŷ .+= estim.model.yop
    ŷ[estim.i_ym] = ym
    return ŷ
end

"Print InternalModel information without i/o integrators."
function print_estim_dim(io::IO, estim::InternalModel, n)
    nu, nd = estim.model.nu, estim.model.nd
    nx̂, nym, nyu = estim.nx̂, estim.nym, estim.nyu
    println(io, "$(lpad(nu, n)) manipulated inputs u")
    println(io, "$(lpad(nx̂, n)) estimated states x̂")
    println(io, "$(lpad(nym, n)) measured outputs ym")
    println(io, "$(lpad(nyu, n)) unmeasured outputs yu")
    print(io,   "$(lpad(nd, n)) measured disturbances d")
end

