struct InternalModel{M<:SimModel} <: StateEstimator
    model::M
    lastu0::Vector{Float64}
    x̂::Vector{Float64}
    x̂d::Vector{Float64}
    x̂s::Vector{Float64}
    i_ym::Vector{Int}
    nx̂::Int
    nym::Int
    nyu::Int
    nxs::Int
    As::Matrix{Float64}
    Bs::Matrix{Float64}
    Cs::Matrix{Float64}
    Ds::Matrix{Float64}
    Â ::Matrix{Float64}
    B̂u::Matrix{Float64}
    Ĉ ::Matrix{Float64}
    B̂d::Matrix{Float64}
    D̂d::Matrix{Float64}
    Âs::Matrix{Float64}
    B̂s::Matrix{Float64}
    function InternalModel{M}(model::M, i_ym, Asm, Bsm, Csm, Dsm) where {M<:SimModel}
        nu, ny = model.nu, model.ny
        nym, nyu = length(i_ym), ny - length(i_ym)
        validate_internalmodel(model)
        validate_ym(model, i_ym)
        if size(Csm,1) ≠ nym || size(Dsm,1) ≠ nym
            error("Stochastic model output quantity ($(size(Csm,1))) is different from "*
                  "measured output quantity ($nym)")
        end
        if iszero(Dsm)
            error("Stochastic model requires a nonzero direct transmission matrix D")
        end
        As, Bs, Cs, Ds = stoch_ym2y(model, i_ym, Asm, Bsm, Csm, Dsm)
        nxs = size(As,1)
        nx̂ = model.nx
        Â, B̂u, Ĉ, B̂d, D̂d = matrices_internalmodel(model)
        Âs, B̂s = init_internalmodel(As, Bs, Cs, Ds)
        lastu0 = zeros(nu)
        x̂d = x̂ = zeros(model.nx) # x̂ and x̂d are same object (updating x̂d will update x̂)
        x̂s = zeros(nxs)
        return new(
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
    InternalModel(model::SimModel; i_ym=1:model.ny, stoch_ym=ss(1,1,1,1,model.Ts).*I)

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
 2 states x̂
 1 measured outputs ym
 1 unmeasured outputs yu
 0 measured disturbances d
```

# Extended Help
`stoch_ym` is a `TransferFunction` or `StateSpace` object that models disturbances on
``\mathbf{y^m}``. Its input is a hypothetical zero mean white noise vector. `stoch_ym` 
supposes 1 integrator per measured outputs by default, assuming that the current stochastic
estimate ``\mathbf{ŷ_s^m}(k) = \mathbf{y^m}(k) - \mathbf{ŷ_d^m}(k)`` is constant in the 
future. This is the dynamic matrix control (DMC) strategy, which is simple but sometimes too
aggressive. Additional poles and zeros in `stoch_ym` can mitigate this.
"""
function InternalModel(
    model::M;
    i_ym::IntRangeOrVector = 1:model.ny,
    stoch_ym::Union{StateSpace, TransferFunction} = ss(1,1,1,1,model.Ts).*I(length(i_ym))
) where {M<:SimModel}
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
    return InternalModel{M}(model, i_ym, stoch_ym.A, stoch_ym.B, stoch_ym.C, stoch_ym.D)
end

"Validate if `model` is asymptotically stable for [`LinModel`](@ref)."
function validate_internalmodel(model::LinModel)
    poles = eigvals(model.A)
    if any(abs.(poles) .≥ 1) 
        error("InternalModel does not support integrating or unstable model")
    end
end
validate_internalmodel(::SimModel) = nothing


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
matrices_internalmodel(::SimModel) = tuple(fill(zeros(0, 0), 5)...)

@doc raw"""
    f̂(estim::InternalModel, x̂, u, d)

State function ``\mathbf{f̂}`` of [`InternalModel`](@ref).

It calls [`f(estim.model, x̂, u ,d)`](@ref) since this estimator does not augment the states.
"""
f̂(estim::InternalModel, x̂, u, d) = f(estim.model, x̂, u, d)

@doc raw"""
    ĥ(estim::InternalModel, x̂, d)

Output function ``\mathbf{ĥ}`` of [`InternalModel`](@ref), it calls [`h`](@ref) directly.
"""
ĥ(estim::InternalModel, x̂, d) = h(estim.model, x̂, d)


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
    vol. 147, no 4, https://doi.org/10.1049/ip-cta:20000443, p. 465–475, ISSN 1350-2379.
"""
function init_internalmodel(As, Bs, Cs, Ds)
    B̂s = Bs/Ds
    Âs = As - B̂s*Cs
    return Âs, B̂s
end

@doc raw"""
    update_estimate!(estim::InternalModel, u, ym, d=empty(estim.x̂)) -> x̂d

Update `estim.x̂` \ `x̂d` \ `x̂s` with current inputs `u`, measured outputs `ym` and dist. `d`.

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
function update_estimate!(estim::InternalModel, u, ym, d=empty(estim.x̂))
    model = estim.model
    x̂d, x̂s = estim.x̂d, estim.x̂s
    # -------------- deterministic model ---------------------
    ŷd = h(model, x̂d, d)
    x̂d[:] = f(model, x̂d, u, d) # this also updates estim.x̂ (they are the same object)
    # --------------- stochastic model -----------------------
    ŷs = zeros(model.ny)
    ŷs[estim.i_ym] = ym - ŷd[estim.i_ym]   # ŷs=0 for unmeasured outputs
    x̂s[:] = estim.Âs*x̂s + estim.B̂s*ŷs
    return x̂d
end

@doc raw"""
    init_estimate!(estim::InternalModel, model::LinModel, u, ym, d)

Init `estim.x̂` \ `x̂d` \ `x̂s` estimate at steady-state for [`InternalModel`](@ref)s.

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
function init_estimate!(estim::InternalModel, model::LinModel, u, ym, d)
    x̂d, x̂s = estim.x̂d, estim.x̂s
    x̂d[:] = (I - model.A)\(model.Bu*u + model.Bd*d)
    ŷd = h(model, x̂d, d)
    ŷs = zeros(model.ny)
    ŷs[estim.i_ym] = ym - ŷd[estim.i_ym]  # ŷs=0 for unmeasured outputs
    x̂s[:] = (I-estim.Âs)\estim.B̂s*ŷs
    return nothing
end

"Print InternalModel information without i/o integrators."
function print_estim_dim(io::IO, estim::InternalModel, n)
    nu, nd = estim.model.nu, estim.model.nd
    nx̂, nym, nyu = estim.nx̂, estim.nym, estim.nyu
    println(io, "$(lpad(nu, n)) manipulated inputs u")
    println(io, "$(lpad(nx̂, n)) states x̂")
    println(io, "$(lpad(nym, n)) measured outputs ym")
    println(io, "$(lpad(nyu, n)) unmeasured outputs yu")
    print(io,   "$(lpad(nd, n)) measured disturbances d")
end

