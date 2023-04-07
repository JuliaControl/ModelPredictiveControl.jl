struct InternalModel{M<:SimModel} <: StateEstimator
    model::M
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
    Âs::Matrix{Float64}
    B̂s::Matrix{Float64}
    function InternalModel{M}(model::M, i_ym, Asm, Bsm, Csm, Dsm) where {M<:SimModel}
        ny = model.ny
        nym, nyu = length(i_ym), ny - length(i_ym)
        if isa(model, LinModel)
            poles = eigvals(model.A)
            if any(abs.(poles) .≥ 1) 
                error("InternalModel does not support integrating or unstable model")
            end
        end
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
        nxs = size(As,1)
        Âs, B̂s = init_internalmodel(As, Bs, Cs, Ds)
        i_ym = collect(i_ym)
        x̂d = x̂ = copy(model.x) # x̂ and x̂d are same object (updating x̂d will update x̂)
        x̂s = zeros(nxs)
        return new(model, x̂, x̂d, x̂s, i_ym, nx̂, nym, nyu, nxs, As, Bs, Cs, Ds, Âs, B̂s)
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
InternalModel{LinModel} estimator with a sample time Ts = 0.5 s and:
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
    if isa(stoch_ym, TransferFunction) 
        stoch_ym = minreal(ss(stoch_ym))
    end
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


@doc raw"""
    init_internalmodel(As, Bs, Cs, Ds)

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
``\mathbf{ŷ_s^u = 0}`` outputs. See [^3].

[^3]: Desbiens, A., D. Hodouin & É. Plamondon. 2000, "Global predictive control : a unified
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
    updatestate!(estim::InternalModel, u, ym, d=Float64[])

Update `estim.x̂` \ `x̂d` \ `x̂s` with current inputs `u`, measured outputs `ym` and dist. `d`.
"""
function updatestate!(estim::InternalModel, u, ym, d=Float64[])
    model = estim.model
    u, d, ym = remove_op(estim, u, d, ym)
    x̂d, x̂s = estim.x̂d, estim.x̂s
    # -------------- deterministic model ---------------------
    ŷd = model.h(x̂d, d)
    x̂d[:] = model.f(x̂d, u, d) # this also updates estim.xhat (they are the same object)
    # --------------- stochastic model -----------------------
    ŷs = zeros(model.ny,1)
    ŷs[estim.i_ym] = ym - ŷd[estim.i_ym]   # ŷs=0 for unmeasured outputs
    x̂s[:] = estim.Âs*x̂s + estim.B̂s*ŷs
    return x̂d
end

@doc raw"""
    initstate!(estim::InternalModel, u, ym, d=Float64[])

Init `estim.x̂d` / `x̂s` states from current inputs `u`, meas. outputs `ym` and disturb. `d`.

The deterministic state `estim.x̂d` initialization method is identical to 
[`initstate!(::StateEstimator)`](@ref). The stochastic states `estim.x̂s` are init at 0. 
"""
function initstate!(estim::InternalModel, u, ym, d=Float64[])
    model = estim.model
    x̂d = isa(model, LinModel) ? steadystate(model, u, d) : estim.x̂[1:model.nx]
    estim.x̂d[:] = x̂d
    # TODO: best method to initialize internal model stochastic states ? not sure...
    estim.x̂s[:] = zeros(estim.nxs)
    return x̂d
end

@doc raw"""
    evaloutput(estim::InternalModel, ym, d=Float64[])

Evaluate `InternalModel` outputs `ŷ` from `estim.x̂d` states and measured outputs `ym`.

[`InternalModel`](@ref) estimator needs current measured outputs ``\mathbf{y^m}(k)`` to 
estimate its outputs ``\mathbf{ŷ}(k)``, since the strategy imposes that 
``\mathbf{ŷ^m}(k) = \mathbf{y^m}(k)`` is always true.
"""
function evaloutput(estim::InternalModel, ym, d=Float64[])
    ŷ = estim.model.h(estim.x̂d, d - estim.model.dop) + estim.model.yop
    ŷ[estim.i_ym] = ym
    return ŷ
end

(estim::InternalModel)(ym, d=Float64[]) = evaloutput(estim::InternalModel, ym, d)

