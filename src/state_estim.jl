abstract type StateEstimator end

abstract type Estimate end

struct InternalModelState <: Estimate
    x̂::Vector{Float64}
    x̂d::Vector{Float64}
    x̂s::Vector{Float64}
end

struct InternalModel <: StateEstimator
    model::SimModel
    state::InternalModelState
    i_ym::IntRangeOrVector
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
    nint_ym::Vector{Int}
    function InternalModel(model, Asm, Bsm, Csm, Dsm, i_ym)
        ny = model.ny
        if isa(model, LinModel)
            poles = eigvals(model.A)
            if any(abs.(poles) .≥ 1) 
                error("InternalModel does not support integrating or unstable model")
            end
        end
        if length(unique(i_ym)) ≠ length(i_ym) || maximum(i_ym) > ny
            error("Measured output indices i_ym should contains valid and unique indices")
        end
        nym = length(i_ym);
        nyu = ny - nym;
        if size(Csm,1) ≠ nym || size(Dsm,1) ≠ nym
            error("Stochastic model output quantity ($(size(Csm,1))) is different from "*
                  "measured output quantity ($nym)")
        end
        if iszero(Dsm)
            error("Stochastic model requires a nonzero direct transmission matrix D")
        end
        # s : all model outputs, sm : measured outputs only
        As = Asm;
        Bs = Bsm;
        Cs = zeros(ny,size(Csm,2));
        Ds = zeros(ny,size(Dsm,2));
        Cs[i_ym,:] = Csm;
        Ds[i_ym,:] = Dsm;
        nxs = size(As,1);
        nx̂ = model.nx
        nxs = size(As,1)
        nint_ym = zeros(nym) # not used for InternalModel
        Âs, B̂s = init_internalmodel(As, Bs, Cs, Ds)
        x̂d = zeros(nx̂)
        x̂s = zeros(nxs)
        state = InternalModelState(x̂d, x̂d, x̂s) # xhat = xhatd for InternalModel
        return new(model, state, i_ym, nx̂, nym, nyu, nxs, As, Bs, Cs, Ds, Âs, B̂s, nint_ym)
    end
end


@doc raw"""
    InternalModel(model::SimModel; i_ym=1:model.ny, stoch_ym=ss(1,1,1,1,model.Ts).*I)

Construct an `InternalModel` estimator based on `model`.

`i_ym` provides the `model` output indices that are measured ``\mathbf{y^m}``, the rest are 
unmeasured ``\mathbf{y^u}``. `model` evaluates the deterministic predictions 
``\mathbf{ŷ_d}``, and `stoch_ym`, the stochastic predictions of the measured outputs 
``\mathbf{ŷ_s^m}`` (the unmeasured ones being ``\mathbf{ŷ_s^u} = \mathbf{0}``). 

`stoch_ym` is a `TransferFunction` or `StateSpace` model that hypothetically filters a zero 
mean white noise vector. Its default value supposes 1 integrator per measured outputs, 
assuming that the current stochastic estimate ``\mathbf{ŷ_s^m}(k) = \mathbf{y^m}(k) - 
\mathbf{ŷ_d^m}(k)`` will be constant in the future. This is the dynamic matrix control (DMC) 
strategy, which is simple but sometimes too aggressive. Additional poles and zeros in 
`stoch_ym` can mitigate this.

!!! warning
    `InternalModel` estimator does not work if `model` is integrating or unstable. The 
    constructor verifies these aspects for `LinModel` but not for `NonLinModel`. Uses any 
    other state estimator in such cases.

See also [`init_internalmodel`](@ref)

# Examples
```jldoctest
julia> estim = InternalModel(LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 0.5), i_ym=[2])
InternalModel state estimator with a sample time Ts = 0.5 s and:
 1 manipulated inputs u
 2 states x̂
 1 measured outputs ym
 1 unmeasured outputs yu
 0 measured disturbances d
```
"""
function InternalModel(
    model::SimModel;
    i_ym::IntRangeOrVector = 1:model.ny,
    stoch_ym::Union{StateSpace, TransferFunction} = ss(1,1,1,1,model.Ts).*I(length(i_ym))
    )
    if isa(stoch_ym, TransferFunction) 
        stoch_ym = minreal(ss(stoch_ym))
    end
    if iscontinuous(stoch_ym)
        stoch_ym = c2d(stoch_ym, model.Ts, :tustin)
    else
        stoch_ym.Ts == model.Ts || error("stoch_ym.Ts must be identical to model.Ts")
    end
    return InternalModel(model, stoch_ym.A, stoch_ym.B, stoch_ym.C, stoch_ym.D, i_ym)
end


@doc raw"""
    init_internalmodel(As, Bs, Cs, Ds)

Calc stochastic model update matrices `Âs` and `B̂s` for `InternalModel` estimator.

`Âs` and `B̂s` are the stochastic model update matrices :
```math
    \mathbf{x̂_s}(k+1) =  \mathbf{Â_s x̂_s}(k) + \mathbf{B̂_s ŷ_s}(k)
```
with current stochastic model states ``\mathbf{x̂_s}(k)`` and outputs ``\mathbf{ŷ_s}(k)``, 
which is in turn composed of the measured ``\mathbf{ŷ_s^m}(k) = \mathbf{y^m}(k) - 
\mathbf{ŷ_d^m}(k)`` and unmeasured ``\mathbf{ŷ_s^u = 0}`` outputs. See [^1].

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

"Update `estim.state` values with current inputs `u`, measured outputs `ym` and dist. `d`."
function updatestate!(estim::InternalModel, u, ym, d=Float64[])
    model = estim.model
    # -------------- deterministic model ---------------------
    ŷd = model.h(estim.state.x̂d, d - model.dop) + model.yop
    estim.state.x̂d[:] = model.f(estim.state.x̂d, u - model.u_op, d - model.dop)
    # --------------- stochastic model -----------------------
    ŷs = zeros(model.ny,1);
    ŷs[estim.i_ym] = ym - ŷd[estim.i_ym];   # ŷs=0 for unmeasured outputs
    model.state.x̂s[:] = estim.Âs*estim.state.x̂s + estim.B̂s*ŷs;
    return estim.state
end


@doc raw"""
    evaloutput(estim::InternalModel, d=Float64[], ym=nothing)

Evaluate `InternalModel` outputs `̂ŷ` from `estim.state` values and measured outputs `ym`.

`InternalModel` estimator needs current measured outputs ``\mathbf{y^m}(k)`` to estimate
its current outputs ``\mathbf{ŷ^m}(k)``, since the strategy imposes that ``\mathbf{ŷ^m}(k) = 
\mathbf{ŷ^m}(k)`` is always true. 
"""
function evaloutput(estim::InternalModel, d=Float64[], ym=nothing)
    if isnothing(ym) 
        error("InternalModel needs current measured outputs ym to estimate its outputs ŷ")
    end
    ŷ = estim.model.h(estim.state.x̂d, d - estim.model.dop) + estim.model.yop
    ŷ[estim.i_ym] = ym
    return ŷ
end


function Base.show(io::IO, estim::StateEstimator)
    println(io, "$(typeof(estim)) state estimator with "*
                "a sample time Ts = $(estim.model.Ts) s and:")
    println(io, " $(estim.model.nu) manipulated inputs u")
    println(io, " $(estim.nx̂) states x̂")
    println(io, " $(estim.nym) measured outputs ym")
    println(io, " $(estim.nyu) unmeasured outputs yu")
    print(io,   " $(estim.model.nd) measured disturbances d")
end