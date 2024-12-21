struct InternalModel{NT<:Real, SM<:SimModel} <: StateEstimator{NT}
    model::SM
    lastu0::Vector{NT}
    x̂op::Vector{NT}
    f̂op::Vector{NT}
    x̂0 ::Vector{NT}
    x̂d::Vector{NT}
    x̂s::Vector{NT}
    ŷs::Vector{NT}
    x̂snext::Vector{NT}
    i_ym::Vector{Int}
    nx̂::Int
    nym::Int
    nyu::Int
    nxs::Int
    As::Matrix{NT}
    Bs::Matrix{NT}
    Cs::Matrix{NT}
    Ds::Matrix{NT}
    Â   ::Matrix{NT}
    B̂u  ::Matrix{NT}
    Ĉ   ::Matrix{NT}
    B̂d  ::Matrix{NT}
    D̂d  ::Matrix{NT}
    Ĉm  ::Matrix{NT}
    D̂dm ::Matrix{NT}
    Âs::Matrix{NT}
    B̂s::Matrix{NT}
    direct::Bool
    corrected::Vector{Bool}
    buffer::StateEstimatorBuffer{NT}
    function InternalModel{NT}(
        model::SM, i_ym, Asm, Bsm, Csm, Dsm
    ) where {NT<:Real, SM<:SimModel}
        nu, ny, nd = model.nu, model.ny, model.nd
        nym, nyu = validate_ym(model, i_ym)
        validate_internalmodel(model, nym, Csm, Dsm)
        As, Bs, Cs, Ds = stoch_ym2y(model, i_ym, Asm, Bsm, Csm, Dsm)
        nxs = size(As,1)
        nx̂ = model.nx
        Â, B̂u, Ĉ, B̂d, D̂d, x̂op, f̂op = matrices_internalmodel(model)
        Ĉm, D̂dm = Ĉ[i_ym,:], D̂d[i_ym,:]
        Âs, B̂s = init_internalmodel(As, Bs, Cs, Ds)
        lastu0 = zeros(NT, nu)
        # x̂0 and x̂d are same object (updating x̂d will update x̂0):
        x̂d = x̂0 = zeros(NT, model.nx) 
        x̂s, x̂snext = zeros(NT, nxs), zeros(NT, nxs)
        ŷs = zeros(NT, ny)
        direct = true # InternalModel always uses direct transmission from ym
        corrected = [false]
        buffer = StateEstimatorBuffer{NT}(nu, nx̂, nym, ny, nd)
        return new{NT, SM}(
            model, 
            lastu0, x̂op, f̂op, x̂0, x̂d, x̂s, ŷs, x̂snext,
            i_ym, nx̂, nym, nyu, nxs, 
            As, Bs, Cs, Ds, 
            Â, B̂u, Ĉ, B̂d, D̂d, Ĉm, D̂dm,
            Âs, B̂s,
            direct, corrected,
            buffer
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
sum both values : ``\mathbf{ŷ = ŷ_d + ŷ_s}``. See the Extended Help for more details.

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
    too aggressive. Additional poles and zeros in `stoch_ym` can mitigate this. The following
    block diagram summarizes the internal model structure.

    ![block diagram of the internal model structure](../assets/imc.svg)
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
    return InternalModel{NT}(model, i_ym, stoch_ym.A, stoch_ym.B, stoch_ym.C, stoch_ym.D)
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
    matrices_internalmodel(model::LinModel) -> Â, B̂u, Ĉ, B̂d, D̂d, x̂op, f̂op

Get state-space matrices of the [`LinModel`](@ref) `model` for [`InternalModel`](@ref).

The [`InternalModel`](@ref) does not augment the state vector, thus:
```math
    \mathbf{Â = A, B̂_u = B_u, Ĉ = C, B̂_d = B_d, D̂_d = D_d, x̂_{op} = x_{op}, f̂_{op} = f_{op}}
```
"""
function matrices_internalmodel(model::LinModel)
    Â, B̂u, Ĉ, B̂d, D̂d = model.A, model.Bu, model.C, model.Bd, model.Dd
    x̂op, f̂op = copy(model.xop), copy(model.fop)
    return Â, B̂u, Ĉ, B̂d, D̂d, x̂op, f̂op
end
"Return empty matrices, and `x̂op` & `f̂op` vectors, if `model` is not a [`LinModel`](@ref)."
function matrices_internalmodel(model::SimModel{NT}) where NT<:Real
    nu, nx, nd, ny = model.nu, model.nx, model.nd, model.ny
    Â, B̂u, Ĉ, B̂d, D̂d = zeros(NT,0,nx), zeros(NT,0,nu), zeros(NT,ny,0), zeros(NT,0,nd), zeros(NT,ny,0)
    x̂op, f̂op = copy(model.xop), copy(model.fop)
    return Â, B̂u, Ĉ, B̂d, D̂d, x̂op, f̂op
end

@doc raw"""
    f̂!(x̂next0, _ , estim::InternalModel, model::NonLinModel, x̂0, u0, d0)

State function ``\mathbf{f̂}`` of [`InternalModel`](@ref) for [`NonLinModel`](@ref).

It calls `model.f!(x̂next0, x̂0, u0 ,d0, model.p)` since this estimator does not augment the states.
"""
f̂!(x̂next0, _, ::InternalModel, model::NonLinModel, x̂0, u0, d0) = model.f!(x̂next0, x̂0, u0, d0, model.p)

@doc raw"""
    ĥ!(ŷ0, estim::InternalModel, model::NonLinModel, x̂0, d0)

Output function ``\mathbf{ĥ}`` of [`InternalModel`](@ref), it calls `model.h!`.
"""
ĥ!(x̂next0, ::InternalModel, model::NonLinModel, x̂0, d0) = model.h!(x̂next0, x̂0, d0, model.p)


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

"Update similar values for [`InternalModel`](@ref) estimator."
function setmodel_estimator!(estim::InternalModel, model, _ , _ , _ , _ , _ )
    Â, B̂u, Ĉ, B̂d, D̂d, x̂op, f̂op = matrices_internalmodel(model)
    # --- update internal model state-space matrices ---
    estim.Â  .= Â
    estim.B̂u .= B̂u
    estim.Ĉ  .= Ĉ
    estim.B̂d .= B̂d
    estim.D̂d .= D̂d
    estim.Ĉm  .= @views Ĉ[estim.i_ym,:]
    estim.D̂dm .= @views D̂d[estim.i_ym,:]
    # --- update state estimate and its operating points ---
    estim.x̂0 .+= estim.x̂op # convert x̂0 to x̂ with the old operating point
    estim.x̂op .= x̂op
    estim.f̂op .= f̂op
    estim.x̂0 .-= estim.x̂op # convert x̂ to x̂0 with the new operating point
    return nothing
end

"""
    correct_estimate!(estim::InternalModel, y0m, d0)

Compute the current stochastic output estimation `ŷs` for [`InternalModel`](@ref).
"""
function correct_estimate!(estim::InternalModel, y0m, d0)
    ŷ0d = estim.buffer.ŷ
    h!(ŷ0d, estim.model, estim.x̂d, d0, estim.model.p)
    ŷs = estim.ŷs
    for j in eachindex(ŷs) # broadcasting was allocating unexpectedly, so we use a loop
        if j in estim.i_ym
            i = estim.i_ym[j]
            ŷs[j] = y0m[i] - ŷ0d[j]
        else
            ŷs[j] = 0
        end
    end
    return nothing
end

@doc raw"""
    update_estimate!(estim::InternalModel, _ , d0, u0)

Update `estim.x̂0`/`x̂d`/`x̂s` with current inputs `u0`, measured outputs `y0m` and dist. `d0`.

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
function update_estimate!(estim::InternalModel, _ , d0, u0)
    model = estim.model
    x̂d, x̂s, ŷs = estim.x̂d, estim.x̂s, estim.ŷs
    # -------------- deterministic model ---------------------
    x̂dnext = estim.buffer.x̂
    f!(x̂dnext, model, x̂d, u0, d0, model.p) 
    x̂d .= x̂dnext # this also updates estim.x̂0 (they are the same object)
    # --------------- stochastic model -----------------------
    x̂snext = estim.x̂snext
    mul!(x̂snext, estim.Âs, x̂s)
    mul!(x̂snext, estim.B̂s, ŷs, 1, 1)
    estim.x̂s .= x̂snext
    # --------------- operating points ---------------------
    x̂0next    = x̂dnext
    x̂0next  .+= estim.f̂op .- estim.x̂op
    estim.x̂0 .= x̂0next
    return nothing
end

@doc raw"""
    init_estimate!(estim::InternalModel, model::LinModel, y0m, d0, u0)

Init `estim.x̂0`/`x̂d`/`x̂s` estimate at steady-state for [`InternalModel`](@ref).

The deterministic estimates `estim.x̂d` start at steady-state using `u0` and `d0` arguments:
```math
    \mathbf{x̂_d} = \mathbf{(I - A)^{-1} (B_u u_0 + B_d d_0 + f_{op} - x_{op})}
```
Based on `y0m` argument and current stochastic outputs estimation ``\mathbf{ŷ_s}``, composed
of the measured ``\mathbf{ŷ_s^m} = \mathbf{y_0^m} - \mathbf{ŷ_{d0}^m}`` and unmeasured 
``\mathbf{ŷ_s^u = 0}`` outputs, the stochastic estimates also start at steady-state:
```math
    \mathbf{x̂_s} = \mathbf{(I - Â_s)^{-1} B̂_s ŷ_s}
```
This estimator does not augment the state vector, thus ``\mathbf{x̂ = x̂_d}``. See
[`init_internalmodel`](@ref) for details.
"""
function init_estimate!(estim::InternalModel, model::LinModel{NT}, y0m, d0, u0) where NT<:Real
    x̂d, x̂s = estim.x̂d, estim.x̂s
    # also updates estim.x̂0 (they are the same object):
    # TODO: use estim.buffer.x̂ to reduce the allocation:
    x̂d .= (I - model.A)\(model.Bu*u0 + model.Bd*d0 + model.fop - model.xop)
    ŷ0d = estim.buffer.ŷ
    h!(ŷ0d, model, x̂d, d0, model.p)
    ŷs = ŷ0d
    ŷs[estim.i_ym] .= @views y0m .- ŷ0d[estim.i_ym]
    # ŷs=0 for unmeasured outputs :
    map(i -> ŷs[i] = (i in estim.i_ym) ? ŷs[i] : 0, eachindex(ŷs))  
    x̂s .= (I-estim.Âs)\estim.B̂s*ŷs # TODO: remove this allocation with a new buffer?
    return nothing
end

# Compute estimated output with current stochastic estimate `estim.ŷs` for `InternalModel`
function evaloutput(estim::InternalModel, d)
    if !estim.corrected[]
        @warn "preparestate! should be called before evaloutput with InternalModel"
    end
    validate_args(estim.model, d)
    ŷ0d, d0 = estim.buffer.ŷ, estim.buffer.d
    d0 .= d .- estim.model.dop
    ĥ!(ŷ0d, estim, estim.model, estim.x̂0, d0)
    ŷ   = ŷ0d
    ŷ .+= estim.model.yop .+ estim.ŷs
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

