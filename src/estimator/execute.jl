"""
    remove_op!(estim::StateEstimator, ym, d, u=nothing) -> y0m, d0, u0

Remove operating pts on measured outputs `ym`, disturbances `d` and inputs `u` (if provided).

If `u` is provided, also store current inputs without operating points `u0` in 
`estim.lastu0`. This field is used for [`PredictiveController`](@ref) computations.
"""
function remove_op!(estim::StateEstimator, ym, d, u=nothing)
    y0m, u0, d0 = estim.buffer.ym, estim.buffer.u, estim.buffer.d
    y0m .= @views ym .- estim.model.yop[estim.i_ym]
    d0  .= d  .- estim.model.dop
    if !isnothing(u)
        u0 .= u .- estim.model.uop
        estim.lastu0 .= u0
    end
    return y0m, d0, u0
end

@doc raw"""
    f̂!(x̂next0, û0, estim::StateEstimator, model::SimModel, x̂0, u0, d0) -> nothing

Mutating state function ``\mathbf{f̂}`` of the augmented model.

By introducing an augmented state vector ``\mathbf{x̂_0}`` like in [`augment_model`](@ref), the
function returns the next state of the augmented model, defined as:
```math
\begin{aligned}
    \mathbf{x̂_0}(k+1) &= \mathbf{f̂}\Big(\mathbf{x̂_0}(k), \mathbf{u_0}(k), \mathbf{d_0}(k)\Big) \\
    \mathbf{ŷ_0}(k)   &= \mathbf{ĥ}\Big(\mathbf{x̂_0}(k), \mathbf{d_0}(k)\Big) 
\end{aligned}
```
where ``\mathbf{x̂_0}(k+1)`` is stored in `x̂next0` argument. The method mutates `x̂next0` and
`û0` in place, the latter stores the input vector of the augmented model 
``\mathbf{u_0 + ŷ_{s_u}}``.
"""
function f̂!(x̂next0, û0, estim::StateEstimator, model::SimModel, x̂0, u0, d0)
    # `@views` macro avoid copies with matrix slice operator e.g. [a:b]
    @views x̂d, x̂s = x̂0[1:model.nx], x̂0[model.nx+1:end]
    @views x̂d_next, x̂s_next = x̂next0[1:model.nx], x̂next0[model.nx+1:end]
    mul!(û0, estim.Cs_u, x̂s)
    û0 .+= u0
    f!(x̂d_next, model, x̂d, û0, d0)
    mul!(x̂s_next, estim.As, x̂s)
    return nothing
end

"""
    f̂!(x̂next0, _ , estim::StateEstimator, model::LinModel, x̂0, u0, d0) -> nothing

Use the augmented model matrices if `model` is a [`LinModel`](@ref).
"""
function f̂!(x̂next0, _ , estim::StateEstimator, ::LinModel, x̂0, u0, d0)
    mul!(x̂next0, estim.Â,  x̂0)
    mul!(x̂next0, estim.B̂u, u0, 1, 1)
    mul!(x̂next0, estim.B̂d, d0, 1, 1)
    return nothing
end

@doc raw"""
    ĥ!(ŷ0, estim::StateEstimator, model::SimModel, x̂0, d0) -> nothing

Mutating output function ``\mathbf{ĥ}`` of the augmented model, see [`f̂!`](@ref).
"""
function ĥ!(ŷ0, estim::StateEstimator, model::SimModel, x̂0, d0)
    # `@views` macro avoid copies with matrix slice operator e.g. [a:b]
    @views x̂d, x̂s = x̂0[1:model.nx], x̂0[model.nx+1:end]
    h!(ŷ0, model, x̂d, d0)
    mul!(ŷ0, estim.Cs_y, x̂s, 1, 1)
    return nothing
end
"""
    ĥ!(ŷ0, estim::StateEstimator, model::LinModel, x̂0, d0) -> nothing

Use the augmented model matrices if `model` is a [`LinModel`](@ref).
"""
function ĥ!(ŷ0, estim::StateEstimator, ::LinModel, x̂0, d0)
    mul!(ŷ0, estim.Ĉ,  x̂0)
    mul!(ŷ0, estim.D̂d, d0, 1, 1)
    return nothing
end


@doc raw"""
    initstate!(estim::StateEstimator, u, ym, d=[]) -> x̂

Init `estim.x̂0` states from current inputs `u`, measured outputs `ym` and disturbances `d`.

The method tries to find a good steady-state for the initial estimate ``\mathbf{x̂}(0)``. It
removes the operating points with [`remove_op!`](@ref) and call [`init_estimate!`](@ref):

- If `estim.model` is a [`LinModel`](@ref), it finds the steady-state of the augmented model
  using `u` and `d` arguments, and uses the `ym` argument to enforce that 
  ``\mathbf{ŷ^m}(0) = \mathbf{y^m}(0)``. For control applications, this solution produces a
  bumpless manual to automatic transfer. See [`init_estimate!`](@ref) for details.
- Else, `estim.x̂0` is left unchanged. Use [`setstate!`](@ref) to manually modify it.

If applicable, it also sets the error covariance `estim.P̂` to `estim.P̂_0`.

# Examples
```jldoctest
julia> estim = SteadyKalmanFilter(LinModel(tf(3, [10, 1]), 0.5), nint_ym=[2]);

julia> u = [1]; y = [3 - 0.1]; x̂ = round.(initstate!(estim, u, y), digits=3)
3-element Vector{Float64}:
  5.0
  0.0
 -0.1

julia> x̂ ≈ updatestate!(estim, u, y)
true

julia> evaloutput(estim) ≈ y
true
```
"""
function initstate!(estim::StateEstimator, u, ym, d=estim.buffer.empty)
    # --- validate arguments ---
    validate_args(estim, ym, d, u)
    # --- init state estimate ----
    y0m, d0, u0 = remove_op!(estim, ym, d, u)
    init_estimate!(estim, estim.model, y0m, d0, u0)
    # --- init covariance error estimate, if applicable ---
    init_estimate_cov!(estim, y0m, d0, u0)
    x̂ = estim.x̂0 + estim.x̂op
    return x̂
end

"By default, [`StateEstimator`](@ref)s do not need covariance error estimate."
init_estimate_cov!(::StateEstimator, _ , _ , _ ) = nothing

@doc raw"""
    init_estimate!(estim::StateEstimator, model::LinModel, y0m, d0, u0)

Init `estim.x̂0` estimate with the steady-state solution if `model` is a [`LinModel`](@ref).

Using `u0`, `y0m` and `d0` arguments (deviation values, see [`setop!`](@ref)), the
steadystate problem combined to the equality constraint ``\mathbf{ŷ_0^m} = \mathbf{y_0^m}``
engenders the following system to solve:
```math
\begin{bmatrix}
    \mathbf{I} - \mathbf{Â}                         \\
    \mathbf{Ĉ^m}
\end{bmatrix} \mathbf{x̂_0} =
\begin{bmatrix}
    \mathbf{B̂_u u_0 + B̂_d d_0 + f̂_{op} - x̂_{op}}    \\
    \mathbf{y_0^m - D̂_d^m d_0}
\end{bmatrix}
```
in which ``\mathbf{Ĉ^m, D̂_d^m}`` are the rows of `estim.Ĉ, estim.D̂d`  that correspond to 
measured outputs ``\mathbf{y^m}``.
"""
function init_estimate!(estim::StateEstimator, ::LinModel, y0m, d0, u0)
    Â, B̂u, Ĉ, B̂d, D̂d = estim.Â, estim.B̂u, estim.Ĉ, estim.B̂d, estim.D̂d
    Ĉm, D̂dm = @views Ĉ[estim.i_ym, :], D̂d[estim.i_ym, :] # measured outputs ym only
    # TODO: use estim.buffer.x̂ to reduce allocations
    estim.x̂0 .= [I - Â; Ĉm]\[B̂u*u0 + B̂d*d0 + estim.f̂op - estim.x̂op; y0m - D̂dm*d0]
    return nothing
end
"""
    init_estimate!(estim::StateEstimator, model::SimModel, _ , _ , _ )

Left `estim.x̂0` estimate unchanged if `model` is not a [`LinModel`](@ref).
"""
init_estimate!(::StateEstimator, ::SimModel, _ , _ , _ ) = nothing

@doc raw"""
    evaloutput(estim::StateEstimator, d=[]) -> ŷ

Evaluate `StateEstimator` outputs `ŷ` from `estim.x̂0` states and disturbances `d`.

Calling a [`StateEstimator`](@ref) object calls this `evaloutput` method.

# Examples
```jldoctest
julia> kf = SteadyKalmanFilter(setop!(LinModel(tf(2, [10, 1]), 5), yop=[20]));

julia> ŷ = evaloutput(kf)
1-element Vector{Float64}:
 20.0
```
"""
function evaloutput(estim::StateEstimator{NT}, d=estim.buffer.empty) where NT <: Real
    validate_args(estim.model, d)
    ŷ0, d0 = estim.buffer.ŷ, estim.buffer.d
    d0 .= d .- estim.model.dop
    ĥ!(ŷ0, estim, estim.model, estim.x̂0, d0)
    ŷ   = ŷ0
    ŷ .+= estim.model.yop
    return ŷ
end

"Functor allowing callable `StateEstimator` object as an alias for `evaloutput`."
(estim::StateEstimator)(d=estim.buffer.empty) = evaloutput(estim, d)

@doc raw"""
    preparestate!(estim::StateEstimator, ym, d=estim.model.dop) -> x̂

Prepare `estim.x̂0` estimate with measured outputs `ym` and dist. `d` for current time step.

This method does nothing if `estim.direct==false` (for delayed estimators). Otherwise, it
removes the operating points with [`remove_op!`](@ref) and call [`prepare_estimate!`](@ref).
"""
function preparestate!(estim::StateEstimator, ym, d=estim.model.dop)
    if estim.direct
        validate_args(estim, ym, d)
        y0m, d0 = remove_op!(estim, ym, d)
        prepare_estimate!(estim, y0m, d0) # compute x̂0corr
    end
    x̂   = copy(estim.x̂0)
    x̂ .+= estim.x̂op
    return x̂
end

@doc raw"""
    updatestate!(estim::StateEstimator, u, ym, d=[]) -> x̂

Update `estim.x̂0` estimate with current inputs `u`, measured outputs `ym` and dist. `d`. 

The method removes the operating points with [`remove_op!`](@ref) and call 
[`update_estimate!`](@ref).

# Examples
```jldoctest
julia> kf = SteadyKalmanFilter(LinModel(ss(0.1, 0.5, 1, 0, 4.0)));

julia> x̂ = updatestate!(kf, [1], [0]) # x̂[2] is the integrator state (nint_ym argument)
2-element Vector{Float64}:
 0.5
 0.0
```
"""
function updatestate!(estim::StateEstimator, u, ym, d=estim.buffer.empty)
    validate_args(estim, ym, d, u)
    y0m, d0, u0 = remove_op!(estim, ym, d, u)
    x̂0next   = update_estimate!(estim, y0m, d0, u0)
    x̂next   = x̂0next
    x̂next .+= estim.x̂op
    return x̂next
end
updatestate!(::StateEstimator, _ ) = throw(ArgumentError("missing measured outputs ym"))

"""
    validate_args(estim::StateEstimator, ym, d, u=nothing)

Check `ym`, `d` and `u` sizes against `estim` dimensions.
"""
function validate_args(estim::StateEstimator, ym, d, u=nothing)
    validate_args(estim.model, d, u)
    nym = estim.nym
    size(ym) ≠ (nym,) && throw(DimensionMismatch("ym size $(size(ym)) ≠ meas. output size ($nym,)"))
end

"""
    setstate!(estim::StateEstimator, x̂) -> estim

Set `estim.x̂0` to `x̂ - estim.x̂op` from the argument `x̂`. 
"""
function setstate!(estim::StateEstimator, x̂)
    size(x̂) == (estim.nx̂,) || error("x̂ size must be $((estim.nx̂,))")
    estim.x̂0 .= x̂ .- estim.x̂op
    return estim
end

@doc raw"""
    setmodel!(estim::StateEstimator, model=estim.model; <keyword arguments>) -> estim

Set `model` and covariance matrices of `estim` [`StateEstimator`](@ref).

Allows model adaptation of estimators based on [`LinModel`](@ref) at runtime. Modification 
of [`NonLinModel`](@ref) state-space functions is not supported. New covariance matrices can
be specified with the keyword arguments (see [`SteadyKalmanFilter`](@ref) documentation for
the nomenclature). Not supported by [`Luenberger`](@ref) and [`SteadyKalmanFilter`](@ref), 
use the time-varying [`KalmanFilter`](@ref) instead. The [`MovingHorizonEstimator`](@ref)
model is kept constant over the estimation horizon ``H_e``. The matrix dimensions and sample
time must stay the same. Note that the observability and controllability of the new
augmented model is not verified (see Extended Help for more info).

# Arguments
!!! info
    Keyword arguments with *`emphasis`* are non-Unicode alternatives.

- `estim::StateEstimator` : estimator to set model and covariances.
- `model=estim.model` : new plant model (not supported by [`NonLinModel`](@ref)).
- `Q̂=nothing` or *`Qhat`* : new augmented model ``\mathbf{Q̂}`` covariance matrix.
- `R̂=nothing` or *`Rhat`* : new augmented model ``\mathbf{R̂}`` covariance matrix.

# Examples
```jldoctest
julia> kf = KalmanFilter(LinModel(ss(0.1, 0.5, 1, 0, 4.0)), σQ=[√4.0], σQint_ym=[√0.25]);

julia> kf.model.A[], kf.Q̂[1, 1], kf.Q̂[2, 2] 
(0.1, 4.0, 0.25)

julia> setmodel!(kf, LinModel(ss(0.42, 0.5, 1, 0, 4.0)), Q̂=[1 0;0 0.5]);

julia> kf.model.A[], kf.Q̂[1, 1], kf.Q̂[2, 2] 
(0.42, 1.0, 0.5)
```

# Extended Help
!!! details "Extended Help"
    Using the default model augmentation computed by the [`default_nint`](@ref) method, 
    switching from a non-integrating plant model to an integrating one will produce
    an augmented model that is not observable. Moving the unmeasured disturbances at the 
    model input (`nint_u` parameter) can fix this issue.
"""
function setmodel!(
        estim::StateEstimator, 
        model = estim.model;
        Qhat = nothing,
        Rhat = nothing,
        Q̂ = Qhat,
        R̂ = Rhat
    )
    uop_old = copy(estim.model.uop)
    yop_old = copy(estim.model.yop)
    dop_old = copy(estim.model.dop)
    setmodel_linmodel!(estim.model, model)
    estim.lastu0 .+= uop_old .- model.uop
    setmodel_estimator!(estim, model, uop_old, yop_old, dop_old, Q̂, R̂)
    return estim
end

"Update LinModel matrices and its operating points."
function setmodel_linmodel!(old::LinModel, new::LinModel)
    new.Ts == old.Ts || throw(ArgumentError("model.Ts must be $(old.Ts) s"))
    new.nu == old.nu || throw(ArgumentError("model.nu must be $(old.nu)"))
    new.nx == old.nx || throw(ArgumentError("model.nx must be $(old.nx)"))
    new.ny == old.ny || throw(ArgumentError("model.ny must be $(old.ny)"))
    new.nd == old.nd || throw(ArgumentError("model.nd must be $(old.nd)"))
    old.A   .= new.A
    old.Bu  .= new.Bu
    old.C   .= new.C
    old.Bd  .= new.Bd
    old.Dd  .= new.Dd
    old.uop .= new.uop
    old.yop .= new.yop
    old.dop .= new.dop
    old.xop .= new.xop
    old.fop .= new.fop
    return nothing
end
function setmodel_linmodel!(old::SimModel, new::SimModel)
    (new !== old) && error("Only LinModel can be modified in setmodel!")
    return nothing
end

"Update the augmented model matrices of `estim` by default."
function setmodel_estimator!(estim::StateEstimator, model, _ , _ , _ , Q̂, R̂)
    As, Cs_u, Cs_y = estim.As, estim.Cs_u, estim.Cs_y
    Â, B̂u, Ĉ, B̂d, D̂d, x̂op, f̂op = augment_model(model, As, Cs_u, Cs_y, verify_obsv=false)
    # --- update augmented state-space matrices ---
    estim.Â  .= Â
    estim.B̂u .= B̂u
    estim.Ĉ  .= Ĉ
    estim.B̂d .= B̂d
    estim.D̂d .= D̂d
    # --- update state estimate and its operating points ---
    estim.x̂0 .+= estim.x̂op # convert x̂0 to x̂ with the old operating point
    estim.x̂op .= x̂op
    estim.f̂op .= f̂op
    estim.x̂0 .-= estim.x̂op # convert x̂ to x̂0 with the new operating point
    # --- update covariance matrices ---
    !isnothing(Q̂) && (estim.Q̂ .= to_hermitian(Q̂))
    !isnothing(R̂) && (estim.R̂ .= to_hermitian(R̂))
    return nothing
end
