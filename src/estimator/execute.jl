"""
    remove_op!(estim::StateEstimator, u, ym, d) -> u0, ym0, d0

Remove operating points on inputs `u`, measured outputs `ym` and disturbances `d`.

Also store current inputs without operating points `u0` in `estim.lastu0`. This field is 
used for [`PredictiveController`](@ref) computations.
"""
function remove_op!(estim::StateEstimator, u, ym, d)
    u0  = u  - estim.model.uop
    ym0 = ym - estim.model.yop[estim.i_ym]
    d0  = d  - estim.model.dop
    estim.lastu0[:] = u0
    return u0, ym0, d0 
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
function initstate!(estim::StateEstimator, u, ym, d=empty(estim.x̂0))
    # --- validate arguments ---
    validate_args(estim, u, ym, d)
    # --- init state estimate ----
    u0, ym0, d0 = remove_op!(estim, u, ym, d)
    init_estimate!(estim, estim.model, u0, ym0, d0)
    # --- init covariance error estimate, if applicable ---
    init_estimate_cov!(estim, u0, ym0, d0)
    x̂ = estim.x̂0 + estim.x̂op
    return x̂
end

"By default, [`StateEstimator`](@ref)s do not need covariance error estimate."
init_estimate_cov!(::StateEstimator, _ , _ , _ ) = nothing

@doc raw"""
    init_estimate!(estim::StateEstimator, model::LinModel, u0, ym0, d0)

Init `estim.x̂0` estimate with the steady-state solution if `model` is a [`LinModel`](@ref).

Using `u0`, `ym0` and `d0` arguments (deviation values, see [`setop!`](@ref)), the
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
function init_estimate!(estim::StateEstimator, ::LinModel, u0, ym0, d0)
    Â, B̂u, Ĉ, B̂d, D̂d = estim.Â, estim.B̂u, estim.Ĉ, estim.B̂d, estim.D̂d
    Ĉm, D̂dm = @views Ĉ[estim.i_ym, :], D̂d[estim.i_ym, :] # measured outputs ym only
    estim.x̂0 .= [I - Â; Ĉm]\[B̂u*u0 + B̂d*d0 + estim.f̂op - estim.x̂op; ym0 - D̂dm*d0]
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
function evaloutput(estim::StateEstimator{NT}, d=empty(estim.x̂0)) where NT <: Real
    validate_args(estim.model, d)
    ŷ0 = Vector{NT}(undef, estim.model.ny)
    d0 = d - estim.model.dop
    ĥ!(ŷ0, estim, estim.model, estim.x̂0, d0)
    ŷ   = ŷ0
    ŷ .+= estim.model.yop
    return ŷ
end

"Functor allowing callable `StateEstimator` object as an alias for `evaloutput`."
(estim::StateEstimator)(d=empty(estim.x̂0)) = evaloutput(estim, d)

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
function updatestate!(estim::StateEstimator, u, ym, d=empty(estim.x̂0))
    validate_args(estim, u, ym, d)
    u0, ym0, d0 = remove_op!(estim, u, ym, d) 
    update_estimate!(estim, u0, ym0, d0)
    estim.x̂0 .+= estim.f̂op .- estim.x̂op
    return estim.x̂0 + estim.x̂op
end
updatestate!(::StateEstimator, _ ) = throw(ArgumentError("missing measured outputs ym"))

"""
    validate_args(estim::StateEstimator, u, ym, d)

Check `u`, `ym` and `d` sizes against `estim` dimensions.
"""
function validate_args(estim::StateEstimator, u, ym, d)
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

"""
    setmodel!(estim::StateEstimator, model::LinModel) -> estim

Set model and operating points of `estim` [`StateEstimator`](@ref) to `model` values.

Allows model adaptation of estimators based on [`LinModel`](@ref) at runtime ([`NonLinModel`](@ref)
is not supported). Not supported by [`Luenberger`](@ref) and [`SteadyKalmanFilter`](@ref),
use the time-varying [`KalmanFilter`](@ref) instead.  The [`MovingHorizonEstimator`] model 
is kept constant over the estimation horizon ``H_e``. The matrix dimensions and sample time
must stay the same. Note that the observability and controllability of the new augmented
model is not verified (see Extended Help for details).

# Examples
```jldoctest
julia> kf = KalmanFilter(LinModel(ss(0.1, 0.5, 1, 0, 4.0)));

julia> kf.model.A
1×1 Matrix{Float64}:
 0.1

julia> setmodel!(kf, LinModel(ss(0.42, 0.5, 1, 0, 4.0))); kf.model.A
1×1 Matrix{Float64}:
 0.42
```

# Extended Help

!!! details "Extended Help"
    Using the default model augmentation computed by the [`default_nint`](@ref) method, 
    switching from a non-integrating plant model to an integrating one will produce
    an augmented model that is not observable. Moving the unmeasured disturbances at the 
    model input (`nint_u` parameter) can fix this issue.
"""
function setmodel!(estim::StateEstimator, model::LinModel)
    validate_model(estim.model, model)
    # --- update model matrices and its operating points ---
    estim.model.A   .= model.A
    estim.model.Bu  .= model.Bu
    estim.model.C   .= model.C
    estim.model.Bd  .= model.Bd
    estim.model.Dd  .= model.Dd
    estim.lastu0   .+= estim.model.uop # convert u0 to u with the old operating point
    estim.model.uop .= model.uop
    estim.lastu0   .-= estim.model.uop # convert u to u0 with the new operating point
    estim.model.yop .= model.yop
    estim.model.dop .= model.dop
    estim.model.xop .= model.xop
    estim.model.fop .= model.fop
    setmodel_estimator!(estim, model)
    return estim
end

"Validate the type and dimension of the new model for adaptation."
function validate_model(old::LinModel, new::LinModel)
    new.Ts == old.Ts || throw(ArgumentError("model.Ts must be $(old.Ts) s"))
    new.nu == old.nu || throw(ArgumentError("model.nu must be $(old.nu)"))
    new.nx == old.nx || throw(ArgumentError("model.nx must be $(old.nx)"))
    new.ny == old.ny || throw(ArgumentError("model.ny must be $(old.ny)"))
    new.nd == old.nd || throw(ArgumentError("model.nd must be $(old.nd)"))
end
validate_model(::SimModel, ::SimModel) = error("Only LinModel is supported for setmodel!")

"Update the augmented model matrices of `estim` by default."
function setmodel_estimator!(estim::StateEstimator, model::LinModel)
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
    return nothing
end
