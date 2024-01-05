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
    f̂(estim::StateEstimator, model::SimModel, x̂, u, d)

State function ``\mathbf{f̂}`` of the augmented model.

By introducing an augmented state vector ``\mathbf{x̂}`` like in [`augment_model`](@ref), the
function returns the next state of the augmented model, defined as:
```math
\begin{aligned}
    \mathbf{x̂}(k+1) &= \mathbf{f̂}\Big(\mathbf{x̂}(k), \mathbf{u}(k), \mathbf{d}(k)\Big) \\
    \mathbf{ŷ}(k)   &= \mathbf{ĥ}\Big(\mathbf{x̂}(k), \mathbf{d}(k)\Big) 
\end{aligned}
```
"""
function f̂(estim::StateEstimator, model::SimModel, x̂, u, d)
    # `@views` macro avoid copies with matrix slice operator e.g. [a:b]
    @views x̂d, x̂s = x̂[1:model.nx], x̂[model.nx+1:end]
    return [f(model, x̂d, u + estim.Cs_u*x̂s, d); estim.As*x̂s]
end
"Use the augmented model matrices if `model` is a [`LinModel`](@ref)."
f̂(estim::StateEstimator, ::LinModel, x̂, u, d) = estim.Â * x̂ + estim.B̂u * u + estim.B̂d * d

@doc raw"""
    ĥ(estim::StateEstimator, model::SimModel, x̂, d)

Output function ``\mathbf{ĥ}`` of the augmented model, see [`f̂`](@ref) for details.
"""
function ĥ(estim::StateEstimator, model::SimModel, x̂, d)
    # `@views` macro avoid copies with matrix slice operator e.g. [a:b]
    @views x̂d, x̂s = x̂[1:model.nx], x̂[model.nx+1:end]
    return h(model, x̂d, d) + estim.Cs_y*x̂s
end
"Use the augmented model matrices if `model` is a [`LinModel`](@ref)."
ĥ(estim::StateEstimator, ::LinModel, x̂, d) = estim.Ĉ * x̂ + estim.D̂d * d


@doc raw"""
    initstate!(estim::StateEstimator, u, ym, d=[]) -> x̂

Init `estim.x̂` states from current inputs `u`, measured outputs `ym` and disturbances `d`.

The method tries to find a good stead-state for the initial esitmate ``\mathbf{x̂}(0)``. It
removes the operating points with [`remove_op!`](@ref) and call [`init_estimate!`](@ref):

- If `estim.model` is a [`LinModel`](@ref), it finds the steady-state of the augmented model
  using `u` and `d` arguments, and uses the `ym` argument to enforce that 
  ``\mathbf{ŷ^m}(0) = \mathbf{y^m}(0)``. For control applications, this solution produces a
  bumpless manual to automatic transfer. See [`init_estimate!`](@ref) for details.
- Else, `estim.x̂` is left unchanged. Use [`setstate!`](@ref) to manually modify it.

If applicable, it also sets the error covariance `estim.P̂` to `estim.P̂0`.

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
function initstate!(estim::StateEstimator, u, ym, d=empty(estim.x̂))
    # --- validate arguments ---
    validate_args(estim, u, ym, d)
    # --- init state estimate ----
    u0, ym0, d0 = remove_op!(estim, u, ym, d)
    init_estimate!(estim, estim.model, u0, ym0, d0)
    # --- init covariance error estimate, if applicable ---
    init_estimate_cov!(estim, u0, ym0, d0)
    return estim.x̂
end

"By default, [`StateEstimator`](@ref)s do not need covariance error estimate."
init_estimate_cov!(::StateEstimator, _ , _ , _ ) = nothing

@doc raw"""
    init_estimate!(estim::StateEstimator, model::LinModel, u, ym, d)

Init `estim.x̂` estimate with the steady-state solution if `model` is a [`LinModel`](@ref).

Using `u`, `ym` and `d` arguments, the steady-state problem combined to the equality 
constraint ``\mathbf{ŷ^m} = \mathbf{y^m}`` engenders the following system to solve:
```math
\begin{bmatrix}
    \mathbf{I} - \mathbf{Â}             \\
    \mathbf{Ĉ^m}
\end{bmatrix} \mathbf{x̂} =
\begin{bmatrix}
    \mathbf{B̂_u u} + \mathbf{B̂_d d}     \\
    \mathbf{y^m} - \mathbf{D̂_d^m d}
\end{bmatrix}
```
in which ``\mathbf{Ĉ^m, D̂_d^m}`` are the rows of `estim.Ĉ, estim.D̂d`  that correspond to 
measured outputs ``\mathbf{y^m}``.
"""
function init_estimate!(estim::StateEstimator, ::LinModel, u, ym, d)
    Â, B̂u, Ĉ, B̂d, D̂d = estim.Â, estim.B̂u, estim.Ĉ, estim.B̂d, estim.D̂d
    Ĉm, D̂dm = Ĉ[estim.i_ym, :], D̂d[estim.i_ym, :] # measured outputs ym only
    estim.x̂[:] = [(I - Â); Ĉm]\[B̂u*u + B̂d*d; ym - D̂dm*d]
    return nothing
end
"""
    init_estimate!(estim::StateEstimator, model::SimModel, _ , _ , _ )

Left `estim.x̂` estimate unchanged if `model` is not a [`LinModel`](@ref).
"""
init_estimate!(::StateEstimator, ::SimModel, _ , _ , _ ) = nothing

@doc raw"""
    evaloutput(estim::StateEstimator, d=[]) -> ŷ

Evaluate `StateEstimator` outputs `ŷ` from `estim.x̂` states and disturbances `d`.

Calling a [`StateEstimator`](@ref) object calls this `evaloutput` method.

# Examples
```jldoctest
julia> kf = SteadyKalmanFilter(setop!(LinModel(tf(2, [10, 1]), 5), yop=[20]));

julia> ŷ = evaloutput(kf)
1-element Vector{Float64}:
 20.0
```
"""
function evaloutput(estim::StateEstimator, d=empty(estim.x̂)) 
    return ĥ(estim, estim.model, estim.x̂, d - estim.model.dop) + estim.model.yop
end

"Functor allowing callable `StateEstimator` object as an alias for `evaloutput`."
(estim::StateEstimator)(d=empty(estim.x̂)) = evaloutput(estim, d)

@doc raw"""
    updatestate!(estim::StateEstimator, u, ym, d=[]) -> x̂

Update `estim.x̂` estimate with current inputs `u`, measured outputs `ym` and dist. `d`. 

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
function updatestate!(estim::StateEstimator, u, ym, d=empty(estim.x̂))
    validate_args(estim, u, ym, d)
    u0, ym0, d0 = remove_op!(estim, u, ym, d) 
    update_estimate!(estim, u0, ym0, d0)
    return estim.x̂
end
updatestate!(::StateEstimator, _ ) = throw(ArgumentError("missing measured outputs ym"))

"""
    validate_args(estim::StateEstimator, u, ym, d)

Check `u`, `ym` and `d` sizes against `estim` dimensions.
"""
function validate_args(estim::StateEstimator, u, ym, d)
    validate_args(estim.model, u, d)
    nym = estim.nym
    size(ym) ≠ (nym,) && throw(DimensionMismatch("ym size $(size(ym)) ≠ meas. output size ($nym,)"))
end
