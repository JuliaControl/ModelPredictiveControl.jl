@doc raw"""
Abstract supertype of all state estimators.

---

    (estim::StateEstimator)(d=[]) -> ŷ

Functor allowing callable `StateEstimator` object as an alias for [`evaloutput`](@ref).

# Examples
```jldoctest
julia> kf = KalmanFilter(setop!(LinModel(tf(3, [10, 1]), 2), yop=[20]));

julia> ŷ = kf() 
1-element Vector{Float64}:
 20.0
```
"""
abstract type StateEstimator{NT<:Real} end

const IntVectorOrInt = Union{Int, Vector{Int}}


"""
    setstate!(estim::StateEstimator, x̂)

Set `estim.x̂` states to values specified by `x̂`. 
"""
function setstate!(estim::StateEstimator, x̂)
    size(x̂) == (estim.nx̂,) || error("x̂ size must be $((estim.nx̂,))")
    estim.x̂[:] = x̂
    return estim
end


function Base.show(io::IO, estim::StateEstimator)
    nu, nd = estim.model.nu, estim.model.nd
    nx̂, nym, nyu = estim.nx̂, estim.nym, estim.nyu
    n = maximum(ndigits.((nu, nx̂, nym, nyu, nd))) + 1
    println(io, "$(typeof(estim).name.name) estimator with a sample time "*
                "Ts = $(estim.model.Ts) s, $(typeof(estim.model).name.name) and:")
    print_estim_dim(io, estim, n)
end

"Print the overall dimensions of the state estimator `estim` with left padding `n`."
function print_estim_dim(io::IO, estim::StateEstimator, n)
    nu, nd = estim.model.nu, estim.model.nd
    nx̂, nym, nyu = estim.nx̂, estim.nym, estim.nyu
    println(io, "$(lpad(nu, n)) manipulated inputs u ($(sum(estim.nint_u)) integrating states)")
    println(io, "$(lpad(nx̂, n)) states x̂")
    println(io, "$(lpad(nym, n)) measured outputs ym ($(sum(estim.nint_ym)) integrating states)")
    println(io, "$(lpad(nyu, n)) unmeasured outputs yu")
    print(io,   "$(lpad(nd, n)) measured disturbances d")
end

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
    init_estimstoch(model, i_ym, nint_u, nint_ym) -> As, Cs_u, Cs_y, nxs, nint_u, nint_ym

Init stochastic model matrices from integrator specifications for state estimation.

The arguments `nint_u` and `nint_ym` specify how many integrators are added to each 
manipulated input and measured outputs. The function returns the state-space matrices `As`, 
`Cs_u` and `Cs_y` of the stochastic model:
```math
\begin{aligned}
\mathbf{x_{s}}(k+1)     &= \mathbf{A_s x_s}(k) + \mathbf{B_s e}(k) \\
\mathbf{y_{s_{u}}}(k)   &= \mathbf{C_{s_{u}}  x_s}(k) \\
\mathbf{y_{s_{ym}}}(k)  &= \mathbf{C_{s_{ym}} x_s}(k) 
\end{aligned}
```
where ``\mathbf{e}(k)`` is an unknown zero mean white noise and ``\mathbf{A_s} = 
\mathrm{diag}(\mathbf{A_{s_{u}}, A_{s_{ym}}})``. The estimations does not use ``\mathbf{B_s}``,
it is thus ignored. The function [`init_integrators`](@ref) builds the state-space matrices.
"""
function init_estimstoch(
    model::SimModel{NT}, i_ym, nint_u::IntVectorOrInt, nint_ym::IntVectorOrInt
) where {NT<:Real}
    nu, ny, nym = model.nu, model.ny, length(i_ym)
    As_u , Cs_u , nint_u  = init_integrators(nint_u , nu , "u")
    As_ym, Cs_ym, nint_ym = init_integrators(nint_ym, nym, "ym")
    As_y, _ , Cs_y = stoch_ym2y(model, i_ym, As_ym, zeros(NT, 0, 0), Cs_ym, zeros(NT, 0, 0))
    nxs_u, nxs_y = size(As_u, 1), size(As_y, 1)
    # combines input and output stochastic models:
    As   = [As_u zeros(NT, nxs_u, nxs_y); zeros(NT, nxs_y, nxs_u) As_y]
    Cs_u = [Cs_u zeros(NT, nu, nxs_y)]
    Cs_y = [zeros(NT, ny, nxs_u) Cs_y]
    return As, Cs_u, Cs_y, nint_u, nint_ym
end

"Validate the specified measured output indices `i_ym`."
function validate_ym(model::SimModel, i_ym)
    if length(unique(i_ym)) ≠ length(i_ym) || maximum(i_ym) > model.ny
        error("Measured output indices i_ym should contains valid and unique indices")
    end
    nym, nyu = length(i_ym), model.ny - length(i_ym)
    return nym, nyu
end

"Convert the measured outputs stochastic model `stoch_ym` to all outputs `stoch_y`."
function stoch_ym2y(model::SimModel{NT}, i_ym, Asm, Bsm, Csm, Dsm) where {NT<:Real}
    As = Asm
    Bs = Bsm
    Cs = zeros(NT, model.ny, size(Csm,2))
    Cs[i_ym,:] = Csm
    if isempty(Dsm)
        Ds = Dsm
    else
        Ds = zeros(NT, model.ny, size(Dsm,2))
        Ds[i_ym,:] = Dsm
    end
    return As, Bs, Cs, Ds
end

@doc raw"""
    init_integrators(nint, ny, varname::String) -> A, C, nint

Calc `A, C` state-space matrices from integrator specifications `nint`.

This function is used to initialize the stochastic part of the augmented model for the
design of state estimators. The vector `nint` provides how many integrators (in series) 
should be incorporated for each output. The argument should have `ny` element, except
for `nint=0` which is an alias for no integrator at all. The specific case of one integrator
per output results in `A = I` and `C = I`. The estimation does not use the `B` matrix, it 
is thus ignored. This function is called twice :

1. for the unmeasured disturbances at manipulated inputs ``\mathbf{u}``
2. for the unmeasured disturbances at measured outputs ``\mathbf{y^m}``
"""
function init_integrators(nint::IntVectorOrInt, ny, varname::String)
    if nint == 0 # alias for no integrator at all
        nint = fill(0, ny)
    end
    if length(nint) ≠ ny
        error("nint_$(varname) size ($(length(nint))) ≠ n$(varname) ($ny)")
    end
    any(nint .< 0) && error("nint_$(varname) values should be ≥ 0")
    nx = sum(nint)
    A, C = zeros(nx, nx), zeros(ny, nx)
    if nx ≠ 0
        i_A, i_g = 1, 1
        for i = 1:ny
            nint_i = nint[i]
            if nint_i ≠ 0
                rows_A = (i_A):(i_A + nint_i - 1)
                A[rows_A, rows_A] = Bidiagonal(ones(nint_i), ones(nint_i-1), :L)
                C[i, i_g+nint_i-1] = 1
                i_A += nint_i
                i_g += nint_i
            end
        end
    end
    return A, C, nint
end

@doc raw"""
    augment_model(model::LinModel, As, Cs; verify_obsv=true) -> Â, B̂u, Ĉ, B̂d, D̂d

Augment [`LinModel`](@ref) state-space matrices with the stochastic ones `As` and `Cs`.

If ``\mathbf{x}`` are `model.x` states, and ``\mathbf{x_s}``, the states defined at
[`init_estimstoch`](@ref), we define an augmented state vector ``\mathbf{x̂} = 
[ \begin{smallmatrix} \mathbf{x} \\ \mathbf{x_s} \end{smallmatrix} ]``. The method
returns the augmented matrices `Â`, `B̂u`, `Ĉ`, `B̂d` and `D̂d`:
```math
\begin{aligned}
    \mathbf{x̂}(k+1) &= \mathbf{Â x̂}(k) + \mathbf{B̂_u u}(k) + \mathbf{B̂_d d}(k) \\
    \mathbf{ŷ}(k)   &= \mathbf{Ĉ x̂}(k) + \mathbf{D̂_d d}(k)
\end{aligned}
```
An error is thrown if the augmented model is not observable and `verify_obsv == true`.
"""
function augment_model(model::LinModel{NT}, As, Cs_u, Cs_y; verify_obsv=true) where NT<:Real
    nu, nx, nd = model.nu, model.nx, model.nd
    nxs = size(As, 1)
    Â   = [model.A model.Bu*Cs_u; zeros(NT, nxs,nx) As]
    B̂u  = [model.Bu; zeros(NT, nxs, nu)]
    Ĉ   = [model.C Cs_y]
    B̂d  = [model.Bd; zeros(NT, nxs, nd)]
    D̂d  = model.Dd
    # observability on Ĉ instead of Ĉm, since it would always return false when nym ≠ ny:
    if verify_obsv && !observability(Â, Ĉ)[:isobservable]
        error("The augmented model is unobservable. You may try to use 0 integrator on "*
              "model integrating outputs with nint_ym parameter. Adding integrators at both "*
              "inputs (nint_u) and outputs (nint_ym) can also violate observability.")
    end
    return Â, B̂u, Ĉ, B̂d, D̂d
end
"Return empty matrices if `model` is not a [`LinModel`](@ref)."
function augment_model(model::SimModel{NT}, As, _ , _ ) where NT<:Real
    nu, nx, nd = model.nu, model.nx, model.nd
    nxs = size(As, 1)
    Â   = zeros(NT, 0, nx+nxs)
    B̂u  = zeros(NT, 0, nu)
    Ĉ   = zeros(NT, 0, nx+nxs)
    B̂d  = zeros(NT, 0, nd)
    D̂d  = zeros(NT, 0, nd)
    return Â, B̂u, Ĉ, B̂d, D̂d
end


@doc raw"""
    default_nint(model::LinModel, i_ym=1:model.ny, nint_u=0) -> nint_ym

Get default integrator quantity per measured outputs `nint_ym` for [`LinModel`](@ref).

The arguments `i_ym` and `nint_u` are the measured output indices and the integrator
quantity on each manipulated input, respectively. By default, one integrator is added on
each measured outputs. If ``\mathbf{Â, Ĉ}`` matrices of the augmented model become
unobservable, the integrator is removed. This approach works well for stable, integrating
and unstable `model` (see Examples).

# Examples
```jldoctest
julia> model = LinModel(append(tf(3, [10, 1]), tf(2, [1, 0]), tf(4,[-5, 1])), 1.0);

julia> nint_ym = default_nint(model)
3-element Vector{Int64}:
 1
 0
 1
```
"""
function default_nint(model::LinModel, i_ym=1:model.ny, nint_u=0)
    validate_ym(model, i_ym)
    nint_ym = fill(0, length(i_ym))
    for i in eachindex(i_ym)
        nint_ym[i]  = 1
        As, Cs_u, Cs_y = init_estimstoch(model, i_ym, nint_u, nint_ym)
        Â, _ , Ĉ = augment_model(model, As, Cs_u, Cs_y, verify_obsv=false)
        # observability on Ĉ instead of Ĉm, since it would always return false when nym ≠ ny
        observability(Â, Ĉ)[:isobservable] || (nint_ym[i] = 0)
    end
    return nint_ym
end
"""
    default_nint(model::SimModel, i_ym=1:model.ny, nint_u=0)

One integrator on each measured output by default if `model` is not a  [`LinModel`](@ref).

If the integrator quantity per manipulated input `nint_u ≠ 0`, the method returns zero
integrator on each measured output.
"""
function default_nint(model::SimModel, i_ym=1:model.ny, nint_u=0)
    validate_ym(model, i_ym)
    nint_ym = iszero(nint_u) ? fill(1, length(i_ym)) : fill(0, length(i_ym))
    return nint_ym
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
    init_estimate!(::StateEstimator, ::SimModel, _ , _ , _ )

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

include("estimator/kalman.jl")
include("estimator/luenberger.jl")
include("estimator/mhe.jl")
include("estimator/internal_model.jl")

"""
    evalŷ(estim::StateEstimator, _ , d) -> ŷ

Evaluate [`StateEstimator`](@ref) output `ŷ` from measured disturbance `d` and `estim.x̂`.

Second argument is ignored, except for [`InternalModel`](@ref).
"""
evalŷ(estim::StateEstimator, _ , d) = evaloutput(estim, d)

@doc raw"""
    evalŷ(estim::InternalModel, ym, d) -> ŷ

Get [`InternalModel`](@ref) output `ŷ` from current measured outputs `ym` and dist. `d`.

[`InternalModel`](@ref) estimator needs current measured outputs ``\mathbf{y^m}(k)`` to 
estimate its outputs ``\mathbf{ŷ}(k)``, since the strategy imposes that 
``\mathbf{ŷ^m}(k) = \mathbf{y^m}(k)`` is always true.
"""
function evalŷ(estim::InternalModel, ym, d)
    ŷ = h(estim.model, estim.x̂d, d - estim.model.dop) + estim.model.yop
    ŷ[estim.i_ym] = ym
    return ŷ
end
    