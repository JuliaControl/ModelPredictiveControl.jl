@doc raw"""
Abstract supertype of all state estimators.

---

    (estim::StateEstimator)(d=Float64[])

Functor allowing callable `StateEstimator` object as an alias for [`evaloutput`](@ref).

# Examples
```jldoctest
julia> kf = KalmanFilter(setop!(LinModel(tf(3, [10, 1]), 2), yop=[20]));

julia> ŷ = kf() 
1-element Vector{Float64}:
 20.0
```
"""
abstract type StateEstimator end

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
    println(io, "$(lpad(nu, n)) manipulated inputs u")
    println(io, "$(lpad(nx̂, n)) states x̂")
    println(io, "$(lpad(nym, n)) measured outputs ym")
    println(io, "$(lpad(nyu, n)) unmeasured outputs yu")
    print(io,   "$(lpad(nd, n)) measured disturbances d")
end

"""
    remove_op!(estim::StateEstimator, u, d, ym) -> u0, d0, ym0

Remove operating points on inputs `u`, measured outputs `ym` and disturbances `d`.

Also store current inputs without operating points `u0` in `estim.lastu0`. This field is 
used for [`PredictiveController`](@ref) computations.
"""
function remove_op!(estim::StateEstimator, u, d, ym)
    u0  = u  - estim.model.uop
    d0  = d  - estim.model.dop
    ym0 = ym - estim.model.yop[estim.i_ym]
    estim.lastu0[:] = u0
    return u0, d0, ym0
end

"""
    init_estimstoch(model, i_ym, nint_u, nint_ym) -> As, Cs_u, Cs_y, nxs, nint_u, nint_ym

Init stochastic model matrices from integrator specifications for state estimation.

TBW. 

The function [`init_integrators`](@ref) builds the state-space matrice of the unmeasured
disturbance models.
"""
function init_estimstoch(model, i_ym, nint_u::IntVectorOrInt, nint_ym::IntVectorOrInt)
    nu, ny, nym = model.nu, model.ny, length(i_ym)
    As_u , Cs_u , nint_u  = init_integrators(nint_u , nu , "u")
    As_ym, Cs_ym, nint_ym = init_integrators(nint_ym, nym, "ym")
    As_y, _ , Cs_y  = stoch_ym2y(model, i_ym, As_ym, [], Cs_ym, [])
    nxs_u, nxs_y = size(As_u, 1), size(As_y, 1)
    # combines input and output stochastic models:
    As   = [As_u zeros(nxs_u, nxs_y); zeros(nxs_y, nxs_u) As_y]
    Cs_u = [Cs_u zeros(nu, nxs_y)]
    Cs_y = [zeros(ny, nxs_u) Cs_y]
    nxs = nxs_u + nxs_y
    return As, Cs_u, Cs_y, nxs, nint_u, nint_ym
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
function stoch_ym2y(model::SimModel, i_ym, Asm, Bsm, Csm, Dsm)
    As = Asm
    Bs = Bsm
    Cs = zeros(model.ny, size(Csm,2))
    Cs[i_ym,:] = Csm
    if isempty(Dsm) || Dsm == 0
        Ds = Dsm
    else
        Ds = zeros(model.ny, size(Dsm,2))
        Ds[i_ym,:] = Dsm
    end
    return As, Bs, Cs, Ds
end

@doc raw"""
    init_integrators(nint, nys, varname::String) -> As, Cs, nint

Calc state-space matrices `As, Cs` (stochastic part) from integrator specifications `nint`.

This function is used to initialize the stochastic part of the augmented model for the
design of state estimators. The vector `nint` provides how many integrators (in series) 
should be incorporated for each stochastic output ``\mathbf{y_s}``:
```math
\begin{aligned}
\mathbf{x_s}(k+1)   &= \mathbf{A_s x_s}(k) + \mathbf{B_s e}(k) \\
\mathbf{y_s}(k)     &= \mathbf{C_s x_s}(k)
\end{aligned}
```
where ``\mathbf{e}(k)`` is an unknown zero mean white noise. The estimations does not use
``\mathbf{B_s}``, it is thus ignored. Note that this function is called twice :

1. for the unmodeled disturbances at measured outputs ``\mathbf{y^m}``
2. for the unmodeled disturbances at manipulated inputs ``\mathbf{u}``
"""
function init_integrators(nint::IntVectorOrInt, nys, varname::String)
    if nint == 0 # alias for no integrator at all
        nint = fill(0, nys)
    end
    if length(nint) ≠ nys
        error("nint_$(varname) size ($(length(nint))) ≠ n$(varname) ($nys)")
    end
    any(nint .< 0) && error("nint_$(varname) values should be ≥ 0")
    nxs = sum(nint)
    As, Cs = zeros(nxs, nxs), zeros(nys, nxs)
    if nxs ≠ 0 # construct stochastic model state-space matrices (integrators) :
        i_As, i_Cs = 1, 1
        for i = 1:nys
            nint_i = nint[i]
            if nint_i ≠ 0
                rows_As = (i_As):(i_As + nint_i - 1)
                As[rows_As, rows_As] = Bidiagonal(ones(nint_i), ones(nint_i-1), :L)
                Cs[i, i_Cs+nint_i-1] = 1
                i_As += nint_i
                i_Cs += nint_i
            end
        end
    end
    return As, Cs, nint
end

@doc raw"""
    augment_model(model::LinModel, As, Cs; verify_obsv=true) -> Â, B̂u, Ĉ, B̂d, D̂d

Augment [`LinModel`](@ref) state-space matrices with the stochastic ones `As` and `Cs`.

If ``\mathbf{x}`` are `model.x` states, and ``\mathbf{x_s}``, the states defined at
[`init_integrators`](@ref), we define an augmented state vector ``\mathbf{x̂} = 
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
function augment_model(model::LinModel, As, Cs_u, Cs_y; verify_obsv=true)
    nu, nx, nd = model.nu, model.nx, model.nd
    nxs = size(As, 1)
    Â   = [model.A model.Bu*Cs_u; zeros(nxs,nx) As]
    B̂u  = [model.Bu; zeros(nxs, nu)]
    Ĉ   = [model.C Cs_y]
    B̂d  = [model.Bd; zeros(nxs, nd)]
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
augment_model(::SimModel, _ , _ , _ ) = tuple(fill(zeros(0, 0),5)...)


@doc raw"""
    default_nint(model::LinModel, i_ym=1:model.ny, nint_u=0)

Get default integrator quantity per measured outputs `nint_ym` for [`LinModel`](@ref).

The measured output ``\mathbf{y^m}`` indices are specified by `i_ym` argument. By default, 
one integrator is added on each measured outputs. If ``\mathbf{Â, Ĉ}`` matrices of the 
augmented model become unobservable, the integrator is removed. This approach works well 
for stable, integrating and unstable `model` (see Examples).

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
"""
function default_nint(model::SimModel, i_ym=1:model.ny, nint_u=0)
    validate_ym(model, i_ym)
    return fill(1, length(i_ym))
end

@doc raw"""
    f̂(estim::StateEstimator, x̂, u, d)

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
function f̂(estim::StateEstimator, x̂, u, d)
    # `@views` macro avoid copies with matrix slice operator e.g. [a:b]
    @views x̂d, x̂s = x̂[1:estim.model.nx], x̂[estim.model.nx+1:end]
    return [f(estim.model, x̂d, u + estim.Cs_u*x̂s, d); estim.As*x̂s]
end

@doc raw"""
    ĥ(estim::StateEstimator, x̂, d)

Output function ``\mathbf{ĥ}`` of the augmented model, see [`f̂`](@ref) for details.
"""
function ĥ(estim::StateEstimator, x̂, d)
    # `@views` macro avoid copies with matrix slice operator e.g. [a:b]
    @views x̂d, x̂s = x̂[1:estim.model.nx], x̂[estim.model.nx+1:end]
    return h(estim.model, x̂d, d) + estim.Cs_y*x̂s
end


@doc raw"""
    initstate!(estim::StateEstimator, u, ym, d=Float64[]) -> x̂

Init `estim.x̂` states from current inputs `u`, measured outputs `ym` and disturbances `d`.

The method set the estimation error covariance to `estim.P̂0` (if applicable) and tries to 
find a good steady-state to initialize `estim.x̂` estimate :

- If `estim.model` is a [`LinModel`](@ref), it evaluates `estim.model` steady-state (using
  [`steadystate`](@ref)) with current inputs `u` and measured disturbances `d`, and saves
  the result to `estim.x̂[1:nx].`
- Else, the current deterministic states `estim.x̂[1:nx]` are left unchanged (use 
  [`setstate!`](@ref) to manually modify them). 
  
It then estimates the measured outputs `ŷm` from these states, and the residual offset with 
current measured outputs `(ym - ŷm)` initializes the integrators of the stochastic model.
This approach ensures that ``\mathbf{ŷ^m}(0) = \mathbf{y^m}(0)``. For [`LinModel`](@ref), it 
also ensures that the estimator starts at steady-state, resulting in a bumpless manual to 
automatic transfer for control applications.

# Examples
```jldoctest
julia> estim = SteadyKalmanFilter(LinModel(tf(3, [10, 1]), 0.5), nint_ym=[2]);

julia> x̂ = initstate!(estim, [1], [3 - 0.1])
3-element Vector{Float64}:
  5.0000000000000115
  0.0
 -0.10000000000000675
```
"""
function initstate!(estim::StateEstimator, u, ym, d=Float64[])
    model = estim.model
    # --- init covariance error estimate (if applicable) ---
    initstate_cov!(estim)
    # --- init lastu0 for PredictiveControllers ---
    estim.lastu0[:] = u - model.uop
    # --- deterministic model states ---
    x̂d = init_deterstate(model, estim, u, d)
    # --- stochastic model states (integrators) ---
    ŷd = h(model, x̂d, d - model.dop) + model.yop
    ŷsm = ym - ŷd[estim.i_ym]
    nint_ym = estim.nint_ym
    i_nint_nonzero = (nint_ym .≠ 0) 
    i_lastint = cumsum(nint_ym[i_nint_nonzero])
    x̂s = zeros(estim.nxs) # xs : integrator states
    x̂s[i_lastint] = ŷsm[i_nint_nonzero]
    # --- combine both results ---
    estim.x̂[:] = [x̂d; x̂s]
    return estim.x̂
end

"By default, state estimators do not need initialization of covariance estimate."
initstate_cov!(estim::StateEstimator) = nothing

"Init deterministic state `x̂d` with steady-state value for `LinModel`."
init_deterstate(model::LinModel, _    , u, d) = steadystate(model, u, d)
"Keep current deterministic state unchanged for `NonLinModel`."
init_deterstate(model::SimModel, estim, _, _) = estim.x̂[1:model.nx]

@doc raw"""
    evaloutput(estim::StateEstimator, d=Float64[]) -> ŷ

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
function evaloutput(estim::StateEstimator, d=Float64[]) 
    return ĥ(estim, estim.x̂, d - estim.model.dop) + estim.model.yop
end

"Functor allowing callable `StateEstimator` object as an alias for `evaloutput`."
(estim::StateEstimator)(d=Float64[]) = evaloutput(estim, d)


@doc raw"""
    updatestate!(estim::StateEstimator, u, ym, d=Float64[]) -> x̂

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
function updatestate!(estim::StateEstimator, u, ym, d=Float64[])
    u0, d0, ym0 = remove_op!(estim, u, d, ym) 
    update_estimate!(estim, u0, ym0, d0)
    return estim.x̂
end
updatestate!(::StateEstimator, _ ) = throw(ArgumentError("missing measured outputs ym"))

include("estimator/kalman.jl")
include("estimator/luenberger.jl")
include("estimator/internal_model.jl")

"Get [`InternalModel`](@ref) output `ŷ` from current measured outputs `ym` and dist. `d`."
evalŷ(estim::InternalModel, ym, d) = evaloutput(estim,ym, d)
"Other [`StateEstimator`](@ref) ignores `ym` to evaluate `ŷ`."
evalŷ(estim::StateEstimator, _, d) = evaloutput(estim, d)