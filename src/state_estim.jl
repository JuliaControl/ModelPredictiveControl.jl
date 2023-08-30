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
    println(io, "$(lpad(nu, n)) manipulated inputs u")
    println(io, "$(lpad(nx̂, n)) states x̂")
    println(io, "$(lpad(nym, n)) measured outputs ym")
    println(io, "$(lpad(nyu, n)) unmeasured outputs yu")
    print(io,   "$(lpad(nd, n)) measured disturbances d")
end

"""
    remove_op!(estim::StateEstimator, u, d, ym)

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

function validate_ym(model::SimModel, i_ym)
    if length(unique(i_ym)) ≠ length(i_ym) || maximum(i_ym) > model.ny
        error("Measured output indices i_ym should contains valid and unique indices")
    end
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
    init_estimstoch(i_ym, nint_ym::Vector{Int})

Calc stochastic model matrices from output integrators specifications for state estimation.

For closed-loop state estimators. `nint_ym is` a vector providing how many integrator should 
be added for each measured output ``\mathbf{y^m}``. The argument generates the `Asm` and 
`Csm` matrices:
```math
\begin{aligned}
\mathbf{x_s}(k+1) &= \mathbf{A_s^m x_s}(k) + \mathbf{B_s^m e}(k) \\
\mathbf{y_s^m}(k) &= \mathbf{C_s^m x_s}(k)
\end{aligned}
```
where ``\mathbf{e}(k)`` is a conceptual and unknown zero mean white noise. 
``\mathbf{B_s^m}`` is not used for closed-loop state estimators thus ignored.
"""
function init_estimstoch(i_ym, nint_ym)
    if nint_ym == 0 # alias for no output integrator at all
        nint_ym = fill(0, length(i_ym))
    end
    nym = length(i_ym);
    if length(nint_ym) ≠ nym
        error("nint_ym size ($(length(nint_ym))) ≠ measured output quantity nym ($nym)")
    end
    any(nint_ym .< 0) && error("nint_ym values should be ≥ 0")
    nxs = sum(nint_ym)
    Asm, Csm = zeros(nxs, nxs), zeros(nym, nxs)
    if nxs ≠ 0 # construct stochastic model state-space matrices (integrators) :
        i_Asm, i_Csm = 1, 1
        for iym = 1:nym
            nint = nint_ym[iym]
            if nint ≠ 0
                rows_Asm = (i_Asm):(i_Asm + nint - 1)
                Asm[rows_Asm, rows_Asm] = Bidiagonal(ones(nint), ones(nint-1), :L)
                Csm[iym, i_Csm+nint-1] = 1
                i_Asm += nint
                i_Csm += nint
            end
        end
    end
    return Asm, Csm, nint_ym
end

@doc raw"""
    augment_model(model::LinModel, As, Cs)

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
"""
function augment_model(model::LinModel, As, Cs)
    nu, nx, nd = model.nu, model.nx, model.nd
    nxs = size(As, 1)
    Â   = [model.A zeros(nx,nxs); zeros(nxs,nx) As]
    B̂u  = [model.Bu; zeros(nxs,nu)]
    Ĉ   = [model.C Cs]
    B̂d  = [model.Bd; zeros(nxs,nd)]
    D̂d  = model.Dd
    return Â, B̂u, Ĉ, B̂d, D̂d
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
    nx = estim.model.nx
    @views return [f(estim.model, x̂[1:nx], u, d); estim.As*x̂[nx+1:end]]
end

@doc raw"""
    ĥ(estim::StateEstimator, x̂, d)

Output function ``\mathbf{ĥ}`` of the augmented model, see [`f̂`](@ref) for details.
"""
function ĥ(estim::StateEstimator, x̂, d)
    # `@views` macro avoid copies with matrix slice operator e.g. [a:b]
    nx = estim.model.nx
    @views return h(estim.model, x̂[1:nx], d) + estim.Cs*x̂[nx+1:end]
end


@doc raw"""
    initstate!(estim::StateEstimator, u, ym, d=Float64[])

Init `estim.x̂` states from current inputs `u`, measured outputs `ym` and disturbances `d`.

The method tries to find a good steady-state to initialize `estim.x̂` estimate :

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
    evaloutput(estim::StateEstimator, d=Float64[])

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
    updatestate!(estim::StateEstimator, u, ym, d=Float64[])

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

include("estimator/kalman.jl")
include("estimator/luenberger.jl")
include("estimator/internal_model.jl")

"Get [`InternalModel`](@ref) output `ŷ` from current measured outputs `ym` and dist. `d`."
evalŷ(estim::InternalModel, ym, d) = evaloutput(estim,ym, d)
"Other [`StateEstimator`](@ref) ignores `ym` to evaluate `ŷ`."
evalŷ(estim::StateEstimator, _, d) = evaloutput(estim, d)