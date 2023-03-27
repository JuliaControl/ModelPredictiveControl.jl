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
    model.x̂[:] = x̂
    return estim
end


function Base.show(io::IO, estim::StateEstimator)
    println(io, "$(typeof(estim)) estimator with "*
                "a sample time Ts = $(estim.model.Ts) s and:")
    println(io, " $(estim.model.nu) manipulated inputs u")
    println(io, " $(estim.nx̂) states x̂")
    println(io, " $(estim.nym) measured outputs ym")
    println(io, " $(estim.nyu) unmeasured outputs yu")
    print(io,   " $(estim.model.nd) measured disturbances d")
end

"Remove operating points on inputs `u`, measured outputs `ym` and disturbances `d`."
function remove_op(estim::StateEstimator, u, d, ym)
    u0  = u  - estim.model.uop
    d0  = d  - estim.model.dop
    ym0 = ym - estim.model.yop[estim.i_ym]
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
    init_estimstoch(model::SimModel, i_ym, nint_ym::Vector{Int})

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
function init_estimstoch(i_ym, nint_ym::Vector{Int})
    nym = length(i_ym);
    if length(nint_ym) ≠ nym
        error("nint_ym size ($(length(nint_ym))) ≠ measured output quantity nym ($nym)")
    end
    any(nint_ym .< 0) && error("nint_ym values should be ≥ 0")
    nxs = sum(nint_ym)
    if nxs ≠ 0 # construct stochastic model state-space matrices (integrators) :
        Asm = zeros(nxs, nxs)
        i_Asm = 1
        for iym = 1:nym
            nint = nint_ym[iym]
            Asm[i_Asm:i_Asm + nint - 1, i_Asm:i_Asm + nint - 1] =
                    Bidiagonal(ones(nint), ones(nint-1), :L)
            i_Asm += nint
        end
        Csm = zeros(nym, nxs);
        i_Csm = 1;
        for iym = 1:nym
            nint = nint_ym[iym];
            if nint ≠ 0
                Csm[iym, i_Csm+nint-1] = 1;
                i_Csm += nint;
            end    
        end
    else    # no stochastic model :
        Asm, Csm = zeros(0, 0), zeros(nym, 0)
    end
    return Asm, Csm
end

@doc raw"""
    augment_model(model::LinModel, As, Cs)

Augment [`LinModel`](ref) state-space matrices with the stochastic ones `As` and `Cs`.

If ``\mathbf{x_d}`` are `model.x` states, and ``\mathbf{x_s}``, the states defined at
[`init_estimstoch`](@ref), we define an augmented state vector ``\mathbf{x} = 
[ \begin{smallmatrix} \mathbf{x_d} \\ \mathbf{x_s} \end{smallmatrix} ]``. The method
returns the augmented model functions `f̂`, `ĥ` and matrices `Â`, `B̂u`, `Ĉ`, `B̂d` and `D̂d`:
```math
\begin{aligned}
    \mathbf{x}(k+1) &= \mathbf{f̂}\Big(\mathbf{x̂}(k), \mathbf{u}(k), \mathbf{d}(k)\Big) \\
                    &= \mathbf{Â x}(k) + \mathbf{B̂_u u}(k) + \mathbf{B̂_d d}(k) \\
    \mathbf{y}(k)   &= \mathbf{ĥ}\Big(\mathbf{x̂}(k), \mathbf{d}(k)\Big) \\
                    &= \mathbf{Ĉ x}(k) + \mathbf{D̂_d d}(k)
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
    f̂(x̂, u, d) = Â*x̂ + B̂u*u + B̂d*d
    ĥ(x̂, d) = Ĉ*x̂ + D̂d*d
    return f̂, ĥ, Â, B̂u, Ĉ, B̂d, D̂d
end

"""
    augment_model(model::NonLinModel, As, Cs)

Only returns the augmented functions `f̂`, `ĥ` when `model` is a [`NonLinModel`](@ref).
"""
function augment_model(model::NonLinModel, As, Cs)
    f̂(x̂, u, d) = [model.f(x̂[1:model.nx], u, d); As*x̂[model.nx+1:end]]
    ĥ(x̂, d) = model.h(x̂[1:model.nx], d) + Cs*x̂[model.nx+1:end]
    return f̂, ĥ
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
    # --- deterministic model states ---
    x̂d = isa(model, LinModel) ? steadystate(model, u, d) : estim.x̂[1:model.nx]
    # --- stochastic model states (integrators) ---
    ŷd = model.h(x̂d, d - model.dop) + model.yop
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
    return estim.ĥ(estim.x̂, d - estim.model.dop) + estim.model.yop
end

"Functor allowing callable `StateEstimator` object as an alias for `evaloutput`."
(estim::StateEstimator)(d=Float64[]) = evaloutput(estim, d)


include("estimator/kalman.jl")
include("estimator/internal_model.jl")
