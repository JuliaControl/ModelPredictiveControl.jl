@doc raw"""
Abstract supertype of all state estimators.

---

    (estim::StateEstimator)(d=Float64[])

Functor allowing callable `StateEstimator` object as an alias for `evaloutput`.

# Examples
```jldoctest
julia> kf = KalmanFilter(setop!(LinModel(tf(3, [10, 1]), 2), yop=[20]));

julia> ŷ = kf() 
1-element Vector{Float64}:
 20.0
```
"""
abstract type StateEstimator end

include("estimator/internal_model.jl")
include("estimator/kalman.jl")

function Base.show(io::IO, estim::StateEstimator)
    println(io, "$(typeof(estim)) state estimator with "*
                "a sample time Ts = $(estim.model.Ts) s and:")
    println(io, " $(estim.model.nu) manipulated inputs u")
    println(io, " $(estim.nx̂) states x̂")
    println(io, " $(estim.nym) measured outputs ym")
    println(io, " $(estim.nyu) unmeasured outputs yu")
    print(io,   " $(estim.model.nd) measured disturbances d")
end

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
    Asm, Csm = init_estimstoch(model::SimModel, i_ym, nint_ym)

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
where ``\mathbf{e}(k)`` is conceptual and unknown zero mean white noise. ``\mathbf{B_s^m}``
is not used for closed-loop state estimators thus ignored.
"""
function init_estimstoch(i_ym, nint_ym)
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

Augment `LinModel` state-space matrices with the stochastic ones `As` and `Cs`.

If ``\mathbf{x_d}`` are `model.x` states, and ``\mathbf{x_s}``, the states defined at
[`init_estimstoch`](@ref), we define an augmented state vector ``\mathbf{x} = 
[ \begin{smallmatrix} \mathbf{x_d} \\ \mathbf{x_s} \end{smallmatrix} ]``. The function
returns the augmented model matrices `Â`, `B̂u`, `Ĉ`, `B̂d` and `D̂d`:
```math
\begin{aligned}
    \mathbf{x}(k+1) &= \mathbf{Â x}(k) + \mathbf{B̂_u u}(k) + \mathbf{B̂_d d}(k) \\
    \mathbf{y}(k)   &= \mathbf{Ĉ x}(k) + \mathbf{D̂_d d}(k)
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


"""
    initstate!(estim::StateEstimator, u, ym, d=Float64[])

TBW
"""
function initstate!(estim::StateEstimator, u, ym, d=Float64[])
    model = estim.model
    if isa(model, LinModel)
        # init deterministic state with steady-states at current input and disturbance :
        x̂d = steadystate(model, u, d)
    else
        # init NonLinModel with with model.x current value :
        x̂d = copy(model.x)
    end


    # TODO: CONTINUER ICI :

    # ŷd = model.h(x̂d, d - model.dop)
    # ŷsm = ym - ̂ŷd[estim.i_ym]

    # nInt_ym_nonZero_i =  (mMPC.nInt_ym .≠ 0) 
    # lastInt_i = cumsum(mMPC.nInt_ym[nInt_ym_nonZero_i])
    # xhats = zeros(mMPC.nxs); # xs : integrator states
    # xhats[lastInt_i] = yhats[nInt_ym_nonZero_i]
    # xhat = [xhatd; xhats]
    # if any(strcmpi(mMPC.feedbackStrategy,{'KF','UKF','MHE'}))
    #     estimStatus = mMPC.P0hat; # estimation error covariance matrix 
    # else
    #     estimStatus = []; # not used for const. gain observers e.g. LO
    # end

    # estim.x̂[:] = [x̂d; x̂s]
    return estim.x̂
end


"Functor allowing callable `StateEstimator` object as an alias for `evaloutput`."
(estim::StateEstimator)(d=Float64[]) = evaloutput(estim, d)