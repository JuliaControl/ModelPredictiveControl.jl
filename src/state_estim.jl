abstract type StateEstimator end

struct InternalModel <: StateEstimator
    model::SimModel
    nx̂::Int
    nxs::Int
    nym::Int
    nint_ym::Vector{Int}
    i_ym::IntRangeOrVector
    As::Matrix{Float64}
    Bs::Matrix{Float64}
    Cs::Matrix{Float64}
    Ds::Matrix{Float64}
    Âs::Matrix{Float64}
    B̂s::Matrix{Float64}
    function InternalModel(model::SimModel, Asm, Bsm, Csm, Dsm, i_ym)
        # TODO: check if integrators or unstable poles in model since not supported
        ny = model.ny
        if length(unique(i_ym)) ≠ length(i_ym) || maximum(i_ym) > ny
            error("Measured output indices i_ym should contains valid and unique indices")
        end
        nym = length(i_ym);
        if size(Csm,1) ≠ nym || size(Dsm,1) ≠ nym
            error("Stochastic model output quantity ($(size(Csm,1))) is different from "*
                  "measured output quantity ($nym)")
        end
        if iszero(Dsm)
            error("Stochastic model requires a direct transmission matrix D")
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
        nint_ym = fill(0,nym,)
        Âs, B̂s = init_internalmodel(As, Bs, Cs, Ds)
        return new(model, nx̂, nxs, nym, nint_ym, i_ym, As, Bs, Cs, Ds, Âs, B̂s)
    end
end

function InternalModel(
    model::SimModel, 
    stoch::StateSpace;
    i_ym::IntRangeOrVector = 1:model.ny
    )
    if iscontinuous(stoch)
        stoch_dis = c2d(stoch, model.Ts, :tustin)
    else
        stoch.Ts == model.Ts || error("stoch.Ts must be identical to model.Ts")
        stoch_dis = stoch
    end
    return InternalModel(model, stoch.A, stoch.B, stoch.C, stoch.D, i_ym)
end


function InternalModel(
    model::SimModel, 
    stoch::TransferFunction; 
    kwargs...
    )
    stoch_ss = minreal(ss(stoch))
    return InternalModel(model, stoch_ss; kwargs...)
end




@doc raw"""
    init_internalmodel(As, Bs, Cs, Ds)

Calc stochastic model update matrices `Âs` and `B̂s` for `InternalModel` estimator.

`Âs` and `B̂s` are the stochastic model update matrices :
```math
    \mathbf{\hat{x}_s}(k+1) =   \mathbf{\hat{A}_s}\mathbf{\hat{x}_s}(k) + 
                                \mathbf{\hat{B}_s}\mathbf{\hat{y}_s}(k)
```
with current stochastic model states ``\mathbf{\hat{x}_s}`` and outputs 
``\mathbf{\hat{y}_s}(k) = \mathbf{y}(k) - \mathbf{\hat{y}_d}(k)``. See Desbiens et al. 
"Model-based predictive control: a general framework" (sec. 4.3.5).
"""
function init_internalmodel(As, Bs, Cs, Ds)
    B̂s = Bs/Ds
    Âs = As - B̂s*Cs
    return Âs, B̂s
end