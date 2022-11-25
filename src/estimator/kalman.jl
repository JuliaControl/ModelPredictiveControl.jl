struct KalmanState <: StateEstimate
    x̂::Vector{Float64}
    P̂::Symmetric{Float64}
end

struct KalmanFilter <: StateEstimator
    model::LinModel
    state::KalmanState
    i_ym::IntRangeOrVector
    nx̂::Int
    nym::Int
    nyu::Int
    nxs::Int
    As::Matrix{Float64}
    Cs::Matrix{Float64}
    Â   ::Matrix{Float64}
    B̂u  ::Matrix{Float64}
    B̂d  ::Matrix{Float64}
    Ĉ   ::Matrix{Float64}
    D̂d  ::Matrix{Float64}
    P̂0  ::Union{Symmetric{Float64}, Diagonal{Float64}}
    Q̂   ::Union{Symmetric{Float64}, Diagonal{Float64}}
    R̂   ::Union{Symmetric{Float64}, Diagonal{Float64}}
    function KalmanFilter(model, i_ym, Asm, Csm, P̂0, Q̂, R̂)
        nx, ny = model.nx, model.ny
        nym = length(i_ym);
        nyu = ny - nym;
        nxs = size(As,1)
        nx̂ = model.nx + nxs
        size(P̂0) ≠ (nx̂, nx̂)   && error("P̂0 size ($(size(P̂0))) ≠ nx̂, nx̂ ($(nx̂, nx̂))")
        size(Q̂)  ≠ (nx̂, nx̂)   && error("Q̂ size ($(size(Q̂))) ≠ nx̂, nx̂ ($(nx̂, nx̂))")
        size(R̂)  ≠ (nym, nym) && error("R̂ size ($(size(R̂))) ≠ nym, nym ($(nym, nym))")
        # s : all model outputs, sm : measured outputs only
        As = Asm;
        Cs = zeros(ny,size(Csm,2));
        Cs[i_ym,:] = Csm;
        Â, B̂u, Ĉ, B̂d, D̂d = augment_model(model, As, Cs)
        x̂ = zeros(nx̂)
        P̂ = zeros(nx̂, nx̂)
        state = KalmanState(x̂, P̂)
        return new(
            model, 
            state, 
            i_ym, 
            nx̂, nym, nyu, nxs, 
            As, Cs,
            Â, B̂u, B̂d, Ĉ, D̂d, 
            P̂0, Q̂, R̂
        )
    end
end

function KalmanFilter(
    model::LinModel,
    i_ym::IntRangeOrVector = 1:model.ny,
    nint_ym::Vector{Int} = fill(1,length(i_ym)),
    σP0::Vector{<:Real} = 10*ones(model.nx),
    σP0_int::Vector{<:Real} = 10*ones(sum(nint_ym)),
    σQ::Vector{<:Real} = 0.1*ones(model.nx),
    σQ_int::Vector{<:Real} = 0.1*ones(sum(nint_ym)),
    σR::Vector{<:Real} = 0.1*ones(length(i_ym))
)
    if isempty(nint_ym) # nint_ym = [] : alias for no output integrator at all
        nint_ym = fill(0, length(i_ym));
    end
    Asm, Csm = init_estimstoch(model::SimModel, i_ym, nint_ym)
    # estimated covariances matrices (variance = σ²) :
    P̂0 = Diagonal([σP0  ; σP0_int   ].^2);
    Q̂  = Diagonal([σQ   ; σQ_int    ].^2);
    R̂  = Diagonal(σR.^2);
    return KalmanFilter(model, i_ym, Asm, Csm, P̂0, Q̂ , R̂)
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
function Asm, Csm = init_estimstoch(i_ym, nint_ym)
    nym = length(i_ym);
    if length(nint_ym) ≠ nym
        error("nint_ym size ($(length(nint_ym))) ≠ measured output quantity ($nym)")
    end
    any(nint_ym .< 0) && error("nint_ym values should be ≥ 0")
    nxs = sum(nint_ym)
    # --- construct stochastic model state-space matrices (integrators) ---
    Asm = Bidiagonal(zeros(nxs), zeros(nxs-1), :L)
    i_Asm = 1
    for iym = 1:nym
        nint = nint_ym[iym]
        Asm[i_Asm:i_Asm + nint - 1, i_Asm:i_Asm + nint - 1] =
                Bidiagnoal(ones(nint), ones(nint-1), :L)
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
end


@doc raw"""
    augment_model(model::LinModel, As, Cs)

Augment `LinModel` state-space matrices with stochastic ones `As` and `Cs`.

We define a augmented state vector:
```math
    \mathbf{x̂} = \begin{bmatrix} \mathbf{x} \\ \mathbf{x_s} \end{bmatrix}
```
and return the augmented matrices `Â`, `B̂u`, `Ĉ`, `B̂d`, `D̂d` for `LinModel`:
```math
\begin{aligned}
    \mathbf{x̂}(k+1) &= \mathbf{Â x̂}(k) + \mathbf{B̂_u u}(k) + \mathbf{B̂_d d}(k) \\
    \mathbf{y}(k)   &= \mathbf{Ĉ x̂}(k) + \mathbf{D̂_d d}(k)
\end{aligned}
```
"""
function augment_model(model::LinModel, As, Cs)
    nx, nxs = model.nx, size(As, 1)
    Â   = [model.A zeros(nx,nxs); zeros(nxs,nx) As]
    B̂u  = [model.B; zeros(nxs,nu)]
    Ĉ   = [model.C Cs]
    B̂d  = [model.Bd; zeros(nxs,nd)]
    D̂d  = model.Dd;
    return Â, B̂u, Ĉ, B̂d, D̂d
end