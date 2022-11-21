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
    Bs::Matrix{Float64}
    Cs::Matrix{Float64}
    Ds::Matrix{Float64}
    Â   ::Matrix{Float64}
    B̂u  ::Matrix{Float64}
    B̂d  ::Matrix{Float64}
    Ĉ   ::Matrix{Float64}
    D̂d  ::Matrix{Float64}
    P̂0  ::Union{Symmetric{Float64}, Diagonal{Float64}}
    Q̂   ::Union{Symmetric{Float64}, Diagonal{Float64}}
    R̂   ::Union{Symmetric{Float64}, Diagonal{Float64}}
    # TODO: Receive Asm, Bsm, Csm and Dsm like InternalModel constructor ??
    function KalmanFilter(model, i_ym, As, Bs, Cs, Ds, P̂0, Q̂, R̂)
        nx, ny = model.nx, model.ny
        nym = length(i_ym);
        nyu = ny - nym;
        nxs = size(As,1)
        nx̂ = model.nx + nxs
        size(P̂0) ≠ (nx̂, nx̂)     && error("P̂0 size ($(size(P̂0))) ≠ nx̂, nx̂ ($(nx̂, nx̂))")
        size(Q̂) ≠ (nx̂, nx̂)      && error("Q̂ size ($(size(Q̂))) ≠ nx̂, nx̂ ($(nx̂, nx̂))")
        size(R̂) ≠ (nym, nym)    && error("R̂ size ($(size(R̂))) ≠ nym, nym ($(nym, nym))")
        x̂ = zeros(nx̂)
        P̂ = zeros(nx̂, nx̂)
        state = KalmanState(x̂, P̂)
        return new(
            model, 
            state, 
            i_ym, 
            nx̂, nym, nyu, nxs, 
            As, Bs, Cs, Ds, 
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
    As, Bs, Cs, Ds = init_estimstoch(model::SimModel, i_ym, nint_ym)
    # estimated covariances matrices (variance = σ²) :
    P̂0 = Diagonal([σP0  ; σP0_int   ].^2);
    Q̂  = Diagonal([σQ   ; σQ_int    ].^2);
    R̂  = Diagonal(σR.^2);
    return KalmanFilter(model, i_ym, As, Bs, Cs, Ds, P̂0, Q̂ , R̂)
end


"""
    As, Bs, Cs, Ds = init_estimstoch(model::SimModel, i_ym, nint_ym)

Calc stochastic model matrices from output integrators specifications.

For closed-loop state estimators. nint_ym is a vector providing how many 
integrator should be added for each measured output.
"""
function As, Bs, Cs, Ds = init_estimstoch(model::SimModel, i_ym, nint_ym)
    nu, ny = model.nu, model.ny
    nym = length(i_ym);
    if length(nint_ym) ≠ nym
        error("nint_ym size ($(length(nint_ym))) ≠ measured output quantity ($nym)")
    end
    any(nint_ym .< 0) && error("nint_ym values should be ≥ 0")
    nxs = sum(nint_ym)
    nint_y = zeros(ny,1);  # zero integrator for unmeasured outputs yu 
    nint_y[i_ym] = nint_ym; 
    # --- construct stochastic model state-space matrices (integrators) ---
    As = zeros(nxs,nxs);
    i_As = 1;
    for iy = 1:ny
        nint = nint_y[iy];
        # TODO: use Bidiagonal constructor
        # lower bidiagonal matrix with ones (generated with 2 diagm calls)
        As[i_As:i_As + nint - 1, i_As:i_As + nint - 1] =
            diagm(0 => ones(nint)) + diagm(-1 => ones(nint - 1))
        i_As += nint;
    end
    Bs = zeros(nxs,nu);
    Cs = zeros(ny,nxs);
    i_Cs = 1;
    for iy = 1:ny
        nint = nint_y[iy];
        if nint ≠ 0
            Cs[iy, i_Cs+nint-1] = 1;
            i_Cs += nint;
        end    
    end
    Ds  = zeros(ny,nu);
end


function augment_model(model::LinModel, As, Bs, Cs, Ds)
    nx, nxs = model.nx, size(As, 1)
    Â   = [mMPC.A zeros(nx,nxs); zeros(nxs,nx) As]
    B̂u  = [mMPC.B; Bs]
    Ĉ   = [mMPC.C Cs]
    B̂d  = [mMPC.Bd; Bds]
    D̂d  = mMPC.Dd;
    return Â, B̂u, Ĉ, B̂d, D̂d
end