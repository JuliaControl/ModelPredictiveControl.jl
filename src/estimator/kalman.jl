struct KalmanFilter <: StateEstimator
    model::LinModel
    x̂::Vector{Float64}
    P̂::Matrix{Float64}
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
    Ĉm  ::Matrix{Float64}
    D̂dm ::Matrix{Float64}
    P̂0  ::Union{Diagonal{Float64}, Matrix{Float64}}
    Q̂   ::Union{Diagonal{Float64}, Matrix{Float64}}
    R̂   ::Union{Diagonal{Float64}, Matrix{Float64}}
    function KalmanFilter(model, i_ym, Asm, Csm, P̂0, Q̂, R̂)
        nx, ny = model.nx, model.ny
        nym = length(i_ym);
        nyu = ny - nym;
        nxs = size(Asm,1)
        nx̂ = nx + nxs
        size(P̂0) ≠ (nx̂, nx̂)   && error("P̂0 size $(size(P̂0)) ≠ nx̂, nx̂ $((nx̂, nx̂))")
        size(Q̂)  ≠ (nx̂, nx̂)   && error("Q̂ size $(size(Q̂)) ≠ nx̂, nx̂ $((nx̂, nx̂))")
        size(R̂)  ≠ (nym, nym) && error("R̂ size $(size(R̂)) ≠ nym, nym $((nym, nym))")
        # s : all model outputs, sm : measured outputs only
        As = Asm;
        Cs = zeros(ny,size(Csm,2));
        Cs[i_ym,:] = Csm;
        Â, B̂u, Ĉ, B̂d, D̂d = augment_model(model, As, Cs)
        Ĉm, D̂dm = Ĉ[i_ym, :], D̂d[i_ym, :] # measured outputs ym only
        x̂ = zeros(nx̂)
        P̂ = zeros(nx̂, nx̂)
        return new(
            model, 
            x̂, P̂, 
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs,
            Â, B̂u, B̂d, Ĉ, D̂d, 
            Ĉm, D̂dm,
            P̂0, Q̂, R̂
        )
    end
end

const IntVectorOrInt = Union{Int, Vector{Int}}

@doc raw"""
    KalmanFilter(model::LinModel; <keyword arguments>)

Construct a time-varying `KalmanFilter` based on `model`.

The process model is :
```
```

# Arguments
- `model::LinModel` : (deterministic) model for the estimations.
- `i_ym=1:model.ny` : `model` output indices that are measured ``\mathbf{y^m}``, the rest 
    are unmeasured ``\mathbf{y^u}``.
- `σP0=fill(10,model.nx)` : standard deviation vector for the initial estimate covariance 
    ``\mathbf{P}(0)`` of `model`.
- `σQ=fill(0.1,model.nx)` : standard deviation vector for the process noise covariance 
    ``\mathbf{Q}`` of `model`.
- `σR=fill(0.1,length(i_ym))` : standard deviation vector for the sensor noise covariance 
    ``\mathbf{R}`` of `model` measured outputs.
- `nint_ym=fill(1,length(i_ym))` : integrator quantity per measured outputs for the 
    stochastic model, `nint_ym=0` means no integrator at all.
- `σP0_int=fill(10,sum(nint_ym))` : standard deviation vector for the initial estimate 
    covariance of the stochastic model (composed of output integrators).
- `σQ_int=fill(0.1,sum(nint_ym))` : standard deviation vector for the process noise 
    covariance of the stochastic model (composed of output integrators).
"""
function KalmanFilter(
    model::LinModel;
    i_ym::IntRangeOrVector = 1:model.ny,
    σP0::Vector{<:Real} = fill(10, model.nx),
    σQ::Vector{<:Real} = fill(0.1, model.nx),
    σR::Vector{<:Real} = fill(0.1, length(i_ym)),
    nint_ym::IntVectorOrInt = fill(1, length(i_ym)),
    σP0_int::Vector{<:Real} = fill(10, sum(nint_ym)),
    σQ_int::Vector{<:Real} = fill(0.1, sum(nint_ym))
)
    if nint_ym == 0 # alias for no output integrator at all :
        nint_ym = fill(0, length(i_ym));
    end
    Asm, Csm = init_estimstoch(i_ym, nint_ym)
    # estimated covariances matrices (variance = σ²) :
    P̂0 = Diagonal{Float64}([σP0  ; σP0_int   ].^2);
    Q̂  = Diagonal{Float64}([σQ   ; σQ_int    ].^2);
    R̂  = Diagonal{Float64}(σR.^2);
    return KalmanFilter(model, i_ym, Asm, Csm, P̂0, Q̂ , R̂)
end

"""
    updatestate!(estim::KalmanFilter, u, ym, d=Float64[])

Update `estim.x̂`\`P̂` estimates with current inputs `u`, measured outputs `ym` and dist. `d`.
"""
function updatestate!(estim::KalmanFilter, u, ym, d=Float64[])
    u, d, ym = remove_op(estim, u, d, ym)
    A, Bu, Bd, C, Dd = estim.Â, estim.B̂u, estim.B̂d, estim.Ĉm, estim.D̂dm
    x̂, P̂, Q̂, R̂ = estim.x̂, estim.P̂, estim.Q̂, estim.R̂ 
    # --- observer gain calculation ---
    M  = (P̂*C')/(C*P̂*C'+R̂)
    Ko = A*M
    # --- next state calculation ---
    x̂[:] = A*x̂ + Bu*u + Bd*d + Ko*(ym - C*x̂ - Dd*d)
    # --- next estimation error covariance calculation ---
    P̂[:] = A*(P̂-M*C*P̂)*A' + Q̂ 
    return x̂
end

@doc raw"""
    evaloutput(estim::KalmanFilter, d=Float64[])

Evaluate `KalmanFilter` outputs `̂ŷ` from `estim.x̂` states and current disturbances `d`.
"""
function evaloutput(estim::KalmanFilter, d=Float64[])
    return estim.Ĉ*estim.x̂ + estim.D̂d*(d - estim.model.dop) + estim.model.yop
end
