abstract type PredictiveController end

struct LinMPC <: PredictiveController
    model::LinModel
    estim::StateEstimator
    Hp::Int
    Hc::Int
    Mwt::Vector{Float64}
    Nwt::Vector{Float64}
    Lwt::Vector{Float64}
    Cwt::Float64
    ru::Vector{Float64}
    M_Hp::Diagonal{Float64}
    N_Hc::Diagonal{Float64}
    L_Hp::Diagonal{Float64}
    R̂u::Vector{Float64}
    Ks::Matrix{Float64}
    Ls::Matrix{Float64}
    function LinMPC(estim, Hp, Hc, Mwt, Nwt, Lwt, Cwt, ru)
        model = estim.model
        nu = model.nu
        ny = model.ny
        Hp < 1  && error("Prediction horizon Hp should be ≥ 1")
        Hc < 1  && error("Control horizon Hc should be ≥ 1")
        Hc > Hp && error("Control horizon Hc should be ≤ prediction horizon Hp")
        size(Mwt) ≠ (ny,) && error("Mwt size $(size(Mwt)) ≠ output size ($ny,)")
        size(Nwt) ≠ (nu,) && error("Nwt size $(size(Nwt)) ≠ manipulated input size ($nu,)")
        size(Lwt) ≠ (nu,) && error("Lwt size $(size(Lwt)) ≠ manipulated input size ($nu,)")
        size(ru)  ≠ (nu,) && error("ru size $(size(ru)) ≠ manipulated input size ($nu,)")
        size(Cwt) ≠ ()    && error("Cwt should be a real scalar")
        any(Mwt.<0) && error("Mwt weights should be ≥ 0")
        any(Nwt.<0) && error("Nwt weights should be ≥ 0")
        any(Lwt.<0) && error("Lwt weights should be ≥ 0")
        Cwt < 0     && error("Cwt weight should be ≥ 0")
        M_Hp = Diagonal(repeat(Mwt, Hp))
        if isinf(Cwt) # no constraint softening nor slack variable ϵ :  
            N_Hc = Diagonal(repeat(Nwt, Hc))
        else # ΔU vector is augmented with slack variable ϵ :
            N_Hc = Diagonal([repeat(Nwt, Hc); Cwt])
        end
        L_Hp = Diagonal(repeat(Lwt, Hp))
        # TODO: quick boolean test for no u setpoints (for NonLinMPC)
        R̂u = repeat(ru,Hp) # constant over Hp
        Ks, Ls = init_stochpred(estim, Hp) 
        return new(
            model, estim, 
            Hp, Hc, 
            Mwt, Nwt, Lwt, Cwt, 
            ru, 
            M_Hp, N_Hc, L_Hp, 
            R̂u, 
            Ks, Ls)
    end
end

@doc raw"""
    LinMPC(model::LinModel; <keyword arguments>)

Construct a linear model predictive controller `LinMPC` based on `model`.

The controller minimizes the following objective function at each discrete time ``k``:
```math
\min_{\mathbf{ΔU}, ϵ}   \mathbf{(R̂_y - Ŷ)}' \mathbf{M}_{H_p} \mathbf{(R̂_y - Ŷ)} + 
                        \mathbf{(ΔU)}' \mathbf{N}_{H_c} \mathbf{(ΔU)} +
                        \mathbf{(R̂_u - U)}' \mathbf{L}_{H_p} \mathbf{(R̂_u - U)} + Cϵ^2
```
in which :

- ``H_p``: prediction horizon
- ``H_c``: control horizon
- ``\mathbf{ΔU}``: manipulated input increments over ``H_c``
- ``\mathbf{Ŷ}``: predicted outputs over ``H_p``
- ``\mathbf{U}``: manipulated inputs over ``H_p``
- ``\mathbf{R̂_y}``: predicted output setpoints over ``H_p``
- ``\mathbf{R̂_u}``: predicted manipulated input setpoints over ``H_p``
- ``\mathbf{M}_{H_p} = \text{diag}\mathbf{(M,M,...,M)}``: output setpoint tracking weights
- ``\mathbf{N}_{H_c} = \text{diag}\mathbf{(N,N,...,N)}``: manipulated input increment weights
- ``\mathbf{L}_{H_p} = \text{diag}\mathbf{(L,L,...,L)}``: manipulated input setpoint 
    tracking weights
- ``C``: slack variable weight
- ``ϵ``: slack variable for constraint softening

The ``\mathbf{ΔU}`` vector includes the manipulated input increments ``\mathbf{Δu}(k+j) = 
\mathbf{u}(k + j) - \mathbf{u}(k + j - 1)`` from ``j = 0`` to ``H_c - 1``. The
manipulated input setpoint predictions ``\mathbf{R̂_u}`` are constant at ``\mathbf{r_u}``.

This method uses the default state estimator, a [`SteadyKalmanFilter`](@ref) with default
arguments.

See [`LinModel`](@ref).

# Arguments
- `model::LinModel` : model used for controller predictions and state estimations.
- `Hp=10+nk`: prediction horizon ``H_p``, `nk` is the number of delays in `model`.
- `Hc=2` : control horizon ``H_c``.
- `Mwt=fill(1.0,model.ny)` : main diagonal of ``\mathbf{M}`` weight matrix (vector)
- `Nwt=fill(0.1,model.nu)` : main diagonal of ``\mathbf{N}`` weight matrix (vector)
- `Lwt=fill(0.0,model.nu)` : main diagonal of ``\mathbf{L}`` weight matrix (vector)
- `Cwt=1e5` : slack variable weight ``C`` (scalar), use `Cwt=Inf` for hard constraints
- `ru=model.uop`: manipulated input setpoints ``\mathbf{r_u}`` (vector)

"""
LinMPC(model::LinModel; kwargs...) = LinMPC(SteadyKalmanFilter(model); kwargs...)


"""
    LinMPC(estim::StateEstimator; <keyword arguments>)

Use custom state estimator `estim` to construct `LinMPC`.

`estim.model` must be a [`LinModel`](@ref). Else, a [`NonLinMPC`](@ref) is required. 
"""
function LinMPC(
    estim::StateEstimator;
    Hp::Union{Int,Nothing} = nothing,
    Hc::Int = 2,
    Mwt = fill(1.0, estim.model.ny),
    Nwt = fill(0.1, estim.model.nu),
    Lwt = fill(0.0, estim.model.nu),
    Cwt = 1e5,
    ru  = estim.model.uop
)
    isa(estim.model, LinModel) || error("estim.model type must be LinModel") 
    poles = eigvals(estim.model.A)
    nk = sum(poles .≈ 0)
    if isnothing(Hp)
        Hp = 10 + nk
    end
    if Hp ≤ nk
        @warn("prediction horizon Hp ($Hp) ≤ number of delays in model "*
              "($nk), the closed-loop system may be zero-gain (unresponsive) or unstable")
    end
    return LinMPC(estim, Hp, Hc, Mwt, Nwt, Lwt, Cwt, ru)
end


#=
    umin  = fill(-Inf, mpc.model.nu),
    umax  = fill(+Inf, mpc.model.nu),
    Δumin = fill(-Inf, mpc.model.nu),
    Δumax = fill(+Inf, mpc.model.nu),
    ŷmin  = fill(-Inf, mpc.model.ny),
    ŷmax  = fill(+Inf, mpc.model.ny),
    c_umin = fill(0.0, mpc.model.nu),
    c_umax = fill(0.0, mpc.model.nu),
    c_ŷmin = fill(1.0, mpc.model.ny),
    c_ŷmax = fill(1.0, mpc.model.ny)
=#


"""
    setconstraint!(mpc::PredictiveController; <keyword arguments>)

TBW
"""
function setconstraint!(
    mpc::PredictiveController; 
    umin  = nothing,
    umax  = nothing,
    Δumin = nothing,
    Δumax = nothing,
    ŷmin  = nothing,
    ŷmax  = nothing,
    c_umin = nothing,
    c_umax = nothing,
    c_ŷmin = nothing,
    c_ŷmax = nothing
)
    model = mpc.model
    nu, ny = model.ny, model.ny
    if !isnothing(umin)
        size(umin)   == (nu,) || error("umin size must be $((nu,))")
        mpc.umin[:] = umin
    end
    if !isnothing(umax)
        size(umax)   == (nu,) || error("umax size must be $((nu,))")
        mpc.umax[:] = umax
    end
    if !isnothing(Δumin)
        size(Δumin)  == (nu,) || error("Δumin size must be $((nu,))")
        mpc.Δumin[:] = Δumin
    end
    if !isnothing(Δumax)
        size(Δumax)  == (nu,) || error("Δumax size must be $((nu,))")
        mpc.Δumax[:] = Δumax
    end
    if !isnothing(ŷmin)
        size(ŷmin)   == (ny,) || error("ŷmin size must be $((ny,))")
        mpc.ŷmin[:] = ŷmin
    end
    if !isnothing(ŷmax)
        size(ŷmax)   == (ny,) || error("ŷmax size must be $((ny,))")
        mpc.ŷmax[:] = ŷmax
    end
    if !isnothing(c_umin)
        size(c_umin) == (nu,) || error("c_umin size must be $((nu,))")
        any(c_umin .< 0) && error("c_umin weights should be non-negative")
        mpc.umin[:] = umin
    end
    if !isnothing(c_umax)
        size(c_umax) == (nu,) || error("c_umax size must be $((nu,))")
        any(c_umax .< 0) && error("c_umax weights should be non-negative")
        mpc.c_umin[:] = c_umin
    end
    if !isnothing(c_ŷmin)
        size(c_ŷmin) == (ny,) || error("c_ŷmin size must be $((ny,))")
        any(c_ŷmin .< 0) && error("c_ŷmin weights should be non-negative")
        mpc.c_ŷmin[:] = c_ŷmin
    end
    if !isnothing(c_ŷmax)
        size(c_ŷmax) == (ny,) || error("c_ŷmax size must be $((ny,))")
        any(c_ŷmax .< 0) && error("c_ŷmax weights should be non-negative")
        mpc.c_ŷmax[:] = c_ŷmax
    end
    init_constraint(mpc::PredictiveController)
    return mpc
end

function init_constraint(mpc::PredictiveController)
#=
    c_U_min          = repmat(c_u_min,Hc,1);
    c_U_max          = repmat(c_u_max,Hc,1);
    c_Yhat_min       = repmat(c_yhat_min,Hp,1);
    c_Yhat_max       = repmat(c_yhat_max,Hp,1);

    mMPC.c_U_min     = c_U_min;
    mMPC.c_U_max     = c_U_max;
    mMPC.c_Yhat_min  = c_Yhat_min;
    mMPC.c_Yhat_max  = c_Yhat_max;

    DU_min   = repmat(reshape(Du_min,nu,1),Hc,1);
    DU_max   = repmat(reshape(Du_max,nu,1),Hc,1);
    Yhat_min = repmat(reshape(yhat_min,ny,1),Hp,1);
    Yhat_max = repmat(reshape(yhat_max,ny,1),Hp,1);
    % Yhat precalculations for nonlinear model constaint function
    Yhat_minNonInf_i    = not(isinf(Yhat_min)) & not(linModel);
    Yhat_maxNonInf_i    = not(isinf(Yhat_max)) & not(linModel);
    Yhat_minNonInf      = Yhat_min(Yhat_minNonInf_i);
    Yhat_maxNonInf      = Yhat_max(Yhat_maxNonInf_i);
    c_Yhat_minNonInf  = c_Yhat_min(Yhat_minNonInf_i);
    c_Yhat_maxNonInf  = c_Yhat_max(Yhat_maxNonInf_i);
    U_min = repmat(reshape(u_min,nu,1),Hc,1);
    U_max = repmat(reshape(u_max,nu,1),Hc,1);
    Mc_Hc = tril(repmat(eye(nu),Hc));
    Nc_Hc = repmat(eye(nu),Hc,1);
    Mc_Hp = [Mc_Hc;repmat(eye(nu),Hp-Hc,Hc)];
    Nc_Hp = [Nc_Hc;repmat(eye(nu),Hp-Hc,1)];
    if not(isinf(mMPC.Cwt)) % delta U is augmented with slack var. eps:
        % 0 <= eps <= +inf :
        DU_min = [DU_min;0];
        DU_max = [DU_max;+inf];
        % eps impacts deltaUhc->Uhc conversion for constraints:
        Mc_Hc_min  = [Mc_Hc, + c_U_min    ];
        Mc_Hc_max  = [Mc_Hc, - c_U_max    ];
        % eps has no effect on deltaUhc->Uhc conversion for predictions:
        Mc_Hc      = [Mc_Hc,   zeros(Hc*nu,1)];
        Mc_Hp      = [Mc_Hp,   zeros(Hp*nu,1)];
    else
        Mc_Hc_min = Mc_Hc;
        Mc_Hc_max = Mc_Hc;
    end

    mMPC.DU_min   = DU_min;
    mMPC.DU_max   = DU_max;
    mMPC.U_min    = U_min;
    mMPC.U_max    = U_max;
    mMPC.Yhat_min = Yhat_min;
    mMPC.Yhat_max = Yhat_max;
    mMPC.Yhat_minNonInf_i      = Yhat_minNonInf_i;
    mMPC.Yhat_maxNonInf_i      = Yhat_maxNonInf_i;
    mMPC.Yhat_minNonInf        = Yhat_minNonInf;
    mMPC.Yhat_maxNonInf        = Yhat_maxNonInf;
    mMPC.c_Yhat_minNonInf      = c_Yhat_minNonInf; 
    mMPC.c_Yhat_maxNonInf      = c_Yhat_maxNonInf;
    mMPC.Mc_Hc      = Mc_Hc;
    mMPC.Mc_Hp      = Mc_Hp;
    mMPC.Mc_Hc_min  = Mc_Hc_min;
    mMPC.Mc_Hc_max  = Mc_Hc_max;
    mMPC.Nc_Hc      = Nc_Hc;
    mMPC.Nc_Hp      = Nc_Hp;
=#
    return nothing
end

@doc raw"""
    init_stoch_pred(estim::StateEstimator, Hp)

Init stochastic prediction matrix `Ks` from `estim` state estimator for predictive control.

``\mathbf{K_s}`` is the prediction matrix of the stochastic model (composed exclusively of 
integrators):
```math
    \mathbf{Ŷ_s} = \mathbf{P_s}[\mathbf{M_s x̂_s}(k) + \mathbf{N_s Ŵ_s}]
                 = \mathbf{K_s x̂_s}(k)
```
since the stochastic process noises ``\mathbf{Ŵ_s = 0}`` during MPC predictions. The 
stochastic predictions ``\mathbf{Ŷ_s}`` are the integrator outputs (from ``k+1`` 
to ``k+H_p``). ``\mathbf{x̂_s}`` is extracted from the current estimate ``\mathbf{x̂}``.

!!! note
    Stochastic predictions are calculated separately and added to ``\mathbf{F̄}`` matrix to 
    reduce MPC optimization computational costs.
"""
function init_stochpred(estim::StateEstimator, Hp)
    As, Cs = estim.As, estim.Cs
    nxs = estim.nxs
    Ms = zeros(Hp*nxs, nxs)
    for i = 1:Hp
        iRow = (1:nxs) .+ nxs*(i-1)
        Ms[iRow, :] = As^i
    end
    Ps = repeatdiag(Cs, Hp)
    Ks = Ps*Ms
    return Ks, zeros(estim.model.ny*Hp, 0)
end


@doc raw"""
    init_stoch_pred(estim::InternalModel, Hp)

Init the stochastic prediction matrices `Ks` and `Ls` for [`InternalModel`](@ref).

`Ks` and `Ls` matrices are defined as:
```math
    \mathbf{Ŷ_s} = \mathbf{K_s x̂_s}(k) + \mathbf{L_s ŷ_s}(k)
```
with ``\mathbf{Ŷ_s}`` as stochastic predictions from ``k + 1`` to ``k + H_p``, current 
stochastic states ``\mathbf{x̂_s}(k)`` and outputs ``\mathbf{ŷ_s}(k)``. ``\mathbf{ŷ_s}(k)``
comprises the measured outputs ``\mathbf{ŷ_s^m}(k) = \mathbf{y^m}(k) - \mathbf{ŷ_d}(k)``
and unmeasured ``\mathbf{ŷ_s^u(k) = 0}``. See [^1].

[^1]: Desbiens, A., D. Hodouin & É. Plamondon. 2000, "Global predictive control : a unified
    control structure for decoupling setpoint tracking, feedforward compensation and 
    disturbance rejection dynamics", *IEE Proceedings - Control Theory and Applications*, 
    vol. 147, no 4, https://doi.org/10.1049/ip-cta:20000443, p. 465–475, ISSN 1350-2379.
"""
function init_stochpred(estim::InternalModel, Hp) 
    As, B̂s, Cs = estim.As, estim.B̂s, estim.Cs
    ny  = estim.model.ny
    nxs = estim.nxs
    Ks = zeros(ny*Hp, nxs)
    Ls = zeros(ny*Hp, ny)
    for i = 1:Hp
        iRow = (1:ny) .+ ny*(i-1)
        Ms = Cs*As^(i-1)*B̂s
        Ks[iRow, :] = Cs*As^i - Ms*Cs
        Ls[iRow, :] = Ms
    end
    return Ks, Ls 
end

"Generate a block diagonal matrix repeating `n` times the matrix `A`."
repeatdiag(A, n::Int) = kron(I(n), A)


function Base.show(io::IO, mpc::PredictiveController)
    println(io, "$(typeof(mpc)) predictive controller with a sample time "*
                "Ts = $(mpc.model.Ts) s, $(typeof(mpc.estim)) estimator and:")
    println(io, " $(mpc.model.nu) manipulated inputs u")
    println(io, " $(mpc.estim.nx̂) states x̂")
    println(io, " $(mpc.estim.nym) measured outputs ym")
    println(io, " $(mpc.estim.nyu) unmeasured outputs yu")
    print(io,   " $(mpc.estim.model.nd) measured disturbances d")
end