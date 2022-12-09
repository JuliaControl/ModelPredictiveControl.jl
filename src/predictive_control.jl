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

"""
    LinMPC(model::LinModel; <keyword arguments>)

Construct a linear model predictive controller `LinMPC` based on `model`.

The default state estimator is a [`SteadyKalmanFilter`](@ref) with its default arguments.

See [`LinModel`](@ref).

# Arguments
- `model::LinModel` : model used for controller predictions and state estimations.
- `Hp=10+nk`: prediction horizon, `nk` is the number of delays in `model`.
- `Hc=2` : control horizon.
- `Mwt=fill(1.0,model.ny)` : output setpoint tracking weights (vector)
- `Nwt=fill(0.1,model.nu)` : manipulated input increment weights (vector)
- `Cwt=1e5` : slack variable weight for constraint softening (scalar) 
- `Lwt=fill(0.0,model.nu)` : manipulated input setpoint tracking weights (vector)
- `ru=model.uop`: manipulated input setpoints (vector)

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
    Cwt = 1e5,
    Lwt = fill(0.0, estim.model.nu),
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
                "Ts = $(mpc.model.Ts) s, $(typeof(mpc.estim)) state estimator and:")
    println(io, " $(mpc.model.nu) manipulated inputs u")
    println(io, " $(mpc.estim.nx̂) states x̂")
    println(io, " $(mpc.estim.nym) measured outputs ym")
    println(io, " $(mpc.estim.nyu) unmeasured outputs yu")
    print(io,   " $(mpc.estim.model.nd) measured disturbances d")
end