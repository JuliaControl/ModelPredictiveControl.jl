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
    function MPC(estim, Hp, Hc, Mwt, Nwt, Lwt, Cwt, ru)
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
        R̂u = repeat(r_u,Hp) # constant over Hp
        Ks, Ls = init_stoch_pred(estim, Hp) 
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
    LinMPC(estim::StateEstimator; <keyword arguments>)

Use custom state estimator `estim` to construct `LinMPC`,

`estim.model` must be a [`LinModel`](@ref). Else, a (`NonLinMPC`) is required.

# Arguments
- `estim::StateEstimator` : state estimator used for `LinMPC` predictions.
- `Hp=nothing`: prediction horizon, the default value is `10 + 
- `Hc=2` : control horizon


"""
function LinMPC(
    estim::StateEstimator;
    Hp::Int = nothing,
    Hc::Int = 2,
    Mwt = fill(1.0, model.ny),
    Nwt = fill(0.1, model.nu),
    Lwt = fill(0.0, model.nu),
    ru  = fill(0.0, model.nu),
    Cwt = 1e5
)
    isa(estim.model, LinModel) || error("estim.model type must be LinModel") 
    if isnothing(Hp)
        poles = eigvals(estim.model.A)
        max_delays = sum(poles .≈ 0)
        Hp= max_delays + 10
    end
    return LinMPC(estim, Hp, Hc, Mwt, Nwt, Lwt, Cwt, ru)
end


"""
    LinMPC(model::LinModel; <keyword arguments>)

Construct a linear model predictive cosntroller `LinMPC` based on `model`.

The default state estimator is a [`SteadyKalmanFilter`](@ref) with default arguments.

See [`LinModel`](@ref).
"""
LinMPC(model::LinModel; kwargs...) = LinMPC(SteadyKalmanFilter(model); kwargs...)


@doc raw"""
    init_stoch_pred(estim::InternalModel, Hp)

Init stochastic prediction matrix `Ks` for `PredictiveController`

``\mathbf{K_s}`` is the prediction matrix of the stochastic model (composed exclusively of 
integrators):
```math
    \mathbf{Ŷ_s} = \mathbf{P_s}[\mathbf{M_s x̂_s}(k) + \mathbf{N_s Ŵs}]
                 = \mathbf{K_s x̂_s}(k)
````
since the stochastic process noises ``\mathbf{Ŵ_s = 0}`` during MPC predictions. The 
stochastic predictions ``\mathbf{Ŷ_s}`` are the integrator outputs (from ``k+1`` 
to ``k+H_p``). ``\mathbf{x̂_s}`` is extracted from the current estimate ``\mathbf{x̂}``.

!!! note
    Stochastic predictions are calculated separately and added to ``\mathbf{F̄}`` matrix to 
    reduce MPC optimization computational costs.
"""
function init_stoch_pred(estim::StateEstimator, Hp)
    As = estim.As
    nxs = estim.nxs
    Ms = zeros(Hp*nxs, nxs)
    for i = 1:Hp
        iRow = (1:nxs) + nxs*(i-1)
        Ms[iRow, :] = As^i
    end
    Ps = repeatdiag(Cs, Hp)
    Ks = Ps*Ms
    return Ks, zeros(ny*Hp,0)
end


@doc raw"""
    init_stoch_pred(estim::InternalModel, Hp)

If `estim` is a `InternalModel`, `Ks` and `Ls`` are the stochastic prediction matrices:
```math
    \mathbf{Ŷ_s} = \mathbf{K_s x̂_s}(k) + \mathbf{L_s ŷ_s}(k)
```
with ``\mathbf{Ŷ_s}`` as stoch. predictions from ``k+1`` to ``k+H_p``, current stoch. model
states ``\mathbf{x̂_s}(k)`` and outputs ``\mathbf{ŷ_s}(k) = \mathbf{y}(k) - \mathbf{ŷ_d}(k)`` 
(except for unmeasured outputs ``\mathbf{ŷ_s^u = 0}``). See Desbiens et al. "Model-based 
predictive control: a general framework" (sec. 4.3.5)

"""
function init_stoch_pred(estim::InternalModel, Hp) 
    As, B̂s, Cs = estim.As, estim.B̂s, estim.Cs
    ny  = estim.model.ny
    nxs = estim.nxs
    Ks = zeros(ny*Hp, nxs)
    Ls = zeros(ny*Hp, ny)
    for i = 1:Hp
        iRow = (1:ny) + ny*(i-1)
        Ms = Cs*As^(i-1)*B̂s
        Ks[iRow, :] = Cs*As^i - Ms*Cs
        Ls[iRow, :] = Ms
    end
    return Ks, Ls 
end

"Generate a block diagonal matrix repeating `n` times the matrix `A`."
repeatdiag(A, n::Int) = kron(I(n), A)


#=
Init mMPC criterion and weights parameters
Mwt and Nwt are setpoint tracking and input increment weights.
Lwt is the input setpoint tracking weight, with input setpoints r_u.
Cwt is the slack variable weight for constraint softening.
Ewt is the economic cost function weight.
=#