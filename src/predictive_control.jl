abstract type PredictiveController end

struct LinMPC <: PredictiveController
    model::LinModel
    estim::StateEstimator
    lastu ::Vector{Float64}
    lastΔŨ::Vector{Float64}
    Hp::Int
    Hc::Int
    M_Hp::Diagonal{Float64}
    Ñ_Hc::Diagonal{Float64}
    L_Hp::Diagonal{Float64}
    C::Float64
    R̂u::Vector{Float64}
    Umin   ::Vector{Float64}
    Umax   ::Vector{Float64}
    ΔŨmin  ::Vector{Float64}
    ΔŨmax  ::Vector{Float64}
    Ŷmin   ::Vector{Float64}
    Ŷmax   ::Vector{Float64}
    c_Umin ::Vector{Float64}
    c_Umax ::Vector{Float64}
    c_ΔUmin::Vector{Float64}
    c_ΔUmax::Vector{Float64}
    c_Ŷmin ::Vector{Float64}
    c_Ŷmax ::Vector{Float64}
    S̃_Hp::Matrix{Bool}
    T_Hp::Matrix{Bool}
    S̃_Hc::Matrix{Bool}
    T_Hc::Matrix{Bool}
    A_umin::Matrix{Float64}
    A_umax::Matrix{Float64}
    A_ŷmin::Matrix{Float64}
    A_ŷmax::Matrix{Float64}
    Ẽ ::Matrix{Float64}
    G ::Matrix{Float64}
    J ::Matrix{Float64}
    Kd::Matrix{Float64}
    Q ::Matrix{Float64}
    P̃ ::Symmetric{Float64}
    Ks::Matrix{Float64}
    Ps::Matrix{Float64}
    Yop::Vector{Float64}
    Dop::Vector{Float64}
    optmodel::OSQP.Model
    function LinMPC(estim, Hp, Hc, Mwt, Nwt, Lwt, Cwt, ru)
        model = estim.model
        nu, ny = model.nu, model.ny
        validate_weights(model, Hp, Hc, Mwt, Nwt, Lwt, Cwt, ru)
        M_Hp = Diagonal(repeat(Mwt, Hp))
        N_Hc = Diagonal(repeat(Nwt, Hc)) 
        L_Hp = Diagonal(repeat(Lwt, Hp))
        C = Cwt
        # manipulated input setpoint predictions are constant over Hp :
        R̂u = ~iszero(Lwt) ? repeat(ru, Hp) : R̂u = Float64[] 
        umin,  umax      = fill(-Inf, nu), fill(+Inf, nu)
        Δumin, Δumax     = fill(-Inf, nu), fill(+Inf, nu)
        ŷmin,  ŷmax      = fill(-Inf, ny), fill(+Inf, ny)
        c_umin, c_umax   = fill(0.0, nu),  fill(0.0, nu)
        c_Δumin, c_Δumax = fill(0.0, nu),  fill(0.0, nu)
        c_ŷmin, c_ŷmax   = fill(1.0, ny),  fill(1.0, ny)
        Umin, Umax, ΔUmin, ΔUmax, Ŷmin, Ŷmax = 
            repeat_constraints(Hp, Hc, umin, umax, Δumin, Δumax, ŷmin, ŷmax)
        c_Umin, c_Umax, c_ΔUmin, c_ΔUmax, c_Ŷmin, c_Ŷmax = 
            repeat_constraints(Hp, Hc, c_umin, c_umax, c_Δumin, c_Δumax, c_ŷmin, c_ŷmax)
        S_Hp, T_Hp, S_Hc, T_Hc = init_ΔUtoU(nu, Hp, Hc)
        E, G, J, Kd, Q = init_deterpred(model, Hp, Hc)
        A_umin, A_umax, S̃_Hp, S̃_Hc = relaxU(C, c_Umin, c_Umax, S_Hp, S_Hc)
        ΔŨmin, ΔŨmax, Ñ_Hc = relaxΔU(C, c_ΔUmin, c_ΔUmax, ΔUmin, ΔUmax, N_Hc)
        A_ŷmin, A_ŷmax, Ẽ = relaxŶ(C, c_Ŷmin, c_Ŷmax, E)
        P̃ = init_quadprog(Ẽ, S̃_Hp, M_Hp, Ñ_Hc, L_Hp)
        Ks, Ps = init_stochpred(estim, Hp)
        Yop, Dop = repeat(model.yop, Hp), repeat(model.dop, Hp)
        lastu  = model.uop
        lastΔŨ = zeros(size(P̃, 1))
        # test with OSQP package :
        optmodel = OSQP.Model()
        A = [A_umin; A_umax; A_ŷmin; A_ŷmax]
        b = [Umin; Umax; Ŷmin; Ŷmax]
        i_nonInf = .!isinf.(b)
        A = A[i_nonInf, :]
        b = b[i_nonInf]
        OSQP.setup!(optmodel; P=sparse(P̃), A=sparse(A), u=b, verbose=false)
        return new(
            model, estim, 
            lastu, lastΔŨ,
            Hp, Hc, 
            M_Hp, Ñ_Hc, L_Hp, C, R̂u,
            Umin,   Umax,   ΔŨmin,   ΔŨmax,   Ŷmin,   Ŷmax, 
            c_Umin, c_Umax, c_ΔUmin, c_ΔUmax, c_Ŷmin, c_Ŷmax, 
            S̃_Hp, T_Hp, S̃_Hc, T_Hc, 
            A_umin, A_umax, A_ŷmin, A_ŷmax,
            Ẽ, G, J, Kd, Q, P̃,
            Ks, Ps,
            Yop, Dop,
            optmodel
        )
    end
end

@doc raw"""
    LinMPC(model::LinModel; <keyword arguments>)

Construct a linear predictive controller `LinMPC` based on [`LinModel`](@ref) `model`.

The controller minimizes the following objective function at each discrete time ``k``:
```math
\min_{\mathbf{ΔU}, ϵ}   \mathbf{(R̂_y - Ŷ)}' \mathbf{M}_{H_p} \mathbf{(R̂_y - Ŷ)}  + 
                             \mathbf{(ΔU)}' \mathbf{N}_{H_c} \mathbf{(ΔU)}  +
                        \mathbf{(R̂_u - U)}' \mathbf{L}_{H_p} \mathbf{(R̂_u - U)}  + Cϵ^2
```
in which :

- ``H_p`` : prediction horizon 
- ``H_c`` : control horizon
- ``\mathbf{ΔU}`` : manipulated input increments over ``H_c``
- ``\mathbf{Ŷ}`` : predicted outputs over ``H_p``
- ``\mathbf{U}`` : manipulated inputs over ``H_p``
- ``\mathbf{R̂_y}`` : predicted output setpoints over ``H_p``
- ``\mathbf{R̂_u}`` : predicted manipulated input setpoints over ``H_p``
- ``\mathbf{M}_{H_p} = \text{diag}\mathbf{(M,M,...,M)}`` : output setpoint tracking weights
- ``\mathbf{N}_{H_c} = \text{diag}\mathbf{(N,N,...,N)}`` : manipulated input increment weights
- ``\mathbf{L}_{H_p} = \text{diag}\mathbf{(L,L,...,L)}`` : manipulated input setpoint tracking weights
- ``C`` : slack variable weight
- ``ϵ`` : slack variable for constraint softening

The ``\mathbf{ΔU}`` vector includes the manipulated input increments ``\mathbf{Δu}(k+j) = 
\mathbf{u}(k+j) - \mathbf{u}(k+j-1)`` from ``j=0`` to ``H_c-1``, the ``\mathbf{Ŷ}`` vector, 
the output predictions ``\mathbf{ŷ(k+j)}`` from ``j=1`` to ``H_p``, and the ``\mathbf{U}`` 
vector, the manipulated inputs ``\mathbf{u}(k+j)`` from ``j=0`` to ``H_p-1``. The 
manipulated input setpoint predictions ``\mathbf{R̂_u}`` are constant at ``\mathbf{r_u}``.

This method uses the default state estimator, a [`SteadyKalmanFilter`](@ref) with default
arguments.

# Arguments
- `model::LinModel` : model used for controller predictions and state estimations.
- `Hp=10+nk`: prediction horizon ``H_p``, `nk` is the number of delays in `model`.
- `Hc=2` : control horizon ``H_c``.
- `Mwt=fill(1.0,model.ny)` : main diagonal of ``\mathbf{M}`` weight matrix (vector)
- `Nwt=fill(0.1,model.nu)` : main diagonal of ``\mathbf{N}`` weight matrix (vector)
- `Lwt=fill(0.0,model.nu)` : main diagonal of ``\mathbf{L}`` weight matrix (vector)
- `Cwt=1e5` : slack variable weight ``C`` (scalar), use `Cwt=Inf` for hard constraints only
- `ru=model.uop`: manipulated input setpoints ``\mathbf{r_u}`` (vector)

# Extended Help
Manipulated inputs setpoints ``\mathbf{r_u}`` are not common but they can be interesting
for over-actuated systems, when `nu > ny` (e.g. prioritize solutions with lower economical 
costs). The default `Lwt` value implies that this feature is disabled by default.
"""
LinMPC(model::LinModel; kwargs...) = LinMPC(SteadyKalmanFilter(model); kwargs...)


"""
    LinMPC(estim::StateEstimator; <keyword arguments>)

Use custom state estimator `estim` to construct `LinMPC`.

`estim.model` must be a [`LinModel`](@ref). Else, a [`NonLinMPC`](@ref) is required. 
"""
function LinMPC(
    estim::StateEstimator;
    Hp::Union{Int, Nothing} = nothing,
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

@doc raw"""
    setconstraint!(mpc::PredictiveController; <keyword arguments>)

Set the constraint parameters of `mpc` predictive controller.

The predictive controllers support both soft and hard constraints, defined by:
```math 
\begin{alignat*}{3}
    \mathbf{u_{min}  - c_{u_{min}}}  ϵ &≤ \mathbf{u}(k+j)  &&≤ \mathbf{u_{max}  + c_{u_{max}}}  ϵ &&\qquad j = 0, 1 ,..., H_c - 1 \\
    \mathbf{Δu_{min} - c_{Δu_{min}}} ϵ &≤ \mathbf{Δu}(k+j) &&≤ \mathbf{Δu_{max} + c_{Δu_{max}}} ϵ &&\qquad j = 0, 1 ,..., H_c - 1 \\
    \mathbf{ŷ_{min}  - c_{ŷ_{min}}}  ϵ &≤ \mathbf{ŷ}(k+j)  &&≤ \mathbf{ŷ_{max}  + c_{ŷ_{max}}}  ϵ &&\qquad j = 1, 2 ,..., H_p \\
\end{alignat*}
```
and also ``ϵ ≥ 0``. All the constraint parameters are vector. Use `±Inf` values when there 
is no bound. The constraint softness parameters ``\mathbf{c}``, also called equal concern 
for relaxation, are non-negative values that specify the softness of the associated bound. 
Use `0.0` values for hard constraints. The predicted output constraints ``\mathbf{ŷ_{min}}`` 
and ``\mathbf{ŷ_{max}}`` are soft by default.

# Arguments
!!! info
    The default constraints are mentioned here for clarity but omitting a keyword argument 
    will not re-assign to its default value (defaults are set at construction only).

- `umin=fill(-Inf,nu)` : manipulated input lower bounds ``\mathbf{u_{min}}`` 
- `umax=fill(+Inf,nu)` : manipulated input upper bounds ``\mathbf{u_{max}}`` 
- `Δumin=fill(-Inf,nu)` : manipulated input increment lower bounds ``\mathbf{Δu_{min}}`` 
- `Δumax=fill(+Inf,nu)` : manipulated input increment upper bounds ``\mathbf{Δu_{max}}`` 
- `ŷmin=fill(-Inf,ny)` : predicted output lower bounds ``\mathbf{ŷ_{min}}`` 
- `ŷmax=fill(+Inf,ny)` : predicted output upper bounds ``\mathbf{ŷ_{max}}`` 
- `c_umin=fill(0.0,nu)` : `umin` softness weights ``\mathbf{c_{u_{min}}}`` 
- `c_umax=fill(0.0,nu)` : `umax` softness weights ``\mathbf{c_{u_{max}}}`` 
- `c_Δumin=fill(0.0,nu)` : `Δumin` softness weights ``\mathbf{c_{Δu_{min}}}`` 
- `c_Δumax=fill(0.0,nu)` : `Δumax` softness weights ``\mathbf{c_{Δu_{max}}}`` 
- `c_ŷmin=fill(1.0,ny)` : `ŷmin` softness weights ``\mathbf{c_{ŷ_{min}}}`` 
- `c_ŷmax=fill(1.0,ny)` : `ŷmax` softness weights ``\mathbf{c_{ŷ_{max}}}``
"""
function setconstraint!(
    mpc::PredictiveController; 
    umin = nothing,    umax  = nothing,
    Δumin = nothing,   Δumax = nothing,
    ŷmin = nothing,    ŷmax  = nothing,
    c_umin = nothing,  c_umax = nothing,
    c_Δumin = nothing, c_Δumax = nothing,
    c_ŷmin = nothing,  c_ŷmax = nothing
)
    model = mpc.model
    nu, ny = model.ny, model.ny
    Hp, Hc = mpc.Hp, mpc.Hc
    C = mpc.C
    ΔUmin, ΔUmax = mpc.ΔŨmin[1:nu*Hc], mpc.ΔŨmax[1:nu*Hc]
    Umin,  Umax  = mpc.Umin, mpc.Umax
    Ŷmin,  Ŷmax  = mpc.Ŷmin, mpc.Ŷmax
    c_Umin,  c_Umax  = mpc.c_Umin, mpc.c_Umax
    c_Ŷmin,  c_Ŷmax  = mpc.c_Ŷmin, mpc.c_Ŷmax
    c_ΔUmin, c_ΔUmax = mpc.c_ΔUmin, mpc.c_ΔUmax
    if !isnothing(umin)
        size(umin)   == (nu,) || error("umin size must be $((nu,))")
        Umin  = repeat(umin, Hc)
        mpc.Umin[:] = Umin
    end
    if !isnothing(umax)
        size(umax)   == (nu,) || error("umax size must be $((nu,))")
        Umax  = repeat(umax, Hc)
        mpc.Umax[:] = Umax
    end
    if !isnothing(Δumin)
        size(Δumin)  == (nu,) || error("Δumin size must be $((nu,))")
        ΔUmin = repeat(Δumin, Hc)
        mpc.ΔŨmin[1:nu*Hc] = ΔUmin
    end
    if !isnothing(Δumax)
        size(Δumax)  == (nu,) || error("Δumax size must be $((nu,))")
        ΔUmax = repeat(Δumax, Hc)
        mpc.ΔŨmax[1:nu*Hc] = ΔUmax
    end
    if !isnothing(ŷmin)
        size(ŷmin)   == (ny,) || error("ŷmin size must be $((ny,))")
        Ŷmin  = repeat(ŷmin, Hp)
        mpc.Ŷmin[:] = Ŷmin 
    end
    if !isnothing(ŷmax)
        size(ŷmax)   == (ny,) || error("ŷmax size must be $((ny,))")
        Ŷmax  = repeat(ŷmax, Hp)
        mpc.Ŷmax[:] = Ŷmax
    end
    if !isnothing(c_umin)
        size(c_umin) == (nu,) || error("c_umin size must be $((nu,))")
        any(c_umin .< 0) && error("c_umin weights should be non-negative")
        c_Umin  = repeat(c_umin, Hc)
        mpc.c_Umin[:] = c_Umin
    end
    if !isnothing(c_umax)
        size(c_umax) == (nu,) || error("c_umax size must be $((nu,))")
        any(c_umax .< 0) && error("c_umax weights should be non-negative")
        c_Umax  = repeat(c_umax, Hc)
        mpc.c_Umax[:] = c_Umax
    end
    if !isnothing(c_Δumin)
        size(c_Δumin) == (nu,) || error("c_Δumin size must be $((nu,))")
        any(c_Δumin .< 0) && error("c_Δumin weights should be non-negative")
        c_ΔUmin  = repeat(c_Δumin, Hc)
        mpc.c_ΔUmin[:] = c_ΔUmin
    end
    if !isnothing(c_Δumax)
        size(c_Δumax) == (nu,) || error("c_Δumax size must be $((nu,))")
        any(c_Δumax .< 0) && error("c_Δumax weights should be non-negative")
        c_ΔUmax  = repeat(c_Δumax, Hc)
        mpc.c_ΔUmax[:] = c_ΔUmax
    end
    if !isnothing(c_ŷmin)
        size(c_ŷmin) == (ny,) || error("c_ŷmin size must be $((ny,))")
        any(c_ŷmin .< 0) && error("c_ŷmin weights should be non-negative")
        c_Ŷmin  = repeat(c_ŷmin, Hp)
        mpc.c_Ŷmin[:] = c_Ŷmin
    end
    if !isnothing(c_ŷmax)
        size(c_ŷmax) == (ny,) || error("c_ŷmax size must be $((ny,))")
        any(c_ŷmax .< 0) && error("c_ŷmax weights should be non-negative")
        c_Ŷmax  = repeat(c_ŷmax, Hp)
        mpc.c_Ŷmax[:] = c_Ŷmax
    end
    if !all(isnothing.((c_umin, c_umax, c_Δumin, c_Δumax, c_ŷmin, c_ŷmax)))
        S_Hp, S_Hc = mpc.S̃_Hp[:, 1:nu*Hc], mpc.S̃_Hc[:, 1:nu*Hc]
        # N_Hc = mpc.Ñ_Hc[1:nu*Hc, 1:nu*Hc]
        E = mpc.Ẽ[:, 1:nu*Hc]
        A_umin, A_umax, _ , _ = relaxU(C, c_Umin, c_Umax, S_Hp, S_Hc)
        # ΔŨmin, ΔŨmax, _ = slackΔU(C, c_ΔUmin, c_ΔUmax, ΔUmin, ΔUmax, N_Hc)
        A_ŷmin, A_ŷmax, _ = relaxŶ(C, c_Ŷmin, c_Ŷmax, E)
        mpc.A_umin[:] = A_umin
        mpc.A_umax[:] = A_umax
        mpc.A_ŷmin[:] = A_ŷmin  
        mpc.A_ŷmax[:] = A_ŷmax
    end
    return mpc
end

"""
    moveinput!(mpc::LinMPC, ry, d=Float64[]; ym=nothing)

TBW.
"""
function moveinput!(mpc::LinMPC, ry::Vector{<:Real}, d::Vector{<:Real}=Float64[]; kwargs...)
    R̂y, D̂ = repeat(ry, 1, mpc.Hp), repeat(d, 1, mpc.Hp) # constant over Hp
    return moveinput!(mpc, ry, R̂y, d, D̂; kwargs...)
end


"""
    moveinput!(mpc::LinMPC, ry, R̂y, d=Float64[], D̂=Float64[]; ym=nothing)

Use custom output setpoints `R̂y` and measured disturbances `D̂` predictions.
"""
function moveinput!(
    mpc::LinMPC, 
    ry::Vector{<:Real}, 
    R̂y::Matrix{<:Real}, 
    d ::Vector{<:Real} = Float64[], 
    D̂ ::Matrix{<:Real} = Float64[]; 
    ym::Union{Vector{<:Real}, Nothing} = nothing
)
    R̂y, D̂ = R̂y[:], D̂[:] # convert matrices to column vectors
    x̂d, x̂s = split_state(mpc.estim)
    ŷs, Ŷs = predict_stoch(mpc, mpc.estim, x̂s, d, ym)
    F, q̃, p = init_prediction(mpc, mpc.model, d, D̂, Ŷs, R̂y, x̂d)
    A, b = init_constraint(mpc, mpc.model, F)
    ΔŨ, J = optim_objective(mpc, A, b, q̃, p)
    Δu = ΔŨ[1:mpc.model.nu] # receding horizon principle: only Δu(k) is used
    u = mpc.lastu + Δu
    mpc.lastu[:] = u
    return u
end



"""
    initstate!(mpc::PredictiveController, u, ym, d=Float64[])

Init `mpc` controller variables and the states of `mpc.estim` [`StateEstimator`](@ref).
"""
function initstate!(mpc::PredictiveController, u, ym, d=Float64[])
    mpc.lastu[:] = u
    mpc.lastΔŨ  .= 0
    return initstate!(mpc.estim, u, ym, d)
end


"""
    updatestate!(mpc::PredictiveController, u, ym, d=Float64[])

Call [`updatestate!`](@ref) on `mpc.estim` [`StateEstimator`](@ref).
"""
updatestate!(mpc::PredictiveController, u, ym, d=Float64[]) = updatestate!(mpc.estim,u,ym,d)


split_state(estim::StateEstimator) = (nx=estim.model.nx; (estim.x̂[1:nx], estim.x̂[nx+1:end]))
split_state(estim::InternalModel)  = (estim.x̂d, estim.x̂s)

predict_stoch(mpc, estim::StateEstimator, x̂s, d, _ ) = (estim.Cs*x̂s, mpc.Ks*x̂s)
function predict_stoch(mpc, estim::InternalModel, x̂s, d, ym )
    isnothing(ym) && error("Predictive controllers with InternalModel need the measured "*
                           "outputs ym in keyword argument to compute control actions u")
    ŷd = estim.model.h(estim.x̂d, d - estim.model.dop) + estim.model.yop 
    ŷs = zeros(estim.model.ny,1)
    ŷs[estim.i_ym] = ym - ŷd[estim.i_ym]  # ŷs=0 for unmeasured outputs
    Ŷs = mpc.Ks*x̂s + mpc.Ps*ŷs
    return ŷs, Ŷs
end


function init_prediction(mpc, model::LinModel, d, D̂, Ŷs, R̂y, x̂d)
    lastu0 = mpc.lastu - model.uop
    F = mpc.Kd*x̂d + mpc.Q*lastu0 + Ŷs + mpc.Yop
    if model.nd ≠ 0
        F += mpc.G*(d - model.dop) + mpc.J*(D̂ - mpc.Dop)
    end
    Ẑ = F - R̂y
    q̃ = 2*(mpc.M_Hp*mpc.Ẽ)'*Ẑ
    p = Ẑ'*mpc.M_Hp*Ẑ
    if ~isempty(mpc.R̂u)
        V̂ = (mpc.T_Hp*mpc.lastu - mpc.R̂u)
        q̃ += 2*(mpc.L_Hp*mpc.T_Hp)'*V̂
        p += V̂'*mpc.L_Hp*V̂
    end
    return F, q̃, p
end


function init_constraint(mpc, model::LinModel, F)
    # === manipulated input constraints ===
    A_u = [mpc.A_umin; mpc.A_umax]
    b_u = [(+mpc.T_Hc*mpc.lastu - mpc.Umin); (-mpc.T_Hc*mpc.lastu + mpc.Umax)]

    # === input increment constraints ===
    # TODO: add manipulated input constraints that support softening

    # === predicted output constraints ====
    A_y = [mpc.A_ŷmin; mpc.A_ŷmax]
    b_y = [(+F - mpc.Ŷmin); (-F + mpc.Ŷmax)]

    # === merging constraints ===
    A = [A_u; A_y]
    b = [b_u; b_y]
    i_nonInf = .!isinf.(b)
    A = A[i_nonInf, :]
    b = b[i_nonInf]
    return A, b
end

function optim_objective(mpc::LinMPC, A, b, q̃, p)
    # initial ΔŨ: [Δu_{k-1}(k); Δu_{k-1}(k+1); ... ; 0]
    ΔŨ0 = [mpc.lastΔŨ[(mpc.model.nu+1):(mpc.Hc*mpc.model.nu)]; zeros(mpc.model.nu)]
    if !isinf(mpc.C) # if soft constraints, append the last slack value ϵ_{k-1}:
        ΔŨ0 = [ΔŨ0; mpc.lastΔŨ[end]]
    end

    OSQP.warm_start!(mpc.optmodel; x=ΔŨ0)
    OSQP.update!(mpc.optmodel; q=q̃, u=b)

    # --- optimization ---
    @info "ModelPredictiveControl: optimizing controller objective function..."
    res = OSQP.solve!(mpc.optmodel)
    ΔŨ = res.x 
    J = res.info.obj_val + p; # optimal objective value by adding constant term p
    
        
    # --- error handling ---
    #=
    if ~isempty(deltaUhc) && all(isnan(deltaUhc))
        flag = -777;
        mess = "Optimal deltaUs are all NaN!"
    end
    if flag <= 0 
        warning("MPC optimization error message (exitflag=%d):\n%s",flag,mess)
    end
    if flag < 0 # if error, we take last value :
        deltaUhc = deltaUhc0
    end
    =#
    mpc.lastΔŨ[:] = ΔŨ
    return ΔŨ, J
end

"Repeat predictive controller constraints over prediction `Hp` and control `Hc` horizons."
function repeat_constraints(Hp, Hc, umin, umax, Δumin, Δumax, ŷmin, ŷmax)
    Umin  = repeat(umin, Hc)
    Umax  = repeat(umax, Hc)
    ΔUmin = repeat(Δumin, Hc)
    ΔUmax = repeat(Δumax, Hc)
    Ŷmin  = repeat(ŷmin, Hp)
    Ŷmax  = repeat(ŷmax, Hp)
    return Umin, Umax, ΔUmin, ΔUmax, Ŷmin, Ŷmax
end


@doc raw"""
    init_ΔUtoU(nu, Hp, Hc, C, c_Umin, c_Umax)

Init manipulated input increments to inputs conversion matrices.

The conversion from the input increments ``\mathbf{ΔU}`` to manipulated inputs over ``H_p`` 
and ``H_c`` are calculated by:
```math
\begin{aligned}
\mathbf{U} = 
    \mathbf{U}_{H_p} &= \mathbf{S}_{H_p} \mathbf{ΔU} + \mathbf{T}_{H_p} \mathbf{u}(k-1) \\
    \mathbf{U}_{H_c} &= \mathbf{S}_{H_c} \mathbf{ΔU} + \mathbf{T}_{H_c} \mathbf{u}(k-1)
\end{aligned}
```
"""
function init_ΔUtoU(nu, Hp, Hc)
    S_Hc = LowerTriangular(repeat(I(nu), Hc, Hc))
    T_Hc = repeat(I(nu), Hc)
    S_Hp = [S_Hc; repeat(I(nu), Hp - Hc, Hc)]
    T_Hp = [T_Hc; repeat(I(nu), Hp - Hc, 1)]
    return S_Hp, T_Hp, S_Hc, T_Hc
end



@doc raw"""
    init_deterpred(model::LinModel, Hp, Hc)

Construct deterministic prediction matrices for [`LinModel`](@ref) `model`.

The linear model predictions are evaluated by :
```math
\begin{aligned}
    \mathbf{Ŷ} &= \mathbf{E ΔU} + \mathbf{G d}(k) + \mathbf{J D̂} + \mathbf{K_d x̂_d}(k) 
                                                  + \mathbf{Q u}(k-1) + \mathbf{Ŷ_s}     \\
               &= \mathbf{E ΔU} + \mathbf{F}
\end{aligned}
```
where predicted outputs ``\mathbf{Ŷ}``, stochastic outputs ``\mathbf{Ŷ_s}``, and 
disturbances ``\mathbf{D̂}`` are from ``k + 1`` to ``k + H_p``. Input increments 
``\mathbf{ΔU}`` are from ``k`` to ``k + H_c - 1``. Deterministic state estimates 
``\mathbf{x̂_d}(k)`` are extracted from current estimates ``\mathbf{x̂}_{k-1}(k)``. Operating
points on ``\mathbf{u}``, ``\mathbf{d}`` and ``\mathbf{y}`` are omitted in above equations.

!!! note
    Stochastic predictions ``\mathbf{Ŷ_s}`` are calculated separately (see 
    [`init_stochpred`](@ref)) and added to ``\mathbf{F}`` matrix to support internal model 
    structure and reduce NonLinMPC computational costs.

# Extended Help
Using the ``\mathbf{A, B_u, C, B_d, D_d}`` matrices in `model` and the equation
``\mathbf{W}_j = \mathbf{C} ( ∑_{i=0}^j \mathbf{A}^i ) \mathbf{B_u}``, the prediction 
matrices are computed by :
```math
\begin{aligned}
\mathbf{E} &= \begin{bmatrix}
\mathbf{W}_{0}      & \mathbf{0}         & \cdots & \mathbf{0}              \\
\mathbf{W}_{1}      & \mathbf{0}         & \cdots & \mathbf{0}              \\
\vdots              & \vdots             & \ddots & \vdots                  \\
\mathbf{W}_{H_p-1}  & \mathbf{W}_{H_p-2} & \cdots & \mathbf{W}_{H_p-H_c+1}
\end{bmatrix}
\\
\mathbf{G} &= \begin{bmatrix}
\mathbf{C}\mathbf{A}^{0} \mathbf{B_d}     \\ 
\mathbf{C}\mathbf{A}^{1} \mathbf{B_d}     \\ 
\vdots                                    \\
\mathbf{C}\mathbf{A}^{H_p-1} \mathbf{B_d}
\end{bmatrix}
\\
\mathbf{J} &= \begin{bmatrix}
\mathbf{D_d}                              & \mathbf{0}                                & \cdots & \mathbf{0}   \\ 
\mathbf{C}\mathbf{A}^{0} \mathbf{B_d}     & \mathbf{D_d}                              & \cdots & \mathbf{0}   \\ 
\vdots                                    & \vdots                                    & \ddots & \mathbf{0}   \\
\mathbf{C}\mathbf{A}^{H_p-2} \mathbf{B_d} & \mathbf{C}\mathbf{A}^{H_p-3} \mathbf{B_d} & \cdots & \mathbf{D_d}
\end{bmatrix}
\\
\mathbf{K_d} &= \begin{bmatrix}
\mathbf{C}\mathbf{A}^{0}      \\
\mathbf{C}\mathbf{A}^{1}      \\
\vdots                        \\
\mathbf{C}\mathbf{A}^{H_p-1}
\end{bmatrix}
\\
\mathbf{Q} &= \begin{bmatrix}
\mathbf{W}_0        \\
\mathbf{W}_1        \\
\vdots              \\
\mathbf{W}_{H_p-1}
\end{bmatrix}
\end{aligned}
```
"""
function init_deterpred(model::LinModel, Hp, Hc)
    A, Bu, C, Bd, Dd = model.A, model.Bu, model.C, model.Bd, model.Dd
    nu, nx, ny, nd = model.nu, model.nx, model.ny, model.nd
    # Apow 3D array : Apow[:,:,1] = A^0, Apow[:,:,2] = A^1, ...
    Apow = Array{Float64}(undef, size(A,1), size(A,2), Hp+1)
    Apow[:,:,1] = I(nx)
    Kd = Matrix{Float64}(undef, Hp*ny, nx)
    for i=1:Hp
        Apow[:,:,i+1] = A^i
        iRow = (1:ny) .+ ny*(i-1)
        Kd[iRow,:] = C*Apow[:,:,i+1]
    end 
    # Apow_csum 3D array : Apow_csum[:,:,1] = A^0, Apow_csum[:,:,2] = A^1 + A^0, ...
    Apow_csum  = cumsum(Apow, dims=3)

    ## === manipulated inputs u ===
    Q = Matrix{Float64}(undef, Hp*ny, nu)
    for i=1:Hp
        iRow = (1:ny) .+ ny*(i-1)
        Q[iRow,:] = C*Apow_csum[:,:,i]*Bu
    end
    E = zeros(Hp*ny, Hc*nu) 
    for i=1:Hc # truncated with control horizon
        iRow = (ny*(i-1)+1):(ny*Hp)
        iCol = (1:nu) .+ nu*(i-1)
        E[iRow,iCol] = Q[iRow .- ny*(i-1),:]
    end

    ## === measured disturbances d ===
    G = Matrix{Float64}(undef, Hp*ny, nd)
    J = repeatdiag(Dd, Hp)
    if nd ≠ 0
        for i=1:Hp
            iRow = (1:ny) .+ ny*(i-1)
            G[iRow,:] = C*Apow[:,:,i]*Bd
        end
        for i=1:Hp
            iRow = (ny*i+1):(ny*Hp)
            iCol = (1:nd) .+ nd*(i-1)
            J[iRow,iCol] = G[iRow .- ny*i,:]
        end
    end
    return E, G, J, Kd, Q
end


@doc raw"""
    relaxU(C, c_Umin, c_Umax, S_Hp, S_Hc)

Augment manipulated inputs constraints with slack variable ϵ for softening.

Denoting the input increments augmented with the slack variable 
``\mathbf{ΔŨ} = [\begin{smallmatrix} \mathbf{ΔU} \\ ϵ \end{smallmatrix}]``, it returns the 
augmented conversion matrices ``\mathbf{S̃}_{H_p}`` and ``\mathbf{S̃}_{H_c}``, similar to the 
ones described at [`init_ΔUtoU`](@ref). It also returns the ``\mathbf{A}`` matrices for the
 inequality constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{u_{min}}} \\ 
    \mathbf{A_{u_{max}}} 
\end{bmatrix} \mathbf{ΔŨ} ≤
\begin{bmatrix}
    + \mathbf{T}_{H_c} \mathbf{u}(k-1) - \mathbf{U_{min}} \\
    - \mathbf{T}_{H_c} \mathbf{u}(k-1) + \mathbf{U_{max}} 
\end{bmatrix}
```
"""
function relaxU(C, c_Umin, c_Umax, S_Hp, S_Hc)
    if !isinf(C) # ΔŨ = [ΔU; ϵ]
        # ϵ impacts ΔU → U conversion for constraint calculations:
        A_umin, A_umax = -[S_Hc +c_Umin], +[S_Hc -c_Umax] 
        # ϵ has no impact on ΔU → U conversion for prediction calculations:
        S̃_Hp, S̃_Hc = [S_Hp falses(size(S_Hp, 1))], [S_Hc falses(size(S_Hc, 1))] 
    else # ΔŨ = ΔU (only hard constraints)
        A_umin, A_umax = -S_Hc, +S_Hc
        S̃_Hp, S̃_Hc = S_Hp, S_Hc
    end
    return A_umin, A_umax, S̃_Hp, S̃_Hc
end

@doc raw"""
    relaxΔU(C, c_ΔUmin, c_ΔUmax, ΔUmin, ΔUmax, N_Hc)

Augment input increments constraints with slack variable ϵ for softening.

Denoting the input increments augmented with the slack variable 
``\mathbf{ΔŨ} = [\begin{smallmatrix} \mathbf{ΔU} \\ ϵ \end{smallmatrix}]``, it returns the 
augmented constraints ``\mathbf{ΔŨ_{min}}`` and ``\mathbf{ΔŨ_{max}}`` over ``H_c``, and the 
``\mathbf{A}`` matrices for the inequality constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{Δu_{min}}} \\ 
    \mathbf{A_{Δu_{max}}}
\end{bmatrix} \mathbf{ΔŨ} ≤
\begin{bmatrix}
    TODO:
\end{bmatrix}
```
"""
function relaxΔU(C, c_ΔUmin, c_ΔUmax, ΔUmin, ΔUmax, N_Hc)
    if !isinf(C) # ΔŨ = [ΔU; ϵ]
        # 0 ≤ ϵ ≤ ∞
        ΔŨmin, ΔŨmax = [ΔUmin; 0.0], [ΔUmax; Inf]
        # the C weight is incorporated into the input increment weights N_Hc
        Ñ_Hc = Diagonal([diag(N_Hc); C])
    else # ΔŨ = ΔU (only hard constraints)
        ΔŨmin, ΔŨmax = ΔUmin, ΔUmax
        Ñ_Hc = N_Hc
    end
    return ΔŨmin, ΔŨmax, Ñ_Hc
end

@doc raw"""
    relaxŶ(C, c_Ŷmin, c_Ŷmax, E)

Augment linear output prediction constraints with slack variable ϵ for softening.

Denoting the input increments augmented with the slack variable 
``\mathbf{ΔŨ} = [\begin{smallmatrix} \mathbf{ΔU} \\ ϵ \end{smallmatrix}]``, it returns the 
``\mathbf{Ẽ}`` matrix that appears in the linear model prediction equation 
``\mathbf{Ŷ = Ẽ ΔŨ + F}``, and the ``\mathbf{A}`` matrices for the inequality constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{ŷ_{min}}} \\ 
    \mathbf{A_{ŷ_{max}}}
\end{bmatrix} \mathbf{ΔŨ} ≤
\begin{bmatrix}
    + \mathbf{F} - \mathbf{Ŷ_{min}} \\
    - \mathbf{F} + \mathbf{Ŷ_{max}}
\end{bmatrix}
```
"""
function relaxŶ(C, c_Ŷmin, c_Ŷmax, E)
    if !isinf(C) # ΔŨ = [ΔU; ϵ]
        # ϵ impacts predicted output constraint calculations:
        A_ŷmin, A_ŷmax = -[E +c_Ŷmin], +[E -c_Ŷmax] 
        # ϵ has not impact on output predictions
        Ẽ = [E zeros(size(E, 1), 1)] 
    else # ΔŨ = ΔU (only hard constraints)
        Ẽ = E
        A_ŷmin, A_ŷmax = -E, +E
    end
    return A_ŷmin, A_ŷmax, Ẽ
end



@doc raw"""
    init_quadprog(E, S_Hp, M_Hp, N_Hc, L_Hp)

Init the quadratic programming optimization matrix `P`.

The `P` matrix appears in the quadratic general form :
```math
    J = \min_{\mathbf{ΔU}} \frac{1}{2}\mathbf{(ΔU)'P(ΔU)} + \mathbf{q'(ΔU)} + p 
```
``\mathbf{P}`` is constant if the model and weights are linear and time invariant (LTI). The 
vector ``\mathbf{q}`` and scalar ``p`` need recalculation each control period ``k``. ``p`` 
does not impact the minima position. It is thus useless at optimization but required to 
evaluate the minimal ``J`` value.
"""
init_quadprog(E, S_Hp, M_Hp, N_Hc, L_Hp) = 2*Symmetric(E'*M_Hp*E + N_Hc + S_Hp'*L_Hp*S_Hp)


@doc raw"""
    init_stochpred(estim::StateEstimator, Hp)

Init the stochastic prediction matrix `Ks` from `estim` estimator for predictive control.

``\mathbf{K_s}`` is the prediction matrix of the stochastic model (composed exclusively of 
integrators):
```math
    \mathbf{Ŷ_s} = \mathbf{K_s x̂_s}(k)
```
The stochastic predictions ``\mathbf{Ŷ_s}`` are the integrator outputs from ``k+1`` to 
``k+H_p``. ``\mathbf{x̂_s}(k)`` is extracted from current estimates ``\mathbf{x̂}_{k-1}(k)``.
The method also returns the matrix ``\mathbf{P_s = 0}``, which is useless except for 
[`InternalModel`] estimators.
"""
function init_stochpred(estim::StateEstimator, Hp)
    As, Cs = estim.As, estim.Cs
    nxs = estim.nxs
    Ms = Matrix{Float64}(undef, Hp*nxs, nxs) 
    for i = 1:Hp
        iRow = (1:nxs) .+ nxs*(i-1)
        Ms[iRow, :] = As^i
    end
    Js = repeatdiag(Cs, Hp)
    Ks = Js*Ms
    Ps = zeros(estim.model.ny*Hp, 0)
    return Ks, Ps
end


@doc raw"""
    init_stochpred(estim::InternalModel, Hp)

Init the stochastic prediction matrices for [`InternalModel`](@ref).

`Ks` and `Ps` matrices are defined as:
```math
    \mathbf{Ŷ_s} = \mathbf{K_s x̂_s}(k) + \mathbf{P_s ŷ_s}(k)
```
with ``\mathbf{Ŷ_s}`` as stochastic predictions from ``k+1`` to ``k+H_p``, current 
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
    Ks = Matrix{Float64}(undef, ny*Hp, nxs)
    Ps = Matrix{Float64}(undef, ny*Hp, ny)
    for i = 1:Hp
        iRow = (1:ny) .+ ny*(i-1)
        Ms = Cs*As^(i-1)*B̂s
        Ks[iRow, :] = Cs*As^i - Ms*Cs
        Ps[iRow, :] = Ms
    end
    return Ks, Ps 
end

"Validate predictive controller weight and horizon specified values."
function validate_weights(model, Hp, Hc, Mwt, Nwt, Lwt, Cwt, ru)
    nu, ny = model.nu, model.ny
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