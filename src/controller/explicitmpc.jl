struct ExplicitMPC{
    NT<:Real, 
    SE<:StateEstimator, 
    CW<:ControllerWeights
} <: PredictiveController{NT}
    estim::SE
    transcription::SingleShooting
    Z̃::Vector{NT}
    ŷ::Vector{NT}
    Hp::Int
    Hc::Int
    nϵ::Int
    nb::Vector{Int}
    weights::CW
    R̂u::Vector{NT}
    R̂y::Vector{NT}
    lastu0::Vector{NT}
    P̃Δu::Matrix{NT}
    P̃u ::Matrix{NT} 
    Tu ::Matrix{NT}
    Tu_lastu0::Vector{NT}
    Ẽ::Matrix{NT}
    F::Vector{NT}
    G::Matrix{NT}
    J::Matrix{NT}
    K::Matrix{NT}
    V::Matrix{NT}
    B::Vector{NT}
    H̃::Hermitian{NT, Matrix{NT}}
    q̃::Vector{NT}
    r::Vector{NT}
    H̃_chol::Cholesky{NT, Matrix{NT}}
    Ks::Matrix{NT}
    Ps::Matrix{NT}
    d0::Vector{NT}
    D̂0::Vector{NT}
    D̂e::Vector{NT}
    Uop::Vector{NT}
    Yop::Vector{NT}
    Dop::Vector{NT}
    buffer::PredictiveControllerBuffer{NT}
    function ExplicitMPC{NT}(
        estim::SE, Hp, Hc, nb, weights::CW
    ) where {NT<:Real, SE<:StateEstimator, CW<:ControllerWeights}
        model = estim.model
        nu, ny, nd, nx̂ = model.nu, model.ny, model.nd, estim.nx̂
        ŷ = copy(model.yop) # dummy vals (updated just before optimization)
        nϵ = 0    # no slack variable ϵ for ExplicitMPC
        # dummy vals (updated just before optimization):
        R̂y, R̂u, Tu_lastu0 = zeros(NT, ny*Hp), zeros(NT, nu*Hp), zeros(NT, nu*Hp)
        lastu0 = zeros(NT, nu)
        transcription = SingleShooting() # explicit MPC only supports SingleShooting
        PΔu = init_ZtoΔU(estim, transcription, Hp, Hc)
        Pu, Tu = init_ZtoU(estim, transcription, Hp, Hc, nb)
        E, G, J, K, V, B = init_predmat(model, estim, transcription, Hp, Hc)
        # dummy val (updated just before optimization):
        F = zeros(NT, ny*Hp)
        P̃Δu, P̃u, Ẽ = PΔu, Pu, E # no slack variable ϵ for ExplicitMPC
        H̃ = init_quadprog(model, weights, Ẽ, P̃Δu, P̃u)
        # dummy vals (updated just before optimization):
        q̃, r = zeros(NT, size(H̃, 1)), zeros(NT, 1)
        H̃_chol = cholesky(H̃)
        Ks, Ps = init_stochpred(estim, Hp)
        # dummy vals (updated just before optimization):
        d0, D̂0, D̂e = zeros(NT, nd), zeros(NT, nd*Hp), zeros(NT, nd + nd*Hp)
        Uop, Yop, Dop = repeat(model.uop, Hp), repeat(model.yop, Hp), repeat(model.dop, Hp)
        nZ̃ = get_nZ(estim, transcription, Hp, Hc) + nϵ
        Z̃ = zeros(NT, nZ̃)
        buffer = PredictiveControllerBuffer(estim, transcription, Hp, Hc, nϵ)
        mpc = new{NT, SE, CW}(
            estim,
            transcription,
            Z̃, ŷ,
            Hp, Hc, nϵ, nb,
            weights,
            R̂u, R̂y,
            lastu0,
            P̃Δu, P̃u, Tu, Tu_lastu0,
            Ẽ, F, G, J, K, V, B,
            H̃, q̃, r,
            H̃_chol,
            Ks, Ps,
            d0, D̂0, D̂e,
            Uop, Yop, Dop,
            buffer
        )
        return mpc
    end
end

@doc raw"""
    ExplicitMPC(model::LinModel; <keyword arguments>)

Construct an explicit linear predictive controller based on [`LinModel`](@ref) `model`.

The controller minimizes the following objective function at each discrete time ``k``:
```math
\begin{aligned}
\min_{\mathbf{ΔU}}   \mathbf{(R̂_y - Ŷ)}' \mathbf{M}_{H_p} \mathbf{(R̂_y - Ŷ)}     
                   + \mathbf{(ΔU)}'      \mathbf{N}_{H_c} \mathbf{(ΔU)}        \\
                   + \mathbf{(R̂_u - U)}' \mathbf{L}_{H_p} \mathbf{(R̂_u - U)} 
\end{aligned}
```

See [`LinMPC`](@ref) for the variable definitions. This controller does not support
constraints but the computational costs are extremely low (array division), therefore 
suitable for applications that require small sample times. The keyword arguments are
identical to [`LinMPC`](@ref), except for `Cwt`, `transcription` and `optim`, which are not
supported. This controller uses a [`SingleShooting`](@ref) transcription method.

This method uses the default state estimator, a [`SteadyKalmanFilter`](@ref) with default
arguments. This controller is allocation-free.

# Examples
```jldoctest
julia> model = LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 4);

julia> mpc = ExplicitMPC(model, Mwt=[0, 1], Nwt=[0.5], Hp=30, Hc=1)
ExplicitMPC controller with a sample time Ts = 4.0 s, SteadyKalmanFilter estimator and:
 30 prediction steps Hp
  1 control steps Hc
  1 manipulated inputs u (0 integrating states)
  4 estimated states x̂
  2 measured outputs ym (2 integrating states)
  0 unmeasured outputs yu
  0 measured disturbances d
```

"""
function ExplicitMPC(
    model::LinModel; 
    Hp::Int = default_Hp(model),
    Hc::IntVectorOrInt = DEFAULT_HC,
    Mwt = fill(DEFAULT_MWT, model.ny),
    Nwt = fill(DEFAULT_NWT, model.nu),
    Lwt = fill(DEFAULT_LWT, model.nu),
    M_Hp = Diagonal(repeat(Mwt, Hp)),
    N_Hc = Diagonal(repeat(Nwt, get_Hc(move_blocking(Hp, Hc)))),
    L_Hp = Diagonal(repeat(Lwt, Hp)),
    kwargs...
) 
    estim = SteadyKalmanFilter(model; kwargs...)
    return ExplicitMPC(estim; Hp, Hc, Mwt, Nwt, Lwt, M_Hp, N_Hc, L_Hp)
end

"""
    ExplicitMPC(estim::StateEstimator; <keyword arguments>)

Use custom state estimator `estim` to construct `ExplicitMPC`.

`estim.model` must be a [`LinModel`](@ref). Else, a [`NonLinMPC`](@ref) is required.
"""
function ExplicitMPC(
    estim::SE;
    Hp::Int = default_Hp(estim.model),
    Hc::IntVectorOrInt = DEFAULT_HC,
    Mwt  = fill(DEFAULT_MWT, estim.model.ny),
    Nwt  = fill(DEFAULT_NWT, estim.model.nu),
    Lwt  = fill(DEFAULT_LWT, estim.model.nu),
    M_Hp = Diagonal(repeat(Mwt, Hp)),
    N_Hc = Diagonal(repeat(Nwt, get_Hc(move_blocking(Hp, Hc)))),
    L_Hp = Diagonal(repeat(Lwt, Hp)),
) where {NT<:Real, SE<:StateEstimator{NT}}
    isa(estim.model, LinModel) || error(MSG_LINMODEL_ERR) 
    nk = estimate_delays(estim.model)
    if Hp ≤ nk
        @warn("prediction horizon Hp ($Hp) ≤ estimated number of delays in model "*
              "($nk), the closed-loop system may be unstable or zero-gain (unresponsive)")
    end
    nb = move_blocking(Hp, Hc)
    Hc = get_Hc(nb)
    weights = ControllerWeights{NT}(estim.model, Hp, Hc, M_Hp, N_Hc, L_Hp)
    return ExplicitMPC{NT}(estim, Hp, Hc, nb, weights)
end

setconstraint!(::ExplicitMPC; kwargs...) = error("ExplicitMPC does not support constraints.")

function Base.show(io::IO, mpc::ExplicitMPC)
    Hp, Hc = mpc.Hp, mpc.Hc
    nu, nd = mpc.estim.model.nu, mpc.estim.model.nd
    nx̂, nym, nyu = mpc.estim.nx̂, mpc.estim.nym, mpc.estim.nyu
    n = maximum(ndigits.((Hp, Hc, nu, nx̂, nym, nyu, nd))) + 1
    println(io, "$(typeof(mpc).name.name) controller with a sample time Ts = "*
                "$(mpc.estim.model.Ts) s, "*
                "$(typeof(mpc.estim).name.name) estimator and:")
    println(io, "$(lpad(Hp, n)) prediction steps Hp")
    println(io, "$(lpad(Hc, n)) control steps Hc")
    print_estim_dim(io, mpc.estim, n)
end

linconstraint!(::ExplicitMPC, ::LinModel, ::TranscriptionMethod) = nothing

@doc raw"""
    optim_objective!(mpc::ExplicitMPC) -> Z̃

Analytically solve the optimization problem for [`ExplicitMPC`](@ref).

The solution is ``\mathbf{Z̃ = - H̃^{-1} q̃}``, see [`init_quadprog`](@ref).
"""
optim_objective!(mpc::ExplicitMPC) = lmul!(-1, ldiv!(mpc.Z̃, mpc.H̃_chol, mpc.q̃))

"Compute the predictions but not the terminal states if `mpc` is an [`ExplicitMPC`](@ref)."
function predict!(
    Ŷ0, x̂0end, _ , _ , _ , mpc::ExplicitMPC, ::LinModel, ::TranscriptionMethod, _ , Z̃
)
    # in-place operations to reduce allocations :
    Ŷ0    .= mul!(Ŷ0, mpc.Ẽ, Z̃) .+ mpc.F
    x̂0end .= NaN 
    return Ŷ0, x̂0end
end

"`ExplicitMPC` does not support custom nonlinear constraint, return `true`."
iszero_nc(mpc::ExplicitMPC) = true

"""
    addinfo!(info, mpc::ExplicitMPC) -> info

For [`ExplicitMPC`](@ref), add nothing to `info`.
"""
addinfo!(info, mpc::ExplicitMPC) = info


"Update the prediction matrices and Cholesky factorization."
function setmodel_controller!(mpc::ExplicitMPC, uop_old, _ )
    model, estim, transcription = mpc.estim.model, mpc.estim, mpc.transcription
    nu, ny, nd, Hp, Hc = model.nu, model.ny, model.nd, mpc.Hp, mpc.Hc
    # --- predictions matrices ---
    E, G, J, K, V, B = init_predmat(model, estim, transcription, Hp, Hc)
    Ẽ = E  # no slack variable ϵ for ExplicitMPC
    mpc.Ẽ .= Ẽ
    mpc.G .= G
    mpc.J .= J
    mpc.K .= K
    mpc.V .= V
    mpc.B .= B
    # --- quadratic programming Hessian matrix ---
    H̃ = init_quadprog(model, mpc.weights, mpc.Ẽ, mpc.P̃Δu, mpc.P̃u)
    mpc.H̃ .= H̃
    set_objective_hessian!(mpc)
    # --- operating points ---
    mpc.lastu0 .+= uop_old .- model.uop
    for i in 0:Hp-1
        mpc.Uop[(1+nu*i):(nu+nu*i)] .= model.uop
        mpc.Yop[(1+ny*i):(ny+ny*i)] .= model.yop
        mpc.Dop[(1+nd*i):(nd+nd*i)] .= model.dop
    end
    return nothing
end

"Update the Cholesky factorization of the Hessian matrix."
function set_objective_hessian!(mpc::ExplicitMPC)
    H̃_chol = cholesky(mpc.H̃)
    mpc.H̃_chol.factors .= H̃_chol.factors
    return nothing
end

"Called by plots recipes for manipulated inputs constraints."
getUcon(mpc::ExplicitMPC, nu) = fill(-Inf, mpc.Hp*nu), fill(+Inf, mpc.Hp*nu)

"Called by plots recipes for predicted output constraints."
getYcon(mpc::ExplicitMPC, ny) = fill(-Inf, mpc.Hp*ny), fill(+Inf, mpc.Hp*ny)