const DEFAULT_NONLINMPC_OPTIMIZER = optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes")

struct NonLinMPC{
    NT<:Real, 
    SE<:StateEstimator, 
    JM<:JuMP.GenericModel, 
    JEfunc<:Function,
    P<:Any
} <: PredictiveController{NT}
    estim::SE
    # note: `NT` and the number type `JNT` in `JuMP.GenericModel{JNT}` can be
    # different since solvers that support non-Float64 are scarce.
    optim::JM
    con::ControllerConstraint{NT}
    ΔŨ::Vector{NT}
    ŷ ::Vector{NT}
    Hp::Int
    Hc::Int
    nϵ::Int
    M_Hp::Hermitian{NT, Matrix{NT}}
    Ñ_Hc::Hermitian{NT, Matrix{NT}}
    L_Hp::Hermitian{NT, Matrix{NT}}
    E::NT
    JE::JEfunc
    p::P
    R̂u0::Vector{NT}
    R̂y0::Vector{NT}
    noR̂u::Bool
    S̃::Matrix{NT}
    T::Matrix{NT}
    T_lastu0::Vector{NT}
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
    Ks::Matrix{NT}
    Ps::Matrix{NT}
    d0::Vector{NT}
    D̂0::Vector{NT}
    D̂E::Vector{NT}
    Uop::Vector{NT}
    Yop::Vector{NT}
    Dop::Vector{NT}
    buffer::PredictiveControllerBuffer{NT}
    function NonLinMPC{NT, SE, JM, JEFunc, P}(
        estim::SE, Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt, Ewt, JE::JEFunc, p::P, optim::JM
    ) where {NT<:Real, SE<:StateEstimator, JM<:JuMP.GenericModel, JEFunc<:Function, P<:Any}
        model = estim.model
        nu, ny, nd, nx̂ = model.nu, model.ny, model.nd, estim.nx̂
        ŷ = copy(model.yop) # dummy vals (updated just before optimization)
        validate_weights(model, Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt, Ewt)
        # convert `Diagonal` to normal `Matrix` if required:
        M_Hp = Hermitian(convert(Matrix{NT}, M_Hp), :L) 
        N_Hc = Hermitian(convert(Matrix{NT}, N_Hc), :L)
        L_Hp = Hermitian(convert(Matrix{NT}, L_Hp), :L)
        # dummy vals (updated just before optimization):
        R̂y0, R̂u0, T_lastu0 = zeros(NT, ny*Hp), zeros(NT, nu*Hp), zeros(NT, nu*Hp)
        noR̂u = iszero(L_Hp)
        S, T = init_ΔUtoU(model, Hp, Hc)
        E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂ = init_predmat(estim, model, Hp, Hc)
        # dummy vals (updated just before optimization):
        F, fx̂  = zeros(NT, ny*Hp), zeros(NT, nx̂)
        con, nϵ, S̃, Ñ_Hc, Ẽ = init_defaultcon_mpc(
            estim, Hp, Hc, Cwt, S, N_Hc, E, ex̂, fx̂, gx̂, jx̂, kx̂, vx̂, bx̂
        )
        H̃ = init_quadprog(model, Ẽ, S̃, M_Hp, Ñ_Hc, L_Hp)
        # dummy vals (updated just before optimization):
        q̃, r = zeros(NT, size(H̃, 1)), zeros(NT, 1)
        Ks, Ps = init_stochpred(estim, Hp)
        # dummy vals (updated just before optimization):
        d0, D̂0, D̂E = zeros(NT, nd), zeros(NT, nd*Hp), zeros(NT, nd + nd*Hp)
        Uop, Yop, Dop = repeat(model.uop, Hp), repeat(model.yop, Hp), repeat(model.dop, Hp)
        nΔŨ = size(Ẽ, 2)
        ΔŨ = zeros(NT, nΔŨ)
        buffer = PredictiveControllerBuffer{NT}(nu, ny, nd, Hp)
        mpc = new{NT, SE, JM, JEFunc, P}(
            estim, optim, con,
            ΔŨ, ŷ,
            Hp, Hc, nϵ,
            M_Hp, Ñ_Hc, L_Hp, Ewt, JE, p,
            R̂u0, R̂y0, noR̂u,
            S̃, T, T_lastu0,
            Ẽ, F, G, J, K, V, B,
            H̃, q̃, r,
            Ks, Ps,
            d0, D̂0, D̂E,
            Uop, Yop, Dop,
            buffer
        )
        init_optimization!(mpc, model, optim)
        return mpc
    end
end

@doc raw"""
    NonLinMPC(model::SimModel; <keyword arguments>)

Construct a nonlinear predictive controller based on [`SimModel`](@ref) `model`.

Both [`NonLinModel`](@ref) and [`LinModel`](@ref) are supported (see Extended Help). The 
controller minimizes the following objective function at each discrete time ``k``:
```math
\begin{aligned}
\min_{\mathbf{ΔU}, ϵ}\ & \mathbf{(R̂_y - Ŷ)}' \mathbf{M}_{H_p} \mathbf{(R̂_y - Ŷ)}   
                       + \mathbf{(ΔU)}'      \mathbf{N}_{H_c} \mathbf{(ΔU)}        \\&
                       + \mathbf{(R̂_u - U)}' \mathbf{L}_{H_p} \mathbf{(R̂_u - U)} 
                       + C ϵ^2  
                       + E J_E(\mathbf{U}_E, \mathbf{Ŷ}_E, \mathbf{D̂}_E, \mathbf{p})
\end{aligned}
```
See [`LinMPC`](@ref) for the variable definitions. The custom economic function ``J_E`` can
penalizes solutions with high economic costs. Setting all the weights to 0 except ``E`` 
creates a pure economic model predictive controller (EMPC). The arguments of ``J_E`` include
the manipulated inputs, the predicted outputs and measured disturbances from ``k`` to 
``k+H_p`` inclusively:
```math
    \mathbf{U}_E = \begin{bmatrix} \mathbf{U}      \\ \mathbf{u}(k+H_p-1)   \end{bmatrix}  , \quad
    \mathbf{Ŷ}_E = \begin{bmatrix} \mathbf{ŷ}(k)   \\ \mathbf{Ŷ}            \end{bmatrix}  , \quad
    \mathbf{D̂}_E = \begin{bmatrix} \mathbf{d}(k)   \\ \mathbf{D̂}            \end{bmatrix}
```
since ``H_c ≤ H_p`` implies that ``\mathbf{Δu}(k+H_p) = \mathbf{0}`` or ``\mathbf{u}(k+H_p)=
\mathbf{u}(k+H_p-1)``. The vector ``\mathbf{D̂}`` includes the predicted measured disturbance
over ``H_p``. The argument ``\mathbf{p}`` is a custom parameter object of any type but use a
mutable one if you want to modify it later e.g.: a vector.

!!! tip
    Replace any of the 4 arguments with `_` if not needed (see `JE` default value below).

This method uses the default state estimator :

- if `model` is a [`LinModel`](@ref), a [`SteadyKalmanFilter`](@ref) with default arguments;
- else, an [`UnscentedKalmanFilter`](@ref) with default arguments. 

!!! warning
    See Extended Help if you get an error like:    
    `MethodError: no method matching Float64(::ForwardDiff.Dual)`.

# Arguments
- `model::SimModel` : model used for controller predictions and state estimations.
- `Hp=nothing`: prediction horizon ``H_p``, must be specified for [`NonLinModel`](@ref).
- `Hc=2` : control horizon ``H_c``.
- `Mwt=fill(1.0,model.ny)` : main diagonal of ``\mathbf{M}`` weight matrix (vector).
- `Nwt=fill(0.1,model.nu)` : main diagonal of ``\mathbf{N}`` weight matrix (vector).
- `Lwt=fill(0.0,model.nu)` : main diagonal of ``\mathbf{L}`` weight matrix (vector).
- `M_Hp=diagm(repeat(Mwt,Hp))` : positive semidefinite symmetric matrix ``\mathbf{M}_{H_p}``.
- `N_Hc=diagm(repeat(Nwt,Hc))` : positive semidefinite symmetric matrix ``\mathbf{N}_{H_c}``.
- `L_Hp=diagm(repeat(Lwt,Hp))` : positive semidefinite symmetric matrix ``\mathbf{L}_{H_p}``.
- `Cwt=1e5` : slack variable weight ``C`` (scalar), use `Cwt=Inf` for hard constraints only.
- `Ewt=0.0` : economic costs weight ``E`` (scalar). 
- `JE=(_,_,_,_)->0.0` : economic function ``J_E(\mathbf{U}_E, \mathbf{Ŷ}_E, \mathbf{D̂}_E, \mathbf{p})``.
- `p=model.p` : ``J_E`` function parameter ``\mathbf{p}`` (any type).
- `optim=JuMP.Model(Ipopt.Optimizer)` : nonlinear optimizer used in the predictive
   controller, provided as a [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.Model)
   (default to [`Ipopt`](https://github.com/jump-dev/Ipopt.jl) optimizer).
- additional keyword arguments are passed to [`UnscentedKalmanFilter`](@ref) constructor 
  (or [`SteadyKalmanFilter`](@ref), for [`LinModel`](@ref)).

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_,_)->0.5x+u, (x,_,_)->2x, 10.0, 1, 1, 1, solver=nothing);

julia> mpc = NonLinMPC(model, Hp=20, Hc=1, Cwt=1e6)
NonLinMPC controller with a sample time Ts = 10.0 s, Ipopt optimizer, UnscentedKalmanFilter estimator and:
 20 prediction steps Hp
  1 control steps Hc
  1 slack variable ϵ (control constraints)
  1 manipulated inputs u (0 integrating states)
  2 estimated states x̂
  1 measured outputs ym (1 integrating states)
  0 unmeasured outputs yu
  0 measured disturbances d
```

# Extended Help
!!! details "Extended Help"
    `NonLinMPC` controllers based on [`LinModel`](@ref) compute the predictions with matrix 
    algebra instead of a `for` loop. This feature can accelerate the optimization, especially
    for the constraint handling, and is not available in any other package, to my knowledge.

    The optimization relies on [`JuMP`](https://github.com/jump-dev/JuMP.jl) automatic 
    differentiation (AD) to compute the objective and constraint derivatives. Optimizers 
    generally benefit from exact derivatives like AD. However, the [`NonLinModel`](@ref) 
    state-space functions must be compatible with this feature. See [Automatic differentiation](https://jump.dev/JuMP.jl/stable/manual/nlp/#Automatic-differentiation)
    for common mistakes when writing these functions.

    Note that if `Cwt≠Inf`, the attribute `nlp_scaling_max_gradient` of `Ipopt` is set to 
    `10/Cwt` (if not already set), to scale the small values of ``ϵ``.
"""
function NonLinMPC(
    model::SimModel;
    Hp::Int = default_Hp(model),
    Hc::Int = DEFAULT_HC,
    Mwt  = fill(DEFAULT_MWT, model.ny),
    Nwt  = fill(DEFAULT_NWT, model.nu),
    Lwt  = fill(DEFAULT_LWT, model.nu),
    M_Hp = diagm(repeat(Mwt, Hp)),
    N_Hc = diagm(repeat(Nwt, Hc)),
    L_Hp = diagm(repeat(Lwt, Hp)),
    Cwt  = DEFAULT_CWT,
    Ewt  = DEFAULT_EWT,
    JE::Function = (_,_,_,_) -> 0.0,
    p = model.p,
    optim::JuMP.GenericModel = JuMP.Model(DEFAULT_NONLINMPC_OPTIMIZER, add_bridges=false),
    kwargs...
)
    estim = UnscentedKalmanFilter(model; kwargs...)
    NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Lwt, Cwt, Ewt, JE, p, M_Hp, N_Hc, L_Hp, optim)
end

function NonLinMPC(
    model::LinModel;
    Hp::Int = default_Hp(model),
    Hc::Int = DEFAULT_HC,
    Mwt  = fill(DEFAULT_MWT, model.ny),
    Nwt  = fill(DEFAULT_NWT, model.nu),
    Lwt  = fill(DEFAULT_LWT, model.nu),
    M_Hp = diagm(repeat(Mwt, Hp)),
    N_Hc = diagm(repeat(Nwt, Hc)),
    L_Hp = diagm(repeat(Lwt, Hp)),
    Cwt  = DEFAULT_CWT,
    Ewt  = DEFAULT_EWT,
    JE::Function = (_,_,_,_) -> 0.0,
    p = model.p,
    optim::JuMP.GenericModel = JuMP.Model(DEFAULT_NONLINMPC_OPTIMIZER, add_bridges=false),
    kwargs...
)
    estim = SteadyKalmanFilter(model; kwargs...)
    NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Lwt, Cwt, Ewt, JE, p, M_Hp, N_Hc, L_Hp, optim)
end


"""
    NonLinMPC(estim::StateEstimator; <keyword arguments>)

Use custom state estimator `estim` to construct `NonLinMPC`.

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_,_)->0.5x+u, (x,_,_)->2x, 10.0, 1, 1, 1, solver=nothing);

julia> estim = UnscentedKalmanFilter(model, σQint_ym=[0.05]);

julia> mpc = NonLinMPC(estim, Hp=20, Hc=1, Cwt=1e6)
NonLinMPC controller with a sample time Ts = 10.0 s, Ipopt optimizer, UnscentedKalmanFilter estimator and:
 20 prediction steps Hp
  1 control steps Hc
  1 slack variable ϵ (control constraints)
  1 manipulated inputs u (0 integrating states)
  2 estimated states x̂
  1 measured outputs ym (1 integrating states)
  0 unmeasured outputs yu
  0 measured disturbances d
```
"""
function NonLinMPC(
    estim::SE;
    Hp::Int = default_Hp(estim.model),
    Hc::Int = DEFAULT_HC,
    Mwt  = fill(DEFAULT_MWT, estim.model.ny),
    Nwt  = fill(DEFAULT_NWT, estim.model.nu),
    Lwt  = fill(DEFAULT_LWT, estim.model.nu),
    M_Hp = diagm(repeat(Mwt, Hp)),
    N_Hc = diagm(repeat(Nwt, Hc)),
    L_Hp = diagm(repeat(Lwt, Hp)),
    Cwt  = DEFAULT_CWT,
    Ewt  = DEFAULT_EWT,
    JE::JEFunc = (_,_,_,_) -> 0.0,
    p::P = estim.model.p,
    optim::JM = JuMP.Model(DEFAULT_NONLINMPC_OPTIMIZER, add_bridges=false),
) where {NT<:Real, SE<:StateEstimator{NT}, JM<:JuMP.GenericModel, JEFunc<:Function, P<:Any}
    nk = estimate_delays(estim.model)
    if Hp ≤ nk
        @warn("prediction horizon Hp ($Hp) ≤ estimated number of delays in model "*
              "($nk), the closed-loop system may be unstable or zero-gain (unresponsive)")
    end
    return NonLinMPC{NT, SE, JM, JEFunc, P}(
        estim, Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt, Ewt, JE, p, optim
    )
end

"""
    addinfo!(info, mpc::NonLinMPC) -> info

For [`NonLinMPC`](@ref), add `:sol` and the optimal economic cost `:JE`.
"""
function addinfo!(info, mpc::NonLinMPC)
    U, Ŷ, D̂, ŷ, d = info[:U], info[:Ŷ], info[:D̂], info[:ŷ], info[:d]
    UE = [U; U[(end - mpc.estim.model.nu + 1):end]]
    ŶE = [ŷ; Ŷ]
    D̂E = [d; D̂]
    info[:JE]  = mpc.JE(UE, ŶE, D̂E, mpc.p)
    info[:sol] = JuMP.solution_summary(mpc.optim, verbose=true)
    return info
end

"""
    init_optimization!(mpc::NonLinMPC, model::SimModel, optim)

Init the nonlinear optimization for [`NonLinMPC`](@ref) controllers.
"""
function init_optimization!(mpc::NonLinMPC, model::SimModel, optim)
    # --- variables and linear constraints ---
    con = mpc.con
    nΔŨ = length(mpc.ΔŨ)
    JuMP.num_variables(optim) == 0 || JuMP.empty!(optim)
    JuMP.set_silent(optim)
    limit_solve_time(mpc.optim, mpc.estim.model.Ts)
    @variable(optim, ΔŨvar[1:nΔŨ])
    A = con.A[con.i_b, :]
    b = con.b[con.i_b]
    @constraint(optim, linconstraint, A*ΔŨvar .≤ b)
    # --- nonlinear optimization init ---
    if mpc.nϵ == 1 && JuMP.solver_name(optim) == "Ipopt"
        C = mpc.Ñ_Hc[end]
        try
            JuMP.get_attribute(optim, "nlp_scaling_max_gradient")
        catch
            # default "nlp_scaling_max_gradient" to `10.0/C` if not already set:
            JuMP.set_attribute(optim, "nlp_scaling_max_gradient", 10.0/C)
        end
    end
    Jfunc, gfunc = get_optim_functions(mpc, mpc.optim)
    @operator(optim, J, nΔŨ, Jfunc)
    @objective(optim, Min, J(ΔŨvar...))
    ny, nx̂, Hp = model.ny, mpc.estim.nx̂, mpc.Hp
    if length(con.i_g) ≠ 0
        for i in eachindex(con.Y0min)
            name = Symbol("g_Y0min_$i")
            optim[name] = JuMP.add_nonlinear_operator(optim, nΔŨ, gfunc[i]; name)
        end
        i_end_Ymin = 1Hp*ny
        for i in eachindex(con.Y0max)
            name = Symbol("g_Y0max_$i")
            optim[name] = JuMP.add_nonlinear_operator(optim, nΔŨ, gfunc[i_end_Ymin+i]; name)
        end
        i_end_Ymax = 2Hp*ny
        for i in eachindex(con.x̂0min)
            name = Symbol("g_x̂0min_$i")
            optim[name] = JuMP.add_nonlinear_operator(optim, nΔŨ, gfunc[i_end_Ymax+i]; name)
        end
        i_end_x̂min = 2Hp*ny + nx̂
        for i in eachindex(con.x̂0max)
            name = Symbol("g_x̂0max_$i")
            optim[name] = JuMP.add_nonlinear_operator(optim, nΔŨ, gfunc[i_end_x̂min+i]; name)
        end
    end
    return nothing
end

"""
    get_optim_functions(mpc::NonLinMPC, ::JuMP.GenericModel) -> Jfunc, gfunc

Get the objective `Jfunc` and constraints `gfunc` functions for [`NonLinMPC`](@ref).

Inspired from: [User-defined operators with vector outputs](https://jump.dev/JuMP.jl/stable/tutorials/nonlinear/tips_and_tricks/#User-defined-operators-with-vector-outputs)
"""
function get_optim_functions(mpc::NonLinMPC, ::JuMP.GenericModel{JNT}) where JNT<:Real
    model = mpc.estim.model
    nu, ny, nx̂, Hp = model.nu, model.ny, mpc.estim.nx̂, mpc.Hp
    ng, nΔŨ, nU, nŶ = length(mpc.con.i_g), length(mpc.ΔŨ), Hp*nu, Hp*ny
    Nc = nΔŨ + 3
    ΔŨ_cache::DiffCache{Vector{JNT}, Vector{JNT}}     = DiffCache(zeros(JNT, nΔŨ), Nc)
    Ŷ0_cache::DiffCache{Vector{JNT}, Vector{JNT}}     = DiffCache(zeros(JNT, nŶ), Nc)
    U0_cache::DiffCache{Vector{JNT}, Vector{JNT}}     = DiffCache(zeros(JNT, nU), Nc)
    g_cache::DiffCache{Vector{JNT}, Vector{JNT}}      = DiffCache(zeros(JNT, ng), Nc)
    x̂0_cache::DiffCache{Vector{JNT}, Vector{JNT}}     = DiffCache(zeros(JNT, nx̂), Nc)
    x̂0next_cache::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, nx̂), Nc)
    u0_cache::DiffCache{Vector{JNT}, Vector{JNT}}     = DiffCache(zeros(JNT, nu), Nc)
    û0_cache::DiffCache{Vector{JNT}, Vector{JNT}}     = DiffCache(zeros(JNT, nu), Nc)
    Ȳ_cache::DiffCache{Vector{JNT}, Vector{JNT}}      = DiffCache(zeros(JNT, nŶ), Nc)
    Ū_cache::DiffCache{Vector{JNT}, Vector{JNT}}      = DiffCache(zeros(JNT, nU), Nc)
    function Jfunc(ΔŨtup::T...) where T<:Real
        ΔŨ1 = ΔŨtup[begin]
        ΔŨ, g = get_tmp(ΔŨ_cache, ΔŨ1), get_tmp(g_cache, ΔŨ1) 
        for i in eachindex(ΔŨtup)
            ΔŨ[i] = ΔŨtup[i] # ΔŨ .= ΔŨtup seems to produce a type instability
        end 
        Ŷ0 = get_tmp(Ŷ0_cache, ΔŨ1)
        x̂0, x̂0next = get_tmp(x̂0_cache, ΔŨ1), get_tmp(x̂0next_cache, ΔŨ1)
        u0, û0 = get_tmp(u0_cache, ΔŨ1), get_tmp(û0_cache, ΔŨ1)
        Ŷ0, x̂0end = predict!(Ŷ0, x̂0, x̂0next, u0, û0, mpc, model, ΔŨ)
        g = get_tmp(g_cache, ΔŨ1)
        g = con_nonlinprog!(g, mpc, model, x̂0end, Ŷ0, ΔŨ)
        U0, Ȳ, Ū = get_tmp(U0_cache, ΔŨ1), get_tmp(Ȳ_cache, ΔŨ1), get_tmp(Ū_cache, ΔŨ1)
        return obj_nonlinprog!(U0, Ȳ, Ū, mpc, model, Ŷ0, ΔŨ)::T
    end
    function gfunc_i(i, ΔŨtup::NTuple{N, T}) where {N, T<:Real}
        ΔŨ1 = ΔŨtup[begin]
        ΔŨ, g = get_tmp(ΔŨ_cache, ΔŨ1), get_tmp(g_cache, ΔŨ1)     
        if any(new !== old for (new, old) in zip(ΔŨtup, ΔŨ)) # new ΔŨtup, update predictions:
            for i in eachindex(ΔŨtup)
                ΔŨ[i] = ΔŨtup[i] # ΔŨ .= ΔŨtup seems to produce a type instability
            end
            Ŷ0 = get_tmp(Ŷ0_cache, ΔŨ1)
            x̂0, x̂0next = get_tmp(x̂0_cache, ΔŨ1), get_tmp(x̂0next_cache, ΔŨ1)
            u0, û0 = get_tmp(u0_cache, ΔŨ1), get_tmp(û0_cache, ΔŨ1)
            Ŷ0, x̂0end = predict!(Ŷ0, x̂0, x̂0next, u0, û0, mpc, model, ΔŨ)
            g = con_nonlinprog!(g, mpc, model, x̂0end, Ŷ0, ΔŨ)
        end
        return g[i]::T
    end
    gfunc = [(ΔŨ...) -> gfunc_i(i, ΔŨ) for i in 1:ng]
    return Jfunc, gfunc
end

"Set the nonlinear constraints on the output predictions `Ŷ` and terminal states `x̂end`."
function setnonlincon!(
    mpc::NonLinMPC, ::NonLinModel, optim::JuMP.GenericModel{JNT}
) where JNT<:Real
    ΔŨvar = optim[:ΔŨvar]
    con = mpc.con
    nonlin_constraints = JuMP.all_constraints(optim, JuMP.NonlinearExpr, MOI.LessThan{JNT})
    map(con_ref -> JuMP.delete(optim, con_ref), nonlin_constraints)
    for i in findall(.!isinf.(con.Y0min))
        gfunc_i = optim[Symbol("g_Y0min_$(i)")]
        @constraint(optim, gfunc_i(ΔŨvar...) <= 0)
    end
    for i in findall(.!isinf.(con.Y0max))
        gfunc_i = optim[Symbol("g_Y0max_$(i)")]
        @constraint(optim, gfunc_i(ΔŨvar...) <= 0)
    end
    for i in findall(.!isinf.(con.x̂0min))
        gfunc_i = optim[Symbol("g_x̂0min_$(i)")]
        @constraint(optim, gfunc_i(ΔŨvar...) <= 0)
    end
    for i in findall(.!isinf.(con.x̂0max))
        gfunc_i = optim[Symbol("g_x̂0max_$(i)")]
        @constraint(optim, gfunc_i(ΔŨvar...) <= 0)
    end
    return nothing
end

"""
    con_nonlinprog!(g, mpc::NonLinMPC, model::SimModel, x̂end, Ŷ, ΔŨ) -> g

Nonlinear constrains for [`NonLinMPC`](@ref) when `model` is not a [`LinModel`](@ref).

The method mutates the `g` vector in argument and returns it.
"""
function con_nonlinprog!(g, mpc::NonLinMPC, ::SimModel, x̂0end, Ŷ0, ΔŨ)
    nx̂, nŶ = mpc.estim.nx̂, length(Ŷ0)
    ϵ = mpc.nϵ == 1 ? ΔŨ[end] : 0 # ϵ = 0 if Cwt=Inf (meaning: no relaxation)
    for i in eachindex(g)
        mpc.con.i_g[i] || continue
        if i ≤ nŶ
            j = i
            g[i] = (mpc.con.Y0min[j] - Ŷ0[j])     - ϵ*mpc.con.C_ymin[j]
        elseif i ≤ 2nŶ
            j = i - nŶ
            g[i] = (Ŷ0[j] - mpc.con.Y0max[j])     - ϵ*mpc.con.C_ymax[j]
        elseif i ≤ 2nŶ + nx̂
            j = i - 2nŶ
            g[i] = (mpc.con.x̂0min[j] - x̂0end[j])  - ϵ*mpc.con.c_x̂min[j]
        else
            j = i - 2nŶ - nx̂
            g[i] = (x̂0end[j] - mpc.con.x̂0max[j])  - ϵ*mpc.con.c_x̂max[j]
        end
    end
    return g
end

"No nonlinear constraints if `model` is a [`LinModel`](@ref), return `g` unchanged."
con_nonlinprog!(g, ::NonLinMPC, ::LinModel, _ , _ , _ ) = g

"Evaluate the economic term of the objective function for [`NonLinMPC`](@ref)."
function obj_econ!(U0, Ȳ, mpc::NonLinMPC, model::SimModel, Ŷ0, ΔŨ)
    if !iszero(mpc.E)
        ny, Hp, ŷ, D̂E = model.ny, mpc.Hp, mpc.ŷ, mpc.D̂E
        U   = U0
        U .+= mpc.Uop
        uend = @views U[(end-model.nu+1):end]
        Ŷ  = Ȳ
        Ŷ .= Ŷ0 .+ mpc.Yop
        UE = [U; uend]
        ŶE = [ŷ; Ŷ]
        E_JE = mpc.E*mpc.JE(UE, ŶE, D̂E, mpc.p)
    else
        E_JE = 0.0
    end
    return E_JE
end