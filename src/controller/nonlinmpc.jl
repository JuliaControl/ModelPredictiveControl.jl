const DEFAULT_NONLINMPC_OPTIMIZER = optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes")

struct NonLinMPC{
    NT<:Real, 
    SE<:StateEstimator, 
    JM<:JuMP.GenericModel, 
    JEfunc<:Function
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
    M_Hp::Diagonal{NT, Vector{NT}}
    Ñ_Hc::Diagonal{NT, Vector{NT}}
    L_Hp::Diagonal{NT, Vector{NT}}
    C::NT
    E::NT
    JE::JEfunc
    R̂u::Vector{NT}
    R̂y::Vector{NT}
    noR̂u::Bool
    S̃::Matrix{NT}
    T::Matrix{NT}
    Ẽ::Matrix{NT}
    F::Vector{NT}
    G::Matrix{NT}
    J::Matrix{NT}
    K::Matrix{NT}
    V::Matrix{NT}
    H̃::Hermitian{NT, Matrix{NT}}
    q̃::Vector{NT}
    p::Vector{NT}
    Ks::Matrix{NT}
    Ps::Matrix{NT}
    d0::Vector{NT}
    D̂0::Vector{NT}
    D̂E::Vector{NT}
    Ŷop::Vector{NT}
    Dop::Vector{NT}
    function NonLinMPC{NT, SE, JM, JEFunc}(
        estim::SE, Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt, Ewt, JE::JEFunc, optim::JM
    ) where {NT<:Real, SE<:StateEstimator, JM<:JuMP.GenericModel, JEFunc<:Function}
        model = estim.model
        nu, ny, nd = model.nu, model.ny, model.nd
        ŷ = copy(model.yop) # dummy vals (updated just before optimization)
        validate_weights(model, Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt, Ewt)
        M_Hp, N_Hc, L_Hp = float(M_Hp), float(N_Hc), float(L_Hp) # debug julia 1.6
        # dummy vals (updated just before optimization):
        R̂y, R̂u = zeros(NT, ny*Hp), zeros(NT, nu*Hp)
        noR̂u = iszero(L_Hp)
        S, T = init_ΔUtoU(model, Hp, Hc)
        E, F, G, J, K, V, ex̂, fx̂, gx̂, jx̂, kx̂, vx̂ = init_predmat(estim, model, Hp, Hc)
        con, S̃, Ñ_Hc, Ẽ = init_defaultcon_mpc(estim, Hp, Hc, Cwt, S, N_Hc, E, ex̂, fx̂, gx̂, jx̂, kx̂, vx̂)
        H̃, q̃, p = init_quadprog(model, Ẽ, S̃, M_Hp, Ñ_Hc, L_Hp)
        Ks, Ps = init_stochpred(estim, Hp)
        # dummy vals (updated just before optimization):
        d0, D̂0, D̂E = zeros(NT, nd), zeros(NT, nd*Hp), zeros(NT, nd + nd*Hp)
        Ŷop, Dop = repeat(model.yop, Hp), repeat(model.dop, Hp)
        nvar = size(Ẽ, 2)
        ΔŨ = zeros(NT, nvar)
        mpc = new{NT, SE, JM, JEFunc}(
            estim, optim, con,
            ΔŨ, ŷ,
            Hp, Hc, 
            M_Hp, Ñ_Hc, L_Hp, Cwt, Ewt, JE, 
            R̂u, R̂y, noR̂u,
            S̃, T,  
            Ẽ, F, G, J, K, V, H̃, q̃, p,
            Ks, Ps,
            d0, D̂0, D̂E,
            Ŷop, Dop,
        )
        init_optimization!(mpc, optim)
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
                       + E J_E(\mathbf{U}_E, \mathbf{Ŷ}_E, \mathbf{D̂}_E)
\end{aligned}
```
See [`LinMPC`](@ref) for the variable definitions. The custom economic function ``J_E`` can
penalizes solutions with high economic costs. Setting all the weights to 0 except ``E`` 
creates a pure economic model predictive controller (EMPC). The arguments of ``J_E`` are 
the manipulated inputs, the predicted outputs and measured disturbances from ``k`` to 
``k+H_p`` inclusively:
```math
    \mathbf{U}_E = \begin{bmatrix} \mathbf{U}      \\ \mathbf{u}(k+H_p-1)   \end{bmatrix}  , \quad
    \mathbf{Ŷ}_E = \begin{bmatrix} \mathbf{ŷ}(k)   \\ \mathbf{Ŷ}            \end{bmatrix}  , \quad
    \mathbf{D̂}_E = \begin{bmatrix} \mathbf{d}(k)   \\ \mathbf{D̂}            \end{bmatrix}
```
since ``H_c ≤ H_p`` implies that ``\mathbf{Δu}(k+H_p) = \mathbf{0}`` or ``\mathbf{u}(k+H_p)=
\mathbf{u}(k+H_p-1)``. The vector ``\mathbf{D̂}`` includes the predicted measured disturbance
over ``H_p``.

!!! tip
    Replace any of the 3 arguments with `_` if not needed (see `JE` default value below).

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
- `Cwt=1e5` : slack variable weight ``C`` (scalar), use `Cwt=Inf` for hard constraints only.
- `Ewt=0.0` : economic costs weight ``E`` (scalar). 
- `JE=(_,_,_)->0.0` : economic function ``J_E(\mathbf{U}_E, \mathbf{Ŷ}_E, \mathbf{D̂}_E)``.
- `M_Hp` / `N_Hc` / `L_Hp` : diagonal matrices ``\mathbf{M}_{H_p}, \mathbf{N}_{H_c},
  \mathbf{L}_{H_p}``, for time-varying weights (generated from `Mwt/Nwt/Lwt` args if omitted).
- `optim=JuMP.Model(Ipopt.Optimizer)` : nonlinear optimizer used in the predictive
   controller, provided as a [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.Model)
   (default to [`Ipopt`](https://github.com/jump-dev/Ipopt.jl) optimizer).
- additional keyword arguments are passed to [`UnscentedKalmanFilter`](@ref) constructor 
  (or [`SteadyKalmanFilter`](@ref), for [`LinModel`](@ref)).

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_)->0.5x+u, (x,_)->2x, 10.0, 1, 1, 1);

julia> mpc = NonLinMPC(model, Hp=20, Hc=1, Cwt=1e6)
NonLinMPC controller with a sample time Ts = 10.0 s, Ipopt optimizer, UnscentedKalmanFilter estimator and:
 20 prediction steps Hp
  1 control steps Hc
  1 manipulated inputs u (0 integrating states)
  2 states x̂
  1 measured outputs ym (1 integrating states)
  0 unmeasured outputs yu
  0 measured disturbances d
```

# Extended Help
`NonLinMPC` controllers based on [`LinModel`](@ref) compute the predictions with matrix 
algebra instead of a `for` loop. This feature can accelerate the optimization, especially
for the constraint handling, and is not available in any other package, to my knowledge.

The optimizations rely on [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl) automatic 
differentiation (AD) to compute the objective and constraint derivatives. Optimizers 
generally benefit from exact derivatives like AD. However, the [`NonLinModel`](@ref) `f` 
and `h` functions must be compatible with this feature. See [Automatic differentiation](https://jump.dev/JuMP.jl/stable/manual/nlp/#Automatic-differentiation)
for common mistakes when writing these functions.
"""
function NonLinMPC(
    model::SimModel;
    Hp::Union{Int, Nothing} = nothing,
    Hc::Int = DEFAULT_HC,
    Mwt = fill(DEFAULT_MWT, model.ny),
    Nwt = fill(DEFAULT_NWT, model.nu),
    Lwt = fill(DEFAULT_LWT, model.nu),
    Cwt = DEFAULT_CWT,
    Ewt = DEFAULT_EWT,
    JE::Function = (_,_,_) -> 0.0,
    M_Hp = nothing,
    N_Hc = nothing,
    L_Hp = nothing,
    optim::JuMP.GenericModel = JuMP.Model(DEFAULT_NONLINMPC_OPTIMIZER, add_bridges=false),
    kwargs...
)
    estim = UnscentedKalmanFilter(model; kwargs...)
    NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Lwt, Cwt, Ewt, JE, M_Hp, N_Hc, L_Hp, optim)
end

function NonLinMPC(
    model::LinModel;
    Hp::Union{Int, Nothing} = nothing,
    Hc::Int = DEFAULT_HC,
    Mwt = fill(DEFAULT_MWT, model.ny),
    Nwt = fill(DEFAULT_NWT, model.nu),
    Lwt = fill(DEFAULT_LWT, model.nu),
    Cwt = DEFAULT_CWT,
    Ewt = DEFAULT_EWT,
    JE::Function = (_,_,_) -> 0.0,
    M_Hp = nothing,
    N_Hc = nothing,
    L_Hp = nothing,
    optim::JuMP.GenericModel = JuMP.Model(DEFAULT_NONLINMPC_OPTIMIZER, add_bridges=false),
    kwargs...
)
    estim = SteadyKalmanFilter(model; kwargs...)
    NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Lwt, Cwt, Ewt, JE, M_Hp, N_Hc, L_Hp, optim)
end


"""
    NonLinMPC(estim::StateEstimator; <keyword arguments>)

Use custom state estimator `estim` to construct `NonLinMPC`.

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_)->0.5x+u, (x,_)->2x, 10.0, 1, 1, 1);

julia> estim = UnscentedKalmanFilter(model, σQint_ym=[0.05]);

julia> mpc = NonLinMPC(estim, Hp=20, Hc=1, Cwt=1e6)
NonLinMPC controller with a sample time Ts = 10.0 s, Ipopt optimizer, UnscentedKalmanFilter estimator and:
 20 prediction steps Hp
  1 control steps Hc
  1 manipulated inputs u (0 integrating states)
  2 states x̂
  1 measured outputs ym (1 integrating states)
  0 unmeasured outputs yu
  0 measured disturbances d
```
"""
function NonLinMPC(
    estim::SE;
    Hp::Union{Int, Nothing} = nothing,
    Hc::Int = DEFAULT_HC,
    Mwt = fill(DEFAULT_MWT, estim.model.ny),
    Nwt = fill(DEFAULT_NWT, estim.model.nu),
    Lwt = fill(DEFAULT_LWT, estim.model.nu),
    Cwt = DEFAULT_CWT,
    Ewt = DEFAULT_EWT,
    JE::JEFunc = (_,_,_) -> 0.0,
    M_Hp = nothing,
    N_Hc = nothing,
    L_Hp = nothing,
    optim::JM = JuMP.Model(DEFAULT_NONLINMPC_OPTIMIZER, add_bridges=false),
) where {NT<:Real, SE<:StateEstimator{NT}, JM<:JuMP.GenericModel, JEFunc<:Function}
    Hp = default_Hp(estim.model, Hp)
    isnothing(M_Hp) && (M_Hp = Diagonal{NT}(repeat(Mwt, Hp)))
    isnothing(N_Hc) && (N_Hc = Diagonal{NT}(repeat(Nwt, Hc)))
    isnothing(L_Hp) && (L_Hp = Diagonal{NT}(repeat(Lwt, Hp)))
    return NonLinMPC{NT, SE, JM, JEFunc}(estim, Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt, Ewt, JE, optim)
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
    info[:JE]  = mpc.JE(UE, ŶE, D̂E)
    info[:sol] = solution_summary(mpc.optim, verbose=true)
    return info
end

"""
    init_optimization!(mpc::NonLinMPC, optim::JuMP.GenericModel)

Init the nonlinear optimization for [`NonLinMPC`](@ref) controllers.
"""
function init_optimization!(mpc::NonLinMPC, optim::JuMP.GenericModel{JNT}) where JNT<:Real
    # --- variables and linear constraints ---
    con = mpc.con
    nvar = length(mpc.ΔŨ)
    set_silent(optim)
    limit_solve_time(mpc)
    @variable(optim, ΔŨvar[1:nvar])
    A = con.A[con.i_b, :]
    b = con.b[con.i_b]
    @constraint(optim, linconstraint, A*ΔŨvar .≤ b)
    # --- nonlinear optimization init ---
    model = mpc.estim.model
    ny, nx̂, Hp, ng = model.ny, mpc.estim.nx̂, mpc.Hp, length(con.i_g)
    # inspired from https://jump.dev/JuMP.jl/stable/tutorials/nonlinear/tips_and_tricks/#User-defined-operators-with-vector-outputs
    Jfunc, gfunc = let mpc=mpc, model=model, ng=ng, nvar=nvar , nŶ=Hp*ny, nx̂=nx̂
        last_ΔŨtup_float, last_ΔŨtup_dual = nothing, nothing
        Ŷ_cache::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, nŶ), nvar + 3)
        g_cache::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, ng), nvar + 3)
        x̂_cache::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, nx̂), nvar + 3)
        function Jfunc(ΔŨtup::JNT...)
            Ŷ = get_tmp(Ŷ_cache, ΔŨtup[1])
            ΔŨ = collect(ΔŨtup)
            if ΔŨtup !== last_ΔŨtup_float
                x̂ = get_tmp(x̂_cache, ΔŨtup[1])
                g = get_tmp(g_cache, ΔŨtup[1])
                Ŷ, x̂end = predict!(Ŷ, x̂, mpc, model, ΔŨ)
                g = con_nonlinprog!(g, mpc, model, x̂end, Ŷ, ΔŨ)
                last_ΔŨtup_float = ΔŨtup
            end
            return obj_nonlinprog(mpc, model, Ŷ, ΔŨ)
        end
        function Jfunc(ΔŨtup::ForwardDiff.Dual...)
            Ŷ = get_tmp(Ŷ_cache, ΔŨtup[1])
            ΔŨ = collect(ΔŨtup)
            if ΔŨtup !== last_ΔŨtup_dual
                x̂ = get_tmp(x̂_cache, ΔŨtup[1])
                g = get_tmp(g_cache, ΔŨtup[1])
                Ŷ, x̂end = predict!(Ŷ, x̂, mpc, model, ΔŨ)
                g = con_nonlinprog!(g, mpc, model, x̂end, Ŷ, ΔŨ)
                last_ΔŨtup_dual = ΔŨtup
            end
            return obj_nonlinprog(mpc, model, Ŷ, ΔŨ)
        end
        function gfunc_i(i, ΔŨtup::NTuple{N, JNT}) where N
            g = get_tmp(g_cache, ΔŨtup[1])
            if ΔŨtup !== last_ΔŨtup_float
                x̂ = get_tmp(x̂_cache, ΔŨtup[1])
                Ŷ = get_tmp(Ŷ_cache, ΔŨtup[1])
                ΔŨ = collect(ΔŨtup)
                Ŷ, x̂end = predict!(Ŷ, x̂, mpc, model, ΔŨ)
                g = con_nonlinprog!(g, mpc, model, x̂end, Ŷ, ΔŨ)
                last_ΔŨtup_float = ΔŨtup
            end
            return g[i]
        end 
        function gfunc_i(i, ΔŨtup::NTuple{N, ForwardDiff.Dual}) where N
            g = get_tmp(g_cache, ΔŨtup[1])
            if ΔŨtup !== last_ΔŨtup_dual
                x̂ = get_tmp(x̂_cache, ΔŨtup[1])
                Ŷ = get_tmp(Ŷ_cache, ΔŨtup[1])
                ΔŨ = collect(ΔŨtup)
                Ŷ, x̂end = predict!(Ŷ, x̂, mpc, model, ΔŨ)
                g = con_nonlinprog!(g, mpc, model, x̂end, Ŷ, ΔŨ)
                last_ΔŨtup_dual = ΔŨtup
            end
            return g[i]
        end
        gfunc = [(ΔŨ...) -> gfunc_i(i, ΔŨ) for i in 1:ng]
        (Jfunc, gfunc)
    end
    register(optim, :Jfunc, nvar, Jfunc, autodiff=true)
    @NLobjective(optim, Min, Jfunc(ΔŨvar...))
    if ng ≠ 0
        for i in eachindex(con.Ymin)
            sym = Symbol("g_Ymin_$i")
            register(optim, sym, nvar, gfunc[i], autodiff=true)
        end
        i_end_Ymin = 1Hp*ny
        for i in eachindex(con.Ymax)
            sym = Symbol("g_Ymax_$i")
            register(optim, sym, nvar, gfunc[i_end_Ymin+i], autodiff=true)
        end
        i_end_Ymax = 2Hp*ny
        for i in eachindex(con.x̂min)
            sym = Symbol("g_x̂min_$i")
            register(optim, sym, nvar, gfunc[i_end_Ymax+i], autodiff=true)
        end
        i_end_x̂min = 2Hp*ny + nx̂
        for i in eachindex(con.x̂max)
            sym = Symbol("g_x̂max_$i")
            register(optim, sym, nvar, gfunc[i_end_x̂min+i], autodiff=true)
        end
    end
    return nothing
end

"Set the nonlinear constraints on the output predictions `Ŷ` and terminal states `x̂end`."
function setnonlincon!(mpc::NonLinMPC, ::NonLinModel)
    optim = mpc.optim
    ΔŨvar = optim[:ΔŨvar]
    con = mpc.con
    map(con -> delete(optim, con), all_nonlinear_constraints(optim))
    for i in findall(.!isinf.(con.Ymin))
        f_sym = Symbol("g_Ymin_$(i)")
        add_nonlinear_constraint(optim, :($(f_sym)($(ΔŨvar...)) <= 0))
    end
    for i in findall(.!isinf.(con.Ymax))
        f_sym = Symbol("g_Ymax_$(i)")
        add_nonlinear_constraint(optim, :($(f_sym)($(ΔŨvar...)) <= 0))
    end
    for i in findall(.!isinf.(con.x̂min))
        f_sym = Symbol("g_x̂min_$(i)")
        add_nonlinear_constraint(optim, :($(f_sym)($(ΔŨvar...)) <= 0))
    end
    for i in findall(.!isinf.(con.x̂max))
        f_sym = Symbol("g_x̂max_$(i)")
        add_nonlinear_constraint(optim, :($(f_sym)($(ΔŨvar...)) <= 0))
    end
    return nothing
end

"""
    con_nonlinprog!(g, mpc::NonLinMPC, model::SimModel, x̂end, Ŷ, ΔŨ) -> g

Nonlinear constrains for [`NonLinMPC`](@ref) when `model` is not a [`LinModel`](@ref).

The method mutates the `g` vector in argument and returns it.
"""
function con_nonlinprog!(g, mpc::NonLinMPC, ::SimModel, x̂end, Ŷ, ΔŨ)
    nx̂, nŶ = mpc.estim.nx̂, length(Ŷ)
    ϵ = !isinf(mpc.C) ? ΔŨ[end] : 0.0 # ϵ = 0.0 if Cwt=Inf (meaning: no relaxation)
    for i in eachindex(g)
        mpc.con.i_g[i] || continue
        if i ≤ nŶ
            j = i
            g[i] = (mpc.con.Ymin[j] - Ŷ[j])     - ϵ*mpc.con.C_ymin[j]
        elseif i ≤ 2nŶ
            j = i - nŶ
            g[i] = (Ŷ[j] - mpc.con.Ymax[j])     - ϵ*mpc.con.C_ymax[j]
        elseif i ≤ 2nŶ + nx̂
            j = i - 2nŶ
            g[i] = (mpc.con.x̂min[j] - x̂end[j])  - ϵ*mpc.con.c_x̂min[j]
        else
            j = i - 2nŶ - nx̂
            g[i] = (x̂end[j] - mpc.con.x̂max[j])  - ϵ*mpc.con.c_x̂max[j]
        end
    end
    return g
end

"No nonlinear constraints if `model` is a [`LinModel`](@ref), return `g` unchanged."
con_nonlinprog!(g, ::NonLinMPC, ::LinModel, _ , _ , _ ) = g