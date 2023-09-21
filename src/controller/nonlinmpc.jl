const DiffCacheType = DiffCache{Vector{Float64}, Vector{Float64}}

struct NonLinMPC{S<:StateEstimator, JEfunc<:Function} <: PredictiveController
    estim::S
    optim::JuMP.Model
    con::ControllerConstraint
    ΔŨ::Vector{Float64}
    ŷ ::Vector{Float64}
    Hp::Int
    Hc::Int
    M_Hp::Diagonal{Float64, Vector{Float64}}
    Ñ_Hc::Diagonal{Float64, Vector{Float64}}
    L_Hp::Diagonal{Float64, Vector{Float64}}
    C::Float64
    E::Float64
    JE::JEfunc
    R̂u::Vector{Float64}
    R̂y::Vector{Float64}
    noR̂u::Bool
    S̃_Hp::Matrix{Bool}
    T_Hp::Matrix{Bool}
    T_Hc::Matrix{Bool}
    Ẽ::Matrix{Float64}
    F::Vector{Float64}
    G::Matrix{Float64}
    J::Matrix{Float64}
    K::Matrix{Float64}
    Q::Matrix{Float64}
    P̃::Hermitian{Float64, Matrix{Float64}}
    q̃::Vector{Float64}
    p::Vector{Float64}
    Ks::Matrix{Float64}
    Ps::Matrix{Float64}
    d::Vector{Float64}
    D̂::Vector{Float64}
    Ŷop::Vector{Float64}
    Dop::Vector{Float64}
    function NonLinMPC{S, JEFunc}(
        estim::S, Hp, Hc, Mwt, Nwt, Lwt, Cwt, Ewt, JE::JEFunc, optim
    ) where {S<:StateEstimator, JEFunc<:Function}
        model = estim.model
        nu, ny, nd = model.nu, model.ny, model.nd
        ŷ = zeros(ny)
        validate_weights(model, Hp, Hc, Mwt, Nwt, Lwt, Cwt, Ewt)
        M_Hp = Diagonal(convert(Vector{Float64}, repeat(Mwt, Hp)))
        N_Hc = Diagonal(convert(Vector{Float64}, repeat(Nwt, Hc)))
        L_Hp = Diagonal(convert(Vector{Float64}, repeat(Lwt, Hp)))
        C = Cwt
        R̂y, R̂u = zeros(ny*Hp), zeros(nu*Hp) # dummy vals (updated just before optimization)
        noR̂u = iszero(L_Hp)
        S_Hp, T_Hp, S_Hc, T_Hc = init_ΔUtoU(nu, Hp, Hc)
        E, F, G, J, K, Q = init_predmat(estim, model, Hp, Hc)
        con, S̃_Hp, Ñ_Hc, Ẽ = init_defaultcon(model, Hp, Hc, C, S_Hp, S_Hc, N_Hc, E)
        P̃, q̃, p = init_quadprog(model, Ẽ, S̃_Hp, M_Hp, Ñ_Hc, L_Hp)
        Ks, Ps = init_stochpred(estim, Hp)
        d, D̂ = zeros(nd), zeros(nd*Hp)
        Ŷop, Dop = repeat(model.yop, Hp), repeat(model.dop, Hp)
        nvar = size(Ẽ, 2)
        ΔŨ = zeros(nvar)
        mpc = new(
            estim, optim, con,
            ΔŨ, ŷ,
            Hp, Hc, 
            M_Hp, Ñ_Hc, L_Hp, Cwt, Ewt, JE, R̂u, R̂y, noR̂u,
            S̃_Hp, T_Hp, T_Hc, 
            Ẽ, F, G, J, K, Q, P̃, q̃, p,
            Ks, Ps,
            d, D̂,
            Ŷop, Dop,
        )
        init_optimization!(mpc)
        return mpc
    end
end

@doc raw"""
    NonLinMPC(model::SimModel; <keyword arguments>)

Construct a nonlinear predictive controller based on [`SimModel`](@ref) `model`.

Both [`NonLinModel`](@ref) and [`LinModel`](@ref) are supported (see Extended Help). The 
controller minimizes the following objective function at each discrete time ``k``:
```math
\min_{\mathbf{ΔU}, ϵ}    \mathbf{(R̂_y - Ŷ)}' \mathbf{M}_{H_p} \mathbf{(R̂_y - Ŷ)}   
                       + \mathbf{(ΔU)}'      \mathbf{N}_{H_c} \mathbf{(ΔU)}  
                       + \mathbf{(R̂_u - U)}' \mathbf{L}_{H_p} \mathbf{(R̂_u - U)} 
                       + C ϵ^2  +  E J_E(\mathbf{U}_E, \mathbf{Ŷ}_E, \mathbf{D̂}_E)
```
See [`LinMPC`](@ref) for the variable definitions. The custom economic function ``J_E`` can
penalizes solutions with high economic costs. Setting all the weights to 0 except ``E`` 
creates a pure economic model predictive controller (EMPC). The arguments of ``J_E`` are 
the manipulated inputs, the predicted outputs and measured disturbances from ``k`` to 
``k+H_p`` inclusively:
```math
    \mathbf{U}_E = \begin{bmatrix} \mathbf{U}      \\ \mathbf{u}(k+H_p-1)   \end{bmatrix}  \text{,} \qquad
    \mathbf{Ŷ}_E = \begin{bmatrix} \mathbf{ŷ}(k)   \\ \mathbf{Ŷ}            \end{bmatrix}  \text{,} \qquad
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
- `Hp=10`: prediction horizon ``H_p``.
- `Hc=2` : control horizon ``H_c``.
- `Mwt=fill(1.0,model.ny)` : main diagonal of ``\mathbf{M}`` weight matrix (vector).
- `Nwt=fill(0.1,model.nu)` : main diagonal of ``\mathbf{N}`` weight matrix (vector).
- `Lwt=fill(0.0,model.nu)` : main diagonal of ``\mathbf{L}`` weight matrix (vector).
- `Cwt=1e5` : slack variable weight ``C`` (scalar), use `Cwt=Inf` for hard constraints only.
- `Ewt=0.0` : economic costs weight ``E`` (scalar). 
- `JE=(_,_,_)->0.0` : economic function ``J_E(\mathbf{U}_E, \mathbf{Ŷ}_E, \mathbf{D̂}_E)``.
- `optim=JuMP.Model(Ipopt.Optimizer)` : nonlinear optimizer used in the predictive
   controller, provided as a [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.Model)
   (default to [`Ipopt.jl`](https://github.com/jump-dev/Ipopt.jl) optimizer).
- additionnal keyword arguments are passed to [`UnscentedKalmanFilter`](@ref) constructor 
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
algebra instead of a `for` loop. This feature can accelerate the optimization and is not 
available in any other package, to my knowledge.

The optimizations rely on [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl) automatic 
differentiation (AD) to compute the objective and constraint derivatives. Optimizers 
generally benefit from exact derivatives like AD. However, the [`NonLinModel`](@ref) `f` 
and `h` functions must be compatible with this feature. See [Automatic differentiation](https://jump.dev/JuMP.jl/stable/manual/nlp/#Automatic-differentiation)
for common mistakes when writing these functions.
"""
function NonLinMPC(
    model::SimModel;
    Hp::Int = DEFAULT_HP,
    Hc::Int = DEFAULT_HC,
    Mwt = fill(DEFAULT_MWT, model.ny),
    Nwt = fill(DEFAULT_NWT, model.nu),
    Lwt = fill(DEFAULT_LWT, model.nu),
    Cwt = DEFAULT_CWT,
    Ewt = DEFAULT_EWT,
    JE::Function = (_,_,_) -> 0.0,
    optim::JuMP.Model = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes")),
    kwargs...
)
    estim = UnscentedKalmanFilter(model; kwargs...)
    NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Lwt, Cwt, Ewt, JE, optim)
end
function NonLinMPC(
    model::LinModel;
    Hp::Int = DEFAULT_HP,
    Hc::Int = DEFAULT_HC,
    Mwt = fill(DEFAULT_MWT, model.ny),
    Nwt = fill(DEFAULT_NWT, model.nu),
    Lwt = fill(DEFAULT_LWT, model.nu),
    Cwt = DEFAULT_CWT,
    Ewt = DEFAULT_EWT,
    JE::Function = (_,_,_) -> 0.0,
    optim::JuMP.Model = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes")),
    kwargs...
)
    estim = SteadyKalmanFilter(model; kwargs...)
    NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Lwt, Cwt, Ewt, JE, optim)
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
    estim::S;
    Hp::Int = DEFAULT_HP,
    Hc::Int = DEFAULT_HC,
    Mwt = fill(DEFAULT_MWT, estim.model.ny),
    Nwt = fill(DEFAULT_NWT, estim.model.nu),
    Lwt = fill(DEFAULT_LWT, estim.model.nu),
    Cwt = DEFAULT_CWT,
    Ewt = DEFAULT_EWT,
    JE::JEFunc = (_,_,_) -> 0.0,
    optim::JuMP.Model = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"))
) where {S<:StateEstimator, JEFunc<:Function}
    return NonLinMPC{S, JEFunc}(estim, Hp, Hc, Mwt, Nwt, Lwt, Cwt, Ewt, JE, optim)
end

"""
    getinfo(mpc::NonLinMPC) -> sol_summary, info

Invoke [`getinfo(::PredictiveController)`](@ref) and add `:JE` the economic optimum ``J_E``.
"""
function getinfo(mpc::NonLinMPC)
    sol_summary, info = invoke(getinfo, Tuple{PredictiveController}, mpc)
    U, Ŷ, D̂ = info[:U], info[:Ŷ], info[:D̂]
    UE = [U; U[(end - mpc.estim.model.nu + 1):end]]
    ŶE = [mpc.ŷ; Ŷ]
    D̂E = [mpc.d; D̂]
    info[:JE] = mpc.JE(UE, ŶE, D̂E)
    return sol_summary, info
end

"""
    init_optimization!(mpc::NonLinMPC)

Init the nonlinear optimization for [`NonLinMPC`](@ref) controllers.
"""
function init_optimization!(mpc::NonLinMPC)
    # --- variables and linear constraints ---
    optim, con = mpc.optim, mpc.con
    nvar = length(mpc.ΔŨ)
    set_silent(optim)
    @variable(optim, ΔŨvar[1:nvar])
    ΔŨvar = optim[:ΔŨvar]
    A = con.A[con.i_b, :]
    b = con.b[con.i_b]
    @constraint(optim, linconstraint, A*ΔŨvar .≤ b)
    # --- nonlinear optimization init ---
    model = mpc.estim.model
    ny, nu, Hp, Hc = model.ny, model.nu, mpc.Hp, mpc.Hc
    nC = (2*Hc*nu + 2*nvar + 2*Hp*ny) - length(mpc.con.b)
    # inspired from https://jump.dev/JuMP.jl/stable/tutorials/nonlinear/tips_and_tricks/#User-defined-operators-with-vector-outputs
    Jfunc, Cfunc = let mpc=mpc, model=model, nC=nC, nvar=nvar , nŶ=Hp*ny
        last_ΔŨtup_float, last_ΔŨtup_dual = nothing, nothing
        Ŷ_cache::DiffCacheType = DiffCache(zeros(nŶ), nvar + 3)
        C_cache::DiffCacheType = DiffCache(zeros(nC), nvar + 3)
        function Jfunc(ΔŨtup::Float64...)
            Ŷ = get_tmp(Ŷ_cache, ΔŨtup[1])
            ΔŨ = collect(ΔŨtup)
            if ΔŨtup != last_ΔŨtup_float
                C = get_tmp(C_cache, ΔŨtup[1])
                Ŷ[:] = predict(mpc, model, ΔŨ)
                C[:] = con_nonlinprog(mpc, model, Ŷ, ΔŨ)
                last_ΔŨtup_float = ΔŨtup
            end
            return obj_nonlinprog(mpc, model, Ŷ, ΔŨ)
        end
        function Jfunc(ΔŨtup::Real...)
            Ŷ = get_tmp(Ŷ_cache, ΔŨtup[1])
            ΔŨ = collect(ΔŨtup)
            if ΔŨtup != last_ΔŨtup_dual
                C = get_tmp(C_cache, ΔŨtup[1])
                Ŷ[:] = predict(mpc, model, ΔŨ)
                C[:] = con_nonlinprog(mpc, model, Ŷ, ΔŨ)
                last_ΔŨtup_dual = ΔŨtup
            end
            return obj_nonlinprog(mpc, model, Ŷ, ΔŨ)
        end
        function con_nonlinprog_i(i, ΔŨtup::NTuple{N, Float64}) where {N}
            C = get_tmp(C_cache, ΔŨtup[1])
            if ΔŨtup != last_ΔŨtup_float
                Ŷ = get_tmp(Ŷ_cache, ΔŨtup[1])
                ΔŨ = collect(ΔŨtup)
                Ŷ[:] = predict(mpc, model, ΔŨ)
                C[:] = con_nonlinprog(mpc, model, Ŷ, ΔŨ)
                last_ΔŨtup_float = ΔŨtup
            end
            return C[i]
        end
        function con_nonlinprog_i(i, ΔŨtup::NTuple{N, Real}) where {N}
            C = get_tmp(C_cache, ΔŨtup[1])
            if ΔŨtup != last_ΔŨtup_dual
                Ŷ = get_tmp(Ŷ_cache, ΔŨtup[1])
                ΔŨ = collect(ΔŨtup)
                Ŷ[:] = predict(mpc, model, ΔŨ)
                C[:] = con_nonlinprog(mpc, model, Ŷ, ΔŨ)
                last_ΔŨtup_dual = ΔŨtup
            end
            return C[i]
        end
        Cfunc = [(ΔŨ...) -> con_nonlinprog_i(i, ΔŨ) for i in 1:nC]
        (Jfunc, Cfunc)
    end
    register(optim, :Jfunc, nvar, Jfunc, autodiff=true)
    @NLobjective(optim, Min, Jfunc(ΔŨvar...))
    if nC ≠ 0
        n = 0
        for i in eachindex(con.Ymin)
            sym = Symbol("C_Ymin_$i")
            register(optim, sym, nvar, Cfunc[n + i], autodiff=true)
        end
        n = lastindex(con.Ymin)
        for i in eachindex(con.Ymax)
            sym = Symbol("C_Ymax_$i")
            register(optim, sym, nvar, Cfunc[n + i], autodiff=true)
        end
    end
    return nothing
end


"No nonlinear constraint for [`NonLinMPC`](@ref) based on [`LinModel`](@ref)."
setnontlincon!(::NonLinMPC, ::LinModel) = nothing

"Set the nonlinear constraints on the output predictions `Ŷ`."
function setnonlincon!(mpc::NonLinMPC, ::NonLinModel)
    optim = mpc.optim
    ΔŨvar = mpc.optim[:ΔŨvar]
    con = mpc.con
    map(con -> delete(optim, con), all_nonlinear_constraints(optim))
    for i in findall(.!isinf.(con.Ymin))
        f_sym = Symbol("C_Ymin_$(i)")
        add_nonlinear_constraint(optim, :($(f_sym)($(ΔŨvar...)) <= 0))
    end
    for i in findall(.!isinf.(con.Ymax))
        f_sym = Symbol("C_Ymax_$(i)")
        add_nonlinear_constraint(optim, :($(f_sym)($(ΔŨvar...)) <= 0))
    end
    return nothing
end


"""
    con_nonlinprog(mpc::NonLinMPC, ::LinModel, ΔŨ::Vector{Real})

Nonlinear constraints for [`NonLinMPC`](@ref) when `model` is a [`LinModel`](@ref).
"""
function con_nonlinprog(mpc::NonLinMPC, model::LinModel, _, ΔŨ::Vector{T}) where {T<:Real}
    return zeros(T, 0)
end
"""
    con_nonlinprog(mpc::NonLinMPC, model::NonLinModel, ΔŨ::Vector{Real})

Nonlinear constrains for [`NonLinMPC`](@ref) when `model` is not a [`LinModel`](@ref).
"""
function con_nonlinprog(mpc::NonLinMPC, ::SimModel, Ŷ, ΔŨ::Vector{T}) where {T<:Real}
    if !isinf(mpc.C) # constraint softening activated :
        ϵ = ΔŨ[end]
        C_Ymin = (mpc.con.Ymin - Ŷ) - ϵ*mpc.con.c_Ymin
        C_Ymax = (Ŷ - mpc.con.Ymax) - ϵ*mpc.con.c_Ymax
    else # no constraint softening :
        C_Ymin = (mpc.con.Ymin - Ŷ)
        C_Ymax = (Ŷ - mpc.con.Ymax)
    end
    # replace -Inf with 0 to avoid INVALID_MODEL error :
    C_Ymin[isinf.(C_Ymin)] .= 0
    C_Ymax[isinf.(C_Ymax)] .= 0
    C = [C_Ymin; C_Ymax]
    return C
end