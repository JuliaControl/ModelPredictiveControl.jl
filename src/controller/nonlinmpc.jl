struct NonLinMPC{S<:StateEstimator, JEFunc<:Function} <: PredictiveController
    estim::S
    optim::JuMP.Model
    info::OptimInfo
    con::ControllerConstraint
    Hp::Int
    Hc::Int
    M_Hp::Diagonal{Float64, Vector{Float64}}
    Ñ_Hc::Diagonal{Float64, Vector{Float64}}
    L_Hp::Diagonal{Float64, Vector{Float64}}
    C::Float64
    E::Float64
    JE::JEFunc
    R̂u::Vector{Float64}
    S̃_Hp::Matrix{Bool}
    T_Hp::Matrix{Bool}
    S̃_Hc::Matrix{Bool}
    T_Hc::Matrix{Bool}
    Ẽ ::Matrix{Float64}
    G ::Matrix{Float64}
    J ::Matrix{Float64}
    Kd::Matrix{Float64}
    Q ::Matrix{Float64}
    P̃ ::Hermitian{Float64, Matrix{Float64}}
    q̃ ::Vector{Float64}
    Ks::Matrix{Float64}
    Ps::Matrix{Float64}
    Yop::Vector{Float64}
    Dop::Vector{Float64}
    function NonLinMPC{S, JEFunc}(
        estim::S, Hp, Hc, Mwt, Nwt, Lwt, Cwt, Ewt, JE::JEFunc, ru, optim
    ) where {S<:StateEstimator, JEFunc<:Function}
        model = estim.model
        nu, nx, ny, nd =  model.nu, model.nx, model.ny, model.nd
        validate_weights(model, Hp, Hc, Mwt, Nwt, Lwt, Cwt, ru, Ewt)
        M_Hp = Diagonal(convert(Vector{Float64}, repeat(Mwt, Hp)))
        N_Hc = Diagonal(convert(Vector{Float64}, repeat(Nwt, Hc)))
        L_Hp = Diagonal(convert(Vector{Float64}, repeat(Lwt, Hp)))
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
        A_Umin, A_Umax, S̃_Hp, S̃_Hc = relaxU(C, c_Umin, c_Umax, S_Hp, S_Hc)
        A_ΔŨmin, A_ΔŨmax, ΔŨmin, ΔŨmax, Ñ_Hc = relaxΔU(C,c_ΔUmin,c_ΔUmax,ΔUmin,ΔUmax,N_Hc)
        A_Ŷmin, A_Ŷmax, Ẽ = relaxŶ(model, C, c_Ŷmin, c_Ŷmax, E)
        i_Umin,  i_Umax  = .!isinf.(Umin),  .!isinf.(Umax)
        i_ΔŨmin, i_ΔŨmax = .!isinf.(ΔŨmin), .!isinf.(ΔŨmin)
        i_Ŷmin,  i_Ŷmax  = .!isinf.(Ŷmin),  .!isinf.(Ŷmax)
        A, i_b = init_linconstraint(model, 
            A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, A_Ŷmin, A_Ŷmax,
            i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ŷmin, i_Ŷmax
        )
        con = ControllerConstraint(
            Umin    , Umax  , ΔŨmin  , ΔŨmax    , Ŷmin  , Ŷmax,
            c_Umin  , c_Umax, c_ΔUmin, c_ΔUmax  , c_Ŷmin, c_Ŷmax,
            A       , i_b   , i_Ŷmin , i_Ŷmax
        )
        nvar = size(Ẽ, 2)
        P̃ = init_quadprog(model, Ẽ, S̃_Hp, M_Hp, Ñ_Hc, L_Hp)
        q̃ = zeros(nvar) # dummy q̃ value (vector updated just before optimization)
        Ks, Ps = init_stochpred(estim, Hp)
        Yop, Dop = repeat(model.yop, Hp), repeat(model.dop, Hp)
        set_silent(optim)
        @variable(optim, ΔŨ[1:nvar])
        A = [A_Umin; A_Umax; A_ΔŨmin; A_ΔŨmax; A_Ŷmin; A_Ŷmax]
        A = con.A[con.i_b, :]
        b = zeros(sum(con.i_b)) # dummy b value (vector updated just before optimization)
        @constraint(optim, linconstraint, A*ΔŨ .≤ b)
        ΔŨ0 = zeros(nvar)
        ϵ = isinf(C) ? NaN : 0.0 # C = Inf means hard constraints only
        u, U = copy(model.uop), repeat(model.uop, Hp)
        ŷ, Ŷ = copy(model.yop), repeat(model.yop, Hp)
        ŷs, Ŷs = zeros(ny), zeros(ny*Hp)
        info = OptimInfo(ΔŨ0, ϵ, 0, u, U, ŷ, Ŷ, ŷs, Ŷs)
        # dummy x̂d, F, d0, D̂0, q̃ values (updated just before optimization)
        mpc = new(
            estim, optim, info, con,
            Hp, Hc, 
            M_Hp, Ñ_Hc, L_Hp, Cwt, Ewt, JE, R̂u,
            S̃_Hp, T_Hp, S̃_Hc, T_Hc, 
            Ẽ, G, J, Kd, Q, P̃, q̃,
            Ks, Ps,
            Yop, Dop,
        )
        J = (ΔŨ...) -> obj_nonlinprog(mpc, model, ΔŨ)
        register(mpc.optim, :J, nvar, J, autodiff=true)
        @NLobjective(mpc.optim, Min, J(ΔŨ...))
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
since ``H_c ≤ H_p`` implies that ``\mathbf{u}(k+H_p) = \mathbf{u}(k+H_p-1)``.

!!! tip
    Replace any of the 3 arguments with `_` if not needed (see `JE` default value below).

This method uses the default state estimator, an [`UnscentedKalmanFilter`](@ref) with 
default arguments.

# Arguments
- `model::SimModel` : model used for controller predictions and state estimations.
- `Hp=10`: prediction horizon ``H_p``.
- `Hc=2` : control horizon ``H_c``.
- `Mwt=fill(1.0,model.ny)` : main diagonal of ``\mathbf{M}`` weight matrix (vector)
- `Nwt=fill(0.1,model.nu)` : main diagonal of ``\mathbf{N}`` weight matrix (vector)
- `Lwt=fill(0.0,model.nu)` : main diagonal of ``\mathbf{L}`` weight matrix (vector)
- `Cwt=1e5` : slack variable weight ``C`` (scalar), use `Cwt=Inf` for hard constraints only
- `Ewt=1.0` : economic costs weight ``E`` (scalar). 
- `JE=(_,_,_)->0.0` : economic function ``J_E(\mathbf{U}_E, \mathbf{D̂}_E, \mathbf{Ŷ}_E)``.
- `ru=model.uop` : manipulated input setpoints ``\mathbf{r_u}`` (vector)
- `optim=JuMP.Model(Ipopt.Optimizer)` : nonlinear optimizer used in the predictive 
   controller, provided as a [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/reference/models/#JuMP.Model)
   (default to [`Ipopt.jl`](https://github.com/jump-dev/Ipopt.jl) optimizer)

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_)->0.5x+u, (x,_)->2x, 10, 1, 1, 1);

julia> mpc = NonLinMPC(model, Hp=20, Hc=1, Cwt=1e6)
NonLinMPC controller with a sample time Ts = 10.0 s, UnscentedKalmanFilter estimator and:
 1 manipulated inputs u
 2 states x̂
 1 measured outputs ym
 0 unmeasured outputs yu
 0 measured disturbances d
```

# Extended Help
`NonLinMPC` controllers based on [`LinModel`](@ref) compute the predictions with matrix 
algebra instead of a `for` loop. This feature can accelerate the optimization and is not 
available in any other package, to my knowledge.
"""
NonLinMPC(model::SimModel; kwargs...) = NonLinMPC(UnscentedKalmanFilter(model); kwargs...)


"""
    NonLinMPC(estim::StateEstimator; <keyword arguments>)

Use custom state estimator `estim` to construct `NonLinMPC`.

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_)->0.5x+u, (x,_)->2x, 10, 1, 1, 1);

julia> estim = UnscentedKalmanFilter(model, σQ_int=[0.05]);

julia> mpc = NonLinMPC(estim, Hp=20, Hc=1, Cwt=1e6)
NonLinMPC controller with a sample time Ts = 10.0 s, UnscentedKalmanFilter estimator and:
 1 manipulated inputs u
 2 states x̂
 1 measured outputs ym
 0 unmeasured outputs yu
 0 measured disturbances d
```
"""
function NonLinMPC(
    estim::S;
    Hp::Int = 10,
    Hc::Int = 2,
    Mwt = fill(1.0, estim.model.ny),
    Nwt = fill(0.1, estim.model.nu),
    Lwt = fill(0.0, estim.model.nu),
    Cwt = 1e5,
    Ewt = 1.0,
    JE::JEFunc = (_,_,_) -> 0.0,
    ru  = estim.model.uop,
    optim::JuMP.Model = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"))
) where {S<:StateEstimator, JEFunc<:Function}
    return NonLinMPC{S, JEFunc}(estim, Hp, Hc, Mwt, Nwt, Lwt, Cwt, Ewt, JE, ru, optim)
end


function init_objective(mpc::NonLinMPC, _  , x̂d, F, d0, D̂0)
    return nothing
end

function obj_nonlinprog(mpc::NonLinMPC, ::LinModel, ΔŨ::NTuple{N, T}) where {T, N}
    ΔŨ = collect(ΔŨ) # convert NTuple to Vector
    Jqp = obj_quadprog(ΔŨ, mpc.P̃, mpc.q̃)
    return Jqp
end


function obj_nonlinprog(mpc::NonLinMPC, model::SimModel, ΔŨ::NTuple{N, T}) where {T,N}
    J = 0.0
    #println("yoSimModel")
    return J
end



function write_info!(mpc::NonLinMPC, ΔŨ, J, ŷs, Ŷs, lastu, F, ym, d)
    return nothing
end