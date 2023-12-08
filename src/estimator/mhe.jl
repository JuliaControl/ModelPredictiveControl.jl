const DEFAULT_MHE_OPTIMIZER = optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes")

struct MovingHorizonEstimator{
    NT<:Real, 
    SM<:SimModel, 
    JM<:JuMP.GenericModel
} <: StateEstimator{NT}
    model::SM
    # note: `NT` and the number type `JNT` in `JuMP.GenericModel{JNT}` can be
    # different since solvers that support non-Float64 are scarce.
    optim::JM
    W̃::Vector{NT}
    lastu0::Vector{NT}
    x̂::Vector{NT}
    P̂::Hermitian{NT, Matrix{NT}}
    He::Int
    i_ym::Vector{Int}
    nx̂ ::Int
    nym::Int
    nyu::Int
    nxs::Int
    As  ::Matrix{NT}
    Cs_u::Matrix{NT}
    Cs_y::Matrix{NT}
    nint_u ::Vector{Int}
    nint_ym::Vector{Int}
    Â ::Matrix{NT}
    B̂u::Matrix{NT}
    Ĉ ::Matrix{NT}
    B̂d::Matrix{NT}
    D̂d::Matrix{NT}
    P̂0::Hermitian{NT, Matrix{NT}}
    Q̂::Hermitian{NT, Matrix{NT}}
    R̂::Hermitian{NT, Matrix{NT}}
    invP̄::Hermitian{NT, Matrix{NT}}
    invQ̂_He::Hermitian{NT, Matrix{NT}}
    invR̂_He::Hermitian{NT, Matrix{NT}}
    K̂::Matrix{NT}
    M̂::Matrix{NT}
    X̂min::Vector{NT}
    X̂max::Vector{NT}
    i_g::BitVector
    X̂ ::Union{Vector{NT}, Missing} 
    Ym::Union{Vector{NT}, Missing}
    U ::Union{Vector{NT}, Missing}
    D ::Union{Vector{NT}, Missing}
    Ŵ ::Union{Vector{NT}, Missing}
    x̂0_past::Vector{NT}
    Nk::Vector{Int}
    function MovingHorizonEstimator{NT, SM, JM}(
        model::SM, He, i_ym, nint_u, nint_ym, P̂0, Q̂, R̂, optim::JM
    ) where {NT<:Real, SM<:SimModel{NT}, JM<:JuMP.GenericModel}
        nu, nd = model.nu, model.nd
        He < 1  && throw(ArgumentError("Estimation horizon He should be ≥ 1"))
        nym, nyu = validate_ym(model, i_ym)
        As, Cs_u, Cs_y, nint_u, nint_ym = init_estimstoch(model, i_ym, nint_u, nint_ym)
        nxs = size(As, 1)
        nx̂  = model.nx + nxs
        Â, B̂u, Ĉ, B̂d, D̂d = augment_model(model, As, Cs_u, Cs_y)
        validate_kfcov(nym, nx̂, Q̂, R̂, P̂0)
        lastu0 = zeros(NT, model.nu)
        x̂ = [zeros(NT, model.nx); zeros(NT, nxs)]
        P̂0 = Hermitian(P̂0, :L)
        Q̂, R̂ = Hermitian(Q̂, :L),  Hermitian(R̂, :L)
        invP̄ = Hermitian(inv(P̂0), :L)
        invQ̂_He = Hermitian(repeatdiag(inv(Q̂), He), :L)
        invR̂_He = Hermitian(repeatdiag(inv(R̂), He), :L)
        P̂ = copy(P̂0)
        K̂, M̂ = zeros(NT, nx̂, nym), zeros(NT, nx̂, nym)
        nvar, nX̂ = nx̂*(He + 1), nx̂*(He + 1)
        X̂min  , X̂max    =   fill(-Inf, nX̂), fill(+Inf, nX̂)
        i_X̂min, i_X̂max  = .!isinf.(X̂min)  , .!isinf.(X̂max)
        i_g = [i_X̂min; i_X̂max]
        W̃ = zeros(NT, nvar)
        X̂, Ym   = zeros(NT, nx̂*He), zeros(NT, nym*He)
        U, D, Ŵ = zeros(NT, nu*He), zeros(NT, nd*He), zeros(NT, nx̂*He)
        x̂0_past = zeros(NT, nx̂)
        Nk = [0]
        estim = new{NT, SM, JM}(
            model, optim, W̃,
            lastu0, x̂, P̂, He,
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs_u, Cs_y, nint_u, nint_ym,
            Â, B̂u, Ĉ, B̂d, D̂d,
            P̂0, Q̂, R̂, invP̄, invQ̂_He, invR̂_He,
            K̂, M̂,
            X̂min, X̂max, i_g,
            X̂, Ym, U, D, Ŵ, 
            x̂0_past, Nk
        )
        init_optimization!(estim, optim)
        return estim
    end
end


@doc raw"""
    MovingHorizonEstimator(model::SimModel; <keyword arguments>)

Construct a moving horizon estimator based on `model` ([`LinModel`](@ref) or [`NonLinModel`](@ref)).

This estimator can handle constraints on the estimates, see [`setconstraint!`](@ref).
Additionally, `model` is not linearized like the [`ExtendedKalmanFilter`](@ref), and the
probability distribution is not approximated like the [`UnscentedKalmanFilter`](@ref). The
computational costs are drastically higher, however, since it minimizes the following
nonlinear objective function at each discrete time ``k``:
```math
\min_{\mathbf{x̂}_k(k-N_k+1), \mathbf{Ŵ}}   \mathbf{x̄}' \mathbf{P̄}^{-1}       \mathbf{x̄} 
                                         + \mathbf{Ŵ}' \mathbf{Q̂}_{N_k}^{-1} \mathbf{Ŵ}  
                                         + \mathbf{V̂}' \mathbf{R̂}_{N_k}^{-1} \mathbf{V̂}
```
in which the arrival costs are defined by:
```math
\begin{aligned}
    \mathbf{x̄} &= \mathbf{x̂}_k(k-N_k+1) - \mathbf{x̂}_{k-N_k}(k-N_k+1) \\
    \mathbf{P̄} &= \mathbf{P̂}_k(k-N_k+1)
\end{aligned}
```
and the covariances are repeated ``N_k`` times:
```math
\begin{aligned}
    \mathbf{Q̂}_{N_k} &= \text{diag}\mathbf{(Q̂,Q̂,...,Q̂)}  \\
    \mathbf{R̂}_{N_k} &= \text{diag}\mathbf{(R̂,R̂,...,R̂)} 
\end{aligned}
```
The vectors ``\mathbf{Ŵ}`` and ``\mathbf{V̂}`` incorporate the estimated process noise
``\mathbf{ŵ}(k-j)`` and sensor noise ``\mathbf{v̂}(k-j)`` from ``j=0`` to ``N_k-1``. The 
estimation horizon ``H_e`` limits the window length:
```math
N_k =                     \begin{cases} 
    k + 1   &  k < H_e    \\
    H_e     &  k ≥ H_e    \end{cases}
```
"""
function MovingHorizonEstimator(
    model::SM;
    He::Union{Int, Nothing}=nothing,
    i_ym::IntRangeOrVector = 1:model.ny,
    σQ ::Vector = fill(1/model.nx, model.nx),
    σR ::Vector = fill(1, length(i_ym)),
    σP0::Vector = σQ,
    nint_u   ::IntVectorOrInt = 0,
    σQint_u  ::Vector = fill(1, max(sum(nint_u), 0)),
    σP0int_u ::Vector = σQint_u,
    nint_ym  ::IntVectorOrInt = default_nint(model, i_ym, nint_u),
    σQint_ym ::Vector = fill(1, max(sum(nint_ym), 0)),
    σP0int_ym::Vector = σQint_ym,
    optim::JM = JuMP.Model(DEFAULT_MHE_OPTIMIZER, add_bridges=false),
) where {NT<:Real, SM<:SimModel{NT}, JM<:JuMP.GenericModel}
    # estimated covariances matrices (variance = σ²) :
    P̂0 = Diagonal{NT}([σP0; σP0int_u; σP0int_ym].^2)
    Q̂  = Diagonal{NT}([σQ;  σQint_u;  σQint_ym].^2)
    R̂  = Diagonal{NT}(σR.^2)
    isnothing(He) && throw(ArgumentError("Estimation horizon He must be explicitly specified"))        
    return MovingHorizonEstimator{NT, SM, JM}(
        model, He, i_ym, nint_u, nint_ym, P̂0, Q̂, R̂, optim
    )
end


"""
    init_optimization!(estim::MovingHorizonEstimator, optim::JuMP.GenericModel)

Init the nonlinear optimization of [`MovingHorizonEstimator`](@ref).
"""
function init_optimization!(
    estim::MovingHorizonEstimator, optim::JuMP.GenericModel{JNT}
) where JNT<:Real
    He = estim.He
    nŶm, nX̂, ng = He*estim.nym, (He+1)*estim.nx̂, length(estim.i_g)
    # --- variables and linear constraints ---
    nvar = length(estim.W̃)
    set_silent(optim)
    #limit_solve_time(estim) #TODO: add this feature
    @variable(optim, W̃var[1:nvar])
    # --- nonlinear optimization init ---
    # see init_optimization!(mpc::NonLinMPC, optim) for details on the inspiration
    Jfunc, gfunc = let estim=estim, model=estim.model, nvar=nvar , nŶm=nŶm, nX̂=nX̂, ng=ng
        last_W̃tup_float, last_W̃tup_dual = nothing, nothing
        Ŷm_cache::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, nŶm), nvar + 3)
        g_cache ::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, ng) , nvar + 3)
        X̂_cache ::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, nX̂) , nvar + 3)
        function Jfunc(W̃tup::JNT...)
            Ŷm = get_tmp(Ŷm_cache, W̃tup[1])
            W̃  = collect(W̃tup)
            if W̃tup !== last_W̃tup_float
                g = get_tmp(g_cache, W̃tup[1])
                X̂ = get_tmp(X̂_cache, W̃tup[1])
                Ŷm, X̂ = predict!(Ŷm, X̂, estim, model, W̃)
                g = con_nonlinprog!(g, estim, model, X̂)
                last_W̃tup_float = W̃tup
            end
            return obj_nonlinprog(estim, model, Ŷm, W̃)
        end
        function Jfunc(W̃tup::ForwardDiff.Dual...)
            Ŷm = get_tmp(Ŷm_cache, W̃tup[1])
            W̃  = collect(W̃tup)
            if W̃tup !== last_W̃tup_dual
                g = get_tmp(g_cache, W̃tup[1])
                X̂ = get_tmp(X̂_cache, W̃tup[1])
                Ŷm, X̂ = predict!(Ŷm, X̂, estim, model, W̃)
                g = con_nonlinprog!(g, estim, model, X̂)
                last_W̃tup_dual = W̃tup
            end
            return obj_nonlinprog(estim, model, Ŷm, W̃)
        end
        function gfunc_i(i, W̃tup::NTuple{N, JNT}) where N
            g = get_tmp(g_cache, W̃tup[1])
            if W̃tup !== last_W̃tup_float
                Ŷm = get_tmp(Ŷm_cache, W̃tup[1])
                X̂ = get_tmp(X̂_cache, W̃tup[1])
                W̃  = collect(W̃tup)
                Ŷm, X̂ = predict!(Ŷm, X̂, estim, model, W̃)
                g = con_nonlinprog!(g, estim, model, X̂)
                last_W̃tup_float = W̃tup
            end
            return g[i]
        end 
        function gfunc_i(i, W̃tup::NTuple{N, ForwardDiff.Dual}) where N
            g = get_tmp(g_cache, W̃tup[1])
            if W̃tup !== last_W̃tup_dual
                Ŷm = get_tmp(Ŷm_cache, W̃tup[1])
                X̂ = get_tmp(X̂_cache, W̃tup[1])
                W̃  = collect(W̃tup)
                Ŷm, X̂ = predict!(Ŷm, X̂, estim, model, W̃)
                g = con_nonlinprog!(g, estim, model, X̂)
                last_W̃tup_dual = W̃tup
            end
            return g[i]
        end
        gfunc = [(W̃...) -> gfunc_i(i, W̃) for i in 1:ng]
        Jfunc, gfunc
    end
    register(optim, :Jfunc, nvar, Jfunc, autodiff=true)
    @NLobjective(optim, Min, Jfunc(W̃var...))
    if ng ≠ 0
        i_end_X̂min = nX̂
        for i in eachindex(estim.X̂min)
            sym = Symbol("g_X̂min_$i")
            register(optim, sym, nvar, gfunc[i], autodiff=true)
        end
        for i in eachindex(estim.X̂max)
            sym = Symbol("g_X̂max_$i")
            register(optim, sym, nvar, gfunc[i_end_X̂min+i], autodiff=true)
        end
    end
    # TODO: moved this call to setconstraint! when the function will handle MHE
    setnonlincon!(estim, estim.model)
    return nothing
end

@doc raw"""
    setconstraint!(estim::MovingHorizonEstimator; <keyword arguments>) -> estim

Set the constraint parameters of `estim` [`MovingHorizonEstimator`](@ref).

The constraints of the moving horizon estimator are:
```math 
\mathbf{x̂_{min}} ≤ \mathbf{x̂}_k(k-j+1) ≤ \mathbf{x̂_{max}} \qquad j = 0, 1, ... , N_k \\
```
Note that state constraints are applied on the augmented state vector ``\mathbf{x̂}`` (see
the extended help of [`SteadyKalmanFilter`](@ref) for details on augmentation).

# Arguments
!!! info
    The default constraints are mentioned here for clarity but omitting a keyword argument 
    will not re-assign to its default value (defaults are set at construction only).

- `estim::MovingHorizonEstimator` : moving horizon estimator to set constraints.
- `x̂min = fill(-Inf,nx̂)` : augmented state vector lower bounds ``\mathbf{x̂_{min}}``.
- `x̂max = fill(+Inf,nx̂)` : augmented state vector upper bounds ``\mathbf{x̂_{max}}``.
- all the keyword arguments above but with a capital letter, e.g. `X̂max` or `X̂min` : for
  time-varying constraints (see Extended Help).

# Examples
```jldoctest
julia> estim = MovingHorizonEstimator(LinModel(ss(0.5,1,1,0,1)), He=3);

julia> estim = setconstraint!(estim, x̂min=[-50, -50], x̂max=[50, 50])
MovingHorizonEstimator estimator with a sample time Ts = 1.0 s, LinModel and:
 3 estimation steps He
 1 manipulated inputs u (0 integrating states)
 2 states x̂
 1 measured outputs ym (1 integrating states)
 0 unmeasured outputs yu
 0 measured disturbances d
```
"""
function setconstraint!(
    estim::MovingHorizonEstimator; 
    x̂min = nothing, x̂max = nothing,
    X̂min = nothing, X̂max = nothing,
)
    model, optim = estim.model, estim.optim
    nx̂, He = estim.nx̂, estim.He
    nX̂ = nx̂*(He+1)
    notSolvedYet = (termination_status(optim) == OPTIMIZE_NOT_CALLED)
    isnothing(X̂min) && !isnothing(x̂min) && (X̂min = repeat(x̂min, He+1))
    isnothing(X̂max) && !isnothing(x̂max) && (X̂max = repeat(x̂max, He+1))
    if !isnothing(X̂min)
        size(X̂min) == (nX̂,) || throw(ArgumentError("X̂min size must be $((nX̂,))"))
        estim.X̂min[:] = X̂min
    end
    if !isnothing(X̂max)
        size(X̂max) == (nX̂,) || throw(ArgumentError("X̂max size must be $((nX̂,))"))
        estim.X̂max[:] = X̂max
    end
    i_X̂min, i_X̂max  = .!isinf.(estim.X̂min)  , .!isinf.(estim.X̂max)
    i_g = [i_X̂min; i_X̂max]
    if notSolvedYet
        estim.i_g[:] = i_g
        setnonlincon!(estim, model)
    else
        if i_g ≠ estim.i_g
            error("Cannot modify ±Inf constraints after calling updatestate!")
        end
    end
    return estim
end

"Set the nonlinear constraints on the output predictions `Ŷ` and terminal states `x̂end`."
function setnonlincon!(estim::MovingHorizonEstimator, ::SimModel)
    optim = estim.optim
    W̃var = optim[:W̃var]
    map(con -> delete(optim, con), all_nonlinear_constraints(optim))
    for i in findall(.!isinf.(estim.X̂min))
        f_sym = Symbol("g_X̂min_$(i)")
        add_nonlinear_constraint(optim, :($(f_sym)($(W̃var...)) <= 0))
    end
    for i in findall(.!isinf.(estim.X̂max))
        f_sym = Symbol("g_X̂max_$(i)")
        add_nonlinear_constraint(optim, :($(f_sym)($(W̃var...)) <= 0))
    end
    return nothing
end

"Print the overall dimensions of the state estimator `estim` with left padding `n`."
function print_estim_dim(io::IO, estim::MovingHorizonEstimator, n)
    nu, nd = estim.model.nu, estim.model.nd
    nx̂, nym, nyu = estim.nx̂, estim.nym, estim.nyu
    He = estim.He
    println(io, "$(lpad(He, n)) estimation steps He")
    println(io, "$(lpad(nu, n)) manipulated inputs u ($(sum(estim.nint_u)) integrating states)")
    println(io, "$(lpad(nx̂, n)) states x̂")
    println(io, "$(lpad(nym, n)) measured outputs ym ($(sum(estim.nint_ym)) integrating states)")
    println(io, "$(lpad(nyu, n)) unmeasured outputs yu")
    print(io,   "$(lpad(nd, n)) measured disturbances d")
end

"Reset `estim.P̂`, `estim.invP̄` and the time windows for the moving horizon estimator."
function init_estimate_cov!(estim::MovingHorizonEstimator, _ , _ , _ ) 
    estim.invP̄.data[:] = Hermitian(inv(estim.P̂0), :L)
    estim.P̂.data[:]    = estim.P̂0
    estim.x̂0_past     .= 0
    estim.W̃           .= 0
    estim.X̂           .= 0
    estim.Ym          .= 0
    estim.U           .= 0
    estim.D           .= 0
    estim.Ŵ           .= 0
    estim.Nk          .= 0
    return nothing
end

@doc raw"""
    update_estimate!(estim::MovingHorizonEstimator, u, ym, d)
    
Update [`MovingHorizonEstimator`](@ref) state `estim.x̂`.

TBW
"""
function update_estimate!(estim::MovingHorizonEstimator{NT}, u, ym, d) where NT<:Real
    model, optim, x̂ = estim.model, estim.optim, estim.x̂
    nx̂, nym, nu, nd, nŵ, He = estim.nx̂, estim.nym, model.nu, model.nd, estim.nx̂, estim.He
    ŵ = zeros(nŵ) # ŵ(k) = 0 for warm-starting
    estim.Nk .+= 1
    Nk = estim.Nk[]
    if Nk > He
        estim.X̂[:]  = [estim.X̂[nx̂+1:end]  ; x̂]
        estim.Ym[:] = [estim.Ym[nym+1:end]; ym]
        estim.U[:]  = [estim.U[nu+1:end]  ; u]
        estim.D[:]  = [estim.D[nd+1:end]  ; d]
        estim.Ŵ[:]  = [estim.Ŵ[nŵ+1:end]  ; ŵ]
        estim.Nk[:] = [He]
    else
        estim.X̂[(1 + nx̂*(Nk-1)):(nx̂*Nk)]    = x̂
        estim.Ym[(1 + nym*(Nk-1)):(nym*Nk)] = ym
        estim.U[(1 + nu*(Nk-1)):(nu*Nk)]    = u
        estim.D[(1 + nd*(Nk-1)):(nd*Nk)]    = d
        estim.Ŵ[(1 + nŵ*(Nk-1)):(nŵ*Nk)]    = ŵ
    end
    Nk = estim.Nk[]
    nŴ, nYm, nX̂ = nx̂*Nk, nym*Nk, nx̂*(Nk+1)
    W̃var::Vector{VariableRef} = optim[:W̃var]
    Ŷm = Vector{NT}(undef, nYm)
    X̂  = Vector{NT}(undef, nX̂)
    estim.x̂0_past[:] = estim.X̂[1:nx̂]
    W̃0 = [estim.x̂0_past; estim.Ŵ]
    Ŷm, X̂ = predict!(Ŷm, X̂, estim, model, W̃0)
    J0 = obj_nonlinprog(estim, model, Ŷm, W̃0)
    # initial W̃0 with Ŵ=0 if objective or constraint function not finite :
    isfinite(J0) || (W̃0 = [estim.x̂0_past; zeros(NT, nŴ)])
    set_start_value.(W̃var, W̃0)
    # at start, when time windows are not filled, some decision variables are fixed at 0:
    unfix.(W̃var[is_fixed.(W̃var)])
    fix.(W̃var[(nx̂*(Nk+1)+1):end], 0.0) 
    try
        optimize!(optim)
    catch err
        if isa(err, MOI.UnsupportedAttribute{MOI.VariablePrimalStart})
            # reset_optimizer to unset warm-start, set_start_value.(nothing) seems buggy
            MOIU.reset_optimizer(optim)
            optimize!(optim)
        else
            rethrow(err)
        end
    end
    status = termination_status(optim)
    W̃curr, W̃last = value.(W̃var), W̃0
    if !(status == OPTIMAL || status == LOCALLY_SOLVED)
        if isfatal(status)
            @error("MHE terminated without solution: estimation in open-loop", 
                   status)         
        else
            @warn("MHE termination status not OPTIMAL or LOCALLY_SOLVED: keeping "*
                  "solution anyway", status)
        end
        @debug solution_summary(optim, verbose=true)
    end
    estim.W̃[:] = !isfatal(status) ? W̃curr : W̃last
    estim.Ŵ[1:nŴ] = estim.W̃[nx̂+1:nx̂+nŴ] # update Ŵ with optimum for next time step
    Ŷm, X̂ = predict!(Ŷm, X̂, estim, model, estim.W̃)
    x̂[:] = X̂[(1 + nx̂*Nk):(nx̂*(Nk+1))]
    if Nk == He
        # update the arrival covariance with an Extended Kalman Filter:
        # TODO: also support UnscentedKalmanFilter, and KalmanFilter for LinModel
        F̂  = ForwardDiff.jacobian(x̂ -> f̂(estim, estim.model, x̂, u, d), estim.x̂)
        Ĥ  = ForwardDiff.jacobian(x̂ -> ĥ(estim, estim.model, x̂, d), estim.x̂)
        Ĥm = Ĥ[estim.i_ym, :] 
        update_estimate_kf!(estim, F̂, Ĥm, u, ym, d, updatestate=false)
        P̄ = estim.P̂
        estim.invP̄.data[:] = Hermitian(inv(P̄), :L) # .data is necessary for Hermitian
    end
    return nothing
end

"""
    obj_nonlinprog(estim::MovingHorizonEstimator, model::SimModel, W̃)

Objective function for [`MovingHorizonEstimator`](@ref).

The function `dot(x, A, x)` is a performant way of calculating `x'*A*x`.
"""
function obj_nonlinprog(
    estim::MovingHorizonEstimator, ::SimModel, Ŷm, W̃::Vector{T}
) where {T<:Real}
    Nk = estim.Nk[]
    nYm, nŴ, nx̂, invP̄ = Nk*estim.nym, Nk*estim.nx̂, estim.nx̂, estim.invP̄
    invQ̂_Nk, invR̂_Nk = @views estim.invQ̂_He[1:nŴ, 1:nŴ], estim.invR̂_He[1:nYm, 1:nYm]
    x̂0 = @views W̃[1:nx̂] # W̃ = [x̂(k-Nk+1|k); Ŵ]
    x̄0 = x̂0 - estim.x̂0_past  
    V̂  = @views estim.Ym[1:nYm] - Ŷm[1:nYm]
    Ŵ  = @views W̃[nx̂+1:nx̂+nŴ]
    return dot(x̄0, invP̄, x̄0) + dot(Ŵ, invQ̂_Nk, Ŵ) + dot(V̂, invR̂_Nk, V̂)
end

function predict!(
    Ŷm, X̂, estim::MovingHorizonEstimator, model::SimModel, W̃::Vector{T}
) where {T<:Real}
    nu, nd, nx̂, nym, Nk = model.nu, model.nd, estim.nx̂, estim.nym, estim.Nk[]
    X̂[1:nx̂] = W̃[1:nx̂] # W̃ = [x̂(k-Nk+1|k); Ŵ]
    for j=1:Nk
        u = @views estim.U[(1 + nu*(j-1)):(nu*j)]
        d = @views estim.D[(1 + nd*(j-1)):(nd*j)]
        ŵ = @views W̃[(1 + nx̂*j):(nx̂*(j+1))]
        x̂ = @views X̂[(1 + nx̂*(j-1)):(nx̂*j)]
        Ŷm[(1 + nym*(j-1)):(nym*j)] = ĥ(estim, model, x̂, d)[estim.i_ym]
        X̂[(1 + nx̂*j):(nx̂*(j+1))]    = f̂(estim, model, x̂, u, d) + ŵ
    end
    return Ŷm, X̂
end

function con_nonlinprog!(g, estim::MovingHorizonEstimator, ::SimModel, X̂)
    nX̂con, nX̂ = length(estim.X̂min), estim.Nk[]*estim.nx̂
    for i in eachindex(g)
        estim.i_g[i] || continue
        if i ≤ nX̂con
            j = i
            if j ≤ nX̂
                g[i] = estim.X̂min[j] - X̂[j]
            end
        else
            j = i - nX̂con
            if j ≤ nX̂
                g[i] = X̂[j] - estim.X̂max[j]
            end
        end
    end
    return g
end