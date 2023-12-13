const DEFAULT_MHE_OPTIMIZER = optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes")

"Include all the data for the constraints of [`MovingHorizonEstimator`](@ref)"
struct EstimatorConstraint{NT<:Real}
    X̂min    ::Vector{NT}
    X̂max    ::Vector{NT}
    x̂min    ::Vector{NT}
    x̂max    ::Vector{NT}
    A_x̂min  ::Matrix{NT}
    A_x̂max  ::Matrix{NT}
    A       ::Matrix{NT}
    b       ::Vector{NT}
    i_b     ::BitVector
    i_g     ::BitVector
end

struct MovingHorizonEstimator{
    NT<:Real, 
    SM<:SimModel, 
    JM<:JuMP.GenericModel
} <: StateEstimator{NT}
    model::SM
    # note: `NT` and the number type `JNT` in `JuMP.GenericModel{JNT}` can be
    # different since solvers that support non-Float64 are scarce.
    optim::JM
    con::EstimatorConstraint{NT}
    W̃::Vector{NT}
    lastu0::Vector{NT}
    x̂::Vector{NT}
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
    Â   ::Matrix{NT}
    B̂u  ::Matrix{NT}
    Ĉ   ::Matrix{NT}
    B̂d  ::Matrix{NT}
    D̂d  ::Matrix{NT}
    P̂0::Hermitian{NT, Matrix{NT}}
    Q̂::Hermitian{NT, Matrix{NT}}
    R̂::Hermitian{NT, Matrix{NT}}
    invP̄::Hermitian{NT, Matrix{NT}}
    invQ̂_He::Hermitian{NT, Matrix{NT}}
    invR̂_He::Hermitian{NT, Matrix{NT}}
    M̂::Matrix{NT}
    X̂ ::Union{Vector{NT}, Missing} 
    Ym::Union{Vector{NT}, Missing}
    U ::Union{Vector{NT}, Missing}
    D ::Union{Vector{NT}, Missing}
    Ŵ ::Union{Vector{NT}, Missing}
    x̂arr_old::Vector{NT}
    P̂arr_old::Hermitian{NT, Matrix{NT}}
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
        M̂ = zeros(NT, nx̂, nym)
        con = init_defaultcon(model, He, nx̂)
        nvar = nx̂*(He+1)
        W̃ = zeros(NT, nvar)
        X̂, Ym   = zeros(NT, nx̂*He), zeros(NT, nym*He)
        U, D, Ŵ = zeros(NT, nu*He), zeros(NT, nd*He), zeros(NT, nx̂*He)
        x̂arr_old = zeros(NT, nx̂)
        P̂arr_old = copy(P̂0)
        Nk = [0]
        estim = new{NT, SM, JM}(
            model, optim, con, 
            W̃, lastu0, x̂, 
            He,
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs_u, Cs_y, nint_u, nint_ym,
            Â, B̂u, Ĉ, B̂d, D̂d,
            P̂0, Q̂, R̂, invP̄, invQ̂_He, invR̂_He,
            M̂,
            X̂, Ym, U, D, Ŵ, 
            x̂arr_old, P̂arr_old, Nk
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
in which the arrival costs are evaluated from the states estimated at time ``k-N_k``:
```math
\begin{aligned}
    \mathbf{x̄} &= \mathbf{x̂}_k(k-N_k+1) - \mathbf{x̂}_{k-N_k}(k-N_k+1) \\
    \mathbf{P̄} &= \mathbf{P̂}_{k-N_k}(k-N_k+1)
\end{aligned}
```
and the covariances are repeated ``N_k`` times:
```math
\begin{aligned}
    \mathbf{Q̂}_{N_k} &= \text{diag}\mathbf{(Q̂,Q̂,...,Q̂)}  \\
    \mathbf{R̂}_{N_k} &= \text{diag}\mathbf{(R̂,R̂,...,R̂)} 
\end{aligned}
```
The estimation horizon ``H_e`` limits the window length ``N_k = \min(k+1, H_e)``. The 
vectors ``\mathbf{Ŵ}`` and ``\mathbf{V̂}`` encompass the estimated process noise
``\mathbf{ŵ}(k-j)`` and sensor noise ``\mathbf{v̂}(k-j)`` from ``j=N_k-1`` to ``0``. The 
Extended Help explicitly defines the two vectors. See [`SteadyKalmanFilter`](@ref) for
details on ``\mathbf{R̂}, \mathbf{Q̂}`` covariances and model augmentation. The process
model is identical to the one in [`UnscentedKalmanFilter`](@ref) documentation. The matrix
``\mathbf{P̂}_{k-N_k}(k-N_k+1)`` is estimated with an [`ExtendedKalmanFilter`](@ref).

!!! warning
    See the Extended Help of [`NonLinMPC`](@ref) function if you get an error like:    
    `MethodError: no method matching (::var"##")(::Vector{ForwardDiff.Dual})`.

# Arguments
- `model::SimModel` : (deterministic) model for the estimations.
- `He=nothing`: estimation horizon ``H_e``, must be specified.
- `optim=JuMP.Model(Ipopt.Optimizer)` : nonlinear optimizer used in the moving horizon
   estimator, provided as a [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.Model)
   (default to [`Ipopt.jl`](https://github.com/jump-dev/Ipopt.jl) optimizer).
- `<keyword arguments>` of [`SteadyKalmanFilter`](@ref) constructor.
- `<keyword arguments>` of [`KalmanFilter`](@ref) constructor.

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_)->0.1x+u, (x,_)->2x, 10.0, 1, 1, 1);

julia> estim = MovingHorizonEstimator(model, He=5, σR=[1], σP0=[0.01])
MovingHorizonEstimator estimator with a sample time Ts = 10.0 s, NonLinModel and:
 5 estimation steps He
 1 manipulated inputs u (0 integrating states)
 2 states x̂
 1 measured outputs ym (1 integrating states)
 0 unmeasured outputs yu
 0 measured disturbances d
```

# Extended Help
The estimated process and sensor noises are defined as:
```math
\mathbf{Ŵ} = 
\begin{bmatrix}
    \mathbf{ŵ}(k-N_k+1)     \\
    \mathbf{ŵ}(k-N_k+2)     \\
    \vdots                  \\
    \mathbf{ŵ}(k)
\end{bmatrix} , \quad
\mathbf{V̂} =
\begin{bmatrix}
    \mathbf{v̂}(k-N_k+1)     \\
    \mathbf{v̂}(k-N_k+2)     \\
    \vdots                  \\
    \mathbf{v̂}(k)
\end{bmatrix}
```
in which ``\mathbf{v̂}(k-j) = 
\mathbf{y^m}(k-j) - \mathbf{ĥ^m}\big(\mathbf{x̂}_k(k-j), \mathbf{d}(k-j)\big)`` from ``j = 
N_k-1`` to ``0``. The augmented model ``\mathbf{f̂}`` and the process noise estimate
recursively generates the state estimates ``\mathbf{x̂}_k(k-j+1) = 
\mathbf{f̂}\big(\mathbf{x̂}_k(k-j), \mathbf{u}(k-j), \mathbf{d}(k-j)\big) + \mathbf{ŵ}(k-j)``
from ``j=N_k-1`` to ``0``, inclusively. 
"""
function MovingHorizonEstimator(
    model::SM;
    He::Union{Int, Nothing}=nothing,
    i_ym::IntRangeOrVector = 1:model.ny,
    σP0::Vector = fill(1/model.nx, model.nx),
    σQ ::Vector = fill(1/model.nx, model.nx),
    σR ::Vector = fill(1, length(i_ym)),
    nint_u   ::IntVectorOrInt = 0,
    σQint_u  ::Vector = fill(1, max(sum(nint_u), 0)),
    σP0int_u ::Vector = fill(1, max(sum(nint_u), 0)),
    nint_ym  ::IntVectorOrInt = default_nint(model, i_ym, nint_u),
    σQint_ym ::Vector = fill(1, max(sum(nint_ym), 0)),
    σP0int_ym::Vector = fill(1, max(sum(nint_ym), 0)),
    optim::JM = JuMP.Model(DEFAULT_MHE_OPTIMIZER, add_bridges=false),
) where {NT<:Real, SM<:SimModel{NT}, JM<:JuMP.GenericModel}
    # estimated covariances matrices (variance = σ²) :
    P̂0 = Hermitian(diagm(NT[σP0; σP0int_u; σP0int_ym].^2), :L)
    Q̂  = Hermitian(diagm(NT[σQ;  σQint_u;  σQint_ym ].^2), :L)
    R̂  = Hermitian(diagm(NT[σR;].^2), :L)
    isnothing(He) && throw(ArgumentError("Estimation horizon He must be explicitly specified"))        
    return MovingHorizonEstimator{NT, SM, JM}(
        model, He, i_ym, nint_u, nint_ym, P̂0, Q̂, R̂, optim
    )
end

@doc raw"""
    MovingHorizonEstimator(model, He, i_ym, nint_u, nint_ym, P̂0, Q̂, R̂, optim)

Construct the estimator from the augmented covariance matrices `P̂0`, `Q̂` and `R̂`.

This syntax allows nonzero off-diagonal elements in ``\mathbf{P̂}_{-1}(0), \mathbf{Q̂, R̂}``.
"""
function MovingHorizonEstimator(
    model::SM, He, i_ym, nint_u, nint_ym, P̂0, Q̂, R̂, optim::JM
) where {NT<:Real, SM<:SimModel{NT}, JM<:JuMP.GenericModel}
    P̂0, Q̂, R̂ = to_mat(P̂0), to_mat(Q̂), to_mat(R̂)
    return MovingHorizonEstimator{NT, SM, JM}(
        model, He, i_ym, nint_u, nint_ym, P̂0, Q̂ , R̂, optim
    )
end


"""
    init_defaultcon(model::SimModel, He)

TBW
"""
function init_defaultcon(model::SimModel{NT}, He, nx̂) where {NT<:Real}
    nvar, nX̂ = nx̂*(He+1), nx̂*He
    X̂min, X̂max = fill(convert(NT,-Inf), nX̂), fill(convert(NT,+Inf), nX̂)
    x̂min, x̂max = fill(convert(NT,-Inf), nx̂), fill(convert(NT,+Inf), nx̂)
    i_X̂min, i_X̂max = .!isinf.(X̂min)  , .!isinf.(X̂max)
    i_g = [i_X̂min; i_X̂max]
    i_x̂min, i_x̂max = .!isinf.(x̂min), .!isinf.(x̂max)
    i_b = [i_x̂min; i_x̂max]
    I_nx̂ = Matrix{NT}(I, nx̂, nx̂)
    A_x̂min, A_x̂max = [-I_nx̂ zeros(NT, nx̂, nvar-nx̂)], [I_nx̂ zeros(NT, nx̂, nvar-nx̂)]
    A = [A_x̂min; A_x̂max]
    b = zeros(NT, size(A, 1)) # dummy b vector (updated just before optimization)
    con = EstimatorConstraint{NT}(X̂min, X̂max, x̂min, x̂max, A_x̂min, A_x̂max, A, b, i_b, i_g)
end

"""
    init_optimization!(estim::MovingHorizonEstimator, optim::JuMP.GenericModel)

Init the nonlinear optimization of [`MovingHorizonEstimator`](@ref).
"""
function init_optimization!(
    estim::MovingHorizonEstimator, optim::JuMP.GenericModel{JNT}
) where JNT<:Real
    He, con = estim.He, estim.con
    nŶm, nX̂, ng = He*estim.nym, He*estim.nx̂, length(con.i_g)
    # --- variables and linear constraints ---
    nvar = length(estim.W̃)
    set_silent(optim)
    #limit_solve_time(estim) #TODO: add this feature
    @variable(optim, W̃var[1:nvar])
    A = con.A[con.i_b, :]
    b = con.b[con.i_b]
    @constraint(optim, linconstraint, A*W̃var .≤ b)
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
        for i in eachindex(con.X̂min)
            sym = Symbol("g_X̂min_$i")
            register(optim, sym, nvar, gfunc[i], autodiff=true)
        end
        for i in eachindex(con.X̂max)
            sym = Symbol("g_X̂max_$i")
            register(optim, sym, nvar, gfunc[i_end_X̂min+i], autodiff=true)
        end
    end
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
    model, optim, con = estim.model, estim.optim, estim.con
    nx̂, He = estim.nx̂, estim.He
    nX̂con = nx̂*(He+1)
    notSolvedYet = (termination_status(optim) == OPTIMIZE_NOT_CALLED)
    isnothing(X̂min) && !isnothing(x̂min) && (X̂min = repeat(x̂min, He+1))
    isnothing(X̂max) && !isnothing(x̂max) && (X̂max = repeat(x̂max, He+1))
    if !isnothing(X̂min)
        size(X̂min) == (nX̂con,) || throw(ArgumentError("X̂min size must be $((nX̂con,))"))
        con.x̂min[:] = X̂min[1:nx̂]
        con.X̂min[:] = X̂min[nx̂+1:end]
    end
    if !isnothing(X̂max)
        size(X̂max) == (nX̂con,) || throw(ArgumentError("X̂max size must be $((nX̂con,))"))
        con.x̂max[:] = X̂max[1:nx̂]
        con.X̂max[:] = X̂max[nx̂+1:end]
    end
    i_X̂min, i_X̂max  = .!isinf.(con.X̂min)  , .!isinf.(con.X̂max)
    i_g = [i_X̂min; i_X̂max]
    i_x̂min, i_x̂max  = .!isinf.(con.x̂min)  , .!isinf.(con.x̂max)
    i_b = [i_x̂min; i_x̂max]
    if notSolvedYet
        con.i_g[:], con.i_b[:] = i_g, i_b
        A = con.A[con.i_b, :]
        b = con.b[con.i_b]
        W̃var = optim[:W̃var]
        delete(optim, optim[:linconstraint])
        unregister(optim, :linconstraint)
        @constraint(optim, linconstraint, A*W̃var .≤ b)
        setnonlincon!(estim, model)
    else
        if i_b ≠ con.i_b || i_g ≠ con.i_g
            error("Cannot modify ±Inf constraints after calling updatestate!")
        end
    end
    return estim
end

"Set the nonlinear constraints on the output predictions `Ŷ` and terminal states `x̂end`."
function setnonlincon!(estim::MovingHorizonEstimator, ::SimModel)
    optim, con = estim.optim, estim.con
    W̃var = optim[:W̃var]
    map(con -> delete(optim, con), all_nonlinear_constraints(optim))
    for i in findall(.!isinf.(con.X̂min))
        f_sym = Symbol("g_X̂min_$(i)")
        add_nonlinear_constraint(optim, :($(f_sym)($(W̃var...)) <= 0))
    end
    for i in findall(.!isinf.(con.X̂max))
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

"Reset `estim.P̂arr_old`, `estim.invP̄` and the windows for the moving horizon estimator."
function init_estimate_cov!(estim::MovingHorizonEstimator, _ , _ , _ ) 
    estim.invP̄.data[:]    = Hermitian(inv(estim.P̂0), :L)
    estim.P̂arr_old.data[:]    = estim.P̂0
    estim.x̂arr_old           .= 0
    estim.W̃                  .= 0
    estim.X̂                  .= 0
    estim.Ym                 .= 0
    estim.U                  .= 0
    estim.D                  .= 0
    estim.Ŵ                  .= 0
    estim.Nk                 .= 0
    return nothing
end

@doc raw"""
    update_estimate!(estim::MovingHorizonEstimator, u, ym, d)
    
Update [`MovingHorizonEstimator`](@ref) state `estim.x̂`.

The optimization problem of [`MovingHorizonEstimator`](@ref) documentation is solved at
each discrete time ``k``. Once solved, the next estimate ``\mathbf{x̂}_k(k+1)`` is computed
by inserting the optimal values of ``\mathbf{x̂}_k(k-N_k+1)`` and ``\mathbf{Ŵ}`` in the
augmented model from ``j = N_k-1`` to ``0`` inclusively. Afterward, if ``k ≥ H_e``, the
arrival covariance for the next time step ``\mathbf{P̂}_{k-N_k+1}(k-N_k+2)`` is estimated
with the equations of [`update_estimate!(::ExtendedKalmanFilter)`](@ref), or `KalmanFilter`,
for `LinModel`.
"""
function update_estimate!(estim::MovingHorizonEstimator{NT}, u, ym, d) where NT<:Real
    model, optim, x̂ = estim.model, estim.optim, estim.x̂
    nx̂, nym, nu, nd, nŵ, He = estim.nx̂, estim.nym, model.nu, model.nd, estim.nx̂, estim.He
    # ------ add new data to the windows -------------
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
    # ---------- update linear constraints ------------
    estim.con.b[:] = [
        -estim.con.x̂min
        +estim.con.x̂max 
    ]
    lincon = estim.optim[:linconstraint]
    set_normalized_rhs.(lincon, estim.con.b[estim.con.i_b])
    # ----------- initial guess -----------------------
    Nk = estim.Nk[]
    nŴ, nYm, nX̂ = nx̂*Nk, nym*Nk, nx̂*Nk
    W̃var::Vector{VariableRef} = optim[:W̃var]
    Ŷm = Vector{NT}(undef, nYm)
    X̂  = Vector{NT}(undef, nX̂)
    estim.x̂arr_old[:] = estim.X̂[1:nx̂]
    W̃0 = [estim.x̂arr_old; estim.Ŵ]
    Ŷm, X̂ = predict!(Ŷm, X̂, estim, model, W̃0)
    J0 = obj_nonlinprog(estim, model, Ŷm, W̃0)
    # initial W̃0 with Ŵ=0 if objective or constraint function not finite :
    isfinite(J0) || (W̃0 = [estim.x̂arr_old; zeros(NT, nŴ)])
    set_start_value.(W̃var, W̃0)
    # ------- solve optimization problem --------------
    # at start, when time windows are not filled, some decision variables are fixed at 0:
    unfix.(W̃var[is_fixed.(W̃var)])
    fix.(W̃var[(1 + nx̂*(Nk+1)):end], 0) 
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
    # -------- error handling -------------------------
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
    # --------- update estimate -----------------------
    estim.Ŵ[1:nŴ] = estim.W̃[nx̂+1:nx̂+nŴ] # update Ŵ with optimum for next time step
    Ŷm, X̂ = predict!(Ŷm, X̂, estim, model, estim.W̃)
    x̂[:] = X̂[(1 + nx̂*(Nk-1)):(nx̂*Nk)]
    if Nk == He
        uarr, ymarr, darr = estim.U[1:nu], estim.Ym[1:nym], estim.D[1:nd]
        update_cov!(estim.P̂arr_old, estim, model, uarr, ymarr, darr)
        estim.invP̄.data[:] = Hermitian(inv(estim.P̂arr_old), :L)
    end
    return nothing
end

"Update the covariance `P̂` with the `KalmanFilter` if `model` is a `LinModel`."
function update_cov!(P̂, estim::MovingHorizonEstimator, ::LinModel, u, ym, d) 
    return update_estimate_kf!(estim, u, ym, d, estim.Â, estim.Ĉ[estim.i_ym, :], P̂)
end
"Update it with the `ExtendedKalmanFilter` if model is not a `LinModel`."
function update_cov!(P̂, estim::MovingHorizonEstimator, ::SimModel, u, ym, d) 
    # TODO: also support UnscentedKalmanFilter
    F̂ = ForwardDiff.jacobian(x̂ -> f̂(estim, estim.model, x̂, u, d), estim.x̂)
    Ĥ = ForwardDiff.jacobian(x̂ -> ĥ(estim, estim.model, x̂, d), estim.x̂)
    return update_estimate_kf!(estim, u, ym, d, F̂, Ĥ[estim.i_ym, :],  P̂)
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
    x̂arr = @views W̃[1:nx̂] # W̃ = [x̂(k-Nk+1|k); Ŵ]
    x̄ = x̂arr - estim.x̂arr_old
    V̂ = @views estim.Ym[1:nYm] - Ŷm[1:nYm]
    Ŵ = @views W̃[nx̂+1:nx̂+nŴ]
    return dot(x̄, invP̄, x̄) + dot(Ŵ, invQ̂_Nk, Ŵ) + dot(V̂, invR̂_Nk, V̂)
end

"""
    predict!(Ŷm, X̂, estim::MovingHorizonEstimator, model::SimModel, W̃) -> Ŷm, X̂

Compute the predicted measured outputs `Ŷm` and states `X̂` for the `MovingHorizonEstimator`.

The method mutates `Ŷm` and `X̂` vector arguments. Note that `Ŷm` goes from ``k-N_k+1`` to 
``k``, and `X̂`, from ``k-N_k+2`` to ``k+1``.
"""
function predict!(
    Ŷm, X̂, estim::MovingHorizonEstimator, model::SimModel, W̃::Vector{T}
) where {T<:Real}
    nu, nd, nx̂, nym, Nk = model.nu, model.nd, estim.nx̂, estim.nym, estim.Nk[]
    x̂ = @views W̃[1:nx̂] # W̃ = [x̂(k-Nk+1|k); Ŵ]
    for j=1:Nk
        u = @views estim.U[(1 + nu*(j-1)):(nu*j)]
        d = @views estim.D[(1 + nd*(j-1)):(nd*j)]
        ŵ = @views W̃[(1 + nx̂*j):(nx̂*(j+1))]
        Ŷm[ (1 + nym*(j-1)):(nym*j)] = ĥ(estim, model, x̂, d)[estim.i_ym]
        X̂[  (1 + nx̂ *(j-1)):(nx̂ *j)] = f̂(estim, model, x̂, u, d) + ŵ
        x̂ = @views X̂[(1 + nx̂*(j-1)):(nx̂*j)]
    end
    return Ŷm, X̂
end

"""
    con_nonlinprog!(g, estim::MovingHorizonEstimator, model::SimModel, X̂)

Nonlinear constrains for [`MovingHorizonEstimator`](@ref).
"""
function con_nonlinprog!(g, estim::MovingHorizonEstimator, ::SimModel, X̂)
    nX̂con, nX̂ = length(estim.con.X̂min), estim.nx̂*estim.Nk[]
    for i in eachindex(g)
        estim.con.i_g[i] || continue
        if i ≤ nX̂con
            j = i
            if j ≤ nX̂
                g[i] = estim.con.X̂min[j] - X̂[j]
            end
        else
            j = i - nX̂con
            if j ≤ nX̂
                g[i] = X̂[j] - estim.con.X̂max[j]
            end
        end
    end
    return g
end