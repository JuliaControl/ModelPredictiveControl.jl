const DEFAULT_NONLINMPC_OPTIMIZER = optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes")
const DEFAULT_NONLINMPC_TRANSCRIPTION = SingleShooting()

struct NonLinMPC{
    NT<:Real, 
    SE<:StateEstimator,
    TM<:TranscriptionMethod,
    JM<:JuMP.GenericModel, 
    PT<:Any,
    JEfunc<:Function,
    GCfunc<:Function
} <: PredictiveController{NT}
    estim::SE
    transcription::TM
    # note: `NT` and the number type `JNT` in `JuMP.GenericModel{JNT}` can be
    # different since solvers that support non-Float64 are scarce.
    optim::JM
    con::ControllerConstraint{NT, GCfunc}
    ΔŨ::Vector{NT}
    ŷ ::Vector{NT}
    Hp::Int
    Hc::Int
    nϵ::Int
    weights::ControllerWeights{NT}
    JE::JEfunc
    p::PT
    R̂u::Vector{NT}
    R̂y::Vector{NT}
    P̃::Matrix{NT}
    S̃::Matrix{NT}
    T::Matrix{NT}
    T_lastu::Vector{NT}
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
    D̂e::Vector{NT}
    Uop::Vector{NT}
    Yop::Vector{NT}
    Dop::Vector{NT}
    buffer::PredictiveControllerBuffer{NT}
    function NonLinMPC{NT}(
        estim::SE, 
        Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt, Ewt, JE::JEfunc, gc!::GCfunc, nc, p::PT, 
        transcription::TM, optim::JM
    ) where {
            NT<:Real, 
            SE<:StateEstimator, 
            TM<:TranscriptionMethod,
            JM<:JuMP.GenericModel,
            PT<:Any,
            JEfunc<:Function, 
            GCfunc<:Function, 
        }
        model = estim.model
        nu, ny, nd, nx̂ = model.nu, model.ny, model.nd, estim.nx̂
        ŷ = copy(model.yop) # dummy vals (updated just before optimization)
        weights = ControllerWeights{NT}(model, Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt, Ewt)
        # dummy vals (updated just before optimization):
        R̂y, R̂u, T_lastu = zeros(NT, ny*Hp), zeros(NT, nu*Hp), zeros(NT, nu*Hp)
        P = init_ZtoΔU(estim, transcription, Hp, Hc)
        S, T = init_ZtoU(estim, transcription, Hp, Hc)
        E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂ = init_predmat(
            model, estim, transcription, Hp, Hc
        )
        Eŝ, Gŝ, Jŝ, Kŝ, Vŝ, Bŝ = init_defectmat(model, estim, transcription, Hp, Hc)
        # dummy vals (updated just before optimization):
        F, fx̂, Fŝ  = zeros(NT, ny*Hp), zeros(NT, nx̂), zeros(NT, nx̂*Hp)
        con, nϵ, P̃, S̃, Ẽ, Ẽŝ = init_defaultcon_mpc(
            estim, Hp, Hc, Cwt, P, S, E, 
            ex̂, fx̂, gx̂, jx̂, kx̂, vx̂, bx̂, 
            Eŝ, Fŝ, Gŝ, Jŝ, Kŝ, Vŝ, Bŝ,
            gc!, nc
        )
        H̃ = init_quadprog(model, weights, Ẽ, S̃)
        # dummy vals (updated just before optimization):
        q̃, r = zeros(NT, size(H̃, 1)), zeros(NT, 1)
        Ks, Ps = init_stochpred(estim, Hp)
        # dummy vals (updated just before optimization):
        d0, D̂0, D̂e = zeros(NT, nd), zeros(NT, nd*Hp), zeros(NT, nd + nd*Hp)
        Uop, Yop, Dop = repeat(model.uop, Hp), repeat(model.yop, Hp), repeat(model.dop, Hp)
        test_custom_functions(NT, model, JE, gc!, nc, Uop, Yop, Dop, p)
        nΔŨ = size(Ẽ, 2)
        ΔŨ = zeros(NT, nΔŨ)
        buffer = PredictiveControllerBuffer{NT}(nu, ny, nd, Hp, Hc, nϵ)
        mpc = new{NT, SE, TM, JM, PT, JEfunc, GCfunc}(
            estim, transcription, optim, con,
            ΔŨ, ŷ,
            Hp, Hc, nϵ,
            weights,
            JE, p,
            R̂u, R̂y,
            P̃, S̃, T, T_lastu,
            Ẽ, F, G, J, K, V, B,
            H̃, q̃, r,
            Ks, Ps,
            d0, D̂0, D̂e,
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
\min_{\mathbf{Z}, ϵ}\ & \mathbf{(R̂_y - Ŷ)}' \mathbf{M}_{H_p} \mathbf{(R̂_y - Ŷ)}   
                       + \mathbf{(ΔU)}'      \mathbf{N}_{H_c} \mathbf{(ΔU)}        \\&
                       + \mathbf{(R̂_u - U)}' \mathbf{L}_{H_p} \mathbf{(R̂_u - U)} 
                       + C ϵ^2  
                       + E J_E(\mathbf{U_e}, \mathbf{Ŷ_e}, \mathbf{D̂_e}, \mathbf{p})
\end{aligned}
```
subject to [`setconstraint!`](@ref) bounds, and the custom inequality constraints:
```math
\mathbf{g_c}(\mathbf{U_e}, \mathbf{Ŷ_e}, \mathbf{D̂_e}, \mathbf{p}, ϵ) ≤ \mathbf{0}
```
with the decision variables ``\mathbf{Z}`` and slack ``ϵ``. By default, a [`SingleShooting`](@ref)
transcription method is used, hence ``\mathbf{Z=ΔU}``. The economic function ``J_E`` can
penalizes solutions with high economic costs. Setting all the weights to 0 except ``E``
creates a pure economic model predictive controller (EMPC). As a matter of fact, ``J_E`` can
be any nonlinear function as a custom objective, even if there is no economic interpretation
to it. The arguments of ``J_E`` and ``\mathbf{g_c}`` include the manipulated inputs,
predicted outputs and measured disturbances, extended from ``k`` to ``k+H_p`` (inclusively,
see Extended Help for more details):
```math
    \mathbf{U_e} = \begin{bmatrix} \mathbf{U}      \\ \mathbf{u}(k+H_p-1)   \end{bmatrix}  , \quad
    \mathbf{Ŷ_e} = \begin{bmatrix} \mathbf{ŷ}(k)   \\ \mathbf{Ŷ}            \end{bmatrix}  , \quad
    \mathbf{D̂_e} = \begin{bmatrix} \mathbf{d}(k)   \\ \mathbf{D̂}            \end{bmatrix}
```
The argument ``\mathbf{p}`` is a custom parameter object of any type, but use a mutable one
if you want to modify it later e.g.: a vector. See [`LinMPC`](@ref) Extended Help for the
definition of the other variables.

!!! tip
    Replace any of the arguments of ``J_E`` and ``\mathbf{g_c}`` functions with `_` if not
    needed (see e.g. the default value of `JE` below).

This method uses the default state estimator:

- if `model` is a [`LinModel`](@ref), a [`SteadyKalmanFilter`](@ref) with default arguments;
- else, an [`UnscentedKalmanFilter`](@ref) with default arguments. 

This controller allocates memory at each time step for the optimization.

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
- `JE=(_,_,_,_)->0.0` : economic or custom cost function ``J_E(\mathbf{U_e}, \mathbf{Ŷ_e},
   \mathbf{D̂_e}, \mathbf{p})``.
- `gc=(_,_,_,_,_,_)->nothing` or `gc!` : custom inequality constraint function 
   ``\mathbf{g_c}(\mathbf{U_e}, \mathbf{Ŷ_e}, \mathbf{D̂_e}, \mathbf{p}, ϵ)``, mutating or 
   not (details in Extended Help).
- `nc=0` : number of custom inequality constraints.
- `p=model.p` : ``J_E`` and ``\mathbf{g_c}`` functions parameter ``\mathbf{p}`` (any type).
- `transcription=SingleShooting()` : a [`TranscriptionMethod`](@ref) for the optimization.
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

    The economic cost ``J_E`` and custom constraint ``\mathbf{g_c}`` functions receive the
    extended vectors ``\mathbf{U_e}`` (`nu*Hp+nu` elements), ``\mathbf{Ŷ_e}`` (`ny+ny*Hp`
    elements) and  ``\mathbf{D̂_e}`` (`nd+nd*Hp` elements) as arguments. They all include the
    values from ``k`` to ``k + H_p`` (inclusively). The custom constraint also receives the
    slack ``ϵ`` (scalar), which is always zero if `Cwt=Inf`.
    
    More precisely, the last two time steps in ``\mathbf{U_e}`` are forced to be equal, i.e.
    ``\mathbf{u}(k+H_p) = \mathbf{u}(k+H_p-1)``, since ``H_c ≤ H_p`` implies that
    ``\mathbf{Δu}(k+H_p) = \mathbf{0}``. The vectors ``\mathbf{ŷ}(k)`` and ``\mathbf{d}(k)``
    are the current state estimator output and measured disturbance, respectively, and 
    ``\mathbf{Ŷ}`` and ``\mathbf{D̂}``, their respective predictions from ``k+1`` to ``k+H_p``. 
    If `LHS` represents the result of the left-hand side in the inequality 
    ``\mathbf{g_c}(\mathbf{U_e}, \mathbf{Ŷ_e}, \mathbf{D̂_e}, \mathbf{p}, ϵ) ≤ \mathbf{0}``,
    the function `gc` can be implemented in two possible ways:
    
    1. **Non-mutating function** (out-of-place): define it as `gc(Ue, Ŷe, D̂e, p, ϵ) -> LHS`.
       This syntax is simple and intuitive but it allocates more memory.
    2. **Mutating function** (in-place): define it as `gc!(LHS, Ue, Ŷe, D̂e, p, ϵ) -> nothing`.
       This syntax reduces the allocations and potentially the computational burden as well.

    The keyword argument `nc` is the number of elements in `LHS`, and `gc!`, an alias for
    the `gc` argument (both `gc` and `gc!` accepts non-mutating and mutating functions). 
    
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
    JE ::Function = (_,_,_,_) -> 0.0,
    gc!::Function = (_,_,_,_,_,_) -> nothing,
    gc ::Function = gc!,
    nc::Int = 0,
    p = model.p,
    transcription::TranscriptionMethod = DEFAULT_NONLINMPC_TRANSCRIPTION,
    optim::JuMP.GenericModel = JuMP.Model(DEFAULT_NONLINMPC_OPTIMIZER, add_bridges=false),
    kwargs...
)
    estim = UnscentedKalmanFilter(model; kwargs...)
    return NonLinMPC(
        estim; 
        Hp, Hc, Mwt, Nwt, Lwt, Cwt, Ewt, JE, gc, nc, p, M_Hp, N_Hc, L_Hp, 
        transcription, optim
    )
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
    JE ::Function = (_,_,_,_) -> 0.0,
    gc!::Function = (_,_,_,_,_,_) -> nothing,
    gc ::Function = gc!,
    nc::Int = 0,
    p = model.p,
    transcription::TranscriptionMethod = DEFAULT_NONLINMPC_TRANSCRIPTION,
    optim::JuMP.GenericModel = JuMP.Model(DEFAULT_NONLINMPC_OPTIMIZER, add_bridges=false),
    kwargs...
)
    estim = SteadyKalmanFilter(model; kwargs...)
    return NonLinMPC(
        estim; 
        Hp, Hc, Mwt, Nwt, Lwt, Cwt, Ewt, JE, gc, nc, p, M_Hp, N_Hc, L_Hp, 
        transcription, optim
    )
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
    JE ::Function = (_,_,_,_) -> 0.0,
    gc!::Function = (_,_,_,_,_,_) -> nothing,
    gc ::Function = gc!,
    nc = 0,
    p = estim.model.p,
    transcription::TranscriptionMethod = DEFAULT_NONLINMPC_TRANSCRIPTION,
    optim::JuMP.GenericModel = JuMP.Model(DEFAULT_NONLINMPC_OPTIMIZER, add_bridges=false),
) where {
    NT<:Real, 
    SE<:StateEstimator{NT}
}
    nk = estimate_delays(estim.model)
    if Hp ≤ nk
        @warn("prediction horizon Hp ($Hp) ≤ estimated number of delays in model "*
              "($nk), the closed-loop system may be unstable or zero-gain (unresponsive)")
    end
    validate_JE(NT, JE)
    gc! = get_mutating_gc(NT, gc)
    return NonLinMPC{NT}(
        estim, Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt, Ewt, JE, gc!, nc, p, 
        transcription, optim
    )
end

"""
    validate_JE(NT, JE) -> nothing

Validate `JE` function argument signature.
"""
function validate_JE(NT, JE)
    #                       Ue,         Ŷe,         D̂e,         p
    if !hasmethod(JE, Tuple{Vector{NT}, Vector{NT}, Vector{NT}, Any})
        error(
            "the economic cost function has no method with type signature "*
            "JE(Ue::Vector{$(NT)}, Ŷe::Vector{$(NT)}, D̂e::Vector{$(NT)}, p::Any)"
        )
    end
    return nothing
end

"""
    validate_gc(NT, gc) -> ismutating

Validate `gc` function argument signature and return `true` if it is mutating.
"""
function validate_gc(NT, gc)
    ismutating = hasmethod(
        gc, 
        #     LHS,        Ue,         Ŷe,         D̂e,         p,   ϵ
        Tuple{Vector{NT}, Vector{NT}, Vector{NT}, Vector{NT}, Any, NT}
    )
    #                                      Ue,         Ŷe,         D̂e,         p,   ϵ
    if !(ismutating || hasmethod(gc, Tuple{Vector{NT}, Vector{NT}, Vector{NT}, Any, NT}))
        error(
            "the custom constraint function has no method with type signature "*
            "gc(Ue::Vector{$(NT)}, Ŷe::Vector{$(NT)}, D̂e::Vector{$(NT)}, p::Any, ϵ::$(NT)) "*
            "or mutating form gc!(LHS::Vector{$(NT)}, Ue::Vector{$(NT)}, Ŷe::Vector{$(NT)}, "*
            "D̂e::Vector{$(NT)}, p::Any, ϵ::$(NT))"
        )
    end
    return ismutating
end

"Get mutating custom constraint function `gc!` from the provided function in argument."
function get_mutating_gc(NT, gc)
    ismutating_gc = validate_gc(NT, gc)
    gc! = if ismutating_gc
        gc
    else
        function gc!(LHS, Ue, Ŷe, D̂e, p, ϵ)
            LHS .= gc(Ue, Ŷe, D̂e, p, ϵ)
            return nothing
        end
    end
    return gc!
end

"""
    test_custom_functions(NT, model::SimModel, JE, gc!, nc, Uop, Yop, Dop, p)

Test the custom functions `JE` and `gc!` at the operating point `Uop`, `Yop`, `Dop`.

This function is called at the end of `NonLinMPC` construction. It warns the user if the
custom cost `JE` and constraint `gc!` functions crash at `model` operating points. This
should ease troubleshooting of simple bugs e.g.: the user forgets to set the `nc` argument.
"""
function test_custom_functions(NT, model::SimModel, JE, gc!, nc, Uop, Yop, Dop, p)
    uop, dop, yop = model.uop, model.dop, model.yop
    Ue, Ŷe, D̂e = [Uop; uop], [yop; Yop], [dop; Dop]
    try
        val::NT = JE(Ue, Ŷe, D̂e, p)
    catch err
        @warn(
            """
            Calling the JE function with Ue, Ŷe, D̂e arguments fixed at uop=$uop, 
            yop=$yop, dop=$dop failed with the following stacktrace. Did you forget
            to set the keyword argument p?
            """, 
            exception=(err, catch_backtrace())
        )
    end
    ϵ, gc = zero(NT), Vector{NT}(undef, nc) 
    try
        gc!(gc, Ue, Ŷe, D̂e, p, ϵ)
    catch err
        @warn(
            """
            Calling the gc function with Ue, Ŷe, D̂e, ϵ arguments fixed at uop=$uop,
            yop=$yop, dop=$dop, ϵ=0 failed with the following stacktrace. Did you 
            forget to set the keyword argument p or nc?
            """, 
            exception=(err, catch_backtrace())
        )
    end
    return nothing
end

"""
    addinfo!(info, mpc::NonLinMPC) -> info

For [`NonLinMPC`](@ref), add `:sol` and the optimal economic cost `:JE`.
"""
function addinfo!(info, mpc::NonLinMPC{NT}) where NT<:Real
    U, Ŷ, D̂, ŷ, d, ϵ = info[:U], info[:Ŷ], info[:D̂], info[:ŷ], info[:d], info[:ϵ]
    Ue = [U; U[(end - mpc.estim.model.nu + 1):end]]
    Ŷe = [ŷ; Ŷ]
    D̂e = [d; D̂] 
    JE = mpc.JE(Ue, Ŷe, D̂e, mpc.p)
    LHS = Vector{NT}(undef, mpc.con.nc)
    mpc.con.gc!(LHS, Ue, Ŷe, D̂e, mpc.p, ϵ)
    info[:JE]  = JE 
    info[:gc] = LHS
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
        C = mpc.weights.Ñ_Hc[end]
        try
            JuMP.get_attribute(optim, "nlp_scaling_max_gradient")
        catch
            # default "nlp_scaling_max_gradient" to `10.0/C` if not already set:
            JuMP.set_attribute(optim, "nlp_scaling_max_gradient", 10.0/C)
        end
    end
    Jfunc, gfuncs = get_optim_functions(mpc, mpc.optim)
    @operator(optim, J, nΔŨ, Jfunc)
    @objective(optim, Min, J(ΔŨvar...))
    init_nonlincon!(mpc, model, gfuncs)
    set_nonlincon!(mpc, model, mpc.optim)
    return nothing
end

"""
    get_optim_functions(mpc::NonLinMPC, ::JuMP.GenericModel) -> Jfunc, gfuncs

Get the objective `Jfunc` function and constraint `gfuncs` function vector for 
[`NonLinMPC`](@ref).

Inspired from: [User-defined operators with vector outputs](https://jump.dev/JuMP.jl/stable/tutorials/nonlinear/tips_and_tricks/#User-defined-operators-with-vector-outputs)
"""
function get_optim_functions(mpc::NonLinMPC, ::JuMP.GenericModel{JNT}) where JNT<:Real
    model = mpc.estim.model
    nu, ny, nx̂, nϵ, Hp = model.nu, model.ny, mpc.estim.nx̂, mpc.nϵ, mpc.Hp
    ng, nc, nΔŨ, nU, nŶ = length(mpc.con.i_g), mpc.con.nc, length(mpc.ΔŨ), Hp*nu, Hp*ny
    nUe, nŶe = nU + nu, nŶ + ny
    Ncache = nΔŨ + 3
    ΔŨ_cache::DiffCache{Vector{JNT}, Vector{JNT}}     = DiffCache(zeros(JNT, nΔŨ), Ncache)
    Ŷe_cache::DiffCache{Vector{JNT}, Vector{JNT}}     = DiffCache(zeros(JNT, nŶe), Ncache)
    Ue_cache::DiffCache{Vector{JNT}, Vector{JNT}}     = DiffCache(zeros(JNT, nUe), Ncache)
    Ȳ_cache::DiffCache{Vector{JNT}, Vector{JNT}}      = DiffCache(zeros(JNT, nŶ),  Ncache)
    Ū_cache::DiffCache{Vector{JNT}, Vector{JNT}}      = DiffCache(zeros(JNT, nU),  Ncache)
    x̂0_cache::DiffCache{Vector{JNT}, Vector{JNT}}     = DiffCache(zeros(JNT, nx̂),  Ncache)
    x̂0next_cache::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, nx̂),  Ncache)
    u0_cache::DiffCache{Vector{JNT}, Vector{JNT}}     = DiffCache(zeros(JNT, nu),  Ncache)
    û0_cache::DiffCache{Vector{JNT}, Vector{JNT}}     = DiffCache(zeros(JNT, nu),  Ncache)
    g_cache::DiffCache{Vector{JNT}, Vector{JNT}}      = DiffCache(zeros(JNT, ng),  Ncache)
    gc_cache::DiffCache{Vector{JNT}, Vector{JNT}}     = DiffCache(zeros(JNT, nc),  Ncache)
    function update_simulations!(ΔŨ, ΔŨtup::NTuple{N, T}) where {N, T<:Real}
        if any(new !== old for (new, old) in zip(ΔŨtup, ΔŨ)) # new ΔŨtup, update predictions:
            ΔŨ1 = ΔŨtup[begin]
            for i in eachindex(ΔŨtup)
                ΔŨ[i] = ΔŨtup[i] # ΔŨ .= ΔŨtup seems to produce a type instability
            end
            Ŷe, Ue     = get_tmp(Ŷe_cache, ΔŨ1), get_tmp(Ue_cache, ΔŨ1)
            Ȳ,  Ū      = get_tmp(Ȳ_cache, ΔŨ1),  get_tmp(Ū_cache, ΔŨ1)
            x̂0, x̂0next = get_tmp(x̂0_cache, ΔŨ1), get_tmp(x̂0next_cache, ΔŨ1)
            u0, û0     = get_tmp(u0_cache, ΔŨ1), get_tmp(û0_cache, ΔŨ1)
            gc, g      = get_tmp(gc_cache, ΔŨ1), get_tmp(g_cache, ΔŨ1)
            Ŷ0, x̂0end  = predict!(Ȳ, x̂0, x̂0next, u0, û0, mpc, model, ΔŨ)
            Ŷe, Ue     = extended_predictions!(Ŷe, Ue, Ū, mpc, model, Ŷ0, ΔŨ)
            ϵ = (nϵ ≠ 0) ? ΔŨ[end] : zero(T) # ϵ = 0 if nϵ == 0 (meaning no relaxation)
            gc = con_custom!(gc, mpc, Ue, Ŷe, ϵ)
            g  = con_nonlinprog!(g, mpc, model, x̂0end, Ŷ0, gc, ϵ)
        end
        return nothing
    end
    function Jfunc(ΔŨtup::Vararg{T, N}) where {N, T<:Real}
        ΔŨ1 = ΔŨtup[begin]
        ΔŨ = get_tmp(ΔŨ_cache, ΔŨ1)
        update_simulations!(ΔŨ, ΔŨtup)
        Ȳ,  Ū  = get_tmp(Ȳ_cache, ΔŨ1),  get_tmp(Ū_cache, ΔŨ1)
        Ŷe, Ue = get_tmp(Ŷe_cache, ΔŨ1), get_tmp(Ue_cache, ΔŨ1)
        return obj_nonlinprog!(Ȳ, Ū, mpc, model, Ue, Ŷe, ΔŨ)::T
    end
    function gfunc_i(i, ΔŨtup::NTuple{N, T}) where {N, T<:Real}
        ΔŨ1 = ΔŨtup[begin]
        ΔŨ = get_tmp(ΔŨ_cache, ΔŨ1)
        update_simulations!(ΔŨ, ΔŨtup)
        g = get_tmp(g_cache, ΔŨ1)
        return g[i]::T
    end
    gfuncs = Vector{Function}(undef, ng)
    for i in 1:ng
        # this is another syntax for anonymous function, allowing parameters T and N:
        gfuncs[i] = function (ΔŨtup::Vararg{T, N}) where {N, T<:Real}
            return gfunc_i(i, ΔŨtup)
        end
    end
    return Jfunc, gfuncs
end

function init_nonlincon!(mpc::NonLinMPC, ::LinModel, gfuncs::Vector{<:Function}) 
    optim, con = mpc.optim, mpc.con
    nΔŨ = length(mpc.ΔŨ)
    if length(con.i_g) ≠ 0
        i_base = 0
        for i in 1:con.nc
            name = Symbol("g_c_$i")
            optim[name] = JuMP.add_nonlinear_operator(optim, nΔŨ, gfuncs[i_base+i]; name)
        end
    end
    return nothing
end

function init_nonlincon!(mpc::NonLinMPC, ::NonLinModel, gfuncs::Vector{<:Function}) 
    optim, con = mpc.optim, mpc.con
    ny, nx̂, Hp, nΔŨ = mpc.estim.model.ny, mpc.estim.nx̂, mpc.Hp, length(mpc.ΔŨ)
    if length(con.i_g) ≠ 0
        i_base = 0
        for i in eachindex(con.Y0min)
            name = Symbol("g_Y0min_$i")
            optim[name] = JuMP.add_nonlinear_operator(optim, nΔŨ, gfuncs[i_base+i]; name)
        end
        i_base = 1Hp*ny
        for i in eachindex(con.Y0max)
            name = Symbol("g_Y0max_$i")
            optim[name] = JuMP.add_nonlinear_operator(optim, nΔŨ, gfuncs[i_base+i]; name)
        end
        i_base = 2Hp*ny
        for i in eachindex(con.x̂0min)
            name = Symbol("g_x̂0min_$i")
            optim[name] = JuMP.add_nonlinear_operator(optim, nΔŨ, gfuncs[i_base+i]; name)
        end
        i_base = 2Hp*ny + nx̂
        for i in eachindex(con.x̂0max)
            name = Symbol("g_x̂0max_$i")
            optim[name] = JuMP.add_nonlinear_operator(optim, nΔŨ, gfuncs[i_base+i]; name)
        end
        i_base = 2Hp*ny + 2nx̂
        for i in 1:con.nc
            name = Symbol("g_c_$i")
            optim[name] = JuMP.add_nonlinear_operator(optim, nΔŨ, gfuncs[i_base+i]; name)
        end
    end
    return nothing
end

"""
    set_nonlincon!(mpc::NonLinMPC, ::LinModel, optim)

Set the custom nonlinear inequality constraints for `LinModel`.
"""
function set_nonlincon!(
    mpc::NonLinMPC, ::LinModel, optim::JuMP.GenericModel{JNT}
) where JNT<:Real
    ΔŨvar = optim[:ΔŨvar]
    con = mpc.con
    nonlin_constraints = JuMP.all_constraints(optim, JuMP.NonlinearExpr, MOI.LessThan{JNT})
    map(con_ref -> JuMP.delete(optim, con_ref), nonlin_constraints)
    for i in 1:con.nc
        gfunc_i = optim[Symbol("g_c_$i")]
        @constraint(optim, gfunc_i(ΔŨvar...) <= 0)
    end
    return nothing
end

"""
    set_nonlincon!(mpc::NonLinMPC, ::NonLinModel, optim)

Also set output prediction `Ŷ` and terminal state `x̂end` constraints when not a `LinModel`.
"""
function set_nonlincon!(
    mpc::NonLinMPC, ::SimModel, optim::JuMP.GenericModel{JNT}
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
    for i in 1:con.nc
        gfunc_i = optim[Symbol("g_c_$i")]
        @constraint(optim, gfunc_i(ΔŨvar...) <= 0)
    end
    return nothing
end

"""
    con_nonlinprog!(g, mpc::NonLinMPC, model::LinModel, _ , _ , gc, ϵ) -> g

Nonlinear constrains for [`NonLinMPC`](@ref) when `model` is a [`LinModel`](@ref).

The method mutates the `g` vectors in argument and returns it. Only the custom constraints
are include in the `g` vector.
"""
function con_nonlinprog!(g, mpc::NonLinMPC, ::LinModel, _ , _ , gc, ϵ)
    for i in eachindex(g)
        g[i] = gc[i]
    end
    return g
end

"""
    con_nonlinprog!(g, mpc::NonLinMPC, model::SimModel, x̂0end, Ŷ0, gc, ϵ) -> g

Nonlinear constrains for [`NonLinMPC`](@ref) when `model` is not a [`LinModel`](@ref).

The method mutates the `g` vectors in argument and returns it. The output prediction, 
the terminal state and the custom constraints are include in the `g` vector.
"""
function con_nonlinprog!(g, mpc::NonLinMPC, ::SimModel, x̂0end, Ŷ0, gc, ϵ)
    nx̂, nŶ = length(x̂0end), length(Ŷ0)
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
        elseif i ≤ 2nŶ + 2nx̂
            j = i - 2nŶ - nx̂
            g[i] = (x̂0end[j] - mpc.con.x̂0max[j])  - ϵ*mpc.con.c_x̂max[j]
        else
            j = i - 2nŶ - 2nx̂
            g[i] = gc[j]
        end
    end
    return g
end


@doc raw"""
    con_custom!(gc, mpc::NonLinMPC, Ue, Ŷe, ϵ) -> gc

Evaluate the custom inequality constraint `gc` in-place and return it.
"""
function con_custom!(gc, mpc::NonLinMPC, Ue, Ŷe, ϵ)
    mpc.con.nc ≠ 0 && mpc.con.gc!(gc, Ue, Ŷe, mpc.D̂e, mpc.p, ϵ)
    return gc
end

"Evaluate the economic term `E*JE` of the objective function for [`NonLinMPC`](@ref)."
function obj_econ(
    mpc::NonLinMPC, model::SimModel, Ue, Ŷe::AbstractVector{NT}
) where NT<:Real
    E_JE = mpc.weights.iszero_E ? zero(NT) : mpc.weights.E*mpc.JE(Ue, Ŷe, mpc.D̂e, mpc.p)
    return E_JE
end