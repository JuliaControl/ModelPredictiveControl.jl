const DEFAULT_NONLINMPC_TRANSCRIPTION = SingleShooting()
const DEFAULT_NONLINMPC_OPTIMIZER = optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes")
const DEFAULT_NONLINMPC_GRADIENT  = AutoForwardDiff()
const DEFAULT_NONLINMPC_JACDENSE  = AutoForwardDiff()
const DEFAULT_NONLINMPC_JACSPARSE = AutoSparse(
    AutoForwardDiff();
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(ALL_COLORING_ORDERS, postprocessing=true),
)
const DEFAULT_NONLINMPC_HESSIAN = DEFAULT_NONLINMPC_JACSPARSE

struct NonLinMPC{
    NT<:Real,
    SE<:StateEstimator,
    CW<:ControllerWeights,
    TM<:TranscriptionMethod,
    JM<:JuMP.GenericModel,
    GB<:AbstractADType,
    JB<:AbstractADType,
    HB<:Union{AbstractADType, Nothing}, 
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
    gradient::GB
    jacobian::JB
    hessian::HB
    oracle::Bool
    Z̃::Vector{NT}
    ŷ::Vector{NT}
    Hp::Int
    Hc::Int
    nϵ::Int
    nb::Vector{Int}
    weights::CW
    JE::JEfunc
    p::PT
    R̂u::Vector{NT}
    R̂y::Vector{NT}
    lastu0::Vector{NT}
    P̃Δu::SparseMatrixCSC{NT, Int}
    P̃u ::SparseMatrixCSC{NT, Int}
    Tu ::SparseMatrixCSC{NT, Int}
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
        estim::SE, Hp, Hc, nb, weights::CW,
        JE::JEfunc, gc!::GCfunc, nc, p::PT, 
        transcription::TM, optim::JM, 
        gradient::GB, jacobian::JB, hessian::HB, oracle
    ) where {
            NT<:Real, 
            SE<:StateEstimator,
            CW<:ControllerWeights,
            TM<:TranscriptionMethod,
            JM<:JuMP.GenericModel,
            GB<:AbstractADType,
            JB<:AbstractADType,
            HB<:Union{AbstractADType, Nothing},
            PT<:Any,
            JEfunc<:Function, 
            GCfunc<:Function, 
        }
        model = estim.model
        nu, ny, nd, nx̂ = model.nu, model.ny, model.nd, estim.nx̂
        ŷ = copy(model.yop) # dummy vals (updated just before optimization)
        # dummy vals (updated just before optimization):
        R̂y, R̂u, Tu_lastu0 = zeros(NT, ny*Hp), zeros(NT, nu*Hp), zeros(NT, nu*Hp)
        lastu0 = zeros(NT, nu)
        validate_transcription(model, transcription)
        PΔu = init_ZtoΔU(estim, transcription, Hp, Hc)
        Pu, Tu = init_ZtoU(estim, transcription, Hp, Hc, nb)
        E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂ = init_predmat(
            model, estim, transcription, Hp, Hc
        )
        Eŝ, Gŝ, Jŝ, Kŝ, Vŝ, Bŝ = init_defectmat(model, estim, transcription, Hp, Hc)
        # dummy vals (updated just before optimization):
        F, fx̂, Fŝ  = zeros(NT, ny*Hp), zeros(NT, nx̂), zeros(NT, nx̂*Hp)
        con, nϵ, P̃Δu, P̃u, Ẽ = init_defaultcon_mpc(
            estim, weights, transcription,
            Hp, Hc, 
            PΔu, Pu, E, 
            ex̂, fx̂, gx̂, jx̂, kx̂, vx̂, bx̂, 
            Eŝ, Fŝ, Gŝ, Jŝ, Kŝ, Vŝ, Bŝ,
            gc!, nc
        )
        warn_cond = iszero(weights.E) ? 1e6 : Inf # condition number warning only if Ewt==0
        H̃ = init_quadprog(model, transcription, weights, Ẽ, P̃Δu, P̃u; warn_cond)
        # dummy vals (updated just before optimization):
        q̃, r = zeros(NT, size(H̃, 1)), zeros(NT, 1)
        Ks, Ps = init_stochpred(estim, Hp)
        # dummy vals (updated just before optimization):
        d0, D̂0, D̂e = zeros(NT, nd), zeros(NT, nd*Hp), zeros(NT, nd + nd*Hp)
        Uop, Yop, Dop = repeat(model.uop, Hp), repeat(model.yop, Hp), repeat(model.dop, Hp)
        test_custom_functions(NT, model, JE, gc!, nc, Uop, Yop, Dop, p)
        nZ̃ = get_nZ(estim, transcription, Hp, Hc) + nϵ
        Z̃ = zeros(NT, nZ̃)
        buffer = PredictiveControllerBuffer(estim, transcription, Hp, Hc, nϵ)
        mpc = new{NT, SE, CW, TM, JM, GB, JB, HB, PT, JEfunc, GCfunc}(
            estim, transcription, optim, con,
            gradient, jacobian, hessian, oracle,
            Z̃, ŷ,
            Hp, Hc, nϵ, nb,
            weights,
            JE, p,
            R̂u, R̂y,
            lastu0,
            P̃Δu, P̃u, Tu, Tu_lastu0,
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
\min_{\mathbf{Z}, ϵ}\ &  \mathbf{(R̂_y - Ŷ)}' \mathbf{M}_{H_p} \mathbf{(R̂_y - Ŷ)}   
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
- `Hp::Int=10+nk` : prediction horizon ``H_p``, `nk` is the number of delays if `model` is a
   [`LinModel`](@ref) (must be specified otherwise).
- `Hc::Union{Int, Vector{Int}}=2` : control horizon ``H_c``, custom move blocking pattern is 
   specified with a vector of integers (see [`move_blocking`](@ref) for details).
- `Mwt=fill(1.0,model.ny)` : main diagonal of ``\mathbf{M}`` weight matrix (vector).
- `Nwt=fill(0.1,model.nu)` : main diagonal of ``\mathbf{N}`` weight matrix (vector).
- `Lwt=fill(0.0,model.nu)` : main diagonal of ``\mathbf{L}`` weight matrix (vector).
- `M_Hp=Diagonal(repeat(Mwt,Hp))` : positive semidefinite symmetric matrix ``\mathbf{M}_{H_p}``.
- `N_Hc=Diagonal(repeat(Nwt,Hc))` : positive semidefinite symmetric matrix ``\mathbf{N}_{H_c}``.
- `L_Hp=Diagonal(repeat(Lwt,Hp))` : positive semidefinite symmetric matrix ``\mathbf{L}_{H_p}``.
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
   controller, provided as a [`JuMP.Model`](@extref) object (default to [`Ipopt`](https://github.com/jump-dev/Ipopt.jl) optimizer).
- `gradient=AutoForwardDiff()` : an `AbstractADType` backend for the gradient of the objective
   function, see [`DifferentiationInterface` doc](@extref DifferentiationInterface List).
- `jacobian=default_jacobian(transcription)` : an `AbstractADType` backend for the Jacobian
   of the nonlinear constraints, see `gradient` above for the options (default in Extended Help).
- `hessian=false` : an `AbstractADType` backend or `Bool` for the Hessian of the Lagrangian, 
   see `gradient` above for the options. The default `false` skip it and use the quasi-Newton
   method of `optim`, which is always the case if `oracle=false` (see Extended Help).
- `oracle=JuMP.solver_name(optim)=="Ipopt"` : a `Bool` to use the [`VectorNonlinearOracle`](@extref MathOptInterface MathOptInterface.VectorNonlinearOracle)
   for efficient nonlinear constraints (not supported by most optimizers for now).
- additional keyword arguments are passed to [`UnscentedKalmanFilter`](@ref) constructor 
  (or [`SteadyKalmanFilter`](@ref), for [`LinModel`](@ref)).

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_,_)->0.5x+u, (x,_,_)->2x, 10.0, 1, 1, 1, solver=nothing);

julia> mpc = NonLinMPC(model, Hp=20, Hc=10, transcription=MultipleShooting())
NonLinMPC controller with a sample time Ts = 10.0 s:
├ estimator: UnscentedKalmanFilter
├ model: NonLinModel
├ optimizer: Ipopt 
├ transcription: MultipleShooting
├ gradient: AutoForwardDiff
├ jacobian: AutoSparse (AutoForwardDiff, TracerSparsityDetector, GreedyColoringAlgorithm)
├ hessian: nothing
└ dimensions:
  ├ 20 prediction steps Hp
  ├ 10 control steps Hc
  ├  1 slack variable ϵ (control constraints)
  ├  1 manipulated inputs u (0 integrating states)
  ├  2 estimated states x̂
  ├  1 measured outputs ym (1 integrating states)
  ├  0 unmeasured outputs yu
  └  0 measured disturbances d
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
    
    By default, the optimization relies on dense [`ForwardDiff`](@extref ForwardDiff)
    automatic differentiation (AD) to compute the objective and constraint derivatives. Two
    exceptions: if `transcription` is not a [`SingleShooting`](@ref), the `jacobian`
    argument defaults to this [sparse backend](@extref DifferentiationInterface AutoSparse-object):
    ```julia
    AutoSparse(
        AutoForwardDiff(); 
        sparsity_detector  = TracerSparsityDetector(), 
        coloring_algorithm = GreedyColoringAlgorithm(
            (
                NaturalOrder(),
                LargestFirst(),
                SmallestLast(),
                IncidenceDegree(),
                DynamicLargestFirst(),
                RandomOrder(StableRNG(0), 0)
            ), 
        postprocessing = true
        )
    )
    ```
    that is, it will test many coloring orders at preparation and keep the best. This is
    also the sparse backend selected for the Hessian of the Lagrangian function if 
    `oracle=true` and `hessian=true`, which is the second exception. Second order 
    derivatives are only supported with `oracle=true` option.
    
    Optimizers generally benefit from exact derivatives like AD. However, the [`NonLinModel`](@ref) 
    state-space functions must be compatible with this feature. See [`JuMP` documentation](@extref JuMP Common-mistakes-when-writing-a-user-defined-operator)
    for common mistakes when writing these functions.

    Note that if `Cwt≠Inf`, the attribute `nlp_scaling_max_gradient` of `Ipopt` is set to 
    `10/Cwt` (if not already set), to scale the small values of ``ϵ``.
"""
function NonLinMPC(
    model::SimModel;
    Hp::Int = default_Hp(model),
    Hc::IntVectorOrInt = DEFAULT_HC,
    Mwt  = fill(DEFAULT_MWT, model.ny),
    Nwt  = fill(DEFAULT_NWT, model.nu),
    Lwt  = fill(DEFAULT_LWT, model.nu),
    M_Hp = Diagonal(repeat(Mwt, Hp)),
    N_Hc = Diagonal(repeat(Nwt, get_Hc(move_blocking(Hp, Hc)))),
    L_Hp = Diagonal(repeat(Lwt, Hp)),
    Cwt  = DEFAULT_CWT,
    Ewt  = DEFAULT_EWT,
    JE ::Function = (_,_,_,_) -> 0.0,
    gc!::Function = (_,_,_,_,_,_) -> nothing,
    gc ::Function = gc!,
    nc::Int = 0,
    p = model.p,
    transcription::TranscriptionMethod = DEFAULT_NONLINMPC_TRANSCRIPTION,
    optim::JuMP.GenericModel = JuMP.Model(DEFAULT_NONLINMPC_OPTIMIZER, add_bridges=false),
    gradient::AbstractADType = DEFAULT_NONLINMPC_GRADIENT,
    jacobian::AbstractADType = default_jacobian(transcription),
    hessian::Union{AbstractADType, Bool, Nothing} = false,
    oracle::Bool = JuMP.solver_name(optim)=="Ipopt",
    kwargs...
)
    estim = default_estimator(model; kwargs...)
    return NonLinMPC(
        estim; 
        Hp, Hc, Mwt, Nwt, Lwt, Cwt, Ewt, JE, gc, nc, p, M_Hp, N_Hc, L_Hp, 
        transcription, optim, gradient, jacobian, hessian, oracle
    )
end

default_estimator(model::SimModel; kwargs...) = UnscentedKalmanFilter(model; kwargs...)
default_estimator(model::LinModel; kwargs...) = SteadyKalmanFilter(model; kwargs...)

"""
    NonLinMPC(estim::StateEstimator; <keyword arguments>)

Use custom state estimator `estim` to construct `NonLinMPC`.

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_,_)->0.5x+u, (x,_,_)->2x, 10.0, 1, 1, 1, solver=nothing);

julia> estim = UnscentedKalmanFilter(model, σQint_ym=[0.05]);

julia> mpc = NonLinMPC(estim, Hp=20, Cwt=1e6)
NonLinMPC controller with a sample time Ts = 10.0 s:
├ estimator: UnscentedKalmanFilter
├ model: NonLinModel
├ optimizer: Ipopt 
├ transcription: SingleShooting
├ gradient: AutoForwardDiff
├ jacobian: AutoForwardDiff
├ hessian: nothing
└ dimensions:
  ├ 20 prediction steps Hp
  ├  2 control steps Hc
  ├  1 slack variable ϵ (control constraints)
  ├  1 manipulated inputs u (0 integrating states)
  ├  2 estimated states x̂
  ├  1 measured outputs ym (1 integrating states)
  ├  0 unmeasured outputs yu
  └  0 measured disturbances d
```
"""
function NonLinMPC(
    estim::SE;
    Hp::Int = default_Hp(estim.model),
    Hc::IntVectorOrInt = DEFAULT_HC,
    Mwt  = fill(DEFAULT_MWT, estim.model.ny),
    Nwt  = fill(DEFAULT_NWT, estim.model.nu),
    Lwt  = fill(DEFAULT_LWT, estim.model.nu),
    M_Hp = Diagonal(repeat(Mwt, Hp)),
    N_Hc = Diagonal(repeat(Nwt, get_Hc(move_blocking(Hp, Hc)))),
    L_Hp = Diagonal(repeat(Lwt, Hp)),
    Cwt  = DEFAULT_CWT,
    Ewt  = DEFAULT_EWT,
    JE ::Function = (_,_,_,_) -> 0.0,
    gc!::Function = (_,_,_,_,_,_) -> nothing,
    gc ::Function = gc!,
    nc = 0,
    p = estim.model.p,
    transcription::TranscriptionMethod = DEFAULT_NONLINMPC_TRANSCRIPTION,
    optim::JuMP.GenericModel = JuMP.Model(DEFAULT_NONLINMPC_OPTIMIZER, add_bridges=false),
    gradient::AbstractADType = DEFAULT_NONLINMPC_GRADIENT,
    jacobian::AbstractADType = default_jacobian(transcription),
    hessian::Union{AbstractADType, Bool, Nothing} = false,
    oracle::Bool = JuMP.solver_name(optim)=="Ipopt"
) where {
    NT<:Real, 
    SE<:StateEstimator{NT}
}
    nk = estimate_delays(estim.model)
    if Hp ≤ nk
        @warn("prediction horizon Hp ($Hp) ≤ estimated number of delays in model "*
              "($nk), the closed-loop system may be unstable or zero-gain (unresponsive)")
    end
    nb = move_blocking(Hp, Hc)
    Hc = get_Hc(nb)
    validate_JE(NT, JE)
    gc! = get_mutating_gc(NT, gc)
    weights = ControllerWeights(estim.model, Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt, Ewt)
    hessian = validate_hessian(hessian, gradient, oracle, DEFAULT_NONLINMPC_HESSIAN)
    return NonLinMPC{NT}(
        estim, Hp, Hc, nb, weights, JE, gc!, nc, p, 
        transcription, optim, gradient, jacobian, hessian, oracle
    )
end

default_jacobian(::SingleShooting)      = DEFAULT_NONLINMPC_JACDENSE
default_jacobian(::TranscriptionMethod) = DEFAULT_NONLINMPC_JACSPARSE

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
    init_optimization!(mpc::NonLinMPC, model::SimModel, optim::JuMP.GenericModel) -> nothing

Init the nonlinear optimization for [`NonLinMPC`](@ref) controllers.
"""
function init_optimization!(
    mpc::NonLinMPC, model::SimModel, optim::JuMP.GenericModel{JNT}
)  where JNT<:Real
    # --- variables and linear constraints ---
    con, transcription = mpc.con, mpc.transcription
    nZ̃ = length(mpc.Z̃)
    JuMP.num_variables(optim) == 0 || JuMP.empty!(optim)
    JuMP.set_silent(optim)
    limit_solve_time(mpc.optim, mpc.estim.model.Ts)
    @variable(optim, Z̃var[1:nZ̃])
    A = con.A[con.i_b, :]
    b = con.b[con.i_b]
    @constraint(optim, linconstraint, A*Z̃var .≤ b)
    Aeq = con.Aeq
    beq = con.beq
    @constraint(optim, linconstrainteq, Aeq*Z̃var .== beq)
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
    if mpc.oracle
        J_op = get_nonlinobj_op(mpc, optim)
        g_oracle, geq_oracle = get_nonlincon_oracle(mpc, optim)
        
    else
        J_func, ∇J_func!, g_funcs, ∇g_funcs!, geq_funcs, ∇geq_funcs! = get_optim_functions(
            mpc, optim
        )
        @operator(optim, J_op, nZ̃, J_func, ∇J_func!)
    end
    @objective(optim, Min, J_op(Z̃var...))
    if mpc.oracle
        set_nonlincon!(mpc, optim, g_oracle, geq_oracle)
    else
        init_nonlincon_leg!(
            mpc, model, transcription, g_funcs, ∇g_funcs!, geq_funcs, ∇geq_funcs!
        )
        set_nonlincon_leg!(mpc, model, transcription, optim)
    end
    return nothing
end

"""
    reset_nonlincon!(mpc::NonLinMPC)

Re-construct nonlinear constraints and add them to `mpc.optim`.
"""
function reset_nonlincon!(mpc::NonLinMPC)
    if mpc.oracle
        g_oracle, geq_oracle = get_nonlincon_oracle(mpc, mpc.optim)
        set_nonlincon!(mpc, mpc.optim, g_oracle, geq_oracle)
    else
        set_nonlincon_leg!(mpc, mpc.estim.model, mpc.transcription, mpc.optim)
    end
end

"""
    get_nonlinobj_op(mpc::NonLinMPC, optim::JuMP.GenericModel{JNT}) -> J_op

Return the nonlinear operator for the objective of `mpc` [`NonLinMPC`](@ref).

It is based on the splatting syntax. This method is really intricate and that's because of:

- These functions are used inside the nonlinear optimization, so they must be type-stable
  and as efficient as possible. All the function outputs and derivatives are cached and
  updated in-place if required to use the efficient [`value_and_gradient!`](@extref DifferentiationInterface DifferentiationInterface.value_and_jacobian!).
- The splatting syntax for objective functions implies the use of `Vararg{T,N}` (see the [performance tip](@extref Julia Be-aware-of-when-Julia-avoids-specializing))
  and memoization to avoid redundant computations. This is already complex, but it's even
  worse knowing that the automatic differentiation tools do not support splatting.
- The signature of gradient and hessian functions is not the same for univariate (`nZ̃ == 1`)
  and multivariate (`nZ̃ > 1`) operators in `JuMP`. Both must be defined.
"""
function get_nonlinobj_op(mpc::NonLinMPC, optim::JuMP.GenericModel{JNT}) where JNT<:Real
    model = mpc.estim.model
    transcription = mpc.transcription
    grad, hess = mpc.gradient, mpc.hessian
    nu, ny, nx̂, nϵ = model.nu, model.ny, mpc.estim.nx̂, mpc.nϵ
    nk = get_nk(model, transcription)
    Hp, Hc = mpc.Hp, mpc.Hc
    ng = length(mpc.con.i_g)
    nc, neq = mpc.con.nc, mpc.con.neq
    nZ̃, nU, nŶ, nX̂, nK = length(mpc.Z̃), Hp*nu, Hp*ny, Hp*nx̂, Hp*nk
    nΔŨ, nUe, nŶe = nu*Hc + nϵ, nU + nu, nŶ + ny  
    strict = Val(true)
    myNaN                            = convert(JNT, NaN)
    J::Vector{JNT}                   = zeros(JNT, 1)
    ΔŨ::Vector{JNT}                  = zeros(JNT, nΔŨ)
    x̂0end::Vector{JNT}               = zeros(JNT, nx̂)
    K0::Vector{JNT}                  = zeros(JNT, nK)
    Ue::Vector{JNT}, Ŷe::Vector{JNT} = zeros(JNT, nUe), zeros(JNT, nŶe)
    U0::Vector{JNT}, Ŷ0::Vector{JNT} = zeros(JNT, nU),  zeros(JNT, nŶ)
    Û0::Vector{JNT}, X̂0::Vector{JNT} = zeros(JNT, nU),  zeros(JNT, nX̂)
    gc::Vector{JNT}, g::Vector{JNT}  = zeros(JNT, nc),  zeros(JNT, ng)
    geq::Vector{JNT}                 = zeros(JNT, neq)
    function J!(Z̃, ΔŨ, x̂0end, Ue, Ŷe, U0, Ŷ0, Û0, K0, X̂0, gc, g, geq)
        update_predictions!(ΔŨ, x̂0end, Ue, Ŷe, U0, Ŷ0, Û0, K0, X̂0, gc, g, geq, mpc, Z̃)
        return obj_nonlinprog!(Ŷ0, U0, mpc, model, Ue, Ŷe, ΔŨ)
    end
    Z̃_J = fill(myNaN, nZ̃)      # NaN to force update at first call
    J_cache = (
        Cache(ΔŨ), Cache(x̂0end), Cache(Ue), Cache(Ŷe), Cache(U0), Cache(Ŷ0), 
        Cache(Û0), Cache(K0), Cache(X̂0), 
        Cache(gc), Cache(g), Cache(geq),
    )
    ∇J_prep = prepare_gradient(J!, grad, Z̃_J, J_cache...; strict)
    ∇J  = Vector{JNT}(undef, nZ̃)
    if !isnothing(hess)
        ∇²J_prep = prepare_hessian(J!, hess, Z̃_J, J_cache...; strict)
        ∇²J = init_diffmat(JNT, hess, ∇²J_prep, nZ̃, nZ̃)
        ∇²J_structure = lowertriangle_indices(init_diffstructure(∇²J))
        @show ncolors(∇²J_prep)
        display(sparsity_pattern(∇²J_prep))
    end
    update_objective! = if !isnothing(hess)
        function (J, ∇J, ∇²J, Z̃_J, Z̃_arg)
            if isdifferent(Z̃_arg, Z̃_J)
                Z̃_J .= Z̃_arg
                J[], _ = value_gradient_and_hessian!(
                    J!, ∇J, ∇²J, ∇²J_prep, hess, Z̃_J, J_cache...
                )
            end
        end
    else
        update_objective! = function (J, ∇J, Z̃_∇J, Z̃_arg)
            if isdifferent(Z̃_arg, Z̃_∇J)
                Z̃_∇J .= Z̃_arg
                J[], _ = value_and_gradient!(
                    J!, ∇J, ∇J_prep, grad, Z̃_∇J, J_cache...
                )
            end
        end
    end
    J_func = if !isnothing(hess)
        function (Z̃_arg::Vararg{T, N}) where {N, T<:Real}
            update_objective!(J, ∇J, ∇²J, Z̃_J, Z̃_arg)
            return J[]::T
        end
    else
        function (Z̃_arg::Vararg{T, N}) where {N, T<:Real}
            update_objective!(J, ∇J, Z̃_J, Z̃_arg)
            return J[]::T
        end
    end
    ∇J_func! = if nZ̃ == 1        # univariate syntax (see JuMP.@operator doc):
        if !isnothing(hess)
            function (Z̃_arg)
                update_objective!(J, ∇J, ∇²J, Z̃_J, Z̃_arg)
                return ∇J[]
            end
        else
            function (Z̃_arg)
                update_objective!(J, ∇J, Z̃_J, Z̃_arg)
                return ∇J[]
            end
        end
    else                        # multivariate syntax (see JuMP.@operator doc):
        if !isnothing(hess)
            function (∇J_arg::AbstractVector{T}, Z̃_arg::Vararg{T, N}) where {N, T<:Real}
                update_objective!(J, ∇J, ∇²J, Z̃_J, Z̃_arg)
                return ∇J_arg .= ∇J
            end
        else
            function (∇J_arg::AbstractVector{T}, Z̃_arg::Vararg{T, N}) where {N, T<:Real}
                update_objective!(J, ∇J, Z̃_J, Z̃_arg)
                return ∇J_arg .= ∇J
            end
        end
    end
    ∇²J_func! = if nZ̃ == 1        # univariate syntax (see JuMP.@operator doc):
        function (Z̃_arg)
            update_objective!(J, ∇J, ∇²J, Z̃_J, Z̃_arg)
            return ∇²J[]
        end
    else                        # multivariate syntax (see JuMP.@operator doc):
        function (∇²J_arg::AbstractMatrix{T}, Z̃_arg::Vararg{T, N}) where {N, T<:Real}
            update_objective!(J, ∇J, ∇²J, Z̃_J, Z̃_arg)
            return fill_diffstructure!(∇²J_arg, ∇²J, ∇²J_structure)
        end
    end
    if !isnothing(hess)
        @operator(optim, J_op, nZ̃, J_func, ∇J_func!, ∇²J_func!)
    else
        @operator(optim, J_op, nZ̃, J_func, ∇J_func!)
    end
    return J_op
end

"""
    get_nonlincon_oracle(mpc::NonLinMPC, optim) -> g_oracle, geq_oracle

Return the nonlinear constraint oracles for [`NonLinMPC`](@ref) `mpc`.

Return `g_oracle` and `geq_oracle`, the inequality and equality [`VectorNonlinearOracle`](@extref MathOptInterface MathOptInterface.VectorNonlinearOracle)
for the two respective constraints. Note that `g_oracle` only includes the non-`Inf`
inequality constraints, thus it must be re-constructed if they change. This method is really
intricate because the oracles are used inside the nonlinear optimization, so they must be
type-stable and as efficient as possible. All the function outputs and derivatives are 
ached and updated in-place if required to use the efficient [`value_and_jacobian!`](@extref DifferentiationInterface DifferentiationInterface.value_and_jacobian!).
"""
function get_nonlincon_oracle(mpc::NonLinMPC, ::JuMP.GenericModel{JNT}) where JNT<:Real
    # ----------- common cache for all functions  ----------------------------------------
    model = mpc.estim.model
    transcription = mpc.transcription
    jac, hess = mpc.jacobian, mpc.hessian
    nu, ny, nx̂, nϵ = model.nu, model.ny, mpc.estim.nx̂, mpc.nϵ
    nk = get_nk(model, transcription)
    Hp, Hc = mpc.Hp, mpc.Hc
    i_g = findall(mpc.con.i_g) # convert to non-logical indices for non-allocating @views
    ng, ngi = length(mpc.con.i_g), sum(mpc.con.i_g)
    nc, neq = mpc.con.nc, mpc.con.neq
    nZ̃, nU, nŶ, nX̂, nK = length(mpc.Z̃), Hp*nu, Hp*ny, Hp*nx̂, Hp*nk
    nΔŨ, nUe, nŶe = nu*Hc + nϵ, nU + nu, nŶ + ny  
    strict = Val(true)
    myNaN, myInf                      = convert(JNT, NaN), convert(JNT, Inf)
    ΔŨ::Vector{JNT}                   = zeros(JNT, nΔŨ)
    x̂0end::Vector{JNT}                = zeros(JNT, nx̂)
    K0::Vector{JNT}                   = zeros(JNT, nK)
    Ue::Vector{JNT}, Ŷe::Vector{JNT}  = zeros(JNT, nUe), zeros(JNT, nŶe)
    U0::Vector{JNT}, Ŷ0::Vector{JNT}  = zeros(JNT, nU),  zeros(JNT, nŶ)
    Û0::Vector{JNT}, X̂0::Vector{JNT}  = zeros(JNT, nU),  zeros(JNT, nX̂)
    gc::Vector{JNT}, g::Vector{JNT}   = zeros(JNT, nc),  zeros(JNT, ng)
    gi::Vector{JNT}, geq::Vector{JNT} = zeros(JNT, ngi), zeros(JNT, neq)
    λi::Vector{JNT}, λeq::Vector{JNT} = rand(JNT, ngi),  rand(JNT, neq)
    # -------------- inequality constraint: nonlinear oracle -----------------------------
    function gi!(gi, Z̃, ΔŨ, x̂0end, Ue, Ŷe, U0, Ŷ0, Û0, K0, X̂0, gc, geq, g)
        update_predictions!(ΔŨ, x̂0end, Ue, Ŷe, U0, Ŷ0, Û0, K0, X̂0, gc, g, geq, mpc, Z̃)
        gi .= @views g[i_g]
        return nothing
    end
    function ℓ_gi(Z̃, λi, ΔŨ, x̂0end, Ue, Ŷe, U0, Ŷ0, Û0, K0, X̂0, gc, geq, g, gi)
        update_predictions!(ΔŨ, x̂0end, Ue, Ŷe, U0, Ŷ0, Û0, K0, X̂0, gc, g, geq, mpc, Z̃)
        gi .= @views g[i_g]
        return dot(λi, gi)
    end
    Z̃_∇gi  = fill(myNaN, nZ̃)      # NaN to force update at first call
    ∇gi_cache = (
        Cache(ΔŨ), Cache(x̂0end), Cache(Ue), Cache(Ŷe), Cache(U0), Cache(Ŷ0), 
        Cache(Û0), Cache(K0), Cache(X̂0), 
        Cache(gc), Cache(geq), Cache(g)
    )
    ∇gi_prep  = prepare_jacobian(gi!, gi, jac, Z̃_∇gi, ∇gi_cache...; strict)
    ∇gi = init_diffmat(JNT, jac, ∇gi_prep, nZ̃, ngi)
    ∇gi_structure  = init_diffstructure(∇gi)
    if !isnothing(hess)
        ∇²gi_cache = (
            Cache(ΔŨ), Cache(x̂0end), Cache(Ue), Cache(Ŷe), Cache(U0), Cache(Ŷ0), 
            Cache(Û0), Cache(K0), Cache(X̂0), 
            Cache(gc), Cache(geq), Cache(g), Cache(gi)
        )
        ∇²gi_prep = prepare_hessian(
            ℓ_gi, hess, Z̃_∇gi, Constant(λi), ∇²gi_cache...; strict
        )
        ∇²ℓ_gi    = init_diffmat(JNT, hess, ∇²gi_prep, nZ̃, nZ̃)
        ∇²gi_structure = lowertriangle_indices(init_diffstructure(∇²ℓ_gi))
    end
    function update_con!(gi, ∇gi, Z̃_∇gi, Z̃_arg)
        if isdifferent(Z̃_arg, Z̃_∇gi)
            Z̃_∇gi .= Z̃_arg
            value_and_jacobian!(gi!, gi, ∇gi, ∇gi_prep, jac, Z̃_∇gi, ∇gi_cache...)
        end
        return nothing
    end
    function gi_func!(gi_arg, Z̃_arg)
        update_con!(gi, ∇gi, Z̃_∇gi, Z̃_arg)
        return gi_arg .= gi
    end
    function ∇gi_func!(∇gi_arg, Z̃_arg)
        update_con!(gi, ∇gi, Z̃_∇gi, Z̃_arg) 
        return fill_diffstructure!(∇gi_arg, ∇gi, ∇gi_structure)
    end
    function ∇²gi_func!(∇²ℓ_arg, Z̃_arg, λ_arg)
        Z̃_∇gi  .= Z̃_arg
        λi     .= λ_arg
        hessian!(ℓ_gi, ∇²ℓ_gi, ∇²gi_prep, hess, Z̃_∇gi, Constant(λi), ∇²gi_cache...)
        return fill_diffstructure!(∇²ℓ_arg, ∇²ℓ_gi, ∇²gi_structure)
    end
    gi_min = fill(-myInf, ngi)
    gi_max = zeros(JNT,   ngi)
    g_oracle = MOI.VectorNonlinearOracle(;
        dimension = nZ̃,
        l = gi_min,
        u = gi_max,
        eval_f = gi_func!,
        jacobian_structure = ∇gi_structure,
        eval_jacobian = ∇gi_func!,
        hessian_lagrangian_structure = isnothing(hess) ? Tuple{Int,Int}[] : ∇²gi_structure,
        eval_hessian_lagrangian      = isnothing(hess) ? nothing          : ∇²gi_func!
    )
    # ------------- equality constraints : nonlinear oracle ------------------------------
    function geq!(geq, Z̃, ΔŨ, x̂0end, Ue, Ŷe, U0, Ŷ0, Û0, K0, X̂0, gc, g) 
        update_predictions!(ΔŨ, x̂0end, Ue, Ŷe, U0, Ŷ0, Û0, K0, X̂0, gc, g, geq, mpc, Z̃)
        return nothing
    end
    function ℓ_geq(Z̃, λeq, ΔŨ, x̂0end, Ue, Ŷe, U0, Ŷ0, Û0, K0, X̂0, gc, geq, g)
        update_predictions!(ΔŨ, x̂0end, Ue, Ŷe, U0, Ŷ0, Û0, K0, X̂0, gc, g, geq, mpc, Z̃)
        return dot(λeq, geq)
    end
    Z̃_∇geq = fill(myNaN, nZ̃)    # NaN to force update at first call
    ∇geq_cache = (
        Cache(ΔŨ), Cache(x̂0end), Cache(Ue), Cache(Ŷe), Cache(U0), Cache(Ŷ0),
        Cache(Û0), Cache(K0),   Cache(X̂0),
        Cache(gc), Cache(g)
    )
    ∇geq_prep = prepare_jacobian(geq!, geq, jac, Z̃_∇geq, ∇geq_cache...; strict)
    ∇geq    = init_diffmat(JNT, jac, ∇geq_prep, nZ̃, neq)
    ∇geq_structure  = init_diffstructure(∇geq)
    if !isnothing(hess)
        ∇²geq_cache = (
            Cache(ΔŨ), Cache(x̂0end), Cache(Ue), Cache(Ŷe), Cache(U0), Cache(Ŷ0),
            Cache(Û0), Cache(K0),   Cache(X̂0),
            Cache(gc), Cache(geq), Cache(g)
        )
        ∇²geq_prep = prepare_hessian(
            ℓ_geq, hess, Z̃_∇geq, Constant(λeq), ∇²geq_cache...; strict
        )
        ∇²ℓ_geq = init_diffmat(JNT, hess, ∇²geq_prep, nZ̃, nZ̃)
        ∇²geq_structure = lowertriangle_indices(init_diffstructure(∇²ℓ_geq))
    end
    function update_con_eq!(geq, ∇geq, Z̃_∇geq, Z̃_arg)
        if isdifferent(Z̃_arg, Z̃_∇geq)
            Z̃_∇geq .= Z̃_arg
            value_and_jacobian!(geq!, geq, ∇geq, ∇geq_prep, jac, Z̃_∇geq, ∇geq_cache...)
        end
        return nothing
    end
    function geq_func!(geq_arg, Z̃_arg)
        update_con_eq!(geq, ∇geq, Z̃_∇geq, Z̃_arg)
        return geq_arg .= geq
    end
    function ∇geq_func!(∇geq_arg, Z̃_arg)
        update_con_eq!(geq, ∇geq, Z̃_∇geq, Z̃_arg)
        return fill_diffstructure!(∇geq_arg, ∇geq, ∇geq_structure)
    end
    function ∇²geq_func!(∇²ℓ_arg, Z̃_arg, λ_arg)
        Z̃_∇geq .= Z̃_arg
        λeq    .= λ_arg
        hessian!(ℓ_geq, ∇²ℓ_geq, ∇²geq_prep, hess, Z̃_∇geq, Constant(λeq), ∇²geq_cache...)
        return fill_diffstructure!(∇²ℓ_arg, ∇²ℓ_geq, ∇²geq_structure)
    end
    geq_min = geq_max = zeros(JNT, neq)
    geq_oracle = MOI.VectorNonlinearOracle(;
        dimension = nZ̃,
        l = geq_min,
        u = geq_max,
        eval_f = geq_func!,
        jacobian_structure = ∇geq_structure,
        eval_jacobian = ∇geq_func!,
        hessian_lagrangian_structure = isnothing(hess) ? Tuple{Int,Int}[] : ∇²geq_structure,
        eval_hessian_lagrangian      = isnothing(hess) ? nothing           : ∇²geq_func!
    )
    return g_oracle, geq_oracle
end

"""
    update_predictions!(
        ΔŨ, x̂0end, Ue, Ŷe, U0, Ŷ0, Û0, K0, X̂0, gc, g, geq, 
        mpc::PredictiveController, Z̃
    ) -> nothing

Update in-place all vectors for the predictions of `mpc` controller at decision vector `Z̃`. 

The method mutates all the arguments before the `mpc` argument.
"""
function update_predictions!(
    ΔŨ, x̂0end, Ue, Ŷe, U0, Ŷ0, Û0, K0, X̂0, gc, g, geq, mpc::PredictiveController, Z̃
)
    model, transcription = mpc.estim.model, mpc.transcription
    U0 = getU0!(U0, mpc, Z̃)
    ΔŨ = getΔŨ!(ΔŨ, mpc, transcription, Z̃)
    Ŷ0, x̂0end  = predict!(Ŷ0, x̂0end, X̂0, Û0, K0, mpc, model, transcription, U0, Z̃)
    Ue, Ŷe = extended_vectors!(Ue, Ŷe, mpc, U0, Ŷ0)
    ϵ = getϵ(mpc, Z̃)
    gc  = con_custom!(gc, mpc, Ue, Ŷe, ϵ)
    g   = con_nonlinprog!(g, mpc, model, transcription, x̂0end, Ŷ0, gc, ϵ)
    geq = con_nonlinprogeq!(geq, X̂0, Û0, K0, mpc, model, transcription, U0, Z̃)
    return nothing
end

"""
    set_nonlincon!(mpc::NonLinMPC, optim, g_oracle, geq_oracle)

Set the nonlinear inequality and equality constraints for `NonLinMPC`, if any.
"""
function set_nonlincon!(
    mpc::NonLinMPC, optim::JuMP.GenericModel{JNT}, g_oracle, geq_oracle
) where JNT<:Real
    Z̃var = optim[:Z̃var]
    nonlin_constraints = JuMP.all_constraints(
        optim, JuMP.Vector{JuMP.VariableRef}, MOI.VectorNonlinearOracle{JNT}
    )
    map(con_ref -> JuMP.delete(optim, con_ref), nonlin_constraints)
    optim[:g_oracle]   = g_oracle
    optim[:geq_oracle] = geq_oracle
    any(mpc.con.i_g) && @constraint(optim, Z̃var in g_oracle)
    mpc.con.neq > 0  && @constraint(optim, Z̃var in geq_oracle)
    return nothing
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
    mpc::NonLinMPC, ::SimModel, Ue, Ŷe::AbstractVector{NT}
) where NT<:Real
    E_JE = mpc.weights.iszero_E ? zero(NT) : mpc.weights.E*mpc.JE(Ue, Ŷe, mpc.D̂e, mpc.p)
    return E_JE
end

"Print the differentiation backends of a [`NonLinMPC`](@ref) controller."
function print_backends(io::IO, mpc::NonLinMPC)
    println(io, "├ gradient: $(backend_str(mpc.gradient))")
    println(io, "├ jacobian: $(backend_str(mpc.jacobian))")
    println(io, "├ hessian: $(backend_str(mpc.hessian))")
end
