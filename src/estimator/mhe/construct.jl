const DEFAULT_MHE_TRANSCRIPTION   = SingleShooting()
const DEFAULT_LINMHE_OPTIMIZER    = OSQP.MathOptInterfaceOSQP.Optimizer
const DEFAULT_NONLINMHE_OPTIMIZER = optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes")
const DEFAULT_NONLINMHE_GRADIENT  = AutoForwardDiff()
const DEFAULT_NONLINMHE_JACOBIAN  = AutoForwardDiff()
const DEFAULT_NONLINMHE_HESSIAN   = AutoForwardDiff()

@doc raw"""
Include all the data for the constraints of [`MovingHorizonEstimator`](@ref).

The bounds on the estimated state at arrival ``\mathbf{x̂}_k(k-N_k+1)`` is separated from
the other state constraints ``\mathbf{x̂}_k(k-N_k+2), \mathbf{x̂}_k(k-N_k+3), ...`` since
the former is always a linear inequality constraint (it's a decision variable). The fields
`x̃min` and `x̃max` refer to the bounds at the arrival (augmented with the slack variable
ε), and `X̂min` and `X̂max`, the others.
"""
struct EstimatorConstraint{NT<:Real, GCfunc<:Union{Nothing, Function}}
    # matrices for the estimated state constraints:
    Ẽx̂      ::Matrix{NT}
    Fx̂      ::Vector{NT}
    Gx̂      ::Matrix{NT}
    Jx̂      ::Matrix{NT}
    Bx̂      ::Vector{NT}
    # matrices for the zero defect constraints (N/A for single shooting transcriptions):
    ẼS      ::Matrix{NT}
    FS      ::Vector{NT}
    GS      ::Matrix{NT}
    JS      ::Matrix{NT}
    BS      ::Vector{NT}
    # bounds over the estimation windows (deviation vectors from operating points):
    x̂0min   ::Vector{NT}
    x̂0max   ::Vector{NT}
    X̂0min   ::Vector{NT}
    X̂0max   ::Vector{NT}
    Ŵmin    ::Vector{NT}
    Ŵmax    ::Vector{NT}
    V̂min    ::Vector{NT}
    V̂max    ::Vector{NT}
    # vectors for the box constraints:
    Z̃min    ::Vector{NT}
    Z̃max    ::Vector{NT}
    # A matrices for the linear inequality constraints:
    A_x̂min  ::Matrix{NT}
    A_x̂max  ::Matrix{NT}
    A_X̂min  ::Matrix{NT}
    A_X̂max  ::Matrix{NT}
    A_Ŵmin  ::SparseMatrixCSC{NT,Int}
    A_Ŵmax  ::SparseMatrixCSC{NT,Int}
    A_V̂min  ::Matrix{NT}
    A_V̂max  ::Matrix{NT}
    A       ::Matrix{NT}
    # b vector for the linear inequality constraints:
    b       ::Vector{NT}
    # indices of finite numbers in the b vector (linear inequality constraints):
    i_b     ::BitVector
    # Aeq matrix for the linear equality constraints:
    Aeq     ::Matrix{NT}
    # beq vector for the linear equality constraints:
    beq     ::Vector{NT}
    # number of nonlinear equality constraints:
    neq     ::Int
    # constraint softness parameter vectors needing separate storage:
    C_x̂min  ::Vector{NT}
    C_x̂max  ::Vector{NT}
    C_v̂min  ::Vector{NT}
    C_v̂max  ::Vector{NT}
    # indices of finite numbers in the g vectors (nonlinear inequality constraints):
    i_g     ::BitVector
    # custom nonlinear inequality constraints:
    gc!     ::GCfunc
    nc      ::Int
end

struct MovingHorizonEstimator{
    NT<:Real, 
    SM<:SimModel,
    KC<:KalmanCovariances,
    TM<:TranscriptionMethod,
    JM<:JuMP.GenericModel,
    GB<:AbstractADType,
    JB<:AbstractADType,
    HB<:Union{AbstractADType, Nothing},
    PT<:Any,
    GCfunc<:Function,
    CE<:KalmanEstimator,
} <: StateEstimator{NT}
    model::SM
    transcription::TM
    # note: `NT` and the number type `JNT` in `JuMP.GenericModel{JNT}` can be
    # different since solvers that support non-Float64 are scarce.
    optim::JM
    con::EstimatorConstraint{NT, GCfunc}
    gradient::GB
    jacobian::JB
    hessian::HB
    cov::KC
    covestim::CE
    Z̃::Vector{NT}
    lastu0::Vector{NT}
    x̂op::Vector{NT}
    f̂op::Vector{NT}
    x̂0 ::Vector{NT}
    He::Int
    nε::Int
    i_ym::Vector{Int}
    nx̂ ::Int
    nym::Int
    nyu::Int
    nxs::Int
    p::PT
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
    Ĉm  ::Matrix{NT}
    D̂dm ::Matrix{NT}
    Tŵ::SparseMatrixCSC{NT, Int}
    Ẽ ::Matrix{NT}
    F ::Vector{NT}
    G ::Matrix{NT}
    J ::Matrix{NT}
    B ::Vector{NT}
    ẽx̄::Matrix{NT}
    fx̄::Vector{NT}
    H̃::Hermitian{NT, Matrix{NT}}
    q̃::Vector{NT}
    r::Vector{NT}
    C::NT
    X̂op::Vector{NT}
    Y0m::Vector{NT}
    Yem::Vector{NT}
    U0 ::Vector{NT}
    Ue ::Vector{NT}
    D0 ::Vector{NT}
    De ::Vector{NT}
    Ŵ  ::Vector{NT}
    X̂0_old   ::Vector{NT}
    x̂0arr_old::Vector{NT}
    P̂arr_old ::Hermitian{NT, Matrix{NT}}
    Nk::Vector{Int}
    direct::Bool
    prepared::Vector{Bool}
    buffer::StateEstimatorBuffer{NT}
    function MovingHorizonEstimator{NT}(
        model::SM, 
        He, i_ym, nint_u, nint_ym, cov::KC, Cwt, 
        gc!::GCfunc, nc, p::PT,
        transcription::TM, optim::JM, 
        gradient::GB, jacobian::JB, hessian::HB, covestim::CE;
        direct=true
    ) where {
            NT<:Real, 
            SM<:SimModel{NT}, 
            KC<:KalmanCovariances,
            TM<:TranscriptionMethod,
            JM<:JuMP.GenericModel, 
            GB<:AbstractADType,
            JB<:AbstractADType,
            HB<:Union{AbstractADType, Nothing},
            PT<:Any,
            GCfunc<:Function,
            CE<:KalmanEstimator{NT}
        }
        nu, ny, nd, nk = model.nu, model.ny, model.nd, model.nk
        He < 1  && throw(ArgumentError("Estimation horizon He should be ≥ 1"))
        Cwt < 0 && throw(ArgumentError("Cwt weight should be ≥ 0"))
        nym, nyu = validate_ym(model, i_ym)
        As, Cs_u, Cs_y, nint_u, nint_ym = init_estimstoch(model, i_ym, nint_u, nint_ym)
        nxs = size(As, 1)
        nx̂  = model.nx + nxs
        Â, B̂u, Ĉ, B̂d, D̂d, x̂op, f̂op = augment_model(model, As, Cs_u, Cs_y)
        Ĉm, D̂dm = Ĉ[i_ym, :], D̂d[i_ym, :]
        lastu0 = zeros(NT, nu)
        x̂0 = [zeros(NT, model.nx); zeros(NT, nxs)]
        Tŵ = init_ZtoŴ(model, transcription, He, nx̂)
        E, G, J, B, ex̄, Ex̂, Gx̂, Jx̂, Bx̂ = init_predmat_mhe(
            model, transcription, He, i_ym, Â, B̂u, Ĉm, B̂d, D̂dm, x̂op, f̂op, direct
        )
        ES, GS, JS, BS = init_defectmat_mhe(
            model, transcription, He, Â, B̂u, B̂d, x̂op, f̂op, direct
        ) 
        # dummy values (updated just before optimization):
        F, fx̄ = zeros(NT, nym*He), zeros(NT, nx̂)
        con, nε, Ẽ, ẽx̄ = init_defaultcon_mhe(
            model, transcription, 
            He, Cwt, nx̂, nym, 
            Tŵ, E, ex̄, 
            Ex̂, Gx̂, Jx̂, Bx̂, 
            ES, GS, JS, BS, 
            gc!, nc
        )
        nZ̃ = size(Ẽ, 2)
        # dummy values, updated before optimization:
        H̃, q̃, r = Hermitian(zeros(NT, nZ̃, nZ̃), :L), zeros(NT, nZ̃), zeros(NT, 1)
        Z̃ = zeros(NT, nZ̃)
        X̂op  = repeat(x̂op, He)
        Y0m, Yem = fill(NT(NaN), nym*He),    fill(NT(NaN), nym*(He+1))
        U0,  Ue  = fill(NT(NaN), nu*He),     fill(NT(NaN),  nu*(He+1))
        D0,  De  = fill(NT(NaN), nd*(He+1)), fill(NT(NaN),  nd*(He+1))
        Ŵ        = fill(NT(NaN), nx̂*He)
        X̂0_old   = fill(NT(NaN), nx̂*He)
        D0[1:nd] .= 0 # D0 start with d0(-1) and it should not be NaN
        x̂0arr_old = zeros(NT, nx̂)
        P̂arr_old = copy(cov.P̂_0)
        Nk = [0]
        prepared = [false]
        test_custom_function_mhe(NT, model, i_ym, He, gc!, nc, x̂op, p, direct)
        buffer = StateEstimatorBuffer{NT}(nu, nx̂, nym, ny, nd, nk, He, nε)
        estim = new{NT, SM, KC, TM, JM, GB, JB, HB, PT, GCfunc, CE}(
            model, transcription, optim, con, 
            gradient, jacobian, hessian,
            cov,
            covestim,  
            Z̃, lastu0, x̂op, f̂op, x̂0, 
            He, nε,
            i_ym, nx̂, nym, nyu, nxs, 
            p,
            As, Cs_u, Cs_y, nint_u, nint_ym,
            Â, B̂u, Ĉ, B̂d, D̂d, Ĉm, D̂dm,
            Tŵ,
            Ẽ, F, G, J, B, ẽx̄, fx̄,
            H̃, q̃, r,
            Cwt,
            X̂op, 
            Y0m, Yem, U0, Ue, D0, De, Ŵ, 
            X̂0_old, x̂0arr_old, P̂arr_old, Nk,
            direct, prepared,
            buffer
        )
        init_optimization!(estim, model, optim)
        return estim
    end
end

@doc raw"""
    MovingHorizonEstimator(model::SimModel; <keyword arguments>)

Construct a moving horizon estimator (MHE) based on `model` ([`LinModel`](@ref) or [`NonLinModel`](@ref)).

It can handle constraints on the estimates. Additionally, `model` is not linearized like the
[`ExtendedKalmanFilter`](@ref), and the probability  distribution is not approximated like
the [`UnscentedKalmanFilter`](@ref). The computational costs are drastically higher, 
however, since it minimizes the following objective function at each discrete time ``k``:
```math
\min_{\mathbf{x̂}_k(k-N_k+p), \mathbf{Ŵ}, ε}   \mathbf{x̄}' \mathbf{P̄}^{-1}       \mathbf{x̄} 
                                            + \mathbf{Ŵ}' \mathbf{Q̂}_{N_k}^{-1} \mathbf{Ŵ}  
                                            + \mathbf{V̂}' \mathbf{R̂}_{N_k}^{-1} \mathbf{V̂}
                                            + C ε^2
```
subject to [`setconstraint!`](@ref) bounds and the custom nonlinear inequality constraints:
```math
\mathbf{g_c}(\mathbf{X̂_e, V̂_e, Ŵ_e, U_e, Y_e^m, D_e, P̄, x̄, p}, ε) ≤ \mathbf{0}
```
and in which the arrival costs are evaluated from the states estimated at time ``k-N_k``:
```math
\begin{aligned}
    \mathbf{x̄} &= \mathbf{x̂}_{k-N_k}(k-N_k+p) - \mathbf{x̂}_k(k-N_k+p) \\
    \mathbf{P̄} &= \mathbf{P̂}_{k-N_k}(k-N_k+p)
\end{aligned}
```
The covariances are repeated ``N_k`` times:
```math
\begin{aligned}
    \mathbf{Q̂}_{N_k} &= \text{diag}\mathbf{(Q̂,Q̂,...,Q̂)}  \\
    \mathbf{R̂}_{N_k} &= \text{diag}\mathbf{(R̂,R̂,...,R̂)} 
\end{aligned}
```
The estimation horizon ``H_e`` limits the window length:
```math
N_k =                     \begin{cases}
    k + 1   &  k < H_e    \\
    H_e     &  k ≥ H_e    \end{cases}
```
The vectors ``\mathbf{Ŵ}`` and ``\mathbf{V̂}`` respectively encompass the estimated process
noises ``\mathbf{ŵ}(k-j+p)`` and sensor noises ``\mathbf{v̂}(k-j+1)`` from ``j=N_k`` to ``1``.
The arguments of ``\mathbf{g_c}`` include the extended vectors of the estimated states 
``\mathbf{X̂_e}``, estimated sensor noises ``\mathbf{V̂_e}``,  estimated process noises
``\mathbf{Ŵ_e}``, manipulated inputs ``\mathbf{U_e}``, measured outputs ``\mathbf{Y_e^m}``
and measured disturbances ``\mathbf{D_e}``. The Extended Help details all these vectors, the
slack variable ``ε`` and the estimation of the covariance at arrival 
``\mathbf{P̂}_{k-N_k}(k-N_k+p)``. If the keyword argument `direct=true` (default value), the
constant ``p=0`` in the equations above, and the MHE is in the current form. Else ``p=1``,
leading to the prediction form.

See [`UnscentedKalmanFilter`](@ref) for details on the augmented process model and 
``\mathbf{R̂}, \mathbf{Q̂}`` covariances. This estimator allocates a fair amount of memory 
at each time step for the optimization, which is hard-coded as a single shooting
transcription for now.

!!! warning
    See the Extended Help if you get an error like:    
    `MethodError: no method matching (::var"##")(::Vector{ForwardDiff.Dual})`.

# Arguments
!!! info
    Keyword arguments with *`emphasis`* are non-Unicode alternatives.

- `model::SimModel` : (deterministic) model for the estimations.
- `He=nothing` : estimation horizon ``H_e``, must be specified.
- `i_ym=1:model.ny` : `model` output indices that are measured ``\mathbf{y^m}``, the rest 
   are unmeasured ``\mathbf{y^u}``.
- `σP_0=fill(1/model.nx,model.nx)` or *`sigmaP_0`* : main diagonal of the initial estimate
   covariance ``\mathbf{P}(0)``, specified as a standard deviation vector.
- `σQ=fill(1/model.nx,model.nx)` or *`sigmaQ`* : main diagonal of the process noise
   covariance ``\mathbf{Q}`` of `model`, specified as a standard deviation vector.
- `σR=fill(1,length(i_ym))` or *`sigmaR`* : main diagonal of the sensor noise covariance
   ``\mathbf{R}`` of `model` measured outputs, specified as a standard deviation vector.
- `nint_u=0`: integrator quantity for the stochastic model of the unmeasured disturbances at
   the manipulated inputs (vector), use `nint_u=0` for no integrator (see Extended Help).
- `nint_ym=default_nint(model,i_ym,nint_u)` : same than `nint_u` but for the unmeasured 
   disturbances at the measured outputs, use `nint_ym=0` for no integrator (see Extended Help).
- `σQint_u=fill(1,sum(nint_u))` or *`sigmaQint_u`* : same than `σQ` but for the unmeasured
   disturbances at manipulated inputs ``\mathbf{Q_{int_u}}`` (composed of integrators).
- `σPint_u_0=fill(1,sum(nint_u))` or *`sigmaPint_u_0`* : same than `σP_0` but for the unmeasured
   disturbances at manipulated inputs ``\mathbf{P_{int_u}}(0)`` (composed of integrators).
- `σQint_ym=fill(1,sum(nint_ym))` or *`sigmaQint_u`* : same than `σQ` for the unmeasured
   disturbances at measured outputs ``\mathbf{Q_{int_{ym}}}`` (composed of integrators).
- `σPint_ym_0=fill(1,sum(nint_ym))` or *`sigmaPint_ym_0`* : same than `σP_0` but for the unmeasured
   disturbances at measured outputs ``\mathbf{P_{int_{ym}}}(0)`` (composed of integrators).
- `Cwt=Inf` : slack variable weight ``C``, default to `Inf` meaning hard constraints only.
- `gc=(_,_,_,_,_,_,_,_,_,_,_)->nothing` or `gc!` : custom nonlinear inequality constraint function 
   ``\mathbf{g_c}(\mathbf{X̂_e, V̂_e, Ŵ_e, U_e, Y_e^m, D_e, P̄, x̄, p}, ε)``, mutating or not 
   (details in Extended Help).
- `nc=0` : number of custom nonlinear inequality constraints.
- `p=model.p` : ``\mathbf{g_c}`` functions parameter ``\mathbf{p}`` (any type).
- `optim=default_optim_mhe(model,nc)` : a [`JuMP.Model`](@extref) object with a quadratic or
   nonlinear optimizer for solving (default to [`Ipopt`](https://github.com/jump-dev/Ipopt.jl),
   or [`OSQP`](https://osqp.org/docs/parsers/jump.html) if `model` is a [`LinModel`](@ref)).
- `gradient=AutoForwardDiff()` : an `AbstractADType` backend for the gradient of the objective
   function when `model` is not a [`LinModel`](@ref), see [`DifferentiationInterface` doc](@extref DifferentiationInterface List).
- `jacobian=AutoForwardDiff()` : an `AbstractADType` backend for the Jacobian of the
   constraints when `model` is not a [`LinModel`](@ref), see `gradient` above for the options.
- `hessian=false` : an `AbstractADType` backend for the Hessian of the Lagrangian, see 
   `gradient` above for the options. The default `false` skip it and use the quasi-Newton
   method of `optim` (see Extended Help).
- `covestim=nothing`: a [`StateEstimator`](@ref) object for the arrival covariance estimation
   ``\mathbf{P̂}_{k-N_k}(k-N_k+p)``, `nothing` means the default choice (see Extended Help).
- `direct=true`: construct with a direct transmission from ``\mathbf{y^m}`` (a.k.a. current
   estimator, in opposition to the delayed/predictor form).

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_,_)->0.1x+u, (x,_,_)->2x, 10.0, 1, 1, 1, solver=nothing);

julia> estim = MovingHorizonEstimator(model, He=5, σR=[1], σP_0=[0.01])
MovingHorizonEstimator estimator with a sample time Ts = 10.0 s:
├ model: NonLinModel
├ optimizer: Ipopt 
├ gradient: AutoForwardDiff
├ jacobian: AutoForwardDiff
├ hessian: nothing
├ arrival covariance: UnscentedKalmanFilter 
├ direct: true
└ dimensions:
  │ ├ 5 estimation steps He
  │ ├ 1 manipulated inputs u (0 integrating states)
  │ ├ 2 estimated states x̂
  │ ├ 1 measured outputs ym (1 integrating states)
  │ ├ 0 unmeasured outputs yu
  │ └ 0 measured disturbances d
  └ optimization:
    ├ 12 decision variables Z̃ (0 slack variable, 0 bounds)
    ├  0 linear inequality constraints A
    └  0 nonlinear inequality constraints g (0 custom)
```

# Extended Help
!!! details "Extended Help"
    The estimated process and sensor noises are defined as:
    ```math
    \mathbf{Ŵ} = 
    \begin{bmatrix}
        \mathbf{ŵ}(k-N_k+p+0)     \\
        \mathbf{ŵ}(k-N_k+p+1)     \\
        \vdots                  \\
        \mathbf{ŵ}(k+p-1)
    \end{bmatrix} , \quad
    \mathbf{V̂} =
    \begin{bmatrix}
        \mathbf{v̂}(k-N_k+1)     \\
        \mathbf{v̂}(k-N_k+2)     \\
        \vdots                  \\
        \mathbf{v̂}(k)
    \end{bmatrix}
    ```
    based on the augmented model functions ``\mathbf{f̂, ĥ^m}``:
    ```math
    \begin{aligned}
        \mathbf{v̂}(k-j)     &= \mathbf{y^m}(k-j) - \mathbf{ĥ^m}\Big(\mathbf{x̂}_k(k-j), \mathbf{d}(k-j)\Big) \\
        \mathbf{x̂}_k(k-j+1) &= \mathbf{f̂}\Big(\mathbf{x̂}_k(k-j), \mathbf{u}(k-j), \mathbf{d}(k-j)\Big) + \mathbf{ŵ}(k-j)
    \end{aligned}
    ```
    The constant ``p`` equals to `!direct`. In other words, ``\mathbf{Ŵ}`` and ``\mathbf{V̂}``
    are shifted by one time step if `direct==true`. The non-default prediction form
    with ``p=1`` is particularly useful for the MHE since it moves its expensive
    computations after the MPC optimization. That is, [`preparestate!`](@ref) will solve the
    optimization by default, but it can be postponed to [`updatestate!`](@ref) with
    `direct=false`. If a `NaN` value appears in the ``\mathbf{y^m}(k-j)`` vectors it will
    be ignored in the objective function. An error will be thrown if it appears in
    ``\mathbf{u}`` or ``\mathbf{d}`` vectors since they are arguments of the dynamics.
    
    The Extended Help of [`SteadyKalmanFilter`](@ref) details the tuning of the covariances
    and the augmentation with `nint_ym` and `nint_u` arguments. The default augmentation
    scheme is identical, that is `nint_u=0` and `nint_ym` computed by [`default_nint`](@ref).
    Note that the constructor does not validate the observability of the resulting augmented
    [`NonLinModel`](@ref). In such cases, it is the user's responsibility to ensure that it
    is still observable.

    The argument ``\mathbf{p}`` in the ``\mathbf{g_c}`` function is a custom parameter
    object of any type, but use a mutable one if you want to modify it later e.g.: a vector.
    The slack variable ``ε`` relaxes the constraints if enabled, see [`setconstraint!`](@ref). 
    It is disabled thus always zero by default for the MHE (from `Cwt=Inf`) but it should be
    activated for problems with two or more types of bounds, to ensure feasibility (e.g. on
    ``\mathbf{x̂}`` and ``\mathbf{v̂}``). The following table details the arguments of 
    ``\mathbf{g_c}``, including the time steps of the first and last sample in them. 

    !!! warning
        The vectors will grows with time until ``N_k = H_e`` is reached. The time series are
        also *artificially aligned* to ease the user life, but some data at boundaries are
        unavailable e.g.: ``\mathbf{u}(k)`` with ``p=0``. They are filled with `NaN` values.
        The exact time steps of the `NaN`s are detailed in the last column below.

    | ARGUMENT           | SIZE            | FIRST SAMPLE    | LAST SAMPLE     | MISSING SAMPLES    |
    | :---------------   | :-------------- | :-------------- | :-------------- | :----------------- |
    | ``\mathbf{X̂_e}``   | `((Nk+1)*nx̂,)`  | ``k - N_k + p`` | ``k + p``       | —                  |
    | ``\mathbf{V̂_e}``   | `((Nk+1)*nym,)` | ``k - N_k + p`` | ``k + p``       | ``k - N_k, k + 1`` |
    | ``\mathbf{Ŵ_e}``   | `((Nk+1)*nx̂,)`  | ``k - N_k + p`` | ``k + p``       | ``k + p``          |
    | ``\mathbf{U_e}``   | `((Nk+1)*nu,)`  | ``k - N_k + p`` | ``k + p``       | ``k + p``          |
    | ``\mathbf{Y_e^m}`` | `((Nk+1)*nym,)` | ``k - N_k + p`` | ``k + p``       | ``k + 1``          |
    | ``\mathbf{D_e}``   | `((Nk+1)*nd,)`  | ``k - N_k + p`` | ``k + p``       | ``k + 1``          |
    | ``\mathbf{P̄}``     | `(nx̂, nx̂)`      | ``k - N_k + p`` | ``k - N_k + p`` | —                  |
    | ``\mathbf{x̄}``     | `(nx̂,)`         | ``k - N_k + p`` | ``k - N_k + p`` | —                  |
    | ``\mathbf{p}``     | var.            | —               | —               | —                  |
    | ``ε``              | `()`            | —               | —               | —                  |

    If `LHS` represents the result of the left-hand side in the inequality 
    ``\mathbf{g_c}(\mathbf{X̂_e, V̂_e, Ŵ_e, U_e, Y_e^m, D_e, P̄, x̄, p}, ε) ≤ \mathbf{0}``,
    the function `gc` can be implemented in two possible ways:
    
    1. **Non-mutating function** (out-of-place): define it as `gc(X̂e, V̂e, Ŵe, Ue, Yem, De, 
       P̄, x̄, p, ε) -> LHS`. This syntax is simple and intuitive but it allocates more memory.
    2. **Mutating function** (in-place): define it as `gc!(LHS, X̂e, V̂e, Ŵe, Ue, Yem, De, P̄,
       x̄, p, ε) -> nothing`. This syntax reduces the allocations and potentially the
       computational burden as well.

    The keyword argument `nc` is the number of elements in `LHS`, and `gc!`, an alias for
    the `gc` argument (both `gc` and `gc!` accepts non-mutating and mutating functions).

    The estimation covariance at arrival ``\mathbf{P̂}_{k-N_k}(k-N_k+p)`` gives an uncertainty
    on the state estimate at the beginning of the window ``k-N_k+p``, that is, in the past.
    It is not the same as the current estimate covariance ``\mathbf{P̂}_k(k)``, a value not
    computed by the MHE (contrarily to e.g. the [`KalmanFilter`](@ref)). Three keyword
    arguments specify its initial value with ``\mathbf{P̂_i} =  \mathrm{diag}\{ \mathbf{P}(0),
    \mathbf{P_{int_{u}}}(0), \mathbf{P_{int_{ym}}}(0) \}``. The initial state estimate
    ``\mathbf{x̂_i}`` can be manually specified with [`setstate!`](@ref), or automatically 
    with [`initstate!`](@ref) for [`LinModel`](@ref). Note the MHE with ``p=0`` is slightly
    inconsistent with all the other estimators here. It interprets the initial values as
    ``\mathbf{x̂_i} = \mathbf{x̂}_{-1}(-1)`` and  ``\mathbf{P̂_i} = \mathbf{P̂}_{-1}(-1)``, an 
    *a posteriori* estimate[^2] from the last time step. The MHE with ``p=1`` is consistent,
    interpreting them as  ``\mathbf{x̂_i} = \mathbf{x̂}_{-1}(0)`` and
    ``\mathbf{P̂_i} = \mathbf{P̂}_{-1}(0)``.

    [^2]: M. Hovd (2012), "A Note On The Smoothing Formulation Of Moving Horizon Estimation",
          *Facta Universitatis*, Vol. 11 №2.

    The optimization and the update of the arrival covariance depend on `model`:

    - If `model` is a [`LinModel`](@ref) and `nc=0`, the optimization is treated as a
      quadratic program with a time-varying Hessian, which is generally cheaper than
      nonlinear programming. By default, a [`KalmanFilter`](@ref) estimates the arrival
      covariance (customizable).
    - Else, a nonlinear program with dense [`ForwardDiff`](@extref ForwardDiff) automatic
      differentiation (AD) compute the objective and constraint derivatives by default 
      (customizable). Optimizers generally benefit from exact derivatives like AD. However, 
      the `f` and `h` functions must be compatible with this feature. See the 
      [`JuMP` documentation](@extref JuMP Common-mistakes-when-writing-a-user-defined-operator)
      for common mistakes when writing these functions. Also, an [`UnscentedKalmanFilter`](@ref)
      estimates the arrival covariance by default.

    Note that if `Cwt≠Inf`, the attribute `nlp_scaling_max_gradient` of `Ipopt` is set to 
    `10/Cwt` (if not already set), to scale the small values of ``ε``. Use the second
    constructor to specify the arrival covariance estimation method.
"""
function MovingHorizonEstimator(
    model::SM;
    He::Union{Int, Nothing} = nothing,
    i_ym::AbstractVector{Int} = 1:model.ny,
    sigmaP_0 = fill(1/model.nx, model.nx),
    sigmaQ   = fill(1/model.nx, model.nx),
    sigmaR   = fill(1, length(i_ym)),
    nint_u ::IntVectorOrInt = 0,
    nint_ym::IntVectorOrInt = default_nint(model, i_ym, nint_u),
    sigmaPint_u_0  = fill(1, max(sum(nint_u),  0)),
    sigmaQint_u    = fill(1, max(sum(nint_u),  0)),
    sigmaPint_ym_0 = fill(1, max(sum(nint_ym), 0)),
    sigmaQint_ym   = fill(1, max(sum(nint_ym), 0)),
    Cwt::Real = Inf,
    gc!::Function = (_,_,_,_,_,_,_,_,_,_,_) -> nothing,
    gc ::Function = gc!,
    nc ::Int = 0,
    p = model.p,
    transcription::ShootingMethod = DEFAULT_MHE_TRANSCRIPTION,
    optim::JM = default_optim_mhe(model, nc),
    gradient::AbstractADType = DEFAULT_NONLINMHE_GRADIENT,
    jacobian::AbstractADType = DEFAULT_NONLINMHE_JACOBIAN,
    hessian::Union{AbstractADType, Bool, Nothing} = false,
    covestim::Union{StateEstimator, Nothing} = nothing,
    direct = true,
    σP_0       = sigmaP_0,
    σQ         = sigmaQ,
    σR         = sigmaR,
    σPint_u_0  = sigmaPint_u_0,
    σQint_u    = sigmaQint_u,
    σPint_ym_0 = sigmaPint_ym_0,
    σQint_ym   = sigmaQint_ym,
) where {NT<:Real, SM<:SimModel{NT}, JM<:JuMP.GenericModel}
    # estimated covariances matrices (variance = σ²) :
    P̂_0 = isnothing(σP_0) ? nothing : Diagonal([σP_0; σPint_u_0; σPint_ym_0].^2)
    Q̂   = Diagonal([σQ;  σQint_u;  σQint_ym ].^2)
    R̂   = Diagonal([σR;].^2)
    isnothing(He) && throw(ArgumentError("Estimation horizon He must be explicitly specified")) 
    return MovingHorizonEstimator(
        model, He, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂, Cwt;
        gc, gc!, nc, p, 
        transcription, optim, gradient, jacobian, hessian, covestim, direct
    )
end

@doc raw"""
    MovingHorizonEstimator(
        model, He, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂, Cwt=Inf;
        gc!=(_,_,_,_,_,_,_,_,_,_,_) -> nothing,
        gc=gc!,
        nc=0,
        optim=default_optim_mhe(model, nc), 
        gradient=AutoForwardDiff(),
        jacobian=AutoForwardDiff(),
        hessian=false,
        covestim=nothing,
        direct=true,
    )

Construct the estimator from the augmented covariance matrices `P̂_0`, `Q̂` and `R̂`.

This syntax allows nonzero off-diagonal elements in ``\mathbf{P̂_i}, \mathbf{Q̂, R̂}``,
where ``\mathbf{P̂_i}`` is the initial estimation covariance. Its value is provided by `P̂_0`
argument. If `isnothing(P̂_0)`, its value will be fetch in `covestim.cov.P̂`.
"""
function MovingHorizonEstimator(
    model::SM, He, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂, Cwt=Inf;
    gc!::Function = (_,_,_,_,_,_,_,_,_,_,_) -> nothing,
    gc ::Function = gc!,
    nc = 0,
    p = model.p,
    transcription::ShootingMethod = DEFAULT_MHE_TRANSCRIPTION,
    optim::JM = default_optim_mhe(model, nc),
    gradient::AbstractADType = DEFAULT_NONLINMHE_GRADIENT,
    jacobian::AbstractADType = DEFAULT_NONLINMHE_JACOBIAN,
    hessian::Union{AbstractADType, Bool, Nothing} = false,
    covestim::Union{StateEstimator, Nothing} = nothing,
    direct = true,
) where {NT<:Real, SM<:SimModel{NT}, JM<:JuMP.GenericModel}
    if isnothing(P̂_0)
        if isnothing(covestim)
            throw(ArgumentError("a covestim argument should be specified to fetch its covariance P̂"))
        end
        P̂_0 = covestim.cov.P̂ 
    end
    P̂_0, Q̂, R̂ = to_mat(P̂_0), to_mat(Q̂), to_mat(R̂)
    cov = KalmanCovariances(model, i_ym, nint_u, nint_ym, Q̂, R̂, P̂_0, He)
    gc! = get_mutating_gc_mhe(NT, gc)
    hessian = validate_hessian(hessian, gradient, DEFAULT_NONLINMHE_HESSIAN)
    if isnothing(covestim)
        covestim = default_covestim_mhe(model, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂; direct)
    end
    validate_covestim(cov, covestim)
    setstate!(covestim, covestim.x̂0 + covestim.x̂op, P̂_0)
    return MovingHorizonEstimator{NT}(
        model, 
        He, i_ym, nint_u, nint_ym, cov, Cwt,
        gc!, nc, p,
        transcription, optim, gradient, jacobian, hessian, covestim; 
        direct
    )
end

"Default optimizer for MHE, depending on the model and the number of custom NL constraints."
function default_optim_mhe(model::SimModel, nc)
    if model isa LinModel && iszero(nc)
        return JuMP.Model(DEFAULT_LINMHE_OPTIMIZER, add_bridges=true)
    else
        return JuMP.Model(DEFAULT_NONLINMHE_OPTIMIZER, add_bridges=false)
    end
end

"Default arrival covariance estimator for MHE, depending on the model type only."
function default_covestim_mhe(model::SimModel, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂; direct)
    if model isa LinModel
        return KalmanFilter(model, i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂; direct)
    else
        return UnscentedKalmanFilter(model,  i_ym, nint_u, nint_ym, P̂_0, Q̂, R̂; direct)
    end
end

"Validate covestim type and dimensions."
function validate_covestim(cov::KalmanCovariances, covestim::KalmanEstimator)
    invP̄, P̂ = cov.invP̄, covestim.cov.P̂
    nx̂ = size(invP̄, 1)
    if size(invP̄) != size(P̂)
        throw(ArgumentError("P̂ covariance size $(size(P̂)) of covestim does match nx̂=$nx̂"))
    end
    return nothing
end
function validate_covestim(::KalmanCovariances, ::StateEstimator)
    error(  "covestim argument must be a SteadyKalmanFilter, KalmanFilter, "*
            "ExtendedKalmanFilter or UnscentedKalmanFilter")
end

"""
    validate_gc_mhe(NT, gc) -> ismutating

Validate `gc` function argument signature for MHE and return `true` if it is mutating.
"""
function validate_gc_mhe(NT, gc)
    ismutating = hasmethod(
        gc, 
        Tuple{
        #   LHS,      , X̂e        , V̂e         , Ŵe
            Vector{NT}, Vector{NT}, Vector{NT}, Vector{NT}, 
        #   Ue        , Yem       , De         , P̄                 , x̄         , p  , ε    
            Vector{NT}, Vector{NT}, Vector{NT}, AbstractMatrix{NT}, Vector{NT}, Any, NT
        }
    )
    isnonmutating = hasmethod(
        gc, 
        Tuple{
        #   X̂e        , V̂e        , Ŵe
            Vector{NT}, Vector{NT}, Vector{NT}, 
        #   Ue        , Yem       , De        , P̄                 , x̄         , p  , ε
            Vector{NT}, Vector{NT}, Vector{NT}, AbstractMatrix{NT}, Vector{NT}, Any, NT
        }
    )
    if !(ismutating || isnonmutating)
        error(
            "the custom constraint function has no method with type signature "*
            "gc(X̂e::Vector{$(NT)}, V̂e::Vector{$(NT)}, Ŵe::Vector{$(NT)}, "*
            "Ue::Vector{$(NT)}, Yem::Vector{$(NT)}, De::Vector{$(NT)}, "*
            "P̄::Vector{$(NT)}, x̄::Vector{$(NT)}, p::Any, ϵ::$(NT)) "*
            "or mutating form gc!(LHS::Vector{$(NT)}, "*
            "X̂e::Vector{$(NT)}, V̂e::Vector{$(NT)}, Ŵe::Vector{$(NT)}, "*
            "Ue::Vector{$(NT)}, Yem::Vector{$(NT)}, De::Vector{$(NT)}, "*
            "P̄::Vector{$(NT)}, x̄::Vector{$(NT)}, p::Any, ϵ::$(NT))"
        )
    end
    return ismutating
end

"Get mutating custom constraint function `gc!` from the provided function in argument."
function get_mutating_gc_mhe(NT, gc)
    ismutating_gc = validate_gc_mhe(NT, gc)
    gc! = if ismutating_gc
        gc
    else
        function gc!(LHS, X̂e, V̂e, Ŵe, Ue, Yem, De, P̄, x̄, p, ϵ)
            LHS .= gc(X̂e, V̂e, Ŵe, Ue, Yem, De, P̄, x̄, p, ϵ)
            return nothing
        end
    end
    return gc!
end

"""
    test_custom_function_mhe(NT, model::SimModel, i_ym, He, gc!, nc, x̂op, p, direct) -> nothing

Test the custom functions `gc!` at the operating points.

This function is called at the end of `MovingHorizonEstimator` construction. It warns the
user if the custom constraint `gc!` function crashes at `model` operating points. It
will also verify the custom function work with the growing windows, and with the `NaN` 
values at the boundaries (see [`MovingHorizonEstimator`](@ref) Extended Help for details on
the data windows). It should ease troubleshooting of simple bugs e.g.: the user forgets to
set the `nc` argument.
"""
function test_custom_function_mhe(NT, model::SimModel, i_ym, He, gc!, nc, x̂op, p, direct)
    nx̂, nŵ, nym = length(x̂op), length(x̂op), length(i_ym)
    nu, nd = model.nu, model.nd
    uop, dop, yop = model.uop, model.dop, model.yop
    yopm = yop[i_ym]
    X̂e_He, V̂e_He,  Ŵe_He = repeat(x̂op, He+1), zeros(NT, (He+1)*nym), zeros(NT, (He+1)*nŵ)
    Ue_He, Yem_He, De_He = repeat(uop, He+1), repeat(yopm, He+1),    repeat(dop, He+1)
    x̄ = zeros(NT, nx̂)
    P̄ = Hermitian(Matrix{NT}(I, nx̂, nx̂), :L)
    ε = zero(NT)
    gc = Vector{NT}(undef, nc) 
    try
        for i in 2:He+1
            X̂e, V̂e, Ŵe  = X̂e_He[1:(i*nx̂)], V̂e_He[1:(i*nym)],  Ŵe_He[1:(i*nŵ)]
            Ue, Yem, De = Ue_He[1:(i*nu)], Yem_He[1:(i*nym)], De_He[1:(i*nd)]
            if direct
                V̂e[1:nym] .= NaN
            else
                V̂e[end-nym+1:end]  .= NaN
                Yem[end-nym+1:end] .= NaN
                De[end-nd+1:end]   .= NaN
            end
            Ŵe[end-nŵ+1:end] .= NaN
            Ue[end-nu+1:end] .= NaN
            gc!(gc, X̂e, V̂e, Ŵe, Ue, Yem, De, P̄, x̄, p, ε)
            all(isfinite, gc) || error("the gc function returned non-finite values: gc = $gc")
        end
    catch err
        @warn(
            """
            Calling the gc function with X̂e, V̂e, Ŵe, Ue, Yem, De, P̄, x̄, ε arguments
            fixed at x̂op=$x̂op, uop=$uop, yop=$yop, dop=$dop, 
            P̄=I, x̄=0, p=$p, ϵ=0 failed with the following stacktrace. 
            Did you forget to set the keyword argument p or nc? 
            Did you handle the growing data windows with the NaN values at the boundaries?
            See the Extended Help of MovingHorizonEstimator for details on the arguments and
            the data windows.
            """, 
            exception=(err, catch_backtrace())
        )
    end
    return nothing
end

@doc raw"""
    setconstraint!(estim::MovingHorizonEstimator; <keyword arguments>) -> estim

Set the bound constraint parameters of the [`MovingHorizonEstimator`](@ref) `estim`.
   
It supports both soft and hard constraints on the estimated state ``\mathbf{x̂}``, process 
noise ``\mathbf{ŵ}`` and sensor noise ``\mathbf{v̂}``:
```math 
\begin{alignat*}{3}
    \mathbf{x̂_{min} - c_{x̂_{min}}} ε ≤&&\   \mathbf{x̂}_k(k-j+p) &≤ \mathbf{x̂_{max} + c_{x̂_{max}}} ε &&\qquad  j = N_k, N_k - 1, ... , 0    \\
    \mathbf{ŵ_{min} - c_{ŵ_{min}}} ε ≤&&\     \mathbf{ŵ}(k-j+p) &≤ \mathbf{ŵ_{max} + c_{ŵ_{max}}} ε &&\qquad  j = N_k, N_k - 1, ... , 1    \\
    \mathbf{v̂_{min} - c_{v̂_{min}}} ε ≤&&\     \mathbf{v̂}(k-j+1) &≤ \mathbf{v̂_{max} + c_{v̂_{max}}} ε &&\qquad  j = N_k, N_k - 1, ... , 1
\end{alignat*}
```
and also ``ε ≥ 0``. All the constraint parameters are vector. Use `±Inf` values when there
is no bound. The constraint softness parameters ``\mathbf{c}``, also called equal concern
for relaxation, are non-negative values that specify the softness of the associated bound.
Use `0.0` values for hard constraints (default for all of them). Notice that constraining
the estimated sensor noises is equivalent to bounding the innovation term, since 
``\mathbf{v̂}(k) = \mathbf{y^m}(k) - \mathbf{ŷ^m}(k)``. See Extended Help for details on
the constant ``p``, on model augmentation and on time-varying constraints.

# Arguments
!!! info
    All the keyword arguments have non-Unicode alternatives e.g. *`xhatmin`* or *`Vhatmax`*. 

    The default constraints are mentioned here for clarity but omitting a keyword argument 
    will not re-assign to its default value (defaults are set at construction only).

- `estim::MovingHorizonEstimator` : moving horizon estimator to set constraints
- `x̂min=fill(-Inf,nx̂)` / `x̂max=fill(+Inf,nx̂)` : estimated state bound ``\mathbf{x̂_{min/max}}``
- `ŵmin=fill(-Inf,nx̂)` / `ŵmax=fill(+Inf,nx̂)` : estimated process noise bound ``\mathbf{ŵ_{min/max}}``
- `v̂min=fill(-Inf,nym)` / `v̂max=fill(+Inf,nym)` : estimated sensor noise bound ``\mathbf{v̂_{min/max}}``
- `c_x̂min=fill(0.0,nx̂)` / `c_x̂max=fill(0.0,nx̂)` : `x̂min` / `x̂max` softness weight ``\mathbf{c_{x̂_{min/max}}}``
- `c_ŵmin=fill(0.0,nx̂)` / `c_ŵmax=fill(0.0,nx̂)` : `ŵmin` / `ŵmax` softness weight ``\mathbf{c_{ŵ_{min/max}}}``
- `c_v̂min=fill(0.0,nym)` / `c_v̂max=fill(0.0,nym)` : `v̂min` / `v̂max` softness weight ``\mathbf{c_{v̂_{min/max}}}``
-  all the keyword arguments above but with a first capital letter, e.g. `X̂max` or `C_ŵmax`:
   for time-varying constraints (see Extended Help)

# Examples
```jldoctest
julia> estim = MovingHorizonEstimator(LinModel(ss(0.5,1,1,0,1)), He=3);

julia> estim = setconstraint!(estim, x̂min=[-50, -50], x̂max=[50, 50])
MovingHorizonEstimator estimator with a sample time Ts = 1.0 s:
├ model: LinModel
├ optimizer: OSQP 
├ arrival covariance: KalmanFilter 
├ direct: true
└ dimensions:
  │ ├ 3 estimation steps He
  │ ├ 1 manipulated inputs u (0 integrating states)
  │ ├ 2 estimated states x̂
  │ ├ 1 measured outputs ym (1 integrating states)
  │ ├ 0 unmeasured outputs yu
  │ └ 0 measured disturbances d
  └ optimization:
    ├  8 decision variables Z̃ (0 slack variable, 4 bounds)
    ├ 12 linear inequality constraints A
    └  0 nonlinear inequality constraints g (0 custom)
```

# Extended Help
!!! details "Extended Help"
    The constant ``p=0`` if `estim.direct==true` (current form), else ``p=1`` (prediction
    form). Note that the state ``\mathbf{x̂}`` and process noise ``\mathbf{ŵ}`` constraints
    are applied on the augmented model, detailed in [`SteadyKalmanFilter`](@ref) Extended
    Help. For variable constraints, the bounds can be modified after calling [`updatestate!`](@ref),
    that is, at runtime, except for `±Inf` bounds. Time-varying constraints over the
    estimation horizon ``H_e`` are also possible, mathematically defined as:
    ```math 
    \begin{alignat*}{3}
        \mathbf{X̂_{min} - C_{x̂_{min}}} ε ≤&&\ \mathbf{X̂} &≤ \mathbf{X̂_{max} + C_{x̂_{max}}} ε \\
        \mathbf{Ŵ_{min} - C_{ŵ_{min}}} ε ≤&&\ \mathbf{Ŵ} &≤ \mathbf{Ŵ_{max} + C_{ŵ_{max}}} ε \\
        \mathbf{V̂_{min} - C_{v̂_{min}}} ε ≤&&\ \mathbf{V̂} &≤ \mathbf{V̂_{max} + C_{v̂_{max}}} ε
    \end{alignat*}
    ```
    For this, use the same keyword arguments as above but with a first capital letter:
    - `X̂min` / `X̂max` / `C_x̂min` / `C_x̂max` : ``\mathbf{X̂}`` constraints `(nx̂*(He+1),)`.
    - `Ŵmin` / `Ŵmax` / `C_ŵmin` / `C_ŵmax` : ``\mathbf{Ŵ}`` constraints `(nx̂*He,)`.
    - `V̂min` / `V̂max` / `C_v̂min` / `C_v̂max` : ``\mathbf{V̂}`` constraints `(nym*He,)`.
"""
function setconstraint!(
    estim::MovingHorizonEstimator; 
    xhatmin   = nothing, xhatmax   = nothing,
    whatmin   = nothing, whatmax   = nothing,
    vhatmin   = nothing, vhatmax   = nothing,
    c_xhatmin = nothing, c_xhatmax = nothing,
    c_whatmin = nothing, c_whatmax = nothing,
    c_vhatmin = nothing, c_vhatmax = nothing,
    Xhatmin   = nothing, Xhatmax   = nothing,
    Whatmin   = nothing, Whatmax   = nothing,
    Vhatmin   = nothing, Vhatmax   = nothing,
    C_xhatmin = nothing, C_xhatmax = nothing,
    C_whatmin = nothing, C_whatmax = nothing,
    C_vhatmin = nothing, C_vhatmax = nothing,
    x̂min   = xhatmin,   x̂max   = xhatmax,
    ŵmin   = whatmin,   ŵmax   = whatmax,
    v̂min   = vhatmin,   v̂max   = vhatmax,
    c_x̂min = c_xhatmin, c_x̂max = c_xhatmax,
    c_ŵmin = c_whatmin, c_ŵmax = c_whatmax,
    c_v̂min = c_vhatmin, c_v̂max = c_vhatmax,
    X̂min   = Xhatmin,   X̂max   = Xhatmax,
    Ŵmin   = Whatmin,   Ŵmax   = Whatmax,
    V̂min   = Vhatmin,   V̂max   = Vhatmax,
    C_x̂min = C_xhatmin, C_x̂max = C_xhatmax,
    C_ŵmin = C_whatmin, C_ŵmax = C_whatmax,
    C_v̂min = C_vhatmin, C_v̂max = C_vhatmax,
)
    model, optim, con = estim.model, estim.optim, estim.con
    transcription = estim.transcription
    nε, nx̂, nŵ, nym, He = estim.nε, estim.nx̂, estim.nx̂, estim.nym, estim.He
    nX̂con = nx̂*(He+1)
    notSolvedYet = (JuMP.termination_status(optim) == JuMP.OPTIMIZE_NOT_CALLED)
    C = estim.C
    if isnothing(X̂min) && !isnothing(x̂min)
        size(x̂min) == (nx̂,) || throw(DimensionMismatch("x̂min size must be $((nx̂,))"))
        con.x̂0min .= x̂min .- estim.x̂op 
        for i in 1:nx̂*He
            con.X̂0min[i] = x̂min[(i-1) % nx̂ + 1] - estim.X̂op[i]
        end
    elseif !isnothing(X̂min)
        size(X̂min) == (nX̂con,) || throw(DimensionMismatch("X̂min size must be $((nX̂con,))"))
        con.x̂0min .= @views X̂min[1:nx̂] .- estim.x̂op
        con.X̂0min .= @views X̂min[nx̂+1:end] .- estim.X̂op
    end
    if isnothing(X̂max) && !isnothing(x̂max)
        size(x̂max) == (nx̂,) || throw(DimensionMismatch("x̂max size must be $((nx̂,))"))
        con.x̂0max .= x̂max .- estim.x̂op 
        for i in 1:nx̂*He
            con.X̂0max[i] = x̂max[(i-1) % nx̂ + 1] - estim.X̂op[i]
        end
    elseif !isnothing(X̂max)
        size(X̂max) == (nX̂con,) || throw(DimensionMismatch("X̂max size must be $((nX̂con,))"))
        con.x̂0max .= @views X̂max[1:nx̂] .- estim.x̂op
        con.X̂0max .= @views X̂max[nx̂+1:end] .- estim.X̂op
    end
    if isnothing(Ŵmin) && !isnothing(ŵmin)
        size(ŵmin) == (nŵ,) || throw(DimensionMismatch("ŵmin size must be $((nŵ,))"))
        for i in 1:nŵ*He
            con.Ŵmin[i] = ŵmin[(i-1) % nŵ + 1]
        end
    elseif !isnothing(Ŵmin)
        size(Ŵmin) == (nŵ*He,) || throw(DimensionMismatch("Ŵmin size must be $((nŵ*He,))"))
        con.Ŵmin .= Ŵmin
    end
    if isnothing(Ŵmax) && !isnothing(ŵmax)
        size(ŵmax) == (nŵ,) || throw(DimensionMismatch("ŵmax size must be $((nŵ,))"))
        for i in 1:nŵ*He
            con.Ŵmax[i] = ŵmax[(i-1) % nŵ + 1]
        end
    elseif !isnothing(Ŵmax)
        size(Ŵmax) == (nŵ*He,) || throw(DimensionMismatch("Ŵmax size must be $((nŵ*He,))"))
        con.Ŵmax .= Ŵmax
    end
    if isnothing(V̂min) && !isnothing(v̂min)
        size(v̂min) == (nym,) || throw(DimensionMismatch("v̂min size must be $((nym,))"))
        for i in 1:nym*He
            con.V̂min[i] = v̂min[(i-1) % nym + 1]
        end
    elseif !isnothing(V̂min)
        size(V̂min) == (nym*He,) || throw(DimensionMismatch("V̂min size must be $((nym*He,))"))
        con.V̂min .= V̂min
    end
    if isnothing(V̂max) && !isnothing(v̂max)
        size(v̂max) == (nym,) || throw(DimensionMismatch("v̂max size must be $((nym,))"))
        for i in 1:nym*He
            con.V̂max[i] = v̂max[(i-1) % nym + 1]
        end
    elseif !isnothing(V̂max)
        size(V̂max) == (nym*He,) || throw(DimensionMismatch("V̂max size must be $((nym*He,))"))
        con.V̂max .= V̂max
    end
    allECRs = (
        c_x̂min, c_x̂max, c_ŵmin, c_ŵmax, c_v̂min, c_v̂max,
        C_x̂min, C_x̂max, C_ŵmin, C_ŵmax, C_v̂min, C_v̂max,
    )
    if any(ECR -> !isnothing(ECR), allECRs)
        !isinf(C) || throw(ArgumentError("Slack variable weight Cwt must be finite to set softness parameters"))
        notSolvedYet || error("Cannot set softness parameters after calling updatestate!")
    end
    if notSolvedYet
        isnothing(C_x̂min) && !isnothing(c_x̂min) && (C_x̂min = repeat(c_x̂min, He+1))
        isnothing(C_x̂max) && !isnothing(c_x̂max) && (C_x̂max = repeat(c_x̂max, He+1))
        isnothing(C_ŵmin) && !isnothing(c_ŵmin) && (C_ŵmin = repeat(c_ŵmin, He))
        isnothing(C_ŵmax) && !isnothing(c_ŵmax) && (C_ŵmax = repeat(c_ŵmax, He))
        isnothing(C_v̂min) && !isnothing(c_v̂min) && (C_v̂min = repeat(c_v̂min, He))
        isnothing(C_v̂max) && !isnothing(c_v̂max) && (C_v̂max = repeat(c_v̂max, He))
        if !isnothing(C_x̂min)
            size(C_x̂min) == (nX̂con,) || throw(DimensionMismatch("C_x̂min size must be $((nX̂con,))"))
            any(C_x̂min .< 0) && error("C_x̂min weights should be non-negative")
            con.A_x̂min[:, begin] .= @. @views -C_x̂min[1:nx̂] 
            con.C_x̂min .= @. @views C_x̂min[nx̂+1:end]
            size(con.A_X̂min, 1) ≠ 0 && (con.A_X̂min[:, begin] = -con.C_x̂min) # for LinModel
        end
        if !isnothing(C_x̂max)
            size(C_x̂max) == (nX̂con,) || throw(DimensionMismatch("C_x̂max size must be $((nX̂con,))"))
            any(C_x̂max .< 0) && error("C_x̂max weights should be non-negative")
            con.A_x̂max[:, begin] .= @. @views -C_x̂max[1:nx̂]
            con.C_x̂max .= @. @views C_x̂max[nx̂+1:end]
            size(con.A_X̂max, 1) ≠ 0 && (con.A_X̂max[:, begin] = -con.C_x̂max) # for LinModel
        end
        if !isnothing(C_ŵmin)
            size(C_ŵmin) == (nŵ*He,) || throw(DimensionMismatch("C_ŵmin size must be $((nŵ*He,))"))
            any(C_ŵmin .< 0) && error("C_ŵmin weights should be non-negative")
            con.A_Ŵmin[:, begin] .= -C_ŵmin
        end
        if !isnothing(C_ŵmax)
            size(C_ŵmax) == (nŵ*He,) || throw(DimensionMismatch("C_ŵmax size must be $((nŵ*He,))"))
            any(C_ŵmax .< 0) && error("C_ŵmax weights should be non-negative")
            con.A_Ŵmax[:, begin] .= -C_ŵmax
        end
        if !isnothing(C_v̂min)
            size(C_v̂min) == (nym*He,) || throw(DimensionMismatch("C_v̂min size must be $((nym*He,))"))
            any(C_v̂min .< 0) && error("C_v̂min weights should be non-negative")
            con.C_v̂min .= C_v̂min
            size(con.A_V̂min, 1) ≠ 0 && (con.A_V̂min[:, begin] = -con.C_v̂min) # for LinModel
        end
        if !isnothing(C_v̂max)
            size(C_v̂max) == (nym*He,) || throw(DimensionMismatch("C_v̂max size must be $((nym*He,))"))
            any(C_v̂max .< 0) && error("C_v̂max weights should be non-negative")
            con.C_v̂max .= C_v̂max
            size(con.A_V̂max, 1) ≠ 0 && (con.A_V̂max[:, begin] = -con.C_v̂max) # for LinModel
        end
    end
    Z̃min, Z̃max = init_boxconstraint_mhe(
        model, He, nx̂, nŵ, nε,
        con.x̂0min,  con.x̂0max,  con.Ŵmin,   con.Ŵmax, 
        con.A_x̂min, con.A_x̂max, con.A_Ŵmin, con.A_Ŵmax 
    )
    Z̃var = optim[:Z̃var]
    if notSolvedYet
        con.i_b[:], con.i_g[:], con.A[:] = init_matconstraint_mhe(
            model, transcription, Z̃min, Z̃max, con.nc,
            con.x̂0min,  con.x̂0max,  con.X̂0min,  con.X̂0max, 
            con.Ŵmin,   con.Ŵmax,   con.V̂min,   con.V̂max,
            con.A_x̂min, con.A_x̂max, con.A_X̂min, con.A_X̂max, 
            con.A_Ŵmin, con.A_Ŵmax, con.A_V̂min, con.A_V̂max,
            con.Aeq
        )
        con.Z̃min[:], con.Z̃max[:] = Z̃min, Z̃max
        A = con.A[con.i_b, :]
        b = zeros(count(con.i_b)) # dummy value, updated before optimization (avoid ±Inf)
        JuMP.delete(optim, optim[:linconstraint])
        JuMP.unregister(optim, :linconstraint)
        @constraint(optim, linconstraint, A*Z̃var .≤ b)
        for i in eachindex(Z̃var)
            JuMP.has_lower_bound(Z̃var[i]) && JuMP.delete_lower_bound(Z̃var[i])
            JuMP.has_upper_bound(Z̃var[i]) && JuMP.delete_upper_bound(Z̃var[i])
            !isinf(Z̃min[i]) && JuMP.set_lower_bound(Z̃var[i], Z̃min[i])
            !isinf(Z̃max[i]) && JuMP.set_upper_bound(Z̃var[i], Z̃max[i])
        end
        reset_nonlincon!(estim, model)
    else
        i_b, i_g = init_matconstraint_mhe(
            model, transcription, Z̃min, Z̃max, con.nc, 
            con.x̂0min,  con.x̂0max,  con.X̂0min,  con.X̂0max, 
            con.Ŵmin,   con.Ŵmax,   con.V̂min,   con.V̂max
        )
        diff_Z̃min, diff_Z̃max = diff_infs(Z̃min, con.Z̃min), diff_infs(Z̃max, con.Z̃max)
        if i_b ≠ con.i_b || i_g ≠ con.i_g || diff_Z̃min || diff_Z̃max
            error("Cannot modify ±Inf constraints after first solve of estimation problem")
        end
        con.Z̃min[:], con.Z̃max[:] = Z̃min, Z̃max
        for i in eachindex(Z̃var)
            !isinf(Z̃min[i]) && JuMP.set_lower_bound(Z̃var[i], Z̃min[i])
            !isinf(Z̃max[i]) && JuMP.set_upper_bound(Z̃var[i], Z̃max[i])
        end
    end
    return estim
end

"By default, no nonlinear constraints or only custom ones, do and return nothing."
reset_nonlincon!(::MovingHorizonEstimator, ::SimModel) = nothing

"""
    reset_nonlincon!(estim::MovingHorizonEstimator, model::NonLinModel)

Re-construct nonlinear constraints and add them to `estim.optim`.
"""
function reset_nonlincon!(estim::MovingHorizonEstimator, model::NonLinModel)
    g_oracle = get_nonlincon_oracle(estim, estim.optim)
    set_nonlincon!(estim, estim.optim, g_oracle)
end

"Unset `i_x̂min` and `i_x̂min` elements if finite box constraints in `Z̃min` and `Z̃max`."
function deletex̂arr_lincon!(i_x̂min, i_x̂max, ::SimModel, Z̃min, Z̃max, nε)
    nx̂ = length(i_x̂min)
    x̂0min, x̂0max = @views Z̃min[(nε+1):(nε+nx̂)], @views Z̃max[(nε+1):(nε+nx̂)]
    foreach(i -> !isinf(x̂0min[i]) && (i_x̂min[i] = false), eachindex(i_x̂min))
    foreach(i -> !isinf(x̂0max[i]) && (i_x̂max[i] = false), eachindex(i_x̂max))
    return i_x̂min, i_x̂max
end
    
"Unset `i_Ŵmin` and `i_Ŵmax` elements if finite box constraints in `Z̃min` and `Z̃max`."
function deleteŴ_lincon!(i_Ŵmin, i_Ŵmax, ::SimModel, Z̃min, Z̃max, nx̂, nε)
    Ŵmin, Ŵmax = @views Z̃min[nε+nx̂+1:end], Z̃max[nε+nx̂+1:end]
    foreach(i -> !isinf(Ŵmin[i]) && (i_Ŵmin[i] = false), eachindex(i_Ŵmin))
    foreach(i -> !isinf(Ŵmax[i]) && (i_Ŵmax[i] = false), eachindex(i_Ŵmax))
    return i_Ŵmin, i_Ŵmax
end

"""
    init_defaultcon_mhe(
        model::SimModel, transcription::TranscriptionMethod, 
        He, Cwt, nx̂, nym, 
        Tŵ, E, ex̄, 
        Ex̂, Gx̂, Jx̂, Bx̂,
        ES, GS, JS, BS,
        gc!::Function, nc
    ) -> con, Ẽ, ẽx̄

    Init `EstimatatorConstraint` struct with default parameters based on model `model`.

Also return `Ẽ` and `ẽx̄` matrices for the the augmented decision vector `Z̃`.
"""
function init_defaultcon_mhe(
    model::SimModel{NT}, transcription::TranscriptionMethod, 
    He, Cwt, nx̂, nym,
    Tŵ, E, ex̄, 
    Ex̂, Gx̂, Jx̂, Bx̂, 
    ES, GS, JS, BS,
    gc!::GCfunc, nc
) where {NT<:Real, GCfunc<:Function}
    nŵ = nx̂
    nX̂, nŴ, nYm = nx̂*He, nŵ*He, nym*He
    nε = isinf(Cwt) ? 0 : 1
    nS = size(ES, 1)
    x̂0min, x̂0max = fill(convert(NT,-Inf), nx̂),  fill(convert(NT,+Inf), nx̂)
    X̂0min, X̂0max = fill(convert(NT,-Inf), nX̂),  fill(convert(NT,+Inf), nX̂)
    Ŵmin, Ŵmax   = fill(convert(NT,-Inf), nŴ),  fill(convert(NT,+Inf), nŴ)
    V̂min, V̂max   = fill(convert(NT,-Inf), nYm), fill(convert(NT,+Inf), nYm)
    c_x̂min, c_x̂max = fill(0.0, nx̂),  fill(0.0, nx̂)
    C_x̂min, C_x̂max = fill(0.0, nX̂),  fill(0.0, nX̂)
    C_ŵmin, C_ŵmax = fill(0.0, nŴ),  fill(0.0, nŴ)
    C_v̂min, C_v̂max = fill(0.0, nYm), fill(0.0, nYm)
    A_x̂min, A_x̂max, ẽx̄ = relaxarrival(ex̄, c_x̂min, c_x̂max, nε)
    A_X̂min, A_X̂max, Ẽx̂ = relaxX̂(Ex̂, C_x̂min, C_x̂max, nε)
    A_Ŵmin, A_Ŵmax     = relaxŴ(Tŵ, C_ŵmin, C_ŵmax, nε)
    A_V̂min, A_V̂max, Ẽ  = relaxV̂(E, C_v̂min, C_v̂max , nε)
    Aeq, ẼS = augmentdefect(ES, nε; slackfirst=true)
    Z̃min, Z̃max = init_boxconstraint_mhe(
        model, He, nx̂, nŵ, nε,
        x̂0min, x̂0max, Ŵmin, Ŵmax, A_x̂min, A_x̂max, A_Ŵmin, A_Ŵmax
    )
    i_b, i_g, A, Aeq, neq = init_matconstraint_mhe(
        model, transcription, Z̃min, Z̃max, nc,
        x̂0min, x̂0max, X̂0min, X̂0max, Ŵmin, Ŵmax, V̂min, V̂max,
        A_x̂min, A_x̂max, A_X̂min, A_X̂max, A_Ŵmin, A_Ŵmax, A_V̂min, A_V̂max, Aeq
    )
    # dummy vectors (updated just before optimization):
    Fx̂, FS = zeros(NT, nx̂*He), zeros(NT, nS)
    b, beq = zeros(NT, size(A, 1)), zeros(NT, size(Aeq, 1))
    con = EstimatorConstraint{NT, GCfunc}(
        Ẽx̂, Fx̂, Gx̂, Jx̂, Bx̂,
        ẼS, FS, GS, JS, BS,
        x̂0min, x̂0max, X̂0min, X̂0max, Ŵmin, Ŵmax, V̂min, V̂max,
        Z̃min, Z̃max,
        A_x̂min, A_x̂max, A_X̂min, A_X̂max, A_Ŵmin, A_Ŵmax, A_V̂min, A_V̂max,
        A, b, i_b,
        Aeq, beq,
        neq,
        C_x̂min, C_x̂max, C_v̂min, C_v̂max,
        i_g,
        gc!, nc
    )
    return con, nε, Ẽ, ẽx̄
end

@doc raw"""
    relaxarrival(ex̄, c_x̂min, c_x̂maxm, nε) -> A_x̂min, A_x̂max, ẽx̄

Augment arrival state constraints with slack variable ε for softening the MHE.

Denoting the MHE decision variable augmented with the slack variable ``\mathbf{Z̃} = 
[\begin{smallmatrix} ε \\ \mathbf{Z} \end{smallmatrix}]``, it returns the ``\mathbf{ẽ_x̄}``
matrix that appears in the estimation error at arrival equation ``\mathbf{x̄} =
\mathbf{ẽ_x̄ Z̃ + f_x̄}``. It also returns the augmented constraints the ``\mathbf{A}``
matrices for the inequality constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{x̂_{min}}} \\ 
    \mathbf{A_{x̂_{max}}}
\end{bmatrix} \mathbf{Z̃} ≤
\begin{bmatrix}
    - \mathbf{(x̂_{min} - x̂_{op})} \\
    + \mathbf{(x̂_{max} - x̂_{op})}
\end{bmatrix}
```
"""
function relaxarrival(ex̄::AbstractMatrix{NT}, c_x̂min, c_x̂max, nε) where NT<:Real
    ex̂ = -ex̄
    if nε ≠ 0 # Z̃ = [ε; Z]
        # ε impacts arrival state constraint calculations:
        A_x̂min, A_x̂max = -[c_x̂min ex̂], [-c_x̂max ex̂]
        # ε has no impact on estimation error at arrival:
        ẽx̄ = [zeros(NT, size(ex̄, 1), 1) ex̄] 
    else # Z̃ = Z (only hard constraints)
        A_x̂min, A_x̂max = -ex̂, ex̂
        ẽx̄ = ex̄
    end
    return A_x̂min, A_x̂max, ẽx̄
end

@doc raw"""
    relaxX̂(Ex̂, C_x̂min, C_x̂max, nε) -> A_X̂min, A_X̂max, Ẽx̂

Augment estimated state constraints with slack variable ε for softening the MHE.

Denoting the MHE decision variable augmented with the slack variable ``\mathbf{Z̃} = 
[\begin{smallmatrix} ε \\ \mathbf{Z} \end{smallmatrix}]``, it returns the ``\mathbf{Ẽ_x̂}``
matrix that appears in estimated states equation ``\mathbf{X̂} = \mathbf{Ẽ_x̂ Z̃ + F_x̂}``. It
also returns the ``\mathbf{A}`` matrices for the inequality constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{X̂_{min}}} \\ 
    \mathbf{A_{X̂_{max}}}
\end{bmatrix} \mathbf{Z̃} ≤
\begin{bmatrix}
    - \mathbf{(X̂_{min} - X̂_{op}) + F_x̂} \\
    + \mathbf{(X̂_{max} - X̂_{op}) - F_x̂}
\end{bmatrix}
```
in which ``\mathbf{X̂_{min}, X̂_{max}}`` and ``\mathbf{X̂_{op}}`` vectors respectively contains
``\mathbf{x̂_{min}, x̂_{max}}`` and ``\mathbf{x̂_{op}}`` repeated ``H_e`` times.
"""
function relaxX̂(Ex̂::AbstractMatrix{NT}, C_x̂min, C_x̂max, nε) where NT<:Real
    if nε ≠ 0 # Z̃ = [ε; Z]
        if iszero(size(Ex̂, 1))
            # model is not a LinModel, thus X̂ constraints are not linear:
            C_x̂min = C_x̂max = zeros(NT, 0, 1)
        end
        # ε impacts estimated process noise constraint calculations:
        A_X̂min, A_X̂max = -[C_x̂min Ex̂], [-C_x̂max Ex̂]
        # ε has no impact on estimated process noises:
        Ẽx̂ = [zeros(NT, size(Ex̂, 1), 1) Ex̂] 
    else # Z̃ = Z (only hard constraints)
        Ẽx̂ = Ex̂
        A_X̂min, A_X̂max = -Ex̂, Ex̂
    end
    return A_X̂min, A_X̂max, Ẽx̂
end

@doc raw"""
    relaxŴ(Tŵ, C_ŵmin, C_ŵmax, nε) -> A_Ŵmin, A_Ŵmax

Augment estimated process noise constraints with slack variable ε for softening the MHE.

Denoting the MHE decision variable augmented with the slack variable ``\mathbf{Z̃} = 
[\begin{smallmatrix} ε \\ \mathbf{Z} \end{smallmatrix}]``, it returns the ``\mathbf{A}`` 
matrices for the inequality constraints:
```math
\begin{bmatrix}
    \mathbf{A_{Ŵ_{min}}} \\ 
    \mathbf{A_{Ŵ_{max}}}
\end{bmatrix} \mathbf{Z̃} ≤
\begin{bmatrix}
    - \mathbf{Ŵ_{min}} \\
    + \mathbf{Ŵ_{max}}
\end{bmatrix}
```
"""
function relaxŴ(Tŵ::AbstractMatrix{NT}, C_ŵmin, C_ŵmax, nε) where NT<:Real
    if nε ≠ 0 # Z̃ = [ε; Z]
        A_Ŵmin, A_Ŵmax = -[C_ŵmin Tŵ], [-C_ŵmax Tŵ]
    else # Z̃ = Z (only hard constraints)
        A_Ŵmin, A_Ŵmax = -Tŵ, Tŵ
    end
    return A_Ŵmin, A_Ŵmax
end

@doc raw"""
    relaxV̂(E, C_v̂min, C_v̂max, nε) -> A_V̂min, A_V̂max, Ẽ

Augment estimated sensor noise constraints with slack variable ε for softening the MHE.

Denoting the MHE decision variable augmented with the slack variable ``\mathbf{Z̃} = 
[\begin{smallmatrix} ε \\ \mathbf{Z} \end{smallmatrix}]``, it returns the ``\mathbf{Ẽ}``
matrix that appears in estimated sensor noise equation ``\mathbf{V̂} = \mathbf{Ẽ Z̃ + F}``. It
also returns the ``\mathbf{A}`` matrices for the inequality constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{V̂_{min}}} \\ 
    \mathbf{A_{V̂_{max}}}
\end{bmatrix} \mathbf{Z̃} ≤
\begin{bmatrix}
    - \mathbf{V̂_{min} + F} \\
    + \mathbf{V̂_{max} - F}
\end{bmatrix}
```
"""
function relaxV̂(E::AbstractMatrix{NT}, C_v̂min, C_v̂max, nε) where NT<:Real
    if nε ≠ 0 # Z̃ = [ε; Z]
        if iszero(size(E, 1))
            # model is not a LinModel, thus V̂ constraints are not linear:
            C_v̂min = C_v̂max = zeros(NT, 0, 1)
        end
        # ε impacts estimated sensor noise constraint calculations:
        A_V̂min, A_V̂max = -[C_v̂min E], [-C_v̂max E]
        # ε has no impact on estimated sensor noises:
        Ẽ = [zeros(NT, size(E, 1), 1) E] 
    else # Z̃ = Z (only hard constraints)
        Ẽ = E
        A_V̂min, A_V̂max = -Ẽ, Ẽ
    end
    return A_V̂min, A_V̂max, Ẽ
end

"""
    init_boxconstraint_mhe(
        model::SimModel, He, nx̂, nŵ, nε,
        x̂0min, x̂0max, Ŵmin, Ŵmax, 
        A_x̂min, A_x̂max, A_Ŵmin, A_Ŵmin 
    ) -> Z̃min, Z̃max

Init the decision variable box constraints `Z̃min` and `Z̃max` for [`MovingHorizonEstimator`](@ref).
"""
function init_boxconstraint_mhe(
    ::SimModel{NT}, He, nx̂, nŵ, nε,
    x̂0min, x̂0max, Ŵmin, Ŵmax, A_x̂min, A_x̂max, A_Ŵmin, A_Ŵmax
) where {NT<:Real}
    nZ̃ = nε + nx̂ + nŵ*He
    Z̃min, Z̃max = fill(convert(NT,-Inf), nZ̃), fill(convert(NT,+Inf), nZ̃)
    nε > 0 && (Z̃min[begin] = 0)
    if nε > 0
        n_C_x̂min = @views A_x̂min[:, begin]
        n_C_x̂max = @views A_x̂max[:, begin]
        n_C_Ŵmin = @views A_Ŵmin[:, begin]
        n_C_Ŵmax = @views A_Ŵmax[:, begin]
        for i in eachindex(x̂0min)
            iszero(n_C_x̂min[i]) && (Z̃min[nε + i] = x̂0min[i])
        end
        for i in eachindex(x̂0max)
            iszero(n_C_x̂max[i]) && (Z̃max[nε + i] = x̂0max[i])
        end
        for i in eachindex(Ŵmin)
            iszero(n_C_Ŵmin[i]) && (Z̃min[nε + nx̂ + i] = Ŵmin[i])
        end
        for i in eachindex(Ŵmax)
            iszero(n_C_Ŵmax[i]) && (Z̃min[nε + nx̂ + i] = Ŵmax[i])
        end
    else
        Z̃min[1:nx̂] .= x̂0min
        Z̃max[1:nx̂] .= x̂0max
        Z̃min[nx̂+1:end] .= Ŵmin
        Z̃max[nx̂+1:end] .= Ŵmax
    end
    return Z̃min, Z̃max
end

"""
    init_optimization!(
        estim::MovingHorizonEstimator, model::LinModel, optim::JuMP.GenericModel
    )

Init the quadratic optimization of [`MovingHorizonEstimator`](@ref).
"""
function init_optimization!(
    estim::MovingHorizonEstimator, model::LinModel, optim::JuMP.GenericModel,
)
    C, con = estim.C, estim.con
    nZ̃ = length(estim.Z̃)
    # --- variables and linear constraints ---
    JuMP.num_variables(optim) == 0 || JuMP.empty!(optim)
    JuMP.set_silent(optim)
    limit_solve_time(optim, model.Ts)
    @variable(optim, Z̃var[1:nZ̃])
    A = con.A[con.i_b, :]
    b = con.b[con.i_b]
    @constraint(optim, linconstraint, A*Z̃var .≤ b)
    Aeq = con.Aeq
    beq = con.beq
    @constraint(optim, linconstrainteq, Aeq*Z̃var .== beq)
    @objective(optim, Min, obj_quadprog(Z̃var, estim.H̃, estim.q̃))
    if con.nc > 0
        # --- nonlinear optimization init for the custom NL constraints ---
        set_scaling_gradient!(optim, C)
        # constraints with vector nonlinear oracle 
        g_oracle = get_nonlincon_oracle(estim, optim)  
        set_nonlincon!(estim, optim, g_oracle)
    end
    return nothing
end

"""
    init_optimization!(
        estim::MovingHorizonEstimator, model::SimModel, optim::JuMP.GenericModel,
    ) -> nothing

Init the nonlinear optimization of [`MovingHorizonEstimator`](@ref).
"""
function init_optimization!(
    estim::MovingHorizonEstimator, model::SimModel, optim::JuMP.GenericModel{JNT}
) where JNT<:Real
    C, con = estim.C, estim.con
    nZ̃ = length(estim.Z̃)
    # --- variables and linear constraints ---
    JuMP.num_variables(optim) == 0 || JuMP.empty!(optim)
    JuMP.set_silent(optim)
    limit_solve_time(optim, model.Ts)
    @variable(optim, Z̃var[1:nZ̃])
    A = con.A[con.i_b, :]
    b = con.b[con.i_b]
    @constraint(optim, linconstraint, A*Z̃var .≤ b)
    Aeq = con.Aeq
    beq = con.beq
    @constraint(optim, linconstrainteq, Aeq*Z̃var .== beq)
    # --- nonlinear optimization init ---
    set_scaling_gradient!(optim, C)
    # constraints with vector nonlinear oracle, objective function with splatting:    
    J_op = get_nonlinobj_op(estim, optim)
    g_oracle = get_nonlincon_oracle(estim, optim)  
    @objective(optim, Min, J_op(Z̃var...))
    set_nonlincon!(estim, optim, g_oracle)
    return nothing
end

"""
    get_nonlinobj_op(estim::MovingHorizonEstimator, optim) -> J_op

Return the nonlinear operator for the objective of `estim` [`MovingHorizonEstimator`](@ref).

It is based on the splatting syntax. This method is really intricate and that's because of:

- These functions are used inside the nonlinear optimization, so they must be type-stable
  and as efficient as possible. All the function outputs and derivatives are cached and
  updated in-place if required to use the efficient [`value_and_gradient!`](@extref DifferentiationInterface DifferentiationInterface.value_and_jacobian!).
- The splatting syntax for objective functions implies the use of `Vararg{T,N}` (see the [performance tip](@extref Julia Be-aware-of-when-Julia-avoids-specializing))
  and memoization to avoid redundant computations. This is already complex, but it's even
  worse knowing that the automatic differentiation tools do not support splatting.
"""
function get_nonlinobj_op(
    estim::MovingHorizonEstimator, optim::JuMP.GenericModel{JNT}
) where JNT<:Real
    model, con = estim.model, estim.con
    grad, hess = estim.gradient, estim.hessian
    nx̂, nym, nŷ, nu, nk, nc = estim.nx̂, estim.nym, model.ny, model.nu, model.nk, con.nc
    He = estim.He
    ng = length(con.i_g)
    nŴ, nV̂, nX̂, ng, nZ̃ = He*nx̂, He*nym, He*nx̂, length(con.i_g), length(estim.Z̃)
    nŴe, nX̂e, nV̂e = (He+1)*nx̂, (He+1)*nx̂, (He+1)*nym
    strict = Val(true)
    myNaN                               = convert(JNT, NaN)
    J::Vector{JNT}                      = zeros(JNT, 1)
    x̂0arr::Vector{JNT}, x̄::Vector{JNT}  = zeros(JNT, nx̂),  zeros(JNT, nx̂)
    Ŵ::Vector{JNT}                      = zeros(JNT, nŴ)
    V̂::Vector{JNT},     X̂0::Vector{JNT} = zeros(JNT, nV̂),  zeros(JNT, nX̂)
    Ŵe::Vector{JNT}                     = zeros(JNT, nŴe)
    V̂e::Vector{JNT},    X̂e::Vector{JNT} = zeros(JNT, nV̂e), zeros(JNT, nX̂e)
    k::Vector{JNT}                      = zeros(JNT, nk)
    û0::Vector{JNT},    ŷ0::Vector{JNT} = zeros(JNT, nu),  zeros(JNT, nŷ)
    gc::Vector{JNT},    g::Vector{JNT}  = zeros(JNT, nc),  zeros(JNT, ng) 
    function J!(Z̃, x̂0arr, x̄, Ŵ, V̂, X̂0, Ŵe, V̂e, X̂e, û0, k, ŷ0, gc, g)
        update_prediction!(x̂0arr, x̄, Ŵ, V̂, X̂0, Ŵe, V̂e, X̂e, û0, k, ŷ0, gc, g, estim, Z̃)
        return obj_nonlinprog(estim, model, x̄, V̂, Ŵ, Z̃)
    end
    Z̃_J = fill(myNaN, nZ̃)      # NaN to force update_predictions! at first call
    J_cache = (
        Cache(x̂0arr), Cache(x̄), 
        Cache(Ŵ), Cache(V̂), Cache(X̂0), 
        Cache(Ŵe), Cache(V̂e), Cache(X̂e),
        Cache(û0), Cache(k), Cache(ŷ0), Cache(gc), Cache(g),
    )
    # temporarily "fill" the estimation window for the preparation of the gradient: 
    estim.Nk[] = He
    ∇J_prep = prepare_gradient(J!, grad, Z̃_J, J_cache...; strict)
    estim.Nk[] = 0
    ∇J = Vector{JNT}(undef, nZ̃)
    if !isnothing(hess)
        estim.Nk[] = He # see comment above
        ∇²J_prep = prepare_hessian(J!, hess, Z̃_J, J_cache...; strict)
        estim.Nk[] = 0
        ∇²J = init_diffmat(JNT, hess, ∇²J_prep, nZ̃, nZ̃)
        ∇²J_structure = lowertriangle_indices(init_diffstructure(∇²J))
    end
    update_objective! = if !isnothing(hess)
        function (J, ∇J, ∇²J, Z̃_∇J, Z̃_arg)
            if isdifferent(Z̃_arg, Z̃_∇J)
                Z̃_∇J .= Z̃_arg
                J[], _ = value_gradient_and_hessian!(
                    J!, ∇J, ∇²J, ∇²J_prep, hess, Z̃_J, J_cache...
                )
            end
        end    
    else
        function (J, ∇J, Z̃_∇J, Z̃_arg)
            if isdifferent(Z̃_arg, Z̃_∇J)
                Z̃_∇J .= Z̃_arg
                J[], _ = value_and_gradient!(J!, ∇J, ∇J_prep, grad, Z̃_J, J_cache...)
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
    # only the multivariate syntax of JuMP.@operator, univariate is impossible for MHE
    # since Z̃ comprises the arrival state estimate AND the estimated process noise:
    ∇J_func! = if !isnothing(hess)
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
    function ∇²J_func!(∇²J_arg::AbstractMatrix{T}, Z̃_arg::Vararg{T, N}) where {N, T<:Real}
        update_objective!(J, ∇J, ∇²J, Z̃_J, Z̃_arg)
        return fill_diffstructure!(∇²J_arg, ∇²J, ∇²J_structure)
    end
    if !isnothing(hess)
        @operator(optim, J_op, nZ̃, J_func, ∇J_func!, ∇²J_func!)
    else
        @operator(optim, J_op, nZ̃, J_func, ∇J_func!)
    end
    return J_op
end

"""
    get_nonlincon_oracle(estim::MovingHorizonEstimator, optim) -> g_oracle, geq_oracle

Return the nonlinear constraint oracles for [`MovingHorizonEstimator`](@ref) `estim`.

Return `g_oracle` and `geq_oracle`, the inequality and equality [`VectorNonlinearOracle`](@extref MathOptInterface MathOptInterface.VectorNonlinearOracle)
for the two respective constraints. Note that `g_oracle` only includes the non-`Inf`
inequality constraints, thus it must be re-constructed if they change. This method is really
intricate because the oracles are used inside the nonlinear optimization, so they must be
type-stable and as efficient as possible. All the function outputs and derivatives are 
ached and updated in-place if required to use the efficient [`value_and_jacobian!`](@extref DifferentiationInterface DifferentiationInterface.value_and_jacobian!).
"""
function get_nonlincon_oracle(
    estim::MovingHorizonEstimator, ::JuMP.GenericModel{JNT}
) where JNT<:Real
    # ----------- common cache for all functions  ----------------------------------------
    model, con = estim.model, estim.con
    jac, hess = estim.jacobian, estim.hessian
    nx̂, nym, nŷ, nu, nk = estim.nx̂, estim.nym, model.ny, model.nu, model.nk
    He = estim.He
    i_g = findall(con.i_g) # convert to non-logical indices for non-allocating @views
    ng, ngi = length(con.i_g), sum(con.i_g)
    nc = con.nc
    nŴ, nV̂, nX̂, nZ̃ = He*nx̂, He*nym, He*nx̂, length(estim.Z̃)
    nŴe, nX̂e, nV̂e = (He+1)*nx̂, (He+1)*nx̂, (He+1)*nym
    strict = Val(true)
    myNaN, myInf                          = convert(JNT, NaN), convert(JNT, Inf)
    x̂0arr::Vector{JNT}, x̄::Vector{JNT}    = zeros(JNT, nx̂), zeros(JNT, nx̂)
    Ŵ::Vector{JNT}                        = zeros(JNT, nŴ)
    V̂::Vector{JNT},     X̂0::Vector{JNT}   = zeros(JNT, nV̂),  zeros(JNT, nX̂)
    Ŵe::Vector{JNT}                       = zeros(JNT, nŴe)
    V̂e::Vector{JNT},    X̂e::Vector{JNT}   = zeros(JNT, nV̂e), zeros(JNT, nX̂e)
    k::Vector{JNT}                        = zeros(JNT, nk)
    û0::Vector{JNT},    ŷ0::Vector{JNT}   = zeros(JNT, nu), zeros(JNT, nŷ)
    gc::Vector{JNT},    g::Vector{JNT}    = zeros(JNT, nc), zeros(JNT, ng)
    gi::Vector{JNT}                       = zeros(JNT, ngi)
    λi::Vector{JNT}                       = rand(JNT, ngi)
    # -------------- inequality constraint: nonlinear oracle -------------------------
    function gi!(gi, Z̃, x̂0arr, x̄, Ŵ, V̂, X̂0, Ŵe, V̂e, X̂e, û0, k, ŷ0, gc, g)
        update_prediction!(x̂0arr, x̄, Ŵ, V̂, X̂0, Ŵe, V̂e, X̂e, û0, k, ŷ0, gc, g, estim, Z̃)
        gi .= @views g[i_g]
        return nothing
    end
    function ℓ_gi(Z̃, λi, x̂0arr, x̄, Ŵ, V̂, X̂0, Ŵe, V̂e, X̂e, û0, k, ŷ0, gc, g, gi)
        update_prediction!(x̂0arr, x̄, Ŵ, V̂, X̂0, Ŵe, V̂e, X̂e, û0, k, ŷ0, gc, g, estim, Z̃)
        gi .= @views g[i_g]
        return dot(λi, gi)
    end
    Z̃_∇gi = fill(myNaN, nZ̃)      # NaN to force update_predictions! at first call
    ∇gi_cache = (
        Cache(x̂0arr), Cache(x̄), 
        Cache(Ŵ), Cache(V̂), Cache(X̂0), 
        Cache(Ŵe), Cache(V̂e), Cache(X̂e),
        Cache(û0), Cache(k), Cache(ŷ0), Cache(gc), Cache(g),
    )
    # temporarily "fill" the estimation window for the preparation of the gradient: 
    estim.Nk[] = He
    ∇gi_prep = prepare_jacobian(gi!, gi, jac, Z̃_∇gi, ∇gi_cache...; strict)
    estim.Nk[] = 0
    ∇gi = init_diffmat(JNT, jac, ∇gi_prep, nZ̃, ngi)
    ∇gi_structure = init_diffstructure(∇gi)
    if !isnothing(hess)
        ∇²gi_cache = (
            Cache(x̂0arr), Cache(x̄), 
            Cache(Ŵ), Cache(V̂), Cache(X̂0), 
            Cache(Ŵe), Cache(V̂e), Cache(X̂e),    
            Cache(û0), Cache(k), Cache(ŷ0), Cache(gc), Cache(g), Cache(gi)
        )
        estim.Nk[] = He # see comment above
        ∇²gi_prep = prepare_hessian(
            ℓ_gi, hess, Z̃_∇gi, Constant(λi), ∇²gi_cache...; strict
        )
        estim.Nk[] = 0
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
    return g_oracle
end

"""
    set_nonlincon!(estim::MovingHorizonEstimator, optim, g_oracle)

Set the nonlinear inequality constraints of `estim`, if any.
"""
function set_nonlincon!(
    estim::MovingHorizonEstimator, optim::JuMP.GenericModel{JNT}, g_oracle
) where JNT<:Real
    Z̃var = optim[:Z̃var]
    nonlin_constraints = JuMP.all_constraints(
        optim, JuMP.Vector{JuMP.VariableRef}, MOI.VectorNonlinearOracle{JNT}
    )
    map(con_ref -> JuMP.delete(optim, con_ref), nonlin_constraints)
    JuMP.unregister(optim, :nonlinconstraint)
    any(estim.con.i_g) && @constraint(optim, nonlinconstraint, Z̃var in g_oracle)
    return nothing
end