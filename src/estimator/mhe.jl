const DEFAULT_LINMHE_OPTIMIZER    = OSQP.MathOptInterfaceOSQP.Optimizer
const DEFAULT_NONLINMHE_OPTIMIZER = optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes")

"Include all the data for the constraints of [`MovingHorizonEstimator`](@ref)"
struct EstimatorConstraint{NT<:Real}
    Ẽx̂      ::Matrix{NT}
    Fx̂      ::Vector{NT}
    Gx̂      ::Matrix{NT}
    Jx̂      ::Matrix{NT}
    x̂min    ::Vector{NT}
    x̂max    ::Vector{NT}
    X̂min    ::Vector{NT}
    X̂max    ::Vector{NT}
    Ŵmin    ::Vector{NT}
    Ŵmax    ::Vector{NT}
    V̂min    ::Vector{NT}
    V̂max    ::Vector{NT}
    A_x̂min  ::Matrix{NT}
    A_x̂max  ::Matrix{NT}
    A_X̂min  ::Matrix{NT}
    A_X̂max  ::Matrix{NT}
    A_Ŵmin  ::Matrix{NT}
    A_Ŵmax  ::Matrix{NT}
    A_V̂min  ::Matrix{NT}
    A_V̂max  ::Matrix{NT}
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
    Z̃::Vector{NT}
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
    Ẽ ::Matrix{NT}
    F ::Vector{NT}
    G ::Matrix{NT}
    J ::Matrix{NT}
    ẽx̄::Matrix{NT}
    fx̄::Vector{NT}
    H̃::Hermitian{NT, Matrix{NT}}
    q̃::Vector{NT}
    p::Vector{NT}
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
        nŵ = nx̂
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
        E, F, G, J, ex̄, fx̄, Ex̂, Fx̂, Gx̂, Jx̂ = init_predmat_mhe(
            model, He, i_ym, Â, B̂u, Ĉ, B̂d, D̂d
        )
        con, Ẽ, ẽx̄ = init_defaultcon_mhe(model, He, nx̂, nym, E, ex̄, Ex̂, Fx̂, Gx̂, Jx̂)
        nZ̃ = nx̂ + nŵ*He
        # dummy values, updated before optimization:
        H̃, q̃, p = Hermitian(zeros(NT, nZ̃, nZ̃), :L), zeros(NT, nZ̃), zeros(NT, 1)
        Z̃ = zeros(NT, nZ̃)
        X̂, Ym   = zeros(NT, nx̂*He), zeros(NT, nym*He)
        U, D, Ŵ = zeros(NT, nu*He), zeros(NT, nd*He), zeros(NT, nx̂*He)
        x̂arr_old = zeros(NT, nx̂)
        P̂arr_old = copy(P̂0)
        Nk = [0]
        estim = new{NT, SM, JM}(
            model, optim, con, 
            Z̃, lastu0, x̂, 
            He,
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs_u, Cs_y, nint_u, nint_ym,
            Â, B̂u, Ĉ, B̂d, D̂d,
            Ẽ, F, G, J, ẽx̄, fx̄,
            H̃, q̃, p,
            P̂0, Q̂, R̂, invP̄, invQ̂_He, invR̂_He,
            M̂,
            X̂, Ym, U, D, Ŵ, 
            x̂arr_old, P̂arr_old, Nk
        )
        init_optimization!(estim, model, optim)
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
    \mathbf{x̄} &= \mathbf{x̂}_{k-N_k}(k-N_k+1) - \mathbf{x̂}_k(k-N_k+1) \\
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
- `optim=default_mhe_optim(model)` : quadratic or nonlinear optimizer used in the moving 
   horizon estimator, provided as a [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.Model)
   (default to [`Ipopt.jl`](https://github.com/jump-dev/Ipopt.jl), or [`OSQP.jl`](https://osqp.org/docs/parsers/jump.html)
   if `model` is a [`LinModel`](@ref)).
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
N_k-1`` to ``0``. The augmented model ``\mathbf{f̂}`` with the process noise recursively
generates the state estimates ``\mathbf{x̂}_k(k-j+1) = 
\mathbf{f̂}\big(\mathbf{x̂}_k(k-j), \mathbf{u}(k-j), \mathbf{d}(k-j)\big) + \mathbf{ŵ}(k-j)``
from ``j=N_k-1`` to ``0``. 
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
    optim::JM = default_mhe_optim(model),
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

"Return a `JuMP.Model` with OSQP optimizer if `model` is a [`LinModel`](@ref)."
default_mhe_optim(::LinModel) = JuMP.Model(DEFAULT_LINMHE_OPTIMIZER, add_bridges=false)
"Else, return it with Ipopt optimizer."
default_mhe_optim(::SimModel) = JuMP.Model(DEFAULT_NONLINMHE_OPTIMIZER, add_bridges=false)

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
    init_defaultcon_mhe(model::SimModel, He, nx̂, nym, E, ex̄, Ex̂, Fx̂, Gx̂, Jx̂)

TBW
"""
function init_defaultcon_mhe(
    model::SimModel{NT}, He, nx̂, nym, E, ex̄, Ex̂, Fx̂, Gx̂, Jx̂
) where {NT<:Real}
    nŵ = nx̂
    nZ̃, nX̂, nŴ, nYm = nx̂+nŵ*He, nx̂*He, nŵ*He, nym*He
    x̂min, x̂max = fill(convert(NT,-Inf), nx̂),  fill(convert(NT,+Inf), nx̂)
    X̂min, X̂max = fill(convert(NT,-Inf), nX̂),  fill(convert(NT,+Inf), nX̂)
    Ŵmin, Ŵmax = fill(convert(NT,-Inf), nŴ),  fill(convert(NT,+Inf), nŴ)
    V̂min, V̂max = fill(convert(NT,-Inf), nYm), fill(convert(NT,+Inf), nYm)
    A_x̂min, A_x̂max = relaxarrival(model, nx̂, nZ̃)
    A_X̂min, A_X̂max = relaxX̂(model, Ex̂)
    A_Ŵmin, A_Ŵmax = relaxŴ(model, He, nx̂, nŵ)
    A_V̂min, A_V̂max = relaxV̂(model, E)
    Ẽ, ẽx̄, Ẽx̂ = E, ex̄, Ex̂
    i_x̂min, i_x̂max = .!isinf.(x̂min), .!isinf.(x̂max)
    i_X̂min, i_X̂max = .!isinf.(X̂min), .!isinf.(X̂max)
    i_Ŵmin, i_Ŵmax = .!isinf.(Ŵmin), .!isinf.(Ŵmax)
    i_V̂min, i_V̂max = .!isinf.(V̂min), .!isinf.(V̂max)
    i_b, i_g, A = init_matconstraint_mhe(model, 
        i_x̂min, i_x̂max, i_X̂min, i_X̂max, i_Ŵmin, i_Ŵmax, i_V̂min, i_V̂max,
        A_x̂min, A_x̂max, A_X̂min, A_X̂max, A_Ŵmin, A_Ŵmax, A_V̂min, A_V̂max
    )
    b = zeros(NT, size(A, 1)) # dummy b vector (updated just before optimization)
    con = EstimatorConstraint{NT}(
        Ẽx̂, Fx̂, Gx̂, Jx̂,
        x̂min, x̂max, X̂min, X̂max, Ŵmin, Ŵmax, V̂min, V̂max,
        A_x̂min, A_x̂max, A_X̂min, A_X̂max, A_Ŵmin, A_Ŵmax, A_V̂min, A_V̂max,
        A, b, i_b, i_g
    )
    return con, Ẽ, ẽx̄ 
end

function relaxarrival(::SimModel{NT}, nx̂, nZ̃) where {NT<:Real}
    I_nx̂ = Matrix{NT}(I, nx̂, nx̂)
    A_x̂min, A_x̂max = [-I_nx̂ zeros(NT, nx̂, nZ̃-nx̂)], [I_nx̂ zeros(NT, nx̂, nZ̃-nx̂)]
    return A_x̂min, A_x̂max
end

function relaxX̂(::SimModel{NT}, Ex̂) where {NT<:Real}
    A_X̂min, A_X̂max = -Ex̂, Ex̂
    return A_X̂min, A_X̂max 
end

function relaxŴ(::SimModel{NT}, He, nx̂, nŵ) where {NT<:Real}
    I_nŴ = Matrix{NT}(I, nŵ*He, nŵ*He)
    A = [zeros(NT, nŵ*He, nx̂) I_nŴ]
    A_Ŵmin, A_Ŵmax = -A, A
    return A_Ŵmin, A_Ŵmax
end

function relaxV̂(::SimModel{NT}, E) where {NT<:Real}
    A_V̂min, A_V̂max = -E, E
    return A_V̂min, A_V̂max
end

@doc raw"""
    init_predmat_mhe(model::LinModel{NT}, He, i_ym, Â, B̂u, Ĉ, B̂d, D̂d) -> E, F, G, J, ex̄, fx̄

Construct the MHE prediction matrices for [`LinModel`](@ref) `model`.

Introducing the vector ``\mathbf{Z} = [\begin{smallmatrix} \mathbf{x̂_k}(k-H_e+1) 
\\ \mathbf{Ŵ} \begin{smallmatrix}]`` with the decision variables, the estimated sensor
noises from time ``k-H_e+1`` to ``k`` are computed by:
```math
\begin{aligned}
    \mathbf{V̂} = \mathbf{Y^m - Ŷ^m} &= \mathbf{E Z + G U + J D + Y^m}     \\
                                    &= \mathbf{E Z + F}
\end{aligned}
in which ``U``, ``D`` and ``Y^m`` contains respectively the manipulated inputs and measured
disturbances and measured outputs from time ``k-H_e+1`` to ``k``. The method also returns
similar matrices but for the estimation error at arrival:
```math
    \mathbf{x̄} = \mathbf{x̂}_{k-He}(k-H_e+1) - \mathbf{x̂}_{k}(k-H_e+1) = \mathbf{e_x̄ Z + f_x̄}
```
Lastly, the estimated states from time ``k-H_e+2`` to ``k+1`` are given by the equation:
```math
\begin{aligned}
    \mathbf{X̂}  &= \mathbf{E_x̂ Z + G_x̂ U + J_x̂ D} \\
                &= \mathbf{E_x̂ Z + F_x̂}
\end{aligned}
All these equations omit the operating points ``\mathbf{u_{op}, y_{op}, d_{op}}``. These
matrices are truncated when ``N_k < H_e`` (at the beginning).

# Extended Help
Using the augmented matrices ``\mathbf{Â, B̂_u, Ĉ, B̂_d, D̂_d}``, the prediction matrices
for the sensor noises are computed by (notice the minus signs after the equalities):
```math
\begin{aligned}
\mathbf{E} &= - \begin{bmatrix}
    \mathbf{Ĉ^m}\mathbf{A}^{0}                  & \mathbf{0}                                    & \cdots & \mathbf{0}   \\ 
    \mathbf{Ĉ^m}\mathbf{Â}^{1}                  & \mathbf{Ĉ^m}                                  & \cdots & \mathbf{0}   \\ 
    \vdots                                      & \vdots                                        & \ddots & \vdots       \\
    \mathbf{Ĉ^m}\mathbf{Â}^{H_e-1}              & \mathbf{Ĉ^m}\mathbf{Â}^{H_e-2}                & \cdots & \mathbf{0}   \end{bmatrix} \\
\mathbf{G} &= - \begin{bmatrix}
    \mathbf{0}                                  & \mathbf{0}                                    & \cdots & \mathbf{0}   \\ 
    \mathbf{Ĉ^m}\mathbf{A}^{0}\mathbf{B̂_u}      & \mathbf{0}                                    & \cdots & \mathbf{0}   \\ 
    \vdots                                      & \vdots                                        & \ddots & \vdots       \\
    \mathbf{Ĉ^m}\mathbf{A}^{H_e-2}\mathbf{B̂_u}  & \mathbf{Ĉ^m}\mathbf{A}^{H_e-3}\mathbf{B̂_u}    & \cdots & \mathbf{0}   \end{bmatrix} \\
\mathbf{J} &= - \begin{bmatrix}
    \mathbf{D̂^m}                                & \mathbf{0}                                    & \cdots & \mathbf{0}   \\ 
    \mathbf{Ĉ^m}\mathbf{A}^{0}\mathbf{B̂_d}      & \mathbf{D̂^m}                                  & \cdots & \mathbf{0}   \\ 
    \vdots                                      & \vdots                                        & \ddots & \vdots       \\
    \mathbf{Ĉ^m}\mathbf{A}^{H_e-2}\mathbf{B̂_d}  & \mathbf{Ĉ^m}\mathbf{A}^{H_e-3}\mathbf{B̂_d}    & \cdots & \mathbf{D̂^m} \end{bmatrix} 
\end{aligned}
```
for the estimation error at arrival:
```math
\mathbf{e_x̄} = \begin{bmatrix}
    -\mathbf{I} & \mathbf{0} & \cdots & \mathbf{0} \end{bmatrix}
```
and, for the estimated states:
```math
\begin{aligned}
\mathbf{E_x̂} &= \begin{bmatrix}
    \mathbf{Â}^{1}                      & \mathbf{I}                        & \cdots & \mathbf{0}                   \\
    \mathbf{Â}^{2}                      & \mathbf{Â}^{1}                    & \cdots & \mathbf{0}                   \\ 
    \vdots                              & \vdots                            & \ddots & \vdots                       \\
    \mathbf{Â}^{H_e}                    & \mathbf{Â}^{H_e-1}                & \cdots & \mathbf{Â}^{1}               \end{bmatrix} \\
\mathbf{G_x̂} &= \begin{bmatrix}
    \mathbf{Â}^{0}\mathbf{B̂_u}          & \mathbf{0}                        & \cdots & \mathbf{0}                   \\ 
    \mathbf{Â}^{1}\mathbf{B̂_u}          & \mathbf{Â}^{0}\mathbf{B̂_u}        & \cdots & \mathbf{0}                   \\ 
    \vdots                              & \vdots                            & \ddots & \vdots                       \\
    \mathbf{Â}^{H_e-1}\mathbf{B̂_u}      & \mathbf{Â}^{H_e-2}\mathbf{B̂_u}    & \cdots & \mathbf{Â}^{0}\mathbf{B̂_u}   \end{bmatrix} \\
\mathbf{J_x̂} &= \begin{bmatrix}
    \mathbf{Â}^{0}\mathbf{B̂_d}          & \mathbf{0}                        & \cdots & \mathbf{0}                   \\ 
    \mathbf{Â}^{1}\mathbf{B̂_d}          & \mathbf{Â}^{0}\mathbf{B̂_d}        & \cdots & \mathbf{0}                   \\ 
    \vdots                              & \vdots                            & \ddots & \vdots                       \\
    \mathbf{Â}^{H_e-1}\mathbf{B̂_d}      & \mathbf{Â}^{H_e-2}\mathbf{B̂_d}    & \cdots & \mathbf{Â}^{0}\mathbf{B̂_d}   \end{bmatrix}
\end{aligned}
```
"""
function init_predmat_mhe(model::LinModel{NT}, He, i_ym, Â, B̂u, Ĉ, B̂d, D̂d) where {NT<:Real}
    nu, nd = model.nu, model.nd
    nym, nx̂ = length(i_ym), size(Â, 1)
    Ĉm, D̂dm = Ĉ[i_ym,:], D̂d[i_ym,:] # measured outputs ym only
    nŵ = nx̂
    # --- pre-compute matrix powers ---
    # Apow 3D array : Apow[:,:,1] = A^0, Apow[:,:,2] = A^1, ... , Apow[:,:,He+1] = A^He
    Âpow = Array{NT}(undef, nx̂, nx̂, He+1)
    Âpow[:,:,1] = I(nx̂)
    for j=2:He+1
        Âpow[:,:,j] = Âpow[:,:,j-1]*Â
    end
    # helper function to improve code clarity and be similar to eqs. in docstring:
    getpower(array3D, power) = array3D[:,:, power+1]
    # --- decision variables Z ---
    nĈm_Âpow = reduce(vcat, -Ĉm*getpower(Âpow, i) for i=0:He-1)
    E = zeros(NT, nym*He, nx̂ + nŵ*He)
    E[:, 1:nx̂] = nĈm_Âpow
    for j=1:He-1
        iRow = (1 + j*nym):(nym*He)
        iCol = (1:nŵ) .+ (j-1)*nŵ .+ nx̂
        E[iRow, iCol] = nĈm_Âpow[1:length(iRow) ,:]
    end
    ex̄ = [-I zeros(NT, nx̂, nŵ*He)]
    Âpow_vec = reduce(vcat, getpower(Âpow, i) for i=0:He)
    Ex̂ = zeros(NT, nx̂*He, nx̂ + nŵ*He)
    Ex̂[:, 1:nx̂] = Âpow_vec[nx̂+1:end, :]
    for j=0:He-1
        iRow = (1 + j*nx̂):(nx̂*He)
        iCol = (1:nŵ) .+ j*nŵ .+ nx̂
        Ex̂[iRow, iCol] = Âpow_vec[1:length(iRow) ,:]
    end
    # --- manipulated inputs U ---
    nĈm_Âpow_B̂u = @views reduce(vcat, nĈm_Âpow[(1+(i*nym)):((i+1)*nym),:]*B̂u for i=0:He-1)
    G = zeros(NT, nym*He, nu*He)
    for j=1:He-1
        iRow = (1 + j*nym):(nym*He)
        iCol = (1:nu) .+ (j-1)*nu
        G[iRow, iCol] = nĈm_Âpow_B̂u[1:length(iRow) ,:]
    end
    Âpow_B̂u = reduce(vcat, getpower(Âpow, i)*B̂u for i=0:He)
    Gx̂ = zeros(NT, nx̂*He, nu*He)
    for j=0:He-1
        iRow = (1 + j*nx̂):(nx̂*He)
        iCol = (1:nu) .+ j*nu
        Gx̂[iRow, iCol] = Âpow_B̂u[1:length(iRow) ,:]
    end
    # --- measured disturbances D ---
    nĈm_Âpow_B̂d = @views reduce(vcat, nĈm_Âpow[(1+(i*nym)):((i+1)*nym),:]*B̂d for i=0:He-1)
    J = repeatdiag(-D̂dm, He)
    for j=1:He-1
        iRow = (1 + j*nym):(nym*He)
        iCol = (1:nd) .+ (j-1)*nd
        J[iRow, iCol] = nĈm_Âpow_B̂d[1:length(iRow) ,:]
    end
    Âpow_B̂d = reduce(vcat, getpower(Âpow, i)*B̂d for i=0:He)
    Jx̂ = zeros(NT, nx̂*He, nd*He)
    for j=0:He-1
        iRow = (1 + j*nx̂):(nx̂*He)
        iCol = (1:nd) .+ j*nd
        Jx̂[iRow, iCol] = Âpow_B̂d[1:length(iRow) ,:]
    end
    # --- F vectors ---
    F  = zeros(NT, nym*He) # dummy F vector (updated just before optimization)
    fx̄ = zeros(NT, nx̂)     # real  fx̄ vector value
    Fx̂ = zeros(NT, nx̂*He)  # dummy Fx̂ vector (updated just before optimization)
    return E, F, G, J, ex̄, fx̄, Ex̂, Fx̂, Gx̂, Jx̂
end

"Return empty matrices if `model` is not a [`LinModel`](@ref)."
function init_predmat_mhe(model::SimModel{NT}, He, i_ym, Â, _ , _ , _ , _ ) where {NT<:Real}
    nym, nx̂ = length(i_ym), size(Â, 1)
    nŵ = nx̂
    E  = zeros(NT, 0, nx̂ + nŵ*He)
    ex̄ = zeros(NT, 0, nx̂ + nŵ*He)
    Ex̂ = zeros(NT, 0, nx̂ + nŵ*He)
    G  = zeros(NT, 0, model.nu*He)
    Gx̂ = zeros(NT, 0, model.nu*He)
    J  = zeros(NT, 0, model.nd*He)
    Jx̂ = zeros(NT, 0, model.nd*He)
    F  = zeros(NT, nym*He)
    fx̄ = zeros(NT, nx̂)
    Fx̂ = zeros(NT, nx̂*He)
    return E, F, G, J, ex̄, fx̄, Ex̂, Fx̂, Gx̂, Jx̂
end

function init_optimization!(
    estim::MovingHorizonEstimator, ::LinModel, optim::JuMP.GenericModel
)
    He, con = estim.He, estim.con
    nŶm, nX̂, ng = He*estim.nym, He*estim.nx̂, length(con.i_g)
    # --- variables and linear constraints ---
    nvar = length(estim.Z̃)
    set_silent(optim)
    #limit_solve_time(estim) #TODO: add this feature
    @variable(optim, Z̃var[1:nvar])
    A = con.A[con.i_b, :]
    b = con.b[con.i_b]
    @constraint(optim, linconstraint, A*Z̃var .≤ b)
    @objective(optim, Min, obj_quadprog(Z̃var, estim.H̃, estim.q̃))
    return nothing
end

"""
    init_optimization!(estim::MovingHorizonEstimator, model::SimModel, optim::JuMP.GenericModel)

Init the nonlinear optimization of [`MovingHorizonEstimator`](@ref).
"""
function init_optimization!(
    estim::MovingHorizonEstimator, model::SimModel, optim::JuMP.GenericModel{JNT},
) where JNT<:Real
    He, con = estim.He, estim.con
    nV̂, nX̂, ng = He*estim.nym, He*estim.nx̂, length(con.i_g)
    # --- variables and linear constraints ---
    nvar = length(estim.Z̃)
    set_silent(optim)
    #limit_solve_time(estim) #TODO: add this feature
    @variable(optim, Z̃var[1:nvar])
    A = con.A[con.i_b, :]
    b = con.b[con.i_b]
    @constraint(optim, linconstraint, A*Z̃var .≤ b)
    # --- nonlinear optimization init ---
    # see init_optimization!(mpc::NonLinMPC, optim) for details on the inspiration
    Jfunc, gfunc = let estim=estim, model=model, nvar=nvar , nV̂=nV̂, nX̂=nX̂, ng=ng
        last_Z̃tup_float, last_Z̃tup_dual = nothing, nothing
        V̂_cache::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, nV̂), nvar + 3)
        g_cache::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, ng), nvar + 3)
        X̂_cache::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, nX̂), nvar + 3)
        function Jfunc(Z̃tup::JNT...)
            V̂ = get_tmp(V̂_cache, Z̃tup[1])
            Z̃ = collect(Z̃tup)
            if Z̃tup !== last_Z̃tup_float
                g = get_tmp(g_cache, Z̃tup[1])
                X̂ = get_tmp(X̂_cache, Z̃tup[1])
                V̂, X̂ = predict!(V̂, X̂, estim, model, Z̃)
                g = con_nonlinprog!(g, estim, model, X̂)
                last_Z̃tup_float = Z̃tup
            end
            return obj_nonlinprog(estim, model, V̂, Z̃)
        end
        function Jfunc(Z̃tup::ForwardDiff.Dual...)
            V̂ = get_tmp(V̂_cache, Z̃tup[1])
            Z̃ = collect(Z̃tup)
            if Z̃tup !== last_Z̃tup_dual
                g = get_tmp(g_cache, Z̃tup[1])
                X̂ = get_tmp(X̂_cache, Z̃tup[1])
                V̂, X̂ = predict!(V̂, X̂, estim, model, Z̃)
                g = con_nonlinprog!(g, estim, model, X̂)
                last_Z̃tup_dual = Z̃tup
            end
            return obj_nonlinprog(estim, model, V̂, Z̃)
        end
        function gfunc_i(i, Z̃tup::NTuple{N, JNT}) where N
            g = get_tmp(g_cache, Z̃tup[1])
            if Z̃tup !== last_Z̃tup_float
                V̂ = get_tmp(V̂_cache, Z̃tup[1])
                X̂ = get_tmp(X̂_cache, Z̃tup[1])
                Z̃ = collect(Z̃tup)
                V̂, X̂ = predict!(V̂, X̂, estim, model, Z̃)
                g = con_nonlinprog!(g, estim, model, X̂)
                last_Z̃tup_float = Z̃tup
            end
            return g[i]
        end 
        function gfunc_i(i, Z̃tup::NTuple{N, ForwardDiff.Dual}) where N
            g = get_tmp(g_cache, Z̃tup[1])
            if Z̃tup !== last_Z̃tup_dual
                V̂ = get_tmp(V̂_cache, Z̃tup[1])
                X̂ = get_tmp(X̂_cache, Z̃tup[1])
                Z̃ = collect(Z̃tup)
                V̂, X̂ = predict!(V̂, X̂, estim, model, Z̃)
                g = con_nonlinprog!(g, estim, model, X̂)
                last_Z̃tup_dual = Z̃tup
            end
            return g[i]
        end
        gfunc = [(Z̃...) -> gfunc_i(i, Z̃) for i in 1:ng]
        Jfunc, gfunc
    end
    register(optim, :Jfunc, nvar, Jfunc, autodiff=true)
    @NLobjective(optim, Min, Jfunc(Z̃var...))
    if ng ≠ 0
        for i in eachindex(con.X̂min)
            sym = Symbol("g_X̂min_$i")
            register(optim, sym, nvar, gfunc[i], autodiff=true)
        end
        i_end_X̂min = nX̂
        for i in eachindex(con.X̂max)
            sym = Symbol("g_X̂max_$i")
            register(optim, sym, nvar, gfunc[i_end_X̂min+i], autodiff=true)
        end
        i_end_X̂max = 2*nX̂
        for i in eachindex(con.V̂min)
            sym = Symbol("g_V̂min_$i")
            register(optim, sym, nvar, gfunc[i_end_X̂max+i], autodiff=true)
        end
        i_end_V̂min = 2*nX̂ + nV̂
        for i in eachindex(con.V̂max)
            sym = Symbol("g_V̂max_$i")
            register(optim, sym, nvar, gfunc[i_end_V̂min+i], autodiff=true)
        end
    end
    return nothing
end

@doc raw"""
    setconstraint!(estim::MovingHorizonEstimator; <keyword arguments>) -> estim

Set the constraint parameters of `estim` [`MovingHorizonEstimator`](@ref).
   
The constraints of the moving horizon estimator are defined as:
```math 
\begin{alignat*}{3}
    \mathbf{x̂_{min}} ≤&&\   \mathbf{x̂}_k(k-j+1) &≤ \mathbf{x̂_{max}}  &&\qquad  j = N_k, N_k - 1, ... , 0    \\
    \mathbf{ŵ_{min}} ≤&&\     \mathbf{ŵ}(k-j+1) &≤ \mathbf{ŵ_{max}}  &&\qquad  j = N_k, N_k - 1, ... , 1    \\
    \mathbf{v̂_{min}} ≤&&\     \mathbf{v̂}(k-j+1) &≤ \mathbf{v̂_{max}}  &&\qquad  j = N_k, N_k - 1, ... , 1
\end{alignat*}
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
    ŵmin = nothing, ŵmax = nothing,
    Ŵmin = nothing, Ŵmax = nothing,
    v̂min = nothing, v̂max = nothing,
    V̂min = nothing, V̂max = nothing,
)
    model, optim, con = estim.model, estim.optim, estim.con
    nx̂, nŵ, nym, He = estim.nx̂, estim.nx̂, estim.nym, estim.He
    nX̂con = nx̂*(He+1)
    notSolvedYet = (termination_status(optim) == OPTIMIZE_NOT_CALLED)
    isnothing(X̂min) && !isnothing(x̂min) && (X̂min = repeat(x̂min, He+1))
    isnothing(X̂max) && !isnothing(x̂max) && (X̂max = repeat(x̂max, He+1))
    isnothing(Ŵmin) && !isnothing(ŵmin) && (Ŵmin = repeat(ŵmin, He))
    isnothing(Ŵmax) && !isnothing(ŵmax) && (Ŵmax = repeat(ŵmax, He))
    isnothing(V̂min) && !isnothing(V̂min) && (X̂min = repeat(v̂min, He))
    isnothing(V̂max) && !isnothing(V̂max) && (X̂max = repeat(v̂max, He))
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
    if !isnothing(Ŵmin)
        size(Ŵmin) == (nŵ*He,) || throw(ArgumentError("Ŵmin size must be $((nŵ*He,))"))
        con.Ŵmin[:] = Ŵmin
    end
    if !isnothing(Ŵmax)
        size(Ŵmax) == (nŵ*He,) || throw(ArgumentError("Ŵmax size must be $((nŵ*He,))"))
        con.Ŵmax[:] = Ŵmax
    end
    if !isnothing(V̂min)
        size(V̂min) == (nym*He,) || throw(ArgumentError("V̂min size must be $((nym*He,))"))
        con.V̂min[:] = V̂min
    end
    if !isnothing(V̂max)
        size(V̂max) == (nym*He,) || throw(ArgumentError("V̂max size must be $((nym*He,))"))
        con.V̂max[:] = V̂max
    end
    i_x̂min, i_x̂max  = .!isinf.(con.x̂min)  , .!isinf.(con.x̂max)
    i_X̂min, i_X̂max  = .!isinf.(con.X̂min)  , .!isinf.(con.X̂max)
    i_Ŵmin, i_Ŵmax  = .!isinf.(con.Ŵmin)  , .!isinf.(con.Ŵmax)
    i_V̂min, i_V̂max  = .!isinf.(con.V̂min)  , .!isinf.(con.V̂max)
    if notSolvedYet
        con.i_b[:], con.i_g[:], con.A[:] = init_matconstraint_mhe(model, 
            i_x̂min, i_x̂max, i_X̂min, i_X̂max, i_Ŵmin, i_Ŵmax, i_V̂min, i_V̂max,
            con.A_x̂min, con.A_x̂max, 
            con.A_X̂min, con.A_X̂max, 
            con.A_Ŵmin, con.A_Ŵmax, 
            con.A_V̂min, con.A_V̂max
        )
        A = con.A[con.i_b, :]
        b = con.b[con.i_b]
        Z̃var = optim[:Z̃var]
        delete(optim, optim[:linconstraint])
        unregister(optim, :linconstraint)
        @constraint(optim, linconstraint, A*Z̃var .≤ b)
        setnonlincon!(estim, model)
    else
        i_b, i_g = init_matconstraint_mhe(model, 
            i_x̂min, i_x̂max, i_X̂min, i_X̂max, i_Ŵmin, i_Ŵmax, i_V̂min, i_V̂max
        )
        if i_b ≠ con.i_b || i_g ≠ con.i_g
            error("Cannot modify ±Inf constraints after calling updatestate!")
        end
    end
    return estim
end

@doc raw"""
    init_matconstrain_mhe(model::LinModel, 
        i_x̂min, i_x̂max, i_X̂min, i_X̂max, i_Ŵmin, i_Ŵmax, i_V̂min, i_V̂max, args...
    ) -> i_b, i_g, A

Init `i_b`, `i_g` and `A` matrices for the MHE linear inequality constraints.

The linear and nonlinear inequality constraints are respectively defined as:
```math
\begin{aligned} 
    \mathbf{A Z̃ } &≤ \mathbf{b} \\ 
    \mathbf{g(Z̃)} &≤ \mathbf{0}
\end{aligned}
```
`i_b` is a `BitVector` including the indices of ``\mathbf{b}`` that are finite numbers. 
`i_g` is a similar vector but for the indices of ``\mathbf{g}`` (empty if `model` is a 
[`LinModel`](@ref)). The method also returns the ``\mathbf{A}`` matrix if `args` is
provided. In such a case, `args`  needs to contain all the inequality constraint matrices: 
`A_x̂min, A_x̂max, A_X̂min, A_X̂max, A_Ŵmin, A_Ŵmax, A_V̂min, A_V̂max`.
"""
function init_matconstraint_mhe(::LinModel{NT}, 
    i_x̂min, i_x̂max, i_X̂min, i_X̂max, i_Ŵmin, i_Ŵmax, i_V̂min, i_V̂max, args...
) where {NT<:Real}
    i_b = [i_x̂min; i_x̂max; i_X̂min; i_X̂max; i_Ŵmin; i_Ŵmax; i_V̂min; i_V̂max]
    i_g = BitVector()
    if isempty(args)
        A = zeros(NT, length(i_b), 0)
    else
        A_x̂min, A_x̂max, A_X̂min, A_X̂max, A_Ŵmin, A_Ŵmax, A_V̂min, A_V̂max = args
        A = [A_x̂min; A_x̂max; A_X̂min; A_X̂max; A_Ŵmin; A_Ŵmax; A_V̂min; A_V̂max]
    end
    return i_b, i_g, A
end

"Init `i_b, A` without state and sensor noise constraints if `model` is not a [`LinModel`](@ref)."
function init_matconstraint_mhe(::SimModel{NT}, 
    i_x̂min, i_x̂max, i_X̂min, i_X̂max, i_Ŵmin, i_Ŵmax, i_V̂min, i_V̂max, args...
) where {NT<:Real}
    i_b = [i_x̂min; i_x̂max; i_Ŵmin; i_Ŵmax]
    i_g = [i_X̂min; i_X̂max; i_V̂min; i_V̂max]
    if isempty(args)
        A = zeros(NT, length(i_b), 0)
    else
        A_x̂min, A_x̂max, _ , _ , A_Ŵmin, A_Ŵmax, _ , _ = args
        A = [A_x̂min; A_x̂max; A_Ŵmin; A_Ŵmax]
    end
    return i_b, i_g, A
end

"By default, no nonlinear constraints in the MHE, thus return nothing."
setnonlincon!(::MovingHorizonEstimator, ::SimModel) = nothing

"Set the nonlinear constraints on the output predictions `Ŷ` and terminal states `x̂end`."
function setnonlincon!(estim::MovingHorizonEstimator, ::NonLinModel)
    optim, con = estim.optim, estim.con
    Z̃var = optim[:Z̃var]
    map(con -> delete(optim, con), all_nonlinear_constraints(optim))
    for i in findall(.!isinf.(con.X̂min))
        f_sym = Symbol("g_X̂min_$(i)")
        add_nonlinear_constraint(optim, :($(f_sym)($(Z̃var...)) <= 0))
    end
    for i in findall(.!isinf.(con.X̂max))
        f_sym = Symbol("g_X̂max_$(i)")
        add_nonlinear_constraint(optim, :($(f_sym)($(Z̃var...)) <= 0))
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
    estim.Z̃                  .= 0
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
    estim.x̂arr_old[:] = estim.X̂[1:nx̂]
    # ---------- initialize estimation vectors ------------
    initpred!(estim, model)
    # ---------- initialize linear constraints ------------
    linconstraint!(estim, model)
    # ----------- initial guess -----------------------
    Nk = estim.Nk[]
    nŴ, nYm, nX̂ = nx̂*Nk, nym*Nk, nx̂*Nk
    Z̃var::Vector{VariableRef} = optim[:Z̃var]
    x̄V̂ = Vector{NT}(undef, nx̂ + nYm)
    X̂  = Vector{NT}(undef, nX̂)
    Z̃0 = [estim.x̂arr_old; estim.Ŵ]
    x̄V̂, X̂ = predict!(x̄V̂, X̂, estim, model, Z̃0)
    J0 = obj_nonlinprog(estim, model, x̄V̂, Z̃0)
    # initial Z̃0 with Ŵ=0 if objective or constraint function not finite :
    isfinite(J0) || (Z̃0 = [estim.x̂arr_old; zeros(NT, nŴ)])
    set_start_value.(Z̃var, Z̃0)
    # ------- solve optimization problem --------------
    # at start, when time windows are not filled, some decision variables are fixed at 0:
    # unfix.(Z̃var[is_fixed.(Z̃var)])
    # fix.(Z̃var[(1 + nx̂*(Nk+1)):end], 0) 
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
    Z̃curr, Z̃last = value.(Z̃var), Z̃0
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
    estim.Z̃[:] = !isfatal(status) ? Z̃curr : Z̃last
    # --------- update estimate -----------------------
    estim.Ŵ[1:nŴ] = estim.Z̃[nx̂+1:nx̂+nŴ] # update Ŵ with optimum for next time step
    x̄V̂, X̂ = predict!(x̄V̂, X̂, estim, model, estim.Z̃)
    x̂[:] = X̂[(1 + nx̂*(Nk-1)):(nx̂*Nk)]
    if Nk == He
        uarr, ymarr, darr = estim.U[1:nu], estim.Ym[1:nym], estim.D[1:nd]
        update_cov!(estim.P̂arr_old, estim, model, uarr, ymarr, darr)
        estim.invP̄.data[:] = Hermitian(inv(estim.P̂arr_old), :L)
    end
    return nothing
end

@doc raw"""
    initpred!(estim::MovingHorizonEstimator, model::LinModel)

The ``H̃`` matrix of the quadratic general form is not constant here because of the 
time-varying ``\mathbf{P̄}`` weight (the estimation error covariance at arrival).
"""
function initpred!(estim::MovingHorizonEstimator, model::LinModel)
    nx̂, nŵ, nym, Nk = estim.nx̂, estim.nx̂, estim.nym, estim.Nk[]
    nYm, nU, nD, nŴ = nym*Nk, model.nu*Nk, model.nd*Nk, nŵ*Nk
    nZ̃ = nx̂ + nŴ
    invQ̂_Nk, invR̂_Nk = @views estim.invQ̂_He[1:nŴ, 1:nŴ], estim.invR̂_He[1:nYm, 1:nYm]
    # --- update F and fx̄ vectors for MHE predictions ---
    estim.F[:] = estim.G*estim.U + estim.Ym
    if model.nd ≠ 0
        estim.F[:] = estim.F + estim.J*estim.D
    end
    estim.fx̄[:] = estim.x̂arr_old
    # --- update H̃, q̃ and p vectors for quadratic optimization ---
    Ẽ_Nk = @views [estim.ẽx̄[:, 1:nZ̃]; estim.Ẽ[1:nYm, 1:nZ̃]]
    F_Nk = @views [estim.fx̄; estim.F[1:nYm]]
    M_Nk = [estim.invP̄ zeros(nx̂, nYm); zeros(nYm, nx̂) invR̂_Nk]
    Ñ_Nk = [zeros(nx̂, nZ̃); zeros(nŴ, nx̂) invQ̂_Nk]
    estim.q̃[1:nZ̃] = 2(M_Nk*Ẽ_Nk)'*F_Nk
    estim.p[] = dot(F_Nk, M_Nk, F_Nk)
    H̃ = 2*(Ẽ_Nk'*M_Nk*Ẽ_Nk + Ñ_Nk)
    estim.H̃.data[1:nZ̃, 1:nZ̃] = H̃
    Z̃var::Vector{VariableRef} = estim.optim[:Z̃var]
    set_objective_function(estim.optim, obj_quadprog(Z̃var, estim.H̃, estim.q̃))
    return nothing
end

initpred!(estim::MovingHorizonEstimator, model::SimModel) = nothing

function linconstraint!(estim::MovingHorizonEstimator, model::LinModel)
    estim.con.Fx̂[:] = estim.con.Gx̂*estim.U
    if model.nd ≠ 0
        estim.con.Fx̂[:] = estim.con.Fx̂ + estim.con.Jx̂*estim.D
    end
    estim.con.b[:] = [
        -estim.con.x̂min
        +estim.con.x̂max
        -estim.con.X̂min + estim.con.Fx̂
        +estim.con.X̂max - estim.con.Fx̂
        -estim.con.Ŵmin
        +estim.con.Ŵmax
        -estim.con.V̂min + estim.F
        +estim.con.V̂max - estim.F
    ]
    lincon = estim.optim[:linconstraint]
    set_normalized_rhs.(lincon, estim.con.b[estim.con.i_b])
end

function linconstraint!(estim::MovingHorizonEstimator, ::SimModel)
    estim.con.b[:] = [
        -estim.con.x̂min
        +estim.con.x̂max
        -estim.con.Ŵmin
        +estim.con.Ŵmax
    ]
    lincon = estim.optim[:linconstraint]
    set_normalized_rhs.(lincon, estim.con.b[estim.con.i_b])
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


function obj_nonlinprog(
    estim::MovingHorizonEstimator, ::LinModel, V̂, Z̃::Vector{T}
) where {T<:Real}
    return obj_quadprog(Z̃, estim.H̃, estim.q̃) + estim.p[]
end

"""
    obj_nonlinprog(estim::MovingHorizonEstimator, model::SimModel, V̂, Z̃)

Objective function for the [`MovingHorizonEstimator`](@ref).

The function `dot(x, A, x)` is a performant way of calculating `x'*A*x`.
"""
function obj_nonlinprog(
    estim::MovingHorizonEstimator, ::SimModel, V̂, Z̃::Vector{T}
) where {T<:Real}
    Nk = estim.Nk[]
    nYm, nŴ, nx̂, invP̄ = Nk*estim.nym, Nk*estim.nx̂, estim.nx̂, estim.invP̄
    invQ̂_Nk, invR̂_Nk = @views estim.invQ̂_He[1:nŴ, 1:nŴ], estim.invR̂_He[1:nYm, 1:nYm]
    x̂arr, Ŵ, V̂ = @views Z̃[1:nx̂], Z̃[nx̂+1:nx̂+nŴ], V̂[1:nYm]
    x̄ = estim.x̂arr_old - x̂arr
    return dot(x̄, invP̄, x̄) + dot(Ŵ, invQ̂_Nk, Ŵ) + dot(V̂, invR̂_Nk, V̂)
end

"""
    predict!(V̂, X̂, estim::MovingHorizonEstimator, model::SimModel, Z̃) -> V̂, X̂

Compute the `V̂` vector and `X̂` vectors for the `MovingHorizonEstimator`.

The method mutates `V̂` and `X̂` vector arguments. The vector `V̂` is the estimated sensor
noises `V̂` from ``k-N_k+1`` to ``k``. The `X̂` vector is estimated states from ``k-N_k+2`` to
``k+1``.
"""
function predict!(
    V̂, X̂, estim::MovingHorizonEstimator, model::SimModel, Z̃::Vector{T}
) where {T<:Real}
    nu, nd, nx̂, nym, Nk = model.nu, model.nd, estim.nx̂, estim.nym, estim.Nk[]
    x̂ = @views Z̃[1:nx̂]
    for j=1:Nk
        u  = @views estim.U[ (1 + nu* (j-1)):(nu*j)]
        ym = @views estim.Ym[(1 + nym*(j-1)):(nym*j)]
        d  = @views estim.D[ (1 + nd* (j-1)):(nd*j)]
        ŵ  = @views Z̃[(1 + nx̂*j):(nx̂*(j+1))]
        V̂[(1 + nym*(j-1)):(nym*j)] = ym - ĥ(estim, model, x̂, d)[estim.i_ym]
        X̂[(1 + nx̂ *(j-1)):(nx̂ *j)] = f̂(estim, model, x̂, u, d) + ŵ
        x̂ = @views X̂[(1 + nx̂*(j-1)):(nx̂*j)]
    end
    return V̂, X̂
end

"""
    con_nonlinprog!(g, estim::MovingHorizonEstimator, model::SimModel, X̂)

Nonlinear constrains for [`MovingHorizonEstimator`](@ref).
"""
function con_nonlinprog!(g, estim::MovingHorizonEstimator, ::SimModel, X̂)
    nX̂con, nX̂ = length(estim.con.X̂min), estim.nx̂ *estim.Nk[]
    nV̂con, nV̂ = length(estim.con.V̂min), estim.nym*estim.Nk[]
    for i in eachindex(g)
        estim.con.i_g[i] || continue
        if i ≤ nX̂con
            j = i
            (j ≤ nX̂) && (g[i] = estim.con.X̂min[j] - X̂[j])
        elseif i ≤ 2nX̂con
            j = i - nX̂con
            (j ≤ nX̂) && (g[i] = X̂[j] - estim.con.X̂max[j])
        elseif i ≤ 2nX̂con + nV̂con
            j = i - 2nX̂con
            (j ≤ nV̂) && (g[i] = estim.con.V̂min[j] - V̂[j])
        else
            j = i - 2nX̂con - nV̂con
            (j ≤ nV̂) && (g[i] = V̂[j] - estim.con.V̂max[j])
        end
    end
    return g
end