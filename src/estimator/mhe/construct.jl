const DEFAULT_LINMHE_OPTIMIZER    = OSQP.MathOptInterfaceOSQP.Optimizer
const DEFAULT_NONLINMHE_OPTIMIZER = optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes")

@doc raw"""
    MovingHorizonEstimator(model::SimModel; <keyword arguments>)

Construct a moving horizon estimator (MHE) based on `model` ([`LinModel`](@ref) or [`NonLinModel`](@ref)).

It can handle constraints on the estimates, see [`setconstraint!`](@ref). Additionally, 
`model` is not linearized like the [`ExtendedKalmanFilter`](@ref), and the probability 
distribution is not approximated like the [`UnscentedKalmanFilter`](@ref). The computational
costs are drastically higher, however, since it minimizes the following objective function
at each discrete time ``k``:
```math
\min_{\mathbf{x̂}_k(k-N_k+1), \mathbf{Ŵ}, ϵ}   \mathbf{x̄}' \mathbf{P̄}^{-1}       \mathbf{x̄} 
                                            + \mathbf{Ŵ}' \mathbf{Q̂}_{N_k}^{-1} \mathbf{Ŵ}  
                                            + \mathbf{V̂}' \mathbf{R̂}_{N_k}^{-1} \mathbf{V̂}
                                            + C ϵ^2
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
The estimation horizon ``H_e`` limits the window length: 
```math
N_k =                     \begin{cases} 
    k + 1   &  k < H_e    \\
    H_e     &  k ≥ H_e    \end{cases}
```
The vectors ``\mathbf{Ŵ}`` and ``\mathbf{V̂}`` encompass the estimated process noise
``\mathbf{ŵ}(k-j)`` and sensor noise ``\mathbf{v̂}(k-j)`` from ``j=N_k-1`` to ``0``. The 
Extended Help defines the two vectors. See [`UnscentedKalmanFilter`](@ref) for details on 
the augmented process model and ``\mathbf{R̂}, \mathbf{Q̂}`` covariances. The matrix 
``\mathbf{P̂}_{k-N_k}(k-N_k+1)`` is estimated with an [`ExtendedKalmanFilter`](@ref).

!!! warning
    See the Extended Help if you get an error like:    
    `MethodError: no method matching (::var"##")(::Vector{ForwardDiff.Dual})`.

# Arguments
- `model::SimModel` : (deterministic) model for the estimations.
- `He=nothing` : estimation horizon ``H_e``, must be specified.
- `optim=default_optim_mhe(model)` : a [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.Model)
   with a quadratic/nonlinear optimizer for solving (default to [`Ipopt`](https://github.com/jump-dev/Ipopt.jl),
   or [`OSQP`](https://osqp.org/docs/parsers/jump.html) if `model` is a [`LinModel`](@ref)).
- `<keyword arguments>` of [`SteadyKalmanFilter`](@ref) constructor.
- `<keyword arguments>` of [`KalmanFilter`](@ref) constructor.

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_)->0.1x+u, (x,_)->2x, 10.0, 1, 1, 1);

julia> estim = MovingHorizonEstimator(model, He=5, σR=[1], σP0=[0.01])
MovingHorizonEstimator estimator with a sample time Ts = 10.0 s, Ipopt optimizer, NonLinModel and:
 5 estimation steps He
 1 manipulated inputs u (0 integrating states)
 2 states x̂
 1 measured outputs ym (1 integrating states)
 0 unmeasured outputs yu
 0 measured disturbances d
```

# Extended Help
!!! details "Extended Help"
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
    based on the augmented model functions ``\mathbf{f̂, ĥ^m}``:
    ```math
    \begin{aligned}
        \mathbf{v̂}(k-j)     &= \mathbf{y^m}(k-j) - \mathbf{ĥ^m}\Big(\mathbf{x̂}_k(k-j), \mathbf{d}(k-j)\Big) \\
        \mathbf{x̂}_k(k-j+1) &= \mathbf{f̂}\Big(\mathbf{x̂}_k(k-j), \mathbf{u}(k-j), \mathbf{d}(k-j)\Big) + \mathbf{ŵ}(k-j)
    \end{aligned}
    ```

    For [`LinModel`](@ref), the optimization is treated as a quadratic program with a
    time-varying Hessian, which is generally cheaper than nonlinear programming. For 
    [`NonLinModel`](@ref), the optimization relies on automatic differentiation (AD).
    Optimizers generally benefit from exact derivatives like AD. However, the `f` and `h` 
    functions must be compatible with this feature. See [Automatic differentiation](https://jump.dev/JuMP.jl/stable/manual/nlp/#Automatic-differentiation)
    for common mistakes when writing these functions.
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
    optim::JM = default_optim_mhe(model),
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
default_optim_mhe(::LinModel) = JuMP.Model(DEFAULT_LINMHE_OPTIMIZER, add_bridges=false)
"Else, return it with Ipopt optimizer."
default_optim_mhe(::SimModel) = JuMP.Model(DEFAULT_NONLINMHE_OPTIMIZER, add_bridges=false)

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
    init_defaultcon_mhe(model::SimModel, He, nx̂, nym, E, ex̄, Ex̂, Fx̂, Gx̂, Jx̂) -> con, Ẽ, ẽx̄

    Init `EstimatatorConstraint` struct with default parameters based on model `model`.

Also return `Ẽ` and `ẽx̄` matrices for the the augmented decision vector `Z̃`.
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

@doc raw"""
    relaxarrival(model::SimModel, nx̂, nZ̃)

TBW
"""
function relaxarrival(::SimModel{NT}, nx̂, nZ̃) where {NT<:Real}
    I_nx̂ = Matrix{NT}(I, nx̂, nx̂)
    A_x̂min, A_x̂max = [-I_nx̂ zeros(NT, nx̂, nZ̃-nx̂)], [I_nx̂ zeros(NT, nx̂, nZ̃-nx̂)]
    return A_x̂min, A_x̂max
end

@doc raw"""
    relaxX̂(model::SimModel, Ex̂)

TBW
"""
function relaxX̂(::SimModel{NT}, Ex̂) where {NT<:Real}
    A_X̂min, A_X̂max = -Ex̂, Ex̂
    return A_X̂min, A_X̂max 
end

@doc raw"""
    relaxŴ(model::SimModel, He, nx̂, nŵ)

TBW
"""
function relaxŴ(::SimModel{NT}, He, nx̂, nŵ) where {NT<:Real}
    I_nŴ = Matrix{NT}(I, nŵ*He, nŵ*He)
    A = [zeros(NT, nŵ*He, nx̂) I_nŴ]
    A_Ŵmin, A_Ŵmax = -A, A
    return A_Ŵmin, A_Ŵmax
end

@doc raw"""
    relaxV̂(model::SimModel, E)

TBW
"""
function relaxV̂(::SimModel{NT}, E) where {NT<:Real}
    A_V̂min, A_V̂max = -E, E
    return A_V̂min, A_V̂max
end

@doc raw"""
    init_predmat_mhe(
        model::LinModel, He, i_ym, Â, B̂u, Ĉ, B̂d, D̂d
    ) -> E, F, G, J, ex̄, fx̄, Ex̂, Fx̂, Gx̂, Jx̂

Construct the MHE prediction matrices for [`LinModel`](@ref) `model`.

Introducing the vector ``\mathbf{Z} = [\begin{smallmatrix} \mathbf{x̂}_k(k-H_e+1) 
\\ \mathbf{Ŵ} \end{smallmatrix}]`` with the decision variables, the estimated sensor
noises from time ``k-H_e+1`` to ``k`` are computed by:
```math
\begin{aligned}
\mathbf{V̂} = \mathbf{Y^m - Ŷ^m} &= \mathbf{E Z + G U + J D + Y^m}     \\
                                &= \mathbf{E Z + F}
\end{aligned}
```
in which ``\mathbf{U, D}`` and ``\mathbf{Y^m}`` contains respectively the manipulated
inputs, measured disturbances and measured outputs from time ``k-H_e+1`` to ``k``. The
method also returns similar matrices but for the estimation error at arrival:
```math
\mathbf{x̄} = \mathbf{x̂}_{k-H_e}(k-H_e+1) - \mathbf{x̂}_{k}(k-H_e+1) = \mathbf{e_x̄ Z + f_x̄}
```
Lastly, the estimated states from time ``k-H_e+2`` to ``k+1`` are given by the equation:
```math
\begin{aligned}
\mathbf{X̂}  &= \mathbf{E_x̂ Z + G_x̂ U + J_x̂ D} \\
            &= \mathbf{E_x̂ Z + F_x̂}
\end{aligned}
```
All these equations omit the operating points ``\mathbf{u_{op}, y_{op}, d_{op}}``. These
matrices are truncated when ``N_k < H_e`` (at the beginning).

# Extended Help
!!! details "Extended Help"
    Using the augmented matrices ``\mathbf{Â, B̂_u, Ĉ, B̂_d, D̂_d}``, the prediction matrices
    for the sensor noises are computed by (notice the minus signs after the equalities):
    ```math
    \begin{aligned}
    \mathbf{E} &= - \begin{bmatrix}
        \mathbf{Ĉ^m}\mathbf{A}^{0}                  & \mathbf{0}                                    & \cdots & \mathbf{0}   \\ 
        \mathbf{Ĉ^m}\mathbf{Â}^{1}                  & \mathbf{Ĉ^m}\mathbf{A}^{0}                    & \cdots & \mathbf{0}   \\ 
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
    fx̄ = zeros(NT, nx̂)     # dummy fx̄ vector (updated just before optimization)
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

"""
    init_optimization!(estim::MovingHorizonEstimator, model::SimModel, optim)

Init the quadratic optimization of [`MovingHorizonEstimator`](@ref).
"""
function init_optimization!(
    estim::MovingHorizonEstimator, ::LinModel, optim::JuMP.GenericModel
)
    nZ̃ = length(estim.Z̃)
    set_silent(optim)
    limit_solve_time(estim.optim, estim.model.Ts)
    @variable(optim, Z̃var[1:nZ̃])
    A = estim.con.A[estim.con.i_b, :]
    b = estim.con.b[estim.con.i_b]
    @constraint(optim, linconstraint, A*Z̃var .≤ b)
    @objective(optim, Min, obj_quadprog(Z̃var, estim.H̃, estim.q̃))
    return nothing
end

"""
    init_optimization!(estim::MovingHorizonEstimator, model::SimModel, optim)

Init the nonlinear optimization of [`MovingHorizonEstimator`](@ref).
"""
function init_optimization!(
    estim::MovingHorizonEstimator, model::SimModel, optim::JuMP.GenericModel{JNT},
) where JNT<:Real
    He, con = estim.He, estim.con
    nV̂, nX̂, ng = He*estim.nym, He*estim.nx̂, length(con.i_g)
    # --- variables and linear constraints ---
    nZ̃ = length(estim.Z̃)
    set_silent(optim)
    limit_solve_time(estim.optim, estim.model.Ts)
    @variable(optim, Z̃var[1:nZ̃])
    A = estim.con.A[con.i_b, :]
    b = estim.con.b[con.i_b]
    @constraint(optim, linconstraint, A*Z̃var .≤ b)
    # --- nonlinear optimization init ---
    # see init_optimization!(mpc::NonLinMPC, optim) for details on the inspiration
    Jfunc, gfunc = let estim=estim, model=model, nZ̃=nZ̃ , nV̂=nV̂, nX̂=nX̂, ng=ng
        last_Z̃tup_float, last_Z̃tup_dual = nothing, nothing
        V̂_cache::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, nV̂), nZ̃ + 3)
        g_cache::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, ng), nZ̃ + 3)
        X̂_cache::DiffCache{Vector{JNT}, Vector{JNT}} = DiffCache(zeros(JNT, nX̂), nZ̃ + 3)
        function Jfunc(Z̃tup::JNT...)
            V̂ = get_tmp(V̂_cache, Z̃tup[1])
            Z̃ = collect(Z̃tup)
            if Z̃tup !== last_Z̃tup_float
                g = get_tmp(g_cache, Z̃tup[1])
                X̂ = get_tmp(X̂_cache, Z̃tup[1])
                V̂, X̂ = predict!(V̂, X̂, estim, model, Z̃)
                g = con_nonlinprog!(g, estim, model, X̂, V̂)
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
                g = con_nonlinprog!(g, estim, model, X̂, V̂)
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
                g = con_nonlinprog!(g, estim, model, X̂, V̂)
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
                g = con_nonlinprog!(g, estim, model, X̂, V̂)
                last_Z̃tup_dual = Z̃tup
            end
            return g[i]
        end
        gfunc = [(Z̃...) -> gfunc_i(i, Z̃) for i in 1:ng]
        Jfunc, gfunc
    end
    register(optim, :Jfunc, nZ̃, Jfunc, autodiff=true)
    @NLobjective(optim, Min, Jfunc(Z̃var...))
    if ng ≠ 0
        for i in eachindex(con.X̂min)
            sym = Symbol("g_X̂min_$i")
            register(optim, sym, nZ̃, gfunc[i], autodiff=true)
        end
        i_end_X̂min = nX̂
        for i in eachindex(con.X̂max)
            sym = Symbol("g_X̂max_$i")
            register(optim, sym, nZ̃, gfunc[i_end_X̂min+i], autodiff=true)
        end
        i_end_X̂max = 2*nX̂
        for i in eachindex(con.V̂min)
            sym = Symbol("g_V̂min_$i")
            register(optim, sym, nZ̃, gfunc[i_end_X̂max+i], autodiff=true)
        end
        i_end_V̂min = 2*nX̂ + nV̂
        for i in eachindex(con.V̂max)
            sym = Symbol("g_V̂max_$i")
            register(optim, sym, nZ̃, gfunc[i_end_V̂min+i], autodiff=true)
        end
    end
    return nothing
end


@doc raw"""
    setconstraint!(estim::MovingHorizonEstimator; <keyword arguments>) -> estim

Set the constraint parameters of the [`MovingHorizonEstimator`](@ref) `estim`.
   
It supports both soft and hard constraints on the estimated state ``\mathbf{x̂}``, process 
noise ``\mathbf{ŵ}`` and sensor noise ``\mathbf{v̂}``:
```math 
\begin{alignat*}{3}
    \mathbf{x̂_{min} - c_{x̂_{min}}} ϵ ≤&&\   \mathbf{x̂}_k(k-j+1) &≤ \mathbf{x̂_{max} + c_{x̂_{max}}} ϵ &&\qquad  j = N_k, N_k - 1, ... , 0    \\
    \mathbf{ŵ_{min} - c_{ŵ_{min}}} ϵ ≤&&\     \mathbf{ŵ}(k-j+1) &≤ \mathbf{ŵ_{max} + c_{ŵ_{max}}} ϵ &&\qquad  j = N_k, N_k - 1, ... , 1    \\
    \mathbf{v̂_{min} - c_{v̂_{min}}} ϵ ≤&&\     \mathbf{v̂}(k-j+1) &≤ \mathbf{v̂_{max} + c_{v̂_{max}}} ϵ &&\qquad  j = N_k, N_k - 1, ... , 1
\end{alignat*}
```
and also ``ϵ ≥ 0``. All the constraint parameters are vector. Use `±Inf` values when there
is no bound. The constraint softness parameters ``\mathbf{c}``, also called equal concern
for relaxation, are non-negative values that specify the softness of the associated bound.
Use `0.0` values for hard constraints. The process and sensor noise constraints are all soft
by default. Notice that constraining the estimated sensor noises is equivalent to bounding 
the innovation term, since ``\mathbf{v̂}(k) = \mathbf{y^m}(k) - \mathbf{ŷ^m}(k)``. See 
Extended Help for details on model augmentation and time-varying constraints.

# Arguments
!!! info
    The default constraints are mentioned here for clarity but omitting a keyword argument 
    will not re-assign to its default value (defaults are set at construction only). The
    same applies for [`PredictiveController`](@ref).

- `estim::MovingHorizonEstimator` : moving horizon estimator to set constraints.
- `x̂min = fill(-Inf,nx̂)`  : augmented state lower bounds ``\mathbf{x̂_{min}}``.
- `x̂max = fill(+Inf,nx̂)`  : augmented state upper bounds ``\mathbf{x̂_{max}}``.
- `ŵmin = fill(-Inf,nx̂)`  : augmented process noise lower bounds ``\mathbf{ŵ_{min}}``.
- `ŵmax = fill(+Inf,nx̂)`  : augmented process noise upper bounds ``\mathbf{ŵ_{max}}``.
- `v̂min = fill(-Inf,nym)` : sensor noise lower bounds ``\mathbf{v̂_{min}}``.
- `v̂max = fill(+Inf,nym)` : sensor noise upper bounds ``\mathbf{v̂_{max}}``.
- `c_x̂min = fill(0.0,nx̂)`  : `x̂min` softness weights ``\mathbf{c_{x̂_{min}}}``.
- `c_x̂max = fill(0.0,nx̂)`  : `x̂max` softness weights ``\mathbf{c_{x̂_{max}}}``.
- `c_ŵmin = fill(1.0,nx̂)`  : `ŵmin` softness weights ``\mathbf{c_{ŵ_{min}}}``.
- `c_ŵmax = fill(1.0,nx̂)`  : `ŵmax` softness weights ``\mathbf{c_{ŵ_{max}}}``.
- `c_v̂min = fill(1.0,nym)` : `v̂min` softness weights ``\mathbf{c_{v̂_{min}}}``.
- `c_v̂max = fill(1.0,nym)` : `v̂max` softness weights ``\mathbf{c_{v̂_{max}}}``.
- all the keyword arguments above but with a capital letter, e.g. `X̂max` or `C_ŵmax` : for
  time-varying constraints (see Extended Help).

# Examples
```jldoctest
julia> estim = MovingHorizonEstimator(LinModel(ss(0.5,1,1,0,1)), He=3);

julia> estim = setconstraint!(estim, x̂min=[-50, -50], x̂max=[50, 50])
MovingHorizonEstimator estimator with a sample time Ts = 1.0 s, OSQP optimizer, LinModel and:
 3 estimation steps He
 1 manipulated inputs u (0 integrating states)
 2 states x̂
 1 measured outputs ym (1 integrating states)
 0 unmeasured outputs yu
 0 measured disturbances d
```

# Extended Help
!!! details "Extended Help"
    Note that the state ``\mathbf{x̂}`` and process noise ``\mathbf{ŵ}`` constraints are 
    applied on the augmented model, detailed in [`SteadyKalmanFilter`](@ref) Extended Help. 

    For variable constraints, the bounds can be modified after calling [`updatestate!`](@ref),
    that is, at runtime, except for `±Inf` bounds. Time-varying constraints over the
    estimation horizon ``H_e`` are also possible, mathematically defined as:
    ```math 
    \begin{alignat*}{3}
        \mathbf{X̂_{min} - C_{x̂_{min}}} ϵ ≤&&\ \mathbf{X̂} &≤ \mathbf{X̂_{max} + C_{x̂_{max}}} ϵ \\
        \mathbf{Ŵ_{min} - C_{ŵ_{min}}} ϵ ≤&&\ \mathbf{Ŵ} &≤ \mathbf{Ŵ_{max} + C_{ŵ_{max}}} ϵ \\
        \mathbf{V̂_{min} - C_{v̂_{min}}} ϵ ≤&&\ \mathbf{V̂} &≤ \mathbf{V̂_{max} + C_{v̂_{max}}} ϵ
    \end{alignat*}
    ```
    For this, use the same keyword arguments as above but with a capital letter:
    - `X̂min` / `X̂max` / `C_x̂min` / `C_x̂max` : ``\mathbf{X̂}`` constraints `(nx̂*(He+1),)`.
    - `Ŵmin` / `Ŵmax` / `C_ŵmin` / `C_ŵmax` : ``\mathbf{Ŵ}`` constraints `(nx̂*He,)`.
    - `V̂min` / `V̂max` / `C_v̂min` / `C_v̂max` : ``\mathbf{V̂}`` constraints `(nym*He,)`.
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
    isnothing(V̂min) && !isnothing(v̂min) && (V̂min = repeat(v̂min, He))
    isnothing(V̂max) && !isnothing(v̂max) && (V̂max = repeat(v̂max, He))
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
    init_matconstraint_mhe(model::LinModel, 
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
    for i in findall(.!isinf.(con.V̂min))
        f_sym = Symbol("g_V̂min_$(i)")
        add_nonlinear_constraint(optim, :($(f_sym)($(Z̃var...)) <= 0))
    end
    for i in findall(.!isinf.(con.V̂max))
        f_sym = Symbol("g_V̂max_$(i)")
        add_nonlinear_constraint(optim, :($(f_sym)($(Z̃var...)) <= 0))
    end
    return nothing
end
