"""
Abstract supertype of all transcription methods of [`PredictiveController`](@ref).

The module currently supports [`SingleShooting`](@ref), [`MultipleShooting`](@ref) and
[`TrapezoidalCollocation`](@ref) transcription methods.
"""
abstract type TranscriptionMethod end
abstract type ShootingMethod    <: TranscriptionMethod end
abstract type CollocationMethod <: TranscriptionMethod end

@doc raw"""
    SingleShooting()

Construct a direct single shooting [`TranscriptionMethod`](@ref).

The decision variable in the optimization problem is (excluding the slack ``ϵ`` and without
any custom move blocking):
```math
\mathbf{Z} = \mathbf{ΔU} =          \begin{bmatrix} 
    \mathbf{Δu}(k+0)                \\ 
    \mathbf{Δu}(k+1)                \\ 
    \vdots                          \\ 
    \mathbf{Δu}(k+H_c-1)            \end{bmatrix}
```
This method computes the predictions by calling the augmented discrete-time model
recursively over the prediction horizon ``H_p`` in the objective function, or by updating
the linear coefficients of the quadratic optimization for [`LinModel`](@ref). It is 
generally  more efficient for small control horizon ``H_c``, stable and mildly nonlinear
plant model/constraints.
"""
struct SingleShooting <: ShootingMethod end

@doc raw"""
    MultipleShooting(; f_threads=false, h_threads=false)

Construct a direct multiple shooting [`TranscriptionMethod`](@ref).

The decision variable is (excluding ``ϵ``):
```math
\mathbf{Z} = \begin{bmatrix} \mathbf{ΔU} \\ \mathbf{X̂_0} \end{bmatrix}
```
thus it also includes the predicted states, expressed as deviation vectors from the
operating point ``\mathbf{x̂_{op}}`` (see [`augment_model`](@ref)):
```math
\mathbf{X̂_0} = \mathbf{X̂ - X̂_{op}} =            \begin{bmatrix} 
    \mathbf{x̂}_i(k+1)     - \mathbf{x̂_{op}}     \\ 
    \mathbf{x̂}_i(k+2)     - \mathbf{x̂_{op}}     \\ 
    \vdots                                      \\ 
    \mathbf{x̂}_i(k+H_p)   - \mathbf{x̂_{op}}     \end{bmatrix}
```
where ``\mathbf{x̂}_i(k+j)`` is the state prediction for time ``k+j``, estimated by the
observer at time ``i=k`` or ``i=k-1`` depending on its `direct` flag. Note that 
``\mathbf{X̂_0 = X̂}`` if the operating point is zero, which is typically the case in practice
for [`NonLinModel`](@ref). 
    
This transcription computes the predictions by calling the augmented discrete-time model
in the equality constraint function recursively over ``H_p``, or by updating the linear
equality constraint vector for [`LinModel`](@ref). It is generally more efficient for large
control horizon ``H_c``, unstable or highly nonlinear models/constraints. Multithreading
with `f_threads` or `h_threads` keyword arguments can be advantageous if ``\mathbf{f}`` or 
``\mathbf{h}`` in the [`NonLinModel`](@ref) is expensive to evaluate, respectively.

Sparse optimizers like `OSQP` or `Ipopt` and sparse Jacobian computations are recommended
for this transcription method.
"""
struct MultipleShooting <: ShootingMethod 
    f_threads::Bool
    h_threads::Bool
    function MultipleShooting(; f_threads=false, h_threads=false)
        return new(f_threads, h_threads)
    end
end

@doc raw"""
    TrapezoidalCollocation(h::Int=0; f_threads=false, h_threads=false)

Construct an implicit trapezoidal [`TranscriptionMethod`](@ref) with `h`th order hold.

This is the simplest collocation method. It supports continuous-time [`NonLinModel`](@ref)s
only. The decision variables are the same as for [`MultipleShooting`](@ref), hence similar
computational costs. See the same docstring for descriptions of `f_threads` and `h_threads`
keywords. The `h` argument is `0` or `1`, for piecewise constant or linear manipulated 
inputs ``\mathbf{u}`` (`h=1` is slightly less expensive). Note that the various [`DiffSolver`](@ref) 
here assume zero-order hold, so `h=1` will induce a plant-model  mismatch if the plant is
simulated with these solvers. 

This transcription computes the predictions by calling the continuous-time model in the
equality constraint function and by using the implicit trapezoidal rule. It can handle
moderately stiff systems and is A-stable. Note that the built-in [`StateEstimator`](@ref)
will still use the `solver` provided at the construction of the [`NonLinModel`](@ref) to
estimate the plant states, not the trapezoidal rule (see `supersample` option of 
[`RungeKutta`](@ref) for stiff systems). See Extended Help for more details.

Sparse optimizers like `Ipopt` and sparse Jacobian computations are recommended for this
transcription method.

# Extended Help
!!! details "Extended Help"
    Note that the stochastic model of the unmeasured disturbances is strictly discrete-time,
    as described in [`ModelPredictiveControl.init_estimstoch`](@ref). Collocation methods
    require continuous-time dynamics. Because of this, the stochastic states are transcribed
    separately using a [`MultipleShooting`](@ref) method. See [`con_nonlinprogeq!`](@ref)
    for more details.
"""
struct TrapezoidalCollocation <: CollocationMethod
    h::Int
    no::Int
    f_threads::Bool
    h_threads::Bool
    function TrapezoidalCollocation(h::Int=0; f_threads=false, h_threads=false)
        if !(h == 0 || h == 1)
            throw(ArgumentError("h argument must be 0 or 1 for TrapezoidalCollocation."))
        end
        no = 2 # 2 collocation points per intervals for trapezoidal rule
        return new(h, no, f_threads, h_threads)
    end
end


@doc raw"""
    OrthogonalCollocation(h::Int=0, no=5; f_threads=false, h_threads=false)

Construct an orthogonal collocation on finite elements [`TranscriptionMethod`](@ref).

The `h` argument is the hold order for ``\mathbf{u}``, and `no`, the number of collocation
points. The decision variable is similar to [`MultipleShooting`](@ref), but it also includes
the collocation points (excluding ``ϵ``):
```math
\mathbf{Z} = \begin{bmatrix} \mathbf{ΔU} \\ \mathbf{X̂_0} \\ \mathbf{K} \end{bmatrix}
```
where ``\mathbf{K}`` comprises all the intermediate stages of the deterministic state only:
```math
\mathbf{K} =                            \begin{bmatrix}
    \mathbf{k}_{1}(k+0)                 \\
    \mathbf{k}_{2}(k+0)                 \\
    \vdots                              \\
    \mathbf{k}_{n_p}(k+0)               \\
    \mathbf{k}_{1}(k+1)                 \\
    \mathbf{k}_{2}(k+1)                 \\
    \vdots                              \\
    \mathbf{k}_{n_p}(k+H_p)             \end{bmatrix}
```
and ``\mathbf{k}_p(k+j)`` is the deterministic state prediction for the ``p``th collocation
point at the ``j``th stage/iterval (details in Extended Help).

# Extended Help
!!! details "Extended Help"
    See the Extended Help of [`TrapezoidalCollocation`](@ref) to understand why the 
    stochastic states are left out of the ``\mathbf{K}`` vector.
"""
struct OrthogonalCollocation <: CollocationMethod
    h::Int
    no::Int
    f_threads::Bool
    h_threads::Bool
    function OrthogonalCollocation(h::Int=0, no=5; f_threads=false, h_threads=false)
        if !(h == 0 || h == 1)
            throw(ArgumentError("h argument must be 0 or 1 for OrthogonalCollocation."))
        end
        return new(h, no, f_threads, h_threads)
    end
end

function validate_transcription(::LinModel, ::CollocationMethod)
    throw(ArgumentError("Collocation methods are not supported for LinModel."))
    return nothing
end
function validate_transcription(::NonLinModel{<:Real, <:EmptySolver}, ::CollocationMethod)
    throw(ArgumentError("Collocation methods require continuous-time NonLinModel."))
    return nothing
end
validate_transcription(::SimModel, ::TranscriptionMethod) = nothing

"Get the number of elements in the optimization decision vector `Z`."
function get_nZ(estim::StateEstimator, ::SingleShooting, Hp, Hc)
    return estim.model.nu*Hc
end
function get_nZ(estim::StateEstimator, ::TranscriptionMethod, Hp, Hc)
    return estim.model.nu*Hc + estim.nx̂*Hp
end
function get_nZ(estim::StateEstimator, transcription::OrthogonalCollocation, Hp, Hc)
    return estim.model.nu*Hc + estim.nx̂*Hp + estim.model.nx*transcription.no*Hp
end

"Get length of the `k` vector with all the solver intermediate steps or all the collocation pts."
get_nk(model::SimModel, ::ShootingMethod) = model.nk
get_nk(model::SimModel, transcription::CollocationMethod) = model.nx*transcription.no

@doc raw"""
    init_ZtoΔU(estim::StateEstimator, transcription::TranscriptionMethod, Hp, Hc) -> PΔu

Init decision variables to input increments over ``H_c`` conversion matrix `PΔu`.

The conversion from the decision variables ``\mathbf{Z}`` to ``\mathbf{ΔU}``, the input
increments over ``H_c``, is computed by:
```math
\mathbf{ΔU} = \mathbf{P_{Δu}} \mathbf{Z}
```

in which ``\mathbf{P_{Δu}}`` is defined in the Extended Help section.

# Extended Help
!!! details "Extended Help"
    Following the decision variable definition of the [`TranscriptionMethod`](@ref), the
    conversion matrix ``\mathbf{P_{Δu}}``, we have:
    - ``\mathbf{P_{Δu}} = \mathbf{I}`` if `transcription` is a [`SingleShooting`](@ref)
    - ``\mathbf{P_{Δu}} = [\begin{smallmatrix}\mathbf{I} & \mathbf{0} \end{smallmatrix}]`` otherwise.
    The matrix is store as as `SparseMatrixCSC` to support both cases efficiently.
"""
function init_ZtoΔU end

function init_ZtoΔU(
    estim::StateEstimator{NT}, ::SingleShooting, _ , Hc
) where {NT<:Real}
    PΔu = sparse(Matrix{NT}(I, estim.model.nu*Hc, estim.model.nu*Hc))
    return PΔu
end

function init_ZtoΔU(
    estim::StateEstimator{NT}, ::TranscriptionMethod, Hp, Hc
) where {NT<:Real}
    I_nu_Hc = sparse(Matrix{NT}(I, estim.model.nu*Hc, estim.model.nu*Hc))
    PΔu = [I_nu_Hc spzeros(NT, estim.model.nu*Hc, estim.nx̂*Hp)]
    return PΔu
end

@doc raw"""
    init_ZtoU(estim, transcription, Hp, Hc, nb) -> Pu, Tu

Init decision variables to inputs over ``H_p`` conversion matrices.

The conversion from the decision variables ``\mathbf{Z}`` to ``\mathbf{U}``, the manipulated
inputs over ``H_p``, is computed by:
```math
\mathbf{U} = \mathbf{P_u} \mathbf{Z} + \mathbf{T_u} \mathbf{u}(k-1)
```
The ``\mathbf{P_u}`` and ``\mathbf{T_u}`` matrices are defined in the Extended Help section.

# Extended Help
!!! details "Extended Help"
    With ``n_i``, the ``i``th element of the ``\mathbf{n_b}`` vector defined in [`move_blocking`](@ref)
    documentation, we introduce the ``\mathbf{Q}(n_i)`` matrix of size `(nu*ni, nu)`:
    ```math
    \mathbf{Q}(n_i) =       \begin{bmatrix}
        \mathbf{I}          \\
        \mathbf{I}          \\
        \vdots              \\
        \mathbf{I}          \end{bmatrix}            
    ```
    The ``\mathbf{U}`` vector and the conversion matrices are defined as:
    ```math
    \mathbf{U} = \begin{bmatrix}
        \mathbf{u}(k + 0)                                                                   \\
        \mathbf{u}(k + 1)                                                                   \\
        \vdots                                                                              \\
        \mathbf{u}(k + H_p - 1)                                                             \end{bmatrix} , \quad
    \mathbf{P_u^†} = \begin{bmatrix}
        \mathbf{Q}(n_1)         & \mathbf{0}            & \cdots    & \mathbf{0}            \\
        \mathbf{Q}(n_2)         & \mathbf{Q}(n_2)       & \cdots    & \mathbf{0}            \\
        \vdots                  & \vdots                & \ddots    & \vdots                \\
        \mathbf{Q}(n_{H_c})     & \mathbf{Q}(n_{H_c})   & \cdots    & \mathbf{Q}(n_{H_c})   \end{bmatrix} , \quad
    \mathbf{T_u} = \begin{bmatrix}
        \mathbf{I}                                                                          \\
        \mathbf{I}                                                                          \\
        \vdots                                                                              \\
        \mathbf{I}                                                                          \end{bmatrix}
    ```
    and, depending on the transcription method, we have:
    - ``\mathbf{P_u} = \mathbf{P_u^†}`` if `transcription` is a [`SingleShooting`](@ref)
    - ``\mathbf{P_u} = [\begin{smallmatrix}\mathbf{P_u^†} & \mathbf{0} \end{smallmatrix}]``
      if `transcription` is a [`MultipleShooting`](@ref)
    The conversion matrices are stored as `SparseMatrixCSC` since it was benchmarked that it
    is generally more performant than normal dense matrices, even for small `nu`, `Hp` and 
    `Hc` values. Using `Bool` element type and `BitMatrix` is also slower.
"""
function init_ZtoU(
    estim::StateEstimator{NT}, transcription::TranscriptionMethod, Hp, Hc, nb
) where {NT<:Real}
    nu = estim.model.nu
    I_nu = sparse(Matrix{NT}(I, nu, nu))
    PuDagger = Matrix{NT}(undef, nu*Hp, nu*Hc)
    for i=1:Hc
        ni    = nb[i]
        Q_ni  = repeat(I_nu, ni, 1)
        iRows = (1:nu*ni) .+ @views nu*sum(nb[1:i-1])
        PuDagger[iRows, :] = [repeat(Q_ni, 1, i) spzeros(nu*ni, nu*(Hc-i))]
    end
    PuDagger = sparse(PuDagger)
    Pu = init_PUmat(estim, transcription, Hp, Hc, PuDagger)
    Tu = repeat(I_nu, Hp)
    return Pu, Tu
end

function init_PUmat(_,::SingleShooting,_,_,PuDagger::AbstractMatrix{NT}) where NT<:Real
    return PuDagger
end
function init_PUmat(
    estim, ::TranscriptionMethod, Hp, _ , PuDagger::AbstractMatrix{NT}
) where NT<:Real
    return [PuDagger spzeros(NT, estim.model.nu*Hp, estim.nx̂*Hp)]
end

@doc raw"""
    init_predmat(
        model::LinModel, estim, transcription::SingleShooting, Hp, Hc, nb
    ) -> E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂ 

Construct the prediction matrices for [`LinModel`](@ref) and [`SingleShooting`](@ref).

The model predictions are evaluated from the deviation vectors (see [`setop!`](@ref)), the
decision variable ``\mathbf{Z = ΔU}`` (with a [`SingleShooting`](@ref) transcription), and:
```math
\begin{aligned}
    \mathbf{Ŷ_0} &= \mathbf{E Z} + \mathbf{G d_0}(k) + \mathbf{J D̂_0} 
                                 + \mathbf{K x̂_0}(k) + \mathbf{V u_0}(k-1) 
                                 + \mathbf{B}        + \mathbf{Ŷ_s}                      \\
                 &= \mathbf{E Z} + \mathbf{F}
\end{aligned}
```
in which ``\mathbf{x̂_0}(k) = \mathbf{x̂}_i(k) - \mathbf{x̂_{op}}``, with ``i = k`` if 
`estim.direct==true`, otherwise ``i = k - 1``. The predicted outputs ``\mathbf{Ŷ_0}`` and
measured disturbances ``\mathbf{D̂_0}`` respectively include ``\mathbf{ŷ_0}(k+j)`` and 
``\mathbf{d̂_0}(k+j)`` values with ``j=1`` to ``H_p``, and input increments ``\mathbf{ΔU}``,
``\mathbf{Δu}(k+j_ℓ)`` from ``ℓ=0`` to ``H_c-1``. The vector ``\mathbf{B}`` contains the
contribution for non-zero state ``\mathbf{x̂_{op}}`` and state update ``\mathbf{f̂_{op}}``
operating points (for linearization at non-equilibrium point, see [`linearize`](@ref)). The
stochastic predictions ``\mathbf{Ŷ_s=0}`` if `estim` is not a [`InternalModel`](@ref), see
[`init_stochpred`](@ref). The method also computes similar matrices for the predicted
terminal state at ``k+H_p``:
```math
\begin{aligned}
    \mathbf{x̂_0}(k+H_p) &= \mathbf{e_x̂ Z}  + \mathbf{g_x̂ d_0}(k)   + \mathbf{j_x̂ D̂_0} 
                                           + \mathbf{k_x̂ x̂_0}(k) + \mathbf{v_x̂ u_0}(k-1)
                                           + \mathbf{b_x̂}                                 \\
                        &= \mathbf{e_x̂ Z}  + \mathbf{f_x̂}
\end{aligned}
```
The matrices ``\mathbf{E, G, J, K, V, B, e_x̂, g_x̂, j_x̂, k_x̂, v_x̂, b_x̂}`` are defined in the
Extended Help section. The ``\mathbf{F}`` and ``\mathbf{f_x̂}`` vectors are  recalculated at
each control period ``k``, see [`initpred!`](@ref) and [`linconstraint!`](@ref).

# Extended Help
!!! details "Extended Help"
    Using the augmented matrices ``\mathbf{Â, B̂_u, Ĉ, B̂_d, D̂_d}`` in `estim` (see 
    [`augment_model`](@ref)), and the following two functions with integer arguments:
    ```math
    \begin{aligned}
    \mathbf{Q}(i, m, b) &= \begin{bmatrix}
        \mathbf{Ĉ S}(i-b+0)\mathbf{B̂_u}             \\
        \mathbf{Ĉ S}(i-b+1)\mathbf{B̂_u}             \\
        \vdots                                      \\
        \mathbf{Ĉ S}(m-b-1)\mathbf{B̂_u}
    \end{bmatrix}                                   \\
    \mathbf{S}(m) &= ∑_{ℓ=0}^m \mathbf{Â}^ℓ      
    \end{aligned}
    ```
    the prediction matrices are computed from the ``j_ℓ`` integers introduced in the 
    [`move_blocking`](@ref) documentation and the following equations:
    ```math
    \begin{aligned}
    \mathbf{E} &= \begin{bmatrix}
        \mathbf{Q}(j_0, j_1, j_0)           & \mathbf{0}                          & \cdots & \mathbf{0}                                \\
        \mathbf{Q}(j_1, j_2, j_0)           & \mathbf{Q}(j_1, j_2, j_1)           & \cdots & \mathbf{0}                                \\
        \vdots                              & \vdots                              & \ddots & \vdots                                    \\
        \mathbf{Q}(j_{H_c-1}, j_{H_c}, j_0) & \mathbf{Q}(j_{H_c-1}, j_{H_c}, j_1) & \cdots & \mathbf{Q}(j_{H_c-1}, j_{H_c}, j_{H_c-1}) \end{bmatrix} \\
    \mathbf{G} &= \begin{bmatrix}
        \mathbf{Ĉ}\mathbf{Â}^{0} \mathbf{B̂_d}     \\ 
        \mathbf{Ĉ}\mathbf{Â}^{1} \mathbf{B̂_d}     \\ 
        \vdots                                    \\
        \mathbf{Ĉ}\mathbf{Â}^{H_p-1} \mathbf{B̂_d} \end{bmatrix} \\
    \mathbf{J} &= \begin{bmatrix}
        \mathbf{D̂_d}                              & \mathbf{0}                                & \cdots & \mathbf{0}   \\ 
        \mathbf{Ĉ}\mathbf{Â}^{0} \mathbf{B̂_d}     & \mathbf{D̂_d}                              & \cdots & \mathbf{0}   \\ 
        \vdots                                    & \vdots                                    & \ddots & \vdots       \\
        \mathbf{Ĉ}\mathbf{Â}^{H_p-2} \mathbf{B̂_d} & \mathbf{Ĉ}\mathbf{Â}^{H_p-3} \mathbf{B̂_d} & \cdots & \mathbf{D̂_d} \end{bmatrix} \\
    \mathbf{K} &= \begin{bmatrix}
        \mathbf{Ĉ}\mathbf{Â}^{1}        \\
        \mathbf{Ĉ}\mathbf{Â}^{2}        \\
        \vdots                          \\
        \mathbf{Ĉ}\mathbf{Â}^{H_p}      \end{bmatrix} \\
    \mathbf{V} &= \mathbf{Q}(0, H_p, 0) \\
    \mathbf{B} &= \begin{bmatrix}
        \mathbf{Ĉ S}(0)                 \\
        \mathbf{Ĉ S}(1)                 \\
        \vdots                          \\
        \mathbf{Ĉ S}(H_p-1)             \end{bmatrix}   \mathbf{\big(f̂_{op} - x̂_{op}\big)} 
    \end{aligned}
    ```
    For the terminal constraints, the matrices are computed with:
    ```math
    \begin{aligned}
    \mathbf{e_x̂} &= \begin{bmatrix} 
        \mathbf{S}(H_p-j_0-1)\mathbf{B̂_u} & \mathbf{S}(H_p-j_1-1)\mathbf{B̂_u} & \cdots & \mathbf{S}(H_p-j_{H_c-1}-1)\mathbf{B̂_u} \end{bmatrix} \\
    \mathbf{g_x̂} &= \mathbf{Â}^{H_p-1} \mathbf{B̂_d} \\
    \mathbf{j_x̂} &= \begin{bmatrix} 
        \mathbf{Â}^{H_p-2}\mathbf{B̂_d} & \mathbf{Â}^{H_p-3}\mathbf{B̂_d} & \cdots & \mathbf{0}                                \end{bmatrix} \\
    \mathbf{k_x̂} &= \mathbf{Â}^{H_p} \\
    \mathbf{v_x̂} &= \mathbf{S}(H_p-1)\mathbf{B̂_u} \\
    \mathbf{b_x̂} &= \mathbf{S}(H_p-1)\mathbf{\big(f̂_{op} - x̂_{op}\big)}
    \end{aligned}
    ```
    The complex structure of the ``\mathbf{E}`` and ``\mathbf{e_x̂}`` matrices is due to the
    move blocking implementation: the decision variable ``\mathbf{Z}`` only contains the
    input increment ``\mathbf{Δu}`` of the free moves (see [`move_blocking`](@ref)).
"""
function init_predmat(
    model::LinModel, estim::StateEstimator{NT}, transcription::SingleShooting, Hp, Hc, nb
) where {NT<:Real}
    Â, B̂u, Ĉ, B̂d, D̂d = estim.Â, estim.B̂u, estim.Ĉ, estim.B̂d, estim.D̂d
    nu, nx̂, ny, nd = model.nu, estim.nx̂, model.ny, model.nd
    # --- pre-compute matrix powers ---
    # Apow 3D array : Apow[:,:,1] = A^0, Apow[:,:,2] = A^1, ... , Apow[:,:,Hp+1] = A^Hp
    Âpow = Array{NT}(undef, nx̂, nx̂, Hp+1)
    Âpow[:,:,1] = I(nx̂)
    for j=2:Hp+1
        Âpow[:,:,j] = @views Âpow[:,:,j-1]*Â
    end
    # Apow_csum 3D array : Apow_csum[:,:,1] = A^0, Apow_csum[:,:,2] = A^1 + A^0, ...
    Âpow_csum  = cumsum(Âpow, dims=3)
    jℓ_data = [0; cumsum(nb)] # introduced in move_blocking docstring
    # four helper functions to improve code clarity and be similar to eqs. in docstring:
    getpower(array3D, power) = @views array3D[:,:, power+1]
    S(m)  = @views Âpow_csum[:,:, m+1]
    jℓ(ℓ) = jℓ_data[ℓ+1]
    function Q!(Q, i, m, b)
        for ℓ=0:m-i-1
            iRows = (1:ny) .+ ny*ℓ
            Q[iRows, :] = Ĉ * S(i-b+ℓ) * B̂u
        end
        return Q
    end
    # --- current state estimates x̂0 ---
    kx̂ = getpower(Âpow, Hp)
    K  = Matrix{NT}(undef, Hp*ny, nx̂)
    for j=1:Hp
        iRow = (1:ny) .+ ny*(j-1)
        K[iRow,:] = Ĉ*getpower(Âpow, j)
    end
    # --- previous manipulated inputs lastu0 ---
    vx̂ = S(Hp-1)*B̂u
    V  = Matrix{NT}(undef, Hp*ny, nu)
    Q!(V, 0, Hp, 0)
    # --- decision variables Z ---
    nZ = get_nZ(estim, transcription, Hp, Hc)
    ex̂ = Matrix{NT}(undef, nx̂, nZ)
    E  = zeros(NT, Hp*ny, nZ) 
    for j=0:Hc-1
        iCol = (1:nu) .+ nu*j
        for i=j:Hc-1
            i_Q, m_Q, b_Q = jℓ(i), jℓ(i+1), jℓ(j)
            iRow = (1:ny*nb[i+1]) .+ ny*i_Q
            Q = @views E[iRow, iCol]
            Q!(Q, i_Q, m_Q, b_Q)
        end
        ex̂[:, iCol] = S(Hp - jℓ(j) - 1)*B̂u
    end    
    # --- current measured disturbances d0 and predictions D̂0 ---
    gx̂ = getpower(Âpow, Hp-1)*B̂d
    G  = Matrix{NT}(undef, Hp*ny, nd)
    jx̂ = Matrix{NT}(undef, nx̂, Hp*nd)
    J  = repeatdiag(D̂d, Hp)
    if nd > 0
        for j=1:Hp
            iRow = (1:ny) .+ ny*(j-1)
            G[iRow,:] = Ĉ*getpower(Âpow, j-1)*B̂d
        end
        for j=1:Hp
            iRow = (ny*j+1):(ny*Hp)
            iCol = (1:nd) .+ nd*(j-1)
            J[iRow, iCol] = G[iRow .- ny*j,:]
            jx̂[:  , iCol] = j < Hp ? getpower(Âpow, Hp-j-1)*B̂d : zeros(NT, nx̂, nd)
        end
    end
    # --- state x̂op and state update f̂op operating points ---
    coef_bx̂ = S(Hp-1)
    coef_B  = Matrix{NT}(undef, ny*Hp, nx̂)
    for j=1:Hp
        iRow = (1:ny) .+ ny*(j-1)
        coef_B[iRow,:] = Ĉ*S(j-1)
    end
    f̂op_n_x̂op = estim.f̂op - estim.x̂op
    bx̂ = coef_bx̂ * f̂op_n_x̂op
    B  = coef_B  * f̂op_n_x̂op
    return E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂
end

@doc raw"""
    init_predmat(
        model::LinModel, estim, transcription::MultipleShooting, Hp, Hc, nb
    ) -> E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂
    
Construct the prediction matrices for [`LinModel`](@ref) and [`MultipleShooting`](@ref).

They are defined in the Extended Help section.

# Extended Help
!!! details "Extended Help"
    They are all appropriately sized zero matrices ``\mathbf{0}``, except for:
    ```math
    \begin{aligned}
    \mathbf{E}     &= [\begin{smallmatrix}\mathbf{0} & \mathbf{E^{x̂}} \end{smallmatrix}]  \\
    \mathbf{E^{x̂}} &= \text{diag}\mathbf{(Ĉ,Ĉ,...,Ĉ)}                                     \\
    \mathbf{J}     &= \text{diag}\mathbf{(D̂_d,D̂_d,...,D̂_d)}                               \\
    \mathbf{e_x̂}   &= [\begin{smallmatrix}\mathbf{0} & \mathbf{I}\end{smallmatrix}]   
    \end{aligned}
    ```
"""
function init_predmat(
    model::LinModel, estim::StateEstimator{NT}, ::MultipleShooting, Hp, Hc, nb
) where {NT<:Real}
    Ĉ, D̂d = estim.Ĉ, estim.D̂d
    nu, nx̂, ny, nd = model.nu, estim.nx̂, model.ny, model.nd
    # --- current state estimates x̂0 ---
    K = zeros(NT, Hp*ny, nx̂)
    kx̂ = zeros(NT, nx̂, nx̂)
    # --- previous manipulated inputs lastu0 ---
    V = zeros(NT, Hp*ny, nu)
    vx̂ = zeros(NT, nx̂, nu)
    # --- decision variables Z ---
    E  = [zeros(NT, Hp*ny, Hc*nu) repeatdiag(Ĉ, Hp)]
    ex̂ = [zeros(NT, nx̂, Hc*nu + (Hp-1)*nx̂) I]
    # --- current measured disturbances d0 and predictions D̂0 ---
    G  = zeros(NT, Hp*ny, nd)
    gx̂ = zeros(NT, nx̂, nd)
    J  = repeatdiag(D̂d, Hp)
    jx̂ = zeros(NT, nx̂, Hp*nd)
    # --- state x̂op and state update f̂op operating points ---
    B  = zeros(NT, Hp*ny)
    bx̂ = zeros(NT, nx̂)
    return E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂
end

"""
    init_predmat(
        model::NonLinModel, estim, transcription::SingleShooting, Hp, Hc, nb
    ) -> E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂

Return empty matrices for [`SingleShooting`](@ref) of [`NonLinModel`](@ref)
"""
function init_predmat(
    model::NonLinModel, estim::StateEstimator{NT}, transcription::SingleShooting, Hp, Hc, _
) where {NT<:Real}
    nu, nx̂, nd = model.nu, estim.nx̂, model.nd
    nZ = get_nZ(estim, transcription, Hp, Hc)
    E  = zeros(NT, 0, nZ)
    G  = zeros(NT, 0, nd)
    J  = zeros(NT, 0, nd*Hp)
    K  = zeros(NT, 0, nx̂)
    V  = zeros(NT, 0, nu)
    B  = zeros(NT, 0)
    ex̂, gx̂, jx̂, kx̂, vx̂, bx̂ = E, G, J, K, V, B
    return E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂
end

@doc raw"""
    init_predmat(
        model::NonLinModel, estim, transcription::TranscriptionMethod, Hp, Hc, nb
    ) -> E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂

Return the terminal state matrices for [`NonLinModel`](@ref) and other [`TranscriptionMethod`](@ref).

The output prediction matrices are all empty matrices. The terminal state matrices are
given in the Extended Help section.

# Extended Help
!!! details "Extended Help"
    The terminal state matrices all appropriately sized zero matrices ``\mathbf{0}``, except
    for ``\mathbf{e_x̂} = [\begin{smallmatrix}\mathbf{0} & \mathbf{I}\end{smallmatrix}]``
"""
function init_predmat(
    model::NonLinModel, estim::StateEstimator{NT}, transcription::TranscriptionMethod, Hp, Hc, _
) where {NT<:Real}
    nu, nx̂, nd = model.nu, estim.nx̂, model.nd
    nZ = get_nZ(estim, transcription, Hp, Hc)
    E  = zeros(NT, 0, nZ)
    G  = zeros(NT, 0, nd)
    J  = zeros(NT, 0, nd*Hp)
    K  = zeros(NT, 0, nx̂)
    V  = zeros(NT, 0, nu)
    B  = zeros(NT, 0)
    ex̂ = [zeros(NT, nx̂, Hc*nu + (Hp-1)*nx̂) I]
    gx̂ = zeros(NT, nx̂, nd)
    jx̂ = zeros(NT, nx̂, nd*Hp)
    kx̂ = zeros(NT, nx̂, nx̂)
    vx̂ = zeros(NT, nx̂, nu)
    bx̂ = zeros(NT, nx̂)
    return E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂
end

@doc raw"""
    init_defectmat(
        model::LinModel, estim, transcription::MultipleShooting, Hp, Hc, nb
    ) -> Eŝ, Gŝ, Jŝ, Kŝ, Vŝ, Bŝ

Init the matrices for computing the defects over the predicted states. 

Knowing that the decision vector ``\mathbf{Z}`` contains both ``\mathbf{ΔU}`` and 
``\mathbf{X̂_0}`` vectors (with a [`MultipleShooting`](@ref) transcription), an equation
similar to the prediction matrices (see [`init_predmat`](@ref)) computes the defects over
the predicted states:
```math
\begin{aligned}
    \mathbf{Ŝ} &= \mathbf{E_ŝ Z} + \mathbf{G_ŝ d_0}(k)  + \mathbf{J_ŝ D̂_0} 
                                 + \mathbf{K_ŝ x̂_0}(k)  + \mathbf{V_ŝ u_0}(k-1) 
                                 + \mathbf{B_ŝ}                                         \\
               &= \mathbf{E_ŝ Z} + \mathbf{F_ŝ}
\end{aligned}
```   
They are forced to be ``\mathbf{Ŝ = 0}`` using the optimization equality constraints. The
matrices ``\mathbf{E_ŝ, G_ŝ, J_ŝ, K_ŝ, V_ŝ, B_ŝ}`` are defined in the Extended Help section.

# Extended Help
!!! details "Extended Help"
    Using the augmented matrices ``\mathbf{Â, B̂_u, Ĉ, B̂_d, D̂_d}`` in `estim` (see 
    [`augment_model`](@ref)), the [`move_blocking`](@ref) vector ``\mathbf{n_b}``, and the
    following ``\mathbf{Q}(n_i)`` matrix of size `(nx̂*ni, nu)`:
    ```math
    \mathbf{Q}(n_i) =       \begin{bmatrix}
        \mathbf{B̂_u}        \\
        \mathbf{B̂_u}        \\
        \vdots              \\
        \mathbf{B̂_u}        \end{bmatrix}            
    ```
    The defect matrices are computed with:
    ```math
    \begin{aligned}
    \mathbf{E_ŝ} &= \begin{bmatrix}
        \mathbf{E_{ŝ}^{Δu}} & \mathbf{E_{ŝ}^{x̂}}                                                    \end{bmatrix} \\
    \mathbf{E_{ŝ}^{Δu}} &= \begin{bmatrix}
        \mathbf{Q}(n_1)     & \mathbf{0}          & \cdots & \mathbf{0}                             \\
        \mathbf{Q}(n_2)     & \mathbf{Q}(n_2)     & \cdots & \mathbf{0}                             \\
        \vdots              & \vdots              & \ddots & \vdots                                 \\
        \mathbf{Q}(n_{H_c}) & \mathbf{Q}(n_{H_c}) & \cdots & \mathbf{Q}(n_{H_c})                    \end{bmatrix} \\
    \mathbf{E_{ŝ}^{x̂}} &= \begin{bmatrix}
        -\mathbf{I} &  \mathbf{0} & \cdots &  \mathbf{0}  &  \mathbf{0}                             \\
         \mathbf{Â} & -\mathbf{I} & \cdots &  \mathbf{0}  &  \mathbf{0}                             \\
         \vdots     &  \vdots     & \ddots &  \vdots      &  \vdots                                 \\
         \mathbf{0} &  \mathbf{0} & \cdots &  \mathbf{Â}  & -\mathbf{I}                             \end{bmatrix} \\
    \mathbf{G_ŝ} &= \begin{bmatrix}
        \mathbf{B̂_d} \\ \mathbf{0} \\ \vdots \\ \mathbf{0}                                          \end{bmatrix} \\
    \mathbf{J_ŝ} &= \begin{bmatrix}
        \mathbf{0}   & \mathbf{0}   & \cdots & \mathbf{0}   & \mathbf{0}                            \\
        \mathbf{B̂_d} & \mathbf{0}   & \cdots & \mathbf{0}   & \mathbf{0}                            \\
        \vdots       & \vdots       & \ddots & \vdots       & \vdots                                \\
        \mathbf{0}   & \mathbf{0}   & \cdots & \mathbf{B̂_d} & \mathbf{0}                            \end{bmatrix} \\
    \mathbf{K_ŝ} &= \begin{bmatrix}
        \mathbf{Â} \\ \mathbf{0} \\ \vdots \\ \mathbf{0}                                            \end{bmatrix} \\
    \mathbf{V_ŝ} &= \begin{bmatrix}
        \mathbf{B̂_u} \\ \mathbf{B̂_u} \\ \vdots \\ \mathbf{B̂_u}                                      \end{bmatrix} \\
    \mathbf{B_ŝ} &= \begin{bmatrix}
        \mathbf{f̂_{op} - x̂_{op}} \\ \mathbf{f̂_{op} - x̂_{op}} \\ \vdots \\ \mathbf{f̂_{op} - x̂_{op}}  \end{bmatrix}
    \end{aligned}
    ```
    The ``\mathbf{E_ŝ^{Δu}}`` matrix structure is due to the move blocking implementation:
    the ``\mathbf{ΔU}`` vector only contains the input increment of the free moves 
    (see [`move_blocking`](@ref)).
"""
function init_defectmat(
    model::LinModel, estim::StateEstimator{NT}, ::MultipleShooting, Hp, Hc, nb
) where {NT<:Real}
    nu, nx̂, nd = model.nu, estim.nx̂, model.nd
    Â, B̂u, B̂d = estim.Â, estim.B̂u, estim.B̂d
    # helper function to be similar to eqs. in docstring:
    function Q!(Q, ni)
        for ℓ=0:ni-1
            iRows = (1:nx̂) .+ nx̂*ℓ
            Q[iRows, :] = B̂u
        end
        return Q
    end
    # --- current state estimates x̂0 ---
    Kŝ = [Â; zeros(NT, nx̂*(Hp-1), nx̂)]
    # --- previous manipulated inputs lastu0 ---
    Vŝ = repeat(B̂u, Hp)
    # --- decision variables Z ---
    nI_nx̂ = Matrix{NT}(-I, nx̂, nx̂)
    Eŝ = [zeros(nx̂*Hp, nu*Hc) repeatdiag(nI_nx̂, Hp)]
    for j=1:Hc
        iCol = (1:nu) .+ nu*(j-1)
        for i=j:Hc
            ni = nb[i]
            iRow = (1:nx̂*ni) .+ nx̂*sum(nb[1:i-1])
            Q = @views Eŝ[iRow, iCol]
            Q!(Q, ni)
        end
    end
    for j=1:Hp-1
        iRow = (1:nx̂) .+ nx̂*j
        iCol = (1:nx̂) .+ nx̂*(j-1) .+ nu*Hc
        Eŝ[iRow, iCol] = Â
    end
    # --- current measured disturbances d0 and predictions D̂0 ---
    Gŝ = [B̂d; zeros(NT, (Hp-1)*nx̂, nd)]
    Jŝ = [zeros(nx̂, nd*Hp); repeatdiag(B̂d, Hp-1) zeros(NT, nx̂*(Hp-1), nd)]
    # --- state x̂op and state update f̂op operating points ---
    Bŝ = repeat(estim.f̂op - estim.x̂op, Hp)
    return Eŝ, Gŝ, Jŝ, Kŝ, Vŝ, Bŝ
end

"""
    init_defectmat(
        model::SimModel, estim, transcription::TranscriptionMethod, Hp, Hc, nb
    ) -> Eŝ, Gŝ, Jŝ, Kŝ, Vŝ, Bŝ

Return empty matrices for all other cases (N/A).
"""
function init_defectmat(
    model::SimModel, estim::StateEstimator{NT}, transcription::TranscriptionMethod, Hp, Hc, _
) where {NT<:Real}
    nx̂, nu, nd = estim.nx̂, model.nu, model.nd
    nZ = get_nZ(estim, transcription, Hp, Hc)
    Eŝ = zeros(NT, 0, nZ)
    Gŝ = zeros(NT, 0, nd)
    Jŝ = zeros(NT, 0, nd*Hp)
    Kŝ = zeros(NT, 0, nx̂)
    Vŝ = zeros(NT, 0, nu)
    Bŝ = zeros(NT, 0)
    return Eŝ, Gŝ, Jŝ, Kŝ, Vŝ, Bŝ
end

@doc raw"""
    init_matconstraint_mpc(
        model::LinModel, transcription::TranscriptionMethod, nc::Int,
        i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ymin, i_Ymax, i_Wmin, i_Wmax, i_x̂min, i_x̂max,
        args...
    ) -> i_b, i_g, A, Aeq, neq

Init `i_b`, `i_g`, `neq`, and `A` and `Aeq` matrices for the all the MPC constraints.

The linear and nonlinear constraints are respectively defined as:
```math
\begin{aligned} 
    \mathbf{A Z̃ }       &≤ \mathbf{b}           \\ 
    \mathbf{A_{eq} Z̃}   &= \mathbf{b_{eq}}      \\
    \mathbf{g(Z̃)}       &≤ \mathbf{0}           \\
    \mathbf{g_{eq}(Z̃)}  &= \mathbf{0}           \\
\end{aligned}
```
The argument `nc` is the number of custom nonlinear inequality constraints in
``\mathbf{g_c}``. `i_b` is a `BitVector` including the indices of ``\mathbf{b}`` that are
finite numbers. `i_g` is a similar vector but for the indices of ``\mathbf{g}``. The method
also returns the ``\mathbf{A, A_{eq}}`` matrices and `neq` if `args` is provided. In such a 
case, `args`  needs to contain all the inequality and equality constraint matrices: 
`A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, A_Ymin, A_Ymax, A_Wmin, A_Wmax, A_x̂min, A_x̂max, A_ŝ`. 
The integer `neq` is the number of nonlinear equality constraints in ``\mathbf{g_{eq}}``.
"""
function init_matconstraint_mpc(
    ::LinModel{NT}, ::TranscriptionMethod, nc::Int,
    i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ymin, i_Ymax, i_Wmin, i_Wmax, i_x̂min, i_x̂max,
    args...
) where {NT<:Real}
    if isempty(args)
        A, Aeq, neq = nothing, nothing, nothing
    else
        (
            A_Umin,  A_Umax, 
            A_ΔŨmin, A_ΔŨmax, 
            A_Ymin,  A_Ymax, 
            A_Wmin,  A_Wmax,
            A_x̂min,  A_x̂max,  
            A_ŝ
        ) = args
        A = [
            A_Umin;  A_Umax; 
            A_ΔŨmin; A_ΔŨmax; 
            A_Ymin;  A_Ymax; 
            A_Wmin;  A_Wmax
            A_x̂min;  A_x̂max;
        ]
        Aeq = A_ŝ
        neq = 0
    end
    i_b = [i_Umin; i_Umax; i_ΔŨmin; i_ΔŨmax; i_Ymin; i_Ymax; i_Wmin; i_Wmax; i_x̂min; i_x̂max]
    i_g = trues(nc)
    return i_b, i_g, A, Aeq, neq
end

"Init `i_b` without output & terminal constraints if `NonLinModel` and `SingleShooting`."
function init_matconstraint_mpc(
    ::NonLinModel{NT}, ::SingleShooting, nc::Int,
    i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ymin, i_Ymax, i_Wmin, i_Wmax, i_x̂min, i_x̂max,
    args...
) where {NT<:Real}
    if isempty(args)
        A, Aeq, neq = nothing, nothing, nothing
    else
        A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, _ , _ , A_Wmin, A_Wmax, _ , _ , A_ŝ = args
        A   = [A_Umin; A_Umax; A_ΔŨmin; A_ΔŨmax; A_Wmin; A_Wmax]
        Aeq = A_ŝ
        neq = 0
    end
    i_b = [i_Umin; i_Umax; i_ΔŨmin; i_ΔŨmax; i_Wmin; i_Wmax]
    i_g = [i_Ymin; i_Ymax; i_x̂min;  i_x̂max; trues(nc)]
    return i_b, i_g, A, Aeq, neq
end

"Init `i_b` without output constraints if `NonLinModel` and other `TranscriptionMethod`."
function init_matconstraint_mpc(
    ::NonLinModel{NT}, ::TranscriptionMethod, nc::Int,
    i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ymin, i_Ymax, i_Wmin, i_Wmax, i_x̂min, i_x̂max,
    args...
) where {NT<:Real}
    if isempty(args)
        A, Aeq, neq = nothing, nothing, nothing
    else    
        A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, _ , _ , A_Wmin, A_Wmax, A_x̂min, A_x̂max, A_ŝ = args
        A   = [A_Umin; A_Umax; A_ΔŨmin; A_ΔŨmax; A_Wmin; A_Wmax; A_x̂min; A_x̂max]
        Aeq = A_ŝ
        nΔŨ, nZ̃ = size(A_ΔŨmin)
        neq = nZ̃ - nΔŨ
    end
    i_b = [i_Umin; i_Umax; i_ΔŨmin; i_ΔŨmax; i_Wmin; i_Wmax; i_x̂min; i_x̂max]
    i_g = [i_Ymin; i_Ymax; trues(nc)]
    return i_b, i_g, A, Aeq, neq
end

@doc raw"""
    linconstraint!(mpc::PredictiveController, model::LinModel)

Set `b` vector for the linear model inequality constraints (``\mathbf{A Z̃ ≤ b}``).

Also init ``\mathbf{f_x̂} = \mathbf{g_x̂ d_0}(k) + \mathbf{j_x̂ D̂_0} + \mathbf{k_x̂ x̂_0}(k) + 
\mathbf{v_x̂ u_0}(k-1) + \mathbf{b_x̂}`` vector for the terminal constraints, see
[`init_predmat`](@ref). The ``\mathbf{F_w}`` vector for the custom linear constraints is
also updated, see [`relaxW`](@ref).
"""
function linconstraint!(mpc::PredictiveController, model::LinModel, ::TranscriptionMethod)
    nU, nΔŨ, nY = length(mpc.con.U0min), length(mpc.con.ΔŨmin), length(mpc.con.Y0min)
    nW = length(mpc.con.Wmin)
    nx̂, fx̂ = mpc.estim.nx̂, mpc.con.fx̂
    fx̂ .= mpc.con.bx̂
    mul!(fx̂, mpc.con.kx̂, mpc.estim.x̂0, 1, 1)
    mul!(fx̂, mpc.con.vx̂, mpc.lastu0, 1, 1)
    if model.nd > 0
        mul!(fx̂, mpc.con.gx̂, mpc.d0, 1, 1)
        mul!(fx̂, mpc.con.jx̂, mpc.D̂0, 1, 1)
    end
    linconstraint_custom!(mpc, model)
    n = 0
    mpc.con.b[(n+1):(n+nU)]  .= @. -mpc.con.U0min + mpc.Tu_lastu0
    n += nU
    mpc.con.b[(n+1):(n+nU)]  .= @. +mpc.con.U0max - mpc.Tu_lastu0
    n += nU
    mpc.con.b[(n+1):(n+nΔŨ)] .= @. -mpc.con.ΔŨmin
    n += nΔŨ
    mpc.con.b[(n+1):(n+nΔŨ)] .= @. +mpc.con.ΔŨmax
    n += nΔŨ
    mpc.con.b[(n+1):(n+nY)]  .= @. -mpc.con.Y0min + mpc.F
    n += nY
    mpc.con.b[(n+1):(n+nY)]  .= @. +mpc.con.Y0max - mpc.F
    n += nY
    mpc.con.b[(n+1):(n+nW)]  .= @. -mpc.con.Wmin + mpc.con.Fw
    n += nW
    mpc.con.b[(n+1):(n+nW)]  .= @. +mpc.con.Wmax - mpc.con.Fw
    n += nW
    mpc.con.b[(n+1):(n+nx̂)]  .= @. -mpc.con.x̂0min + fx̂
    n += nx̂
    mpc.con.b[(n+1):(n+nx̂)]  .= @. +mpc.con.x̂0max - fx̂
    if any(mpc.con.i_b) 
        lincon = mpc.optim[:linconstraint]
        JuMP.set_normalized_rhs(lincon, mpc.con.b[mpc.con.i_b])
    end
    return nothing
end

"Set `b` excluding predicted output constraints for `NonLinModel` and not `SingleShooting`."
function linconstraint!(mpc::PredictiveController, model::NonLinModel, ::TranscriptionMethod)
    nU, nΔŨ = length(mpc.con.U0min), length(mpc.con.ΔŨmin)
    nW = length(mpc.con.Wmin)
    nx̂ = mpc.estim.nx̂
    # here, updating fx̂ is not necessary since fx̂ = 0
    linconstraint_custom!(mpc, model)
    n = 0
    mpc.con.b[(n+1):(n+nU)]  .= @. -mpc.con.U0min + mpc.Tu_lastu0
    n += nU
    mpc.con.b[(n+1):(n+nU)]  .= @. +mpc.con.U0max - mpc.Tu_lastu0
    n += nU
    mpc.con.b[(n+1):(n+nΔŨ)] .= @. -mpc.con.ΔŨmin
    n += nΔŨ
    mpc.con.b[(n+1):(n+nΔŨ)] .= @. +mpc.con.ΔŨmax
    n += nΔŨ
    mpc.con.b[(n+1):(n+nW)]  .= @. -mpc.con.Wmin + mpc.con.Fw
    n += nW
    mpc.con.b[(n+1):(n+nW)]  .= @. +mpc.con.Wmax - mpc.con.Fw
    n += nW
    mpc.con.b[(n+1):(n+nx̂)]  .= @. -mpc.con.x̂0min
    n += nx̂
    mpc.con.b[(n+1):(n+nx̂)]  .= @. +mpc.con.x̂0max
    if any(mpc.con.i_b) 
        lincon = mpc.optim[:linconstraint]
        JuMP.set_normalized_rhs(lincon, mpc.con.b[mpc.con.i_b])
    end
end

"Also exclude terminal constraints for `NonLinModel` and `SingleShooting`."
function linconstraint!(mpc::PredictiveController, model::NonLinModel, ::SingleShooting)
    nU, nΔŨ = length(mpc.con.U0min), length(mpc.con.ΔŨmin)
    nW = length(mpc.con.Wmin)
    linconstraint_custom!(mpc, model)
    n = 0
    mpc.con.b[(n+1):(n+nU)]  .= @. -mpc.con.U0min + mpc.Tu_lastu0
    n += nU
    mpc.con.b[(n+1):(n+nU)]  .= @. +mpc.con.U0max - mpc.Tu_lastu0
    n += nU
    mpc.con.b[(n+1):(n+nΔŨ)] .= @. -mpc.con.ΔŨmin
    n += nΔŨ
    mpc.con.b[(n+1):(n+nΔŨ)] .= @. +mpc.con.ΔŨmax
    n += nΔŨ
    mpc.con.b[(n+1):(n+nW)]  .= @. -mpc.con.Wmin + mpc.con.Fw
    n += nW
    mpc.con.b[(n+1):(n+nW)]  .= @. +mpc.con.Wmax - mpc.con.Fw
    if any(mpc.con.i_b) 
        lincon = mpc.optim[:linconstraint]
        @views JuMP.set_normalized_rhs(lincon, mpc.con.b[mpc.con.i_b])
    end
    return nothing
end

@doc raw"""
    linconstrainteq!(
        mpc::PredictiveController, model::LinModel, transcription::MultipleShooting
    )

Set `beq` vector for the linear model equality constraints (``\mathbf{A_{eq} Z̃ = b_{eq}}``).

Also init ``\mathbf{F_ŝ} = \mathbf{G_ŝ d_0}(k) + \mathbf{J_ŝ D̂_0} + \mathbf{K_ŝ x̂_0}(k) + 
\mathbf{V_ŝ u_0}(k-1) + \mathbf{B_ŝ}``, see [`init_defectmat`](@ref).
"""
function linconstrainteq!(mpc::PredictiveController, model::LinModel, ::MultipleShooting)
    Fŝ  = mpc.con.Fŝ
    Fŝ .= mpc.con.Bŝ
    mul!(Fŝ, mpc.con.Kŝ, mpc.estim.x̂0, 1, 1)
    mul!(Fŝ, mpc.con.Vŝ, mpc.lastu0, 1, 1)
    if model.nd > 0
        mul!(Fŝ, mpc.con.Gŝ, mpc.d0, 1, 1)
        mul!(Fŝ, mpc.con.Jŝ, mpc.D̂0, 1, 1)
    end
    mpc.con.beq .= @. -Fŝ
    linconeq = mpc.optim[:linconstrainteq]
    JuMP.set_normalized_rhs(linconeq, mpc.con.beq)
    return nothing
end
linconstrainteq!(::PredictiveController, ::SimModel, ::TranscriptionMethod) = nothing

@doc raw"""
    set_warmstart!(mpc::PredictiveController, ::SingleShooting, Z̃var) -> Z̃s

Set and return the warm-start value of `Z̃var` for [`SingleShooting`](@ref) transcription.

If supported by `mpc.optim`, it warm-starts the solver at:
```math
\mathbf{Z̃_s} = 
\begin{bmatrix}
    \mathbf{Δu}(k+0|k-1)        \\ 
    \mathbf{Δu}(k+1|k-1)        \\ 
    \vdots                      \\
    \mathbf{Δu}(k+H_c-2|k-1)    \\
    \mathbf{0}                  \\
    ϵ(k-1)
\end{bmatrix}
```
where ``\mathbf{Δu}(k+j|k-1)`` is the input increment for time ``k+j`` computed at the 
last control period ``k-1``, and ``ϵ(k-1)``, the slack variable of the last control period.
"""
function set_warmstart!(mpc::PredictiveController, ::SingleShooting, Z̃var)
    nu, Hc, Z̃s = mpc.estim.model.nu, mpc.Hc, mpc.buffer.Z̃
    # --- input increments ΔU ---
    Z̃s[1:(Hc*nu-nu)] .= @views mpc.Z̃[nu+1:Hc*nu]
    Z̃s[(Hc*nu-nu+1):(Hc*nu)] .= 0
    # --- slack variable ϵ ---
    mpc.nϵ == 1 && (Z̃s[end] = mpc.Z̃[end])
    JuMP.set_start_value.(Z̃var, Z̃s)
    return Z̃s
end

@doc raw"""
    set_warmstart!(mpc::PredictiveController, ::TranscriptionMethod, Z̃var) -> Z̃s

Set and return the warm-start value of `Z̃var` for other [`TranscriptionMethod`](@ref).

It warm-starts the solver at:
```math
\mathbf{Z̃_s} =
\begin{bmatrix}
    \mathbf{Δu}(k+0|k-1)        \\ 
    \mathbf{Δu}(k+1|k-1)        \\ 
    \vdots                      \\
    \mathbf{Δu}(k+H_c-2|k-1)    \\
    \mathbf{0}                  \\
    \mathbf{x̂_0}(k+1|k-1)       \\
    \mathbf{x̂_0}(k+2|k-1)       \\
    \vdots                      \\
    \mathbf{x̂_0}(k+H_p-1|k-1)   \\
    \mathbf{x̂_0}(k+H_p-1|k-1)   \\
    ϵ(k-1)
\end{bmatrix}
```
where ``\mathbf{x̂_0}(k+j|k-1)`` is the predicted state for time ``k+j`` computed at the
last control period ``k-1``, expressed as a deviation from the operating point 
``\mathbf{x̂_{op}}``.
"""
function set_warmstart!(mpc::PredictiveController, ::TranscriptionMethod, Z̃var)
    nu, nx̂, Hp, Hc, Z̃s = mpc.estim.model.nu, mpc.estim.nx̂, mpc.Hp, mpc.Hc, mpc.buffer.Z̃
    # --- input increments ΔU ---
    Z̃s[1:(Hc*nu-nu)] .= @views mpc.Z̃[nu+1:Hc*nu]
    Z̃s[(Hc*nu-nu+1):(Hc*nu)] .= 0
    # --- predicted states X̂0 ---
    Z̃s[(Hc*nu+1):(Hc*nu+Hp*nx̂-nx̂)]       .= @views mpc.Z̃[(Hc*nu+nx̂+1):(Hc*nu+Hp*nx̂)]
    Z̃s[(Hc*nu+Hp*nx̂-nx̂+1):(Hc*nu+Hp*nx̂)] .= @views mpc.Z̃[(Hc*nu+Hp*nx̂-nx̂+1):(Hc*nu+Hp*nx̂)]
    # --- slack variable ϵ ---
    mpc.nϵ == 1 && (Z̃s[end] = mpc.Z̃[end])
    JuMP.set_start_value.(Z̃var, Z̃s)
    return Z̃s
end

getΔŨ!(ΔŨ, ::PredictiveController, ::SingleShooting, Z̃) = (ΔŨ .= Z̃)
function getΔŨ!(ΔŨ, mpc::PredictiveController, ::TranscriptionMethod, Z̃)
    # avoid explicit matrix multiplication with mpc.P̃Δu for performance:
    nΔU = mpc.Hc*mpc.estim.model.nu
    ΔŨ[1:nΔU] .= @views Z̃[1:nΔU]
    mpc.nϵ == 1 && (ΔŨ[end] = Z̃[end])
    return ΔŨ
end
getU0!(U0, mpc::PredictiveController, Z̃) = (mul!(U0, mpc.P̃u, Z̃) .+ mpc.Tu_lastu0)

@doc raw"""
    predict!(
        Ŷ0, x̂0end, _ , _ , _ ,
        mpc::PredictiveController, model::LinModel, transcription::TranscriptionMethod, 
        _ , Z̃
    ) -> Ŷ0, x̂0end

Compute the predictions `Ŷ0`, terminal states `x̂0end` if model is a [`LinModel`](@ref).

The method mutates `Ŷ0` and `x̂0end` vector arguments. The `x̂end` vector is used for
the terminal constraints applied on ``\mathbf{x̂_0}(k+H_p)``. The computations are
identical for any [`TranscriptionMethod`](@ref) if the model is linear:
```math
\begin{aligned}
\mathbf{Ŷ_0}        &= \mathbf{Ẽ Z̃}   + \mathbf{F} \\
\mathbf{x̂_0}(k+H_p) &= \mathbf{ẽ_x̂ Z̃} + \mathbf{f_x̂}
\end{aligned}
```
"""
function predict!(
    Ŷ0, x̂0end, _, _, _,
    mpc::PredictiveController, ::LinModel, ::TranscriptionMethod, 
    _ , Z̃
)
    # in-place operations to reduce allocations :
    Ŷ0    .= mul!(Ŷ0, mpc.Ẽ, Z̃) .+ mpc.F
    x̂0end .= mul!(x̂0end, mpc.con.ẽx̂, Z̃) .+ mpc.con.fx̂
    return Ŷ0, x̂0end
end

@doc raw"""
    predict!(
        Ŷ0, x̂0end, X̂0, Û0, K0,
        mpc::PredictiveController, model::NonLinModel, transcription::SingleShooting,
        U0, _
    ) -> Ŷ0, x̂0end

Compute vectors if `model` is a [`NonLinModel`](@ref) and for [`SingleShooting`](@ref).
    
The method mutates `Ŷ0`, `x̂0end`, `X̂0`, `Û0` and `K0` arguments. The augmented model of
[`f̂!`](@ref) and [`ĥ!`](@ref) functions is called recursively in a `for` loop:
```math
\begin{aligned}
\mathbf{x̂_0}(k+1) &= \mathbf{f̂}\Big(\mathbf{x̂_0}(k), \mathbf{u_0}(k), \mathbf{d̂_0}(k) \Big) \\
\mathbf{ŷ_0}(k)   &= \mathbf{ĥ}\Big(\mathbf{x̂_0}(k), \mathbf{d̂_0}(k) \Big)
\end{aligned}
```
"""
function predict!(
    Ŷ0, x̂0end, X̂0, Û0, K0,
    mpc::PredictiveController, model::NonLinModel, ::SingleShooting,
    U0, _
)
    nu, nx̂, ny, nd, nk, Hp = model.nu, mpc.estim.nx̂, model.ny, model.nd, model.nk, mpc.Hp
    D̂0 = mpc.D̂0
    x̂0 = @views mpc.estim.x̂0[1:nx̂]
    d̂0 = @views mpc.d0[1:nd]
    for j=1:Hp
        u0     = @views U0[(1 + nu*(j-1)):(nu*j)]
        û0     = @views Û0[(1 + nu*(j-1)):(nu*j)]
        k0     = @views K0[(1 + nk*(j-1)):(nk*j)]
        x̂0next = @views X̂0[(1 + nx̂*(j-1)):(nx̂*j)]
        f̂!(x̂0next, û0, k0, mpc.estim, model, x̂0, u0, d̂0)
        x̂0 = @views X̂0[(1 + nx̂*(j-1)):(nx̂*j)]
        d̂0 = @views D̂0[(1 + nd*(j-1)):(nd*j)]
        ŷ0 = @views Ŷ0[(1 + ny*(j-1)):(ny*j)]
        ĥ!(ŷ0, mpc.estim, model, x̂0, d̂0)
    end
    Ŷ0    .+= mpc.F # F = Ŷs if mpc.estim is an InternalModel, else F = 0.
    x̂0end  .= x̂0
    return Ŷ0, x̂0end
end

@doc raw"""
    predict!(
        Ŷ0, x̂0end, _ , _ , _ , 
        mpc::PredictiveController, model::NonLinModel, transcription::TranscriptionMethod,
        _ , Z̃
    ) -> Ŷ0, x̂0end

Compute vectors if `model` is a [`NonLinModel`](@ref) and other [`TranscriptionMethod`](@ref).
    
The method mutates `Ŷ0` and `x̂0end` arguments. The augmented output function [`ĥ!`](@ref) 
is called multiple times in a `for` loop:
```math
\mathbf{ŷ_0}(k) = \mathbf{ĥ}\Big(\mathbf{x̂_0}(k), \mathbf{d̂_0}(k) \Big)
```
in which ``\mathbf{x̂_0}`` is the augmented state extracted from the decision variable `Z̃`.
"""
function predict!(
    Ŷ0, x̂0end, _, _, _,
    mpc::PredictiveController, model::NonLinModel, transcription::TranscriptionMethod,
    _ , Z̃
)
    nu, ny, nd, nx̂, Hp, Hc = model.nu, model.ny, model.nd, mpc.estim.nx̂, mpc.Hp, mpc.Hc
    h_threads = transcription.h_threads
    X̂0 = @views Z̃[(nu*Hc+1):(nu*Hc+nx̂*Hp)] # Z̃ = [ΔU; X̂0; ϵ]
    D̂0 = mpc.D̂0
    @threadsif h_threads for j=1:Hp
        x̂0 = @views X̂0[(1 +  nx̂*(j-1)):(nx̂*j)]
        d̂0 = @views D̂0[(1 +  nd*(j-1)):(nd*j)]
        ŷ0 = @views Ŷ0[(1 +  ny*(j-1)):(ny*j)]
        ĥ!(ŷ0, mpc.estim, model, x̂0, d̂0)
    end
    Ŷ0    .+= mpc.F # F = Ŷs if mpc.estim is an InternalModel, else F = 0.
    x̂0end  .= @views X̂0[(1+nx̂*(Hp-1)):(nx̂*Hp)]
    return Ŷ0, x̂0end
end

"""
    con_nonlinprog!(
        g, mpc::PredictiveController, model::LinModel, ::TranscriptionMethod, _ , _ , gc, ϵ
    ) -> g

Nonlinear constrains when `model` is a [`LinModel`](@ref).

The method mutates the `g` vectors in argument and returns it. Only the custom constraints
`gc` are include in the `g` vector.
"""
function con_nonlinprog!(
    g, ::PredictiveController, ::LinModel, ::TranscriptionMethod, _ , _ , gc, ϵ
)
    for i in eachindex(g)
        g[i] = gc[i]
    end
    return g
end

"""
    con_nonlinprog!(
        g, mpc::PredictiveController, model::NonLinModel, ::TranscriptionMethod, x̂0end, Ŷ0, gc, ϵ
    ) -> g

Nonlinear constrains when `model` is a [`NonLinModel`](@ref) with non-[`SingleShooting`](@ref).

The method mutates the `g` vectors in argument and returns it. The output prediction and the
custom constraints are include in the `g` vector.
"""
function con_nonlinprog!(
    g, mpc::PredictiveController, ::NonLinModel, ::TranscriptionMethod, x̂0end, Ŷ0, gc, ϵ
)
    nŶ = length(Ŷ0)
    for i in eachindex(g)
        mpc.con.i_g[i] || continue
        if i ≤ nŶ
            j = i
            g[i] = (mpc.con.Y0min[j] - Ŷ0[j])     - ϵ*mpc.con.C_ymin[j]
        elseif i ≤ 2nŶ
            j = i - nŶ
            g[i] = (Ŷ0[j] - mpc.con.Y0max[j])     - ϵ*mpc.con.C_ymax[j]
        else
            j = i - 2nŶ
            g[i] = gc[j]
        end
    end
    return g
end

"""
    con_nonlinprog!(
        g, mpc::PredictiveController, model::NonLinModel, ::SingleShooting, x̂0end, Ŷ0, gc, ϵ
    ) -> g

Nonlinear constrains when `model` is [`NonLinModel`](@ref) with [`SingleShooting`](@ref).

The method mutates the `g` vectors in argument and returns it. The output prediction, 
the terminal state and the custom constraints are include in the `g` vector.
"""
function con_nonlinprog!(
    g, mpc::PredictiveController, ::NonLinModel, ::SingleShooting, x̂0end, Ŷ0, gc, ϵ
)
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
    con_nonlinprogeq!(
        geq, X̂0, Û0, K0
        mpc::PredictiveController, model::NonLinModel, transcription::MultipleShooting, 
        U0, Z̃
    )

Nonlinear equality constrains for [`NonLinModel`](@ref) and [`MultipleShooting`](@ref).

The method mutates the `geq`, `X̂0`, `Û0` and `K0` vectors in argument. The nonlinear 
equality constraints `geq` only includes the augmented state defects, computed with:
```math
\mathbf{ŝ}(k+1) = \mathbf{f̂}\Big(\mathbf{x̂_0}(k), \mathbf{u_0}(k), \mathbf{d̂_0}(k)\Big) 
                    - \mathbf{x̂_0}(k+1)
```
in which the augmented state ``\mathbf{x̂_0}`` are extracted from the decision variables 
`Z̃`, and ``\mathbf{f̂}`` is the augmented state function defined in [`f̂!`](@ref).
"""
function con_nonlinprogeq!(
    geq, X̂0, Û0, K0, 
    mpc::PredictiveController, model::NonLinModel, transcription::MultipleShooting, U0, Z̃
)
    nu, nx̂, nd, nk = model.nu, mpc.estim.nx̂, model.nd, model.nk
    Hp, Hc = mpc.Hp, mpc.Hc
    nΔU, nX̂ = nu*Hc, nx̂*Hp
    f_threads = transcription.f_threads
    D̂0 = mpc.D̂0
    X̂0_Z̃ = @views Z̃[(nΔU+1):(nΔU+nX̂)] 
    @threadsif f_threads for j=1:Hp
        if j < 2
            x̂0 = @views mpc.estim.x̂0[1:nx̂]
            d̂0 = @views mpc.d0[1:nd]
        else
            x̂0 = @views X̂0_Z̃[(1 + nx̂*(j-2)):(nx̂*(j-1))]
            d̂0 = @views   D̂0[(1 + nd*(j-2)):(nd*(j-1))]
        end
        u0       = @views   U0[(1 + nu*(j-1)):(nu*j)]
        û0       = @views   Û0[(1 + nu*(j-1)):(nu*j)]
        k0       = @views   K0[(1 + nk*(j-1)):(nk*j)]
        x̂0next   = @views   X̂0[(1 + nx̂*(j-1)):(nx̂*j)]
        x̂0next_Z̃ = @views X̂0_Z̃[(1 + nx̂*(j-1)):(nx̂*j)]
        ŝnext    = @views  geq[(1 + nx̂*(j-1)):(nx̂*j)]
        f̂!(x̂0next, û0, k0, mpc.estim, model, x̂0, u0, d̂0)
        ŝnext .= x̂0next .- x̂0next_Z̃
    end
    return geq
end

@doc raw"""
    con_nonlinprogeq!(
        geq, X̂0, Û0, K0
        mpc::PredictiveController, model::NonLinModel, transcription::TrapezoidalCollocation, 
        U0, Z̃
    )

Nonlinear equality constrains for [`NonLinModel`](@ref) and [`TrapezoidalCollocation`](@ref).

The method mutates the `geq`, `X̂0`, `Û0` and `K0` vectors in argument. 

The nonlinear equality constraints `geq` only includes the state defects. The deterministic
and stochastic states are handled separately since collocation methods require continuous-
time state-space models, and the stochastic model of the unmeasured disturbances
is discrete-time. The deterministic and stochastic defects are respectively computed with:
```math
\begin{aligned}
\mathbf{s_d}(k+1) &= \mathbf{x_0}(k) - \mathbf{x_0}(k+1) 
                      + 0.5 T_s (\mathbf{k}_1 + \mathbf{k}_2) \\
\mathbf{s_s}(k+1) &= \mathbf{A_s x_s}(k) - \mathbf{x_s}(k+1)
\end{aligned}
```
in which ``\mathbf{x_0}`` and ``\mathbf{x_s}`` are the deterministic and stochastic states 
extracted from the decision variables `Z̃`. The ``\mathbf{k}`` coefficients are 
evaluated from the continuous-time function `model.f!` and:
```math
\begin{aligned}
\mathbf{k}_1 &= \mathbf{f}\Big(\mathbf{x_0}(k),   \mathbf{û_0}(k),   \mathbf{d̂_0}(k)  \Big) \\
\mathbf{k}_2 &= \mathbf{f}\Big(\mathbf{x_0}(k+1), \mathbf{û_0}(k+h), \mathbf{d̂_0}(k+1)\Big) 
\end{aligned}
```
in which ``h`` is the hold order `transcription.h` and the disturbed input is:
```math
\mathbf{û_0}(k) = \mathbf{u_0}(k) + \mathbf{C_{s_u} x_s}(k)
``` 
the ``\mathbf{A_s, C_{s_u}}`` matrices are defined in [`init_estimstoch`](@ref) doc.
"""
function con_nonlinprogeq!(
    geq, X̂0, Û0, K0, 
    mpc::PredictiveController, model::NonLinModel, transcription::TrapezoidalCollocation, 
    U0, Z̃
)
    nu, nx̂, nd, nx, h = model.nu, mpc.estim.nx̂, model.nd, model.nx, transcription.h
    Hp, Hc = mpc.Hp, mpc.Hc
    nΔU, nX̂ = nu*Hc, nx̂*Hp
    f_threads = transcription.f_threads
    Ts, p = model.Ts, model.p
    As, Cs_u = mpc.estim.As, mpc.estim.Cs_u
    nk = get_nk(model, transcription)
    D̂0 = mpc.D̂0
    X̂0_Z̃ = @views Z̃[(nΔU+1):(nΔU+nX̂)]
    @threadsif f_threads for j=1:Hp
        if j < 2
            x̂0 = @views mpc.estim.x̂0[1:nx̂]
            d̂0 = @views mpc.d0[1:nd]
        else
            x̂0 = @views X̂0_Z̃[(1 + nx̂*(j-2)):(nx̂*(j-1))] 
            d̂0 = @views   D̂0[(1 + nd*(j-2)):(nd*(j-1))]
        end
        k0       = @views   K0[(1 + nk*(j-1)):(nk*j)]
        d̂0next   = @views   D̂0[(1 + nd*(j-1)):(nd*j)]
        x̂0next   = @views   X̂0[(1 + nx̂*(j-1)):(nx̂*j)]
        x̂0next_Z̃ = @views X̂0_Z̃[(1 + nx̂*(j-1)):(nx̂*j)]  
        ŝnext    = @views  geq[(1 + nx̂*(j-1)):(nx̂*j)]  
        x0, xs              = @views x̂0[1:nx], x̂0[nx+1:end]
        x0next_Z̃, xsnext_Z̃  = @views x̂0next_Z̃[1:nx], x̂0next_Z̃[nx+1:end]
        sdnext, ssnext      = @views ŝnext[1:nx], ŝnext[nx+1:end]
        k1, k2              = @views k0[1:nx], k0[nx+1:2*nx]
        # ----------------- stochastic defects -----------------------------------------
        xsnext = @views x̂0next[nx+1:end]
        mul!(xsnext, As, xs)
        ssnext .= @. xsnext - xsnext_Z̃
        # ----------------- deterministic defects --------------------------------------
        u0 = @views U0[(1 + nu*(j-1)):(nu*j)]
        û0 = @views Û0[(1 + nu*(j-1)):(nu*j)]
        mul!(û0, Cs_u, xs)                 # ys_u(k) = Cs_u*xs(k)
        û0 .+= u0                          #   û0(k) = u0(k) + ys_u(k)
        if f_threads || h < 1 || j < 2
            # we need to recompute k1 with multi-threading, even with h==1, since the 
            # last iteration (j-1) may not be executed (iterations are re-orderable)
            model.f!(k1, x0, û0, d̂0, p)
        else
            k1 .= @views K0[(1 + nk*(j-1)-nx):(nk*(j-1))] # k2 of of the last iter. j-1
        end
        if h < 1 || j ≥ Hp
            # j = Hp special case: u(k+Hp-1) = u(k+Hp) since Hc ≤ Hp implies Δu(k+Hp) = 0
            û0next = û0
        else
            u0next = @views U0[(1 + nu*j):(nu*(j+1))]
            û0next = @views Û0[(1 + nu*j):(nu*(j+1))]
            mul!(û0next, Cs_u, xsnext_Z̃)      # ys_u(k+1) = Cs_u*xs(k+1)
            û0next .+= u0next                 #   û0(k+1) = u0(k+1) + ys_u(k+1)
        end
        model.f!(k2, x0next_Z̃, û0next, d̂0next, p)
        sdnext .= @. x0 - x0next_Z̃ + 0.5*Ts*(k1 + k2)
    end
    return geq
end

"No eq. constraints for other cases e.g. [`SingleShooting`](@ref), returns `geq` unchanged."
con_nonlinprogeq!(geq,_,_,_,::PredictiveController,::SimModel,::TranscriptionMethod,_,_)=geq
