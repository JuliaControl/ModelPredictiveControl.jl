const COLLOCATION_NODE_TYPE = Float64

"""
Abstract supertype of all transcription methods of [`PredictiveController`](@ref).

The module currently supports [`SingleShooting`](@ref), [`MultipleShooting`](@ref),
[`TrapezoidalCollocation`](@ref) and [`OrthogonalCollocation`](@ref) transcription methods.
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
here assume zero-order hold, so `h=1` will induce a plant-model mismatch if the plant is
simulated with these solvers. Measured disturbances ``\mathbf{d}`` are piecewise linear.

This transcription computes the predictions by calling the continuous-time model in the
equality constraint function and by using the implicit trapezoidal rule. It can handle
moderately stiff systems and is A-stable. See Extended Help for more details.

!!! warning
    The built-in [`StateEstimator`](@ref) will still use the `solver` provided at the
    construction of the [`NonLinModel`](@ref) to estimate the plant states, not the 
    trapezoidal rule (see `supersample` option of  [`RungeKutta`](@ref) for stiff systems).

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
    OrthogonalCollocation(
        h::Int=0, no::Int=3; f_threads=false, h_threads=false, roots=:gaussradau
    )

Construct an orthogonal collocation on finite elements [`TranscriptionMethod`](@ref).

Also known as pseudo-spectral method. It supports continuous-time [`NonLinModel`](@ref)s
only. The `h` argument is the hold order for ``\mathbf{u}`` (`0` or `1`), and the `no`
argument, the number of collocation points ``n_o``. The decision variable is similar to
[`MultipleShooting`](@ref), but it also includes the collocation points:
```math
\mathbf{Z} = \begin{bmatrix} \mathbf{ΔU} \\ \mathbf{X̂_0} \\ \mathbf{K} \end{bmatrix}
```
where ``\mathbf{K}`` encompasses all the intermediate stages of the deterministic states
(the first `nx` elements of ``\mathbf{x̂}``):
```math
\mathbf{K} =                            \begin{bmatrix}
    \mathbf{k}_{1}(k+0)                 \\
    \mathbf{k}_{2}(k+0)                 \\
    \vdots                              \\
    \mathbf{k}_{n_o}(k+0)               \\
    \mathbf{k}_{1}(k+1)                 \\
    \mathbf{k}_{2}(k+1)                 \\
    \vdots                              \\
    \mathbf{k}_{n_o}(k+H_p-1)           \end{bmatrix}
```
and ``\mathbf{k}_i(k+j)`` is the deterministic state prediction for the ``i``th collocation
point at the ``j``th stage/interval/finite element (details in Extended Help). The `roots`
keyword argument is either `:gaussradau` or `:gausslegendre`, for Gauss-Radau or 
Gauss-Legendre quadrature, respectively. See [`MultipleShooting`](@ref) docstring for
descriptions of `f_threads` and `h_threads` keywords. This transcription computes the
predictions by enforcing the collocation and continuity constraints at the collocation
points. It is efficient for highly stiff systems, but generally more expensive than the
other methods for non-stiff systems. See Extended Help for more details.

!!! warning
    The built-in [`StateEstimator`](@ref) will still use the `solver` provided at the
    construction of the [`NonLinModel`](@ref) to estimate the plant states, not orthogonal
    collocation (see `supersample` option of  [`RungeKutta`](@ref) for stiff systems).

Sparse optimizers like `Ipopt` and sparse Jacobian computations are highly recommended for
this transcription method (sparser formulation than [`MultipleShooting`](@ref)).

# Extended Help
!!! details "Extended Help"
    As explained in the Extended Help of [`TrapezoidalCollocation`](@ref), the stochastic
    states are left out of the ``\mathbf{K}`` vector since collocation methods require
    continuous-time dynamics and the stochastic model is discrete.

    The collocation points are located at the roots of orthogonal polynomials, which is 
    "optimal" for approximating the state trajectories with polynomials of degree ``n_o``.
    The method then enforces the system dynamics at these points. The Gauss-Legendre scheme
    is more accurate than Gauss-Radau but only A-stable, while the latter being L-stable. 
    See [`con_nonlinprogeq!`](@ref) for implementation details.
"""
struct OrthogonalCollocation <: CollocationMethod
    h::Int
    no::Int
    f_threads::Bool
    h_threads::Bool
    τ::Vector{COLLOCATION_NODE_TYPE}
    function OrthogonalCollocation(
        h::Int=0, no::Int=3; f_threads=false, h_threads=false, roots=:gaussradau
    )
        if !(h == 0 || h == 1)
            throw(ArgumentError("h argument must be 0 or 1 for OrthogonalCollocation."))
        end
        if roots==:gaussradau            
            x, _ = FastGaussQuadrature.gaussradau(no, COLLOCATION_NODE_TYPE)
            # we reverse the nodes to include the τ=1.0 node:
            τ = (reverse(-x) .+ 1) ./ 2
        elseif roots==:gausslegendre
            x, _ = FastGaussQuadrature.gausslegendre(no)
            # converting [-1, 1] to [0, 1] (see 
            # https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval):
            τ = (x .+ 1) ./ 2
        else
            throw(ArgumentError("roots argument must be :gaussradau or :gausslegendre."))
        end
        return new(h, no, f_threads, h_threads, τ)
    end
end

@doc raw"""
    init_orthocolloc(model::SimModel, transcription::OrthogonalCollocation) -> Mo, Co, λo

Init the differentiation and continuity matrices for [`OrthogonalCollocation`](@ref).

Introducing ``τ_i``, the ``i``th root of the orthogonal polynomial normalized to the
interval ``[0, 1]``, and ``τ_0=0``, each state trajectories are approximated by a distinct
polynomial of degree ``n_o``. The differentiation matrix ``\mathbf{M_o}``, continuity
matrix ``\mathbf{C_o}`` and continuity coefficient ``λ_o`` are pre-computed with:
```math
\begin{aligned}
    \mathbf{P_o} &=                                                                               \begin{bmatrix}
        τ_1^1 \mathbf{I}       & τ_1^2 \mathbf{I}       & \cdots & τ_1^{n_o} \mathbf{I}           \\
        τ_2^1 \mathbf{I}       & τ_2^2 \mathbf{I}       & \cdots & τ_2^{n_o} \mathbf{I}           \\
        \vdots                 & \vdots                 & \ddots & \vdots                         \\
        τ_{n_o}^1 \mathbf{I}   & τ_{n_o}^2 \mathbf{I}   & \cdots & τ_{n_o}^{n_o} \mathbf{I}       \end{bmatrix} \\
    \mathbf{Ṗ_o} &=                                                                               \begin{bmatrix}
        τ_1^0 \mathbf{I}       & 2τ_1^1 \mathbf{I}      & \cdots & n_o τ_1^{n_o-1} \mathbf{I}     \\
        τ_2^0 \mathbf{I}       & 2τ_2^1 \mathbf{I}      & \cdots & n_o τ_2^{n_o-1} \mathbf{I}     \\
        \vdots                 & \vdots                 & \ddots & \vdots                         \\
        τ_{n_o}^0 \mathbf{I} & 2τ_{n_o}^1 \mathbf{I} & \cdots & n_o τ_{n_o}^{n_o-1} \mathbf{I}    \end{bmatrix} \\
    \mathbf{M_o} &= \frac{1}{T_s} \mathbf{Ṗ_o} \mathbf{P_o}^{-1}                                  \\
    \mathbf{C_o} &=                                                                               \begin{bmatrix}
        L_1(1) \mathbf{I}      & L_2(1) \mathbf{I}      & \cdots & L_{n_o}(1) \mathbf{I}          \end{bmatrix} \\
            λ_o  &= L_0(1)                                                                        
\end{aligned}
```
where ``\mathbf{P_o}`` is a matrix to evaluate the polynamial values w/o the coefficients
and Y-intercept, and ``\mathbf{Ṗ_o}``, to evaluate its derivatives. The Lagrange polynomial
``L_j(τ)`` bases are defined as:
```math
L_j(τ) = \prod_{i=0, i≠j}^{n_o} \frac{τ - τ_i}{τ_j - τ_i}
```
"""
function init_orthocolloc(
    model::SimModel{NT}, transcription::OrthogonalCollocation
) where {NT<:Real}
    nx, no = model.nx, transcription.no
    τ = transcription.τ
    Po = Matrix{NT}(undef, nx*no, nx*no) # polynomial matrix (w/o the Y-intercept term)
    Ṗo = Matrix{NT}(undef, nx*no, nx*no) # polynomial derivative matrix
    for j=1:no, i=1:no
        iRows = (1:nx) .+ nx*(i-1)
        iCols = (1:nx) .+ nx*(j-1)
        Po[iRows, iCols] = (τ[i]^j)*I(nx)
        Ṗo[iRows, iCols] = (j*τ[i]^(j-1))*I(nx)
    end
    Mo = sparse((Ṗo/Po)/model.Ts)
    Co = Matrix{NT}(undef, nx, nx*no)
    for j=1:no
        iCols = (1:nx) .+ nx*(j-1)
        Co[:, iCols] = lagrange_end(j, transcription)*I(nx)
    end
    Co = sparse(Co)
    λo = lagrange_end(0, transcription)
    return Mo, Co, λo
end
"Return empty sparse matrices and `NaN` for other [`TranscriptionMethod`](@ref)"
init_orthocolloc(::SimModel, ::TranscriptionMethod) = spzeros(0,0), spzeros(0,0), NaN

"Evaluate the Lagrange basis polynomial ``L_j`` at `τ=1`."
function lagrange_end(j, transcription::OrthogonalCollocation)
    τ_val = 1
    τ_values = [0; transcription.τ] # including the τ=0 node for the Lagrange polynomials
    j_index = j + 1 # because of the τ=0 node
    τj = τ_values[j_index]
    Lj = 1
    for i in eachindex(τ_values)
        i == j_index && continue
        τi = τ_values[i]
        Lj *= (τ_val - τi)/(τj - τi)
    end
    return Lj
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