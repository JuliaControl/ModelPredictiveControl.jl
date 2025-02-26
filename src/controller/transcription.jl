"""
Abstract supertype of all transcription methods of [`PredictiveController`](@ref).

The module currently supports [`SingleShooting`](@ref) and [`MultipleShooting`](@ref).
"""
abstract type TranscriptionMethod end

@doc raw"""
    SingleShooting()

Construct a direct single shooting [`TranscriptionMethod`](@ref).

The decision variable in the optimization problem is (excluding the slack ``ϵ``):
```math
\mathbf{Z} = \mathbf{ΔU} =          \begin{bmatrix} 
    \mathbf{Δu}(k+0)                \\ 
    \mathbf{Δu}(k+1)                \\ 
    \vdots                          \\ 
    \mathbf{Δu}(k+H_c-1)            \end{bmatrix}
```
This method is generally more efficient for small control horizon ``H_c``, stable or mildly
nonlinear plant model/constraints.
"""
struct SingleShooting <: TranscriptionMethod end

@doc raw"""
    MultipleShooting()

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
observer at time ``i=k`` or ``i=k-1`` depending on its `direct` flag. This transcription
method is generally more efficient for large control horizon ``H_c``, unstable or highly
nonlinear plant models/constraints. Sparse optimizers like `OSQP` or `Ipopt` are recommended
for large-scale problems.
"""
struct MultipleShooting <: TranscriptionMethod end

"Get the number of elements in the optimization decision vector `Z`."
function get_nZ(estim::StateEstimator, transcription::SingleShooting, Hp, Hc)
    return estim.model.nu*Hc
end
function get_nZ(estim::StateEstimator, transcription::MultipleShooting, Hp, Hc)
    return estim.model.nu*Hc + estim.nx̂*Hp
end

@doc raw"""
    init_ZtoΔU(estim::StateEstimator, transcription::TranscriptionMethod, Hp, Hc) -> P

Init decision variables to input increments over ``H_c`` conversion matrix `P`.

The conversion from the decision variables ``\mathbf{Z}`` to ``\mathbf{ΔU}``, the input
increments over ``H_c``, is computed by:
```math
\mathbf{ΔU} = \mathbf{P} \mathbf{Z}
```
in which ``\mathbf{P}`` is defined in the Extended Help section.

# Extended Help
!!! details "Extended Help"
    Following the decision variable definition of the [`TranscriptionMethod`](@ref), the
    conversion matrix ``\mathbf{P}``, we have:
    - ``\mathbf{P} = \mathbf{I}`` if `transcription` is a [`SingleShooting`](@ref)
    - ``\mathbf{P} = [\begin{smallmatrix}\mathbf{I} & \mathbf{0} \end{smallmatrix}]`` if 
      `transcription` is a [`MultipleShooting`](@ref)
"""
function init_ZtoΔU end

function init_ZtoΔU(
    estim::StateEstimator{NT}, transcription::SingleShooting, _ , Hc
) where {NT<:Real}
    P = Matrix{NT}(I, estim.model.nu*Hc, estim.model.nu*Hc)
    return P
end

function init_ZtoΔU(
    estim::StateEstimator{NT}, transcription::MultipleShooting, Hp, Hc
) where {NT<:Real}
    I_nu_Hc = Matrix{NT}(I, estim.model.nu*Hc, estim.model.nu*Hc)
    P = [I_nu_Hc zeros(NT, estim.model.nu*Hc, estim.nx̂*Hp)]
    return P
end

@doc raw"""
    init_ZtoU(estim, transcription, Hp, Hc) -> S, T

Init decision variables to inputs over ``H_p`` conversion matrices.

The conversion from the decision variables ``\mathbf{Z}`` to ``\mathbf{U}``, the manipulated
inputs over ``H_p``, is computed by:
```math
\mathbf{U} = \mathbf{S} \mathbf{Z} + \mathbf{T} \mathbf{u}(k-1)
```
The ``\mathbf{S}`` and ``\mathbf{T}`` matrices are defined in the Extended Help section.

# Extended Help
!!! details "Extended Help"
    The ``\mathbf{U}`` vector and the conversion matrices are defined as:
    ```math
    \mathbf{U} = \begin{bmatrix}
        \mathbf{u}(k + 0)                                           \\
        \mathbf{u}(k + 1)                                           \\
        \vdots                                                      \\
        \mathbf{u}(k + H_c - 1)                                     \\
        \vdots                                                      \\
        \mathbf{u}(k + H_p - 1)                                     \end{bmatrix} , \quad
    \mathbf{S^†} = \begin{bmatrix}
        \mathbf{I}  & \mathbf{0}    & \cdots    & \mathbf{0}        \\
        \mathbf{I}  & \mathbf{I}    & \cdots    & \mathbf{0}        \\
        \vdots      & \vdots        & \ddots    & \vdots            \\
        \mathbf{I}  & \mathbf{I}    & \cdots    & \mathbf{I}        \\
        \vdots      & \vdots        & \ddots    & \vdots            \\
        \mathbf{I}  & \mathbf{I}    & \cdots    & \mathbf{I}        \end{bmatrix} , \quad
    \mathbf{T} = \begin{bmatrix}
        \mathbf{I}                                                  \\
        \mathbf{I}                                                  \\
        \vdots                                                      \\
        \mathbf{I}                                                  \\
        \vdots                                                      \\
        \mathbf{I}                                                  \end{bmatrix}
    ```
    and, depending on the transcription method, we have:
    - ``\mathbf{S} = \mathbf{S^†}`` if `transcription` is a [`SingleShooting`](@ref)
    - ``\mathbf{S} = [\begin{smallmatrix}\mathbf{S^†} & \mathbf{0} \end{smallmatrix}]`` if 
      `transcription` is a [`MultipleShooting`](@ref)
"""
function init_ZtoU(
    estim::StateEstimator{NT}, transcription::TranscriptionMethod, Hp, Hc
) where {NT<:Real}
    model = estim.model
    # S and T are `Matrix{NT}` since conversion is faster than `Matrix{Bool}` or `BitMatrix`
    I_nu = Matrix{NT}(I, model.nu, model.nu)
    S_Hc = LowerTriangular(repeat(I_nu, Hc, Hc))
    Sdagger = [S_Hc; repeat(I_nu, Hp - Hc, Hc)]
    S = init_Smat(estim, transcription, Hp, Hc, Sdagger)
    T = repeat(I_nu, Hp)
    return S, T
end

init_Smat( _ , transcription::SingleShooting, _ , _ , Sdagger) = Sdagger

function init_Smat(estim, transcription::MultipleShooting, Hp, _ , Sdagger)
    return [Sdagger zeros(eltype(Sdagger), estim.model.nu*Hp, estim.nx̂*Hp)]
end

@doc raw"""
    init_predmat(
        model::LinModel, estim, transcription::SingleShooting, Hp, Hc
    ) -> E, G, J, K, V, ex̂, gx̂, jx̂, kx̂, vx̂

Construct the prediction matrices for [`LinModel`](@ref) and [`SingleShooting`](@ref).

The model predictions are evaluated from the deviation vectors (see [`setop!`](@ref)), the
decision variable ``\mathbf{Z}`` (see [`TranscriptionMethod`](@ref)), and:
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
``\mathbf{Δu}(k+j)`` from ``j=0`` to ``H_c-1``. The vector ``\mathbf{B}`` contains the
contribution for non-zero state ``\mathbf{x̂_{op}}`` and state update ``\mathbf{f̂_{op}}``
operating points (for linearization at non-equilibrium point, see [`linearize`](@ref)). The
stochastic predictions ``\mathbf{Ŷ_s=0}`` if `estim` is not a [`InternalModel`](@ref), see
[`init_stochpred`](@ref). The method also computes similar matrices for the predicted
terminal states at ``k+H_p``:
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
    [`augment_model`](@ref)), and the function ``\mathbf{W}(j) = ∑_{i=0}^j \mathbf{Â}^i``,
    the prediction matrices are computed by :
    ```math
    \begin{aligned}
    \mathbf{E} &= \begin{bmatrix}
        \mathbf{Ĉ W}(0)\mathbf{B̂_u}     & \mathbf{0}                      & \cdots & \mathbf{0}                                        \\
        \mathbf{Ĉ W}(1)\mathbf{B̂_u}     & \mathbf{Ĉ W}(0)\mathbf{B̂_u}     & \cdots & \mathbf{0}                                        \\
        \vdots                          & \vdots                          & \ddots & \vdots                                            \\
        \mathbf{Ĉ W}(H_p-1)\mathbf{B̂_u} & \mathbf{Ĉ W}(H_p-2)\mathbf{B̂_u} & \cdots & \mathbf{Ĉ W}(H_p-H_c+1)\mathbf{B̂_u} \end{bmatrix} \\
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
    \mathbf{V} &= \begin{bmatrix}
        \mathbf{Ĉ W}(0)\mathbf{B̂_u}     \\
        \mathbf{Ĉ W}(1)\mathbf{B̂_u}     \\
        \vdots                          \\
        \mathbf{Ĉ W}(H_p-1)\mathbf{B̂_u} \end{bmatrix} \\
    \mathbf{B} &= \begin{bmatrix}
        \mathbf{Ĉ W}(0)                 \\
        \mathbf{Ĉ W}(1)                 \\
        \vdots                          \\
        \mathbf{Ĉ W}(H_p-1)             \end{bmatrix}   \mathbf{\big(f̂_{op} - x̂_{op}\big)} 
    \end{aligned}
    ```
    For the terminal constraints, the matrices are computed with:
    ```math
    \begin{aligned}
    \mathbf{e_x̂} &= \begin{bmatrix} 
                        \mathbf{W}(H_p-1)\mathbf{B̂_u} & 
                        \mathbf{W}(H_p-2)\mathbf{B̂_u} & 
                        \cdots & 
                        \mathbf{W}(H_p-H_c+1)\mathbf{B̂_u} \end{bmatrix} \\
    \mathbf{g_x̂} &= \mathbf{Â}^{H_p-1} \mathbf{B̂_d} \\
    \mathbf{j_x̂} &= \begin{bmatrix} 
                        \mathbf{Â}^{H_p-2} \mathbf{B̂_d} & 
                        \mathbf{Â}^{H_p-3} \mathbf{B̂_d} & 
                        \cdots & 
                        \mathbf{0} 
                    \end{bmatrix} \\
    \mathbf{k_x̂} &= \mathbf{Â}^{H_p} \\
    \mathbf{v_x̂} &= \mathbf{W}(H_p-1)\mathbf{B̂_u} \\
    \mathbf{b_x̂} &= \mathbf{W}(H_p-1)    \mathbf{\big(f̂_{op} - x̂_{op}\big)}
    \end{aligned}
    ```
"""
function init_predmat(
    model::LinModel, estim::StateEstimator{NT}, transcription::SingleShooting, Hp, Hc
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
    # helper function to improve code clarity and be similar to eqs. in docstring:
    getpower(array3D, power) = @views array3D[:,:, power+1]
    # --- current state estimates x̂0 ---
    kx̂ = getpower(Âpow, Hp)
    K  = Matrix{NT}(undef, Hp*ny, nx̂)
    for j=1:Hp
        iRow = (1:ny) .+ ny*(j-1)
        K[iRow,:] = Ĉ*getpower(Âpow, j)
    end    
    # --- previous manipulated inputs lastu0 ---
    vx̂ = getpower(Âpow_csum, Hp-1)*B̂u
    V  = Matrix{NT}(undef, Hp*ny, nu)
    for j=1:Hp
        iRow = (1:ny) .+ ny*(j-1)
        V[iRow,:] = Ĉ*getpower(Âpow_csum, j-1)*B̂u
    end
    # --- decision variables Z ---
    nZ = get_nZ(estim, transcription, Hp, Hc)
    ex̂ = Matrix{NT}(undef, nx̂, nZ)
    E  = zeros(NT, Hp*ny, nZ) 
    for j=1:Hc # truncated with control horizon
        iRow = (ny*(j-1)+1):(ny*Hp)
        iCol = (1:nu) .+ nu*(j-1)
        E[iRow, iCol] = V[iRow .- ny*(j-1),:]
        ex̂[:  , iCol] = getpower(Âpow_csum, Hp-j)*B̂u
    end
    # --- current measured disturbances d0 and predictions D̂0 ---
    gx̂ = getpower(Âpow, Hp-1)*B̂d
    G  = Matrix{NT}(undef, Hp*ny, nd)
    jx̂ = Matrix{NT}(undef, nx̂, Hp*nd)
    J  = repeatdiag(D̂d, Hp)
    if nd ≠ 0
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
    coef_bx̂ = getpower(Âpow_csum, Hp-1)
    coef_B  = Matrix{NT}(undef, ny*Hp, nx̂)
    for j=1:Hp
        iRow = (1:ny) .+ ny*(j-1)
        coef_B[iRow,:] = Ĉ*getpower(Âpow_csum, j-1)
    end
    f̂op_n_x̂op = estim.f̂op - estim.x̂op
    bx̂ = coef_bx̂ * f̂op_n_x̂op
    B  = coef_B  * f̂op_n_x̂op
    return E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂
end

@doc raw"""
    init_predmat(
        model::LinModel, estim, transcription::MultipleShooting, Hp, Hc
    ) -> E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂
    
Construct the prediction matrices for [`LinModel`](@ref) and [`MultipleShooting`](@ref).

They are defined in the Extended Help section.

# Extended Help
!!! details "Extended Help"
    They are all appropriately sized zero matrices ``\mathbf{0}``, except for:
    ```math
    \begin{aligned}
    \mathbf{E}   &= [\begin{smallmatrix}\mathbf{0} & \mathbf{E^†} \end{smallmatrix}]    \\
    \mathbf{E^†} &= \text{diag}\mathbf{(Ĉ,Ĉ,...,Ĉ)}                                     \\
    \mathbf{J}   &= \text{diag}\mathbf{(D̂_d,D̂_d,...,D̂_d)}                               \\
    \mathbf{ex̂}  &= [\begin{smallmatrix}\mathbf{0} & \mathbf{I}\end{smallmatrix}]   
    \end{aligned}
    ```
"""
function init_predmat(
    model::LinModel, estim::StateEstimator{NT}, transcription::MultipleShooting, Hp, Hc
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

@doc raw"""
    init_predmat(model::SimModel, estim, transcription::MultipleShooting, Hp, Hc)

Return empty matrices except `ex̂` for [`SimModel`](@ref) and [`MultipleShooting`](@ref).

The matrix is ``\mathbf{e_x̂} = [\begin{smallmatrix}\mathbf{0} & \mathbf{I}\end{smallmatrix}]``.
"""
function init_predmat(
    model::SimModel, estim::StateEstimator{NT}, transcription::MultipleShooting, Hp, Hc
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
    gx̂, jx̂, kx̂, vx̂, bx̂ = G, J, K, V, B
    return E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂
end

"""
    init_predmat(model::SimModel, estim, transcription::TranscriptionMethod, Hp, Hc) 

Return empty matrices for all other cases.
"""
function init_predmat(
    model::SimModel, estim::StateEstimator{NT}, transcription::TranscriptionMethod, Hp, Hc
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
    init_defectmat(model::LinModel, estim, transcription::MultipleShooting, Hp, Hc)

Init the matrices for computing the defects over the predicted states. 

An equation similar to the prediction matrices (see 
[`init_predmat`](@ref)) computes the defects over the predicted states:
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
    The matrices are computed by:
    ```math
    \begin{aligned}
    \mathbf{E_ŝ} &= \begin{bmatrix}
        \mathbf{B̂_u} & \mathbf{0}   & \cdots & \mathbf{0}   & -\mathbf{I} &  \mathbf{0} & \cdots &  \mathbf{0}  &  \mathbf{0}      \\
        \mathbf{B̂_u} & \mathbf{B̂_u} & \cdots & \mathbf{0}   &  \mathbf{Â} & -\mathbf{I} & \cdots &  \mathbf{0}  &  \mathbf{0}      \\
        \vdots       & \vdots       & \ddots & \vdots       &  \vdots     &  \vdots     & \ddots &  \vdots      &  \vdots          \\
        \mathbf{B̂_u} & \mathbf{B̂_u} & \cdots & \mathbf{B̂_u} &  \mathbf{0} &  \mathbf{0} & \cdots &  \mathbf{Â}  & -\mathbf{I}      \end{bmatrix} \\
    \mathbf{G_ŝ} &= \begin{bmatrix}
        \mathbf{B̂_d} \\ \mathbf{0} \\ \vdots \\ \mathbf{0}                                                          \end{bmatrix} \\
    \mathbf{J_ŝ} &= \begin{bmatrix}
        \mathbf{0}   & \mathbf{0}   & \cdots & \mathbf{0}   & \mathbf{0}                                            \\
        \mathbf{B̂_d} & \mathbf{0}   & \cdots & \mathbf{0}   & \mathbf{0}                                            \\
        \vdots       & \vdots       & \ddots & \vdots       & \vdots                                                \\
        \mathbf{0}   & \mathbf{0}   & \cdots & \mathbf{B̂_d} & \mathbf{0}                                            \end{bmatrix} \\
    \mathbf{K_ŝ} &= \begin{bmatrix}
        \mathbf{Â} \\ \mathbf{0} \\ \vdots \\ \mathbf{0}                                                            \end{bmatrix} \\
    \mathbf{V_ŝ} &= \begin{bmatrix}
        \mathbf{B̂_u} \\ \mathbf{B̂_u} \\ \vdots \\ \mathbf{B̂_u}                                                      \end{bmatrix} \\
    \mathbf{B_ŝ} &= \begin{bmatrix}
        \mathbf{f̂_{op} - x̂_{op}} \\ \mathbf{f̂_{op} - x̂_{op}} \\ \vdots \\ \mathbf{f̂_{op} - x̂_{op}}                  \end{bmatrix}
    \end{aligned}
    ```
"""
function init_defectmat(
    model::LinModel, estim::StateEstimator{NT}, transcription::MultipleShooting, Hp, Hc
) where {NT<:Real}
    nu, nx̂, nd = model.nu, estim.nx̂, model.nd
    Â, B̂u, B̂d = estim.Â, estim.B̂u, estim.B̂d
    # --- current state estimates x̂0 ---
    Kŝ = [Â; zeros(NT, nx̂*(Hp-1), nx̂)]
    # --- previous manipulated inputs lastu0 ---
    Vŝ = repeat(B̂u, Hp)
    # --- decision variables Z ---
    nI_nx̂ = Matrix{NT}(-I, nx̂, nx̂)
    Eŝ = [zeros(nx̂*Hp, nu*Hc) repeatdiag(nI_nx̂, Hp)]
    for j=1:Hc, i=j:Hp
        iRow = (1:nx̂) .+ nx̂*(i-1)
        iCol = (1:nu) .+ nu*(j-1)
        Eŝ[iRow, iCol] = B̂u
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
    init_defectmat(model::SimModel, estim, transcription::TranscriptionMethod, Hp, Hc)

Return empty matrices for all other cases (N/A).
"""
function init_defectmat(
    model::SimModel, estim::StateEstimator{NT}, transcription::TranscriptionMethod, Hp, Hc
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
    linconstrainteq!(
        mpc::PredictiveController, model::LinModel, transcription::MultipleShooting
    )

Set `beq` vector for the linear model equality constraints (``\mathbf{Aeq Z̃ = beq}``).

Also init ``\mathbf{F_ŝ} = \mathbf{G_ŝ d_0}(k) + \mathbf{J_ŝ D̂_0} + \mathbf{K_ŝ x̂_0}(k) + 
\mathbf{V_ŝ u_0}(k-1) + \mathbf{B_ŝ}``, see [`init_defectmat`](@ref).
"""
function linconstrainteq!(mpc::PredictiveController, model::LinModel, ::MultipleShooting)
    Fŝ  = mpc.con.Fŝ
    Fŝ .= mpc.con.Bŝ
    mul!(Fŝ, mpc.con.Kŝ, mpc.estim.x̂0, 1, 1)
    mul!(Fŝ, mpc.con.Vŝ, mpc.estim.lastu0, 1, 1)
    if model.nd ≠ 0
        mul!(Fŝ, mpc.con.Gŝ, mpc.d0, 1, 1)
        mul!(Fŝ, mpc.con.Jŝ, mpc.D̂0, 1, 1)
    end
    mpc.con.beq .= @. -Fŝ
    linconeq = mpc.optim[:linconstrainteq]
    JuMP.set_normalized_rhs(linconeq, mpc.con.beq)
    return nothing
end
linconstrainteq!(::PredictiveController, ::SimModel, ::SingleShooting) = nothing
linconstrainteq!(::PredictiveController, ::SimModel, ::MultipleShooting) = nothing

@doc raw"""
    set_warmstart!(mpc::PredictiveController, transcription::SingleShooting, Z̃var) -> Z̃0

Set and return the warm start value of `Z̃var` for [`SingleShooting`](@ref) transcription.

If supported by `mpc.optim`, it warm-starts the solver at:
```math
\mathbf{ΔŨ} = 
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
function set_warmstart!(mpc::PredictiveController, transcription::SingleShooting, Z̃var)
    nu, Hc, Z̃0 = mpc.estim.model.nu, mpc.Hc, mpc.buffer.Z̃
    # --- input increments ΔU ---
    Z̃0[1:(Hc*nu-nu)] .= @views mpc.Z̃[nu+1:Hc*nu]
    Z̃0[(Hc*nu-nu+1):(Hc*nu)] .= 0
    # --- slack variable ϵ ---
    mpc.nϵ == 1 && (Z̃0[end] = mpc.Z̃[end])
    JuMP.set_start_value.(Z̃var, Z̃0)
    return Z̃0
end

@doc raw"""
    set_warmstart!(mpc::PredictiveController, transcription::MultipleShooting, Z̃var) -> Z̃0

Set and return the warm start value of `Z̃var` for [`MultipleShooting`](@ref) transcription.

It warm-starts the solver at:
```math
\mathbf{ΔŨ} =
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
function set_warmstart!(mpc::PredictiveController, transcription::MultipleShooting, Z̃var)
    nu, nx̂, Hp, Hc, Z̃0 = mpc.estim.model.nu, mpc.estim.nx̂, mpc.Hp, mpc.Hc, mpc.buffer.Z̃
    # --- input increments ΔU ---
    Z̃0[1:(Hc*nu-nu)] .= @views mpc.Z̃[nu+1:Hc*nu]
    Z̃0[(Hc*nu-nu+1):(Hc*nu)] .= 0
    # --- predicted states X̂0 ---
    Z̃0[(Hc*nu+1):(Hc*nu+Hp*nx̂-nx̂)]       .= @views mpc.Z̃[(Hc*nu+nx̂+1):(Hc*nu+Hp*nx̂)]
    Z̃0[(Hc*nu+Hp*nx̂-nx̂+1):(Hc*nu+Hp*nx̂)] .= @views mpc.Z̃[(Hc*nu+Hp*nx̂-nx̂+1):(Hc*nu+Hp*nx̂)]
    # --- slack variable ϵ ---
    mpc.nϵ == 1 && (Z̃0[end] = mpc.Z̃[end])
    JuMP.set_start_value.(Z̃var, Z̃0)
    return Z̃0
end

@doc raw"""
    predict!(
        Ŷ0, x̂0, _, _, _, 
        mpc::PredictiveController, model::LinModel, transcription::TranscriptionMethod, 
        Z̃
    ) -> Ŷ0, x̂0end

Compute the predictions `Ŷ0` and terminal states `x̂0end` if model is a [`LinModel`](@ref).

The method mutates `Ŷ0` and `x̂0` vector arguments. The `x̂end` vector is used for
the terminal constraints applied on ``\mathbf{x̂}_{k-1}(k+H_p)``. The computations are
identical for any [`TranscriptionMethod`](@ref).
"""
function predict!(
    Ŷ0, x̂0, _ , _ , _ , 
    mpc::PredictiveController, ::LinModel, ::TranscriptionMethod, 
    Z̃
)
    # in-place operations to reduce allocations :
    Ŷ0 .= mul!(Ŷ0, mpc.Ẽ, Z̃) .+ mpc.F
    x̂0 .= mul!(x̂0, mpc.con.ẽx̂, Z̃) .+ mpc.con.fx̂
    x̂0end = x̂0
    return Ŷ0, x̂0end
end

@doc raw"""
    predict!(
        Ŷ0, x̂0, x̂0next, u0, û0, 
        mpc::PredictiveController, model::NonLinModel, transcription::SingleShooting,
        Z̃
    ) -> Ŷ0, x̂0end

Compute vectors if `model` is a [`NonLinModel`](@ref) and for [`SingleShooting`](@ref).
    
The method mutates `Ŷ0`, `x̂0`, `x̂0next`, `u0` and `û0` arguments.
"""
function predict!(
    Ŷ0, x̂0, x̂0next, u0, û0, 
    mpc::PredictiveController, model::NonLinModel, ::SingleShooting,
    Z̃
)
    nu, ny, nd, Hp, Hc = model.nu, model.ny, model.nd, mpc.Hp, mpc.Hc
    ΔŨ  = Z̃ # only true for SingleShooting transcription method
    D̂0  = mpc.D̂0
    x̂0 .= mpc.estim.x̂0
    u0 .= mpc.estim.lastu0
    d0  = @views mpc.d0[1:end]
    for j=1:Hp
        if j ≤ Hc
            u0 .+= @views ΔŨ[(1 + nu*(j-1)):(nu*j)]
        end
        f̂!(x̂0next, û0, mpc.estim, model, x̂0, u0, d0)
        x̂0next .+= mpc.estim.f̂op .- mpc.estim.x̂op
        x̂0 .= x̂0next
        d0  = @views D̂0[(1 + nd*(j-1)):(nd*j)]
        ŷ0  = @views Ŷ0[(1 + ny*(j-1)):(ny*j)]
        ĥ!(ŷ0, mpc.estim, model, x̂0, d0)
    end
    Ŷ0 .+= mpc.F # F = Ŷs if mpc.estim is an InternalModel, else F = 0.
    x̂0end = x̂0
    return Ŷ0, x̂0end
end

@doc raw"""
    predict!(
        Ŷ0, x̂0, _ , _ , _ , 
        mpc::PredictiveController, model::NonLinModel, transcription::MultipleShooting,
        Z̃
    ) -> Ŷ0, x̂0end

Compute vectors if `model` is a [`NonLinModel`](@ref) and for [`MultipleShooting`](@ref).
    
The method mutates `Ŷ0` and `x̂0` arguments.
"""
function predict!(
    Ŷ0, x̂0, x̂0next, u0, û0,
    mpc::PredictiveController, model::NonLinModel, ::MultipleShooting,
    Z̃
)
    nu, ny, nd, nx̂, Hp, Hc = model.nu, model.ny, model.nd, mpc.estim.nx̂, mpc.Hp, mpc.Hc
    X̂0 = @views Z̃[(nu*Hc+1):(nu*Hc+nx̂*Hp)] # Z̃ = [ΔU; X̂0; ϵ]
    D̂0 = mpc.D̂0
    for j=1:Hp
        x̂0 .= @views X̂0[(1 + nx̂*(j-1)):(nx̂*j)]
        d0  = @views D̂0[(1 + nd*(j-1)):(nd*j)]
        ŷ0  = @views Ŷ0[(1 + ny*(j-1)):(ny*j)]
        ĥ!(ŷ0, mpc.estim, model, x̂0, d0)
    end
    Ŷ0 .+= mpc.F # F = Ŷs if mpc.estim is an InternalModel, else F = 0.
    x̂0end = x̂0
    return Ŷ0, x̂0end
end