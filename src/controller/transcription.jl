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
This method generally more efficient for small control horizon ``H_c``, stable or mildly
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
\mathbf{X̂_0} =                                  \begin{bmatrix} 
    \mathbf{x̂}(k+1)     - \mathbf{x̂_{op}}       \\ 
    \mathbf{x̂}(k+2)     - \mathbf{x̂_{op}}       \\ 
    \vdots                                      \\ 
    \mathbf{x̂}(k+H_p)   - \mathbf{x̂_{op}}       \end{bmatrix}
```
This method is generally more efficient for large control horizon ``H_c``, unstable or
highly nonlinear plant models/constraints.
"""
struct MultipleShooting <: TranscriptionMethod end


@doc raw"""
    init_ZtoΔU(estim::StateEstimator, transcription::TranscriptionMethod, Hp, Hc) -> P

Init decision variables to input increments over ``H_c`` conversion matrix `P`.

The conversion from the decision variables ``\mathbf{Z}`` to ``\mathbf{ΔU}``, the input
increments over ``H_c``, is computed by:
```math
\mathbf{ΔU} = \mathbf{P} \mathbf{Z}
```
in which ``\mathbf{P} is defined in the Extended Help section.

# Extended Help
!!! details "Extended Help"
    Following the decision variable definition of the [`TranscriptionMethod`](@ref), the
    conversion matrix ``\mathbf{P}``, we have:
    - ``\mathbf{P} = \mathbf{I}`` if `transcription isa SingleShooting`
    - ``\mathbf{P} = [\begin{smallmatrix}\mathbf{I} \mathbf{0} \end{smallmatrix}]`` if 
      `transcription isa MultipleShooting`
"""
function init_ZtoU end

function init_ZtoΔU(
    estim::StateEstimator{NT}, transcription::SingleShooting, _ , Hc
) where {NT<:Real}
    return Matrix{NT}(I, estim.model.nu*Hc, estim.model.nu*Hc)
end

function init_ZtoΔU(
    estim::StateEstimator{NT}, transcription::MultipleShooting, Hp, Hc
) where {NT<:Real}
    I_nu_Hc = Matrix{NT}(I, estim.model.nu*Hc, estim.model.nu*Hc)
    return [I_nu_Hc zeros(NT, estim.model.nu*Hc, estim.nx̂*Hp)]
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
    - ``\mathbf{S} = \mathbf{S^†}`` if `transcription isa SingleShooting`
    - ``\mathbf{S} = [\begin{smallmatrix}\mathbf{S^†} \mathbf{0} \end{smallmatrix}]`` if 
      `transcription isa MultipleShooting`
"""
function init_ZtoU(
    estim::StateEstimator{NT}, transcription::TranscriptionMethod, Hp, Hc
) where {NT<:Real}
    model = estim.model
    # S and T are `Matrix{NT}` since conversion is faster than `Matrix{Bool}` or `BitMatrix`
    I_nu = Matrix{NT}(I, model.nu, model.nu)
    S_Hc = LowerTriangular(repeat(I_nu, Hc, Hc))
    Sdagger = [S_Hc; repeat(I_nu, Hp - Hc, Hc)]
    S = init_ZtoU_Smat(estim, transcription, Hp, Hc, Sdagger)
    T = repeat(I_nu, Hp)
    return S, T
end

init_ZtoU_Smat( _ , transcription::SingleShooting, _ , _ , Sdagger) = Sdagger

function init_ZtoU_Smat(estim, transcription::MultipleShooting, Hp, _ , Sdagger)
    return [Sdagger; zeros(eltype(Sdagger), estim.model.nu*Hp, estim.nx̂*Hp)]
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
    # --- state estimates x̂ ---
    kx̂ = getpower(Âpow, Hp)
    K  = Matrix{NT}(undef, Hp*ny, nx̂)
    for j=1:Hp
        iRow = (1:ny) .+ ny*(j-1)
        K[iRow,:] = Ĉ*getpower(Âpow, j)
    end    
    # --- manipulated inputs u ---
    vx̂ = getpower(Âpow_csum, Hp-1)*B̂u
    V  = Matrix{NT}(undef, Hp*ny, nu)
    for j=1:Hp
        iRow = (1:ny) .+ ny*(j-1)
        V[iRow,:] = Ĉ*getpower(Âpow_csum, j-1)*B̂u
    end
    ex̂ = Matrix{NT}(undef, nx̂, Hc*nu)
    # --- decision variables Z ---
    E  = zeros(NT, Hp*ny, Hc*nu) 
    for j=1:Hc # truncated with control horizon
        iRow = (ny*(j-1)+1):(ny*Hp)
        iCol = (1:nu) .+ nu*(j-1)
        E[iRow, iCol] = V[iRow .- ny*(j-1),:]
        ex̂[:  , iCol] = getpower(Âpow_csum, Hp-j)*B̂u
    end
    # --- measured disturbances d ---
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
"""
function init_predmat(
    model::LinModel, estim::StateEstimator{NT}, transcription::MultipleShooting, Hp, Hc
) where {NT<:Real}
    Ĉ, D̂d = estim.Ĉ, estim.D̂d
    nu, nx̂, ny, nd = model.nu, estim.nx̂, model.ny, model.nd
    # --- state estimates x̂ ---
    K = zeros(NT, Hp*ny, nx̂)
    # --- manipulated inputs u ---
    V = zeros(NT, Hp*ny, nu)
    # --- decision variables Z ---
    E  = [zeros(NT, Hp*ny, Hc*nu) repeatdiag(Ĉ, Hp)]
    ex̂ = [zeros(NT, nx̂, Hc*nu + (Hp-1)*nx̂) I]
    # --- measured disturbances d ---
    G  = zeros(NT, Hp*ny, nd)
    gx̂ = zeros(NT, nx̂, Hp*nd)
    J  = repeatdiag(D̂d, Hp)
    jx̂ = zeros(NT, nx̂, Hp*nd)
    # --- state x̂op and state update f̂op operating points ---
    B  = zeros(NT, Hp*ny, 1)
    bx̂ = zeros(NT, nx̂, 1)
    return E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂
end

"""
    init_predmat(model::SimModel, estim, transcription::SingleShooting, Hp, Hc) 

Return empty matrices for [`SimModel`](@ref) and [`SingleShooting`](@ref) (N/A).
"""
function init_predmat(
    model::SimModel, estim::StateEstimator{NT}, transcription::SingleShooting, Hp, Hc
) where {NT<:Real}
    nu, nx̂, nd = model.nu, estim.nx̂, model.nd
    E  = zeros(NT, 0, nu*Hc)
    G  = zeros(NT, 0, nd)
    J  = zeros(NT, 0, nd*Hp)
    K  = zeros(NT, 0, nx̂)
    V  = zeros(NT, 0, nu)
    B  = zeros(NT, 0)
    ex̂, gx̂, jx̂, kx̂, vx̂, bx̂ = E, G, J, K, V, B
    return E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂
end

"""
    init_predmat(model::SimModel, estim, transcription::MultipleShooting, Hp, Hc)

Return empty matrices except `ex̂` for [`SimModel`](@ref) and [`MultipleShooting`](@ref).
"""
function init_predmat(
    model::SimModel, estim::StateEstimator{NT}, transcription::MultipleShooting, Hp, Hc
) where {NT<:Real}
    nu, nx̂, nd = model.nu, estim.nx̂, model.nd
    E  = zeros(NT, 0, nu*Hc)
    G  = zeros(NT, 0, nd)
    J  = zeros(NT, 0, nd*Hp)
    K  = zeros(NT, 0, nx̂)
    V  = zeros(NT, 0, nu)
    B  = zeros(NT, 0)
    ex̂ = [zeros(NT, nx̂, Hc*nu + (Hp-1)*nx̂) I]
    gx̂, jx̂, kx̂, vx̂, bx̂ = E, G, J, K, V
    return E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂
end

"""
    init_defectmat(model::SimModel, estim, transcription::SingleShooting, Hp, Hc)

Return empty matrices if `transcription` is a [`SingleShooting`](@ref) (N/A).
"""
function init_defectmat(
    model::SimModel, estim::StateEstimator{NT}, transcription::SingleShooting, Hp, Hc
) where {NT<:Real}
    nx̂, nu, nd = estim.nx̂, model.nu, model.nd
    Eŝ = zeros(NT, 0, nu*Hc)
    Gŝ = zeros(NT, 0, nd)
    Jŝ = zeros(NT, 0, nd*Hp)
    Kŝ = zeros(NT, 0, nx̂)
    Vŝ = zeros(NT, 0, nu)
    Bŝ = zeros(NT, 0)
    return Eŝ, Gŝ, Jŝ, Kŝ, Vŝ, Bŝ
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
        \mathbf{B̂_u} & \mathbf{0}   & \cdots & \mathbf{0}   & -\mathbf{I} &  \mathbf{0} & \cdots &  \mathbf{0}    \\
        \mathbf{B̂_u} & \mathbf{B̂_u} & \cdots & \mathbf{0}   &  \mathbf{Â} & -\mathbf{I} & \cdots &  \mathbf{0}    \\
        \vdots       & \vdots       & \ddots & \vdots       &  \vdots     &  \vdots     & \ddots & \vdots         \\
        \mathbf{B̂_u} & \mathbf{B̂_u} & \cdots & \mathbf{B̂_u} &  \mathbf{0} &  \mathbf{0} & \cdots & -\mathbf{I}    \end{bmatrix} \\
    \mathbf{G_ŝ} &= \begin{bmatrix}
        \mathbf{B̂_d} \\ \mathbf{0} \\ \vdots \\ \mathbf{0}                                                      \end{bmatrix} \\
    \mathbf{J_ŝ} &= \begin{bmatrix}
        \mathbf{0}   & \mathbf{0}   & \cdots & \mathbf{0}   & \mathbf{0}                                        \\
        \mathbf{B̂_d} & \mathbf{0}   & \cdots & \mathbf{0}   & \mathbf{0}                                        \\
        \vdots       & \vdots       & \ddots & \vdots       & \vdots                                            \\
        \mathbf{0}   & \mathbf{0}   & \cdots & \mathbf{B̂_d} & \mathbf{0}                                        \end{bmatrix} \\
    \mathbf{K_ŝ} &= \begin{bmatrix}
        \mathbf{Â} \\ \mathbf{0} \\ \vdots \\ \mathbf{0}                                                        \end{bmatrix} \\
    \mathbf{V_ŝ} &= \begin{bmatrix}
        \mathbf{B̂_u} \\ \mathbf{B̂_u} \\ \vdots \\ \mathbf{B̂_u}                                                  \end{bmatrix} \\
    \mathbf{B_ŝ} &= \begin{bmatrix}
        \mathbf{f̂_{op} - x̂_{op}} \\ \mathbf{f̂_{op} - x̂_{op}} \\ \vdots \\ \mathbf{f̂_{op} - x̂_{op}}              \end{bmatrix}
    \end{aligned}
    ```
"""
function init_defectmat(
    model::LinModel, estim::StateEstimator{NT}, transcription::MultipleShooting, Hp, Hc
) where {NT<:Real}
    nu, nx̂, nd = model.nu, estim.nx̂, model.nd
    Â, B̂u, B̂d = estim.Â, estim.B̂u, estim.B̂d
    # --- state estimates x̂ ---
    Kŝ = [Â; zeros(NT, nx̂*(Hp-1), nx̂)]
    # --- manipulated inputs u ---
    Vŝ = repeat(B̂u, Hp)
    # --- decision variables Z ---
    nI_nu = Matrix{NT}(-I, nu, nu)
    Eŝ = [LowerTriangular(repeat(B̂u, Hc, Hc)) repeatdiag(nI_nu, Hp)]
    # --- measured disturbances d ---
    Gŝ = [B̂d; zeros(NT, (Hp-1)*nx̂, nd)]
    Jŝ = [zeros(NT, nx̂, Hp*nd); repeatdiag(B̂d, Hp-1) zeros(NT, (Hp-1)*nd, nd)]
    # --- state x̂op and state update f̂op operating points ---
    B̂s = repeat(estim.f̂op - estim.x̂op, Hp)
    return Eŝ, Gŝ, Jŝ, Kŝ, Vŝ, Bŝ
end

"Return empty matrices if `model` is not a [`LinModel`](@ref) (N/A)."
function init_defectmat(
    model::SimModel, estim::StateEstimator{NT}, transcription::TranscriptionMethod, Hp, Hc
) where {NT<:Real}
    nx̂, nu, nd = estim.nx̂, model.nu, model.nd
    Eŝ = zeros(NT, 0, nu*Hc)
    Gŝ = zeros(NT, 0, nd)
    Jŝ = zeros(NT, 0, nd*Hp)
    Kŝ = zeros(NT, 0, nx̂)
    Vŝ = zeros(NT, 0, nu)
    Bŝ = zeros(NT, 0)
    return Eŝ, Gŝ, Jŝ, Kŝ, Vŝ, Bŝ
end