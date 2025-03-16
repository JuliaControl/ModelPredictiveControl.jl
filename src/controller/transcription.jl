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
This method is generally more efficient for small control horizon ``H_c``, stable and mildly
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
observer at time ``i=k`` or ``i=k-1`` depending on its `direct` flag. Note that 
``\mathbf{X̂_0 = X̂}`` if the operating points is zero, which is typically the case in 
practice for [`NonLinModel`](@ref). This transcription method is generally more efficient
for large control horizon ``H_c``, unstable or highly nonlinear plant models/constraints. 

Sparse optimizers like `OSQP` or `Ipopt` are recommended for this method.
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
    - ``\mathbf{P_{Δu}} = [\begin{smallmatrix}\mathbf{I} & \mathbf{0} \end{smallmatrix}]``
      if `transcription` is a [`MultipleShooting`](@ref)
"""
function init_ZtoΔU end

function init_ZtoΔU(
    estim::StateEstimator{NT}, transcription::SingleShooting, _ , Hc
) where {NT<:Real}
    PΔu = Matrix{NT}(I, estim.model.nu*Hc, estim.model.nu*Hc)
    return PΔu
end

function init_ZtoΔU(
    estim::StateEstimator{NT}, transcription::MultipleShooting, Hp, Hc
) where {NT<:Real}
    I_nu_Hc = Matrix{NT}(I, estim.model.nu*Hc, estim.model.nu*Hc)
    PΔu = [I_nu_Hc zeros(NT, estim.model.nu*Hc, estim.nx̂*Hp)]
    return PΔu
end

@doc raw"""
    init_ZtoU(estim, transcription, Hp, Hc) -> Pu, Tu

Init decision variables to inputs over ``H_p`` conversion matrices.

The conversion from the decision variables ``\mathbf{Z}`` to ``\mathbf{U}``, the manipulated
inputs over ``H_p``, is computed by:
```math
\mathbf{U} = \mathbf{P_u} \mathbf{Z} + \mathbf{T_u} \mathbf{u}(k-1)
```
The ``\mathbf{P_u}`` and ``\mathbf{T_u}`` matrices are defined in the Extended Help section.

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
    \mathbf{P_u^†} = \begin{bmatrix}
        \mathbf{I}  & \mathbf{0}    & \cdots    & \mathbf{0}        \\
        \mathbf{I}  & \mathbf{I}    & \cdots    & \mathbf{0}        \\
        \vdots      & \vdots        & \ddots    & \vdots            \\
        \mathbf{I}  & \mathbf{I}    & \cdots    & \mathbf{I}        \\
        \vdots      & \vdots        & \ddots    & \vdots            \\
        \mathbf{I}  & \mathbf{I}    & \cdots    & \mathbf{I}        \end{bmatrix} , \quad
    \mathbf{T_u} = \begin{bmatrix}
        \mathbf{I}                                                  \\
        \mathbf{I}                                                  \\
        \vdots                                                      \\
        \mathbf{I}                                                  \\
        \vdots                                                      \\
        \mathbf{I}                                                  \end{bmatrix}
    ```
    and, depending on the transcription method, we have:
    - ``\mathbf{P_u} = \mathbf{P_u^†}`` if `transcription` is a [`SingleShooting`](@ref)
    - ``\mathbf{P_u} = [\begin{smallmatrix}\mathbf{P_u^†} & \mathbf{0} \end{smallmatrix}]``
      if `transcription` is a [`MultipleShooting`](@ref)
"""
function init_ZtoU(
    estim::StateEstimator{NT}, transcription::TranscriptionMethod, Hp, Hc
) where {NT<:Real}
    model = estim.model
    # Pu and Tu are `Matrix{NT}`, conversion is faster than `Matrix{Bool}` or `BitMatrix`
    I_nu = Matrix{NT}(I, model.nu, model.nu)
    PU_Hc = LowerTriangular(repeat(I_nu, Hc, Hc))
    PUdagger = [PU_Hc; repeat(I_nu, Hp - Hc, Hc)]
    Pu = init_PUmat(estim, transcription, Hp, Hc, PUdagger)
    Tu = repeat(I_nu, Hp)
    return Pu, Tu
end

init_PUmat( _ , transcription::SingleShooting, _ , _ , PUdagger) = PUdagger
function init_PUmat(estim, transcription::MultipleShooting, Hp, _ , PUdagger)
    return [PUdagger zeros(eltype(PUdagger), estim.model.nu*Hp, estim.nx̂*Hp)]
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
    \mathbf{e_x̂} &= [\begin{smallmatrix}\mathbf{0} & \mathbf{I}\end{smallmatrix}]   
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

Return the terminal state matrices for [`SimModel`](@ref) and [`MultipleShooting`](@ref).

The output prediction matrices are all empty matrices. The terminal state matrices are
given in the Extended Help section.

# Extended Help
!!! details "Extended Help"
    The terminal state matrices all appropriately sized zero matrices ``\mathbf{0}``, except
    for ``\mathbf{e_x̂} = [\begin{smallmatrix}\mathbf{0} & \mathbf{I}\end{smallmatrix}]``
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
    gx̂ = zeros(NT, nx̂, nd)
    jx̂ = zeros(NT, nx̂, nd*Hp)
    kx̂ = zeros(NT, nx̂, nx̂)
    vx̂ = zeros(NT, nx̂, nu)
    bx̂ = zeros(NT, nx̂)
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
    init_matconstraint_mpc(
        model::LinModel, transcription::TranscriptionMethod, nc::Int,
        i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ymin, i_Ymax, i_x̂min, i_x̂max, 
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
`A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, A_Ymin, A_Ymax, A_x̂min, A_x̂max, A_ŝ`. The integer `neq`
is the number of nonlinear equality constraints in ``\mathbf{g_{eq}}``.
"""
function init_matconstraint_mpc(
    ::LinModel{NT}, ::TranscriptionMethod, nc::Int,
    i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ymin, i_Ymax, i_x̂min, i_x̂max, 
    args...
) where {NT<:Real}
    if isempty(args)
        A, Aeq, neq = nothing, nothing, nothing
    else
        A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, A_Ymin, A_Ymax, A_x̂min, A_x̂max, A_ŝ = args
        A   = [A_Umin; A_Umax; A_ΔŨmin; A_ΔŨmax; A_Ymin; A_Ymax; A_x̂min; A_x̂max]
        Aeq = A_ŝ
        neq = 0
    end
    i_b = [i_Umin; i_Umax; i_ΔŨmin; i_ΔŨmax; i_Ymin; i_Ymax; i_x̂min; i_x̂max]
    i_g = trues(nc)
    return i_b, i_g, A, Aeq, neq
end

"Init `i_b` without output constraints if `NonLinModel` and not `SingleShooting`."
function init_matconstraint_mpc(
    ::NonLinModel{NT}, ::TranscriptionMethod, nc::Int,
    i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ymin, i_Ymax, i_x̂min, i_x̂max, 
    args...
) where {NT<:Real}
    if isempty(args)
        A, Aeq, neq = nothing, nothing, nothing
    else
        A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, A_Ymin, A_Ymax, A_x̂min, A_x̂max, A_ŝ = args
        A   = [A_Umin; A_Umax; A_ΔŨmin; A_ΔŨmax; A_Ymin; A_Ymax; A_x̂min; A_x̂max]
        Aeq = A_ŝ
        nΔŨ, nZ̃ = size(A_ΔŨmin)
        neq = nZ̃ - nΔŨ
    end
    i_b = [i_Umin; i_Umax; i_ΔŨmin; i_ΔŨmax; i_x̂min; i_x̂max]
    i_g = [i_Ymin; i_Ymax; trues(nc)]
    return i_b, i_g, A, Aeq, neq
end

"Init `i_b` without output & terminal constraints if `NonLinModel` and `SingleShooting`."
function init_matconstraint_mpc(
    ::NonLinModel{NT}, ::SingleShooting, nc::Int,
    i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ymin, i_Ymax, i_x̂min, i_x̂max, 
    args...
) where {NT<:Real}
    if isempty(args)
        A, Aeq, neq = nothing, nothing, nothing
    else
        A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, A_Ymin, A_Ymax, A_x̂min, A_x̂max, A_ŝ = args
        A   = [A_Umin; A_Umax; A_ΔŨmin; A_ΔŨmax; A_Ymin; A_Ymax; A_x̂min; A_x̂max]
        Aeq = A_ŝ
        neq = 0
    end
    i_b = [i_Umin; i_Umax; i_ΔŨmin; i_ΔŨmax]
    i_g = [i_Ymin; i_Ymax; i_x̂min;  i_x̂max; trues(nc)]
    return i_b, i_g, A, Aeq, neq
end

"""
    init_nonlincon!(
        mpc::PredictiveController, model::LinModel, transcription::TranscriptionMethod, 
        gfuncs  , ∇gfuncs!,   
        geqfuncs, ∇geqfuncs!
    )

Init nonlinear constraints for [`LinModel`](@ref) for all [`TranscriptionMethod`](@ref).

The only nonlinear constraints are the custom inequality constraints `gc`.
"""
function init_nonlincon!(
    mpc::PredictiveController, ::LinModel, ::TranscriptionMethod,
    gfuncs, ∇gfuncs!, 
    _ , _    
) 
    optim, con = mpc.optim, mpc.con
    nZ̃ = length(mpc.Z̃)
    if length(con.i_g) ≠ 0
        i_base = 0
        for i in 1:con.nc
            name = Symbol("g_c_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_base+i]; name
            )
        end
    end
    return nothing
end

"""
    init_nonlincon!(
        mpc::PredictiveController, model::NonLinModel, transcription::MultipleShooting, 
        gfuncs,   ∇gfuncs!,
        geqfuncs, ∇geqfuncs!
    )
    
Init nonlinear constraints for [`NonLinModel`](@ref) and [`MultipleShooting`](@ref).

The nonlinear constraints are the output prediction `Ŷ` bounds, the custom inequality
constraints `gc` and all the nonlinear equality constraints `geq`.
"""
function init_nonlincon!(
    mpc::PredictiveController, ::NonLinModel, ::MultipleShooting, 
    gfuncs,     ∇gfuncs!,
    geqfuncs,   ∇geqfuncs!
) 
    optim, con = mpc.optim, mpc.con
    ny, nx̂, Hp, nZ̃ = mpc.estim.model.ny, mpc.estim.nx̂, mpc.Hp, length(mpc.Z̃)
    # --- nonlinear inequality constraints ---
    if length(con.i_g) ≠ 0
        i_base = 0
        for i in eachindex(con.Y0min)
            name = Symbol("g_Y0min_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_base+i], ∇gfuncs![i_base+i]; name
            )
        end
        i_base = 1Hp*ny
        for i in eachindex(con.Y0max)
            name = Symbol("g_Y0max_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_base+i], ∇gfuncs![i_base+i]; name
            )
        end
        i_base = 2Hp*ny
        for i in 1:con.nc
            name = Symbol("g_c_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_base+i], ∇gfuncs![i_base+i]; name
            )
        end
    end
    # --- nonlinear equality constraints ---
    Z̃var = optim[:Z̃var]
    for i in eachindex(geqfuncs)
        name = Symbol("geq_$i")
        geqfunc_i = optim[name] = JuMP.add_nonlinear_operator(
            optim, nZ̃, geqfuncs[i], ∇geqfuncs![i]; name
        )
        # set with @constrains here instead of set_nonlincon!, since the number of nonlinear 
        # equality constraints is known and constant (±Inf are impossible):
        @constraint(optim, geqfunc_i(Z̃var...) == 0)
    end
    return nothing
end

"""
    init_nonlincon!(
        mpc::PredictiveController, model::NonLinModel, ::SingleShooting, 
        gfuncs,   ∇gfuncs!,
        geqfuncs, ∇geqfuncs!
    )

Init nonlinear constraints for [`NonLinModel`](@ref) and [`SingleShooting`](@ref).

The nonlinear constraints are the custom inequality constraints `gc`, the output
prediction `Ŷ` bounds and the terminal state `x̂end` bounds.
"""
function init_nonlincon!(
    mpc::PredictiveController, ::NonLinModel, ::SingleShooting, gfuncs, ∇gfuncs!, _ , _
)
    optim, con = mpc.optim, mpc.con
    ny, nx̂, Hp, nZ̃ = mpc.estim.model.ny, mpc.estim.nx̂, mpc.Hp, length(mpc.Z̃)
    if length(con.i_g) ≠ 0
        i_base = 0
        for i in eachindex(con.Y0min)
            name = Symbol("g_Y0min_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_base+i], ∇gfuncs![i_base+i]; name
            )
        end
        i_base = 1Hp*ny
        for i in eachindex(con.Y0max)
            name = Symbol("g_Y0max_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_base+i], ∇gfuncs![i_base+i]; name
            )
        end
        i_base = 2Hp*ny
        for i in eachindex(con.x̂0min)
            name = Symbol("g_x̂0min_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_base+i], ∇gfuncs![i_base+i]; name
            )
        end
        i_base = 2Hp*ny + nx̂
        for i in eachindex(con.x̂0max)
            name = Symbol("g_x̂0max_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_base+i], ∇gfuncs![i_base+i]; name
            )
        end
        i_base = 2Hp*ny + 2nx̂
        for i in 1:con.nc
            name = Symbol("g_c_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_base+i], ∇gfuncs![i_base+i]; name
            )
        end
    end
    return nothing
end


"By default, there is no nonlinear constraint, thus do nothing."
function set_nonlincon!(
    ::PredictiveController, ::SimModel, ::TranscriptionMethod, ::JuMP.GenericModel, 
    )
    return nothing
end

"""
    set_nonlincon!(mpc::PredictiveController, ::LinModel, ::TranscriptionMethod, optim)

Set the custom nonlinear inequality constraints for `LinModel`.
"""
function set_nonlincon!(
    mpc::PredictiveController, ::LinModel, ::TranscriptionMethod, ::JuMP.GenericModel{JNT}
) where JNT<:Real
    optim = mpc.optim
    Z̃var = optim[:Z̃var]
    con = mpc.con
    nonlin_constraints = JuMP.all_constraints(optim, JuMP.NonlinearExpr, MOI.LessThan{JNT})
    map(con_ref -> JuMP.delete(optim, con_ref), nonlin_constraints)
    for i in 1:con.nc
        gfunc_i = optim[Symbol("g_c_$i")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    return nothing
end

"""
    set_nonlincon!(mpc::PredictiveController, ::NonLinModel, ::MultipleShooting, optim)

Also set output prediction `Ŷ` constraints for `NonLinModel` and non-`SingleShooting`.
"""
function set_nonlincon!(
    mpc::PredictiveController, ::NonLinModel, ::TranscriptionMethod, ::JuMP.GenericModel{JNT}
) where JNT<:Real
    optim = mpc.optim
    Z̃var = optim[:Z̃var]
    con = mpc.con
    nonlin_constraints = JuMP.all_constraints(optim, JuMP.NonlinearExpr, MOI.LessThan{JNT})
    map(con_ref -> JuMP.delete(optim, con_ref), nonlin_constraints)
    for i in findall(.!isinf.(con.Y0min))
        gfunc_i = optim[Symbol("g_Y0min_$(i)")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    for i in findall(.!isinf.(con.Y0max))
        gfunc_i = optim[Symbol("g_Y0max_$(i)")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    for i in 1:con.nc
        gfunc_i = optim[Symbol("g_c_$i")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    return nothing
end

"""
    set_nonlincon!(mpc::PredictiveController, ::NonLinModel, ::SingleShooting, optim)

Also set output prediction `Ŷ` and terminal state `x̂end` constraint for `SingleShooting`.
"""
function set_nonlincon!(
    mpc::PredictiveController, ::NonLinModel, ::SingleShooting, ::JuMP.GenericModel{JNT}
) where JNT<:Real
    optim = mpc.optim
    Z̃var = optim[:Z̃var]
    con = mpc.con
    nonlin_constraints = JuMP.all_constraints(optim, JuMP.NonlinearExpr, MOI.LessThan{JNT})
    map(con_ref -> JuMP.delete(optim, con_ref), nonlin_constraints)
    for i in findall(.!isinf.(con.Y0min))
        gfunc_i = optim[Symbol("g_Y0min_$(i)")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    for i in findall(.!isinf.(con.Y0max))
        gfunc_i = optim[Symbol("g_Y0max_$(i)")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    for i in findall(.!isinf.(con.x̂0min))
        gfunc_i = optim[Symbol("g_x̂0min_$(i)")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    for i in findall(.!isinf.(con.x̂0max))
        gfunc_i = optim[Symbol("g_x̂0max_$(i)")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    for i in 1:con.nc
        gfunc_i = optim[Symbol("g_c_$i")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    return nothing
end

@doc raw"""
    linconstraint!(mpc::PredictiveController, model::LinModel)

Set `b` vector for the linear model inequality constraints (``\mathbf{A Z̃ ≤ b}``).

Also init ``\mathbf{f_x̂} = \mathbf{g_x̂ d_0}(k) + \mathbf{j_x̂ D̂_0} + \mathbf{k_x̂ x̂_0}(k) + 
\mathbf{v_x̂ u_0}(k-1) + \mathbf{b_x̂}`` vector for the terminal constraints, see
[`init_predmat`](@ref).
"""
function linconstraint!(mpc::PredictiveController, model::LinModel, ::TranscriptionMethod)
    nU, nΔŨ, nY = length(mpc.con.U0min), length(mpc.con.ΔŨmin), length(mpc.con.Y0min)
    nx̂, fx̂ = mpc.estim.nx̂, mpc.con.fx̂
    fx̂ .= mpc.con.bx̂
    mul!(fx̂, mpc.con.kx̂, mpc.estim.x̂0, 1, 1)
    mul!(fx̂, mpc.con.vx̂, mpc.estim.lastu0, 1, 1)
    if model.nd ≠ 0
        mul!(fx̂, mpc.con.gx̂, mpc.d0, 1, 1)
        mul!(fx̂, mpc.con.jx̂, mpc.D̂0, 1, 1)
    end
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
    nU, nΔŨ, nY = length(mpc.con.U0min), length(mpc.con.ΔŨmin), length(mpc.con.Y0min)
    nx̂, fx̂ = mpc.estim.nx̂, mpc.con.fx̂
    # here, updating fx̂ is not necessary since fx̂ = 0
    n = 0
    mpc.con.b[(n+1):(n+nU)]  .= @. -mpc.con.U0min + mpc.Tu_lastu0
    n += nU
    mpc.con.b[(n+1):(n+nU)]  .= @. +mpc.con.U0max - mpc.Tu_lastu0
    n += nU
    mpc.con.b[(n+1):(n+nΔŨ)] .= @. -mpc.con.ΔŨmin
    n += nΔŨ
    mpc.con.b[(n+1):(n+nΔŨ)] .= @. +mpc.con.ΔŨmax
    n += nΔŨ
    mpc.con.b[(n+1):(n+nx̂)]  .= @. -mpc.con.x̂0min
    n += nx̂
    mpc.con.b[(n+1):(n+nx̂)]  .= @. +mpc.con.x̂0max
    if any(mpc.con.i_b) 
        lincon = mpc.optim[:linconstraint]
        JuMP.set_normalized_rhs(lincon, mpc.con.b[mpc.con.i_b])
    end
end

"Also exclude terminal constraints for `NonLinModel` and `SingleShooting`."
function linconstraint!(mpc::PredictiveController, ::NonLinModel, ::SingleShooting)
    nU, nΔŨ = length(mpc.con.U0min), length(mpc.con.ΔŨmin)
    n = 0
    mpc.con.b[(n+1):(n+nU)]  .= @. -mpc.con.U0min + mpc.Tu_lastu0
    n += nU
    mpc.con.b[(n+1):(n+nU)]  .= @. +mpc.con.U0max - mpc.Tu_lastu0
    n += nU
    mpc.con.b[(n+1):(n+nΔŨ)] .= @. -mpc.con.ΔŨmin
    n += nΔŨ
    mpc.con.b[(n+1):(n+nΔŨ)] .= @. +mpc.con.ΔŨmax
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

getΔŨ!(ΔŨ, mpc::PredictiveController, ::SingleShooting, Z̃) = (ΔŨ .= Z̃) # since mpc.P̃Δu = I
getΔŨ!(ΔŨ, mpc::PredictiveController, ::TranscriptionMethod, Z̃) = mul!(ΔŨ, mpc.P̃Δu, Z̃)
getU0!(U0, mpc::PredictiveController, Z̃) = (mul!(U0, mpc.P̃u, Z̃) .+ mpc.Tu_lastu0)

@doc raw"""
    predict!(
        Ŷ0, x̂0end, _ , _ ,
        mpc::PredictiveController, model::LinModel, transcription::TranscriptionMethod, 
        _ , Z̃
    ) -> Ŷ0, x̂0end

Compute the predictions `Ŷ0`, terminal states `x̂0end` if model is a [`LinModel`](@ref).

The method mutates `Ŷ0` and `x̂0end` vector arguments. The `x̂end` vector is used for
the terminal constraints applied on ``\mathbf{x̂}_{k-1}(k+H_p)``. The computations are
identical for any [`TranscriptionMethod`](@ref) if the model is linear.
"""
function predict!(
    Ŷ0, x̂0end, _ , _ , 
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
        Ŷ0, x̂0end, X̂0, Û0, 
        mpc::PredictiveController, model::NonLinModel, transcription::SingleShooting,
        U0, _
    ) -> Ŷ0, x̂0end

Compute vectors if `model` is a [`NonLinModel`](@ref) and for [`SingleShooting`](@ref).
    
The method mutates `Ŷ0`, `x̂0end`, `X̂0` and `Û0` arguments.
"""
function predict!(
    Ŷ0, x̂0end, X̂0, Û0,
    mpc::PredictiveController, model::NonLinModel, ::SingleShooting,
    U0, _
)
    nu, nx̂, ny, nd, Hp, Hc = model.nu, mpc.estim.nx̂, model.ny, model.nd, mpc.Hp, mpc.Hc
    D̂0 = mpc.D̂0
    x̂0 = @views mpc.estim.x̂0[1:nx̂]
    d0 = @views mpc.d0[1:nd]
    for j=1:Hp
        u0     = @views U0[(1 + nu*(j-1)):(nu*j)]
        û0     = @views Û0[(1 + nu*(j-1)):(nu*j)]
        x̂0next = @views X̂0[(1 + nx̂*(j-1)):(nx̂*j)]
        f̂!(x̂0next, û0, mpc.estim, model, x̂0, u0, d0)
        x̂0next .+= mpc.estim.f̂op .- mpc.estim.x̂op
        x̂0 = @views X̂0[(1 + nx̂*(j-1)):(nx̂*j)]
        d0 = @views D̂0[(1 + nd*(j-1)):(nd*j)]
        ŷ0 = @views Ŷ0[(1 + ny*(j-1)):(ny*j)]
        ĥ!(ŷ0, mpc.estim, model, x̂0, d0)
    end
    Ŷ0    .+= mpc.F # F = Ŷs if mpc.estim is an InternalModel, else F = 0.
    x̂0end  .= x̂0
    return Ŷ0, x̂0end
end

@doc raw"""
    predict!(
        Ŷ0, x̂0end, _ , _ , 
        mpc::PredictiveController, model::NonLinModel, transcription::MultipleShooting,
        U0, Z̃
    ) -> Ŷ0, x̂0end

Compute vectors if `model` is a [`NonLinModel`](@ref) and for [`MultipleShooting`](@ref).
    
The method mutates `Ŷ0` and `x̂0end` arguments.
"""
function predict!(
    Ŷ0, x̂0end, _, _,
    mpc::PredictiveController, model::NonLinModel, ::MultipleShooting,
    U0, Z̃
)
    nu, ny, nd, nx̂, Hp, Hc = model.nu, model.ny, model.nd, mpc.estim.nx̂, mpc.Hp, mpc.Hc
    X̂0 = @views Z̃[(nu*Hc+1):(nu*Hc+nx̂*Hp)] # Z̃ = [ΔU; X̂0; ϵ]
    D̂0 = mpc.D̂0
    local x̂0
    for j=1:Hp
        x̂0 = @views X̂0[(1 +  nx̂*(j-1)):(nx̂*j)]
        d0 = @views D̂0[(1 +  nd*(j-1)):(nd*j)]
        ŷ0 = @views Ŷ0[(1 +  ny*(j-1)):(ny*j)]
        ĥ!(ŷ0, mpc.estim, model, x̂0, d0)
    end
    Ŷ0    .+= mpc.F # F = Ŷs if mpc.estim is an InternalModel, else F = 0.
    x̂0end  .= x̂0
    return Ŷ0, x̂0end
end


"""
    con_nonlinprogeq!(
        geq, X̂0, Û0, mpc::PredictiveController, model::NonLinModel, ::MultipleShooting, U0, Z̃
    )

Nonlinear equality constrains for [`NonLinModel`](@ref) and [`MultipleShooting`](@ref).

The method mutates the `geq`, `X̂0` and `Û0` vectors in argument.
"""
function con_nonlinprogeq!(
    geq, X̂0, Û0, mpc::PredictiveController, model::NonLinModel, ::MultipleShooting, U0, Z̃
)
    nx̂, nu, nd, Hp, Hc = mpc.estim.nx̂, model.nu, model.nd, mpc.Hp, mpc.Hc
    nΔU, nX̂ = nu*Hc, nx̂*Hp
    D̂0 = mpc.D̂0
    X̂0_Z̃ = @views Z̃[(nΔU+1):(nΔU+nX̂)]
    x̂0 = @views mpc.estim.x̂0[1:nx̂]
    d0 = @views mpc.d0[1:nd]
    #TODO: allow parallel for loop or threads? 
    for j=1:Hp
        u0     = @views U0[(1 + nu*(j-1)):(nu*j)]
        û0     = @views Û0[(1 + nu*(j-1)):(nu*j)]
        x̂0next = @views X̂0[(1 + nx̂*(j-1)):(nx̂*j)]
        f̂!(x̂0next, û0, mpc.estim, model, x̂0, u0, d0)
        x̂0next .+= mpc.estim.f̂op .- mpc.estim.x̂op
        x̂0next_Z̃ = @views X̂0_Z̃[(1 + nx̂*(j-1)):(nx̂*j)]       
        ŝnext    = @views  geq[(1 + nx̂*(j-1)):(nx̂*j)]
        ŝnext   .= x̂0next .- x̂0next_Z̃
        x̂0 = x̂0next_Z̃ # using states in Z̃ for next iteration (allow parallel for)
        d0 = @views D̂0[(1 + nd*(j-1)):(nd*j)]
    end
    return geq
end
con_nonlinprogeq!(geq,_,_,::PredictiveController,::NonLinModel,::SingleShooting,  _,_) = geq
con_nonlinprogeq!(geq,_,_,::PredictiveController,::LinModel,::TranscriptionMethod,_,_) = geq