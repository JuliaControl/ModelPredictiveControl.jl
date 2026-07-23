"Get the number of elements in the optimization decision vector `Z`"
get_nZ_mhe(::SingleShooting, He, nx̂, nŵ) = nx̂ + nŵ*He
get_nZ_mhe(::TranscriptionMethod, He, nx̂, nŵ) = nx̂ + nx̂*He + nŵ*He

@doc raw"""
    init_predmat_mhe(
        model::LinModel, transcription::SingleShooting,
        He, i_ym, Â, B̂u, Ĉm, B̂d, D̂dm, x̂op, f̂op, direct
    ) -> E, G, J, B, ex̄, Ex̂, Gx̂, Jx̂, Bx̂

Construct the MHE prediction matrices for [`LinModel`](@ref) and [`SingleShooting`](@ref).

We first introduce the deviation vector of the estimated state at arrival 
``\mathbf{x̂_0}(k-N_k+p) = \mathbf{x̂}_k(k-N_k+p) - \mathbf{x̂_{op}}`` (see [`setop!`](@ref)),
and the vector ``\mathbf{Z} = [\begin{smallmatrix} \mathbf{x̂_0}(k-N_k+p)
\\ \mathbf{Ŵ} \end{smallmatrix}]`` with the decision variables. Setting the constant ``p=0``
produces an estimator in the current form, while the prediction form is obtained with
``p=1``. The estimated sensor noises from time ``k-N_k+1`` to ``k`` are computed by:
```math
\begin{aligned}
    \mathbf{V̂} = \mathbf{Y_0^m - Ŷ_0^m} &= \mathbf{E Z + G U_0 + J D_0 + Y_0^m + B}     \\
                                        &= \mathbf{E Z + F}
\end{aligned}
```
in which ``\mathbf{U_0}`` and ``\mathbf{Y_0^m}`` respectively include the deviation values
of the manipulated inputs ``\mathbf{u_0}(k-j+p)`` from ``j=N_k`` to ``1`` and measured
outputs ``\mathbf{y_0^m}(k-j+1)`` from ``j=N_k`` to ``1``. The vector ``\mathbf{D_0}``
includes the the measured disturbance deviation values ``\mathbf{d_0}(k-j)`` from from
``j=N_k`` to ``0``, thus one additional data point. The constant ``\mathbf{B}`` is the
contribution for non-zero state ``\mathbf{x̂_{op}}`` and state update ``\mathbf{f̂_{op}}``
operating points (for linearization, see [`augment_model`](@ref) and [`linearize`](@ref)).
The method also returns the matrices for the estimation error at arrival:
```math
    \mathbf{x̄} = \mathbf{x̂_0^†}(k-N_k+p) - \mathbf{x̂_0}(k-N_k+p) = \mathbf{e_x̄ Z + f_x̄}
```
in which ``\mathbf{e_x̄} = [\begin{smallmatrix} -\mathbf{I} & \mathbf{0} & \cdots & \mathbf{0} \end{smallmatrix}]``,
and ``\mathbf{f_x̄} = \mathbf{x̂_0^†}(k-N_k+p)``. The latter is the deviation vector of the
state at arrival, estimated at time ``k-N_k``, i.e. ``\mathbf{x̂_0^†}(k-N_k+p) = 
\mathbf{x̂}_{k-N_k}(k-N_k+p) - \mathbf{x̂_{op}}``. Lastly, the estimates ``\mathbf{x̂_0}(k-j+p)``
from ``j=N_k-1`` to ``0``, also in deviation form, are computed with:
```math
\begin{aligned}
    \mathbf{X̂_0}  &= \mathbf{E_x̂ Z + G_x̂ U_0 + J_x̂ D_0 + B_x̂} \\
                  &= \mathbf{E_x̂ Z + F_x̂}
\end{aligned}
```
The matrices ``\mathbf{E, G, J, B, E_x̂, G_x̂, J_x̂, B_x̂}`` are defined in the Extended Help 
section. The vectors ``\mathbf{F, F_x̂, f_x̄}`` are recalculated at each discrete time step, 
see [`initpred!(::MovingHorizonEstimator, ::LinModel)`](@ref) and [`linconstraint!(::MovingHorizonEstimator, ::LinModel)`](@ref).

# Extended Help
!!! details "Extended Help"
    Using the augmented process model matrices ``\mathbf{Â, B̂_u, Ĉ^m, B̂_d, D̂_d^m}``, and the
    function ``\mathbf{S}(j) = ∑_{i=0}^j \mathbf{Â}^i``, the prediction matrices for the
    sensor noises depend on the constant ``p``. For ``p=0``, the matrices are computed by
    (notice the minus signs after the equalities):
    ```math
    \begin{aligned}
    \mathbf{E} &= - \begin{bmatrix}
        \mathbf{Ĉ^m}\mathbf{Â}^{1}                  & \mathbf{Ĉ^m}\mathbf{Â}^{0}                    & \cdots & \mathbf{0}                               \\ 
        \mathbf{Ĉ^m}\mathbf{Â}^{2}                  & \mathbf{Ĉ^m}\mathbf{Â}^{1}                    & \cdots & \mathbf{0}                               \\ 
        \vdots                                      & \vdots                                        & \ddots & \vdots                                   \\
        \mathbf{Ĉ^m}\mathbf{Â}^{H_e}                & \mathbf{Ĉ^m}\mathbf{Â}^{H_e-1}                & \cdots & \mathbf{Ĉ^m}\mathbf{Â}^{0}               \end{bmatrix} \\
    \mathbf{G} &= - \begin{bmatrix}
        \mathbf{Ĉ^m}\mathbf{Â}^{0}\mathbf{B̂_u}      & \mathbf{0}                                    & \cdots & \mathbf{0}                               \\ 
        \mathbf{Ĉ^m}\mathbf{Â}^{1}\mathbf{B̂_u}      & \mathbf{Ĉ^m}\mathbf{Â}^{0}\mathbf{B̂_u}        & \cdots & \mathbf{0}                               \\ 
        \vdots                                      & \vdots                                        & \ddots & \vdots                                   \\
        \mathbf{Ĉ^m}\mathbf{Â}^{H_e-1}\mathbf{B̂_u}  & \mathbf{Ĉ^m}\mathbf{Â}^{H_e-2}\mathbf{B̂_u}    & \cdots & \mathbf{Ĉ^m}\mathbf{Â}^{0}\mathbf{B̂_u}   \end{bmatrix} \\
    \mathbf{J} &= - \begin{bmatrix}
        \mathbf{Ĉ^m}\mathbf{Â}^{0}\mathbf{B̂_d}      & \mathbf{D̂_d^m}                              & \mathbf{0}                                    & \cdots & \mathbf{0}     \\ 
        \mathbf{Ĉ^m}\mathbf{Â}^{1}\mathbf{B̂_d}      & \mathbf{Ĉ^m}\mathbf{Â}^{0}\mathbf{B̂_d}      & \mathbf{D̂_d^m}                                & \cdots & \mathbf{0}     \\ 
        \vdots                                      & \vdots                                      & \vdots                                        & \ddots & \vdots         \\
        \mathbf{Ĉ^m}\mathbf{Â}^{H_e-1}\mathbf{B̂_d}  & \mathbf{Ĉ^m}\mathbf{Â}^{H_e-2}\mathbf{B̂_d}  & \mathbf{Ĉ^m}\mathbf{Â}^{H_e-3}\mathbf{B̂_d}    & \cdots & \mathbf{D̂_d^m} \end{bmatrix} \\
    \mathbf{B} &= - \begin{bmatrix}
        \mathbf{Ĉ^m S}(0)                    \\
        \mathbf{Ĉ^m S}(1)                    \\
        \vdots                               \\
        \mathbf{Ĉ^m S}(H_e-1) \end{bmatrix}  \mathbf{\big(f̂_{op} - x̂_{op}\big)}
    \end{aligned}
    ```
    or, for ``p=1``, the matrices are given by:
    ```math
    \begin{aligned}
    \mathbf{E} &= - \begin{bmatrix}
        \mathbf{Ĉ^m}\mathbf{Â}^{0}                  & \mathbf{0}                                    & \cdots & \mathbf{0}   \\ 
        \mathbf{Ĉ^m}\mathbf{Â}^{1}                  & \mathbf{Ĉ^m}\mathbf{Â}^{0}                    & \cdots & \mathbf{0}   \\ 
        \vdots                                      & \vdots                                        & \ddots & \vdots       \\
        \mathbf{Ĉ^m}\mathbf{Â}^{H_e-1}              & \mathbf{Ĉ^m}\mathbf{Â}^{H_e-2}                & \cdots & \mathbf{0}   \end{bmatrix} \\
    \mathbf{G} &= - \begin{bmatrix}
        \mathbf{0}                                  & \mathbf{0}                                    & \cdots & \mathbf{0}   \\ 
        \mathbf{Ĉ^m}\mathbf{Â}^{0}\mathbf{B̂_u}      & \mathbf{0}                                    & \cdots & \mathbf{0}   \\ 
        \vdots                                      & \vdots                                        & \ddots & \vdots       \\
        \mathbf{Ĉ^m}\mathbf{Â}^{H_e-2}\mathbf{B̂_u}  & \mathbf{Ĉ^m}\mathbf{Â}^{H_e-3}\mathbf{B̂_u}    & \cdots & \mathbf{0}   \end{bmatrix} \\
    \mathbf{J} &= - \begin{bmatrix}
        \mathbf{0}  & \mathbf{D̂_d^m}                              & \mathbf{0}                                    & \cdots & \mathbf{0}     \\ 
        \mathbf{0}  & \mathbf{Ĉ^m}\mathbf{Â}^{0}\mathbf{B̂_d}      & \mathbf{D̂_d^m}                                & \cdots & \mathbf{0}     \\ 
        \vdots      & \vdots                                      & \vdots                                        & \ddots & \vdots         \\
        \mathbf{0}  & \mathbf{Ĉ^m}\mathbf{Â}^{H_e-2}\mathbf{B̂_d}  & \mathbf{Ĉ^m}\mathbf{Â}^{H_e-3}\mathbf{B̂_d}    & \cdots & \mathbf{D̂_d^m} \end{bmatrix} \\
    \mathbf{B} &= - \begin{bmatrix}
        \mathbf{0}                           \\  
        \mathbf{Ĉ^m S}(0)                    \\
        \vdots                               \\
        \mathbf{Ĉ^m S}(H_e-2) \end{bmatrix}  \mathbf{\big(f̂_{op} - x̂_{op}\big)}
    \end{aligned}
    ```
    The matrices for the estimated states are computed by:
    ```math
    \begin{aligned}
    \mathbf{E_x̂} &= \begin{bmatrix}
        \mathbf{Â}^{1}                      & \mathbf{A}^{0}                    & \cdots & \mathbf{0}                   \\
        \mathbf{Â}^{2}                      & \mathbf{Â}^{1}                    & \cdots & \mathbf{0}                   \\ 
        \vdots                              & \vdots                            & \ddots & \vdots                       \\
        \mathbf{Â}^{H_e}                    & \mathbf{Â}^{H_e-1}                & \cdots & \mathbf{Â}^{0}               \end{bmatrix} \\
    \mathbf{G_x̂} &= \begin{bmatrix}
        \mathbf{Â}^{0}\mathbf{B̂_u}          & \mathbf{0}                        & \cdots & \mathbf{0}                   \\ 
        \mathbf{Â}^{1}\mathbf{B̂_u}          & \mathbf{Â}^{0}\mathbf{B̂_u}        & \cdots & \mathbf{0}                   \\ 
        \vdots                              & \vdots                            & \ddots & \vdots                       \\
        \mathbf{Â}^{H_e-1}\mathbf{B̂_u}      & \mathbf{Â}^{H_e-2}\mathbf{B̂_u}    & \cdots & \mathbf{Â}^{0}\mathbf{B̂_u}   \end{bmatrix} \\
    \mathbf{J_x̂^†} &= \begin{bmatrix}
        \mathbf{Â}^{0}\mathbf{B̂_d}          & \mathbf{0}                        & \cdots & \mathbf{0}                   \\ 
        \mathbf{Â}^{1}\mathbf{B̂_d}          & \mathbf{Â}^{0}\mathbf{B̂_d}        & \cdots & \mathbf{0}                   \\ 
        \vdots                              & \vdots                            & \ddots & \vdots                       \\
        \mathbf{Â}^{H_e-1}\mathbf{B̂_d}      & \mathbf{Â}^{H_e-2}\mathbf{B̂_d}    & \cdots & \mathbf{Â}^{0}\mathbf{B̂_d}   \end{bmatrix} \ , \quad
    \mathbf{J_x̂} = \begin{cases}
        [\begin{smallmatrix} \mathbf{J_x̂^†} & \mathbf{0}      \end{smallmatrix}]   & p=0                                \\
        [\begin{smallmatrix} \mathbf{0}     & \mathbf{J_x̂^†}  \end{smallmatrix}]   & p=1                                \end{cases}   \\
    \mathbf{B_x̂} &= \begin{bmatrix}
        \mathbf{S}(0)                    \\
        \mathbf{S}(1)                    \\
        \vdots                           \\
        \mathbf{S}(H_e-1) \end{bmatrix}  \mathbf{\big(f̂_{op} - x̂_{op}\big)}
    \end{aligned}
    ```
    All these matrices are truncated when ``N_k < H_e`` (at the beginning).
"""
function init_predmat_mhe(
    model::LinModel{NT}, ::SingleShooting, He, i_ym, Â, B̂u, Ĉm, B̂d, D̂dm, x̂op, f̂op, direct
) where {NT<:Real}
    nu, nd = model.nu, model.nd
    nym, nx̂ = length(i_ym), size(Â, 2)
    nŵ = nx̂
    p = direct ? 0 : 1
    # --- pre-compute matrix powers ---
    # Apow3D array : Apow[:,:,1] = A^0, Apow[:,:,2] = A^1, ... , Apow[:,:,He+1] = A^He
    Âpow3D = Array{NT}(undef, nx̂, nx̂, He+1)
    Âpow3D[:,:,1] = I(nx̂)
    for j=2:He+1
        Âpow3D[:,:,j] = @views Âpow3D[:,:,j-1]*Â
    end
    # nĈm_Âpow3D array : similar indices as Apow3D
    nĈm_Âpow3D = Array{NT}(undef, nym, nx̂, He+1)
    nĈm_Âpow3D[:,:,1] = -Ĉm
    for j=2:He+1
        nĈm_Âpow3D[:,:,j] = @views -Ĉm*Âpow3D[:,:,j]
    end
    # helper function to improve code clarity and be similar to eqs. in docstring:
    getpower(array3D, power) = @views array3D[:,:, power+1]
    # --- decision variables Z ---
    nĈm_Âpow = reduce(vcat, getpower(nĈm_Âpow3D, i) for i=0:He)
    E = zeros(NT, nym*He, nx̂ + nŵ*He)
    col_begin = iszero(p) ? 1    : 0
    col_end   = iszero(p) ? He : He-1
    i = 0
    for j=col_begin:col_end
        iRow = (1 + i*nym):(nym*He)
        iCol = (1:nŵ) .+ j*nŵ
        E[iRow, iCol] = @views nĈm_Âpow[1:length(iRow) ,:]
        i += 1
    end
    iszero(p) && @views (E[:, 1:nx̂] = @views nĈm_Âpow[nym+1:end, :])
    ex̄ = [-I zeros(NT, nx̂, nŵ*He)]
    Âpow_vec = reduce(vcat, getpower(Âpow3D, i) for i=0:He)
    Ex̂ = zeros(NT, nx̂*He, nx̂ + nŵ*He)
    i=0
    for j=1:He
        iRow = (1 + i*nx̂):(nx̂*He)
        iCol = (1:nŵ) .+ j*nŵ
        Ex̂[iRow, iCol] = @views Âpow_vec[1:length(iRow) ,:]
        i+=1
    end
    Ex̂[:, 1:nx̂] = @views Âpow_vec[nx̂+1:end, :] 
    # --- manipulated inputs U ---
    nĈm_Âpow_B̂u = reduce(vcat, getpower(nĈm_Âpow3D, i)*B̂u for i=0:He-1)
    nĈm_Âpow_B̂u = [zeros(nym, nu) ; nĈm_Âpow_B̂u]
    G = zeros(NT, nym*He, nu*He)
    i=0
    col_begin = iszero(p) ? 1    : 0
    col_end   = iszero(p) ? He-1 : He-2
    for j=col_begin:col_end
        iRow = (1 + i*nym):(nym*He)
        iCol = (1:nu) .+ j*nu
        G[iRow, iCol] = @views nĈm_Âpow_B̂u[1:length(iRow) ,:]
        i+=1
    end
    iszero(p) && @views (G[:, 1:nu] = nĈm_Âpow_B̂u[nym+1:end, :])
    Âpow_B̂u = reduce(vcat, getpower(Âpow3D, i)*B̂u for i=0:He-1)
    Gx̂ = zeros(NT, nx̂*He, nu*He)
    for j=0:He-1
        iRow = (1 + j*nx̂):(nx̂*He)
        iCol = (1:nu) .+ j*nu
        Gx̂[iRow, iCol] = @views Âpow_B̂u[1:length(iRow) ,:]
    end
    # --- measured disturbances D ---
    nĈm_Âpow_B̂d = reduce(vcat, getpower(nĈm_Âpow3D, i)*B̂d for i=0:He-1)
    nĈm_Âpow_B̂d = [-D̂dm; nĈm_Âpow_B̂d]
    J = zeros(NT, nym*He, nd*(He+1))
    i = 0
    for j=1:He
        iRow = (1 + i*nym):(nym*He)
        iCol = (1:nd) .+ j*nd
        J[iRow, iCol] = nĈm_Âpow_B̂d[1:length(iRow) ,:]
        i+=1
    end
    iszero(p) && @views (J[:, 1:nd] = nĈm_Âpow_B̂d[nym+1:end, :])
    Âpow_B̂d = reduce(vcat, getpower(Âpow3D, i)*B̂d for i=0:He-1)
    Jx̂ = zeros(NT, nx̂*He, nd*(He+1))
    for j=0:He-1
        iRow = (1 + j*nx̂):(nx̂*He)
        iCol = (1:nd) .+ j*nd .+ p
        Jx̂[iRow, iCol] = Âpow_B̂d[1:length(iRow) ,:]
    end
    # --- state x̂op and state update f̂op operating points ---
    # Apow_csum 3D array : Apow_csum[:,:,1] = A^0, Apow_csum[:,:,2] = A^1 + A^0, ...
    Âpow_csum  = cumsum(Âpow3D, dims=3)
    # helper function to improve code clarity and be similar to eqs. in docstring:
    S(j) = @views Âpow_csum[:,:, j+1]
    f̂_op_n_x̂op = (f̂op - x̂op)
    coef_B  = zeros(NT, nym*He, nx̂)
    row_begin = iszero(p) ? 0    : 1
    row_end   = iszero(p) ? He-1 : He-2
    j=0
    for i=row_begin:row_end
        iRow = (1:nym) .+ nym*i
        coef_B[iRow,:] = -Ĉm*S(j)
        j+=1
    end
    B = coef_B*f̂_op_n_x̂op
    coef_Bx̂ = Matrix{NT}(undef, nx̂*He, nx̂)
    for j=0:He-1
        iRow = (1:nx̂)  .+ nx̂*j
        coef_Bx̂[iRow,:] = S(j)
    end
    Bx̂ = coef_Bx̂*f̂_op_n_x̂op
    return E, G, J, B, ex̄, Ex̂, Gx̂, Jx̂, Bx̂
end

"""
    init_predmat_mhe(
        model::LinModel, transcription::MultipleShooting, 
        He, i_ym, Â, B̂u, Ĉm, B̂d, D̂dm, x̂op, f̂op, direct
    ) -> E, G, J, B, ex̄, Ex̂, Gx̂, Jx̂, Bx̂

TBW
"""
function init_predmat_mhe(
    model::LinModel{NT}, ::MultipleShooting, 
    He, i_ym, Â, B̂u, Ĉm, B̂d, D̂dm, x̂op, f̂op, direct
) where {NT<:Real}
    nu, nd = model.nu, model.nd
    nym, nx̂ = length(i_ym), size(Â, 2)
    nŵ = nx̂
    p = direct ? 0 : 1
    nX̂, nŴ, nV̂, nU, nD = nx̂*He, nŵ*He, nym*He, nu*He, nd*(He+1)
    # --- decision variables Z ---
    E  = [zeros(NT, nV̂, (1-p)*nx̂) repeatdiag(-Ĉm, He) zeros(NT, nV̂, p*nx̂ + nŴ)]
    ex̄ = [-I zeros(NT, nx̂, nX̂ + nŴ)]
    Ex̂ = [zeros(NT, nX̂, nx̂) I zeros(NT, nX̂, nŴ)]
    # --- manipulated inputs U ---
    G  = zeros(NT, nV̂, nU)
    Gx̂ = zeros(NT, nX̂, nU)
    # --- measured disturbances D ---
    J  = [zeros(NT, nV̂, nd) repeatdiag(-D̂dm, He)]
    Jx̂ = zeros(NT, nX̂, nD)
    # --- state x̂op and state update f̂op operating points ---
    B  = zeros(NT, nV̂)
    Bx̂ = zeros(NT, nX̂)
    return E, G, J, B, ex̄, Ex̂, Gx̂, Jx̂, Bx̂
end

"""
    init_predmat_mhe(
        model::SimModel, ::SingleShooting, He, i_ym, Â, _ , _ , _ , _ , _ , _ , _
    ) -> E, G, J, B, ex̄, Ex̂, Gx̂, Jx̂, Bx̂

Return empty matrices if `model` is not a [`LinModel`](@ref), except for `ex̄`.
"""
function init_predmat_mhe(
    model::SimModel{NT}, ::SingleShooting, He, i_ym, Â, _ , _ , _ , _ , _ , _ , _
) where {NT<:Real}
    nym, nx̂ = length(i_ym), size(Â, 2)
    nŵ = nx̂
    E  = zeros(NT, 0, nx̂ + nŵ*He)
    ex̄ = [-I zeros(NT, nx̂, nŵ*He)]
    Ex̂ = zeros(NT, 0, nx̂ + nŵ*He)
    G  = zeros(NT, 0, model.nu*He)
    Gx̂ = zeros(NT, 0, model.nu*He)
    J  = zeros(NT, 0, model.nd*(He+1))
    Jx̂ = zeros(NT, 0, model.nd*(He+1))
    B  = zeros(NT, nym*He)
    Bx̂ = zeros(NT, nx̂*He)
    return E, G, J, B, ex̄, Ex̂, Gx̂, Jx̂, Bx̂
end

"""
    init_defectmat_mhe(
        model::LinModel, transcription::MultipleShooting, 
        He, i_ym, Â, B̂u, Ĉm, B̂d, D̂dm, x̂op, f̂op, direct
    ) -> ES, GS, JS, BS

TBW
"""
function init_defectmat_mhe(
    model::LinModel{NT}, ::MultipleShooting, He, Â, B̂u, B̂d, x̂op, f̂op, direct
) where {NT<:Real}
    nd = model.nd
    nx̂ = size(Â, 2)
    nŵ = nx̂
    nX̂ = nx̂*He
    p = direct ? 0 : 1
    # --- decision variables Z ---
    nI_nx̂ = Matrix{NT}(-I, nx̂, nx̂)
    I_nx̂  = Matrix{NT}(I, nŵ, nŵ)
    ES = [zeros(NT, nX̂, nx̂) repeatdiag(nI_nx̂, He) repeatdiag(I_nx̂, He)]
    for j=1:He
        iRowCol = (1:nx̂) .+ nx̂*(j-1)
        ES[iRowCol, iRowCol] = Â
    end
    # --- manipulated inputs U ---
    GS = repeatdiag(B̂u, He)
    # --- measured disturbances D ---
    JS = [zeros(NT, nX̂, p*nd) repeatdiag(B̂d, He) zeros(NT, nX̂, (1-p)*nd)]
    # --- state x̂op and state update f̂op operating points ---
    BS = repeat(f̂op - x̂op, He)
    return ES, GS, JS, BS
end

"Return empty matrices for [`SingleShooting`](@ref) transcription (N/A)."
function init_defectmat_mhe(
    model::SimModel{NT}, ::SingleShooting, He, Â, _ , _ , _ , _ , _
) where {NT<:Real}
    nu, nd = model.nu, model.nd
    nx̂ = size(Â, 2)
    nŵ = nx̂
    ES = zeros(NT, 0, nx̂ + nŵ*He)
    GS = zeros(NT, 0, nu*He)
    JS = zeros(NT, 0, nd*(He+1))
    BS = zeros(NT, 0)
    return ES, GS, JS, BS
end

@doc raw"""
    init_matconstraint_mhe(
        model::LinModel, transcription::SingleShooting, Z̃min, Z̃max, nc,
        x̂0min, x̂0max, X̂0min, X̂0max, Ŵmin, Ŵmax, V̂min, V̂max, args...
    ) -> i_b, i_g, A, Aeq, neq

Init `i_b`, `i_g`, `neq`, and `A` and `Aeq` matrices for all the MHE constraints.

The linear and nonlinear inequality constraints are respectively defined as:
```math
\begin{aligned} 
    \mathbf{A Z̃ }       &≤ \mathbf{b}           \\ 
    \mathbf{A_{eq} Z̃}   &= \mathbf{b_{eq}}      \\
    \mathbf{g(Z̃)}       &≤ \mathbf{0}           \\
    \mathbf{g_{eq}(Z̃)}  &= \mathbf{0}           
\end{aligned}
```
The argument `nc` is the number of custom nonlinear inequality constraints in
``\mathbf{g_c}``. `i_b` is a `BitVector` including the indices of ``\mathbf{b}`` that are
finite numbers. `i_g` is a similar vector but for the indices of ``\mathbf{g}``. The method
also returns the `\mathbf{A, A_{eq}}`` matrices and `neq` if `args` is provided. In such a
case, `args`  needs to contain all the inequality and equality constraint matrices: 
`A_x̂min, A_x̂max, A_X̂min, A_X̂max, A_Ŵmin, A_Ŵmax, A_V̂min, A_V̂max, Aeq`. The integer `neq` is
the number of nonlinear equality constraints in ``\mathbf{g_{eq}}``.
"""
function init_matconstraint_mhe(
    model::LinModel{NT}, transcription::TranscriptionMethod, Z̃min, Z̃max, nc,
    x̂0min, x̂0max, X̂0min, X̂0max, Ŵmin, Ŵmax, V̂min, V̂max, args...
) where {NT<:Real}
    if isempty(args)
        A, Aeq, neq = nothing, nothing, nothing
    else
        A_x̂min, A_x̂max, A_X̂min, A_X̂max, A_Ŵmin, A_Ŵmax, A_V̂min, A_V̂max, Aeq = args
        A = [A_x̂min; A_x̂max; A_X̂min; A_X̂max; A_Ŵmin; A_Ŵmax; A_V̂min; A_V̂max]
        neq = 0 # number of nonlinear equality constraints
    end
    i_x̂min, i_x̂max  = @. !isinf(x̂0min), !isinf(x̂0max)
    i_X̂min, i_X̂max  = @. !isinf(X̂0min), !isinf(X̂0max)
    i_Ŵmin, i_Ŵmax  = @. !isinf(Ŵmin),  !isinf(Ŵmax)
    i_V̂min, i_V̂max  = @. !isinf(V̂min),  !isinf(V̂max)
    nx̂ = length(x̂0min)
    nε = length(Z̃min) - length(Ŵmin) - nx̂
    deletex̂arr_lincon!(i_x̂min, i_x̂max, model, Z̃min, Z̃max, nε)
    deleteŴ_lincon!(i_Ŵmin, i_Ŵmax, model, Z̃min, Z̃max, nx̂, nε)
    i_b = [i_x̂min; i_x̂max; i_X̂min; i_X̂max; i_Ŵmin; i_Ŵmax; i_V̂min; i_V̂max]
    i_g = trues(nc)
    return i_b, i_g, A, Aeq, neq
end

"Init `i_b, A` without state and sensor noise constraints if `model` is not a [`LinModel`](@ref)."
function init_matconstraint_mhe(
    model::NonLinModel{NT}, transcription::SingleShooting, Z̃min, Z̃max, nc,
    x̂0min, x̂0max, X̂0min, X̂0max, Ŵmin, Ŵmax, V̂min, V̂max, args...
) where {NT<:Real}
    if isempty(args)
        A, Aeq, neq = nothing, nothing, nothing
    else
        A_x̂min, A_x̂max, _ , _ , A_Ŵmin, A_Ŵmax, _ , _ , Aeq = args
        A = [A_x̂min; A_x̂max; A_Ŵmin; A_Ŵmax]
        neq = 0 # number of nonlinear equality constraints
    end
    i_x̂min, i_x̂max  = @. !isinf(x̂0min), !isinf(x̂0max)
    i_X̂min, i_X̂max  = @. !isinf(X̂0min), !isinf(X̂0max)
    i_Ŵmin, i_Ŵmax  = @. !isinf(Ŵmin),  !isinf(Ŵmax)
    i_V̂min, i_V̂max  = @. !isinf(V̂min),  !isinf(V̂max)
    nx̂ = length(x̂0min)
    nε = length(Z̃min) - length(Ŵmin) - nx̂
    deletex̂arr_lincon!(i_x̂min, i_x̂max, model, Z̃min, Z̃max, nε)
    deleteŴ_lincon!(i_Ŵmin, i_Ŵmax, model, Z̃min, Z̃max, nx̂, nε)
    i_b = [i_x̂min; i_x̂max; i_Ŵmin; i_Ŵmax]
    i_g = [i_X̂min; i_X̂max; i_V̂min; i_V̂max; trues(nc)]
    return i_b, i_g, i_g, A, Aeq, neq
end

"For [`SingleShooting`](@ref), truncate the end of prediction matrices if `Nk < He`"
function trunc_predmat(estim::MovingHorizonEstimator, ::SingleShooting)
    model, F = estim.model, estim.F
    nx̂, nŵ, nym, nε, Nk = estim.nx̂, estim.nx̂, estim.nym, estim.nε, estim.Nk[]
    nU, nYm, nŴ, nD = model.nu*Nk, nym*Nk, nŵ*Nk, model.nd*(Nk+1)
    nZ = get_nZ_mhe(transcription, Nk, nx̂, nŵ)
    nZ̃ = nε + nZ
    if Nk < estim.He # avoid views since allocations only when Nk < He and we want fast mul!
        Ẽ       = estim.Ẽ[1:nYm, 1:nZ̃]
        G, J, B = estim.G[1:nYm, 1:nU], estim.J[1:nYm, 1:nD], estim.B[1:nYm]
        ẽx̄      = estim.ẽx̄[:, 1:nZ̃]
        Tŵ      = estim.Tŵ[1:nŴ, 1:nZ]
        F       = @views estim.F[1:nYm] # views here since they will store results
        H̃_data  = @views estim.H̃.data[1:nZ̃, 1:nZ̃]
        H̃       = @views estim.H̃[1:nZ̃, 1:nZ̃]
        q̃       = @views estim.q̃[1:nZ̃]
        Z̃var    = @views estim.optim[:Z̃var][1:nZ̃]
    else
        Ẽ, F, G, J, B = estim.Ẽ, estim.F, estim.G, estim.J, estim.B
        ẽx̄, Tŵ        = estim.ẽx̄, estim.Tŵ
        H̃, H̃_data, q̃  = estim.H̃, estim.H̃.data, estim.q̃
        Z̃var          = estim.optim[:Z̃var]
    end
    return Ẽ, F, G, J, B, ẽx̄, Tŵ, H̃, H̃_data, q̃, Z̃var
end

"For [`MultipleShooting`](@ref), extract subparts of the prediction matrices if `Nk < He`."
function trunc_predmat(estim::MovingHorizonEstimator, ::MultipleShooting)
    model, F = estim.model, estim.F
    nx̂, nŵ, nym, nε, Nk = estim.nx̂, estim.nx̂, estim.nym, estim.nε, estim.Nk[]
    nU, nYm, nŴ, nD = model.nu*Nk, nym*Nk, nŵ*Nk, model.nd*(Nk+1)
    nx̂_nX̂    = nx̂ + nx̂*Nk 
    nx̂_nX̂_He = nx̂ + nx̂*estim.He
    if Nk < estim.He # avoid views since allocations only when Nk < He and we want fast mul!
        i_Z̃_He  = [(1):(nε + nx̂_nX̂); (nε + nx̂_nX̂_He + 1):(nε + nx̂_nX̂_He + nŴ)]
        i_Z_He  = [(1):(nx̂_nX̂); (nx̂_nX̂_He + 1):(nx̂_nX̂_He + nŴ)]
        Ẽ       = estim.Ẽ[1:nYm, i_Z̃_He]
        G, J, B = estim.G[1:nYm, 1:nU], estim.J[1:nYm, 1:nD], estim.B[1:nYm]
        ẽx̄      = estim.ẽx̄[:, i_Z̃_He]
        Tŵ      = estim.Tŵ[1:nŴ, i_Z_He]
        F       = @views estim.F[1:nYm] # views here since they will store results
        H̃_data  = @views estim.H̃.data[i_Z̃_He, i_Z̃_He]
        H̃       = @views estim.H̃[i_Z̃_He, i_Z̃_He]
        q̃       = @views estim.q̃[i_Z̃_He]
        Z̃var    = @views estim.optim[:Z̃var][i_Z̃_He]
    else
        Ẽ, F, G, J, B = estim.Ẽ, estim.F, estim.G, estim.J, estim.B
        ẽx̄, Tŵ        = estim.ẽx̄, estim.Tŵ
        H̃, H̃_data, q̃  = estim.H̃, estim.H̃.data, estim.q̃
        Z̃var          = estim.optim[:Z̃var]
    end
    return Ẽ, F, G, J, B, ẽx̄, Tŵ, H̃, H̃_data, q̃, Z̃var

end

@doc raw"""
    linconstraint!(
        estim::MovingHorizonEstimator, model::LinModel, transcription::TranscriptionMethod
    )

Set `b` vector for the linear model inequality constraints (``\mathbf{A Z̃ ≤ b}``) of MHE.

Also init ``\mathbf{F_x̂ = G_x̂ U_0 + J_x̂ D_0 + B_x̂}`` vector for the state constraints, see 
[`init_predmat_mhe`](@ref).
"""
function linconstraint!(
    estim::MovingHorizonEstimator, model::LinModel, ::TranscriptionMethod
)
    nx̂, nŵ, nym, Nk = estim.nx̂, estim.nx̂, estim.nym, estim.Nk[]
    nU, nX̂, nD = model.nu*Nk, estim.nx̂*Nk, model.nd*(Nk+1)
    # --- truncate vector and matrices if necessary ---
    if Nk < estim.He
        # avoid views since allocations only when Nk < He and we want fast mul!:
        Bx̂     = estim.con.Bx̂[1:nX̂]
        Gx̂, U0 = estim.con.Gx̂[1:nX̂, 1:nU], estim.U0[1:nU]
        Jx̂, D0 = estim.con.Jx̂[1:nX̂, 1:nD], estim.D0[1:nD]
        Fx̂     = @views estim.con.Fx̂[1:nX̂]
    else
        Bx̂     = estim.con.Bx̂
        Gx̂, U0 = estim.con.Gx̂, estim.U0
        Jx̂, D0 = estim.con.Jx̂, estim.D0
        Fx̂     = estim.con.Fx̂
    end
    X̂0min, X̂0max = trunc_bounds(estim, estim.con.X̂0min, estim.con.X̂0max, nx̂)
    Ŵmin, Ŵmax   = trunc_bounds(estim, estim.con.Ŵmin,  estim.con.Ŵmax,  nŵ)
    V̂min, V̂max   = trunc_bounds(estim, estim.con.V̂min,  estim.con.V̂max,  nym)
    # --- update Fx̂ vectors for MHE state constraints ---
    Fx̂ .= Bx̂
    mul!(Fx̂, Gx̂, U0, 1, 1)
    model.nd > 0 && mul!(Fx̂, Jx̂, D0, 1, 1)
    # --- update b vector for linear inequality constraints ---
    nX̂_He, nŴ_He, nV̂_He = length(X̂0min), length(Ŵmin), length(V̂min)
    nx̂ = length(estim.con.x̂0min)
    n = 0
    estim.con.b[(n+1):(n+nx̂)] .= @. -estim.con.x̂0min
    n += nx̂
    estim.con.b[(n+1):(n+nx̂)] .= @. +estim.con.x̂0max
    n += nx̂
    estim.con.b[(n+1):(n+nX̂_He)] .= @. -X̂0min + estim.con.Fx̂
    n += nX̂_He
    estim.con.b[(n+1):(n+nX̂_He)] .= @. +X̂0max - estim.con.Fx̂
    n += nX̂_He
    estim.con.b[(n+1):(n+nŴ_He)] .= @. -Ŵmin
    n += nŴ_He
    estim.con.b[(n+1):(n+nŴ_He)] .= @. +Ŵmax
    n += nŴ_He
    estim.con.b[(n+1):(n+nV̂_He)] .= @. -V̂min + estim.F
    n += nV̂_He
    estim.con.b[(n+1):(n+nV̂_He)] .= @. +V̂max - estim.F
    if any(estim.con.i_b) 
        lincon = estim.optim[:linconstraint]
        JuMP.set_normalized_rhs(lincon, estim.con.b[estim.con.i_b])
    end
    return nothing
end

"Set `b` excluding state and sensor noise bounds if `model` is not a [`LinModel`](@ref)."
function linconstraint!(
    estim::MovingHorizonEstimator, ::SimModel, ::TranscriptionMethod
)
    # --- truncate vector and matrices if necessary ---
    Ŵmin, Ŵmax = trunc_bounds(estim, estim.con.Ŵmin, estim.con.Ŵmax, estim.nx̂)
    # --- update b vector for linear inequality constraints ---
    nx̂, nŴ_He = length(estim.con.x̂0min), length(Ŵmin)
    n = 0
    estim.con.b[(n+1):(n+nx̂)] .= @. -estim.con.x̂0min
    n += nx̂
    estim.con.b[(n+1):(n+nx̂)] .= @. +estim.con.x̂0max
    n += nx̂
    estim.con.b[(n+1):(n+nŴ_He)] .= @. -Ŵmin
    n += nŴ_He
    estim.con.b[(n+1):(n+nŴ_He)] .= @. +Ŵmax
    if any(estim.con.i_b) 
        lincon = estim.optim[:linconstraint]
        JuMP.set_normalized_rhs(lincon, estim.con.b[estim.con.i_b])
    end
    return nothing
end

"""
    linconstrainteq!(
        estim::MovingHorizonEstimator, model::LinModel, ::TranscriptionMethod
    )

TBW
"""
function linconstrainteq!(
    estim::MovingHorizonEstimator, model::LinModel, ::TranscriptionMethod
)
    return nothing
end


@doc raw"""
    set_warmstart_mhe!(
        estim::MovingHorizonEstimator, transcription::SingleShooting, Z̃var
    ) -> Z̃s

Set and return the warm-start value of `Z̃var` for [`MovingHorizonEstimator`](@ref).

If supported by `estim.optim` and based a [`SingleShooting`](@ref) transcription, it
warm-starts the solver at:
```math
\mathbf{Z̃_s} = 
\begin{bmatrix}
    ε_{k-1}                         \\
    \mathbf{x̂_0^†}(k-N_k+p)         \\
    \mathbf{ŵ}(k-N_k+p+0|k-1)       \\
    \mathbf{ŵ}(k-N_k+p+1|k-1)       \\
    \vdots                          \\
    \mathbf{ŵ}(k+p-3|k-1)           \\
    \mathbf{ŵ}(k+p-2|k-1)           \\
    \mathbf{0}                      \\
\end{bmatrix}
```
where ``ε_{k-1}`` and ``\mathbf{ŵ}(k-j|k-1)`` are respectively the slack variable and the
process noise estimates computed at the last time step ``k-1``. The vector 
``\mathbf{x̂_0^†}(k-N_k+p)`` is the deviation vector of the state at the arrival estimated
at time ``k-N_k``. If the objective function is not finite at this point, all the process
noises ``\mathbf{ŵ}_{k-1}(k-j)`` are warm-started at zeros. The method mutates all the
arguments.
"""
function set_warmstart_mhe!(
    estim::MovingHorizonEstimator{NT}, ::SingleShooting, Z̃var
) where NT<:Real
    model, buffer = estim.model, estim.buffer
    nε, nx̂, nŵ, Nk = estim.nε, estim.nx̂, estim.nx̂, estim.Nk[]
    nx̃ = nε + nx̂
    Z̃s = estim.buffer.Z̃
    û0, ŷ0, x̄, k = buffer.û, buffer.ŷ, buffer.x̂, buffer.k
    # --- slack variable ε ---
    estim.nε == 1 && (Z̃s[begin] = estim.Z̃[begin])
    # --- arrival state estimate x̂0arr ---
    Z̃s[nε+1:nx̃] = estim.x̂0arr_old
    # --- process noise estimates Ŵ ---
    Z̃s[nx̃+1:end] = estim.Ŵ
    # --- verify definiteness of objective function ---
    V̂, X̂0 = estim.buffer.V̂, estim.buffer.X̂
    x̄ .= 0 # x̂0arr == x̂arr_old implies the error at arrival x̄ is zero
    predict_mhe!(V̂, X̂0, û0, k, ŷ0, estim, model, estim.x̂0arr_old, estim.Ŵ, Z̃s)
    Js = obj_nonlinprog(estim, model, x̄, V̂, estim.Ŵ, Z̃s)
    if !isfinite(Js)
        Z̃s[nx̃+1:end] .= 0
    end
    # --- unused variable in Z̃ (applied only when Nk < He) ---
    # We force the update of the NLP gradient and jacobian by warm-starting the unused 
    # variable in Z̃ at 1. Since estim.Ŵ is initialized with 0s, at least 1 variable in Z̃s
    # will be inevitably different at the following time step.
    Z̃s[nx̃+Nk*nŵ+1:end] .= 1
    JuMP.set_start_value.(Z̃var, Z̃s)
    return Z̃s
end

@doc raw"""
    set_warmstart_mhe!(
        estim::MovingHorizonEstimator, transcription::MultipleShooting, Z̃var
    ) -> Z̃s

Do the same but based on a [`MultipleShooting`](@ref) transcription.

If supported by `estim.optim`, it warm-starts the solver at:
```math
\mathbf{Z̃_s} = 
\begin{bmatrix}
    ε_{k-1}                         \\
    \mathbf{x̂_0^†}(k-N_k+p)         \\
    \mathbf{x̂_0}(k-N_k+p+1|k-1)     \\
    \mathbf{x̂_0}(k-N_k+p+2|k-1)     \\
    \vdots                          \\
    \mathbf{x̂_0}(k+p-2|k-1)         \\
    \mathbf{x̂_0}(k+p-1|k-1)         \\
    \mathbf{x̂_0}(k+p-1|k-1)         \\
    \mathbf{ŵ}(k-N_k+p+0|k-1)       \\
    \mathbf{ŵ}(k-N_k+p+1|k-1)       \\
    \vdots                          \\
    \mathbf{ŵ}(k+p-3|k-1)           \\
    \mathbf{ŵ}(k+p-2|k-1)           \\
    \mathbf{0}                      \\
\end{bmatrix}
```
where ``\mathbf{x̂_0}(k-j|k-1)`` is the predicted state for time ``k-j`` computed at the
last control period ``k-1``, expressed as a deviation from the operating point 
``\mathbf{x̂_{op}}``. 
"""
function set_warmstart_mhe!(
    estim::MovingHorizonEstimator{NT}, ::MultipleShooting, Z̃var
) where NT<:Real
    model, buffer = estim.model, estim.buffer
    nε, nx̂, nŵ, Nk = estim.nε, estim.nx̂, estim.nx̂, estim.Nk[]
    nx̃, nX̂ = nε + nx̂, nx̂*estim.He
    Z̃s = estim.buffer.Z̃
    û0, ŷ0, x̄, k = buffer.û, buffer.ŷ, buffer.x̂, buffer.k
    # --- slack variable ε ---
    estim.nε == 1 && (Z̃s[begin] = estim.Z̃[begin])
    # --- arrival state estimate x̂0arr ---
    Z̃s[nε+1:nx̃] = estim.x̂0arr_old
    # --- state estimates X̂0 --- # mpc.Z̃[(nΔU+nx̂+1):(nΔU+nX̂)]
    Z̃s[(nx̃+1):(nx̃+nX̂-nx̂)]    .= @views estim.Z̃[(nx̃+nx̂+1):(nx̃+nX̂)]
    Z̃s[(nx̃+nX̂-nx̂+1):(nx̃+nX̂)] .= @views estim.Z̃[(nx̃+nX̂-nx̂+1):(nx̃+nX̂)]
    # --- process noise estimates Ŵ ---
    Z̃s[(nx̃+nX̂+1):end] = estim.Ŵ
    # --- verify definiteness of objective function ---
    V̂, X̂0 = estim.buffer.V̂, estim.buffer.X̂
    x̄ .= 0 # x̂0arr == x̂arr_old implies the error at arrival x̄ is zero
    predict_mhe!(V̂, X̂0, û0, k, ŷ0, estim, model, estim.x̂0arr_old, estim.Ŵ, Z̃s)
    Js = obj_nonlinprog(estim, model, x̄, V̂, estim.Ŵ, Z̃s)
    if !isfinite(Js)
        Z̃s[nx̃+1:end] .= 0
    end
    # --- unused variable in Z̃ (applied only when Nk < He) ---
    # We force the update of the NLP gradient and jacobian by warm-starting the unused 
    # variable in Z̃ at 1. Since estim.Ŵ is initialized with 0s, at least 1 variable in Z̃s
    # will be inevitably different at the following time step.
    Z̃s[nx̃+Nk*nŵ+1:end] .= 1
    JuMP.set_start_value.(Z̃var, Z̃s)
    return Z̃s
end