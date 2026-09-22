"Get the number of elements in the optimization decision vector `Z`"
get_nZ_mhe(::SingleShooting, He, nx̂ , ::Any, nŵ, ::Any=0) = nx̂ + nŵ*He
get_nZ_mhe(::MultipleShooting, He, nx̂ , ::Any, nŵ, ::Any=0) = nx̂ + nx̂*He + nŵ*He
function get_nZ_mhe(::TrapezoidalCollocation, He, nx̂, ::Any, nŵ, na=0)
    return nx̂ + nx̂*He + na + na*He + na*He + nŵ*He
end
function get_nZ_mhe(transcription::OrthogonalCollocation, He, nx̂, nk̄, nŵ, na=0) 
    nā = na*transcription.no
    return nx̂ + nx̂*He + na + na*He + nk̄*He + nā*He + nŵ*He
end

"Get the element indices in the decision vector `Z̃` that applies to a `Nk` window length."
function get_i_Z̃_Nk(estim::MovingHorizonEstimator, ::TranscriptionMethod)
    nx̂, nx̃, nŵ, Nk = estim.nx̂, estim.nx̃, estim.nx̂, estim.Nk[]
    na = get_na(estim.model)
    nŴ, nX̂, nA   = nŵ*Nk, nx̂*Nk, na*Nk
    nx̃_nX̂_He          = nx̃ + nx̂*estim.He
    na_nA_He          = na + na*estim.He
    nx̃_nX̂_na_nA_He    = nx̃_nX̂_He + na_nA_He
    nx̃_nX̂_na_nA_nĀ_He = nx̃_nX̂_na_nA_He + na*estim.He
    i_Z̃_NK = [
        (1):(nx̃ + nX̂);
        (1 + nx̃_nX̂_He):(nx̃_nX̂_He + na + nA)
        (1 + nx̃_nX̂_na_nA_He):(nx̃_nX̂_na_nA_He + nA)
        (1 + nx̃_nX̂_na_nA_nĀ_He):(nx̃_nX̂_na_nA_nĀ_He + nŴ)
    ]
    return i_Z̃_NK
end
function get_i_Z̃_Nk(estim::MovingHorizonEstimator, transcription::OrthogonalCollocation)
    nx̂, nx̃, nŵ, Nk = estim.nx̂, estim.nx̃, estim.nx̂, estim.Nk[]
    na = get_na(estim.model)
    nk̄, nā = get_nk̄(estim.model, transcription), get_nā(estim.model, transcription)
    nŴ, nX̂, nA, nK̄, nĀ  = nŵ*Nk, nx̂*Nk, na*Nk, nk̄*Nk, nā*Nk
    nx̃_nX̂_He             = nx̃ + nx̂*estim.He
    na_nA_He             = na + na*estim.He
    nx̃_nX̂_na_nA_He       = nx̃_nX̂_He          + na_nA_He
    nx̃_nX̂_na_nA_nK̄_He    = nx̃_nX̂_na_nA_He    + nk̄*estim.He
    nx̃_nX̂_na_nA_nK̄_nĀ_He = nx̃_nX̂_na_nA_nK̄_He + na*estim.He
    i_Z̃_NK = [
        (1):(nx̃ + nX̂);
        (1 + nx̃_nX̂_He):(nx̃_nX̂_He + na + nA);
        (1 + nx̃_nX̂_na_nA_He):(nx̃_nX̂_na_nA_He + nK̄);
        (1 + nx̃_nX̂_na_nA_nK̄_He):(nx̃_nX̂_na_nA_nK̄_He + nĀ);
        (1 + nx̃_nX̂_na_nA_nK̄_nĀ_He):(nx̃_nX̂_na_nA_nK̄_nĀ_He + nŴ);
    ]
    return i_Z̃_NK
end
function get_i_Z̃_Nk(estim::MovingHorizonEstimator, ::SingleShooting) 
    nx̃, nŵ, Nk = estim.nx̃, estim.nx̂, estim.Nk[]
    return (1):(nx̃ + nŵ*Nk)
end

@doc raw"""
    init_predmat_mhe(
        model::LinModel, transcription::SingleShooting, direct::Bool,
        He, Â, B̂u, Ĉm, B̂d, D̂dm, x̂op, f̂op
    ) -> E, G, J, B, ex̄, EX̂, GX̂, JX̂, BX̂

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
    \mathbf{X̂_0}  &= \mathbf{E_X̂ Z + G_X̂ U_0 + J_X̂ D_0 + B_X̂} \\
                  &= \mathbf{E_X̂ Z + F_X̂}
\end{aligned}
```
The matrices ``\mathbf{E, G, J, B, E_X̂, G_X̂, J_X̂, B_X̂}`` are defined in the Extended Help 
section. The vectors ``\mathbf{F, F_X̂, f_x̄}`` are recalculated at each discrete time step, 
see [`initpred!`](@ref) and [`linconstraint!`](@ref).

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
    \mathbf{E_X̂} &= \begin{bmatrix}
        \mathbf{Â}^{1}                      & \mathbf{A}^{0}                    & \cdots & \mathbf{0}                   \\
        \mathbf{Â}^{2}                      & \mathbf{Â}^{1}                    & \cdots & \mathbf{0}                   \\ 
        \vdots                              & \vdots                            & \ddots & \vdots                       \\
        \mathbf{Â}^{H_e}                    & \mathbf{Â}^{H_e-1}                & \cdots & \mathbf{Â}^{0}               \end{bmatrix} \\
    \mathbf{G_X̂} &= \begin{bmatrix}
        \mathbf{Â}^{0}\mathbf{B̂_u}          & \mathbf{0}                        & \cdots & \mathbf{0}                   \\ 
        \mathbf{Â}^{1}\mathbf{B̂_u}          & \mathbf{Â}^{0}\mathbf{B̂_u}        & \cdots & \mathbf{0}                   \\ 
        \vdots                              & \vdots                            & \ddots & \vdots                       \\
        \mathbf{Â}^{H_e-1}\mathbf{B̂_u}      & \mathbf{Â}^{H_e-2}\mathbf{B̂_u}    & \cdots & \mathbf{Â}^{0}\mathbf{B̂_u}   \end{bmatrix} \\
    \mathbf{J_X̂^†} &= \begin{bmatrix}
        \mathbf{Â}^{0}\mathbf{B̂_d}          & \mathbf{0}                        & \cdots & \mathbf{0}                   \\ 
        \mathbf{Â}^{1}\mathbf{B̂_d}          & \mathbf{Â}^{0}\mathbf{B̂_d}        & \cdots & \mathbf{0}                   \\ 
        \vdots                              & \vdots                            & \ddots & \vdots                       \\
        \mathbf{Â}^{H_e-1}\mathbf{B̂_d}      & \mathbf{Â}^{H_e-2}\mathbf{B̂_d}    & \cdots & \mathbf{Â}^{0}\mathbf{B̂_d}   \end{bmatrix} \ , \quad
    \mathbf{J_X̂} = \begin{cases}
        [\begin{smallmatrix} \mathbf{J_X̂^†} & \mathbf{0}      \end{smallmatrix}]   & p=0                                \\
        [\begin{smallmatrix} \mathbf{0}     & \mathbf{J_X̂^†}  \end{smallmatrix}]   & p=1                                \end{cases}   \\
    \mathbf{B_X̂} &= \begin{bmatrix}
        \mathbf{S}(0)                    \\
        \mathbf{S}(1)                    \\
        \vdots                           \\
        \mathbf{S}(H_e-1) \end{bmatrix}  \mathbf{\big(f̂_{op} - x̂_{op}\big)}
    \end{aligned}
    ```
    All these matrices are truncated when ``N_k < H_e`` (at the beginning).
"""
function init_predmat_mhe(
    model::LinModel{NT}, ::SingleShooting, direct::Bool, He, Â, B̂u, Ĉm, B̂d, D̂dm, x̂op, f̂op
) where {NT<:Real}
    nu, nd = model.nu, model.nd
    nym, nx̂ = size(Ĉm, 1), size(Â, 2)
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
    EX̂ = zeros(NT, nx̂*He, nx̂ + nŵ*He)
    i=0
    for j=1:He
        iRow = (1 + i*nx̂):(nx̂*He)
        iCol = (1:nŵ) .+ j*nŵ
        EX̂[iRow, iCol] = @views Âpow_vec[1:length(iRow) ,:]
        i+=1
    end
    EX̂[:, 1:nx̂] = @views Âpow_vec[nx̂+1:end, :] 
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
    GX̂ = zeros(NT, nx̂*He, nu*He)
    for j=0:He-1
        iRow = (1 + j*nx̂):(nx̂*He)
        iCol = (1:nu) .+ j*nu
        GX̂[iRow, iCol] = @views Âpow_B̂u[1:length(iRow) ,:]
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
    JX̂ = zeros(NT, nx̂*He, nd*(He+1))
    for j=0:He-1
        iRow = (1 + j*nx̂):(nx̂*He)
        iCol = (1:nd) .+ j*nd .+ p
        JX̂[iRow, iCol] = Âpow_B̂d[1:length(iRow) ,:]
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
    BX̂ = coef_Bx̂*f̂_op_n_x̂op
    return E, G, J, B, ex̄, EX̂, GX̂, JX̂, BX̂
end

@doc raw"""
    init_predmat_mhe(
        model::LinModel, transcription::MultipleShooting, direct::Bool,
        He, Â, B̂u, Ĉm, B̂d, D̂dm, x̂op, f̂op
    ) -> E, G, J, B, ex̄, EX̂, GX̂, JX̂, BX̂

Construct them for [`LinModel`](@ref) and [`MultipleShooting`](@ref).

The ``\mathbf{e_x̄}`` is identical to the [`SingleShooting`](@ref) transcription. The other
matrices are defined in the Extended Help section.

# Extended Help
!!! details "Extended Help"
    The matrices are compute by (notice the minus signs after the equalities):
    ```math
    \begin{aligned}
    \mathbf{E} &= - \begin{bmatrix} 
        \mathbf{E^x̂} & \mathbf{E^ŵ}                                                          \end{bmatrix} \\
    \mathbf{G} &=   \mathbf{0}                                                               \\
    \mathbf{J} &= - \begin{bmatrix}
        \mathbf{0}  & \mathbf{D̂_d^m}    & \mathbf{0}     & \cdots & \mathbf{0}               \\ 
        \mathbf{0}  & \mathbf{0}        & \mathbf{D̂_d^m} & \cdots & \mathbf{0}               \\ 
        \vdots      & \vdots            & \vdots         & \ddots & \vdots                   \\
        \mathbf{0}  & \mathbf{0}        & \mathbf{0}     & \cdots & \mathbf{D̂_d^m}           \end{bmatrix} \\
    \mathbf{B} &=   \mathbf{0}
    \end{aligned}
    ```
    The ``\mathbf{E^ŵ}`` matrix is an appropriately size ``\mathbf{0}`` matrix and, for 
    ``p=0``, we have:
    ```math
    \mathbf{E^x̂} = \begin{bmatrix}
        \mathbf{0}      & \mathbf{Ĉ^m}      & \mathbf{0}     & \cdots       & \mathbf{0}     \\ 
        \mathbf{0}      & \mathbf{0}        & \mathbf{Ĉ^m}   & \cdots       & \mathbf{0}     \\ 
        \vdots          & \vdots            & \vdots         & \ddots       & \vdots         \\
        \mathbf{0}      & \mathbf{0}        & \mathbf{0}     & \cdots       & \mathbf{Ĉ^m}   \end{bmatrix}
    ```
    or, for ``p=1``:
    ```math
    \mathbf{E^x̂} = \begin{bmatrix}
        \mathbf{Ĉ^m}   & \mathbf{0}         & \cdots        & \mathbf{0}   & \mathbf{0}      \\ 
        \mathbf{0}     & \mathbf{Ĉ^m}       & \cdots        & \mathbf{0}   & \mathbf{0}      \\ 
        \vdots         & \vdots             & \ddots        & \vdots       & \vdots          \\
        \mathbf{0}     & \mathbf{0}         & \cdots        & \mathbf{Ĉ^m} & \mathbf{0}      \end{bmatrix}
    ```
    The matrices for the estimated states are computed by:
    ```math
    \begin{aligned}
    \mathbf{E_X̂} &= \begin{bmatrix} 
        \mathbf{E_X̂^x̂} & \mathbf{E_X̂^ŵ}                                                      \end{bmatrix} \\
    \mathbf{G_X̂} &= \mathbf{0}                                                               \\
    \mathbf{J_X̂} &= \mathbf{0}                                                               \\
    \mathbf{B_X̂} &= \mathbf{0}
    \end{aligned}
    ```
    The ``\mathbf{E_X̂^ŵ}`` matrix is an appropriately size ``\mathbf{0}`` matrix and:
    ```math
    \mathbf{E_X̂^x̂} = \begin{bmatrix}
        \mathbf{0}      & \mathbf{I}        & \mathbf{0}     & \cdots       & \mathbf{0}     \\ 
        \mathbf{0}      & \mathbf{0}        & \mathbf{I}     & \cdots       & \mathbf{0}     \\ 
        \vdots          & \vdots            & \vdots         & \ddots       & \vdots         \\
        \mathbf{0}      & \mathbf{0}        & \mathbf{0}     & \cdots       & \mathbf{I}     \end{bmatrix}
    ```
    The appropriate rows and columns on these matrices are selected using the slicing
    operator `A[i_rows, i_cols]` when ``N_k < H_e`` (at the beginning).
"""
function init_predmat_mhe(
    model::LinModel{NT}, ::MultipleShooting, direct::Bool,
    He, Â, _ , Ĉm, _ , D̂dm, _ , _ 
) where {NT<:Real}
    nu, nd = model.nu, model.nd
    nym, nx̂ = size(Ĉm, 1), size(Â, 2)
    nŵ = nx̂
    p = direct ? 0 : 1
    nX̂, nŴ, nV̂, nU, nD = nx̂*He, nŵ*He, nym*He, nu*He, nd*(He+1)
    # --- decision variables Z ---
    E  = [zeros(NT, nV̂, (1-p)*nx̂) repeatdiag(-Ĉm, He) zeros(NT, nV̂, p*nx̂ + nŴ)]
    ex̄ = [-I zeros(NT, nx̂, nX̂ + nŴ)]
    EX̂ = [zeros(NT, nX̂, nx̂) I zeros(NT, nX̂, nŴ)]
    # --- manipulated inputs U ---
    G  = zeros(NT, nV̂, nU)
    GX̂ = zeros(NT, nX̂, nU)
    # --- measured disturbances D ---
    J  = [zeros(NT, nV̂, nd) repeatdiag(-D̂dm, He)]
    JX̂ = zeros(NT, nX̂, nD)
    # --- state x̂op and state update f̂op operating points ---
    B  = zeros(NT, nV̂)
    BX̂ = zeros(NT, nX̂)
    return E, G, J, B, ex̄, EX̂, GX̂, JX̂, BX̂
end

"""
    init_predmat_mhe(
        model::SimModel, transcription::SingleShooting, direct::Bool,
        He, Â, B̂u, Ĉm, B̂d, D̂dm, x̂op, f̂op
    ) -> E, G, J, B, ex̄, EX̂, GX̂, JX̂, BX̂

Return empty matrices for [`SingleShooting`](@ref) and non-`LinModel`, except for `ex̄`.
"""
function init_predmat_mhe(
    model::SimModel{NT}, transcription::SingleShooting, ::Bool,
    He, Â, _ , Ĉm, _ , _ , _ , _ 
) where {NT<:Real}
    nym, nx̂ = size(Ĉm, 1), size(Â, 2)
    nŵ = nx̂
    nk̄ = get_nk̄(model, transcription)
    nZ = get_nZ_mhe(transcription, He, nx̂, nk̄, nŵ)
    E  = zeros(NT, 0, nZ)
    ex̄ = [-I zeros(NT, nx̂, nZ - nx̂)]
    EX̂ = zeros(NT, 0, nZ)
    G  = zeros(NT, 0, model.nu*He)
    GX̂ = zeros(NT, 0, model.nu*He)
    J  = zeros(NT, 0, model.nd*(He+1))
    JX̂ = zeros(NT, 0, model.nd*(He+1))
    B  = zeros(NT, nym*He)
    BX̂ = zeros(NT, nx̂*He)
    return E, G, J, B, ex̄, EX̂, GX̂, JX̂, BX̂
end

"""
    init_predmat_mhe(
        model::SimModel, transcription::TranscriptionMethod, direct::Bool
        He, Â, B̂u, Ĉm, B̂d, D̂dm, x̂op, f̂op
    ) -> E, G, J, B, ex̄, EX̂, GX̂, JX̂, BX̂

Return `ex̄, EX̂, GX̂, JX̂, BX̂` and empty matrices non-`LinModel` and other [`TranscriptionMethod`](@ref).
"""
function init_predmat_mhe(
    model::SimModel{NT}, transcription::TranscriptionMethod, ::Bool,
    He, Â, _ , Ĉm, _ , _ , _ , _ 
) where {NT<:Real}
    nym, nx̂ = size(Ĉm, 1), size(Â, 2)
    nŵ = nx̂
    nk̄, na = get_nk̄(model, transcription), get_na(model)
    nZ = get_nZ_mhe(transcription, He, nx̂, nk̄, nŵ, na)
    E  = zeros(NT, 0, nZ)
    ex̄ = [-I zeros(NT, nx̂, nZ - nx̂)]
    EX̂ = [zeros(NT, nx̂*He, nx̂) I zeros(NT, nx̂*He, nZ - nx̂ - nx̂*He)]
    G  = zeros(NT, 0, model.nu*He)
    GX̂ = zeros(NT, nx̂*He, model.nu*He)
    J  = zeros(NT, 0, model.nd*(He+1))
    JX̂ = zeros(NT, nx̂*He, model.nd*(He+1))
    B  = zeros(NT, nym*He)
    BX̂ = zeros(NT, nx̂*He)
    return E, G, J, B, ex̄, EX̂, GX̂, JX̂, BX̂
end

@doc raw"""
    init_defectmat_mhe(
        model::LinModel, transcription::MultipleShooting, direct::Bool,
        He, Â, B̂u, B̂d, x̂op, f̂op, As, Co, λo
    ) -> ES, GS, JS, BS

Init the matrices for computing the defects over the predicted states.

With a [`MultipleShooting`](@ref) transcription, the decision vector ``\mathbf{Z}`` contains
the arrival state estimate ``\mathbf{x̂_0}(k-N_k+p)``, the stage states ``\mathbf{X̂_0}`` 
(both defined as deviation vector from ``\mathbf{x̂_{op}}``) and the estimated process noises
``\mathbf{Ŵ}``. Knowing this, an equation similar to the prediction matrices (see 
[`init_predmat_mhe`](@ref)) computes the defects of the estimated states over ``H_e``:
```math
\begin{aligned}
    \mathbf{Ŝ}  &= \mathbf{E_S Z} + \mathbf{G_S U_0}  + \mathbf{J_S D_0} + \mathbf{B_S}       \\
                &= \mathbf{E_S Z} + \mathbf{F_S}
\end{aligned}
```   
They are forced to be ``\mathbf{Ŝ = 0}`` using the optimization equality constraints. The
matrices ``\mathbf{E_S, G_S, J_S, B_S}`` are defined in the Extended Help section.

# Extended Help
!!! details "Extended Help"
    The defect matrices are computed with:
    ```math
    \begin{aligned}
    \mathbf{E_S} &= \begin{bmatrix}
        \mathbf{E_S^x̂} & \mathbf{E_S^ŵ}                                                             \end{bmatrix} \\
    \mathbf{E_S^x̂} &= \begin{bmatrix}
        \mathbf{Â}     & \mathbf{-I}    & \mathbf{0}    & \cdots    & \mathbf{0} & \mathbf{0}       \\
        \mathbf{0}     & \mathbf{Â}     & \mathbf{-I}   & \cdots    & \mathbf{0} & \mathbf{0}       \\
        \vdots         & \vdots         & \vdots        & \ddots    & \vdots     & \vdots           \\
        \mathbf{0}     & \mathbf{0}     & \mathbf{0}    & \cdots    & \mathbf{Â} & \mathbf{-I}      \end{bmatrix} \\
    \mathbf{E_S^ŵ} &= \begin{bmatrix}
        \mathbf{I}     &  \mathbf{0}    & \cdots        & \mathbf{0}                                \\
        \mathbf{0}     &  \mathbf{I}    & \cdots        & \mathbf{0}                                \\
        \vdots         &  \vdots        & \ddots        & \vdots                                    \\
        \mathbf{0}     &  \mathbf{0}    & \cdots        & \mathbf{I}                                \end{bmatrix} \\
    \mathbf{G_S} &= \begin{bmatrix}
        \mathbf{B̂_u}   &  \mathbf{0}    & \cdots        & \mathbf{0}                                \\
        \mathbf{0}     &  \mathbf{B̂_u}  & \cdots        & \mathbf{0}                                \\
        \vdots         &  \vdots        & \ddots        & \vdots                                    \\
        \mathbf{0}     &  \mathbf{0}    & \cdots        & \mathbf{B̂_u}                              \end{bmatrix} \\
    \mathbf{J_S^†} &= \begin{bmatrix}
        \mathbf{B̂_d}   & \mathbf{0}     & \cdots        & \mathbf{0}                                \\
        \mathbf{0}     & \mathbf{B̂_d}   & \cdots        & \mathbf{0}                                \\
        \vdots         & \vdots         & \ddots        & \vdots                                    \\
        \mathbf{0}     & \mathbf{0}     & \cdots        & \mathbf{B̂_d}                              \end{bmatrix} \ , \quad
    \mathbf{J_S} &= \begin{cases}
        [\begin{smallmatrix} \mathbf{J_S^†} & \mathbf{0}      \end{smallmatrix}]   & p=0            \\
        [\begin{smallmatrix} \mathbf{0}     & \mathbf{J_S^†}  \end{smallmatrix}]   & p=1            \end{cases}   \\
    \mathbf{B_S} &= \begin{bmatrix}
        \mathbf{f̂_{op} - x̂_{op}} \\ \mathbf{f̂_{op} - x̂_{op}} \\ \vdots \\ \mathbf{f̂_{op} - x̂_{op}}  \end{bmatrix}
    \end{aligned}
    ```
    The appropriate rows and columns on these matrices are selected using the slicing
    operator `A[i_rows, i_cols]` when ``N_k < H_e`` (at the beginning).
"""
function init_defectmat_mhe(
    model::LinModel{NT}, ::MultipleShooting, direct::Bool, 
    He, Â, B̂u, B̂d, x̂op, f̂op, _ , _ , _
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

@doc raw"""
    init_defectmat_mhe(
        model::SimModel, transcription::TranscriptionMethod, direct::Bool,
        He, Â, _ , _ , _ , _ , As, _ , _ 
    ) -> ES, GS, JS, BS

Init the matrices for computing the defects of the stochastic states only.

The documentation of [`init_estimstoch`](@ref) shows that the stochastic model of the 
unmeasured disturbances is linear and discrete-time. The defect of the stochastic states
over ``H_e`` is therefore computed by:
```math
    \mathbf{Ŝ} = \mathbf{E_S Z} 
```   
The matrix ``\mathbf{E_S}`` is defined in the Extended Help section.

# Extended Help
!!! details "Extended Help"
    Using the stochastic matrix ``\mathbf{A_s}`` of [`init_estimstoch`](@ref)), and
    updating the stochastic states by adding their associated process noise estimates 
    ``\mathbf{ŵ_s}``, the defect matrices are computed with:
    ```math
    \begin{aligned}
    \mathbf{E_S^x̂} &= \begin{bmatrix}
        \mathbf{0} & \mathbf{A_s} & \mathbf{0} & \mathbf{-I}  & \mathbf{0}  & \mathbf{0}  & \cdots & \mathbf{0}   & \mathbf{0} & \mathbf{0}   \\
        \mathbf{0} & \mathbf{0}   & \mathbf{0} & \mathbf{A_s} & \mathbf{0}  & \mathbf{-I} & \cdots & \mathbf{0}   & \mathbf{0} & \mathbf{0}   \\
        \vdots     & \vdots       & \vdots     & \vdots       & \mathbf{0}  & \mathbf{0}  & \ddots & \vdots       & \vdots     & \vdots       \\
        \mathbf{0} & \mathbf{0}   & \mathbf{0} & \mathbf{0}   & \mathbf{0}  & \mathbf{0}  & \cdots & \mathbf{A_s} & \mathbf{0} & \mathbf{-I}  \end{bmatrix} \\
    \mathbf{E_S^ŵ} &= \begin{bmatrix}
        \mathbf{0} & \mathbf{I}   & \mathbf{0} & \mathbf{0}   & \cdots      & \mathbf{0}  & \mathbf{0}                                        \\
        \mathbf{0} & \mathbf{0}   & \mathbf{0} & \mathbf{I}   & \cdots      & \mathbf{0}  & \mathbf{0}                                        \\
        \vdots     & \vdots       & \vdots     & \vdots       & \ddots      & \vdots      & \vdots                                            \\
        \mathbf{0} & \mathbf{0}   & \mathbf{0} & \mathbf{0}   & \cdots      & \mathbf{0}  & \mathbf{I}                                        \end{bmatrix} \\
    \mathbf{E_S^a} &= \mathbf{0}                                                                                                              \\
    \mathbf{E_S^ā} &= \mathbf{0}                                                                                                              \\
    \mathbf{E_S}   &= \begin{bmatrix} \mathbf{E_S^x̂} & \mathbf{E_S^a} & \mathbf{E_S^ā} & \mathbf{E_S^ŵ}                                       \end{bmatrix}
    \end{aligned}
    ```
"""
function init_defectmat_mhe(
    model::SimModel{NT}, transcription::TranscriptionMethod, ::Bool, 
    He, Â, _ , _ , _ , _ , As, _ , _
) where {NT<:Real}
    nx̂, nxs = size(Â, 2), size(As, 2)
    na, nā = get_na(model), get_nā(model, transcription)
    nx  = nx̂ - nxs
    nŵ  = nx̂
    nŵd = nŵ - nxs
    ESx̂ = [zeros(NT, nxs*He, nx̂) repeatdiag([zeros(NT, nxs, nx) -I], He)]
    for j=1:He
        iRow = (1:nxs)   .+ nxs*(j-1)
        iCol = (nx+1:nx̂) .+  nx̂*(j-1)
        ESx̂[iRow, iCol] = As
    end
    ESŵ = repeatdiag([zeros(NT, nxs, nŵd) I], He)
    ESa = zeros(nxs*He, na + na*He)
    ESā = zeros(nxs*He, nā*He)
    ES = [ESx̂ ESa ESā ESŵ]
    GS = zeros(NT, nxs*He, model.nu*He)
    JS = zeros(NT, nxs*He, model.nd*(He+1))
    BS = zeros(NT, nxs*He)
    return ES, GS, JS, BS
end

@doc raw"""
    init_defectmat_mhe(
        model::SimModel, transcription::OrthogonalCollocation, direct::Bool
        He, Â, _ , _ , _ , _ , As, Co, λo
    ) -> ES, GS, JS, BS

Init the matrices for computing the continuity constraints and stochastic state defects.

The documentation of [`init_orthocolloc`](@ref) shows that continuity constraints of the
[`OrthogonalCollocation`](@ref) are in fact linear. Combined with the stochastic state
defects, the linear equality constraints for this transcription is given by:
```math
    \mathbf{Ŝ} = \mathbf{E_S Z} 
```   
The matrix ``\mathbf{E_S}`` is defined in the Extended Help section.

# Extended Help
!!! details "Extended Help"
    Using the stochastic matrix ``\mathbf{A_s}`` of [`init_estimstoch`](@ref)), and by
    updating the states by adding the process noise estimates ``\mathbf{ŵ}``, the defect
    matrices are computed with:
    ```math
    \begin{aligned}
    \mathbf{E_S^x̂}  &= \begin{bmatrix}
        λ_o\mathbf{I} & \mathbf{0}   &\mathbf{-I}    & \mathbf{0}   & \cdots & \mathbf{0}    & \mathbf{0}   & \mathbf{0} & \mathbf{0}           \\
        \mathbf{0}    & \mathbf{A_s} & \mathbf{0}    & \mathbf{-I}  & \cdots & \mathbf{0}    & \mathbf{0}   & \mathbf{0} & \mathbf{0}           \\
        \mathbf{0}    & \mathbf{0}   & λ_o\mathbf{I} & \mathbf{0}   & \cdots & \mathbf{0}    & \mathbf{0}   & \mathbf{0} & \mathbf{0}           \\
        \mathbf{0}    & \mathbf{0}   & \mathbf{0}    & \mathbf{A_s} & \cdots & \mathbf{0}    & \mathbf{0}   & \mathbf{0} & \mathbf{0}           \\
        \vdots        & \vdots       & \vdots        & \vdots       & \ddots & \vdots        & \vdots       & \vdots     & \vdots               \\
        \mathbf{0}    & \mathbf{0}   & \mathbf{0}    & \mathbf{0}   & \cdots & λ_o\mathbf{I} & \mathbf{0}   & \mathbf{-I} & \mathbf{0}          \\   
        \mathbf{0}    & \mathbf{0}   & \mathbf{0}    & \mathbf{0}   & \cdots & \mathbf{0}    & \mathbf{A_s} & \mathbf{0}  & \mathbf{-I}         \end{bmatrix} \\
    \mathbf{E_S^k̄} &= \begin{bmatrix}
       \mathbf{C_o}   & \mathbf{0}   & \cdots & \mathbf{0}                                                                                      \\
        \mathbf{0}    & \mathbf{0}   & \cdots & \mathbf{0}                                                                                      \\
        \mathbf{0}    & \mathbf{C_o} & \cdots & \mathbf{0}                                                                                      \\
        \mathbf{0}    & \mathbf{0}   & \cdots & \mathbf{0}                                                                                      \\
        \vdots        & \vdots       & \ddots & \vdots                                                                                          \\
        \mathbf{0}    & \mathbf{0}   & \cdots & \mathbf{C_o}                                                                                    \\ 
        \mathbf{0}    & \mathbf{0}   & \cdots & \mathbf{0}                                                                                      \end{bmatrix} \\
    \mathbf{E_S^ŵ} &= \mathbf{I}                                                                                                                \\
    \mathbf{E_S^a} &= \mathbf{0}                                                                                                                \\
    \mathbf{E_S^ā} &= \mathbf{0}                                                                                                                \\
    \mathbf{E_S}   &= \begin{bmatrix} \mathbf{E_S^x̂} & \mathbf{E_S^a} & \mathbf{E_S^k̄} & \mathbf{E_S^ā} & \mathbf{E_S^ŵ}                        \end{bmatrix} \\
    \end{aligned}
    ```
"""
function init_defectmat_mhe(
    model::SimModel{NT}, transcription::OrthogonalCollocation, ::Bool,
    He, Â, _ , _ , _ , _ , As, Co, λo
) where {NT<:Real}
    nx̂, nxs = size(Â, 2), size(As, 2)
    nx  = nx̂ - nxs
    na, nā = get_na(model), get_nā(model, transcription)
    nk̄ = get_nk̄(model, transcription)
    λo_I = λo*I(nx)
    ESx̂ = [zeros(NT, nx̂*He, nx̂) -I]
    for j=1:He
        iRow = (1:nx) .+ (j-1)*nx̂
        iCol = (1:nx) .+ (j-1)*nx̂
        ESx̂[iRow, iCol] = λo_I
        iRow = (nx+1:nx̂) .+ (j-1)*nx̂
        iCol = (nx+1:nx̂) .+ (j-1)*nx̂
        ESx̂[iRow, iCol] = As
    end
    ESk̄ = repeatdiag([Co; zeros(NT, nxs, nk̄)], He)
    ESŵ = I # will be different if nŵ ≠ nx̂ is implemented
    ESa = zeros(nx̂*He, na + na*He)
    ESā = zeros(nx̂*He, nā*He)
    ES = [ESx̂ ESa ESk̄ ESā ESŵ]
    GS = zeros(NT, nxs*He, model.nu*He)
    JS = zeros(NT, nxs*He, model.nd*(He+1))
    BS = zeros(NT, nxs*He)
    return ES, GS, JS, BS
end

"Return empty matrices for [`SingleShooting`](@ref) transcription on any `SimModelODE` (N/A)."
function init_defectmat_mhe(
    model::SimModelODE{NT}, transcription::SingleShooting, ::Bool, 
    He, Â, _ , _ , _ , _ , _ , _ , _
) where {NT<:Real}
    nx̂ = size(Â, 2)
    nŵ = nx̂
    return init_defectmat_mhe_empty(model, transcription, He, nx̂, nŵ)
end

function init_defectmat_mhe_empty(
    model::SimModel{NT}, transcription::TranscriptionMethod, He, nx̂, nŵ
) where {NT<:Real}
    nu, nd = model.nu, model.nd
    nk̄, na = get_nk̄(model, transcription), get_na(model)
    nZ = get_nZ_mhe(transcription, He, nx̂, nk̄, nŵ, na)
    ES = zeros(NT, 0, nZ)
    GS = zeros(NT, 0, nu*He)
    JS = zeros(NT, 0, nd*(He+1))
    BS = zeros(NT, 0)
    return ES, GS, JS, BS
end

@doc raw"""
    init_matconstraint_mhe(
        model::LinModel, transcription::ShootingMethod, Z̃min, Z̃max, nc, nε,
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
also returns the ``\mathbf{A, A_{eq}}`` matrices and `neq` if `args` is provided. In such a
case, `args`  needs to contain all the inequality and equality constraint matrices: 
`A_x̂min, A_x̂max, A_X̂min, A_X̂max, A_Ŵmin, A_Ŵmax, A_V̂min, A_V̂max, Aeq`. The integer `neq` is
the number of nonlinear equality constraints in ``\mathbf{g_{eq}}``.
"""
function init_matconstraint_mhe(
    model::LinModel{NT}, transcription::ShootingMethod, Z̃min, Z̃max, nc, nε,
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
    deletex̂arr_lincon!(i_x̂min, i_x̂max, model, transcription, Z̃min, Z̃max, nε)
    deleteX̂_lincon!(i_X̂min, i_X̂max, model, transcription, Z̃min, Z̃max, nε, nx̂)
    deleteŴ_lincon!(i_Ŵmin, i_Ŵmax, model, transcription, Z̃min, Z̃max)
    i_b = [i_x̂min; i_x̂max; i_X̂min; i_X̂max; i_Ŵmin; i_Ŵmax; i_V̂min; i_V̂max]
    i_g = trues(nc)
    return i_b, i_g, A, Aeq, neq
end

"Init `i_b, A` without sensor noise and state constraints if `NonLinModel` and `SingleShooting`."
function init_matconstraint_mhe(
    model::NonLinModel{NT}, transcription::SingleShooting, Z̃min, Z̃max, nc, nε,
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
    deletex̂arr_lincon!(i_x̂min, i_x̂max, model, transcription, Z̃min, Z̃max, nε)
    deleteX̂_lincon!(i_X̂min, i_X̂max, model, transcription, Z̃min, Z̃max, nε, nx̂)
    deleteŴ_lincon!(i_Ŵmin, i_Ŵmax, model, transcription, Z̃min, Z̃max)
    i_b = [i_x̂min; i_x̂max; i_Ŵmin; i_Ŵmax]
    i_g = [i_X̂min; i_X̂max; i_V̂min; i_V̂max; trues(nc)]
    return i_b, i_g, A, Aeq, neq
end

"Init `i_b, A` without sensor noise constraints if not `LinModel` and other `TranscriptionMethod`."
function init_matconstraint_mhe(
    model::SimModel{NT}, transcription::TranscriptionMethod, Z̃min, Z̃max, nc, nε,
    x̂0min, x̂0max, X̂0min, X̂0max, Ŵmin, Ŵmax, V̂min, V̂max, args...
) where {NT<:Real}
    if isempty(args)
        A, Aeq, neq = nothing, nothing, nothing
    else
        A_x̂min, A_x̂max, A_X̂min, A_X̂max, A_Ŵmin, A_Ŵmax, _ , _ , Aeq = args
        A = [A_x̂min; A_x̂max; A_X̂min; A_X̂max; A_Ŵmin; A_Ŵmax]
        nŴ, nZ̃ = size(A_Ŵmin)
        nx̂ = length(x̂0min)
        nAeq = size(Aeq, 1)                  # number of linear equality constraints
        neq  = nZ̃ - nŴ - nε - nx̂ - nAeq      # number of nonlinear equality constraints
    end
    i_x̂min, i_x̂max  = @. !isinf(x̂0min), !isinf(x̂0max)
    i_X̂min, i_X̂max  = @. !isinf(X̂0min), !isinf(X̂0max)
    i_Ŵmin, i_Ŵmax  = @. !isinf(Ŵmin),  !isinf(Ŵmax)
    i_V̂min, i_V̂max  = @. !isinf(V̂min),  !isinf(V̂max)
    nx̂ = length(x̂0min)
    deletex̂arr_lincon!(i_x̂min, i_x̂max, model, transcription, Z̃min, Z̃max, nε)
    deleteX̂_lincon!(i_X̂min, i_X̂max, model, transcription, Z̃min, Z̃max, nε, nx̂)
    deleteŴ_lincon!(i_Ŵmin, i_Ŵmax, model, transcription, Z̃min, Z̃max)
    i_b = [i_x̂min; i_x̂max; i_X̂min; i_X̂max; i_Ŵmin; i_Ŵmax]
    i_g = [i_V̂min; i_V̂max; trues(nc)]
    return i_b, i_g, A, Aeq, neq
end

"Modify `Z̃min` and `Z̃max` in-place to include state estimate constraints if applicable."
function boxconstraint_states!(
    Z̃min, Z̃max, ::TranscriptionMethod, nx̂, nε, X̂0min, X̂0max, C_x̂min, C_x̂max
)
    nx̃, nX̂ = nε + nx̂, length(X̂0min)
    if nε > 0
        for i in eachindex(X̂0min)
            iszero(C_x̂min[i]) && (Z̃min[nx̃ + i] = X̂0min[i])
        end
        for i in eachindex(X̂0max)
            iszero(C_x̂max[i]) && (Z̃max[nx̃ + i] = X̂0max[i])
        end
    else
        Z̃min[(nx̃+1):(nx̃+nX̂)] .= X̂0min
        Z̃max[(nx̃+1):(nx̃+nX̂)] .= X̂0max
    end
    return Z̃min, Z̃max
end
boxconstraint_states!(Z̃min, Z̃max, ::SingleShooting, _, _, _, _, _, _) = Z̃min, Z̃max

"Unset `i_x̂min` and `i_x̂max` elements if finite box constraints in `Z̃min` and `Z̃max`."
function deletex̂arr_lincon!(
    i_x̂min, i_x̂max, ::SimModel, ::TranscriptionMethod, Z̃min, Z̃max, nε
)
    nx̂ = length(i_x̂min)
    x̂0min, x̂0max = @views Z̃min[(nε+1):(nε+nx̂)], @views Z̃max[(nε+1):(nε+nx̂)]
    foreach(i -> !isinf(x̂0min[i]) && (i_x̂min[i] = false), eachindex(i_x̂min))
    foreach(i -> !isinf(x̂0max[i]) && (i_x̂max[i] = false), eachindex(i_x̂max))
    return i_x̂min, i_x̂max
end

"Unset `i_X̂min` and `i_X̂max` elements if finite box constraints in `Z̃min` and `Z̃max`."
function deleteX̂_lincon!(
    i_X̂min, i_X̂max, ::SimModel, ::TranscriptionMethod, Z̃min, Z̃max, nε, nx̂
)
    nx̃ = nε + nx̂
    nX̂ = length(i_X̂min)
    X̂0min, X̂0max = @views Z̃min[(nx̃+1):(nx̃+nX̂)], @views Z̃max[(nx̃+1):(nx̃+nX̂)]
    foreach(i -> !isinf(X̂0min[i]) && (i_X̂min[i] = false), eachindex(i_X̂min))
    foreach(i -> !isinf(X̂0max[i]) && (i_X̂max[i] = false), eachindex(i_X̂max))
    return i_X̂min, i_X̂max
end
deleteX̂_lincon!(i_X̂min, i_X̂max, ::SimModel, ::SingleShooting, _, _, _, _) = i_X̂min, i_X̂max
    
"Unset `i_Ŵmin` and `i_Ŵmax` elements if finite box constraints in `Z̃min` and `Z̃max`."
function deleteŴ_lincon!(i_Ŵmin, i_Ŵmax, ::SimModel, ::TranscriptionMethod, Z̃min, Z̃max)
    nŴ = length(i_Ŵmin)
    Ŵmin, Ŵmax = @views Z̃min[end-nŴ+1:end], Z̃max[end-nŴ+1:end]
    foreach(i -> !isinf(Ŵmin[i]) && (i_Ŵmin[i] = false), eachindex(i_Ŵmin))
    foreach(i -> !isinf(Ŵmax[i]) && (i_Ŵmax[i] = false), eachindex(i_Ŵmax))
    return i_Ŵmin, i_Ŵmax
end

@doc raw"""
    linconstraint!(estim::MovingHorizonEstimator, model::LinModel, ::TranscriptionMethod)

Set `b` vector for the linear inequality constraints (``\mathbf{A Z̃ ≤ b}``) of MHE.

Also init ``\mathbf{F_X̂ = G_X̂ U_0 + J_X̂ D_0 + B_X̂}`` vector for the state constraints, see 
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
        BX̂     = estim.con.BX̂[1:nX̂]
        GX̂, U0 = estim.con.GX̂[1:nX̂, 1:nU], estim.U0[1:nU]
        JX̂, D0 = estim.con.JX̂[1:nX̂, 1:nD], estim.D0[1:nD]
        FX̂     = @views estim.con.FX̂[1:nX̂]
    else
        BX̂     = estim.con.BX̂
        GX̂, U0 = estim.con.GX̂, estim.U0
        JX̂, D0 = estim.con.JX̂, estim.D0
        FX̂     = estim.con.FX̂
    end
    X̂0min, X̂0max = trunc_bounds(estim, estim.con.X̂0min, estim.con.X̂0max, nx̂)
    Ŵmin, Ŵmax   = trunc_bounds(estim, estim.con.Ŵmin,  estim.con.Ŵmax,  nŵ)
    V̂min, V̂max   = trunc_bounds(estim, estim.con.V̂min,  estim.con.V̂max,  nym)
    # --- update FX̂ vectors for MHE state constraints ---
    FX̂ .= BX̂
    mul!(FX̂, GX̂, U0, 1, 1)
    model.nd > 0 && mul!(FX̂, JX̂, D0, 1, 1)
    # --- update b vector for linear inequality constraints ---
    nX̂_He, nŴ_He, nV̂_He = length(X̂0min), length(Ŵmin), length(V̂min)
    nx̂ = length(estim.con.x̂0min)
    n = 0
    estim.con.b[(n+1):(n+nx̂)] .= @. -estim.con.x̂0min
    n += nx̂
    estim.con.b[(n+1):(n+nx̂)] .= @. +estim.con.x̂0max
    n += nx̂
    estim.con.b[(n+1):(n+nX̂_He)] .= @. -X̂0min + estim.con.FX̂
    n += nX̂_He
    estim.con.b[(n+1):(n+nX̂_He)] .= @. +X̂0max - estim.con.FX̂
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

"Set `b` excluding sensor noise bounds if not a `LinModel` and non-`SingleShooting`."
function linconstraint!(
    estim::MovingHorizonEstimator, ::SimModel, ::TranscriptionMethod
)
    nx̂, nŵ = estim.nx̂, estim.nx̂
    # --- truncate vector and matrices if necessary ---
    X̂0min, X̂0max = trunc_bounds(estim, estim.con.X̂0min, estim.con.X̂0max, nx̂)
    Ŵmin, Ŵmax = trunc_bounds(estim, estim.con.Ŵmin, estim.con.Ŵmax, nŵ)
    # --- update b vector for linear inequality constraints ---
    nx̂, nŴ_He, nX̂_He = length(estim.con.x̂0min), length(Ŵmin), length(X̂0min)
    n = 0
    estim.con.b[(n+1):(n+nx̂)] .= @. -estim.con.x̂0min
    n += nx̂
    estim.con.b[(n+1):(n+nx̂)] .= @. +estim.con.x̂0max
    n += nx̂
    estim.con.b[(n+1):(n+nX̂_He)] .= @. -X̂0min + estim.con.FX̂
    n += nX̂_He
    estim.con.b[(n+1):(n+nX̂_He)] .= @. +X̂0max - estim.con.FX̂
    n += nX̂_He
    estim.con.b[(n+1):(n+nŴ_He)] .= @. -Ŵmin
    n += nŴ_He
    estim.con.b[(n+1):(n+nŴ_He)] .= @. +Ŵmax
    if any(estim.con.i_b) 
        lincon = estim.optim[:linconstraint]
        JuMP.set_normalized_rhs(lincon, estim.con.b[estim.con.i_b])
    end
    return nothing
end

"Set `b` excluding state and sensor noise bounds for `NonLinModel` and `SingleShooting`."
function linconstraint!(
    estim::MovingHorizonEstimator, ::NonLinModel, ::SingleShooting
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

@doc raw"""
    linconstrainteq!(
        estim::MovingHorizonEstimator, model::LinModel, ::MultipleShooting
    )

Set `Aeq` matrix and `beq` vector for the linear equality constraints of MHE.

They are defined by ``\mathbf{A_{eq} Z̃ ≤ b_{eq}}``. The method also inits 
``\mathbf{F_S = G_S U_0 + J_S D_0 + B_S}`` vector for the state defect constraints, see 
[`init_defectmat_mhe`](@ref). 

The number of linear equality constraints grows when ``N_k < H_e``. A temporary
`:linconstrainteq_temp` structure is overwritten at each time
step during this period. A permanent `:linconstrainteq` structure is created when 
``N_k = H_e`` is reached. From this point on, only the the ``beq`` vector is updated
with `JuMP.set_normalized_rhs` function for efficiency.
"""
function linconstrainteq!(
    estim::MovingHorizonEstimator, model::LinModel, ::MultipleShooting
)
    optim = estim.optim
    ẼS, FS, GS, JS, BS, Aeq, beq, Z̃var = trunc_defectmat(estim)
    U0, D0 = trunc_windows(estim)
    FS .= BS
    mul!(FS, GS, U0, 1, 1)
    if model.nd > 0
        mul!(FS, JS, D0, 1, 1)
    end
    beq .= @. -FS
    Aeq .= @.  ẼS
    if haskey(optim, :linconstrainteq_temp)
        JuMP.delete(optim, optim[:linconstrainteq_temp])
        JuMP.unregister(optim, :linconstrainteq_temp)
    end
    if estim.Nk[] < estim.He
        if haskey(optim, :linconstrainteq)
            JuMP.delete(optim, optim[:linconstrainteq])
            JuMP.unregister(optim, :linconstrainteq)
        end
        @constraint(optim, linconstrainteq_temp, Aeq*Z̃var .== beq)
    else
        if haskey(optim, :linconstrainteq)
            JuMP.set_normalized_rhs(optim[:linconstrainteq], beq)
        else
            @constraint(optim, linconstrainteq, Aeq*Z̃var .== beq)
        end
    end
    return nothing
end

"""
    linconstrainteq!(
        estim::MovingHorizonEstimator, ::SimModel, transcription::TranscriptionMethod
    )

By default, only update `Aeq` when `Nk < He` for other [`TranscriptionMethod`](@ref).

The linear equality constraints include the stochastic defects only, and the `beq`
vector is only zeros for this specific case. See [`init_defectmat_mhe`](@ref) for the 
equations.
"""
function linconstrainteq!(
    estim::MovingHorizonEstimator, ::SimModel, transcription::TranscriptionMethod
)
    optim, con, Nk = estim.optim, estim.con, estim.Nk[]
    nŝ = size(con.Aeq, 1) ÷ estim.He # number of state defects per time step
    nŜ = nŝ*Nk
    if Nk < estim.He # avoid views since allocations only when Nk < He and we want fast mul!
        i_Z̃_Nk = get_i_Z̃_Nk(estim, transcription)
        ẼS = con.ẼS[1:nŜ, i_Z̃_Nk]
        Aeq, beq = @views con.Aeq[1:nŜ, i_Z̃_Nk], con.beq[1:nŜ]
        Z̃var     = @views optim[:Z̃var][i_Z̃_Nk]
    else
        ẼS       = con.ẼS
        Aeq, beq = con.Aeq, con.beq
        Z̃var     = optim[:Z̃var]
    end
    Aeq .= ẼS
    if haskey(optim, :linconstrainteq_temp)
        JuMP.delete(optim, optim[:linconstrainteq_temp])
        JuMP.unregister(optim, :linconstrainteq_temp)
    end
    if Nk < estim.He
        if haskey(optim, :linconstrainteq)
            JuMP.delete(optim, optim[:linconstrainteq])
            JuMP.unregister(optim, :linconstrainteq)
        end
        @constraint(optim, linconstrainteq_temp, Aeq*Z̃var .== beq)
    else
        if !haskey(optim, :linconstrainteq)
            @constraint(optim, linconstrainteq, Aeq*Z̃var .== beq)
        end
    end
    return nothing
end
"No linear equality constraints for all cases of [`SingleShooting`](@ref)."
linconstrainteq!(::MovingHorizonEstimator, ::SimModel, ::SingleShooting) = nothing

"""
    reset_warmstart!(estim::MovingHorizonEstimator, ::TranscriptionMethod)

Reset warm-starting values `estim.Z̃` at values stored in `estim.model`
"""
function reset_warmstart!(estim::MovingHorizonEstimator, ::MultipleShooting)
    model = estim.model
    nx, nx̂, nx̃, nε, He = model.nx, estim.nx̂, estim.nx̃, estim.nε, estim.He
    x0s  = model.buffer.x
    x0s .= model.xs_0 .- model.xop
    estim.Z̃ .= 0
    estim.Z̃[nε+1:nε+nx] = x0s
    for j in 1:He
        estim.Z̃[(nx̃+(j-1)*nx̂+1):(nx̃+(j-1)*nx̂+nx)] .= x0s
    end
    return nothing
end
function reset_warmstart!(estim::MovingHorizonEstimator, ::TrapezoidalCollocation)
    model = estim.model
    nx, nx̂, nx̃, nε, He, na = model.nx, estim.nx̂, estim.nx̃, estim.nε, estim.He, get_na(model)
    as_0 = get_as_0(model)
    x0s  = model.buffer.x
    x0s .= model.xs_0 .- model.xop
    a0s  = as_0
    estim.Z̃ .= 0
    estim.Z̃[nε+1:nε+nx] = x0s
    estim.Z̃[(nx̃ + nx̂*He + 1):(nx̃ + nx̂*He + na)] = a0s
    for j in 1:He
        estim.Z̃[(nx̃+(j-1)*nx̂+1):(nx̃+(j-1)*nx̂+nx)] .= x0s
        estim.Z̃[(nx̃+nx̂*He+na+(j-1)*na+1):(nx̃+nx̂*He+na+j*na)] .= a0s
        estim.Z̃[(nx̃+nx̂*He+na+na*He+(j-1)*na+1):(nx̃+nx̂*He+na+na*He+j*na)] .= a0s
    end
    return nothing
end
function reset_warmstart!(estim::MovingHorizonEstimator, ::OrthogonalCollocation)
    model = estim.model
    nx, nx̂, nx̃, nε, He, na = model.nx, estim.nx̂, estim.nx̃, estim.nε, estim.He, get_na(model)
    as_0 = get_as_0(model)
    x0s  = model.buffer.x
    x0s .= model.xs_0 .- model.xop
    a0s  = as_0
    estim.Z̃ .= 0
    estim.Z̃[nε+1:nε+nx] = x0s
    estim.Z̃[(nx̃ + nx̂*He + 1):(nx̃ + nx̂*He + na)] = a0s
    for j in 1:He
        estim.Z̃[(nx̃+(j-1)*nx̂+1):(nx̃+(j-1)*nx̂+nx)] .= x0s
        estim.Z̃[(nx̃+nx̂*He+na+(j-1)*na+1):(nx̃+nx̂*He+na+j*na)] .= a0s
        estim.Z̃[(nx̃+nx̂*He+na+na*He+(j-1)*nx+1):(nx̃+nx̂*He+na+na*He+j*nx)] .= x0s
        estim.Z̃[(nx̃+nx̂*He+na+na*He+nx*He+(j-1)*na+1):(nx̃+nx̂*He+na+na*He+nx*He+j*na)] .= a0s
    end
    return nothing
end
function reset_warmstart!(estim::MovingHorizonEstimator, ::SingleShooting)
    model = estim.model
    nx, nε = model.nx, estim.nε
    x0s  = model.buffer.x
    x0s .= model.xs_0 .- model.xop
    estim.Z̃ .= 0
    estim.Z̃[nε+1:nε+nx] = x0s
    return nothing
end

@doc raw"""
    set_warmstart_mhe!(
        estim::MovingHorizonEstimator, transcription::SingleShooting, Z̃var
    ) -> Z̃s

Set and return the warm-start value of `Z̃var` for [`MovingHorizonEstimator`](@ref).

If supported by `estim.optim` and based a [`SingleShooting`](@ref) transcription, it
warm-starts the solver at the value provided in Extended Help.

# Extended Help
!!! details "Extended Help"
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
        \mathbf{0_ŵ}
    \end{bmatrix}
    ```
    where ``ε_{k-1}`` and ``\mathbf{ŵ}(k-j|k-1)`` are respectively the slack variable and the
    process noise estimates computed at the last time step ``k-1``. The vector 
    ``\mathbf{x̂_0^†}(k-N_k+p)`` is the deviation vector of the state at the arrival estimated
    at time ``k-N_k``. If the objective function is not finite at this point, all the process
    noises ``\mathbf{ŵ}_{k-1}(k-j)`` are warm-started at zeros. See the Extended Help of 
    [`SingleShooting`](@ref) for the defintion of vector ``\mathbf{0_ŵ}``. The method mutates
    all the arguments.
"""
function set_warmstart_mhe!(
    estim::MovingHorizonEstimator{NT}, transcription::SingleShooting, Z̃var
) where NT<:Real
    model, buffer = estim.model, estim.buffer
    nk̄ = get_nk̄(estim.model, transcription)
    nε, nx̂, nŵ, He, Nk = estim.nε, estim.nx̂, estim.nx̂, estim.He, estim.Nk[]
    nx̃, nŴ = nε + nx̂, nŵ*He
    Z̃s = estim.buffer.Z̃
    # --- slack variable ε ---
    estim.nε == 1 && (Z̃s[begin] = estim.Z̃[begin])
    # --- arrival state estimate x̂0arr ---
    Z̃s[nε+1:nx̃] = estim.x̂0arr_old
    # --- process noise estimates Ŵ ---
    Z̃s[(nx̃+1):(nx̃+nŴ-nŵ)] .= @views estim.Z̃[(nx̃+nŵ+1):(nx̃+nŴ)]
    Z̃s[(nx̃+nŴ-nŵ+1):end]  .= 0
    # --- verify definiteness of objective function ---
    x̄, â0arr = buffer.x̂, buffer.â
    V̂, Ŵ, X̂0, Â0, Û0, Ŷ0 = buffer.V̂, buffer.Ŵ, buffer.X̂, buffer.Â, buffer.U, buffer.Ŷ
    K = Vector{NT}(undef, nk̄*Nk) # TODO: remove the allocation
    x̂0arr = estim.x̂0arr_old
    x̄ .= 0 # x̂0arr == x̂arr_old implies the error at arrival x̄ is zero
    getŴ!(Ŵ, estim, transcription, Z̃s) 
    predict_mhe!(
        V̂, X̂0, Â0, Û0, K, Ŷ0, estim, model, estim.transcription, x̂0arr, â0arr, Ŵ, Z̃s
    )
    Js = obj_nonlinprog(estim, model, x̄, V̂, Ŵ, Z̃s)
    if !isfinite(Js)
        Z̃s[nx̃+1:end] .= 0 # Ŵ = 0
    end
    # --- unused variable in Z̃ (applied only when Nk < He) ---
    # We force the update of the NLP gradient and jacobian by warm-starting the unused 
    # variable of Ŵ in Z̃ at 1. Since Ŵ is initialized with 0s, at least 1 variable in Z̃s
    # will be inevitably different at the following time step.
    Z̃s[nx̃+nŵ*Nk+1:end] .= 1
    JuMP.set_start_value.(Z̃var, Z̃s)
    return Z̃s
end

@doc raw"""
    set_warmstart_mhe!(
        estim::MovingHorizonEstimator, transcription::OrthogonalCollocation, Z̃var
    ) -> Z̃s

Do the same but for [`OrthogonalCollocation`](@ref).

The warm-starting value is provided in Extended Help.

# Extended Help
!!! details "Extended Help"
    It warm-starts the solver at:
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
        \mathbf{0_x̂}                    \\
        \mathbf{â_0}(k-N_k+p|k-1)       \\
        \mathbf{â_0}(k-N_k+p+1|k-1)     \\
        \mathbf{â_0}(k-N_k+p+2|k-1)     \\
        \vdots
        \mathbf{â_0}(k-p-2|k-1)         \\
        \mathbf{â_0}(k-p-1|k-1)         \\
        \mathbf{a_0}(k-p-1|k-1)         \\
        \mathbf{0_â}                    \\
        \mathbf{k̄}(k-N_k+p+0|k-1)       \\
        \mathbf{k̄}(k-N_k+p+1|k-1)       \\
        \vdots                          \\
        \mathbf{k̄}(k+p-3|k-1)           \\
        \mathbf{k̄}(k+p-2|k-1)           \\
        \mathbf{k̄}(k+p-2|k-1)           \\
        \mathbf{0_k̄}                    \\
        \mathbf{ā}(k-N_k+p+0|k-1)       \\
        \mathbf{ā}(k-N_k+p+1|k-1)       \\
        \vdots                          \\
        \mathbf{ā}(k+p-3|k-1)           \\
        \mathbf{ā}(k+p-2|k-1)           \\
        \mathbf{ā}(k+p-2|k-1)           \\
        \mathbf{0_ā}                    \\
        \mathbf{ŵ}(k-N_k+p+0|k-1)       \\
        \mathbf{ŵ}(k-N_k+p+1|k-1)       \\
        \vdots                          \\
        \mathbf{ŵ}(k+p-3|k-1)           \\
        \mathbf{ŵ}(k+p-2|k-1)           \\
        \mathbf{0}                      \\
        \mathbf{0_ŵ}
    \end{bmatrix}
    ```
    where ``\mathbf{x̂_0}(k-j|k-1)`` is the predicted state for time ``k-j`` computed at the
    last control period ``k-1``, expressed as a deviation from the operating point 
    ``\mathbf{x̂_{op}}``. The vector ``\mathbf{k̄}(k-j|k-1)`` and ``\mathbf{ā}(k-j|k-1)
    include the ``n_o`` intermediate stage predictions for the interval ``k-j``, and is also
    computed at the last control period. See the Extended Help of [`MultipleShooting`](@ref)
    and [`OrthogonalCollocation`](@ref) for the defintion of vectors ``\mathbf{0_x̂}``,
    ``\mathbf{0_â}``, ``\mathbf{0_k}`` and ``\mathbf{0_ŵ}``.
"""
function set_warmstart_mhe!(
    estim::MovingHorizonEstimator{NT}, transcription::OrthogonalCollocation, Z̃var
) where NT<:Real
    model, buffer = estim.model, estim.buffer
    na = get_na(model)
    nk̄, nā = get_nk̄(estim.model, transcription), get_nā(estim.model, transcription)
    nε, nx̂, nŵ, He, Nk = estim.nε, estim.nx̂, estim.nx̂, estim.He, estim.Nk[]
    nx̃, nŴ, nX̂, nA, nK̄, nĀ = nε + nx̂, nŵ*He, nx̂*He, na*He, nk̄*He, nā*He
    Z̃s = estim.buffer.Z̃
    # --- slack variable ε ---
    estim.nε == 1 && (Z̃s[begin] = estim.Z̃[begin])
    # --- arrival state estimate x̂0arr ---
    i_base = nε
    Z̃s[i_base+1:nx̃] = estim.x̂0arr_old
    # --- state estimates X̂0 ---
    i_base = nx̃
    Z̃s[(i_base+1):(i_base+nX̂-nx̂)]    .= @views estim.Z̃[(i_base+nx̂+1):(i_base+nX̂)]
    Z̃s[(i_base+nX̂-nx̂+1):(i_base+nX̂)] .= @views estim.Z̃[(i_base+nX̂-nx̂+1):(i_base+nX̂)]
    # --- algebraic variables â0arr and Â0 ---
    i_base = nx̃ + nX̂ 
    Z̃s[(i_base+1):(i_base+nA)]       .= @views estim.Z̃[(i_base+na+1):(i_base+na+nA)]
    Z̃s[(i_base+nA+1):(i_base+na+nA)] .= @views estim.Z̃[(i_base+nA+1):(i_base+na+nA)]
    # --- deterministic states at collocation points K̄ --- 
    i_base = nx̃ + nX̂ + na + nA
    Z̃s[(i_base+1):(i_base+nK̄-nk̄)]    .= @views estim.Z̃[(i_base+nk̄+1):(i_base+nK̄)]
    Z̃s[(i_base+nK̄-nk̄+1):(i_base+nK̄)] .= @views estim.Z̃[(i_base+nK̄-nk̄+1):(i_base+nK̄)]
    # --- algebraic variables at collocation points Ā --- 
    i_base = nx̃ + nX̂ + na + nA + nK̄
    Z̃s[(i_base+1):(i_base+nĀ-nā)]    .= @views estim.Z̃[(i_base+nā+1):(i_base+nĀ)]
    Z̃s[(i_base+nĀ-nā+1):(i_base+nĀ)] .= @views estim.Z̃[(i_base+nĀ-nā+1):(i_base+nĀ)]
    # --- process noise estimates Ŵ ---
    i_base = nx̃ + nX̂ + na + nA + nK̄ + nĀ
    Z̃s[(i_base+1):(i_base+nŴ-nŵ)]   .= @views estim.Z̃[(i_base+nŵ+1):(i_base+nŴ)]
    Z̃s[(i_base+nŴ-nŵ+1):end]        .= 0
    # --- verify definiteness of objective function --- 
    x̄, â0arr = buffer.x̂, buffer.â
    V̂, Ŵ, X̂0, Â0, Û0, Ŷ0 = buffer.V̂, buffer.Ŵ, buffer.X̂, buffer.Â, buffer.U, buffer.Ŷ
    K̄ = Vector{NT}(undef, nk̄*Nk) # TODO: remove the allocation
    x̂0arr = estim.x̂0arr_old
    â0arr = getâ0arr!(â0arr, estim, estim.transcription, Z̃s)  
    x̄ .= 0 # x̂0arr == x̂arr_old implies the error at arrival x̄ is zero
    getŴ!(Ŵ, estim, transcription, Z̃s)
    predict_mhe!(V̂, X̂0, Â0, Û0, K̄, Ŷ0, estim, model, estim.transcription, x̂0arr, â0arr, Ŵ, Z̃s)
    Js = obj_nonlinprog(estim, model, x̄, V̂, Ŵ, Z̃s)
    if !isfinite(Js)
        Z̃s[nx̃+nX̂+nK̄+1:end] .= 0 # Ŵ = 0
    end
    # --- unused variable in Z̃ (applied only when Nk < He) ---
    # We force the update of the NLP gradient and jacobian by warm-starting the unused 
    # variable of Ŵ in Z̃ at 1. Since Ŵ is initialized with 0s, at least 1 variable in Z̃s
    # will be inevitably different at the following time step.
    Z̃s[nx̃+nX̂+nK̄+nŵ*Nk+1:end] .= 1
    JuMP.set_start_value.(Z̃var, Z̃s)
    return Z̃s
end

@doc raw"""
    set_warmstart_mhe!(
        estim::MovingHorizonEstimator, transcription::TranscriptionMethod, Z̃var
    ) -> Z̃s

Do the same but for other transcription [`TranscriptionMethod`](@ref).

The warm-starting value is provided in Extended Help.

# Extended Help
!!! details "Extended Help"
    It warm-starts the solver at:
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
        \mathbf{0_x̂}                    \\
        \mathbf{â_0}(k-N_k+p|k-1)       \\
        \mathbf{â_0}(k-N_k+p+1|k-1)     \\
        \mathbf{â_0}(k-N_k+p+2|k-1)     \\
        \vdots
        \mathbf{â_0}(k-p-2|k-1)         \\
        \mathbf{â_0}(k-p-1|k-1)         \\
        \mathbf{â_0}(k-p-1|k-1)         \\
        \mathbf{0_â}                    \\
        \mathbf{ŵ}(k-N_k+p+0|k-1)       \\
        \mathbf{ŵ}(k-N_k+p+1|k-1)       \\
        \vdots                          \\
        \mathbf{ŵ}(k+p-3|k-1)           \\
        \mathbf{ŵ}(k+p-2|k-1)           \\
        \mathbf{0}                      \\
        \mathbf{0_ŵ}
    \end{bmatrix}
    ```
"""
function set_warmstart_mhe!(
    estim::MovingHorizonEstimator{NT}, transcription::TranscriptionMethod, Z̃var
) where NT<:Real
    model, buffer = estim.model, estim.buffer
    na = get_na(model)
    nk̄ = get_nk̄(estim.model, transcription)
    nε, nx̂, nŵ, He, Nk = estim.nε, estim.nx̂, estim.nx̂, estim.He, estim.Nk[]
    nx̃, nŴ, nX̂ = nε + nx̂, nŵ*He, nx̂*He
    Z̃s = estim.buffer.Z̃
    # --- slack variable ε ---
    estim.nε == 1 && (Z̃s[begin] = estim.Z̃[begin])
    # --- arrival state estimate x̂0arr ---
    i_base = nε
    Z̃s[i_base+1:nx̃] = estim.x̂0arr_old
    # --- state estimates X̂0 ---
    i_base = nx̃
    Z̃s[(i_base+1):(i_base+nX̂-nx̂)]    .= @views estim.Z̃[(i_base+nx̂+1):(i_base+nX̂)]
    Z̃s[(i_base+nX̂-nx̂+1):(i_base+nX̂)] .= @views estim.Z̃[(i_base+nX̂-nx̂+1):(i_base+nX̂)]
    # --- algebraic variables â0arr and Â0 ---
    i_base = nx̃ + nX̂ 
    Z̃s[(i_base+1):(i_base+nA)]       .= @views estim.Z̃[(i_base+na+1):(i_base+na+nA)]
    Z̃s[(i_base+nA+1):(i_base+na+nA)] .= @views estim.Z̃[(i_base+nA+1):(i_base+na+nA)]
    # --- algebraic variables Ā --- 
    i_base = nx̃ + nX̂ + na + nA
    Z̃s[(i_base+1):(i_base+nĀ-nā)]    .= @views estim.Z̃[(i_base+nā+1):(i_base+nĀ)]
    Z̃s[(i_base+nĀ-nā+1):(i_base+nĀ)] .= @views estim.Z̃[(i_base+nĀ-nā+1):(i_base+nĀ)]
    # --- process noise estimates Ŵ ---
    i_base = nx̃ + nX̂ + na + nA + nĀ
    Z̃s[(i_base+1):(i_base+nŴ-nŵ)]   .= @views estim.Z̃[(i_base+nŵ+1):(i_base+nŴ)]
    Z̃s[(i_base+nŴ-nŵ+1):end]        .= 0
    # --- verify definiteness of objective function ---
    x̄, â0arr = buffer.x̂, buffer.â
    V̂, Ŵ, X̂0, Â0, Û0, Ŷ0 = buffer.V̂, buffer.Ŵ, buffer.X̂, buffer.Â, buffer.U, buffer.Ŷ
    K̄ = Vector{NT}(undef, nk̄*Nk) # TODO: remove the allocation
    x̂0arr = estim.x̂0arr_old
    â0arr = getâ0arr!(â0arr, estim, transcription, Z̃s)
    x̄ .= 0 # x̂0arr == x̂arr_old implies the error at arrival x̄ is zero
    getŴ!(Ŵ, estim, transcription, Z̃s)
    predict_mhe!(
        V̂, X̂0, Â0, Û0, K̄, Ŷ0, estim, model, estim.transcription, x̂0arr, â0arr, Ŵ, Z̃s
    )
    Js = obj_nonlinprog(estim, model, x̄, V̂, Ŵ, Z̃s)
    if !isfinite(Js)
        Z̃s[nx̃+nX̂+1:end] .= 0 # Ŵ = 0
    end
    # --- unused variable in Z̃ (applied only when Nk < He) ---
    # We force the update of the NLP gradient and jacobian by warm-starting the unused 
    # variable of Ŵ in Z̃ at 1. Since Ŵ is initialized with 0s, at least 1 variable in Z̃s
    # will be inevitably different at the following time step.
    Z̃s[nx̃+nX̂+nŵ*Nk+1:end] .= 1
    JuMP.set_start_value.(Z̃var, Z̃s)
    return Z̃s
end

"Fill the unused decision variables in `Z̃` with `0`s (only when `Nk < He`)."
function fill0unused!(Z̃, estim::MovingHorizonEstimator, ::SingleShooting)
    nŵ, nx̃, Nk =  estim.nx̂, estim.nx̃, estim.Nk[]
    Z̃[(1 + nx̃ + nŵ*Nk):end] .= 0 # unused vars after Ŵ vector
    return nothing
end
function fill0unused!(Z̃, estim::MovingHorizonEstimator, ::TranscriptionMethod)
    nŵ, nx̂, nx̃, He, Nk =  estim.nx̂, estim.nx̂, estim.nx̃, estim.He, estim.Nk[]
    na = get_na(estim.model)
    nx̃_nX̂_He          = nx̃ + nx̂*He
    nx̃_nX̂_na_nA_He    = nx̃ + nx̂*He + na + na*He
    nx̃_nX̂_na_nA_nĀ_He = nx̃ + nx̂*He + na + na*He + na*He
    Z̃[(1 + nx̃ + nx̂*Nk):(nx̃_nX̂_He)]                      .= 0 # unused vars after X̂0 vector
    Z̃[(1 + nx̃_nX̂_He + na + na*Nk):(nx̃_nX̂_na_nA_He)]     .= 0 # unused vars after Â0 vector
    Z̃[(1 + nx̃_nX̂_na_nA_He + na*Nk):(nx̃_nX̂_na_nA_nĀ_He)] .= 0 # unused vars after Ā vector
    Z̃[(1 + nx̃_nX̂_na_nA_nĀ_He + nŵ*Nk):end]              .= 0 # unused vars after Ŵ vector
    return nothing
end
function fill0unused!(Z̃, estim::MovingHorizonEstimator, transcription::OrthogonalCollocation)
    nŵ, nx̂, nx̃, He, Nk =  estim.nx̂, estim.nx̂, estim.nx̃, estim.He, estim.Nk[]
    na = get_na(estim.model)
    nk̄ = get_nk̄(estim.model, transcription)
    nā = transcription.no*na
    nx̃_nX̂_He             = nx̃ + nx̂*He
    nx̃_nX̂_na_nA_He       = nx̃ + nx̂*He + na + na*He
    nx̃_nX̂_na_nA_nK̄_He    = nx̃ + nx̂*He + na + na*He + nk̄*He
    nx̃_nX̂_na_nA_nK̄_nĀ_He = nx̃ + nx̂*He + na + na*He + nk̄*He + nā*He
    Z̃[(1 + nx̃ + nx̂*Nk):(nx̃_nX̂_He)]                            .= 0 # unused vars after X̂0 vector
    Z̃[(1 + nx̃_nX̂_He + na + na*Nk):(nx̃_nX̂_na_nA_He)]           .= 0 # unused vars after Â0 vector
    Z̃[(1 + nx̃_nX̂_na_nA_He + nk̄*Nk):(nx̃_nX̂_na_nA_nK̄_He)]       .= 0 # unused vars after K̄ vector
    Z̃[(1 + nx̃_nX̂_na_nA_nK̄_He + nā*Nk):(nx̃_nX̂_na_nA_nK̄_nĀ_He)] .= 0 # unused vars after Ā vector
    Z̃[(1 + nx̃_nX̂_na_nA_nK̄_nĀ_He + nŵ*Nk):end]                 .= 0 # unused vars after Ŵ vector
    return nothing
end

@doc raw"""
    predict_mhe!(
        V̂, X̂0, Â0, Û0, K̄, Ŷ0, 
        estim::MovingHorizonEstimator, model::LinModel, transcription::TranscriptionMethod, 
        x̂0arr, â0arr, Ŵ, Z̃ 
    ) -> V̂, X̂0, Â0

Compute the `V̂` vector and `X̂0` vectors for the `MovingHorizonEstimator` and `LinModel`.

The function mutates `V̂` and `X̂0` vector arguments. The vector `V̂` is the estimated sensor
noises from ``k-N_k+1`` to ``k``. The `X̂0` vector is estimated states from ``k-N_k+2`` to 
``k+1``. The computations are (by truncating the matrices when `N_k < H_e`):
```math
\begin{aligned}
\mathbf{V̂}   &= \mathbf{Ẽ Z̃}   + \mathbf{F}     \\
\mathbf{X̂_0} &= \mathbf{Ẽ_X̂ Z̃} + \mathbf{F_X̂}
\end{aligned}
```
The vector `Â0` is ignored and returned unchanged.
"""
function predict_mhe!(
    V̂, X̂0, Â0, _ , _ , _ ,
    estim::MovingHorizonEstimator, ::LinModel, ::TranscriptionMethod, 
    _ , _ , _ , Z̃
)
    nε, Nk = estim.nε, estim.Nk[]
    if Nk < estim.He
        # avoid views since allocations only when Nk < He and we want fast mul!:
        nX̂, nŴ, nYm = estim.nx̂*Nk, estim.nx̂*Nk, estim.nym*Nk
        nZ̃ = nε + estim.nx̂ + nŴ
        Ẽ,  F  = estim.Ẽ[1:nYm, 1:nZ̃],     estim.F[1:nYm]
        ẼX̂, FX̂ = estim.con.ẼX̂[1:nX̂, 1:nZ̃], estim.con.FX̂[1:nX̂]
        Z̃ = Z̃[1:nZ̃]
        V̂_res, X̂0_res = @views V̂[1:nYm], X̂0[1:nX̂]
    else
        Ẽ, F = estim.Ẽ, estim.F
        ẼX̂, FX̂ = estim.con.ẼX̂, estim.con.FX̂
        V̂_res, X̂0_res = V̂, X̂0
    end
    V̂_res  .= mul!(V̂_res, Ẽ, Z̃) .+ F
    X̂0_res .= mul!(X̂0_res, ẼX̂, Z̃) .+ FX̂
    return V̂, X̂0, Â0
end

@doc raw"""
    predict_mhe!(
        V̂, X̂0, Â0, Û0, K̄, Ŷ0, 
        estim::MovingHorizonEstimator, model::NonLinModel, ::SingleShooting, 
        x̂0arr, â0arr, Ŵ, Z̃ 
    ) -> V̂, X̂0, Â0

Compute the vectors when `model` is a [`NonLinModel`](@ref) with [`SingleShooting`](@ref).

The function mutates `V̂`, `X̂0`, `Û0`, `K` and `Ŷ0` vector arguments. The vector `Â0` is
ignored and returned unchanged. The augmented model of [`f̂!`](@ref) and [`ĥ!`](@ref) is
called recursively in a `for` loop from ``j=1`` to ``N_k``, and by adding the estimated
process noise ``\mathbf{ŵ}``.
"""
function predict_mhe!(
    V̂, X̂0, Â0, Û0, K̄, Ŷ0, 
    estim::MovingHorizonEstimator, model::NonLinModel, ::SingleShooting, 
    x̂0arr, _ , Ŵ, _ 
)
    nu, nd, ny, nk̄ = model.nu, model.nd, model.ny, model.nk̄
    nx̂, nŵ, nym, Nk = estim.nx̂, estim.nx̂, estim.nym, estim.Nk[]
    p = estim.direct ? 0 : 1
    x̂0 = @views x̂0arr[1:nx̂]
    for j=1:Nk
        u0      = @views  estim.U0[(1+nu*(j-1)):(nu*j)]
        d0      = @views  estim.D0[(1+nd*(j+p-1)):(nd*(j+p))]
        ŵ       = @views         Ŵ[(1+nŵ*(j-1)):(nŵ*j)]
        k̄       = @views         K̄[(1+nk̄*(j-1)):(nk̄*j)]
        û0      = @views        Û0[(1+nu*(j-1)):(nu*j)]
        x̂0next  = @views        X̂0[(1+nx̂*(j-1)):(nx̂*j)]
        f̂!(x̂0next, û0, k̄, estim, model, x̂0, u0, d0)
        x̂0next .+= ŵ
        if estim.direct
            ŷ0next  = @views        Ŷ0[(1 +  ny*(j-1)):(ny*j)]
            y0nextm = @views estim.Y0m[(1 + nym*(j-1)):(nym*j)]
            v̂next   = @views         V̂[(1 + nym*(j-1)):(nym*j)]
            d0next  = @views  estim.D0[(1 + nd*j):(nd*(j+1))]
            ĥ!(ŷ0next, estim, model, x̂0next, d0next)
            ŷ0nextm = @views ŷ0next[estim.i_ym]
            if any(isnan, y0nextm) # nan in Y0m: y0m=ŷ0m => associated v̂ value = 0
                y0nextm = [isnan(y) ? ŷ : y for (y, ŷ) in zip(y0nextm, ŷ0nextm)]
            end
            v̂next .= y0nextm .- ŷ0nextm
        else
            ŷ0      = @views        Ŷ0[(1 +  ny*(j-1)):(ny*j)]
            y0m     = @views estim.Y0m[(1 + nym*(j-1)):(nym*j)]
            v̂       = @views         V̂[(1 + nym*(j-1)):(nym*j)]
            ĥ!(ŷ0, estim, model, x̂0, d0)
            ŷ0m = @views ŷ0[estim.i_ym]
            if any(isnan, y0m)
                y0m = [isnan(y) ? ŷ : y for (y, ŷ) in zip(y0m, ŷ0m)]
            end
            v̂ .= y0m .- ŷ0m
        end
        x̂0 = x̂0next
    end
    if Nk < estim.He  # fill unused values with 0s for tracer sparsity detection:
        V̂[nym*Nk+1:end] .= 0
        X̂0[nx̂*Nk+1:end] .= 0
    end
    return V̂, X̂0, Â0
end

@doc raw"""
    predict_mhe!(
        V̂, X̂0, Â0, _ , _ , Ŷ0, 
        estim::MovingHorizonEstimator, model::SimModel, ::TranscriptionMethod, 
        x̂0arr, â0arr, _ , Z̃ 
    ) -> V̂, X̂0, Â0

Compute the vectors for all other cases.

The function mutates `V̂`, `X̂0`, and `Ŷ0` vector arguments. The augmented output function 
[`ĥ!`](@ref) is called multiple times in a `for` loop from ``j=1`` to ``N_k``.
"""
function predict_mhe!(
    V̂, X̂0, Â0, _ , _ , Ŷ0, 
    estim::MovingHorizonEstimator, model::SimModel, transcription::TranscriptionMethod, 
    x̂0arr, â0arr, _ , Z̃ 
)
    nd, ny, na = model.nd, model.ny, get_na(model)
    nx̂, nx̃, nym, Nk = estim.nx̂, estim.nx̃, estim.nym, estim.Nk[]
    nx̃_nX̂ = nx̃ + nx̂*estim.He
    h_threads = transcription.h_threads
    X̂0[1:nx̂*Nk] .= @views Z̃[(1 + nx̃):(nx̃ + nx̂*Nk)]                   # skip x0arr
    Â0[1:na*Nk] .= @views Z̃[(1 + nx̃_nX̂ + na):(nx̃_nX̂ + na + na*Nk)]   # skip â0arr
    @threadsif h_threads for j=1:Nk
        if estim.direct
            x̂0 = @views X̂0[(1+nx̂*(j-1)):(nx̂*j)]
            â0 = @views Â0[(1+na*(j-1)):(na*j)]
        else
            x̂0 = @views j < 2 ? x̂0arr[1:nx̂] : X̂0[(1+nx̂*(j-2)):(nx̂*(j-1))]
            â0 = @views j < 2 ? â0arr[1:na] : Â0[(1+na*(j-2)):(na*(j-1))]
        end
        d0  = @views  estim.D0[(1+nd*j):(nd*(j+1))] # the 1st nd elements are not needed here
        ŷ0  = @views        Ŷ0[(1 +  ny*(j-1)):(ny*j)]
        v̂   = @views         V̂[(1 + nym*(j-1)):(nym*j)]
        y0m = @views estim.Y0m[(1 + nym*(j-1)):(nym*j)]
        ĥ!(ŷ0, estim, model, x̂0, â0, d0)
        ŷ0m = @views ŷ0[estim.i_ym]
        if any(isnan, y0m) # nan in Y0m: y0m=ŷ0m => associated v̂ value = 0
            y0m = [isnan(y) ? ŷ : y for (y, ŷ) in zip(y0m, ŷ0m)]
        end
        v̂ .= y0m .- ŷ0m
    end
    if Nk < estim.He  # fill unused values with 0s for tracer sparsity detection:
        V̂[nym*Nk+1:end] .= 0
        X̂0[nx̂*Nk+1:end] .= 0
        Â0[na*Nk+1:end] .= 0
    end
    return V̂, X̂0, Â0
end

"""
    con_nonlinprog_mhe!(
        g, estim::MovingHorizonEstimator, model::NonLinModel, ::SingleShooting, X̂0, V̂, gc, ε
    ) -> g

Nonlinear MHE constraint when `model` is [`NonLinModel`](@ref) with [`SingleShooting`](@ref).

The method mutates the `g` vectors in argument and returns it. The estimated states,
estimated sensor noises and custom constraints are included in the `g` vector.
"""
function con_nonlinprog_mhe!(
    g, estim::MovingHorizonEstimator, ::NonLinModel, ::SingleShooting, X̂0, V̂, gc, ε
)
    nX̂con, nX̂ = length(estim.con.X̂0min), estim.nx̂ *estim.Nk[]
    nV̂con, nV̂ = length(estim.con.V̂min),  estim.nym*estim.Nk[]
    for i in eachindex(g)
        estim.con.i_g[i] || continue
        if i ≤ nX̂con
            j = i
            jcon = nX̂con-nX̂+j
            g[i] = j > nX̂ ? 0 : estim.con.X̂0min[jcon] - X̂0[j] - ε*estim.con.C_x̂min[jcon]
        elseif i ≤ 2nX̂con
            j = i - nX̂con
            jcon = nX̂con-nX̂+j
            g[i] = j > nX̂ ? 0 : X̂0[j] - estim.con.X̂0max[jcon] - ε*estim.con.C_x̂max[jcon]
        elseif i ≤ 2nX̂con + nV̂con
            j = i - 2nX̂con
            jcon = nV̂con-nV̂+j
            g[i] = j > nV̂ ? 0 : estim.con.V̂min[jcon] - V̂[j] - ε*estim.con.C_v̂min[jcon]
        elseif i ≤ 2nX̂con + 2nV̂con
            j = i - 2nX̂con - nV̂con
            jcon = nV̂con-nV̂+j
            g[i] = j > nV̂ ? 0 : V̂[j] - estim.con.V̂max[jcon] - ε*estim.con.C_v̂max[jcon]
        else
            j = i - 2nX̂con - 2nV̂con
            g[i] = gc[j]
        end
    end
    return g
end

"""
    con_nonlinprog_mhe!(
        g, ::MovingHorizonEstimator, ::LinModel, ::TranscriptionMethod, _ , _ , gc, _ 
    ) -> g

Compute the same but for [`LinModel`](@ref). 

The nonlinear custom inequality constraints in `gc` are the only nonlinear constraints
for this case. 
"""
function con_nonlinprog_mhe!(
    g, ::MovingHorizonEstimator, ::LinModel, ::TranscriptionMethod, _ , _ , gc , _ 
)
    for i in eachindex(g)
        g[i] = gc[i]
    end
    return g
end

"""
    con_nonlinprog_mhe!(
        g, estim::MovingHorizonEstimator, ::SimModel, ::TranscriptionMethod, _ , V̂, gc, ε
    ) -> g

Nonlinear MHE constraint for all other cases.

The method mutates the `g` vectors in argument and returns it. The estimated sensor noises 
and custom constraints are included in the `g` vector.
"""
function con_nonlinprog_mhe!(
    g, estim::MovingHorizonEstimator, ::SimModel, ::TranscriptionMethod, _ , V̂, gc, ε
)
    nV̂con, nV̂ = length(estim.con.V̂min),  estim.nym*estim.Nk[]
    for i in eachindex(g)
        estim.con.i_g[i] || continue
        if i ≤ nV̂con
            j = i
            jcon = nV̂con-nV̂+j
            g[i] = j > nV̂ ? 0 : estim.con.V̂min[jcon] - V̂[j] - ε*estim.con.C_v̂min[jcon]
        elseif i ≤ 2nV̂con
            j = i - nV̂con
            jcon = nV̂con-nV̂+j
            g[i] = j > nV̂ ? 0 : V̂[j] - estim.con.V̂max[jcon] - ε*estim.con.C_v̂max[jcon]
        else
            j = i - 2nV̂con
            g[i] = gc[j]
        end
    end
    return g
end

@doc raw"""
    con_nonlinprogeq_mhe!(
        geq, X̂0, _ , Û0, K̄,
        estim::MovingHorizonEstimator, model::NonLinModel, ::MultipleShooting, 
        x̂0arr, _ , Ŵ, Z̃
    ) -> geq

Nonlinear MHE equality constrains for [`NonLinModel`](@ref) and [`MultipleShooting`](@ref).

The method mutates the `geq`, `X̂0`, `Û0` and `K` vectors in argument. By introducing 
the integer ``ℓ = k - N_k + p`` to shorten the notation, the defects of the estimated states
are computed with:
```math
\mathbf{ŝ}(ℓ+j+1) = \mathbf{f̂}\Big(\mathbf{x̂_0}(ℓ+j), \mathbf{u_0}(ℓ+j), \mathbf{d_0}(ℓ+j)\Big) 
                           + \mathbf{ŵ}(ℓ+j) - \mathbf{x̂_0}(ℓ+j+1)
```
for ``j = 0, 1, ... , N_k-1`` and in which the augmented state vectors ``\mathbf{x̂_0}`` are 
extracted from the decision variable `Z̃`. The function ``\mathbf{f̂}`` is defined at [`f̂!`](@ref).
"""
function con_nonlinprogeq_mhe!(
    geq, X̂0, _ , Û0, K̄,
    estim::MovingHorizonEstimator, model::NonLinModel, transcription::MultipleShooting, 
    x̂0arr, _ , Ŵ, Z̃
)
    nu, nx, nd, nk̄ = model.nu, model.nx, model.nd, model.nk̄
    nx̂, nx̃, nxs, nŵ, He = estim.nx̂, estim.nx̃, estim.nxs, estim.nx̂, estim.He
    Nk = estim.Nk[]
    f_threads = transcription.f_threads
    nw = nŵ - nxs
    p = estim.direct ? 0 : 1
    X̂0_Z̃ = @views Z̃[(nx̃+1):(nx̃+nx̂*He)]
    Û0 = disturbedinput!(Û0, estim, x̂0arr, X̂0_Z̃, estim.U0)
    @threadsif f_threads for j=1:Nk
        if j < 2
            x̂d_Z̃ = @views x̂0arr[1:nx]
        else
            x̂d_Z̃ = @views X̂0_Z̃[(1 + nx̂*(j-2)):(nx̂*(j-2) + nx)]
        end
        d0       = @views   estim.D0[(1 + nd*(j+p-1)):(nd*(j+p))]
        k̄        = @views          K̄[(1 + nk̄*(j-1)):(nk̄*j)]
        û0       = @views         Û0[(1 + nu*(j-1)):(nu*j)]
        ŵd       = @views          Ŵ[(1 + nŵ*(j-1)):(nŵ*(j-1) + nw)]
        x̂dnext   = @views         X̂0[(1 + nx̂*(j-1)):(nx̂*(j-1) + nx)]
        x̂dnext_Z̃ = @views       X̂0_Z̃[(1 + nx̂*(j-1)):(nx̂*(j-1) + nx)]
        ŝdnext   = @views        geq[(1 + nx*(j-1)):(nx*j)]
        f!(x̂dnext, k̄, model, x̂d_Z̃, û0, d0, model.p)
        x̂dnext .+= ŵd
        ŝdnext  .= @. x̂dnext - x̂dnext_Z̃
    end
    Nk < He && (geq[nx*Nk+1:end] .= 0)
    return geq
end

@doc raw"""
    con_nonlinprogeq_mhe!(
        geq, _ , _ , Û0, K̄,
        estim::MovingHorizonEstimator, model::SimModel, ::TrapezoidalCollocation, 
        x̂0arr, â0arr, Ŵ, Z̃
    ) -> geq

Nonlinear MHE equality constrains for [`SimModel`](@ref) and [`TrapezoidalCollocation`](@ref).

By introducing the integer ``ℓ = k - N_k + p`` to shorten the notation, the deterministic
state defects are computed with:
```math
\mathbf{ŝ_k}(ℓ+j) = \mathbf{x̂_d}(ℓ+j) + 0.5 T_s [\mathbf{k̇_0}(ℓ+j) + \mathbf{k̇_1}(ℓ+j)] 
                      - \mathbf{x̂_d}(ℓ+j+1) + \mathbf{ŵ_d}(ℓ+j)                                              
```
for ``j = 0, 1, ... , N_k-1``, and in which ``\mathbf{x̂_d}`` and ``\mathbf{ŵ_d}`` are the
deterministic state and process noise estimates, respectively, extracted from the decision
variable `Z̃`. The ``\mathbf{k̇}`` coefficients are evaluated from the continuous-time
function [`fq_dae!`](@ref) and:
```math
\begin{aligned}
\mathbf{k̇_0}(ℓ+j) &= \mathbf{f}\Big(\mathbf{x̂_d}(ℓ+j),   \mathbf{a_0}(ℓ+j), \mathbf{û_0}(ℓ+j),   \mathbf{d_0}(ℓ+j),   \mathbf{p}\Big) \\
\mathbf{k̇_1}(ℓ+j) &= \mathbf{f}\Big(\mathbf{x̂_d}(ℓ+j+1), \mathbf{a_1}(ℓ+j), \mathbf{û_0}(ℓ+j+h), \mathbf{d_0}(ℓ+j+1), \mathbf{p}\Big) 
\end{aligned}
```
in which ``h`` is the hold order `transcription.h` and the disturbed input ``\mathbf{û_0}``
is defined in [`f̂!`](@ref) documentation. The residuals for [`NonLinModelDAE`](@ref) are
also computed from ``j = 0, 1, ... , N_k-1`` and:
```math
\begin{aligned}
\mathbf{q_0}(ℓ+j) &= \mathbf{q}\Big(\mathbf{x̂_d}(ℓ+j),   \mathbf{a_0}(ℓ+j), \mathbf{û_0}(ℓ+j),   \mathbf{d_0}(ℓ+j),   \mathbf{p}\Big) \\
\mathbf{q_1}(ℓ+j) &= \mathbf{q}\Big(\mathbf{x̂_d}(ℓ+j+1), \mathbf{a_1}(ℓ+j), \mathbf{û_0}(ℓ+j+h), \mathbf{d_0}(ℓ+j+1), \mathbf{p}\Big) 
\end{aligned}
```
and also one final residual at `k+p`:
```math
\mathbf{q_0}(k+p) = \mathbf{q}\Big(\mathbf{x̂_d}(k+p), \mathbf{a_0}(k+p), \mathbf{û_0}(k), \mathbf{d_0}(k), \mathbf{p}\Big)
```
"""
function con_nonlinprogeq_mhe!(
    geq, _ , _ , Û0, K̄,
    estim::MovingHorizonEstimator, model::SimModel, transcription::TrapezoidalCollocation, 
    x̂0arr, â0arr, Ŵ, Z̃
)
    nu, nx, nd, h = model.nu, model.nx, model.nd, transcription.h
    nx̂, nx̃, nxs, nŵ, He = estim.nx̂, estim.nx̃, estim.nxs, estim.nx̂, estim.He
    Nk = estim.Nk[]
    f_threads = transcription.f_threads
    Ts = model.Ts
    na = get_na(model)
    nk̄ = get_nk̄(model, transcription)
    nw = nŵ - nxs
    nx̃_nX̂  = nx̃ + nx̂*estim.He
    nŜk, nA, nĀ = nx*He, na*He, na*He
    i_d0arr = estim.direct ? 0 : nd # the first nd elements in D0 are useless if p=1
    X̂0_Z̃ = @views Z̃[(1 + nx̃):(nx̃_nX̂)]                   # skipping x̂0arr components
    Â0_Z̃ = @views Z̃[(1 + nx̃_nX̂ + na):(nx̃_nX̂ + na + nA)] # skipping â0arr components
    A1_Z̃ = @views Z̃[(1 + nx̃_nX̂ + na + nA):(nx̃_nX̂ + na + nA + nĀ)]
    Û0   = disturbedinput!(Û0, estim, x̂0arr, X̂0_Z̃, estim.U0)
    (Nk < He) && (geq .= 0) 
    Ŝk   = @views geq[1:nŜk]
    Q0   = @views geq[(1 + nŜk):(nŜk + nA + na)]
    Q̄    = @views geq[(1 + nŜk + nA + na):end]    
    @threadsif f_threads for j=1:Nk
        if j < 2
            x̂d_Z̃ = @views x̂0arr[1:nx]
            â0_Z̃ = @views â0arr[1:na]
        else
            x̂d_Z̃ = @views X̂0_Z̃[(1 + nx̂*(j-2)):(nx̂*(j-2) + nx)]
            â0_Z̃ = @views Â0_Z̃[(1 + na*(j-2)):(na*(j-2) + na)]
        end
        d0       = @views   estim.D0[(1 + nd*(j-1) + i_d0arr):(nd*j + i_d0arr)]
        û0       = @views         Û0[(1 + nu*(j-1)):(nu*j)]
        k̄dot     = @views          K̄[(1 + nk̄*(j-1)):(nk̄*j)]
        a1_Z̃     = @views       A1_Z̃[(1 + na*(j-1)):(na*j)]
        q0       = @views         Q0[(1 + na*(j-1)):(na*j)]
        q1       = @views          Q̄[(1 + na*(j-1)):(na*j)]
        ŵd       = @views          Ŵ[(1 + nŵ*(j-1)):(nŵ*(j-1) + nw)]
        x̂dnext_Z̃ = @views       X̂0_Z̃[(1 + nx̂*(j-1)):(nx̂*(j-1) + nx)]
        ŝk       = @views         Ŝk[(1 + nx*(j-1)):(nx*j)]
        k̇0, k̇1   = @views       k̄dot[1:nx], k̄dot[nx+1:2nx]
        if estim.direct || j < Nk
            d1 = @views estim.D0[(1 + nd*j + i_d0arr):(nd*(j+1) + i_d0arr)]
        else # special case, d0(k+1)≈d0(k), since d0(k+1) is not available at time k:
            d1 = d0 
        end
        # we need to recompute k̇0 & q0 with multi-threading, even with h>0, since 
        # the previous iteration (j-1) may not be executed (iterations are re-orderable)
        computeDynamicsAtBegin = f_threads || h < 1 || j < 2
        if computeDynamicsAtBegin
            fq_dae!(k̇0, q0, model, x̂d_Z̃, â0_Z̃, û0, d0)
        else # piecewise linear inputs u and disturbances d:
            k̇0 .= @views K̄[(1 + nk̄*(j-1)-nx):(nk̄*(j-1))] # k̇1[j-1] (prev. iter)
            q0 .= @views â0_Z̃ .- A1_Z̃[(1 + na*(j-2)):(na*(j-1))] # a0[j] = a1[j-1]
        end
        if h > 0 && j < Nk
            û1 = @views Û0[(1 + nu*j):(nu*(j+1))]
        else # special case, û0(k+p)≈û0(k+p-1), since û0(k+p) is not available at time k:
            û1 = û0 
        end
        fq_dae!(k̇1, q1, model, x̂dnext_Z̃, a1_Z̃, û1, d1)
        ŝk .= @. x̂d_Z̃ - x̂dnext_Z̃ + 0.5*Ts*(k̇0 + k̇1) + ŵd
    end
    # keep "end" in names, it solves a weird race cond. with f_threads on Julia v1.13.0
    q0end = @views   Q0[(1 + na*Nk):(na*Nk + na)]
    a0end = @views Â0_Z̃[(1 + na*(Nk-1)):(na*Nk)]
    if na > 0 && h < 1 # compute the final residual at k+p:
        x̂dend   = @views     X̂0_Z̃[(1 + nx̂*(Nk-1)):((nx̂*(Nk-1) + nx))]
        û0end   = @views       Û0[(1 + nu*(Nk-1)):((nu*(Nk-1) + nu))] # û0(k+p) ≈ û0(k+p-1)
        d0end   = @views estim.D0[(1 + nd*Nk):(nd*Nk + nd)]           # d0(k+1) ≈ d0(k)
        k̇0end   = @views        K̄[(end-nx+1):end]
        fq_dae!(k̇0end, q0end, model, x̂dend, a0end, û0end, d0end)
    else # piecewise linear inputs u and disturvances d:
        q0end .= a0end .- @views A1_Z̃[(1 + na*(Nk-1)):(na*Nk)]
    end
    return geq
end

@doc raw"""
    con_nonlinprogeq_mhe!(
        geq, _ , _ , Û0, K̄,
        estim::MovingHorizonEstimator, model::SimModel, ::OrthogonalCollocation, 
        x̂0arr, â0arr, _ , Z̃
    ) -> geq

Nonlinear MHE equality constrains for [`SimModel`](@ref) and [`OrthogonalCollocation`](@ref).

By introducing the integer ``ℓ = k - N_k + p`` to shorten the notation, the defects between
the deterministic state derivative at the ``n_o`` collocation points and the model dynamics
are computed by:
```math
\mathbf{ŝ_k̄}(ℓ+j)                                                                                 
    = \mathbf{M_o} \begin{bmatrix}                                          
        \mathbf{k}_1(ℓ+j) - \mathbf{x̂_d}(ℓ+j)                       \\
        \mathbf{k}_2(ℓ+j) - \mathbf{x̂_d}(ℓ+j)                       \\
        \vdots                                                      \\
        \mathbf{k}_{n_o}(ℓ+j) - \mathbf{x̂_d}(ℓ+j)                   \end{bmatrix}                                                                                     
    - \begin{bmatrix}
        \mathbf{k̇}_1(ℓ+j)                                           \\
        \mathbf{k̇}_2(ℓ+j)                                           \\
        \vdots                                                      \\
        \mathbf{k̇}_{n_o}(ℓ+j)                                       \end{bmatrix}
```
for ``j = 0, 1, ... , N_k-1``, and knowing that the ``\mathbf{k}_i(ℓ+j)`` and 
``\mathbf{x̂_d}(ℓ+j)`` vectors are extracted from the decision variables in `Z̃`. The
``\mathbf{k̇}_i`` vectors are evaluated from the continuous-time function [`fq_dae!`](@ref),
as described in [`init_orthocolloc`](@ref). The defects for the continuity constraints and the
stochastic states are linear equality constraints (see [`init_defectmat_mhe`](@ref)). The
estimated process noise ``\mathbf{ŵ}(ℓ+j)`` are incorporated in the continuity constraint.
"""
function con_nonlinprogeq_mhe!(
    geq, _ , _ , Û0, K̄,
    estim::MovingHorizonEstimator, model::SimModel, transcription::OrthogonalCollocation, 
    x̂0arr, â0arr, _ , Z̃
)
    nu, nx, nd, h = model.nu, model.nx, model.nd, transcription.h
    nx̂, nx̃, He = estim.nx̂, estim.nx̃, estim.He
    Nk = estim.Nk[]
    f_threads = transcription.f_threads
    no, τ, τendIsNotOne =  transcription.no, transcription.τ, transcription.τendIsNotOne
    Mo = estim.Mo
    na = get_na(model)
    nā, nk̄ = na*no, get_nk̄(model, transcription)
    nx̃_nX̂  = nx̃ + nx̂*estim.He
    nŜk̄, nA, nK̄, nĀ = nk̄*He, na*He, nk̄*He, nā*He
    i_d0arr = estim.direct ? 0 : nd # the first nd elements in D0 are useless if p=1
    X̂0_Z̃ = @views Z̃[(1 + nx̃):(nx̃_nX̂)]                   # skipping x̂0arr components
    Â0_Z̃ = @views Z̃[(1 + nx̃_nX̂ + na):(nx̃_nX̂ + na + nA)] # skipping â0arr components
    K̄_Z̃  = @views Z̃[(1 + nx̃_nX̂ + na + nA):(nx̃_nX̂ + na + nA + nK̄)]
    Ā_Z̃  = @views Z̃[(1 + nx̃_nX̂ + na + nA + nK̄):(nx̃_nX̂ + na + nA + nK̄ + nĀ)]
    Dtemp = estim.buffer.D
    Û0 = disturbedinput!(Û0, estim, x̂0arr, X̂0_Z̃, estim.U0)
    (Nk < He) && (geq .= 0) 
    Ŝk̄   = @views geq[1:nŜk̄]
    Q0   = @views geq[(1 + nŜk̄):(nŜk̄ + nA + na)]
    Q̄    = @views geq[(1 + nŜk̄ + nA + na):end]
    @threadsif f_threads for j=1:Nk
        if j < 2
            x̂d_Z̃ = @views x̂0arr[1:nx]
            â0_Z̃ = @views â0arr[1:na]
        else
            x̂d_Z̃ = @views X̂0_Z̃[(1 + nx̂*(j-2)):(nx̂*(j-2) + nx)]
            â0_Z̃ = @views Â0_Z̃[(1 + na*(j-2)):(na*(j-2) + na)]
        end
        d0   = @views estim.D0[(1 + nd*(j-1) + i_d0arr):(nd*j + i_d0arr)]
        û0   = @views       Û0[(1 + nu*(j-1)):(nu*j)]
        k̄dot = @views        K̄[(1 + nk̄*(j-1)):(nk̄*j)]
        k̄_Z̃  = @views      K̄_Z̃[(1 + nk̄*(j-1)):(nk̄*j)]
        ā_Z̃  = @views      Ā_Z̃[(1 + nā*(j-1)):(nā*j)]
        q0   = @views       Q0[(1 + na*(j-1)):(na*j)]
        q̄    = @views        Q̄[(1 + nā*(j-1)):(nā*j)]
        ŝk̄   = @views       Ŝk̄[(1 + nk̄*(j-1)):(nk̄*j)]
        if estim.direct || j < Nk
            d1 = @views estim.D0[(1 + nd*j + i_d0arr):(nd*(j+1) + i_d0arr)]
        else
            d1 = d0 # special case: d0(k+1)≈d0(k), since d0(k+1) is not available at time k:
        end
        # ----------------- residual at sampling instants (τ=0) -----------------------
        # we need to recompute q0 with multi-threading, even with h>0 and τ[end]≈1, since 
        # the previous iteration (j-1) may not be executed (iterations are re-orderable)
        computeDynamicsAtBegin = τendIsNotOne || f_threads || h < 1 || j < 2 
        if na > 0 && computeDynamicsAtBegin
            @views fq_dae!(k̄dot[1:nx] ,q0 , model, x̂d_Z̃, â0_Z̃, û0, d0)
        else
            q0 .= @views â0_Z̃ .- Ā_Z̃[(1 + nā*(j-1) - na):(nā*(j-1))] # a0[j] = a_no[j-1]
        end
        # ----------------- collocation constraint defects -----------------------------
        Δk = k̄dot
        for i=1:no
            Δk[(1 + (i-1)*nx):(i*nx)] = @views k̄_Z̃[(1 + (i-1)*nx):(i*nx)] .- x̂d_Z̃
        end
        mul!(ŝk̄, Mo, Δk)
        di = @views Dtemp[(1 + nd*(j-1)):(nd*j)]
        if h > 0
            ûi = similar(û0) # TODO: remove this allocation
        end
        for i=1:no
            k̇i, ki_Z̃  = @views k̄dot[(1 + (i-1)*nx):(i*nx)], k̄_Z̃[(1 + (i-1)*nx):(i*nx)]
            qi, ai_Z̃  = @views    q̄[(1 + (i-1)*na):(i*na)], ā_Z̃[(1 + (i-1)*na):(i*na)]
            di  .= (1-τ[i]).*d0 .+ τ[i].*d1
            if h > 0 && j < Nk
                û1  = @views Û0[(1 + nu*j):(nu*(j+1))]
                ûi .= (1-τ[i]).*û0 .+ τ[i].*û1
            else # special case, û0(k+p)≈û0(k+p-1), since û0(k+p) is not available:
                ûi  = û0 
            end
            fq_dae!(k̇i, qi, model, ki_Z̃, ai_Z̃, ûi, di)
        end
        ŝk̄ .-= k̄dot
    end
    # keep "end" in names, it solves a weird race cond. with f_threads on Julia v1.13.0
    q0end = @views   Q0[(1 + na*Nk):(na*Nk + na)]
    a0end = @views Â0_Z̃[(1 + na*(Nk-1)):(na*Nk)]
    if na > 0 && h < 1 # compute the final residual at k+p:
        x̂dend   = @views     X̂0_Z̃[(1 + nx̂*(Nk-1)):((nx̂*(Nk-1) + nx))]
        û0end   = @views       Û0[(1 + nu*(Nk-1)):((nu*(Nk-1) + nu))] # û0(k+p) ≈ û0(k+p-1)
        d0end   = @views estim.D0[(1 + nd*Nk):(nd*Nk + nd)]           # d0(k+1) ≈ d0(k)
        k̇0end   = @views        K̄[(end-nx+1):end]
        fq_dae!(k̇0end, q0end, model, x̂dend, a0end, û0end, d0end)
    else # piecewise linear inputs u and disturvances d:
        q0end .= @views a0end .- Ā_Z̃[(1 + nā*Nk - na):(nā*Nk)] # a0[j] = a_no[j-1]
    end
    return geq
end

"No nonlinear eq. const. for other cases e.g. [`SingleShooting`](@ref), returns `geq` unchanged."
con_nonlinprogeq_mhe!(geq,_,_,_,_,::MovingHorizonEstimator, ::SimModel, ::TranscriptionMethod,_,_,_,_) = geq