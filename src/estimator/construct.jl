struct StateEstimatorBuffer{NT<:Real}
    u ::Vector{NT}
    û ::Vector{NT}
    k::Vector{NT}
    x̂ ::Vector{NT}
    P̂ ::Matrix{NT}
    Q̂ ::Matrix{NT}
    R̂ ::Matrix{NT}
    K̂ ::Matrix{NT}
    ym::Vector{NT}
    ŷ ::Vector{NT}
    d ::Vector{NT}
    empty::Vector{NT}
end

@doc raw"""
    StateEstimatorBuffer{NT}(nu::Int, nx̂::Int, nym::Int, ny::Int, nd::Int, nk::Int=0)

Create a buffer for `StateEstimator` objects for estimated states and measured outputs.

The buffer is used to store intermediate results during estimation without allocating.
"""
function StateEstimatorBuffer{NT}(
    nu::Int, nx̂::Int, nym::Int, ny::Int, nd::Int, nk::Int=0
) where NT <: Real
    u  = Vector{NT}(undef, nu)
    û  = Vector{NT}(undef, nu)
    k  = Vector{NT}(undef, nk)
    x̂  = Vector{NT}(undef, nx̂)
    P̂  = Matrix{NT}(undef, nx̂, nx̂)
    Q̂  = Matrix{NT}(undef, nx̂, nx̂)
    R̂  = Matrix{NT}(undef, nym, nym)
    K̂  = Matrix{NT}(undef, nx̂, nym)
    ym = Vector{NT}(undef, nym)
    ŷ  = Vector{NT}(undef, ny)
    d  = Vector{NT}(undef, nd)
    empty = Vector{NT}(undef, 0)
    return StateEstimatorBuffer{NT}(u, û, k, x̂, P̂, Q̂, R̂, K̂, ym, ŷ, d, empty)
end

"Include all the covariance matrices for the Kalman filters and moving horizon estimator."
struct KalmanCovariances{
    NT<:Real,
    # parameters to support both dense and Diagonal matrices (with specialization):
    Q̂C<:AbstractMatrix{NT},
    R̂C<:AbstractMatrix{NT},
}
    P̂_0::Hermitian{NT, Matrix{NT}}
    P̂::Hermitian{NT, Matrix{NT}}
    Q̂::Hermitian{NT, Q̂C}
    R̂::Hermitian{NT, R̂C}
    invP̄::Hermitian{NT, Matrix{NT}}
    invQ̂_He::Hermitian{NT, Q̂C}
    invR̂_He::Hermitian{NT, R̂C}
    function KalmanCovariances{NT}(
        model, i_ym, nint_u, nint_ym, Q̂::Q̂C, R̂::R̂C, P̂_0=nothing, He=1
    ) where {NT<:Real, Q̂C<:AbstractMatrix{NT}, R̂C<:AbstractMatrix{NT}}
        validate_kfcov(model, i_ym, nint_u, nint_ym, Q̂, R̂, P̂_0)
        if isnothing(P̂_0)
            P̂_0 = zeros(NT, 0, 0)
        end
        P̂_0 = Hermitian(P̂_0, :L)
        P̂   = copy(P̂_0)
        Q̂   = Hermitian(Q̂, :L)
        R̂   = Hermitian(R̂, :L)
        # the following variables are only for the moving horizon estimator:
        invP̄, invQ̂, invR̂ = copy(P̂_0), copy(Q̂), copy(R̂)
        inv!(invP̄)
        inv!(invQ̂)
        inv!(invR̂)
        invQ̂_He = repeatdiag(invQ̂, He)
        invR̂_He = repeatdiag(invR̂, He)
        isdiag(invQ̂_He) && (invQ̂_He = Diagonal(invQ̂_He)) # Q̂C(invQ̂_He) does not work on Julia 1.10
        isdiag(invR̂_He) && (invR̂_He = Diagonal(invR̂_He)) # R̂C(invR̂_He) does not work on Julia 1.10
        invQ̂_He = Hermitian(invQ̂_He, :L)
        invR̂_He = Hermitian(invR̂_He, :L)
        return new{NT, Q̂C, R̂C}(P̂_0, P̂, Q̂, R̂, invP̄, invQ̂_He, invR̂_He)
    end
end

"Outer constructor to convert covariance matrix number type to `NT` if necessary."
function KalmanCovariances(
        model::SimModel{NT}, i_ym, nint_u, nint_ym, Q̂, R̂, P̂_0=nothing, He=1
    ) where {NT<:Real}
    return KalmanCovariances{NT}(model, i_ym, nint_u, nint_ym, NT.(Q̂), NT.(R̂), P̂_0, He)
end

"""
    validate_kfcov(model, i_ym, nint_u, nint_ym, Q̂, R̂, P̂_0=nothing)

Validate sizes and Hermitianity of process `Q̂`` and sensor `R̂` noises covariance matrices.

Also validate initial estimate covariance `P̂_0`, if provided.
"""
function validate_kfcov(model, i_ym, nint_u, nint_ym, Q̂, R̂, P̂_0=nothing)
    nym = length(i_ym)
    nx̂  = model.nx + sum(nint_u) + sum(nint_ym)
    size(Q̂)  ≠ (nx̂, nx̂)     && error("Q̂ size $(size(Q̂)) ≠ nx̂, nx̂ $((nx̂, nx̂))")
    !ishermitian(Q̂)         && error("Q̂ is not Hermitian")
    size(R̂)  ≠ (nym, nym)   && error("R̂ size $(size(R̂)) ≠ nym, nym $((nym, nym))")
    !ishermitian(R̂)         && error("R̂ is not Hermitian")
    if ~isnothing(P̂_0)
        size(P̂_0) ≠ (nx̂, nx̂) && error("P̂_0 size $(size(P̂_0)) ≠ nx̂, nx̂ $((nx̂, nx̂))")
        !ishermitian(P̂_0)    && error("P̂_0 is not Hermitian")
    end
end

@doc raw"""
    init_estimstoch(model, i_ym, nint_u, nint_ym) -> As, Cs_u, Cs_y, nxs, nint_u, nint_ym

Init stochastic model matrices from integrator specifications for state estimation.

The arguments `nint_u` and `nint_ym` specify how many integrators are added to each 
manipulated input and measured outputs. The function returns the state-space matrices `As`, 
`Cs_u` and `Cs_y` of the stochastic model:
```math
\begin{aligned}
\mathbf{x_{s}}(k+1)     &= \mathbf{A_s x_s}(k) + \mathbf{B_s e}(k) \\
\mathbf{y_{s_{u}}}(k)   &= \mathbf{C_{s_{u}} x_s}(k) \\
\mathbf{y_{s_{y}}}(k)   &= \mathbf{C_{s_{y}} x_s}(k) 
\end{aligned}
```
where ``\mathbf{e}(k)`` is an unknown zero mean white noise and ``\mathbf{A_s} = 
\mathrm{diag}(\mathbf{A_{s_{u}}, A_{s_{y}}})``. The estimations does not use ``\mathbf{B_s}``,
it is thus ignored. The function [`init_integrators`](@ref) builds the state-space matrices.
"""
function init_estimstoch(
    model::SimModel{NT}, i_ym, nint_u::IntVectorOrInt, nint_ym::IntVectorOrInt
) where {NT<:Real}
    nu, ny, nym = model.nu, model.ny, length(i_ym)
    As_u , Cs_u , nint_u  = init_integrators(nint_u , nu , "u")
    As_ym, Cs_ym, nint_ym = init_integrators(nint_ym, nym, "ym")
    As_y, _ , Cs_y = stoch_ym2y(model, i_ym, As_ym, zeros(NT, 0, 0), Cs_ym, zeros(NT, 0, 0))
    nxs_u, nxs_y = size(As_u, 1), size(As_y, 1)
    # combines input and output stochastic models:
    As   = [As_u zeros(NT, nxs_u, nxs_y); zeros(NT, nxs_y, nxs_u) As_y]
    Cs_u = [Cs_u zeros(NT, nu, nxs_y)]
    Cs_y = [zeros(NT, ny, nxs_u) Cs_y]
    return As, Cs_u, Cs_y, nint_u, nint_ym
end

"Validate the specified measured output indices `i_ym`."
function validate_ym(model::SimModel, i_ym)
    if length(unique(i_ym)) ≠ length(i_ym) || maximum(i_ym) > model.ny
        error("Measured output indices i_ym should contains valid and unique indices")
    end
    nym, nyu = length(i_ym), model.ny - length(i_ym)
    return nym, nyu
end

"Convert the measured outputs stochastic model `stoch_ym` to all outputs `stoch_y`."
function stoch_ym2y(model::SimModel{NT}, i_ym, Asm, Bsm, Csm, Dsm) where {NT<:Real}
    As = Asm
    Bs = Bsm
    Cs = zeros(NT, model.ny, size(Csm,2))
    Cs[i_ym,:] = Csm
    if isempty(Dsm)
        Ds = Dsm
    else
        Ds = zeros(NT, model.ny, size(Dsm,2))
        Ds[i_ym,:] = Dsm
    end
    return As, Bs, Cs, Ds
end

@doc raw"""
    init_integrators(nint, ny, varname::String) -> A, C, nint

Calc `A, C` state-space matrices from integrator specifications `nint`.

This function is used to initialize the stochastic part of the augmented model for the
design of state estimators. The vector `nint` provides how many integrators (in series) 
should be incorporated for each output. The argument should have `ny` element, except
for `nint=0` which is an alias for no integrator at all. The specific case of one integrator
per output results in `A = I` and `C = I`. The estimation does not use the `B` matrix, it 
is thus ignored. This function is called twice :

1. for the unmeasured disturbances at manipulated inputs ``\mathbf{u}``
2. for the unmeasured disturbances at measured outputs ``\mathbf{y^m}``
"""
function init_integrators(nint::IntVectorOrInt, ny, varname::String)
    if nint == 0 # alias for no integrator at all
        nint = fill(0, ny)
    end
    if length(nint) ≠ ny
        error("nint_$(varname) length ($(length(nint))) ≠ n$(varname) ($ny)")
    end
    any(nint .< 0) && error("nint_$(varname) values should be ≥ 0")
    nx = sum(nint)
    A, C = zeros(nx, nx), zeros(ny, nx)
    if nx ≠ 0
        i_A, i_g = 1, 1
        for i = 1:ny
            nint_i = nint[i]
            if nint_i ≠ 0
                rows_A = (i_A):(i_A + nint_i - 1)
                A[rows_A, rows_A] = Bidiagonal(ones(nint_i), ones(nint_i-1), :L)
                C[i, i_g+nint_i-1] = 1
                i_A += nint_i
                i_g += nint_i
            end
        end
    end
    return A, C, nint
end


@doc raw"""
    augment_model(
        model::LinModel, As, Cs_u, Cs_y; verify_obsv=true
    ) -> Â, B̂u, Ĉ, B̂d, D̂d, x̂op, f̂op

Augment [`LinModel`](@ref) state-space matrices with stochastic ones `As`, `Cs_u`, `Cs_y`.

If ``\mathbf{x_0}`` are `model.x0` states, and ``\mathbf{x_s}``, the states defined at
[`init_estimstoch`](@ref), we define an augmented state vector ``\mathbf{x̂} = 
[ \begin{smallmatrix} \mathbf{x_0} \\ \mathbf{x_s} \end{smallmatrix} ]``. The method
returns the augmented matrices `Â`, `B̂u`, `Ĉ`, `B̂d` and `D̂d`:
```math
\begin{aligned}
    \mathbf{x̂_0}(k+1) &= \mathbf{Â x̂_0}(k) + \mathbf{B̂_u u_0}(k) + \mathbf{B̂_d d_0}(k) \\
    \mathbf{ŷ_0}(k)   &= \mathbf{Ĉ x̂_0}(k) + \mathbf{D̂_d d_0}(k)
\end{aligned}
```
An error is thrown if the augmented model is not observable and `verify_obsv == true`. The
augmented operating points ``\mathbf{x̂_{op}}`` and ``\mathbf{f̂_{op}}`` are simply 
``\mathbf{x_{op}}`` and ``\mathbf{f_{op}}`` vectors appended with zeros (see [`setop!`](@ref)). 
See Extended Help for a detailed definition of the augmented matrices and vectors.

# Extended Help
!!! details "Extended Help"
    Using the `As`, `Cs_u` and `Cs_y` matrices of the stochastic model provided in argument
    and the `model.A`, `model.Bu`, `model.Bd`, `model.C`, `model.Dd` matrices, the 
    state-space matrices of the augmented model are defined as follows:
    ```math
    \begin{aligned}
    \mathbf{Â}   &=                                    \begin{bmatrix} 
        \mathbf{A} & \mathbf{B_u C_{s_u}}              \\ 
        \mathbf{0} & \mathbf{A_s}                      \end{bmatrix} \\
    \mathbf{B̂_u} &=                                    \begin{bmatrix}
        \mathbf{B_u}                                   \\
        \mathbf{0}                                     \end{bmatrix} \\
    \mathbf{Ĉ}   &=                                    \begin{bmatrix}
        \mathbf{C} & \mathbf{C_{s_y}}                  \end{bmatrix} \\
    \mathbf{B̂_d} &=                                    \begin{bmatrix} 
        \mathbf{B_d}                                   \\
        \mathbf{0}                                     \end{bmatrix} \\
    \mathbf{D̂_d} &= \mathbf{D_d}
    \end{aligned}
    ```
    and the operating points of the augmented model are:
    ```math
    \begin{aligned}
        \mathbf{x̂_{op}} &= \begin{bmatrix} \mathbf{x_{op}} \\ \mathbf{0} \end{bmatrix} \\
        \mathbf{f̂_{op}} &= \begin{bmatrix} \mathbf{f_{op}} \\ \mathbf{0} \end{bmatrix}
    \end{aligned}
    ```
"""
function augment_model(model::LinModel{NT}, As, Cs_u, Cs_y; verify_obsv=true) where NT<:Real
    nu, nx, nd = model.nu, model.nx, model.nd
    nxs = size(As, 1)
    Â   = [model.A model.Bu*Cs_u; zeros(NT, nxs,nx) As]
    B̂u  = [model.Bu; zeros(NT, nxs, nu)]
    Ĉ   = [model.C Cs_y]
    B̂d  = [model.Bd; zeros(NT, nxs, nd)]
    D̂d  = model.Dd
    # observability on Ĉ instead of Ĉm, since it would always return false when nym ≠ ny:
    if verify_obsv && !ControlSystemsBase.observability(Â, Ĉ)[:isobservable]
        error("The augmented model is unobservable. You may try to use 0 integrator on "*
              "model integrating outputs with nint_ym parameter. Adding integrators at both "*
              "inputs (nint_u) and outputs (nint_ym) can also violate observability. If the "*
              "model is still unobservable without any integrators, you may need to call "*
              "sminreal or minreal on your system.")
    end
    x̂op, f̂op = [model.xop; zeros(nxs)], [model.fop; zeros(nxs)]
    return Â, B̂u, Ĉ, B̂d, D̂d, x̂op, f̂op
end
"""
    augment_model(
        model::SimModel, As, Cs_u, Cs_y; verify_obsv=false
    ) -> Â, B̂u, Ĉ, B̂d, D̂d, x̂op, f̂op

Return empty matrices, and `x̂op` & `f̂op` vectors, if `model` is not a [`LinModel`](@ref).
"""
function augment_model(model::SimModel{NT}, As, args... ; verify_obsv=false) where NT<:Real
    nu, nx, nd, ny = model.nu, model.nx, model.nd, model.ny
    nxs = size(As, 1)
    Â   = zeros(NT, 0, nx+nxs)
    B̂u  = zeros(NT, 0, nu)
    Ĉ   = zeros(NT, ny, 0)
    B̂d  = zeros(NT, 0, nd)
    D̂d  = zeros(NT, ny, 0)
    x̂op, f̂op = [model.xop; zeros(nxs)], [model.fop; zeros(nxs)]
    return Â, B̂u, Ĉ, B̂d, D̂d, x̂op, f̂op
end

@doc raw"""
    default_nint(model::LinModel, i_ym=1:model.ny, nint_u=0) -> nint_ym

Get default integrator quantity per measured outputs `nint_ym` for [`LinModel`](@ref).

The arguments `i_ym` and `nint_u` are the measured output indices and the integrator
quantity on each manipulated input, respectively. By default, one integrator is added on
each measured outputs. If ``\mathbf{Â, Ĉ}`` matrices of the augmented model become
unobservable, the integrator is removed. This approach works well for stable, integrating
and unstable `model` (see Examples).

# Examples
```jldoctest
julia> model = LinModel(append(tf(3, [10, 1]), tf(2, [1, 0]), tf(4,[-5, 1])), 1.0);

julia> nint_ym = default_nint(model)
3-element Vector{Int64}:
 1
 0
 1
```
"""
function default_nint(model::LinModel, i_ym=1:model.ny, nint_u=0)
    validate_ym(model, i_ym)
    nint_ym = fill(0, length(i_ym))
    for i in eachindex(i_ym)
        nint_ym[i]  = 1
        As, Cs_u, Cs_y = init_estimstoch(model, i_ym, nint_u, nint_ym)
        Â, _ , Ĉ = augment_model(model, As, Cs_u, Cs_y, verify_obsv=false)
        # observability on Ĉ instead of Ĉm, since it would always return false when nym ≠ ny
        ControlSystemsBase.observability(Â, Ĉ)[:isobservable] || (nint_ym[i] = 0)
    end
    return nint_ym
end

"""
    default_nint(model::SimModel, i_ym=1:model.ny, nint_u=0)

One integrator on each measured output by default if `model` is not a  [`LinModel`](@ref).

Theres is no verification the augmented model remains observable. If the integrator quantity
per manipulated input `nint_u ≠ 0`, the method returns zero integrator on each measured
output.
"""
function default_nint(model::SimModel, i_ym=1:model.ny, nint_u=0)
    validate_ym(model, i_ym)
    nint_ym = iszero(nint_u) ? fill(1, length(i_ym)) : fill(0, length(i_ym))
    return nint_ym
end