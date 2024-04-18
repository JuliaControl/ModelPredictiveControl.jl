

@doc raw"""
    init_estimstoch(model, i_ym, nint_u, nint_ym) -> As, Cs_u, Cs_y, nxs, nint_u, nint_ym

Init stochastic model matrices from integrator specifications for state estimation.

The arguments `nint_u` and `nint_ym` specify how many integrators are added to each 
manipulated input and measured outputs. The function returns the state-space matrices `As`, 
`Cs_u` and `Cs_y` of the stochastic model:
```math
\begin{aligned}
\mathbf{x_{s}}(k+1)     &= \mathbf{A_s x_s}(k) + \mathbf{B_s e}(k) \\
\mathbf{y_{s_{u}}}(k)   &= \mathbf{C_{s_{u}}  x_s}(k) \\
\mathbf{y_{s_{ym}}}(k)  &= \mathbf{C_{s_{ym}} x_s}(k) 
\end{aligned}
```
where ``\mathbf{e}(k)`` is an unknown zero mean white noise and ``\mathbf{A_s} = 
\mathrm{diag}(\mathbf{A_{s_{u}}, A_{s_{ym}}})``. The estimations does not use ``\mathbf{B_s}``,
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
        error("nint_$(varname) size ($(length(nint))) ≠ n$(varname) ($ny)")
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
    augment_model(model::LinModel, As, Cs; verify_obsv=true) -> Â, B̂u, Ĉ, B̂d, D̂d

Augment [`LinModel`](@ref) state-space matrices with the stochastic ones `As` and `Cs`.

If ``\mathbf{x}`` are `model.x0` states, and ``\mathbf{x_s}``, the states defined at
[`init_estimstoch`](@ref), we define an augmented state vector ``\mathbf{x̂} = 
[ \begin{smallmatrix} \mathbf{x} \\ \mathbf{x_s} \end{smallmatrix} ]``. The method
returns the augmented matrices `Â`, `B̂u`, `Ĉ`, `B̂d` and `D̂d`:
```math
\begin{aligned}
    \mathbf{x̂}(k+1) &= \mathbf{Â x̂}(k) + \mathbf{B̂_u u}(k) + \mathbf{B̂_d d}(k) \\
    \mathbf{ŷ}(k)   &= \mathbf{Ĉ x̂}(k) + \mathbf{D̂_d d}(k)
\end{aligned}
```
An error is thrown if the augmented model is not observable and `verify_obsv == true`.
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
              "inputs (nint_u) and outputs (nint_ym) can also violate observability.")
    end
    return Â, B̂u, Ĉ, B̂d, D̂d
end
"Return empty matrices if `model` is not a [`LinModel`](@ref)."
function augment_model(model::SimModel{NT}, As, _ , _ ) where NT<:Real
    nu, nx, nd = model.nu, model.nx, model.nd
    nxs = size(As, 1)
    Â   = zeros(NT, 0, nx+nxs)
    B̂u  = zeros(NT, 0, nu)
    Ĉ   = zeros(NT, 0, nx+nxs)
    B̂d  = zeros(NT, 0, nd)
    D̂d  = zeros(NT, 0, nd)
    return Â, B̂u, Ĉ, B̂d, D̂d
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

If the integrator quantity per manipulated input `nint_u ≠ 0`, the method returns zero
integrator on each measured output.
"""
function default_nint(model::SimModel, i_ym=1:model.ny, nint_u=0)
    validate_ym(model, i_ym)
    nint_ym = iszero(nint_u) ? fill(1, length(i_ym)) : fill(0, length(i_ym))
    return nint_ym
end