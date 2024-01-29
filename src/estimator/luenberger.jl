struct Luenberger{NT<:Real, SM<:LinModel} <: StateEstimator{NT}
    model::SM
    lastu0::Vector{NT}
    x̂::Vector{NT}
    i_ym::Vector{Int}
    nx̂::Int
    nym::Int
    nyu::Int
    nxs::Int
    As  ::Matrix{NT}
    Cs_u::Matrix{NT}
    Cs_y::Matrix{NT}
    nint_u ::Vector{Int}
    nint_ym::Vector{Int}
    Â   ::Matrix{NT}
    B̂u  ::Matrix{NT}
    Ĉ   ::Matrix{NT}
    B̂d  ::Matrix{NT}
    D̂d  ::Matrix{NT}
    Ĉm  ::Matrix{NT}
    D̂dm ::Matrix{NT}
    K̂::Matrix{NT}
    function Luenberger{NT, SM}(
        model, i_ym, nint_u, nint_ym, p̂
    ) where {NT<:Real, SM<:LinModel}
        nym, nyu = validate_ym(model, i_ym)
        validate_luenberger(model, nint_u, nint_ym, p̂)
        As, Cs_u, Cs_y, nint_u, nint_ym = init_estimstoch(model, i_ym, nint_u, nint_ym)
        nxs = size(As, 1)
        nx̂  = model.nx + nxs
        Â, B̂u, Ĉ, B̂d, D̂d = augment_model(model, As, Cs_u, Cs_y)
        K̂ = try
            place(Â, Ĉ, p̂, :o)[:, i_ym]
        catch
            error("Cannot compute the Luenberger gain K̂ with specified poles p̂.")
        end
        Ĉm, D̂dm = Ĉ[i_ym, :], D̂d[i_ym, :] # measured outputs ym only
        lastu0 = zeros(NT, model.nu)
        x̂ = [zeros(NT, model.nx); zeros(NT, nxs)]
        return new{NT, SM}(
            model, 
            lastu0, x̂,
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs_u, Cs_y, nint_u, nint_ym,
            Â, B̂u, Ĉ, B̂d, D̂d,
            Ĉm, D̂dm,
            K̂
        )
    end
end

@doc raw"""
    Luenberger(
        model::LinModel; 
        i_ym = 1:model.ny, 
        nint_u  = 0,
        nint_ym = default_nint(model, i_ym),
        p̂ = 1e-3*(1:(model.nx + sum(nint_u) + sum(nint_ym))) .+ 0.5
    )

Construct a Luenberger observer with the [`LinModel`](@ref) `model`.

`i_ym` provides the `model` output indices that are measured ``\mathbf{y^m}``, the rest are
unmeasured ``\mathbf{y^u}``. `model` matrices are augmented with the stochastic model, which
is specified by the numbers of integrator `nint_u` and `nint_ym` (see [`SteadyKalmanFilter`](@ref)
Extended Help). The argument `p̂` is a vector of `model.nx + sum(nint_u) + sum(nint_ym)`
elements specifying the observer poles/eigenvalues (near ``z=0.5`` by default). The method
computes the observer gain `K̂` with [`place`](https://juliacontrol.github.io/ControlSystems.jl/stable/lib/synthesis/#ControlSystemsBase.place).

# Examples
```jldoctest
julia> model = LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 0.5);

julia> estim = Luenberger(model, nint_ym=[1, 1], p̂=[0.61, 0.62, 0.63, 0.64])
Luenberger estimator with a sample time Ts = 0.5 s, LinModel and:
 1 manipulated inputs u (0 integrating states)
 4 estimated states x̂
 2 measured outputs ym (2 integrating states)
 0 unmeasured outputs yu
 0 measured disturbances d
```
"""
function Luenberger(
    model::SM;
    i_ym::IntRangeOrVector  = 1:model.ny,
    nint_u ::IntVectorOrInt = 0,
    nint_ym::IntVectorOrInt = default_nint(model, i_ym, nint_u),
    p̂ = 1e-3*(1:(model.nx + sum(nint_u) + sum(nint_ym))) .+ 0.5
) where{NT<:Real, SM<:LinModel{NT}}
    return Luenberger{NT, SM}(model, i_ym, nint_u, nint_ym, p̂)
end

"Validate the quantity and stability of the Luenberger poles `p̂`."
function validate_luenberger(model, nint_u, nint_ym, p̂)
    if length(p̂) ≠ model.nx + sum(nint_u) +  sum(nint_ym)
        error("p̂ length ($(length(p̂))) ≠ nx ($(model.nx)) + "*
              "integrator quantity ($(sum(nint_ym)))")
    end
    any(abs.(p̂) .≥ 1) && error("Observer poles p̂ should be inside the unit circles.")
end


"""
    update_estimate!(estim::Luenberger, u, ym, d=empty(estim.x̂))

Same than [`update_estimate!(::SteadyKalmanFilter)`](@ref) but using [`Luenberger`](@ref).
"""
function update_estimate!(estim::Luenberger, u, ym, d=empty(estim.x̂))
    Â, B̂u, B̂d, Ĉm, D̂dm = estim.Â, estim.B̂u, estim.B̂d, estim.Ĉm, estim.D̂dm
    x̂, K̂ = estim.x̂, estim.K̂
    v̂, ŷm, x̂LHS = similar(ym), similar(ym), similar(x̂)
    # in-place operations to recuce allocations:
    ŷm  .= mul!(v̂, Ĉm, x̂) 
    ŷm .+= mul!(v̂, D̂dm, d)
    v̂   .= ym .- ŷm
    x̂   .= mul!(x̂LHS, Â, x̂)
    x̂  .+= mul!(x̂LHS, B̂u, u)
    x̂  .+= mul!(x̂LHS, B̂d, d)
    x̂  .+= mul!(x̂LHS, K̂, v̂)
    return nothing
end