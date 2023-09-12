struct Luenberger <: StateEstimator
    model::LinModel
    lastu0::Vector{Float64}
    x̂::Vector{Float64}
    i_ym::Vector{Int}
    nx̂::Int
    nym::Int
    nyu::Int
    nxs::Int
    As  ::Matrix{Float64}
    Cs_u::Matrix{Float64}
    Cs_y::Matrix{Float64}
    nint_u ::Vector{Int}
    nint_ym::Vector{Int}
    Â   ::Matrix{Float64}
    B̂u  ::Matrix{Float64}
    Ĉ   ::Matrix{Float64}
    B̂d  ::Matrix{Float64}
    D̂d  ::Matrix{Float64}
    Ĉm  ::Matrix{Float64}
    D̂dm ::Matrix{Float64}
    K̂::Matrix{Float64}
    function Luenberger(model, i_ym, nint_u, nint_ym, p̂)
        nym, nyu = validate_ym(model, i_ym)
        As, Cs_u, Cs_y, nxs, nint_u, nint_ym = init_estimstoch(model, i_ym, nint_u, nint_ym)
        nx̂ = model.nx + nxs
        Â, B̂u, Ĉ, B̂d, D̂d = augment_model(model, As, Cs_u, Cs_y)
        K̂ = try
            place(Â, Ĉ, p̂, :o)[:, i_ym]
        catch
            error("Cannot compute the Luenberger gain K̂ with specified poles p̂.")
        end
        Ĉm, D̂dm = Ĉ[i_ym, :], D̂d[i_ym, :] # measured outputs ym only
        lastu0 = zeros(model.nu)
        x̂ = [zeros(model.nx); zeros(nxs)]
        return new(
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
Extended Help). The argument `p̂` is a vector of `model.nx + sum(nint_ym)` elements 
specifying the observer poles/eigenvalues (near ``z=0.5`` by default). The method computes 
the observer gain `K̂` with [`place`](https://juliacontrol.github.io/ControlSystems.jl/stable/lib/synthesis/#ControlSystemsBase.place).

# Examples
```jldoctest
julia> model = LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 0.5);

julia> estim = Luenberger(model, nint_ym=[1, 1], p̂=[0.61, 0.62, 0.63, 0.64])
Luenberger estimator with a sample time Ts = 0.5 s, LinModel and:
 1 manipulated inputs u
 4 states x̂
 2 measured outputs ym
 0 unmeasured outputs yu
 0 measured disturbances d
```
"""
function Luenberger(
    model::LinModel;
    i_ym::IntRangeOrVector  = 1:model.ny,
    nint_u ::IntVectorOrInt = 0,
    nint_ym::IntVectorOrInt = default_nint(model, i_ym, nint_u),
    p̂ = 1e-3*(1:(model.nx + sum(nint_u) + sum(nint_ym))) .+ 0.5
)
    nx = model.nx
    if length(p̂) ≠ model.nx + sum(nint_u) +  sum(nint_ym)
        error("p̂ length ($(length(p̂))) ≠ nx ($nx) + integrator quantity ($(sum(nint_ym)))")
    end
    any(abs.(p̂) .≥ 1) && error("Observer poles p̂ should be inside the unit circles.")
    return Luenberger(model, i_ym, nint_u, nint_ym, p̂)
end


"""
    update_estimate!(estim::Luenberger, u, ym, d=Float64[])

Same than [`update_estimate!(::SteadyKalmanFilter)`](@ref) but using [`Luenberger`](@ref).
"""
function update_estimate!(estim::Luenberger, u, ym, d=Float64[])
    Â, B̂u, B̂d, Ĉm, D̂dm = estim.Â, estim.B̂u, estim.B̂d, estim.Ĉm, estim.D̂dm
    x̂, K̂ = estim.x̂, estim.K̂
    x̂[:] = Â*x̂ + B̂u*u + B̂d*d + K̂*(ym - Ĉm*x̂ - D̂dm*d)
    return x̂
end