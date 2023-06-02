struct Luenberger <: StateEstimator
    model::LinModel
    lastu0::Vector{Float64}
    x̂::Vector{Float64}
    i_ym::Vector{Int}
    nx̂::Int
    nym::Int
    nyu::Int
    nxs::Int
    As::Matrix{Float64}
    Cs::Matrix{Float64}
    nint_ym::Vector{Int}
    Â   ::Matrix{Float64}
    B̂u  ::Matrix{Float64}
    B̂d  ::Matrix{Float64}
    Ĉ   ::Matrix{Float64}
    D̂d  ::Matrix{Float64}
    Ĉm  ::Matrix{Float64}
    D̂dm ::Matrix{Float64}
    K::Matrix{Float64}
    function Luenberger(model, i_ym, nint_ym, Asm, Csm, p)
        nu, nx, ny = model.nu, model.nx, model.ny
        nym, nyu = length(i_ym), ny - length(i_ym)
        nxs = size(Asm,1)
        nx̂ = nx + nxs
        As, _ , Cs, _  = stoch_ym2y(model, i_ym, Asm, [], Csm, [])
        Â, B̂u, Ĉ, B̂d, D̂d = augment_model(model, As, Cs)
        Ĉm, D̂dm = Ĉ[i_ym, :], D̂d[i_ym, :] # measured outputs ym only
        K = try
            place(Â, Ĉ, p, :o)
        catch
            error("Cannot compute the Luenberger gain L with specified poles p.")        
        end
        i_ym = collect(i_ym)
        lastu0 = zeros(nu)
        x̂ = [copy(model.x); zeros(nxs)]
        return new(
            model, 
            lastu0, x̂,
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs, nint_ym,
            Â, B̂u, B̂d, Ĉ, D̂d, 
            Ĉm, D̂dm,
            K
        )
    end
end

@doc raw"""
    Luenberger(
        model::LinModel; 
        i_ym = 1:model.ny, 
        nint_ym = fill(1, length(i_ym)),
        p̂ = 1e-3*(0:(model.nx + sum(nint_ym)-1)) .+ 0.5)
    )

Construct a Luenberger observer with the [`LinModel`](@ref) `model`.

`i_ym` provides the `model` output indices that are measured ``\mathbf{y^m}``, the rest are 
unmeasured ``\mathbf{y^u}``. `model` matrices are augmented with the stochastic model, which
is specified by the numbers of output integrator `nint_ym` (see [`SteadyKalmanFilter`](@ref)
Extended Help). The argument `p̂` is a vector of `model.nx + sum(nint_ym)` elements 
specifying the observer poles/eigenvalues (near ``z=0.5`` by default). The method computes 
the observer gain ``\mathbf{K}`` with [`place`](https://juliacontrol.github.io/ControlSystems.jl/stable/lib/synthesis/#ControlSystemsBase.place).

# Examples
```jldoctest
julia> model = LinModel([tf(3, [30, 1]); tf(-2, [5, 1])], 0.5);

julia> lo = Luenberger(model, nint_ym=[1, 1], p̂=[0.61, 0.62, 0.63, 0.64])
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
    nint_ym::IntVectorOrInt = fill(1, length(i_ym)),
    p̂ = 1e-3*(0:(model.nx + sum(nint_ym)-1)) .+ 0.5
)
    if nint_ym == 0 # alias for no output integrator at all :
        nint_ym = fill(0, length(i_ym));
    end
    Asm, Csm = init_estimstoch(i_ym, nint_ym)
    nx = model.nx
    if length(p̂) ≠ model.nx + sum(nint_ym)
        error("p̂ length ($(length(p̂))) ≠ nx ($nx) + integrator quantity ($(sum(nint_ym)))")
    end
    any(abs.(p̂) .≥ 1) && error("Observer poles p̂ should be inside the unit circles.")
    return Luenberger(model, i_ym, nint_ym, Asm, Csm, p̂)
end


"""
    updatestate!(estim::Luenberger, u, ym, d=Float64[])

Same than `SteadyKalmanFilter` but using the Luenberger observer.
"""
function updatestate!(estim::Luenberger, u, ym, d=Float64[])
    u, d, ym = remove_op!(estim, u, d, ym)
    Â, B̂u, B̂d, Ĉm, D̂dm = estim.Â, estim.B̂u, estim.B̂d, estim.Ĉm, estim.D̂dm
    x̂, K = estim.x̂, estim.K
    x̂[:] = Â*x̂ + B̂u*u + B̂d*d + K*(ym - Ĉm*x̂ - D̂dm*d)
    return x̂    
end