struct NonLinMHE{M<:SimModel} <: StateEstimator
    model::M
    optim::JuMP.Model
    lastu0::Vector{Float64}
    x̂::Vector{Float64}
    P̂::Hermitian{Float64, Matrix{Float64}}
    i_ym::Vector{Int}
    nx̂::Int
    nym::Int
    nyu::Int
    nxs::Int
    As::Matrix{Float64}
    Cs::Matrix{Float64}
    nint_ym::Vector{Int}
    P̂0::Hermitian{Float64, Matrix{Float64}}
    Q̂::Hermitian{Float64, Matrix{Float64}}
    R̂::Hermitian{Float64, Matrix{Float64}}
    He::Int 
    X̂max::Vector{Float64}
    X̂min::Vector{Float64}
    function NonLinMHE{M}(model::M, i_ym, nint_ym, P̂0, Q̂, R̂, He) where {M<:SimModel}
        nu, nx, ny = model.nu, model.nx, model.ny
        nym, nyu = length(i_ym), ny - length(i_ym)
        Asm, Csm, nint_ym = init_estimstoch(i_ym, nint_ym)
        nxs = size(Asm,1)
        nx̂ = nx + nxs
        validate_kfcov(nym, nx̂, Q̂, R̂, P̂0)
        As, _ , Cs, _  = stoch_ym2y(model, i_ym, Asm, [], Csm, [])
        nσ, γ, m̂, Ŝ = init_mhe(nx̂, He)
        i_ym = collect(i_ym)
        lastu0 = zeros(nu)
        x̂ = [zeros(model.nx); zeros(nxs)]
        P̂0 = Hermitian(P̂0, :L)
        Q̂ = Hermitian(Q̂, :L)
        R̂ = Hermitian(R̂, :L)
        P̂ = copy(P̂0)
        return new(
            model,
            lastu0, x̂, P̂, 
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs, nint_ym,
            P̂0, Q̂, R̂,
            nσ, γ, m̂, Ŝ
        )
    end
end

@doc raw"""
    NonLinMHE(model::SimModel; <keyword arguments>)

Construct a nonlinear moving horizon estimator with the [`SimModel`](@ref) `model`.

Both [`LinModel`](@ref) and [`NonLinModel`](@ref) are supported. The process model is 
identical to [`UnscentedKalmanFilter`](@ref).

# Arguments
- `model::SimModel` : (deterministic) model for the estimations.
- `He::Int=10` : estimation horizon.
- `<keyword arguments>` of [`SteadyKalmanFilter`](@ref) constructor.
- `<keyword arguments>` of [`KalmanFilter`](@ref) constructor.

# Examples
```jldoctest
julia> model = NonLinModel((x,u,_)->0.1x+u, (x,_)->2x, 10.0, 1, 1, 1);

```
"""
function NonLinMHE(
    model::M;
    i_ym::IntRangeOrVector = 1:model.ny,
    σP0::Vector = fill(1/model.nx, model.nx),
    σQ::Vector  = fill(1/model.nx, model.nx),
    σR::Vector  = fill(1, length(i_ym)),
    nint_ym::IntVectorOrInt = fill(1, length(i_ym)),
    σP0_int::Vector = fill(1, max(sum(nint_ym), 0)),
    σQ_int::Vector  = fill(1, max(sum(nint_ym), 0)),
    He::Int = 10
) where {M<:SimModel}
    # estimated covariances matrices (variance = σ²) :
    P̂0 = Diagonal{Float64}([σP0  ; σP0_int   ].^2);
    Q̂  = Diagonal{Float64}([σQ   ; σQ_int    ].^2);
    R̂  = Diagonal{Float64}(σR.^2);
    return NonLinMHE{M}(model, i_ym, nint_ym, P̂0, Q̂ , R̂, He)
end

@doc raw"""
    NonLinMHE{M<:SimModel}(model::M, i_ym, nint_ym, P̂0, Q̂, R̂, He)

Construct the estimator from the augmented covariance matrices `P̂0`, `Q̂` and `R̂`.

This syntax allows nonzero off-diagonal elements in ``\mathbf{P̂}_{-1}(0), \mathbf{Q̂, R̂}``.
"""
NonLinMHE{M}(model::M, i_ym, nint_ym, P̂0, Q̂, R̂, He) where {M<:SimModel}