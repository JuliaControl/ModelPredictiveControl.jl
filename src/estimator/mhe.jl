@doc raw"""
Include all the data for the constraints of [`MovingHorizonEstimator`](@ref).

The bounds on the estimated state at arrival ``\mathbf{x̂}_k(k-N_k+1)`` is separated from
the other state constraints ``\mathbf{x̂}_k(k-N_k+2), \mathbf{x̂}_k(k-N_k+3), ...`` since
the former is always a linear inequality constraint (it's a decision variable). The fields
`x̂min` and `x̂max` refer to the bounds at the arrival, and `X̂min` and `X̂max`, the others.
"""
struct EstimatorConstraint{NT<:Real}
    Ẽx̂      ::Matrix{NT}
    Fx̂      ::Vector{NT}
    Gx̂      ::Matrix{NT}
    Jx̂      ::Matrix{NT}
    x̂min    ::Vector{NT}
    x̂max    ::Vector{NT}
    X̂min    ::Vector{NT}
    X̂max    ::Vector{NT}
    Ŵmin    ::Vector{NT}
    Ŵmax    ::Vector{NT}
    V̂min    ::Vector{NT}
    V̂max    ::Vector{NT}
    A_x̂min  ::Matrix{NT}
    A_x̂max  ::Matrix{NT}
    A_X̂min  ::Matrix{NT}
    A_X̂max  ::Matrix{NT}
    A_Ŵmin  ::Matrix{NT}
    A_Ŵmax  ::Matrix{NT}
    A_V̂min  ::Matrix{NT}
    A_V̂max  ::Matrix{NT}
    A       ::Matrix{NT}
    b       ::Vector{NT}
    i_b     ::BitVector
    i_g     ::BitVector
end

struct MovingHorizonEstimator{
    NT<:Real, 
    SM<:SimModel, 
    JM<:JuMP.GenericModel
} <: StateEstimator{NT}
    model::SM
    # note: `NT` and the number type `JNT` in `JuMP.GenericModel{JNT}` can be
    # different since solvers that support non-Float64 are scarce.
    optim::JM
    con::EstimatorConstraint{NT}
    Z̃::Vector{NT}
    lastu0::Vector{NT}
    x̂::Vector{NT}
    He::Int
    i_ym::Vector{Int}
    nx̂ ::Int
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
    Ẽ ::Matrix{NT}
    F ::Vector{NT}
    G ::Matrix{NT}
    J ::Matrix{NT}
    ẽx̄::Matrix{NT}
    fx̄::Vector{NT}
    H̃::Hermitian{NT, Matrix{NT}}
    q̃::Vector{NT}
    p::Vector{NT}
    P̂0::Hermitian{NT, Matrix{NT}}
    Q̂::Hermitian{NT, Matrix{NT}}
    R̂::Hermitian{NT, Matrix{NT}}
    invP̄::Hermitian{NT, Matrix{NT}}
    invQ̂_He::Hermitian{NT, Matrix{NT}}
    invR̂_He::Hermitian{NT, Matrix{NT}}
    M̂::Matrix{NT}
    X̂ ::Union{Vector{NT}, Missing} 
    Ym::Union{Vector{NT}, Missing}
    U ::Union{Vector{NT}, Missing}
    D ::Union{Vector{NT}, Missing}
    Ŵ ::Union{Vector{NT}, Missing}
    x̂arr_old::Vector{NT}
    P̂arr_old::Hermitian{NT, Matrix{NT}}
    Nk::Vector{Int}
    function MovingHorizonEstimator{NT, SM, JM}(
        model::SM, He, i_ym, nint_u, nint_ym, P̂0, Q̂, R̂, optim::JM
    ) where {NT<:Real, SM<:SimModel{NT}, JM<:JuMP.GenericModel}
        nu, nd = model.nu, model.nd
        He < 1  && throw(ArgumentError("Estimation horizon He should be ≥ 1"))
        nym, nyu = validate_ym(model, i_ym)
        As, Cs_u, Cs_y, nint_u, nint_ym = init_estimstoch(model, i_ym, nint_u, nint_ym)
        nxs = size(As, 1)
        nx̂  = model.nx + nxs
        nŵ = nx̂
        Â, B̂u, Ĉ, B̂d, D̂d = augment_model(model, As, Cs_u, Cs_y)
        validate_kfcov(nym, nx̂, Q̂, R̂, P̂0)
        lastu0 = zeros(NT, model.nu)
        x̂ = [zeros(NT, model.nx); zeros(NT, nxs)]
        P̂0 = Hermitian(P̂0, :L)
        Q̂, R̂ = Hermitian(Q̂, :L),  Hermitian(R̂, :L)
        invP̄ = Hermitian(inv(P̂0), :L)
        invQ̂_He = Hermitian(repeatdiag(inv(Q̂), He), :L)
        invR̂_He = Hermitian(repeatdiag(inv(R̂), He), :L)
        M̂ = zeros(NT, nx̂, nym)
        E, F, G, J, ex̄, fx̄, Ex̂, Fx̂, Gx̂, Jx̂ = init_predmat_mhe(
            model, He, i_ym, Â, B̂u, Ĉ, B̂d, D̂d
        )
        con, Ẽ, ẽx̄ = init_defaultcon_mhe(model, He, nx̂, nym, E, ex̄, Ex̂, Fx̂, Gx̂, Jx̂)
        nZ̃ = nx̂ + nŵ*He
        # dummy values, updated before optimization:
        H̃, q̃, p = Hermitian(zeros(NT, nZ̃, nZ̃), :L), zeros(NT, nZ̃), zeros(NT, 1)
        Z̃ = zeros(NT, nZ̃)
        X̂, Ym   = zeros(NT, nx̂*He), zeros(NT, nym*He)
        U, D, Ŵ = zeros(NT, nu*He), zeros(NT, nd*He), zeros(NT, nx̂*He)
        x̂arr_old = zeros(NT, nx̂)
        P̂arr_old = copy(P̂0)
        Nk = [0]
        estim = new{NT, SM, JM}(
            model, optim, con, 
            Z̃, lastu0, x̂, 
            He,
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs_u, Cs_y, nint_u, nint_ym,
            Â, B̂u, Ĉ, B̂d, D̂d,
            Ẽ, F, G, J, ẽx̄, fx̄,
            H̃, q̃, p,
            P̂0, Q̂, R̂, invP̄, invQ̂_He, invR̂_He,
            M̂,
            X̂, Ym, U, D, Ŵ, 
            x̂arr_old, P̂arr_old, Nk
        )
        init_optimization!(estim, model, optim)
        return estim
    end
end

function Base.show(io::IO, estim::MovingHorizonEstimator)
    nu, nd = estim.model.nu, estim.model.nd
    nx̂, nym, nyu = estim.nx̂, estim.nym, estim.nyu
    n = maximum(ndigits.((nu, nx̂, nym, nyu, nd))) + 1
    println(io, "$(typeof(estim).name.name) estimator with a sample time "*
                "Ts = $(estim.model.Ts) s, $(solver_name(estim.optim)) optimizer, "*
                "$(typeof(estim.model).name.name) and:")
    print_estim_dim(io, estim, n)
end

"Print the overall dimensions of the state estimator `estim` with left padding `n`."
function print_estim_dim(io::IO, estim::MovingHorizonEstimator, n)
    nu, nd = estim.model.nu, estim.model.nd
    nx̂, nym, nyu = estim.nx̂, estim.nym, estim.nyu
    He = estim.He
    println(io, "$(lpad(He, n)) estimation steps He")
    println(io, "$(lpad(nu, n)) manipulated inputs u ($(sum(estim.nint_u)) integrating states)")
    println(io, "$(lpad(nx̂, n)) states x̂")
    println(io, "$(lpad(nym, n)) measured outputs ym ($(sum(estim.nint_ym)) integrating states)")
    println(io, "$(lpad(nyu, n)) unmeasured outputs yu")
    print(io,   "$(lpad(nd, n)) measured disturbances d")
end

include("mhe/construct.jl")
include("mhe/execute.jl")