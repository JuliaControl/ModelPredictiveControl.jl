include("mhe/construct.jl")
include("mhe/execute.jl")

function Base.show(io::IO, estim::MovingHorizonEstimator)
    nu, nd = estim.model.nu, estim.model.nd
    nx̂, nym, nyu = estim.nx̂, estim.nym, estim.nyu
    n = maximum(ndigits.((nu, nx̂, nym, nyu, nd))) + 1
    println(io, "$(typeof(estim).name.name) estimator with a sample time "*
                "Ts = $(estim.model.Ts) s, $(solver_name(estim.optim)) optimizer, "*
                "$(typeof(estim.model).name.name) and:")
    print_estim_dim(io, estim, n)
end

"Print the overall dimensions of the MHE `estim` with left padding `n`."
function print_estim_dim(io::IO, estim::MovingHorizonEstimator, n)
    nu, nd = estim.model.nu, estim.model.nd
    nx̂, nym, nyu = estim.nx̂, estim.nym, estim.nyu
    He = estim.He
    nϵ = isinf(estim.C) ? 0 : 1
    println(io, "$(lpad(He, n)) estimation steps He")
    println(io, "$(lpad(nϵ, n)) slack variable ϵ (estimation constraints)")
    println(io, "$(lpad(nu, n)) manipulated inputs u ($(sum(estim.nint_u)) integrating states)")
    println(io, "$(lpad(nx̂, n)) estimated states x̂")
    println(io, "$(lpad(nym, n)) measured outputs ym ($(sum(estim.nint_ym)) integrating states)")
    println(io, "$(lpad(nyu, n)) unmeasured outputs yu")
    print(io,   "$(lpad(nd, n)) measured disturbances d")
end