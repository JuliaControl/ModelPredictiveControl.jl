include("mhe/construct.jl")
include("mhe/execute.jl")

function Base.show(io::IO, estim::MovingHorizonEstimator)
    model = estim.model
    nu, nd = model.nu, model.nd
    nx̂, nym, nyu = estim.nx̂, estim.nym, estim.nyu
    n = maximum(ndigits.((nu, nx̂, nym, nyu, nd))) + 1
    println(io, "$(nameof(typeof(estim))) estimator with a sample time Ts = $(model.Ts) s:")
    println(io, "├ model: $(nameof(typeof(model)))")
    println(io, "├ optimizer: $(JuMP.solver_name(estim.optim)) ")
    print_backends(io, estim, model)
    println(io, "└ dimensions:")
    print_estim_dim(io, estim, n)
end

function print_backends(io::IO, estim::MovingHorizonEstimator, ::SimModel)
    println(io, "├ gradient: $(backend_str(estim.gradient))")
    println(io, "├ jacobian: $(backend_str(estim.jacobian))")
end
print_backends(::IO, ::MovingHorizonEstimator, ::LinModel) = nothing


"Print the overall dimensions of the MHE `estim` with left padding `n`."
function print_estim_dim(io::IO, estim::MovingHorizonEstimator, n)
    nu, nd = estim.model.nu, estim.model.nd
    nx̂, nym, nyu = estim.nx̂, estim.nym, estim.nyu
    He, nε = estim.He, estim.nε
    niu, niym = sum(estim.nint_u), sum(estim.nint_ym)
    println(io, "  ├$(lpad(He, n)) estimation steps He")
    println(io, "  ├$(lpad(nε, n)) slack variable ε (estimation constraints)")
    println(io, "  ├$(lpad(nu, n)) manipulated inputs u ($niu integrating states)")
    println(io, "  ├$(lpad(nx̂, n)) estimated states x̂")
    println(io, "  ├$(lpad(nym, n)) measured outputs ym ($niym integrating states)")
    println(io, "  ├$(lpad(nyu, n)) unmeasured outputs yu")
    print(io,   "  └$(lpad(nd, n)) measured disturbances d")
end