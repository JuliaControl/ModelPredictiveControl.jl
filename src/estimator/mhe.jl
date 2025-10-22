include("mhe/construct.jl")
include("mhe/execute.jl")
include("mhe/legacy.jl")

"Print optimizer and other information for `MovingHorizonEstimator`."
function print_details(io::IO, estim::MovingHorizonEstimator)
    println(io, "├ optimizer: $(JuMP.solver_name(estim.optim)) ")
    print_backends(io, estim, estim.model)
    println(io, "├ arrival covariance: $(nameof(typeof(estim.covestim))) ")
end

"Print the differentiation backends for `SimModel`."
function print_backends(io::IO, estim::MovingHorizonEstimator, ::SimModel)
    println(io, "├ gradient: $(backend_str(estim.gradient))")
    println(io, "├ jacobian: $(backend_str(estim.jacobian))")
end
"No differentiation backends to print for `LinModel`."
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