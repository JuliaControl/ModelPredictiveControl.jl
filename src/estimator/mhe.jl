include("mhe/construct.jl")
include("mhe/execute.jl")
include("mhe/transcription.jl")

"Return estimation horizon He and slack variables length nε for `MovingHorizonEstimator`."
get_other_dims(estim::MovingHorizonEstimator) = (estim.He, estim.nε)

"Print optimizer and other information for `MovingHorizonEstimator`."
function print_details(io::IO, estim::MovingHorizonEstimator)
    println(io, "├ optimizer: $(JuMP.solver_name(estim.optim)) ")
    println(io, "├ transcription: $(nameof(typeof(estim.transcription)))")
    print_backends(io, estim, estim.model)
    println(io, "├ arrival covariance: $(nameof(typeof(estim.covestim))) ")
    println(io, "├ direct: $(estim.direct)")
end

"Print the differentiation backends of `MovingHorizonEstimator` for `SimModel`."
function print_backends(io::IO, estim::MovingHorizonEstimator, ::SimModel)
    println(io, "├ gradient: $(backend_str(estim.gradient))")
    println(io, "├ jacobian: $(backend_str(estim.jacobian))")
    println(io, "├ hessian: $(backend_str(estim.hessian))")
end
"No differentiation backends to print for `LinModel`."
print_backends(::IO, ::MovingHorizonEstimator, ::LinModel) = nothing

"Print the overall dimensions of the MHE `estim` with left padding `n`."
function print_estim_dim(io::IO, estim::MovingHorizonEstimator, n; firstchars=nothing)
    nu, nd = estim.model.nu, estim.model.nd
    nx̂, nym, nyu = estim.nx̂, estim.nym, estim.nyu
    He, nε = estim.He, estim.nε
    niu, niym = sum(estim.nint_u), sum(estim.nint_ym)
    println(io, "  │ ├$(lpad(He, n)) estimation steps He")
    println(io, "  │ ├$(lpad(nu, n)) manipulated inputs u ($niu integrating states)")
    println(io, "  │ ├$(lpad(nx̂, n)) estimated states x̂")
    println(io, "  │ ├$(lpad(nym, n)) measured outputs ym ($niym integrating states)")
    println(io, "  │ ├$(lpad(nyu, n)) unmeasured outputs yu")
    print(io,   "  │ └$(lpad(nd, n)) measured disturbances d")
    if isnothing(firstchars) # the user prints the MHE object itself, not a controller:
        nZ̃, nε = length(estim.Z̃), estim.nε
        nA, nAeq     = sum(estim.con.i_b), size(estim.con.Aeq, 1)
        ng, nc, neq  = sum(estim.con.i_g), estim.con.nc, estim.con.neq
        m = maximum(ndigits.((nZ̃, nA, ng))) + 1
        i_nZ̃min, i_nZ̃max = @. !isinf(estim.con.Z̃min), !isinf(estim.con.Z̃max)
        nZ̃bounds = sum(i_nZ̃min) + sum(i_nZ̃max)
        println(io)
        println(io, "  └ optimization:")
        println(io, "    ├$(lpad(nZ̃, m)) decision variables Z̃ ($nε slack variable, $nZ̃bounds bounds)")
        println(io, "    ├$(lpad(nA, m)) linear inequality constraints A")
        println(io, "    ├$(lpad(nAeq, m)) linear equality constraints Aeq")
        println(io, "    ├$(lpad(ng, m)) nonlinear inequality constraints g ($nc custom)")
        print(io,   "    └$(lpad(neq, m)) nonlinear equality constraints geq")
    end
end