abstract type StateEstimator end

abstract type StateEstimate end

include("estimator/internal_model.jl")
#include("estimator/kalman.jl")

function Base.show(io::IO, estim::StateEstimator)
    println(io, "$(typeof(estim)) state estimator with "*
                "a sample time Ts = $(estim.model.Ts) s and:")
    println(io, " $(estim.model.nu) manipulated inputs u")
    println(io, " $(estim.nx̂) states x̂")
    println(io, " $(estim.nym) measured outputs ym")
    println(io, " $(estim.nyu) unmeasured outputs yu")
    print(io,   " $(estim.model.nd) measured disturbances d")
end

function validate_ym(model::SimModel, i_ym)
    if length(unique(i_ym)) ≠ length(i_ym) || maximum(i_ym) > model.ny
        error("Measured output indices i_ym should contains valid and unique indices")
    end
end