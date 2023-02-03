struct UnscentedKalmanFilter <: StateEstimator
    model::SimModel
    x̂::Vector{Float64}
    P̂::Matrix{Float64}
    i_ym::IntRangeOrVector
    nx̂::Int
    nym::Int
    nyu::Int
    nxs::Int
    As::Matrix{Float64}
    Cs::Matrix{Float64}
    nint_ym::Vector{Int}
    f̂::Function
    ĥ::Function
    P̂0::Union{Diagonal{Float64}, Matrix{Float64}}
    Q̂::Union{Diagonal{Float64}, Matrix{Float64}}
    R̂::Union{Diagonal{Float64}, Matrix{Float64}}
    function UnscentedKalmanFilter(model, i_ym, nint_ym, Asm, Csm, P̂0 ,Q̂, R̂)
        nx, ny = model.nx, model.ny
        nym, nyu = length(i_ym), ny - length(i_ym)
        nxs = size(Asm,1)
        nx̂ = nx + nxs
        validate_kfcov(nym, nx̂, Q̂, R̂)
        As, _ , Cs, _  = stoch_ym2y(model, i_ym, Asm, [], Csm, [])
        Â, B̂u, Ĉ, B̂d, D̂d = augment_model(model, As, Cs)
        Ĉm, D̂dm = Ĉ[i_ym, :], D̂d[i_ym, :] # measured outputs ym only
        f̂(x̂,u,d) = Â*x̂ + B̂u*u + B̂d*d
        ĥ(x̂,d) = Ĉ*x̂ + D̂d*d
        K = try
            kalman(Discrete, Â, Ĉm, Matrix(Q̂), Matrix(R̂)) # Matrix() required for Julia 1.6
        catch my_error
            if isa(my_error, ErrorException)
                error("Cannot compute the optimal Kalman gain K for the "* 
                      "SteadyKalmanFilter. You may try to remove integrators with nint_ym "*
                      "parameter or use the time-varying KalmanFilter.")
            end
        end
        x̂ = [copy(model.x); zeros(nxs)]
        return new(
            model, 
            x̂,
            i_ym, nx̂, nym, nyu, nxs, 
            As, Cs, nint_ym,
            Â, B̂u, B̂d, Ĉ, D̂d, 
            Ĉm, D̂dm,
            f̂, ĥ,
            Q̂, R̂,
            K
        )
    end