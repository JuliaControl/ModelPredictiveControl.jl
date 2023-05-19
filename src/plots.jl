struct SimResult
    T_data ::Vector{Float64}
    Y_data ::Matrix{Float64}
    Ry_data::Matrix{Float64}
    Ŷ_data ::Matrix{Float64}
    U_data ::Matrix{Float64}
    Ru_data::Matrix{Float64}
    D_data ::Matrix{Float64}
    X_data ::Matrix{Float64}
    X̂_data ::Matrix{Float64}
end

#=
@recipe function f(res::SimResult)
    y = map(xi -> pdf(dist,xi), x)
    seriestype --> :path  # there is always an attribute dictionary `d` available...
                          # If the user didn't specify a seriestype, we choose :path
    return x, y
end
=#
