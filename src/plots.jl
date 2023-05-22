struct SimResult{C<:PredictiveController}
    mpc    ::C
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


@recipe function simresultplot(
    res::SimResult; 
    plotRy          = true,
    plotŶminŶmax    = true,
    plotŶ           = false,
    plotRu          = true,
    plotUminUmax    = true,
    plotD           = true
)

    mpc = res.mpc
    t   = res.T_data
    Ns  = length(t)


    # set up the subplots
    # legend := false
    # grid := false
    ny = size(res.Y_data, 1)
    nu = size(res.U_data, 1)
    nd = size(res.D_data, 1)


    layout := @layout (nd ≠ 0 && plotD) ? [(ny,1) (nu,1) (nd, 1)] : [(ny,1) (nu,1)]

    # these are common to both marginal histograms
    #fillcolor := :black
    
    subplot_base = 0
    for i in 1:ny
        @series begin
            xguide  --> "Time (s)"
            yguide  --> "\$y_$i\$"
            color   --> 1
            subplot --> subplot_base + i
            label   --> "\$\\mathbf{y}\$"
            t, res.Y_data[i, :]
        end
        if plotRy && !iszero(mpc.M_Hp)
            @series begin
                xguide    --> "Time (s)"
                yguide    --> "\$y_$i\$"
                color     --> 2
                subplot   --> subplot_base + i
                linestyle --> :dash
                label     --> "\$\\mathbf{r_y}\$"
                t, res.Ry_data[i, :]
            end
        end
        if plotŶminŶmax && !isinf(mpc.con.Ŷmin[i])
            @series begin
                xguide    --> "Time (s)"
                yguide    --> "\$y_$i\$"
                color     --> 3
                subplot   --> subplot_base + i
                linestyle --> :dot
                linewidth --> 2.0
                label     --> "\$\\mathbf{\\hat{y}_{min}}\$"
                t, fill(mpc.con.Ŷmin[i], Ns)
            end
        end
        if plotŶminŶmax && !isinf(mpc.con.Ŷmax[i])
            @series begin
                xguide    --> "Time (s)"
                yguide    --> "\$y_$i\$"
                color     --> 4
                subplot   --> subplot_base + i
                linestyle --> :dot
                linewidth --> 2.0
                label     --> "\$\\mathbf{\\hat{y}_{max}}\$"
                t, fill(mpc.con.Ŷmax[i], Ns)
            end
        end
        if plotŶ
            @series begin
                xguide    --> "Time (s)"
                yguide    --> "\$y_$i\$"
                color     --> 5
                subplot   --> subplot_base + i
                linestyle --> :dashdot
                label     --> "\$\\mathbf{\\hat{y}}\$"
                t, res.Ŷ_data[i, :]
            end
        end
    end
    subplot_base += ny
    for i in 1:nu
        @series begin
            xguide     --> "Time (s)"
            yguide     --> "\$u_$i\$"
            color      --> 1
            subplot    --> subplot_base + i
            seriestype --> :steppost
            label      --> "\$\\mathbf{u}\$"
            t, res.U_data[i, :]
        end
        if plotRu && !iszero(mpc.L_Hp)
            @series begin
                xguide    --> "Time (s)"
                yguide    --> "\$u_$i\$"
                color     --> 2
                subplot   --> subplot_base + i
                linestyle --> :dash
                label     --> "\$\\mathbf{r_{u}}\$"
                t, res.Ry_data[i, :]
            end
        end
        if plotUminUmax && !isinf(mpc.con.Umin[i])
            @series begin
                xguide    --> "Time (s)"
                yguide    --> "\$u_$i\$"
                color     --> 3
                subplot   --> subplot_base + i
                linestyle --> :dot
                linewidth --> 2.0
                label     --> "\$\\mathbf{u_{min}}\$"
                t, fill(mpc.con.Umin[i], Ns)
            end
        end
        if plotUminUmax && !isinf(mpc.con.Umax[i])
            @series begin
                xguide    --> "Time (s)"
                yguide    --> "\$u_$i\$"
                color     --> 4
                subplot   --> subplot_base + i
                linestyle --> :dot
                linewidth --> 2.0
                label     --> "\$\\mathbf{u_{max}}\$"
                t, fill(mpc.con.Umax[i], Ns)
            end
        end
    end
    subplot_base += nu
    if plotD
        for i in 1:nd
            @series begin
                xguide  --> "Time (s)"
                yguide  --> "\$d_$i\$"
                color   --> 1
                subplot --> subplot_base + i
                label   --> ""
                t, res.D_data[i, :]
            end
        end
    end
end

