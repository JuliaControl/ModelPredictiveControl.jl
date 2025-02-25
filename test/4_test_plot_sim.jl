@testitem "SimModel quick simulation" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    model = LinModel(sys, Ts, i_d=[3])
    res = sim!(model, 15)

    @test repr(res) == "Simulation results of LinModel with 15 time steps."
    @test isa(res.obj, LinModel)
    @test length(res.T_data) == 15
    @test res.U_data[:, 1] ≈ model.uop .+ 1
    @test res.D_data[:, 1] ≈ model.dop
    @test res.X_data[:, 1] ≈ zeros(model.nx)

    res_man = SimResult(model, res.U_data, res.Y_data, res.D_data; X_data=res.X_data)
    @test res_man.U_data ≈ res.U_data
    @test res_man.Y_data ≈ res.Y_data
    @test res_man.D_data ≈ res.D_data
    @test res_man.X_data ≈ res.X_data

    @test_throws ArgumentError SimResult(model, [res.U_data model.uop], res.Y_data, res.D_data)
end

@testitem "SimModel Plots" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra, Plots
    model = LinModel(sys, Ts, i_d=[3])
    res = sim!(model, 15, [1, 3], [-10])
    p = plot(res, plotx=true)
    @test p[1][1][:x] ≈ res.T_data
    @test p[1][1][:y] ≈ res.Y_data[1, :]
    @test p[2][1][:y] ≈ res.Y_data[2, :]
    @test p[3][1][:y][1:2:end] ≈ res.U_data[1, :]
    @test p[4][1][:y][1:2:end] ≈ res.U_data[2, :]
    @test p[5][1][:y] ≈ res.D_data[1, :]
    @test p[6][1][:y] ≈ res.X_data[1, :]
    @test p[7][1][:y] ≈ res.X_data[2, :]
    @test p[8][1][:y] ≈ res.X_data[3, :]
    @test p[9][1][:y] ≈ res.X_data[4, :]
    p = plot(res, ploty=[2])
    @test p[1][1][:x] ≈ res.T_data
    @test p[1][1][:y] ≈ res.Y_data[2, :]
    p = plot(res, ploty=false, plotu=false, plotd=false, plotx=2:4)
    @test p[1][1][:x] ≈ res.T_data
    @test p[1][1][:y] ≈ res.X_data[2, :]
    @test p[2][1][:y] ≈ res.X_data[3, :]
    @test p[3][1][:y] ≈ res.X_data[4, :]
end

@testitem "StateEstimator quick simulation" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    estim = SteadyKalmanFilter(LinModel(sys, Ts, i_d=[3]))
    res = sim!(estim, 15)
    @test isa(res.obj, SteadyKalmanFilter)
    @test length(res.T_data) == 15
    @test res.U_data[:, 1]  ≈ estim.model.uop .+ 1
    @test res.Ud_data[:, 1] ≈ estim.model.uop .+ 1
    @test res.D_data[:, 1]  ≈ estim.model.dop
    @test res.X_data[:, 1]  ≈ zeros(estim.model.nx)
    @test res.X̂_data[:, 1]  ≈ zeros(estim.nx̂)

    res_man = SimResult(
        estim, res.U_data, res.Y_data, res.D_data; 
        X_data=res.X_data, X̂_data=res.X̂_data
    )
    @test res_man.U_data ≈ res.U_data
    @test res_man.Y_data ≈ res.Y_data
    @test res_man.D_data ≈ res.D_data
    @test res_man.X_data ≈ res.X_data
    @test res_man.X̂_data ≈ res.X̂_data
end

@testitem "StateEstimator Plots" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra, Plots
    estim = MovingHorizonEstimator(LinModel(sys, Ts, i_d=[3]), He=5)
    estim = setconstraint!(estim, x̂min=[-100,-101,-102,-103,-Inf,-Inf])
    estim = setconstraint!(estim, x̂max=[+100,+101,+102,+103,+Inf,+Inf])
    res = sim!(estim, 15, [1, 3], [-10])
    p1 = plot(res, plotx=true)
    @test p1[1][1][:x] ≈ res.T_data
    @test p1[end-3][1][:y] ≈ res.X_data[1,:]
    @test p1[end-2][1][:y] ≈ res.X_data[2,:]
    @test p1[end-1][1][:y] ≈ res.X_data[3,:]
    @test p1[end-0][1][:y] ≈ res.X_data[4,:]
    p2 = plot(res, plotx̂=true, plotx̂min=false, plotx̂max=false)
    @test p2[1][1][:x] ≈ res.T_data
    @test p2[end-5][1][:y] ≈ res.X̂_data[1,:]
    @test p2[end-4][1][:y] ≈ res.X̂_data[2,:]
    @test p2[end-3][1][:y] ≈ res.X̂_data[3,:]
    @test p2[end-2][1][:y] ≈ res.X̂_data[4,:]
    @test p2[end-1][1][:y] ≈ res.X̂_data[5,:]
    @test p2[end-0][1][:y] ≈ res.X̂_data[6,:]
    p3 = plot(res, plotxwithx̂=true, plotx̂min=false, plotx̂max=false)
    @test p3[1][1][:x] ≈ res.T_data
    @test p3[end-5][1][:y] ≈ res.X_data[1,:]
    @test p3[end-5][2][:y] ≈ res.X̂_data[1,:]
    @test p3[end-4][1][:y] ≈ res.X_data[2,:]
    @test p3[end-4][2][:y] ≈ res.X̂_data[2,:]
    @test p3[end-3][1][:y] ≈ res.X_data[3,:]
    @test p3[end-3][2][:y] ≈ res.X̂_data[3,:]
    @test p3[end-2][1][:y] ≈ res.X_data[4,:]
    @test p3[end-2][2][:y] ≈ res.X̂_data[4,:]
    @test p3[end-1][1][:y] ≈ res.X̂_data[5,:]
    @test p3[end-0][1][:y] ≈ res.X̂_data[6,:]
    p4 = plot(res, plotx̂=true, plotx̂min=true, plotx̂max=false)
    @test p4[1][1][:x] ≈ res.T_data
    @test all(p4[end-5][2][:y] .≈ -100) 
    @test all(p4[end-4][2][:y] .≈ -101)
    @test all(p4[end-3][2][:y] .≈ -102)
    @test all(p4[end-2][2][:y] .≈ -103)
    p5 = plot(res, plotx̂=true, plotx̂min=false, plotx̂max=true)
    @test p5[1][1][:x] ≈ res.T_data
    @test all(p5[end-5][2][:y] .≈ +100)
    @test all(p5[end-4][2][:y] .≈ +101)
    @test all(p5[end-3][2][:y] .≈ +102)
    @test all(p5[end-2][2][:y] .≈ +103)
    p6 = plot(res, plotxwithx̂=true, plotx̂min=true, plotx̂max=false)
    @test p6[1][1][:x] ≈ res.T_data
    @test all(p6[end-5][3][:y] .≈ -100)
    @test all(p6[end-4][3][:y] .≈ -101)
    @test all(p6[end-3][3][:y] .≈ -102)
    @test all(p6[end-2][3][:y] .≈ -103)
    p7 = plot(res, plotxwithx̂=true, plotx̂min=false, plotx̂max=true)
    @test p7[1][1][:x] ≈ res.T_data
    @test all(p7[end-5][3][:y] .≈ +100)
    @test all(p7[end-4][3][:y] .≈ +101)
    @test all(p7[end-3][3][:y] .≈ +102)
    @test all(p7[end-2][3][:y] .≈ +103)
end

@testitem "PredictiveController quick simulation" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    mpc1 = LinMPC(LinModel(sys, Ts, i_d=[3]))
    res = sim!(mpc1, 15)
    @test isa(res.obj, LinMPC)
    @test length(res.T_data) == 15
    @test res.Ry_data[:, 1] ≈ mpc1.estim.model.yop .+ 1
    @test res.Ud_data ≈ res.U_data
    @test res.D_data[:, 1]  ≈ mpc1.estim.model.dop
    @test res.X_data[:, 1]  ≈ zeros(mpc1.estim.model.nx)
    @test res.X̂_data[:, 1]  ≈ zeros(mpc1.estim.nx̂)

    mpc2 = ExplicitMPC(LinModel(sys, Ts, i_d=[3]))
    res = sim!(mpc2, 15)
    @test isa(res.obj, ExplicitMPC)
    @test length(res.T_data) == 15
    @test res.Ry_data[:, 1] ≈ mpc2.estim.model.yop .+ 1
    @test res.Ud_data ≈ res.U_data
    @test res.D_data[:, 1]  ≈ mpc2.estim.model.dop
    @test res.X_data[:, 1]  ≈ zeros(mpc2.estim.model.nx)
    @test res.X̂_data[:, 1]  ≈ zeros(mpc2.estim.nx̂)

    res_man = SimResult(
        mpc1, res.U_data, res.Y_data, res.D_data; 
        X_data=res.X_data, X̂_data=res.X̂_data,
        Ry_data=res.Ry_data
    )
    @test res_man.U_data ≈ res.U_data
    @test res_man.Y_data ≈ res.Y_data
    @test res_man.D_data ≈ res.D_data
    @test res_man.X_data ≈ res.X_data
    @test res_man.X̂_data ≈ res.X̂_data
    @test res_man.Ry_data ≈ res.Ry_data
end

@testitem "PredictiveController Plots" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra, Plots
    estim = MovingHorizonEstimator(LinModel(sys, Ts, i_d=[3]), He=5)
    estim = setconstraint!(estim, x̂min=[-100,-101,-102,-103,-104,-105])
    estim = setconstraint!(estim, x̂max=[+100,+101,+102,+103,+104,+105])
    mpc = LinMPC(estim, Lwt=[0.01, 0.01])
    mpc = setconstraint!(mpc, umin=[-50, -51], umax=[52, 53], ymin=[-54,-55], ymax=[56,57])
    res = sim!(mpc, 15)
    p1 = plot(res, plotŷ=true)
    @test p1[1][1][:x] ≈ res.T_data
    @test p1[1][1][:y] ≈ res.Y_data[1,:]
    @test p1[1][2][:y] ≈ res.Ŷ_data[1,:]
    @test p1[2][1][:y] ≈ res.Y_data[2,:]
    @test p1[2][2][:y] ≈ res.Ŷ_data[2,:]
    p2 = plot(res, plotx=true)
    @test p2[1][1][:x] ≈ res.T_data
    @test p2[end-3][1][:y] ≈ res.X_data[1,:]
    @test p2[end-2][1][:y] ≈ res.X_data[2,:]
    @test p2[end-1][1][:y] ≈ res.X_data[3,:]
    @test p2[end-0][1][:y] ≈ res.X_data[4,:]
    p3 = plot(res, plotx̂=true)
    @test p3[1][1][:x] ≈ res.T_data
    @test p3[end-5][1][:y] ≈ res.X̂_data[1,:]
    @test p3[end-4][1][:y] ≈ res.X̂_data[2,:]
    @test p3[end-3][1][:y] ≈ res.X̂_data[3,:]
    @test p3[end-2][1][:y] ≈ res.X̂_data[4,:]
    @test p3[end-1][1][:y] ≈ res.X̂_data[5,:]
    @test p3[end-0][1][:y] ≈ res.X̂_data[6,:]
    p4 = plot(res, plotxwithx̂=true)
    @test p4[1][1][:x] ≈ res.T_data
    @test p4[end-5][1][:y] ≈ res.X_data[1,:]
    @test p4[end-5][2][:y] ≈ res.X̂_data[1,:]
    @test p4[end-4][1][:y] ≈ res.X_data[2,:]
    @test p4[end-4][2][:y] ≈ res.X̂_data[2,:]
    @test p4[end-3][1][:y] ≈ res.X_data[3,:]
    @test p4[end-3][2][:y] ≈ res.X̂_data[3,:]
    @test p4[end-2][1][:y] ≈ res.X_data[4,:]
    @test p4[end-2][2][:y] ≈ res.X̂_data[4,:]
    @test p4[end-1][1][:y] ≈ res.X̂_data[5,:]
    @test p4[end-0][1][:y] ≈ res.X̂_data[6,:]
    p5 = plot(res, plotumin=true, plotumax=false, plotymin=false, plotymax=false)
    @test p5[1][1][:x] ≈ res.T_data
    @test all(p5[end-2][3][:y] .≈ -50.0)
    @test all(p5[end-1][3][:y] .≈ -51.0)
    p6 = plot(res, plotumin=false, plotumax=true, plotymin=false, plotymax=false)
    @test p6[1][1][:x] ≈ res.T_data
    @test all(p6[end-2][3][:y] .≈ 52.0)
    @test all(p6[end-1][3][:y] .≈ 53.0)
    p7 = plot(res, plotumin=false, plotumax=false, plotymin=true, plotymax=false)
    @test p7[1][1][:x] ≈ res.T_data
    @test all(p7[end-4][3][:y] .≈ -54.0)
    @test all(p7[end-3][3][:y] .≈ -55.0)
    p8 = plot(res, plotumin=false, plotumax=false, plotymin=false, plotymax=true)
    @test p8[1][1][:x] ≈ res.T_data
    @test all(p8[end-4][3][:y] .≈ 56.0)
    @test all(p8[end-3][3][:y] .≈ 57.0)
    p9 = plot(res, plotx̂=true, plotx̂min=true, plotx̂max=false)
    @test p9[1][1][:x] ≈ res.T_data
    @test all(p9[end-5][2][:y] .≈ -100.0)
    @test all(p9[end-4][2][:y] .≈ -101.0)
    @test all(p9[end-3][2][:y] .≈ -102.0)
    @test all(p9[end-2][2][:y] .≈ -103.0)
    @test all(p9[end-1][2][:y] .≈ -104.0)
    @test all(p9[end-0][2][:y] .≈ -105.0)
    p10 = plot(res, plotx̂=true, plotx̂min=false, plotx̂max=true)
    @test p10[1][1][:x] ≈ res.T_data
    @test all(p10[end-5][2][:y] .≈ +100.0)
    @test all(p10[end-4][2][:y] .≈ +101.0)
    @test all(p10[end-3][2][:y] .≈ +102.0)
    @test all(p10[end-2][2][:y] .≈ +103.0)
    @test all(p10[end-1][2][:y] .≈ +104.0)
    @test all(p10[end-0][2][:y] .≈ +105.0)
    p11 = plot(res, plotxwithx̂=true, plotx̂min=true, plotx̂max=false)
    @test p11[1][1][:x] ≈ res.T_data
    @test all(p11[end-5][3][:y] .≈ -100.0)
    @test all(p11[end-4][3][:y] .≈ -101.0)
    @test all(p11[end-3][3][:y] .≈ -102.0)
    @test all(p11[end-2][3][:y] .≈ -103.0)
    @test all(p11[end-1][2][:y] .≈ -104.0)
    @test all(p11[end-0][2][:y] .≈ -105.0)
    p12 = plot(res, plotxwithx̂=true, plotx̂min=false, plotx̂max=true)
    @test p12[1][1][:x] ≈ res.T_data
    @test all(p12[end-5][3][:y] .≈ +100.0)
    @test all(p12[end-4][3][:y] .≈ +101.0)
    @test all(p12[end-3][3][:y] .≈ +102.0)
    @test all(p12[end-2][3][:y] .≈ +103.0)
    @test all(p12[end-1][2][:y] .≈ +104.0)
    @test all(p12[end-0][2][:y] .≈ +105.0)
end