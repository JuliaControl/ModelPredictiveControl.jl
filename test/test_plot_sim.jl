Ts = 4.0
sys = [ tf(1.90,[18.0,1])   tf(1.90,[18.0,1])   tf(1.90,[18.0,1]);
        tf(-0.74,[8.0,1])   tf(0.74,[8.0,1])    tf(-0.74,[8.0,1])   ]

@testset "SimModel quick simulation" begin
    model = LinModel(sys, Ts, i_d=[3])
    res = sim!(model, 15)
    @test isa(res.obj, LinModel)
    @test length(res.T_data) == 15
    @test res.U_data[:, 1] ≈ model.uop .+ 1
    @test res.D_data[:, 1] ≈ model.dop
    @test res.X_data[:, 1] ≈ zeros(model.nx)
end

@testset "SimModel Plots" begin
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
end

@testset "StateEstimator quick simulation" begin
    estim = SteadyKalmanFilter(LinModel(sys, Ts, i_d=[3]))
    res = sim!(estim, 15)
    @test isa(res.obj, SteadyKalmanFilter)
    @test length(res.T_data) == 15
    @test res.U_data[:, 1]  ≈ estim.model.uop .+ 1
    @test res.Ud_data[:, 1] ≈ estim.model.uop .+ 1
    @test res.D_data[:, 1]  ≈ estim.model.dop
    @test res.X_data[:, 1]  ≈ zeros(estim.model.nx)
    @test res.X̂_data[:, 1]  ≈ zeros(estim.nx̂)
end

@testset "StateEstimator Plots" begin
    estim = SteadyKalmanFilter(LinModel(sys, Ts, i_d=[3]))
    res = sim!(estim, 15, [1, 3], [-10])
    p1 = plot(res, plotx=true)
    @test p1[1][1][:x] ≈ res.T_data
    @test p1[end-3][1][:y] ≈ res.X_data[1,:]
    @test p1[end-2][1][:y] ≈ res.X_data[2,:]
    @test p1[end-1][1][:y] ≈ res.X_data[3,:]
    @test p1[end-0][1][:y] ≈ res.X_data[4,:]
    p2 = plot(res, plotx̂=true)
    @test p2[1][1][:x] ≈ res.T_data
    @test p2[end-5][1][:y] ≈ res.X̂_data[1,:]
    @test p2[end-4][1][:y] ≈ res.X̂_data[2,:]
    @test p2[end-3][1][:y] ≈ res.X̂_data[3,:]
    @test p2[end-2][1][:y] ≈ res.X̂_data[4,:]
    @test p2[end-1][1][:y] ≈ res.X̂_data[5,:]
    @test p2[end-0][1][:y] ≈ res.X̂_data[6,:]
    p3 = plot(res, plotxwithx̂=true)
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
end

@testset "PredictiveController quick simulation" begin
    mpc = LinMPC(LinModel(sys, Ts, i_d=[3]))
    res = sim!(mpc, 15)
    @test isa(res.obj, LinMPC)
    @test length(res.T_data) == 15
    @test res.Ry_data[:, 1] ≈ mpc.estim.model.yop .+ 1
    @test res.Ud_data ≈ res.U_data
    @test res.D_data[:, 1]  ≈ mpc.estim.model.dop
    @test res.X_data[:, 1]  ≈ zeros(mpc.estim.model.nx)
    @test res.X̂_data[:, 1]  ≈ zeros(mpc.estim.nx̂)
end

@testset "PredictiveController Plots" begin
    mpc = LinMPC(LinModel(sys, Ts, i_d=[3]), Lwt=[0.01, 0.01])
    mpc = setconstraint!(mpc, umin=[-50, -51], umax=[52, 53], ŷmin=[-54,-55], ŷmax=[56,57])
    # TODO: ajouter des tests pour umin umax ŷmin ŷmax
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
    p5 = plot(res, plotumin=true, plotumax=false, plotŷmin=false, plotŷmax=false)
    @test p5[1][1][:x] ≈ res.T_data
    @test all(p5[end-2][3][:y] .≈ -50.0)
    @test all(p5[end-1][3][:y] .≈ -51.0)
    p6 = plot(res, plotumin=false, plotumax=true, plotŷmin=false, plotŷmax=false)
    @test p6[1][1][:x] ≈ res.T_data
    @test all(p6[end-2][3][:y] .≈ 52.0)
    @test all(p6[end-1][3][:y] .≈ 53.0)
    p7 = plot(res, plotumin=false, plotumax=false, plotŷmin=true, plotŷmax=false)
    @test p7[1][1][:x] ≈ res.T_data
    @test all(p7[end-4][3][:y] .≈ -54.0)
    @test all(p7[end-3][3][:y] .≈ -55.0)
    p8 = plot(res, plotumin=false, plotumax=false, plotŷmin=false, plotŷmax=true)
    @test p8[1][1][:x] ≈ res.T_data
    @test all(p8[end-4][3][:y] .≈ 56.0)
    @test all(p8[end-3][3][:y] .≈ 57.0)
end