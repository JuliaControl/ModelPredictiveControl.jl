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