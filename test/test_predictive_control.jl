Ts = 400.0
sys = [ tf(1.90,[1800.0,1])   tf(1.90,[1800.0,1])   tf(1.90,[1800.0,1]);
        tf(-0.74,[800.0,1])   tf(0.74,[800.0,1])    tf(-0.74,[800.0,1])   ]

@testset "LinMPC construction" begin
    model = LinModel(sys, Ts, i_d=[3])
    mpc1 = LinMPC(model, Hp=15)
    @test isa(mpc1.estim, SteadyKalmanFilter)
    @test size(mpc1.Ẽ,1) == 15*mpc1.estim.model.ny
    mpc2 = LinMPC(model, Hc=4, Cwt=Inf)
    @test size(mpc2.Ẽ,2) == 4*mpc2.estim.model.nu
    mpc3 = LinMPC(model, Hc=4, Cwt=1e6)
    @test size(mpc3.Ẽ,2) == 4*mpc3.estim.model.nu + 1
    @test mpc3.Ñ_Hc[end] ≈ 1e6
    mpc4 = LinMPC(model, Mwt=[1,2], Hp=15)
    @test mpc4.M_Hp ≈ Diagonal(diagm(repeat(Float64[1, 2], 15)))
    mpc5 = LinMPC(model, Nwt=[3,4], Cwt=1e3, Hc=5)
    @test mpc5.Ñ_Hc ≈ Diagonal(diagm([repeat(Float64[3, 4], 5); [1e3]]))
    mpc6 = LinMPC(model, Lwt=[0,1], Hp=15)
    @test mpc6.L_Hp ≈ Diagonal(diagm(repeat(Float64[0, 1], 15)))
    mpc7 = LinMPC(model, optim=JuMP.Model(DAQP.Optimizer))
    @test solver_name(mpc7.optim) == "DAQP"
    kf = KalmanFilter(model)
    mpc8 = LinMPC(kf)
    @test isa(mpc8.estim, KalmanFilter)
    mpc9 = LinMPC(model, nint_u=[1, 1], nint_ym=[0, 0])
    @test mpc9.estim.nint_u  == [1, 1]
    @test mpc9.estim.nint_ym == [0, 0]
    mpc10 = LinMPC(model, M_Hp=Diagonal(collect(1.01:0.01:1.2)))
    @test mpc10.M_Hp ≈ Diagonal(collect(1.01:0.01:1.2))
    mpc11 = LinMPC(model, N_Hc=Diagonal([0.1,0.11,0.12,0.13]), Cwt=Inf)
    @test mpc11.Ñ_Hc ≈ Diagonal([0.1,0.11,0.12,0.13])
    mcp12 = LinMPC(model, L_Hp=Diagonal(collect(0.001:0.001:0.02)))
    @test mcp12.L_Hp ≈ Diagonal(collect(0.001:0.001:0.02))
    model2 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    mpc13  = LinMPC(model2)
    @test isa(mpc13, LinMPC{Float32})
    @test isa(mpc13.optim, JuMP.GenericModel{Float64}) # OSQP does not support Float32

    @test_throws ArgumentError LinMPC(model, Hp=0)
    @test_throws ArgumentError LinMPC(model, Hc=0)
    @test_throws ArgumentError LinMPC(model, Hp=1, Hc=2)
    @test_throws ArgumentError LinMPC(model, Mwt=[1])
    @test_throws ArgumentError LinMPC(model, Mwt=[1])
    @test_throws ArgumentError LinMPC(model, Lwt=[1])
    @test_throws ArgumentError LinMPC(model, Cwt=[1])
    @test_throws ArgumentError LinMPC(model, Mwt=[-1,1])
    @test_throws ArgumentError LinMPC(model, Nwt=[-1,1])
    @test_throws ArgumentError LinMPC(model, Lwt=[-1,1])
    @test_throws ArgumentError LinMPC(model, Cwt=-1)
end

@testset "LinMPC moves and getinfo" begin
    linmodel = setop!(LinModel(tf(5, [2, 1]), 3), yop=[10])
    mpc1 = LinMPC(linmodel, Nwt=[0], Hp=1000, Hc=1)
    r = [15]
    preparestate!(mpc1, [10])
    u = moveinput!(mpc1, r)
    @test u ≈ [1] atol=1e-2
    u = mpc1(r)
    @test u ≈ [1] atol=1e-2
    info = getinfo(mpc1)
    @test info[:u] ≈ u
    @test info[:Ŷ][end] ≈ r[1] atol=1e-2
    mpc2 = LinMPC(linmodel, Nwt=[0], Cwt=Inf, Hp=1000, Hc=1)
    preparestate!(mpc2, [10])
    u = moveinput!(mpc2, r)
    @test u ≈ [1] atol=1e-2
    mpc3 = LinMPC(linmodel, Mwt=[0], Nwt=[0], Lwt=[1])
    preparestate!(mpc3, [10])
    u = moveinput!(mpc3, [0], R̂u=fill(12, mpc3.Hp))
    @test u ≈ [12] atol=1e-2
    model2 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    mpc4  = LinMPC(model2)
    preparestate!(mpc4, [0])
    moveinput!(mpc4, [0]) ≈ [0.0]

    @test_throws DimensionMismatch moveinput!(mpc1, [0,0,0])
    @test_throws DimensionMismatch moveinput!(mpc1, [0], [0,0])
    @test_throws DimensionMismatch moveinput!(mpc1; D̂  = fill(0, mpc1.Hp+1))
    @test_throws DimensionMismatch moveinput!(mpc1; R̂y = fill(0, mpc1.Hp+1))
    @test_throws DimensionMismatch moveinput!(mpc3; R̂u = fill(0, mpc1.Hp+1))
end

@testset "LinMPC step disturbance rejection" begin
    linmodel = setop!(LinModel(tf(5, [2, 1]), 3.0), yop=[10])
    r = [15]
    outdist = [5]
    mpc_im = LinMPC(InternalModel(linmodel))
    linmodel.x0 .= 0
    ym, u = linmodel() - outdist, [0.0]
    for i=1:25
        ym = linmodel() - outdist
        preparestate!(mpc_im, ym)
        u = moveinput!(mpc_im, r)
        updatestate!(mpc_im, u, ym)
        updatestate!(linmodel, u)
    end
    @test u  ≈ [2] atol=1e-2
    @test ym ≈ r   atol=1e-2
    mpc_nint_u = LinMPC(SteadyKalmanFilter(LinModel(tf(5, [2, 1]), 3), nint_u=[1]))
    linmodel.x0 .= 0
    ym, u = linmodel() - outdist, [0.0]
    for i=1:25
        ym = linmodel() - outdist
        preparestate!(mpc_nint_u, ym)
        u = moveinput!(mpc_nint_u, r)
        updatestate!(mpc_nint_u, u, ym)
        updatestate!(linmodel, u)
    end
    @test u  ≈ [2] atol=1e-2
    @test ym ≈ r   atol=1e-2 
    mpc_nint_ym = LinMPC(SteadyKalmanFilter(LinModel(tf(5, [2, 1]), 3), nint_ym=[1]))
    linmodel.x0 .= 0
    ym, u = linmodel() - outdist, [0.0]
    for i=1:25
        ym = linmodel() - outdist
        preparestate!(mpc_nint_ym, ym)
        u = moveinput!(mpc_nint_ym, r)
        updatestate!(mpc_nint_ym, u, ym)
        updatestate!(linmodel, u)
    end
    @test u  ≈ [2] atol=1e-2
    @test ym ≈ r   atol=1e-2 
end

@testset "LinMPC other methods" begin
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    mpc1 = LinMPC(linmodel1)
    @test initstate!(mpc1, [10, 50], [50, 30+1]) ≈ [zeros(3); [1]]
    setstate!(mpc1, [1,2,3,4])
    @test mpc1.estim.x̂0 ≈ [1,2,3,4]
    setstate!(mpc1, [0,0,0,0])
    preparestate!(mpc1, [50, 30])
    updatestate!(mpc1, mpc1.estim.model.uop, [50, 30])
    @test mpc1.estim.x̂0 ≈ [0,0,0,0]
    # do not call preparestate! before moveinput! for the warning:
    moveinput!(mpc1, [10, 50])
    @test_throws ArgumentError updatestate!(mpc1, [0,0])
end

@testset "LinMPC set constraints" begin
    model = LinModel(sys, Ts, i_d=[3])
    mpc = LinMPC(model, Hp=1, Hc=1)

    setconstraint!(mpc, umin=[-5, -9.9], umax=[100,99])
    @test all((mpc.con.U0min, mpc.con.U0max) .≈ ([-5, -9.9], [100,99]))
    setconstraint!(mpc, Δumin=[-5,-10], Δumax=[6,11])
    @test all((mpc.con.ΔŨmin, mpc.con.ΔŨmax) .≈ ([-5,-10,0], [6,11,Inf]))
    setconstraint!(mpc, ymin=[-6, -11],ymax=[55, 35])
    @test all((mpc.con.Y0min, mpc.con.Y0max) .≈ ([-6,-11], [55,35]))
    setconstraint!(mpc, x̂min=[-21,-22,-23,-24,-25,-26], x̂max=[21,22,23,24,25,26])
    @test all((mpc.con.x̂0min, mpc.con.x̂0max) .≈ ([-21,-22,-23,-24,-25,-26], [21,22,23,24,25,26]))

    setconstraint!(mpc, c_umin=[0.01,0.02], c_umax=[0.03,0.04])
    @test all((-mpc.con.A_Umin[:, end], -mpc.con.A_Umax[:, end]) .≈ ([0.01,0.02], [0.03,0.04]))
    setconstraint!(mpc, c_Δumin=[0.05,0.06], c_Δumax=[0.07,0.08])
    @test all((-mpc.con.A_ΔŨmin[1:end-1, end], -mpc.con.A_ΔŨmax[1:end-1, end]) .≈ ([0.05,0.06], [0.07,0.08]))
    setconstraint!(mpc, c_ymin=[1.00,1.01], c_ymax=[1.02,1.03])
    @test all((-mpc.con.A_Ymin[:, end], -mpc.con.A_Ymax[:, end]) .≈ ([1.00,1.01], [1.02,1.03]))
    setconstraint!(mpc, c_x̂min=[0.21,0.22,0.23,0.24,0.25,0.26], c_x̂max=[0.31,0.32,0.33,0.34,0.35,0.36])
    @test all((-mpc.con.A_x̂min[:, end], -mpc.con.A_x̂max[:, end]) .≈ ([0.21,0.22,0.23,0.24,0.25,0.26], [0.31,0.32,0.33,0.34,0.35,0.36]))

    model2 = LinModel(tf([2], [10, 1]), 3.0)
    mpc2 = LinMPC(model2, Hp=50, Hc=5)

    setconstraint!(mpc2, Umin=-1(1:50).-1, Umax=+1(1:50).+1)
    @test all((mpc2.con.U0min, mpc2.con.U0max) .≈ (-1(1:50).-1, +1(1:50).+1))
    setconstraint!(mpc2, ΔUmin=-1(1:5).-2, ΔUmax=+1(1:5).+2)
    @test all((mpc2.con.ΔŨmin, mpc2.con.ΔŨmax) .≈ ([-1(1:5).-2; 0], [+1(1:5).+2; Inf]))
    setconstraint!(mpc2, Ymin=-1(1:50).-3, Ymax=+1(1:50).+3)
    @test all((mpc2.con.Y0min, mpc2.con.Y0max) .≈ (-1(1:50).-3, +1(1:50).+3))

    setconstraint!(mpc2, C_umin=+1(1:50).+4, C_umax=+1(1:50).+4)
    @test all((-mpc2.con.A_Umin[:, end], -mpc2.con.A_Umax[:, end]) .≈ (+1(1:50).+4, +1(1:50).+4))
    setconstraint!(mpc2, C_Δumin=+1(1:5).+5, C_Δumax=+1(1:5).+5)
    @test all((-mpc2.con.A_ΔŨmin[1:end-1, end], -mpc2.con.A_ΔŨmax[1:end-1, end]) .≈ (+1(1:5).+5, +1(1:5).+5))
    setconstraint!(mpc2, C_ymin=+1(1:50).+6, C_ymax=+1(1:50).+6)
    @test all((-mpc2.con.A_Ymin[:, end], -mpc2.con.A_Ymax[:, end]) .≈ (+1(1:50).+6, +1(1:50).+6))
    setconstraint!(mpc2, c_umin=[0], c_umax=[0], c_Δumin=[0], c_Δumax=[0], c_ymin=[1], c_ymax=[1])

    @test_throws ArgumentError setconstraint!(mpc, umin=[0,0,0])
    @test_throws ArgumentError setconstraint!(mpc, umax=[0,0,0])
    @test_throws ArgumentError setconstraint!(mpc, Δumin=[0,0,0])
    @test_throws ArgumentError setconstraint!(mpc, Δumax=[0,0,0])
    @test_throws ArgumentError setconstraint!(mpc, ymin=[0,0,0])
    @test_throws ArgumentError setconstraint!(mpc, ymax=[0,0,0])
    @test_throws ArgumentError setconstraint!(mpc, c_umin=[0,0,0])
    @test_throws ArgumentError setconstraint!(mpc, c_umax=[0,0,0])
    @test_throws ArgumentError setconstraint!(mpc, c_Δumin=[0,0,0])
    @test_throws ArgumentError setconstraint!(mpc, c_Δumax=[0,0,0])
    @test_throws ArgumentError setconstraint!(mpc, c_ymin=[0,0,0])
    @test_throws ArgumentError setconstraint!(mpc, c_ymax=[0,0,0])

    preparestate!(mpc, mpc.estim.model.yop, mpc.estim.model.dop)
    moveinput!(mpc, [0, 0], [0])
    @test_throws ErrorException setconstraint!(mpc, c_umin=[1, 1], c_umax=[1, 1])
    @test_throws ErrorException setconstraint!(mpc, umin=[-Inf,-Inf], umax=[+Inf,+Inf])

    mpc3 = LinMPC(model, Cwt=Inf)
    @test_throws ArgumentError setconstraint!(mpc3, c_umin=[1, 1])
    @test_throws ArgumentError setconstraint!(mpc3, c_umax=[1, 1])
    @test_throws ArgumentError setconstraint!(mpc3, c_Δumin=[1, 1])
    @test_throws ArgumentError setconstraint!(mpc3, c_Δumax=[1, 1])
    @test_throws ArgumentError setconstraint!(mpc3, c_ymin=[1, 1])
    @test_throws ArgumentError setconstraint!(mpc3, c_ymax=[1, 1])
end

@testset "LinMPC constraint violation" begin
    model = LinModel(tf([2], [10, 1]), 3.0)
    mpc = LinMPC(model, Hp=50, Hc=5)

    setconstraint!(mpc, x̂min=[-1e3,-Inf], x̂max=[1e3,+Inf])
    setconstraint!(mpc, umin=[-3], umax=[3])
    setconstraint!(mpc, Δumin=[-1.5], Δumax=[1.5])
    setconstraint!(mpc, ymin=[-100], ymax=[100])
    preparestate!(mpc, [0])
    moveinput!(mpc, [-10])
    info = getinfo(mpc)
    @test info[:ΔU][begin] ≈ -1.5 atol=1e-1
    @test info[:U][end] ≈ -3 atol=1e-1

    setconstraint!(mpc, umin=[-10], umax=[10])
    setconstraint!(mpc, Δumin=[-15], Δumax=[15])
    setconstraint!(mpc, ymin=[-0.5], ymax=[0.5])
    moveinput!(mpc, [-10])
    info = getinfo(mpc)
    @test info[:Ŷ][end] ≈ -0.5 atol=1e-1

    setconstraint!(mpc, umin=[-10], umax=[10])
    setconstraint!(mpc, Δumin=[-15], Δumax=[15])
    setconstraint!(mpc, Ymin=[-0.5; fill(-100, 49)], Ymax=[0.5; fill(+100, 49)])
    moveinput!(mpc, [-10])
    info = getinfo(mpc)
    @test info[:Ŷ][end]   ≈ -10  atol=1e-1
    @test info[:Ŷ][begin] ≈ -0.5 atol=1e-1

    setconstraint!(mpc, umin=[-1e3], umax=[+1e3])
    setconstraint!(mpc, Δumin=[-1e3], Δumax=[+1e3])
    setconstraint!(mpc, ymin=[-1e3], ymax=[+1e3])
    setconstraint!(mpc, x̂min=[-1e-6,-Inf], x̂max=[+1e-6,+Inf])
    moveinput!(mpc, [-10])
    info = getinfo(mpc)
    @test info[:x̂end][1] ≈ 0 atol=1e-1
end

@testset "LinMPC terminal cost" begin
    model = LinModel(ss([0.5 -0.4;0.6 0.5], [1 0;0 1], [1 0; 0 1], 0, 1))
    K = lqr(Discrete, model.A, model.Bu, I, 0.5I)
    M_end = ControlSystemsBase.are(Discrete, model.A, model.Bu, I, 0.5I)
    M_Hp = [I(4) zeros(4,2); zeros(2,4) M_end]
    mpc = LinMPC(model; Hp=3, Hc=3, M_Hp, Nwt=[0; 0], Lwt=[0.5, 0.5], nint_ym=0)
    X_mpc = zeros(2,20)
    setstate!(mpc,[1,1])
    setstate!(model, [1,1])
    for i=1:20
        y = model()
        preparestate!(mpc, y)
        u = moveinput!(mpc, [0, 0])
        X_mpc[:,i] = model.x0
        updatestate!(mpc, u, y)
        updatestate!(model, u)
    end
    X_lqr = zeros(2,20)
    x=[1,1]
    for i=1:20
        u = -K*x
        X_lqr[:,i] = x
        x = model.A*x + model.Bu*u
    end
    @test all(isapprox.(X_mpc, X_lqr, atol=1e-5))
end

@testset "LinMPC set model" begin
    estim = KalmanFilter(setop!(LinModel(tf(5, [2, 1]), 3), yop=[10], uop=[1]))
    mpc = LinMPC(estim, Nwt=[0], Cwt=1e4, Hp=1000, Hc=1)
    mpc = setconstraint!(mpc, umin=[-24], umax=[26])
    mpc = setconstraint!(mpc, ymin=[-54], ymax=[56])
    @test mpc.Yop ≈ fill(10.0, 1000)
    @test mpc.Uop ≈ fill(1.0, 1000)
    @test mpc.con.U0min ≈ fill(-24.0 -1,  1000)
    @test mpc.con.U0max ≈ fill(26.0  -1,  1000)
    @test mpc.con.Y0min ≈ fill(-54.0 -10, 1000)
    @test mpc.con.Y0max ≈ fill(56.0  -10, 1000)
    r = [15]
    preparestate!(mpc, [10])
    u = moveinput!(mpc, r)
    @test u ≈ [2] atol=1e-2
    setmodel!(mpc, setop!(LinModel(tf(5, [2, 1]), 3), yop=[20], uop=[11]))
    @test mpc.Yop ≈ fill(20.0, 1000)
    @test mpc.Uop ≈ fill(11.0, 1000)
    @test mpc.con.U0min ≈ fill(-24.0 - 1  + 1  - 11,  1000)
    @test mpc.con.U0max ≈ fill(26.0  - 1  + 1  - 11,  1000)
    @test mpc.con.Y0min ≈ fill(-54.0 - 10 + 10 - 20, 1000)
    @test mpc.con.Y0max ≈ fill(56.0  - 10 + 10 - 20, 1000)
    r = [40]
    u = moveinput!(mpc, r)
    @test u ≈ [15] atol=1e-2
    setmodel!(mpc, setop!(LinModel(tf(10, [2, 1]), 3), yop=[20], uop=[11]))
    r = [40]
    u = moveinput!(mpc, r)
    @test u ≈ [13] atol=1e-2
    setmodel!(mpc, Mwt=[100], Nwt=[200], Lwt=[300])
    @test mpc.M_Hp ≈ diagm(fill(100, 1000))
    @test mpc.Ñ_Hc ≈ diagm([200, 1e4])
    @test mpc.L_Hp ≈ diagm(fill(300, 1000))
    setmodel!(mpc, M_Hp=diagm(1:1000), Ñ_Hc=diagm([0.1;1e6]), L_Hp=diagm(1.1:1000.1))
    @test mpc.M_Hp ≈ diagm(1:1000)
    @test mpc.Ñ_Hc ≈ diagm([0.1;1e6])
    @test mpc.L_Hp ≈ diagm(1.1:1000.1)
end

@testset "LinMPC real-time simulations" begin
    linmodel1 = LinModel(tf(2, [10, 1]), 0.1)
    mpc1 = LinMPC(linmodel1)
    times1 = zeros(5)
    for i=1:5
        times1[i] = savetime!(mpc1)
        preparestate!(mpc1, [1])
        updatestate!(mpc1, [1], [1])
        periodsleep(mpc1)
    end
    @test all(isapprox.(diff(times1[2:end]), 0.1, atol=0.01))
end

@testset "ExplicitMPC construction" begin
    model = LinModel(sys, Ts, i_d=[3])
    mpc1 = ExplicitMPC(model, Hp=15)
    @test isa(mpc1.estim, SteadyKalmanFilter)
    @test size(mpc1.Ẽ,1) == 15*mpc1.estim.model.ny
    mpc4 = ExplicitMPC(model, Mwt=[1,2], Hp=15)
    @test mpc4.M_Hp ≈ Diagonal(diagm(repeat(Float64[1, 2], 15)))
    mpc5 = ExplicitMPC(model, Nwt=[3,4], Hc=5)
    @test mpc5.Ñ_Hc ≈ Diagonal(diagm(repeat(Float64[3, 4], 5)))
    mpc6 = ExplicitMPC(model, Lwt=[0,1], Hp=15)
    @test mpc6.L_Hp ≈ Diagonal(diagm(repeat(Float64[0, 1], 15)))
    kf = KalmanFilter(model)
    mpc8 = ExplicitMPC(kf)
    @test isa(mpc8.estim, KalmanFilter)
    mpc9 = ExplicitMPC(model, nint_u=[1, 1], nint_ym=[0, 0])
    @test mpc9.estim.nint_u  == [1, 1]
    @test mpc9.estim.nint_ym == [0, 0]
    mpc10 = ExplicitMPC(model, M_Hp=Diagonal(collect(1.01:0.01:1.2)))
    @test mpc10.M_Hp ≈ Diagonal(collect(1.01:0.01:1.2))
    mpc11 = ExplicitMPC(model, N_Hc=Diagonal([0.1,0.11,0.12,0.13]))
    @test mpc11.Ñ_Hc ≈ Diagonal([0.1,0.11,0.12,0.13])
    mcp12 = ExplicitMPC(model, L_Hp=Diagonal(collect(0.001:0.001:0.02)))
    @test mcp12.L_Hp ≈ Diagonal(collect(0.001:0.001:0.02))
    model2 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    mpc13  = ExplicitMPC(model2)
    @test isa(mpc13, ExplicitMPC{Float32})

    @test_throws ArgumentError LinMPC(model, Hp=0)
end

@testset "ExplicitMPC moves and getinfo" begin
    mpc1 = ExplicitMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1000, Hc=1)
    r = [5]
    preparestate!(mpc1, [0])
    u = moveinput!(mpc1, r)
    @test u ≈ [1] atol=1e-2
    u = mpc1(r)
    @test u ≈ [1] atol=1e-2
    info = getinfo(mpc1)
    @test info[:u] ≈ u
    @test info[:Ŷ][end] ≈ r[1] atol=1e-2
    mpc2 = ExplicitMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1000, Hc=1)
    preparestate!(mpc2, [0])
    u = moveinput!(mpc2, [5])
    @test u ≈ [1] atol=1e-2
    mpc3 = ExplicitMPC(LinModel(tf(5, [2, 1]), 3), Mwt=[0], Nwt=[0], Lwt=[1])
    preparestate!(mpc3, [0])
    u = moveinput!(mpc3, [0], R̂u=fill(12, mpc3.Hp))
    @test u ≈ [12] atol=1e-2
    model2 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    mpc4  = ExplicitMPC(model2)
    preparestate!(mpc4, [0])
    moveinput!(mpc4, [0]) ≈ [0.0]
end


@testset "ExplicitMPC step disturbance rejection" begin
    linmodel = setop!(LinModel(tf(5, [2, 1]), 3.0), yop=[10])
    r = [15]
    outdist = [5]
    mpc_im = ExplicitMPC(InternalModel(linmodel))
    linmodel.x0 .= 0
    ym, u = linmodel() - outdist, [0.0]
    for i=1:25
        ym = linmodel() - outdist
        preparestate!(mpc_im, ym)
        u = moveinput!(mpc_im, r)
        updatestate!(mpc_im, u, ym)
        updatestate!(linmodel, u)
    end
    @test u  ≈ [2] atol=1e-2
    @test ym ≈ r   atol=1e-2
    mpc_nint_u = ExplicitMPC(SteadyKalmanFilter(LinModel(tf(5, [2, 1]), 3), nint_u=[1]))
    linmodel.x0 .= 0
    ym, u = linmodel() - outdist, [0.0]
    for i=1:25
        ym = linmodel() - outdist
        preparestate!(mpc_nint_u, ym)
        u = moveinput!(mpc_nint_u, r)
        updatestate!(mpc_nint_u, u, ym)
        updatestate!(linmodel, u)
    end
    @test u  ≈ [2] atol=1e-2
    @test ym ≈ r   atol=1e-2 
    mpc_nint_ym = ExplicitMPC(SteadyKalmanFilter(LinModel(tf(5, [2, 1]), 3), nint_ym=[1]))
    linmodel.x0 .= 0
    ym, u = linmodel() - outdist, [0.0]
    for i=1:25
        ym = linmodel() - outdist
        preparestate!(mpc_nint_ym, ym)
        u = moveinput!(mpc_nint_ym, r)
        updatestate!(mpc_nint_ym, u, ym)
        updatestate!(linmodel, u)
    end
    @test u  ≈ [2] atol=1e-2
    @test ym ≈ r   atol=1e-2 
end

@testset "ExplicitMPC other methods" begin
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    mpc1 = ExplicitMPC(linmodel1)
    @test initstate!(mpc1, [10, 50], [50, 30+1]) ≈ [zeros(3); [1]]
    setstate!(mpc1, [1,2,3,4])
    @test mpc1.estim.x̂0 ≈ [1,2,3,4]
    setstate!(mpc1, [0,0,0,0])
    preparestate!(mpc1, [50, 30])
    updatestate!(mpc1, mpc1.estim.model.uop, [50, 30])
    @test mpc1.estim.x̂0 ≈ [0,0,0,0]
    preparestate!(mpc1, [50, 30])
    @test_throws ArgumentError updatestate!(mpc1, [0,0])
end

@testset "ExplicitMPC constraints" begin
    model = LinModel(sys, Ts, i_d=[3])
    mpc = ExplicitMPC(model, Hp=1, Hc=1)
    @test_throws ErrorException setconstraint!(mpc, umin=[0.0, 0.0])
end

@testset "ExplicitMPC set model" begin
    estim = KalmanFilter(setop!(LinModel(tf(5, [2, 1]), 3), yop=[10], uop=[1]))
    mpc = ExplicitMPC(estim, Nwt=[0], Hp=1000, Hc=1)
    @test mpc.Yop ≈ fill(10.0, 1000)
    @test mpc.Uop ≈ fill(1.0, 1000)
    r = [15]
    preparestate!(mpc, [10])
    u = moveinput!(mpc, r)
    @test u ≈ [2] atol=1e-2
    setmodel!(mpc, setop!(LinModel(tf(5, [2, 1]), 3), yop=[20], uop=[11]))
    @test mpc.Yop ≈ fill(20.0, 1000)
    @test mpc.Uop ≈ fill(11.0, 1000)
    r = [40]
    u = moveinput!(mpc, r)
    @test u ≈ [15] atol=1e-2
    setmodel!(mpc, setop!(LinModel(tf(10, [2, 1]), 3), yop=[20], uop=[11]))
    r = [40]
    u = moveinput!(mpc, r)
    @test u ≈ [13] atol=1e-2
    setmodel!(mpc, Mwt=[100], Nwt=[200], Lwt=[300])
    @test mpc.M_Hp ≈ diagm(fill(100, 1000))
    @test mpc.Ñ_Hc ≈ diagm([200])
    @test mpc.L_Hp ≈ diagm(fill(300, 1000))
    setmodel!(mpc, M_Hp=diagm(1:1000), Ñ_Hc=[0.1], L_Hp=diagm(1.1:1000.1))
    @test mpc.M_Hp ≈ diagm(1:1000)
    @test mpc.Ñ_Hc ≈ [0.1]
    @test mpc.L_Hp ≈ diagm(1.1:1000.1)
end

@testset "NonLinMPC construction" begin
    linmodel1 = LinModel(sys,Ts,i_d=[3])
    nmpc0 = NonLinMPC(linmodel1, Hp=15)
    @test isa(nmpc0.estim, SteadyKalmanFilter)
    f = (x,u,d,model) -> model.A*x + model.Bu*u + model.Bd*d
    h = (x,d,model)   -> model.C*x + model.Dd*d
    nonlinmodel = NonLinModel(f, h, Ts, 2, 4, 2, 1, p=linmodel1, solver=nothing)
    nmpc1 = NonLinMPC(nonlinmodel, Hp=15)
    @test isa(nmpc1.estim, UnscentedKalmanFilter)
    @test size(nmpc1.R̂y, 1) == 15*nmpc1.estim.model.ny
    nmpc2 = NonLinMPC(nonlinmodel, Hp=15, Hc=4, Cwt=Inf)
    @test size(nmpc2.Ẽ, 2) == 4*nonlinmodel.nu
    nmpc3 = NonLinMPC(nonlinmodel, Hp=15, Hc=4, Cwt=1e6)
    @test size(nmpc3.Ẽ, 2) == 4*nonlinmodel.nu + 1
    @test nmpc3.Ñ_Hc[end] == 1e6
    nmpc4 = NonLinMPC(nonlinmodel, Hp=15, Mwt=[1,2])
    @test nmpc4.M_Hp ≈ Diagonal(diagm(repeat(Float64[1, 2], 15)))
    nmpc5 = NonLinMPC(nonlinmodel, Hp=15 ,Nwt=[3,4], Cwt=1e3, Hc=5)
    @test nmpc5.Ñ_Hc ≈ Diagonal(diagm([repeat(Float64[3, 4], 5); [1e3]]))
    nmpc6 = NonLinMPC(nonlinmodel, Hp=15, Lwt=[0,1])
    @test nmpc6.L_Hp ≈ Diagonal(diagm(repeat(Float64[0, 1], 15)))
    nmpc7 = NonLinMPC(nonlinmodel, Hp=15, Ewt=1e-3, JE=(Ue,Ŷe,D̂e,p) -> p*Ue.*Ŷe.*D̂e, p=2)
    @test nmpc7.E == 1e-3
    @test nmpc7.JE([1,2],[3,4],[4,6],2) == 2*[1,2].*[3,4].*[4,6]
    optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer, "nlp_scaling_max_gradient"=>1.0))
    nmpc8 = NonLinMPC(nonlinmodel, Hp=15, optim=optim)
    @test solver_name(nmpc8.optim) == "Ipopt"
    @test get_attribute(nmpc8.optim, "nlp_scaling_max_gradient") == 1.0
    im = InternalModel(nonlinmodel)
    nmpc9 = NonLinMPC(im, Hp=15)
    @test isa(nmpc9.estim, InternalModel)
    nmpc10 = NonLinMPC(linmodel1, nint_u=[1, 1], nint_ym=[0, 0])
    @test nmpc10.estim.nint_u  == [1, 1]
    @test nmpc10.estim.nint_ym == [0, 0]
    nmpc11 = NonLinMPC(nonlinmodel, Hp=15, nint_u=[1, 1], nint_ym=[0, 0])
    @test nmpc11.estim.nint_u  == [1, 1]
    @test nmpc11.estim.nint_ym == [0, 0]
    nmpc12 = NonLinMPC(nonlinmodel, Hp=10, M_Hp=Diagonal(collect(1.01:0.01:1.2)))
    @test nmpc12.M_Hp ≈ Diagonal(collect(1.01:0.01:1.2))
    nmpc13 = NonLinMPC(nonlinmodel, Hp=10, N_Hc=Diagonal([0.1,0.11,0.12,0.13]), Cwt=Inf)
    @test nmpc13.Ñ_Hc ≈ Diagonal([0.1,0.11,0.12,0.13])
    nmcp14 = NonLinMPC(nonlinmodel, Hp=10, L_Hp=Diagonal(collect(0.001:0.001:0.02)))
    @test nmcp14.L_Hp ≈ Diagonal(collect(0.001:0.001:0.02))

    nonlinmodel2 = NonLinModel{Float32}(f, h, Ts, 2, 4, 2, 1, solver=nothing)
    nmpc15  = NonLinMPC(nonlinmodel2, Hp=15)
    @test isa(nmpc15, NonLinMPC{Float32})
    @test isa(nmpc15.optim, JuMP.GenericModel{Float64}) # Ipopt does not support Float32

    @test_throws ArgumentError NonLinMPC(nonlinmodel, Hp=15, Ewt=[1, 1])
    @test_throws ArgumentError NonLinMPC(nonlinmodel)
    @test_throws ErrorException NonLinMPC(nonlinmodel, Hp=15, JE=(_,_,_)->0.0)
end

@testset "NonLinMPC moves and getinfo" begin
    linmodel = setop!(LinModel(tf(5, [2000, 1]), 3000.0), yop=[10])
    nmpc_lin = NonLinMPC(linmodel, Nwt=[0], Hp=1000, Hc=1)
    r = [15]
    preparestate!(nmpc_lin, [10])
    u = moveinput!(nmpc_lin, r)
    @test u ≈ [1] atol=5e-2
    u = nmpc_lin(r)
    @test u ≈ [1] atol=5e-2
    info = getinfo(nmpc_lin)
    @test info[:u] ≈ u
    @test info[:Ŷ][end] ≈ r[1] atol=5e-2
    Hp = 1000
    R̂y = fill(r[1], Hp)
    JE = (_ , Ŷe, _ , R̂y) -> sum((Ŷe[2:end] - R̂y).^2)
    nmpc = NonLinMPC(linmodel, Mwt=[0], Nwt=[0], Cwt=Inf, Ewt=1, JE=JE, p=R̂y, Hp=Hp, Hc=1)
    preparestate!(nmpc, [10])
    u = moveinput!(nmpc)
    @test u ≈ [1] atol=5e-2
    # ensure that the current estimated output is updated for correct JE values:
    @test nmpc.ŷ ≈ evaloutput(nmpc.estim, Float64[])
    linmodel2 = LinModel([tf(5, [2000, 1]) tf(7, [8000,1])], 3000.0, i_d=[2])
    f = (x,u,d,_) -> linmodel2.A*x + linmodel2.Bu*u + linmodel2.Bd*d
    h = (x,d,_)   -> linmodel2.C*x + linmodel2.Dd*d
    nonlinmodel = NonLinModel(f, h, 3000.0, 1, 2, 1, 1, solver=nothing)
    nmpc2 = NonLinMPC(nonlinmodel, Nwt=[0], Hp=1000, Hc=1)
    d = [0.1]
    preparestate!(nmpc2, [0], [0])
    u = moveinput!(nmpc2, 7d, d)
    @test u ≈ [0] atol=5e-2
    u = nmpc2(7d, d)
    @test u ≈ [0] atol=5e-2
    info = getinfo(nmpc2)
    @test info[:u] ≈ u
    @test info[:Ŷ][end] ≈ 7d[1] atol=5e-2
    nmpc3 = NonLinMPC(nonlinmodel, Nwt=[0], Cwt=Inf, Hp=1000, Hc=1)
    preparestate!(nmpc3, [0], [0])
    u = moveinput!(nmpc3, 7d, d)
    @test u ≈ [0] atol=5e-2
    nmpc4 = NonLinMPC(nonlinmodel, Hp=15, Mwt=[0], Nwt=[0], Lwt=[1])
    preparestate!(nmpc4, [0], [0])
    u = moveinput!(nmpc4, [0], d, R̂u=fill(12, nmpc4.Hp))
    @test u ≈ [12] atol=5e-2
    nmpc5 = setconstraint!(NonLinMPC(nonlinmodel, Hp=15, Cwt=Inf), ymin=[1])
    g_Y0min_end = nmpc5.optim[:g_Y0min_15].func
    # test gfunc_i(i,::NTuple{N, Float64}):
    @test g_Y0min_end(20.0, 10.0) ≤ 0.0 
    # test gfunc_i(i,::NTuple{N, ForwardDiff.Dual}) : 
    @test ForwardDiff.gradient(vec->g_Y0min_end(vec...), [20.0, 10.0]) ≈ [-5, -5] atol=1e-3
    linmodel3 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    nmpc6  = NonLinMPC(linmodel3, Hp=10)
    preparestate!(nmpc6, [0])
    @test moveinput!(nmpc6, [0]) ≈ [0.0]
    nonlinmodel2 = NonLinModel{Float32}(f, h, 3000.0, 1, 2, 1, 1, solver=nothing)
    nmpc7  = NonLinMPC(nonlinmodel2, Hp=10)
    y = similar(nonlinmodel2.yop)
    nonlinmodel2.h!(y, Float32[0,0], Float32[0], Float32[])
    preparestate!(nmpc7, [0], [0])
    @test moveinput!(nmpc7, [0], [0]) ≈ [0.0]
end

@testset "NonLinMPC step disturbance rejection" begin
    linmodel = setop!(LinModel(tf(5, [2000, 1]), 3000.0), yop=[10])
    r = [15]
    outdist = [5]
    nmpc_im = NonLinMPC(InternalModel(linmodel))
    linmodel.x0 .= 0
    ym, u = linmodel() - outdist, [0.0]
    for i=1:25
        ym = linmodel() - outdist
        preparestate!(nmpc_im, ym)
        u = moveinput!(nmpc_im, r)
        updatestate!(nmpc_im, u, ym)
        updatestate!(linmodel, u)
    end
    @test u  ≈ [2] atol=1e-2
    @test ym ≈ r   atol=1e-2
    nmpc_nint_u = NonLinMPC(SteadyKalmanFilter(linmodel, nint_u=[1]))
    linmodel.x0 .= 0
    ym, u = linmodel() - outdist, [0.0]
    for i=1:25
        ym = linmodel() - outdist
        preparestate!(nmpc_nint_u, ym)
        u = moveinput!(nmpc_nint_u, r)
        updatestate!(nmpc_nint_u, u, ym)
        updatestate!(linmodel, u)
    end
    @test u  ≈ [2] atol=1e-2
    @test ym ≈ r   atol=1e-2 
    nmpc_nint_ym = NonLinMPC(SteadyKalmanFilter(linmodel, nint_ym=[1]))
    linmodel.x0 .= 0
    ym, u = linmodel() - outdist, [0.0]
    for i=1:25
        ym = linmodel() - outdist
        preparestate!(nmpc_nint_ym, ym)
        u = moveinput!(nmpc_nint_ym, r)
        updatestate!(nmpc_nint_ym, u, ym)
        updatestate!(linmodel, u)
    end
    @test u  ≈ [2] atol=1e-2
    @test ym ≈ r   atol=1e-2
end

@testset "NonLinMPC other methods" begin
    linmodel = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    f = (x,u,_,_) -> linmodel.A*x + linmodel.Bu*u
    h = (x,_,_)   -> linmodel.C*x
    nonlinmodel = setop!(
        NonLinModel(f, h, Ts, 2, 2, 2, solver=nothing), uop=[10,50], yop=[50,30]
    )
    nmpc1 = NonLinMPC(nonlinmodel, Hp=15)
    @test initstate!(nmpc1, [10, 50], [20, 25]) ≈ zeros(4)
    setstate!(nmpc1, [1,2,3,4])
    @test nmpc1.estim.x̂0 ≈ [1,2,3,4]
    setstate!(nmpc1, [0,0,0,0])
    preparestate!(nmpc1, [50, 30])
    updatestate!(nmpc1, nmpc1.estim.model.uop, [50, 30])
    @test nmpc1.estim.x̂0 ≈ [0,0,0,0] atol=1e-6
end

@testset "NonLinMPC set constraints" begin
    linmodel1 = LinModel(sys,Ts,i_d=[3])
    nmpc_lin = NonLinMPC(linmodel1, Hp=1, Hc=1)

    setconstraint!(nmpc_lin, ymin=[5,10],ymax=[55, 35])
    @test all((nmpc_lin.con.Y0min, nmpc_lin.con.Y0max) .≈ ([5,10], [55,35]))
    setconstraint!(nmpc_lin, c_ymin=[1.0,1.1], c_ymax=[1.2,1.3])
    @test all((-nmpc_lin.con.A_Ymin[:, end], -nmpc_lin.con.A_Ymax[:, end]) .≈ ([1.0,1.1], [1.2,1.3]))

    f = (x,u,d,_) -> linmodel1.A*x + linmodel1.Bu*u + linmodel1.Bd*d
    h = (x,d,_)   -> linmodel1.C*x + linmodel1.Dd*d
    nonlinmodel = NonLinModel(f, h, Ts, 2, 4, 2, 1, solver=nothing)
    nmpc = NonLinMPC(nonlinmodel, Hp=1, Hc=1)

    setconstraint!(nmpc, umin=[-5, -9.9], umax=[100,99])
    @test all((nmpc.con.U0min, nmpc.con.U0max) .≈ ([-5, -9.9], [100,99]))
    setconstraint!(nmpc, Δumin=[-5,-10], Δumax=[6,11])
    @test all((nmpc.con.ΔŨmin, nmpc.con.ΔŨmax) .≈ ([-5,-10,0], [6,11,Inf]))
    setconstraint!(nmpc, ymin=[-6, -11],ymax=[55, 35])
    @test all((nmpc.con.Y0min, nmpc.con.Y0max) .≈ ([-6,-11], [55,35]))
    setconstraint!(nmpc, x̂min=[-21,-22,-23,-24,-25,-26], x̂max=[21,22,23,24,25,26])
    @test all((nmpc.con.x̂0min, nmpc.con.x̂0max) .≈ ([-21,-22,-23,-24,-25,-26], [21,22,23,24,25,26]))

    setconstraint!(nmpc, c_umin=[0.01,0.02], c_umax=[0.03,0.04])
    @test all((-nmpc.con.A_Umin[:, end], -nmpc.con.A_Umax[:, end]) .≈ ([0.01,0.02], [0.03,0.04]))
    setconstraint!(nmpc, c_Δumin=[0.05,0.06], c_Δumax=[0.07,0.08])
    @test all((-nmpc.con.A_ΔŨmin[1:end-1, end], -nmpc.con.A_ΔŨmax[1:end-1, end]) .≈ ([0.05,0.06], [0.07,0.08]))
    setconstraint!(nmpc, c_ymin=[1.00,1.01], c_ymax=[1.02,1.03])
    @test all((-nmpc.con.A_Ymin, -nmpc.con.A_Ymax) .≈ (zeros(0,3), zeros(0,3)))
    @test all((nmpc.con.C_ymin, nmpc.con.C_ymax) .≈ ([1.00,1.01], [1.02,1.03]))
    setconstraint!(nmpc, c_x̂min=[0.21,0.22,0.23,0.24,0.25,0.26], c_x̂max=[0.31,0.32,0.33,0.34,0.35,0.36])
    @test all((nmpc.con.c_x̂min, nmpc.con.c_x̂max) .≈ ([0.21,0.22,0.23,0.24,0.25,0.26], [0.31,0.32,0.33,0.34,0.35,0.36]))

end

@testset "NonLinMPC constraint violation" begin
    linmodel = LinModel(tf([2], [10000, 1]), 3000.0)
    nmpc_lin = NonLinMPC(linmodel, Hp=50, Hc=5)

    setconstraint!(nmpc_lin, x̂min=[-1e3,-Inf], x̂max=[1e3,+Inf])
    setconstraint!(nmpc_lin, umin=[-3], umax=[3])
    setconstraint!(nmpc_lin, Δumin=[-1.5], Δumax=[1.5])
    setconstraint!(nmpc_lin, ymin=[-100], ymax=[100])
    preparestate!(nmpc_lin, [0])
    moveinput!(nmpc_lin, [-20])
    info = getinfo(nmpc_lin)
    @test info[:ΔU][begin] ≈ -1.5 atol=1e-2
    @test info[:U][end] ≈ -3 atol=1e-2

    setconstraint!(nmpc_lin, umin=[-10], umax=[10])
    setconstraint!(nmpc_lin, Δumin=[-15], Δumax=[15])
    setconstraint!(nmpc_lin, ymin=[-0.5], ymax=[0.5])
    moveinput!(nmpc_lin, [-20])
    info = getinfo(nmpc_lin)
    @test info[:Ŷ][end] ≈ -0.5 atol=1e-2

    setconstraint!(nmpc_lin, umin=[-10], umax=[10])
    setconstraint!(nmpc_lin, Δumin=[-15], Δumax=[15])
    setconstraint!(nmpc_lin, Ymin=[-0.5; fill(-100, 49)], Ymax=[0.5; fill(+100, 49)])
    moveinput!(nmpc_lin, [-10])
    info = getinfo(nmpc_lin)
    @test info[:Ŷ][end]   ≈ -10  atol=1e-2
    @test info[:Ŷ][begin] ≈ -0.5 atol=1e-2

    setconstraint!(nmpc_lin, umin=[-1e3], umax=[+1e3])
    setconstraint!(nmpc_lin, Δumin=[-1e3], Δumax=[+1e3])
    setconstraint!(nmpc_lin, ymin=[-1e3], ymax=[+1e3])
    setconstraint!(nmpc_lin, x̂min=[-1e-6,-Inf], x̂max=[+1e-6,+Inf])
    moveinput!(nmpc_lin, [-10])
    info = getinfo(nmpc_lin)
    @test info[:x̂end][1] ≈ 0 atol=1e-1

    f = (x,u,_,_) -> linmodel.A*x + linmodel.Bu*u
    h = (x,_,_)   -> linmodel.C*x
    nonlinmodel = NonLinModel(f, h, linmodel.Ts, 1, 1, 1, solver=nothing)
    nmpc = NonLinMPC(nonlinmodel, Hp=50, Hc=5)

    setconstraint!(nmpc, x̂min=[-1e3,-Inf], x̂max=[1e3,+Inf])
    setconstraint!(nmpc, umin=[-3], umax=[3])
    setconstraint!(nmpc, Δumin=[-1.5], Δumax=[1.5])
    setconstraint!(nmpc, ymin=[-100], ymax=[100])
    preparestate!(nmpc, [0])
    moveinput!(nmpc, [-20])
    info = getinfo(nmpc)
    @test info[:ΔU][begin] ≈ -1.5 atol=1e-2
    @test info[:U][end] ≈ -3 atol=1e-2
    
    setconstraint!(nmpc, umin=[-10], umax=[10])
    setconstraint!(nmpc, Δumin=[-15], Δumax=[15])
    setconstraint!(nmpc, ymin=[-0.5], ymax=[0.5])
    moveinput!(nmpc, [-20])
    info = getinfo(nmpc)
    @test info[:Ŷ][end] ≈ -0.5 atol=1e-2
    
    setconstraint!(nmpc, umin=[-10], umax=[10])
    setconstraint!(nmpc, Δumin=[-15], Δumax=[15])
    setconstraint!(nmpc, Ymin=[-0.5; fill(-100, 49)], Ymax=[0.5; fill(+100, 49)])
    moveinput!(nmpc, [-10])
    info = getinfo(nmpc)
    @test info[:Ŷ][end]   ≈ -10  atol=1e-2
    @test info[:Ŷ][begin] ≈ -0.5 atol=1e-2
    
    setconstraint!(nmpc, umin=[-1e3], umax=[+1e3])
    setconstraint!(nmpc, Δumin=[-1e3], Δumax=[+1e3])
    setconstraint!(nmpc, ymin=[-1e3], ymax=[+1e3])
    setconstraint!(nmpc, x̂min=[-1e-6,-Inf], x̂max=[+1e-6,+Inf])
    moveinput!(nmpc, [-10])
    info = getinfo(nmpc)
    @test info[:x̂end][1] ≈ 0 atol=1e-1
end

@testset "NonLinMPC set model" begin
    estim = KalmanFilter(setop!(LinModel(tf(5, [2, 1]), 3), yop=[10], uop=[1]))
    mpc = NonLinMPC(estim, Nwt=[0], Cwt=1e4, Hp=1000, Hc=1)
    mpc = setconstraint!(mpc, umin=[-24], umax=[26])
    mpc = setconstraint!(mpc, ymin=[-54], ymax=[56])
    @test mpc.Yop ≈ fill(10.0, 1000)
    @test mpc.Uop ≈ fill(1.0, 1000)
    @test mpc.con.U0min ≈ fill(-24.0 -1,  1000)
    @test mpc.con.U0max ≈ fill(26.0  -1,  1000)
    @test mpc.con.Y0min ≈ fill(-54.0 -10, 1000)
    @test mpc.con.Y0max ≈ fill(56.0  -10, 1000)
    r = [15]
    preparestate!(mpc, [10])
    u = moveinput!(mpc, r)
    @test u ≈ [2] atol=1e-2
    setmodel!(mpc, setop!(LinModel(tf(5, [2, 1]), 3), yop=[20], uop=[11]))
    @test mpc.Yop ≈ fill(20.0, 1000)
    @test mpc.Uop ≈ fill(11.0, 1000)
    @test mpc.con.U0min ≈ fill(-24.0 - 1  + 1  - 11,  1000)
    @test mpc.con.U0max ≈ fill(26.0  - 1  + 1  - 11,  1000)
    @test mpc.con.Y0min ≈ fill(-54.0 - 10 + 10 - 20, 1000)
    @test mpc.con.Y0max ≈ fill(56.0  - 10 + 10 - 20, 1000)
    r = [40]
    u = moveinput!(mpc, r)
    @test u ≈ [15] atol=1e-2
    setmodel!(mpc, setop!(LinModel(tf(10, [2, 1]), 3), yop=[20], uop=[11]))
    r = [40]
    u = moveinput!(mpc, r)
    @test u ≈ [13] atol=1e-2
    setmodel!(mpc, Mwt=[100], Nwt=[200], Lwt=[300])
    @test mpc.M_Hp ≈ diagm(fill(100, 1000))
    @test mpc.Ñ_Hc ≈ diagm([200, 1e4])
    @test mpc.L_Hp ≈ diagm(fill(300, 1000))
    setmodel!(mpc, M_Hp=diagm(1:1000), Ñ_Hc=diagm([0.1;1e6]), L_Hp=diagm(1.1:1000.1))
    @test mpc.M_Hp ≈ diagm(1:1000)
    @test mpc.Ñ_Hc ≈ diagm([0.1;1e6])
    @test mpc.L_Hp ≈ diagm(1.1:1000.1)
    f = (x,u,d,_) -> estim.model.A*x + estim.model.Bu*u + estim.model.Bd*d
    h = (x,d,_)   -> estim.model.C*x + estim.model.Du*d
    nonlinmodel = NonLinModel(f, h, 10.0, 1, 1, 1)
    nmpc = NonLinMPC(nonlinmodel, Nwt=[0], Cwt=1e4, Hp=1000, Hc=10)
    setmodel!(nmpc, Mwt=[100], Nwt=[200], Lwt=[300])
    @test nmpc.M_Hp ≈ diagm(fill(100, 1000))
    @test nmpc.Ñ_Hc ≈ diagm([fill(200, 10); 1e4])
    @test nmpc.L_Hp ≈ diagm(fill(300, 1000))
    setmodel!(nmpc, M_Hp=diagm(1:1000), Ñ_Hc=diagm([fill(0.1, 10);1e6]), L_Hp=diagm(1.1:1000.1))
    @test nmpc.M_Hp ≈ diagm(1:1000)
    @test nmpc.Ñ_Hc ≈ diagm([fill(0.1, 10);1e6])
    @test nmpc.L_Hp ≈ diagm(1.1:1000.1)
    @test_throws ErrorException setmodel!(nmpc, deepcopy(nonlinmodel))
end

@testset "LinMPC v.s. NonLinMPC" begin
    linmodel = setop!(LinModel(sys,Ts,i_d=[3]), uop=[10,50], yop=[50,30], dop=[20])
    f = (x,u,d,_) -> linmodel.A*x + linmodel.Bu*u + linmodel.Bd*d
    h = (x,d,_)   -> linmodel.C*x + linmodel.Dd*d
    nonlinmodel = NonLinModel(f, h, Ts, 2, 4, 2, 1, solver=nothing)
    nonlinmodel = setop!(nonlinmodel, uop=[10,50], yop=[50,30], dop=[20])
    optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer, "sb"=>"yes"))
    linmpc = LinMPC(KalmanFilter(linmodel), Hp=15, optim=optim)
    linmpc = setconstraint!(linmpc, ymax=[55,35], Δumax=[5, 5])
    nonlinmpc1 = NonLinMPC(UnscentedKalmanFilter(nonlinmodel), Hp=15)
    nonlinmpc1 = setconstraint!(nonlinmpc1, ymax=[55,35], Δumax=[5, 5])
    nonlinmpc2 = NonLinMPC(KalmanFilter(linmodel), Hp=15)
    nonlinmpc2 = setconstraint!(nonlinmpc2, ymax=[55,35], Δumax=[5, 5])
    U_linmpc = zeros(2, 30)
    U_nonlinmpc1 = zeros(2, 30)
    U_nonlinmpc2 = zeros(2, 30)
    for i=1:30
        r = [55, 35]
        d = i > 15 ? [0] : [20]
        y = linmodel(d) + randn(2)
        preparestate!(linmpc, y, d)
        preparestate!(nonlinmpc1, y, d)
        preparestate!(nonlinmpc2, y, d)
        u1 = moveinput!(linmpc, r, d)
        u2 = moveinput!(nonlinmpc1, r, d)
        u3 = moveinput!(nonlinmpc2, r, d)
        U_linmpc[:, i] = u1
        U_nonlinmpc1[:, i] = u2 
        U_nonlinmpc2[:, i] = u3
        updatestate!(linmpc, u1, y, d)
        updatestate!(nonlinmpc1, u2, y, d)
        updatestate!(nonlinmpc2, u3, y, d)
        updatestate!(linmodel, u1, d)
    end
    @test U_linmpc ≈ U_nonlinmpc1 rtol=1e-3 atol=1e-3
    @test U_linmpc ≈ U_nonlinmpc2 rtol=1e-3 atol=1e-3
    @test U_nonlinmpc1 ≈ U_nonlinmpc2 rtol=1e-3 atol=1e-3
end
