Ts = 4.0
sys = [ tf(1.90,[18.0,1])   tf(1.90,[18.0,1])   tf(1.90,[18.0,1]);
        tf(-0.74,[8.0,1])   tf(0.74,[8.0,1])    tf(-0.74,[8.0,1])   ]

@testset "LinMPC construction" begin
    model = LinModel(sys, Ts, i_d=[3])
    mpc1 = LinMPC(model, Hp=15)
    @test isa(mpc1.estim, SteadyKalmanFilter)
    @test size(mpc1.Ẽ,1) == 15*mpc1.estim.model.ny
    mpc2 = LinMPC(model, Hc=4, Cwt=Inf)
    @test size(mpc2.Ẽ,2) == 4*mpc2.estim.model.nu
    mpc3 = LinMPC(model, Hc=4, Cwt=1e6)
    @test size(mpc3.Ẽ,2) == 4*mpc3.estim.model.nu + 1
    @test mpc3.C ≈ 1e6
    mpc4 = LinMPC(model, Mwt=[1,2], Hp=15)
    @test mpc4.M_Hp ≈ Diagonal(diagm(repeat(Float64[1, 2], 15)))
    mpc5 = LinMPC(model, Nwt=[3,4], Cwt=1e3, Hc=5)
    @test mpc5.Ñ_Hc ≈ Diagonal(diagm([repeat(Float64[3, 4], 5); [1e3]]))
    mpc6 = LinMPC(model, Lwt=[0,1], Hp=15)
    @test mpc6.L_Hp ≈ Diagonal(diagm(repeat(Float64[0, 1], 15)))
    mpc7 = LinMPC(model, optim=JuMP.Model(Ipopt.Optimizer))
    @test solver_name(mpc7.optim) == "Ipopt"
    kf = KalmanFilter(model)
    mpc8 = LinMPC(kf)
    @test isa(mpc8.estim, KalmanFilter)
    mpc9 = LinMPC(model, nint_u=[1, 1], nint_ym=[0, 0])
    @test mpc9.estim.nint_u  == [1, 1]
    @test mpc9.estim.nint_ym == [0, 0]

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
    mpc1 = LinMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1000, Hc=1)
    r = [5]
    u = moveinput!(mpc1, r)
    @test u ≈ [1] atol=1e-2
    u = mpc1(r)
    @test u ≈ [1] atol=1e-2
    info = getinfo(mpc1)
    @test info[:u] ≈ u
    @test info[:Ŷ][end] ≈ r[1] atol=1e-2
    mpc2 = LinMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Cwt=Inf, Hp=1000, Hc=1)
    u = moveinput!(mpc2, [5])
    @test u ≈ [1] atol=1e-2
    mpc3 = LinMPC(LinModel(tf(5, [2, 1]), 3), Mwt=[0], Nwt=[0], Lwt=[1])
    u = moveinput!(mpc3, [0], R̂u=fill(12, mpc3.Hp))
    @test u ≈ [12] atol=1e-2
    mpc_im = LinMPC(InternalModel(LinModel(tf(5, [2, 1]), 3)))
    ym, u = mpc_im.estim.model() - [5], [0.0]
    for i=1:25
        ym = mpc_im.estim.model() - [5]
        u = moveinput!(mpc_im, r; ym)
        updatestate!(mpc_im, u, ym)
        updatestate!(mpc_im.estim.model, u)
    end
    @test u  ≈ [2] atol=1e-2
    @test ym ≈ [5] atol=1e-2
    mpc_nint_u = LinMPC(SteadyKalmanFilter(LinModel(tf(5, [2, 1]), 3), nint_u=[1]))
    ym, u = mpc_nint_u.estim.model() - [5], [0.0]
    for i=1:25
        ym = mpc_nint_u.estim.model() - [5]
        u = moveinput!(mpc_nint_u, r; ym)
        updatestate!(mpc_nint_u, u, ym)
        updatestate!(mpc_nint_u.estim.model, u)
    end
    @test u  ≈ [2] atol=1e-2
    @test ym ≈ [5] atol=1e-2 
    mpc_nint_ym = LinMPC(SteadyKalmanFilter(LinModel(tf(5, [2, 1]), 3), nint_ym=[1]))
    ym, u = mpc_nint_ym.estim.model() - [5], [0.0]
    for i=1:25
        ym = mpc_nint_ym.estim.model() - [5]
        u = moveinput!(mpc_nint_ym, r; ym)
        updatestate!(mpc_nint_ym, u, ym)
        updatestate!(mpc_nint_ym.estim.model, u)
    end
    @test u  ≈ [2] atol=1e-2
    @test ym ≈ [5] atol=1e-2 

    @test_throws ArgumentError moveinput!(mpc1, [0,0,0])
    @test_throws ArgumentError moveinput!(mpc1, [0], [0,0])
    @test_throws ArgumentError moveinput!(mpc1; D̂  = fill(0, mpc1.Hp+1))
    @test_throws ArgumentError moveinput!(mpc1; R̂y = fill(0, mpc1.Hp+1))
    @test_throws ArgumentError moveinput!(mpc1; R̂u = fill(0, mpc1.Hp+1))
end

@testset "LinMPC other methods" begin
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    mpc1 = LinMPC(linmodel1)
    @test initstate!(mpc1, [10, 50], [50, 30+1]) ≈ [zeros(3); [1]]
    setstate!(mpc1, [1,2,3,4])
    @test mpc1.estim.x̂ ≈ [1,2,3,4]
    setstate!(mpc1, [0,0,0,0])
    updatestate!(mpc1, mpc1.estim.model.uop, mpc1.estim())
    @test mpc1.estim.x̂ ≈ [0,0,0,0]
    @test_throws ArgumentError updatestate!(mpc1, [0,0])
end

@testset "LinMPC constraints" begin
    model = LinModel(sys, Ts, i_d=[3])
    mpc = LinMPC(model, Hp=1, Hc=1)
    setconstraint!(mpc, umin=[-5, -9.9], umax=[100,99])
    @test all((mpc.con.Umin, mpc.con.Umax) .≈ ([-5, -9.9], [100,99]))
    setconstraint!(mpc, Δumin=[-5,-10], Δumax=[6,11])
    @test all((mpc.con.ΔŨmin, mpc.con.ΔŨmax) .≈ ([-5,-10,0], [6,11,Inf]))
    setconstraint!(mpc, ymin=[-6, -11],ymax=[55, 35])
    @test all((mpc.con.Ymin, mpc.con.Ymax) .≈ ([-6,-11], [55,35]))



    setconstraint!(mpc, x̂min=[-21,-22,-23,-24,-25,-26], x̂max=[21,22,23,24,25,26])
    @test all((mpc.con.x̂min, mpc.con.x̂max) .≈ ([-21,-22,-23,-24,-25,-26], [21,22,23,24,25,26]))


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
    @test all((mpc2.con.Umin, mpc2.con.Umax) .≈ (-1(1:50).-1, +1(1:50).+1))
    setconstraint!(mpc2, ΔUmin=-1(1:5).-2, ΔUmax=+1(1:5).+2)
    @test all((mpc2.con.ΔŨmin, mpc2.con.ΔŨmax) .≈ ([-1(1:5).-2; 0], [+1(1:5).+2; Inf]))
    setconstraint!(mpc2, Ymin=-1(1:50).-3, Ymax=+1(1:50).+3)
    @test all((mpc2.con.Ymin, mpc2.con.Ymax) .≈ (-1(1:50).-3, +1(1:50).+3))
    setconstraint!(mpc2, c_Umin=+1(1:50).+4, c_Umax=+1(1:50).+4)
    @test all((-mpc2.con.A_Umin[:, end], -mpc2.con.A_Umax[:, end]) .≈ (+1(1:50).+4, +1(1:50).+4))
    setconstraint!(mpc2, c_ΔUmin=+1(1:5).+5, c_ΔUmax=+1(1:5).+5)
    @test all((-mpc2.con.A_ΔŨmin[1:end-1, end], -mpc2.con.A_ΔŨmax[1:end-1, end]) .≈ (+1(1:5).+5, +1(1:5).+5))
    setconstraint!(mpc2, c_Ymin=+1(1:50).+6, c_Ymax=+1(1:50).+6)
    @test all((-mpc2.con.A_Ymin[:, end], -mpc2.con.A_Ymax[:, end]) .≈ (+1(1:50).+6, +1(1:50).+6))
    setconstraint!(mpc2, c_umin=[0], c_umax=[0], c_Δumin=[0], c_Δumax=[0], c_ymin=[1], c_ymax=[1])


    setconstraint!(mpc2, umin=[-3], umax=[3])
    setconstraint!(mpc2, Δumin=[-1.5], Δumax=[1.5])
    setconstraint!(mpc2, ymin=[-100], ymax=[100])
    moveinput!(mpc2, [-10])
    info = getinfo(mpc2)
    @test info[:ΔU][begin] ≈ -1.5 atol=1e-2
    @test info[:U][end] ≈ -3 atol=1e-2


    setconstraint!(mpc2, umin=[-10], umax=[10])
    setconstraint!(mpc2, Δumin=[-15], Δumax=[15])
    setconstraint!(mpc2, ymin=[-0.5], ymax=[0.5])
    moveinput!(mpc2, [-10])
    info = getinfo(mpc2)
    @test info[:Ŷ][end] ≈ -0.5 atol=1e-2


    setconstraint!(mpc2, umin=[-10], umax=[10])
    setconstraint!(mpc2, Δumin=[-15], Δumax=[15])
    setconstraint!(mpc2, Ymin=[-0.5; fill(-100, 49)], Ymax=[0.5; fill(+100, 49)])
    moveinput!(mpc2, [-10])
    info = getinfo(mpc2)
    @test info[:Ŷ][end]   ≈ -10  atol=1e-2
    @test info[:Ŷ][begin] ≈ -0.5 atol=1e-2






    setconstraint!(mpc2, umin=[-Inf], umax=[+Inf])
    setconstriant!(mpc2, Δumin=[-Inf], Δumax=[+Inf])
    setconstraint!(mpc2, ymin=[-Inf], ymax=[+Inf])
    setconstraint!(mpc2, x̂min=[-1e-6], x̂max=[+1e-6])
    moveinput!(mpc2, [+1000])
    info = getinfo(mpc2)
    @test info[:x̂end] ≈ 0 atol=1e-5






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

#=
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
end

@testset "ExplicitMPC moves and getinfo" begin
    mpc1 = ExplicitMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1000, Hc=1)
    r = [5]
    u = moveinput!(mpc1, r)
    @test u ≈ [1] atol=1e-2
    u = mpc1(r)
    @test u ≈ [1] atol=1e-2
    info = getinfo(mpc1)
    @test info[:u] ≈ u
    @test info[:Ŷ][end] ≈ r[1] atol=1e-2
    mpc2 = ExplicitMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1000, Hc=1)
    u = moveinput!(mpc2, [5])
    @test u ≈ [1] atol=1e-2
    mpc3 = ExplicitMPC(LinModel(tf(5, [2, 1]), 3), Mwt=[0], Nwt=[0], Lwt=[1])
    u = moveinput!(mpc3, [0], R̂u=fill(12, mpc3.Hp))
    @test u ≈ [12] atol=1e-2
    mpc_im = ExplicitMPC(InternalModel(LinModel(tf(5, [2, 1]), 3)))
    ym, u = mpc_im.estim.model() - [5], [0.0]
    for i=1:25
        ym = mpc_im.estim.model() - [5]
        u = moveinput!(mpc_im, r; ym)
        updatestate!(mpc_im, u, ym)
        updatestate!(mpc_im.estim.model, u)
    end
    @test u  ≈ [2] atol=1e-2
    @test ym ≈ [5] atol=1e-2
    mpc_nint_u = ExplicitMPC(SteadyKalmanFilter(LinModel(tf(5, [2, 1]), 3), nint_u=[1]))
    ym, u = mpc_nint_u.estim.model() - [5], [0.0]
    for i=1:25
        ym = mpc_nint_u.estim.model() - [5]
        u = moveinput!(mpc_nint_u, r; ym)
        updatestate!(mpc_nint_u, u, ym)
        updatestate!(mpc_nint_u.estim.model, u)
    end
    @test u  ≈ [2] atol=1e-2
    @test ym ≈ [5] atol=1e-2 
    mpc_nint_ym = ExplicitMPC(SteadyKalmanFilter(LinModel(tf(5, [2, 1]), 3), nint_ym=[1]))
    ym, u = mpc_nint_ym.estim.model() - [5], [0.0]
    for i=1:25
        ym = mpc_nint_ym.estim.model() - [5]
        u = moveinput!(mpc_nint_ym, r; ym)
        updatestate!(mpc_nint_ym, u, ym)
        updatestate!(mpc_nint_ym.estim.model, u)
    end
    @test u  ≈ [2] atol=1e-2
    @test ym ≈ [5] atol=1e-2
end

@testset "ExplicitMPC other methods" begin
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    mpc1 = ExplicitMPC(linmodel1)
    @test initstate!(mpc1, [10, 50], [50, 30+1]) ≈ [zeros(3); [1]]
    setstate!(mpc1, [1,2,3,4])
    @test mpc1.estim.x̂ ≈ [1,2,3,4]
    setstate!(mpc1, [0,0,0,0])
    updatestate!(mpc1, mpc1.estim.model.uop, mpc1.estim())
    @test mpc1.estim.x̂ ≈ [0,0,0,0]
    @test_throws ArgumentError updatestate!(mpc1, [0,0])
end

@testset "ExplicitMPC constraints" begin
    model = LinModel(sys, Ts, i_d=[3])
    mpc = ExplicitMPC(model, Hp=1, Hc=1)
    @test_throws ErrorException setconstraint!(mpc, umin=[0.0, 0.0])
end

@testset "NonLinMPC construction" begin
    linmodel1 = LinModel(sys,Ts,i_d=[3])
    nmpc0 = NonLinMPC(linmodel1, Hp=15)
    @test isa(nmpc0.estim, SteadyKalmanFilter)
    f(x,u,d) = linmodel1.A*x + linmodel1.Bu*u + linmodel1.Bd*d
    h(x,d)   = linmodel1.C*x + linmodel1.Dd*d
    nonlinmodel = NonLinModel(f, h, Ts, 2, 4, 2, 1)
    nmpc1 = NonLinMPC(nonlinmodel, Hp=15)
    @test isa(nmpc1.estim, UnscentedKalmanFilter)
    @test size(nmpc1.R̂y, 1) == 15*nmpc1.estim.model.ny
    nmpc2 = NonLinMPC(nonlinmodel, Hp=15, Hc=4, Cwt=Inf)
    @test size(nmpc2.Ẽ, 2) == 4*nonlinmodel.nu
    nmpc3 = NonLinMPC(nonlinmodel, Hp=15, Hc=4, Cwt=1e6)
    @test size(nmpc3.Ẽ, 2) == 4*nonlinmodel.nu + 1
    @test nmpc3.C == 1e6
    nmpc4 = NonLinMPC(nonlinmodel, Hp=15, Mwt=[1,2])
    @test nmpc4.M_Hp ≈ Diagonal(diagm(repeat(Float64[1, 2], 15)))
    nmpc5 = NonLinMPC(nonlinmodel, Hp=15 ,Nwt=[3,4], Cwt=1e3, Hc=5)
    @test nmpc5.Ñ_Hc ≈ Diagonal(diagm([repeat(Float64[3, 4], 5); [1e3]]))
    nmpc6 = NonLinMPC(nonlinmodel, Hp=15, Lwt=[0,1])
    @test nmpc6.L_Hp ≈ Diagonal(diagm(repeat(Float64[0, 1], 15)))
    nmpc7 = NonLinMPC(nonlinmodel, Hp=15, Ewt=1e-3, JE=(UE,ŶE,D̂E) -> UE.*ŶE.*D̂E)
    @test nmpc7.E == 1e-3
    @test nmpc7.JE([1,2],[3,4],[4,6]) == [12, 48]
    nmpc8 = NonLinMPC(nonlinmodel, Hp=15, optim=JuMP.Model(OSQP.MathOptInterfaceOSQP.Optimizer))
    @test solver_name(nmpc8.optim) == "OSQP"
    im = InternalModel(nonlinmodel)
    nmpc9 = NonLinMPC(im, Hp=15)
    @test isa(nmpc9.estim, InternalModel)
    nmpc10 = NonLinMPC(linmodel1, nint_u=[1, 1], nint_ym=[0, 0])
    @test nmpc10.estim.nint_u  == [1, 1]
    @test nmpc10.estim.nint_ym == [0, 0]
    nmpc11 = NonLinMPC(nonlinmodel, Hp=15, nint_u=[1, 1], nint_ym=[0, 0])
    @test nmpc11.estim.nint_u  == [1, 1]
    @test nmpc11.estim.nint_ym == [0, 0]

    @test_throws ArgumentError NonLinMPC(nonlinmodel, Hp=15, Ewt=[1, 1])
    # to uncomment when deprecated constructor is removed:
    # @test_throws ArgumentError NonLinMPC(nonlinmodel, Hp=nothing)
end

@testset "NonLinMPC moves and getinfo" begin
    linmodel = LinModel(tf(5, [2, 1]), 3.0)
    nmpc_lin = NonLinMPC(linmodel, Nwt=[0], Hp=1000, Hc=1)
    r = [5]
    u = moveinput!(nmpc_lin, r)
    @test u ≈ [1] atol=5e-2
    u = nmpc_lin(r)
    @test u ≈ [1] atol=5e-2
    info = getinfo(nmpc_lin)
    @test info[:u] ≈ u
    @test info[:Ŷ][end] ≈ r[1] atol=5e-2
    Hp = 1000
    R̂y = fill(5, Hp)
    JE = (_ , ŶE, _ ) -> sum((ŶE[2:end] - R̂y).^2)
    nmpc = NonLinMPC(linmodel, Mwt=[0], Nwt=[0], Cwt=Inf, Ewt=1, JE=JE, Hp=Hp, Hc=1)
    u = moveinput!(nmpc)
    @test u ≈ [1] atol=5e-2
    linmodel = LinModel([tf(5, [2, 1]) tf(7, [8,1])], 3.0, i_d=[2])
    f(x,u,d) = linmodel.A*x + linmodel.Bu*u + linmodel.Bd*d
    h(x,d)   = linmodel.C*x + linmodel.Dd*d
    nonlinmodel = NonLinModel(f, h, 3.0, 1, 2, 1, 1)
    nmpc2 = NonLinMPC(nonlinmodel, Nwt=[0], Hp=1000, Hc=1)
    d = [0.1]
    u = moveinput!(nmpc2, 7d, d)
    @test u ≈ [0] atol=5e-2
    u = nmpc2(7d, d)
    @test u ≈ [0] atol=5e-2
    info = getinfo(nmpc2)
    @test info[:u] ≈ u
    @test info[:Ŷ][end] ≈ 7d[1] atol=5e-2
    nmpc3 = NonLinMPC(nonlinmodel, Nwt=[0], Cwt=Inf, Hp=1000, Hc=1)
    u = moveinput!(nmpc3, 7d, d)
    @test u ≈ [0] atol=5e-2
    nmpc4 = NonLinMPC(nonlinmodel, Hp=15, Mwt=[0], Nwt=[0], Lwt=[1])
    u = moveinput!(nmpc4, [0], d, R̂u=fill(12, nmpc4.Hp))
    @test u ≈ [12] atol=5e-2
    nmpc5 = setconstraint!(NonLinMPC(nonlinmodel, Hp=15, Cwt=Inf), ymin=[-1])
    C_Ymax_end = nmpc5.optim.nlp_model.operators.registered_multivariate_operators[end].f
    @test C_Ymax_end(Float64.((1.0, 1.0))) ≤ 0.0 # test con_nonlinprog_i(i,::NTuple{N, Float64})
    @test C_Ymax_end(Float32.((1.0, 1.0))) ≤ 0.0 # test con_nonlinprog_i(i,::NTuple{N, Real})
    nmpc_im = NonLinMPC(InternalModel(LinModel(tf(5, [2, 1]), 3)))
    ym, u = nmpc_im.estim.model() - [5], [0.0]
    for i=1:25
        ym = nmpc_im.estim.model() - [5]
        u = moveinput!(nmpc_im, r; ym)
        updatestate!(nmpc_im, u, ym)
        updatestate!(nmpc_im.estim.model, u)
    end
    @test u  ≈ [2] atol=1e-2
    @test ym ≈ [5] atol=1e-2    
    nmpc_nint_u = NonLinMPC(SteadyKalmanFilter(LinModel(tf(5, [2, 1]), 3), nint_u=[1]))
    ym, u = nmpc_nint_u.estim.model() - [5], [0.0]
    for i=1:25
        ym = nmpc_nint_u.estim.model() - [5]
        u = moveinput!(nmpc_nint_u, r; ym)
        updatestate!(nmpc_nint_u, u, ym)
        updatestate!(nmpc_nint_u.estim.model, u)
    end
    @test u  ≈ [2] atol=1e-2
    @test ym ≈ [5] atol=1e-2 
    nmpc_nint_ym = NonLinMPC(SteadyKalmanFilter(LinModel(tf(5, [2, 1]), 3), nint_ym=[1]))
    ym, u = nmpc_nint_ym.estim.model() - [5], [0.0]
    for i=1:25
        ym = nmpc_nint_ym.estim.model() - [5]
        u = moveinput!(nmpc_nint_ym, r; ym)
        updatestate!(nmpc_nint_ym, u, ym)
        updatestate!(nmpc_nint_ym.estim.model, u)
    end
    @test u  ≈ [2] atol=1e-2
    @test ym ≈ [5] atol=1e-2
    
end

@testset "NonLinMPC other methods" begin
    linmodel = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    f(x,u,_) = linmodel.A*x + linmodel.Bu*u
    h(x,_)   = linmodel.C*x
    nonlinmodel = NonLinModel(f, h, Ts, 2, 2, 2) 
    nmpc1 = NonLinMPC(nonlinmodel, Hp=15)
    @test initstate!(nmpc1, [10, 50], [20, 25]) ≈ zeros(4)
    setstate!(nmpc1, [1,2,3,4])
    @test nmpc1.estim.x̂ ≈ [1,2,3,4]
    setstate!(nmpc1, [0,0,0,0])
    updatestate!(nmpc1, nmpc1.estim.model.uop, nmpc1.estim())
    @test nmpc1.estim.x̂ ≈ [0,0,0,0] atol=1e-6
end

@testset "NonLinMPC constraints" begin
    linmodel1 = LinModel(sys,Ts,i_d=[3])
    nmpc_lin = NonLinMPC(linmodel1, Hp=1, Hc=1)
    setconstraint!(nmpc_lin, ymin=[5,10],ymax=[55, 35])
    @test all((nmpc_lin.con.Ymin, nmpc_lin.con.Ymax) .≈ ([5,10], [55,35]))
    setconstraint!(nmpc_lin, c_ymin=[1.0,1.1], c_ymax=[1.2,1.3])
    @test all((-nmpc_lin.con.A_Ymin[:, end], -nmpc_lin.con.A_Ymax[:, end]) .≈ ([1.0,1.1], [1.2,1.3]))
    f(x,u,d) = linmodel1.A*x + linmodel1.Bu*u + linmodel1.Bd*d
    h(x,d)   = linmodel1.C*x + linmodel1.Dd*d
    nonlinmodel = NonLinModel(f, h, Ts, 2, 4, 2, 1)
    nmpc = NonLinMPC(nonlinmodel, Hp=1, Hc=1)
    setconstraint!(nmpc, umin=[-5, -9.9], umax=[100,99])
    @test all((nmpc.con.Umin, nmpc.con.Umax) .≈ ([-5, -9.9], [100,99]))
    setconstraint!(nmpc, Δumin=[-5,-10], Δumax=[6,11])
    @test all((nmpc.con.ΔŨmin, nmpc.con.ΔŨmax) .≈ ([-5,-10,0], [6,11,Inf]))
    setconstraint!(nmpc, ymin=[-6, -11],ymax=[55, 35])
    @test all((nmpc.con.Ymin, nmpc.con.Ymax) .≈ ([-6,-11], [55,35]))
    setconstraint!(nmpc, c_umin=[0.01,0.02], c_umax=[0.03,0.04])
    @test all((-nmpc.con.A_Umin[:, end], -nmpc.con.A_Umax[:, end]) .≈ ([0.01,0.02], [0.03,0.04]))
    setconstraint!(nmpc, c_Δumin=[0.05,0.06], c_Δumax=[0.07,0.08])
    @test all((-nmpc.con.A_ΔŨmin[1:end-1, end], -nmpc.con.A_ΔŨmax[1:end-1, end]) .≈ ([0.05,0.06], [0.07,0.08]))
    setconstraint!(nmpc, c_ymin=[1.00,1.01], c_ymax=[1.02,1.03])
    @test all((-nmpc.con.A_Ymin, -nmpc.con.A_Ymax) .≈ (zeros(0,3), zeros(0,3)))
    @test all((nmpc.con.c_Ymin, nmpc.con.c_Ymax) .≈ ([1.00,1.01], [1.02,1.03]))

    linmodel2 = LinModel(tf([2], [10, 1]), 3.0)
    nmpc_lin2 = NonLinMPC(linmodel2, Hp=50, Hc=5)

    setconstraint!(nmpc_lin2, umin=[-3], umax=[3])
    setconstraint!(nmpc_lin2, Δumin=[-1.5], Δumax=[1.5])
    setconstraint!(nmpc_lin2, ymin=[-100], ymax=[100])
    moveinput!(nmpc_lin2, [-20])
    info = getinfo(nmpc_lin2)
    @test info[:ΔU][begin] ≈ -1.5 atol=1e-2
    @test info[:U][end] ≈ -3 atol=1e-2

    setconstraint!(nmpc_lin2, umin=[-10], umax=[10])
    setconstraint!(nmpc_lin2, Δumin=[-15], Δumax=[15])
    setconstraint!(nmpc_lin2, ymin=[-0.5], ymax=[0.5])
    moveinput!(nmpc_lin2, [-20])
    info = getinfo(nmpc_lin2)
    @test info[:Ŷ][end] ≈ -0.5 atol=1e-2

    setconstraint!(nmpc_lin2, umin=[-10], umax=[10])
    setconstraint!(nmpc_lin2, Δumin=[-15], Δumax=[15])
    setconstraint!(nmpc_lin2, Ymin=[-0.5; fill(-100, 49)], Ymax=[0.5; fill(+100, 49)])
    moveinput!(nmpc_lin2, [-10])
    info = getinfo(nmpc_lin2)
    @test info[:Ŷ][end]   ≈ -10  atol=1e-2
    @test info[:Ŷ][begin] ≈ -0.5 atol=1e-2

    f2(x,u,_) = linmodel2.A*x + linmodel2.Bu*u
    h2(x,_)   = linmodel2.C*x
    nonlinmodel2 = NonLinModel(f2, h2, Ts, 1, 1, 1)
    nmpc2 = NonLinMPC(nonlinmodel2, Hp=50, Hc=5)

    setconstraint!(nmpc2, umin=[-3], umax=[3])
    setconstraint!(nmpc2, Δumin=[-1.5], Δumax=[1.5])
    setconstraint!(nmpc2, ymin=[-100], ymax=[100])
    moveinput!(nmpc2, [-20])
    info = getinfo(nmpc2)
    @test info[:ΔU][begin] ≈ -1.5 atol=1e-2
    @test info[:U][end] ≈ -3 atol=1e-2
    
    setconstraint!(nmpc2, umin=[-10], umax=[10])
    setconstraint!(nmpc2, Δumin=[-15], Δumax=[15])
    setconstraint!(nmpc2, ymin=[-0.5], ymax=[0.5])
    moveinput!(nmpc2, [-20])
    info = getinfo(nmpc2)
    @test info[:Ŷ][end] ≈ -0.5 atol=1e-2
    
    setconstraint!(nmpc2, umin=[-10], umax=[10])
    setconstraint!(nmpc2, Δumin=[-15], Δumax=[15])
    setconstraint!(nmpc2, Ymin=[-0.5; fill(-100, 49)], Ymax=[0.5; fill(+100, 49)])
    moveinput!(nmpc2, [-10])
    info = getinfo(nmpc2)
    @test info[:Ŷ][end]   ≈ -10  atol=1e-2
    @test info[:Ŷ][begin] ≈ -0.5 atol=1e-2
    
end
=#