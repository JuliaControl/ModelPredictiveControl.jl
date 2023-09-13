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
    mpc6 = LinMPC(model, Lwt=[0,1], ru=[0,50], Hp=15)
    @test mpc6.L_Hp ≈ Diagonal(diagm(repeat(Float64[0, 1], 15)))
    @test mpc6.R̂u ≈ repeat([0,50], 15)
    mpc7 = LinMPC(model, optim=JuMP.Model(Ipopt.Optimizer))
    @test solver_name(mpc7.optim) == "Ipopt"
    kf = KalmanFilter(model)
    mpc8 = LinMPC(kf)
    @test isa(mpc8.estim, KalmanFilter)

    @test_throws ErrorException LinMPC(model, Hp=0)
    @test_throws ErrorException LinMPC(model, Hc=0)
    @test_throws ErrorException LinMPC(model, Hp=1, Hc=2)
    @test_throws ErrorException LinMPC(model, Mwt=[1])
    @test_throws ErrorException LinMPC(model, Mwt=[1])
    @test_throws ErrorException LinMPC(model, Lwt=[1])
    @test_throws ErrorException LinMPC(model, ru=[1])
    @test_throws ErrorException LinMPC(model, Cwt=[1])
    @test_throws ErrorException LinMPC(model, Mwt=[-1,1])
    @test_throws ErrorException LinMPC(model, Nwt=[-1,1])
    @test_throws ErrorException LinMPC(model, Lwt=[-1,1])
    @test_throws ErrorException LinMPC(model, Cwt=-1)
end

@testset "LinMPC constraints" begin
    model = LinModel(sys, Ts, i_d=[3])
    mpc = LinMPC(model, Hp=1, Hc=1)
    setconstraint!(mpc, umin=[5, 9.9], umax=[100,99])
    @test all((mpc.con.Umin, mpc.con.Umax) .≈ ([5, 9.9], [100,99]))
    setconstraint!(mpc, Δumin=[-5,-10], Δumax=[6,11])
    @test all((mpc.con.ΔŨmin, mpc.con.ΔŨmax) .≈ ([-5,-10,0], [6,11,Inf]))
    setconstraint!(mpc, ymin=[5,10],ymax=[55, 35])
    @test all((mpc.con.Ymin, mpc.con.Ymax) .≈ ([5,10], [55,35]))
    setconstraint!(mpc, c_umin=[0.1,0.2], c_umax=[0.3,0.4])
    @test all((-mpc.con.A_Umin[:, end], -mpc.con.A_Umax[:, end]) .≈ ([0.1,0.2], [0.3,0.4]))
    setconstraint!(mpc, c_Δumin=[0.05,0.15], c_Δumax=[0.25,0.35])
    @test all((-mpc.con.A_ΔŨmin[1:end-1, end], -mpc.con.A_ΔŨmax[1:end-1, end]) .≈ ([0.05,0.15], [0.25,0.35]))
    setconstraint!(mpc, c_ymin=[1.0,1.1], c_ymax=[1.2,1.3])
    @test all((-mpc.con.A_Ymin[:, end], -mpc.con.A_Ymax[:, end]) .≈ ([1.0,1.1], [1.2,1.3]))
end

@testset "LinMPC moves and getinfo" begin
    mpc1 = LinMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1000, Hc=1)
    r = [5]
    u = moveinput!(mpc1, r)
    @test u ≈ [1] atol=1e-2
    u = mpc1(r)
    @test u ≈ [1] atol=1e-2
    _ , info = getinfo(mpc1)
    @test info[:u] ≈ u
    @test info[:Ŷ][end] ≈ r[1] atol=1e-2
    mpc2 = LinMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Cwt=Inf, Hp=1000, Hc=1)
    u = moveinput!(mpc2, [5])
    @test u ≈ [1] atol=1e-2
    mpc3 = LinMPC(LinModel(tf(5, [2, 1]), 3), Mwt=[0], Nwt=[0], Lwt=[1], ru=[12])
    u = moveinput!(mpc3, [0])
    @test u ≈ [12] atol=1e-2
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

@testset "ExplicitMPC construction" begin
    model = LinModel(sys, Ts, i_d=[3])
    mpc1 = ExplicitMPC(model, Hp=15)
    @test isa(mpc1.estim, SteadyKalmanFilter)
    @test size(mpc1.Ẽ,1) == 15*mpc1.estim.model.ny
    mpc4 = ExplicitMPC(model, Mwt=[1,2], Hp=15)
    @test mpc4.M_Hp ≈ Diagonal(diagm(repeat(Float64[1, 2], 15)))
    mpc5 = ExplicitMPC(model, Nwt=[3,4], Hc=5)
    @test mpc5.Ñ_Hc ≈ Diagonal(diagm(repeat(Float64[3, 4], 5)))
    mpc6 = ExplicitMPC(model, Lwt=[0,1], ru=[0,50], Hp=15)
    @test mpc6.L_Hp ≈ Diagonal(diagm(repeat(Float64[0, 1], 15)))
    @test mpc6.R̂u ≈ repeat([0,50], 15)
    kf = KalmanFilter(model)
    mpc8 = ExplicitMPC(kf)
    @test isa(mpc8.estim, KalmanFilter)

    @test_throws ErrorException ExplicitMPC(model, Hp=0)
    @test_throws ErrorException ExplicitMPC(model, Hc=0)
    @test_throws ErrorException ExplicitMPC(model, Hp=1, Hc=2)
    @test_throws ErrorException ExplicitMPC(model, Mwt=[1])
    @test_throws ErrorException ExplicitMPC(model, Mwt=[1])
    @test_throws ErrorException ExplicitMPC(model, Lwt=[1])
    @test_throws ErrorException ExplicitMPC(model, ru=[1])
    @test_throws ErrorException ExplicitMPC(model, Mwt=[-1,1])
    @test_throws ErrorException ExplicitMPC(model, Nwt=[-1,1])
    @test_throws ErrorException ExplicitMPC(model, Lwt=[-1,1])
end

@testset "ExplicitMPC constraints" begin
    model = LinModel(sys, Ts, i_d=[3])
    mpc = ExplicitMPC(model, Hp=1, Hc=1)
    @test_throws ErrorException setconstraint!(mpc, umin=[0.0, 0.0])
end

@testset "ExplicitMPC moves and getinfo" begin
    mpc1 = ExplicitMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1000, Hc=1)
    r = [5]
    u = moveinput!(mpc1, r)
    @test u ≈ [1] atol=1e-2
    u = mpc1(r)
    @test u ≈ [1] atol=1e-2
    _ , info = getinfo(mpc1)
    @test info[:u] ≈ u
    @test info[:Ŷ][end] ≈ r[1] atol=1e-2
    mpc2 = ExplicitMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1000, Hc=1)
    u = moveinput!(mpc2, [5])
    @test u ≈ [1] atol=1e-2
    mpc3 = ExplicitMPC(LinModel(tf(5, [2, 1]), 3), Mwt=[0], Nwt=[0], Lwt=[1], ru=[12])
    u = moveinput!(mpc3, [0])
    @test u ≈ [12] atol=1e-2
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
    nmpc2 = NonLinMPC(nonlinmodel, Hc=4, Cwt=Inf)
    @test size(nmpc2.Ẽ, 2) == 4*nonlinmodel.nu
    nmpc3 = NonLinMPC(nonlinmodel, Hc=4, Cwt=1e6)
    @test size(nmpc3.Ẽ, 2) == 4*nonlinmodel.nu + 1
    @test nmpc3.C == 1e6
    nmpc4 = NonLinMPC(nonlinmodel, Mwt=[1,2], Hp=15)
    @test nmpc4.M_Hp ≈ Diagonal(diagm(repeat(Float64[1, 2], 15)))
    nmpc5 = NonLinMPC(nonlinmodel, Nwt=[3,4], Cwt=1e3, Hc=5)
    @test nmpc5.Ñ_Hc ≈ Diagonal(diagm([repeat(Float64[3, 4], 5); [1e3]]))
    nmpc6 = NonLinMPC(nonlinmodel, Lwt=[0,1], ru=[0,50], Hp=15)
    @test nmpc6.L_Hp ≈ Diagonal(diagm(repeat(Float64[0, 1], 15)))
    @test nmpc6.R̂u ≈ repeat([0,50], 15)
    nmpc7 = NonLinMPC(nonlinmodel, Ewt=1e-3, JE=(UE,ŶE,D̂E) -> UE.*ŶE.*D̂E)
    @test nmpc7.E == 1e-3
    @test nmpc7.JE([1,2],[3,4],[4,6]) == [12, 48]
    nmpc8 = NonLinMPC(nonlinmodel, optim=JuMP.Model(OSQP.MathOptInterfaceOSQP.Optimizer))
    @test solver_name(nmpc8.optim) == "OSQP"
    im = InternalModel(nonlinmodel)
    nmpc9 = NonLinMPC(im)
    @test isa(nmpc9.estim, InternalModel)
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
    setconstraint!(nmpc, umin=[5, 9.9], umax=[100,99])
    @test all((nmpc.con.Umin, nmpc.con.Umax) .≈ ([5, 9.9], [100,99]))
    setconstraint!(nmpc, Δumin=[-5,-10], Δumax=[6,11])
    @test all((nmpc.con.ΔŨmin, nmpc.con.ΔŨmax) .≈ ([-5,-10,0], [6,11,Inf]))
    setconstraint!(nmpc, ymin=[5,10],ymax=[55, 35])
    @test all((nmpc.con.Ymin, nmpc.con.Ymax) .≈ ([5,10], [55,35]))
    setconstraint!(nmpc, c_umin=[0.1,0.2], c_umax=[0.3,0.4])
    @test all((-nmpc.con.A_Umin[:, end], -nmpc.con.A_Umax[:, end]) .≈ ([0.1,0.2], [0.3,0.4]))
    setconstraint!(nmpc, c_Δumin=[0.05,0.15], c_Δumax=[0.25,0.35])
    @test all((-nmpc.con.A_ΔŨmin[1:end-1, end], -nmpc.con.A_ΔŨmax[1:end-1, end]) .≈ ([0.05,0.15], [0.25,0.35]))
    setconstraint!(nmpc, c_ymin=[1.0,1.1], c_ymax=[1.2,1.3])
    @test all((-nmpc.con.A_Ymin, -nmpc.con.A_Ymax) .≈ (zeros(0,3), zeros(0,3)))
    @test all((nmpc.con.c_Ymin, nmpc.con.c_Ymax) .≈ ([1.0,1.1], [1.2,1.3]))
end

@testset "NonLinMPC moves and getinfo" begin
    linmodel = LinModel(tf(5, [2, 1]), 3.0)
    nmpc_lin = NonLinMPC(linmodel, Nwt=[0], Hp=1000, Hc=1)
    r = [5]
    u = moveinput!(nmpc_lin, r)
    @test u ≈ [1] atol=5e-2
    u = nmpc_lin(r)
    @test u ≈ [1] atol=5e-2
    _ , info = getinfo(nmpc_lin)
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
    r = 7*d
    u = moveinput!(nmpc2, r, d)
    @test u ≈ [0] atol=5e-2
    u = nmpc2(r, d)
    @test u ≈ [0] atol=5e-2
    _ , info = getinfo(nmpc2)
    @test info[:u] ≈ u
    @test info[:Ŷ][end] ≈ r[1] atol=5e-2
    nmpc3 = NonLinMPC(nonlinmodel, Nwt=[0], Cwt=Inf, Hp=1000, Hc=1)
    u = moveinput!(nmpc3, r, d)
    @test u ≈ [0] atol=5e-2
    nmpc4 = NonLinMPC(nonlinmodel, Mwt=[0], Nwt=[0], Lwt=[1], ru=[12])
    u = moveinput!(nmpc4, [0], d)
    @test u ≈ [12] atol=5e-2
    nmpc5 = setconstraint!(NonLinMPC(nonlinmodel, Cwt=Inf), ymin=[-1])
    C_Ymax_end = nmpc5.optim.nlp_model.operators.registered_multivariate_operators[end].f
    @test C_Ymax_end(Float64.((1.0, 1.0))) ≤ 0.0 # test con_nonlinprog_i(i,::NTuple{N, Float64})
    @test C_Ymax_end(Float32.((1.0, 1.0))) ≤ 0.0 # test con_nonlinprog_i(i,::NTuple{N, Real})
end

@testset "NonLinMPC other methods" begin
    linmodel = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    f(x,u,_) = linmodel.A*x + linmodel.Bu*u
    h(x,_)   = linmodel.C*x
    nonlinmodel = NonLinModel(f, h, Ts, 2, 2, 2) 
    nmpc1 = NonLinMPC(nonlinmodel)
    @test initstate!(nmpc1, [10, 50], [20, 25]) ≈ zeros(4)
    setstate!(nmpc1, [1,2,3,4])
    @test nmpc1.estim.x̂ ≈ [1,2,3,4]
    setstate!(nmpc1, [0,0,0,0])
    updatestate!(nmpc1, nmpc1.estim.model.uop, nmpc1.estim())
    @test nmpc1.estim.x̂ ≈ [0,0,0,0] atol=1e-6
end
