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
    setconstraint!(mpc, ŷmin=[5,10],ŷmax=[55, 35])
    @test all((mpc.con.Ŷmin, mpc.con.Ŷmax) .≈ ([5,10], [55,35]))
    setconstraint!(mpc, c_umin=[0.1,0.2], c_umax=[0.3,0.4])
    @test all((-mpc.con.A_Umin[:, end], -mpc.con.A_Umax[:, end]) .≈ ([0.1,0.2], [0.3,0.4]))
    setconstraint!(mpc, c_Δumin=[0.05,0.15], c_Δumax=[0.25,0.35])
    @test all((-mpc.con.A_ΔŨmin[1:end-1, end], -mpc.con.A_ΔŨmax[1:end-1, end]) .≈ ([0.05,0.15], [0.25,0.35]))
    setconstraint!(mpc, c_ŷmin=[1.0,1.1], c_ŷmax=[1.2,1.3])
    @test all((-mpc.con.A_Ŷmin[:, end], -mpc.con.A_Ŷmax[:, end]) .≈ ([1.0,1.1], [1.2,1.3]))
end

@testset "LinMPC moves and getinfo" begin
    mpc = LinMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1000, Hc=1)
    r = [5]
    u = moveinput!(mpc, r)
    @test u ≈ [1] atol=1e-2
    u = mpc(r)
    @test u ≈ [1] atol=1e-2
    info, _ = getinfo(mpc)
    @test info[:u] ≈ u
    @test info[:Ŷ][end] ≈ 5 atol=1e-2
end

@testset "LinMPC other methods" begin
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    mpc1 = LinMPC(linmodel1)
    @test initstate!(mpc1, [10, 50], [50, 30+1]) ≈ [zeros(3); [1]]
    setstate!(mpc1, [1,2,3,4])
    @test mpc1.estim.x̂ ≈ [1,2,3,4]
end


@testset "NonLinMPC construction" begin
    linmodel1 = LinModel(sys,Ts,i_d=[3])
    f(x,u,d) = linmodel1.A*x + linmodel1.Bu*u + linmodel1.Bd*d
    h(x,d)   = linmodel1.C*x + linmodel1.Du*d
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
    f(x,u,d) = linmodel1.A*x + linmodel1.Bu*u + linmodel1.Bd*d
    h(x,d)   = linmodel1.C*x + linmodel1.Du*d
    nonlinmodel = NonLinModel(f, h, Ts, 2, 4, 2, 1)
    nmpc = NonLinMPC(nonlinmodel, Hp=1, Hc=1)
    setconstraint!(nmpc, umin=[5, 9.9], umax=[100,99])
    @test all((nmpc.con.Umin, nmpc.con.Umax) .≈ ([5, 9.9], [100,99]))
    setconstraint!(nmpc, Δumin=[-5,-10], Δumax=[6,11])
    @test all((nmpc.con.ΔŨmin, nmpc.con.ΔŨmax) .≈ ([-5,-10,0], [6,11,Inf]))
    setconstraint!(nmpc, ŷmin=[5,10],ŷmax=[55, 35])
    @test all((nmpc.con.Ŷmin, nmpc.con.Ŷmax) .≈ ([5,10], [55,35]))
    setconstraint!(nmpc, c_umin=[0.1,0.2], c_umax=[0.3,0.4])
    @test all((-nmpc.con.A_Umin[:, end], -nmpc.con.A_Umax[:, end]) .≈ ([0.1,0.2], [0.3,0.4]))
    setconstraint!(nmpc, c_Δumin=[0.05,0.15], c_Δumax=[0.25,0.35])
    @test all((-nmpc.con.A_ΔŨmin[1:end-1, end], -nmpc.con.A_ΔŨmax[1:end-1, end]) .≈ ([0.05,0.15], [0.25,0.35]))
    setconstraint!(nmpc, c_ŷmin=[1.0,1.1], c_ŷmax=[1.2,1.3])
    println(nmpc.con.A_Ŷmin)
    @test all((-nmpc.con.A_Ŷmin, -nmpc.con.A_Ŷmax) .≈ (zeros(0,3), zeros(0,3)))
    @test all((nmpc.con.c_Ŷmin, nmpc.con.c_Ŷmax) .≈ ([1.0,1.1], [1.2,1.3]))
end

@testset "NonLinMPC moves and getinfo" begin
    linmodel = LinModel(tf(5, [2, 1]), 3)
    nmpc_lin = NonLinMPC(linmodel, Nwt=[0], Hp=1000, Hc=1)
    r = [5]
    u = moveinput!(nmpc_lin, r)
    @test u ≈ [1] atol=1e-3
    u = nmpc_lin(r)
    @test u ≈ [1] atol=1e-3
    info, _ = getinfo(nmpc_lin)
    @test info[:u] ≈ u
    @test info[:Ŷ][end] ≈ 5 atol=1e-2

    f(x,u,_) = linmodel.A*x + linmodel.Bu*u
    h(x,_)   = linmodel.C*x 
    nonlinmodel = NonLinModel(f, h, 3, 1, 1, 1)
    nmpc_nonlin = NonLinMPC(nonlinmodel, Nwt=[0], Hp=1000, Hc=1)
    r = [5]
    u = moveinput!(nmpc_nonlin, r)
    @test u ≈ [1] atol=1e-2
    u = nmpc_nonlin(r)
    @test u ≈ [1] atol=1e-2
    info, _ = getinfo(nmpc_nonlin)
    @test info[:u] ≈ u
    @test info[:Ŷ][end] ≈ 5 atol=1e-2
end

@testset "NonLinMPC other methods" begin
    linmodel = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    f(x,u,_) = linmodel.A*x + linmodel.Bu*u
    h(x,_)   = linmodel.C*x
    nonlinmodel = NonLinModel(f, h, Ts, 2, 2, 2) 
    nmpc1 = NonLinMPC(nonlinmodel)
    @test initstate!(nmpc1, [10, 50], [20, 25]) ≈ [zeros(2); [20, 25]]
    setstate!(nmpc1, [1,2,3,4])
    @test nmpc1.estim.x̂ ≈ [1,2,3,4]
end