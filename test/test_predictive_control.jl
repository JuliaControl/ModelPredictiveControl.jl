Ts = 4.0
sys = [ tf(1.90,[18.0,1])   tf(1.90,[18.0,1])   tf(1.90,[18.0,1]);
        tf(-0.74,[8.0,1])   tf(0.74,[8.0,1])    tf(-0.74,[8.0,1])   ]

@testset "LinMPC construction" begin

model = LinModel(sys, Ts, i_d=[3])
mpc1 = LinMPC(model, Hp=15)
@test isa(mpc1.estim, SteadyKalmanFilter)
@test size(mpc1.Ẽ,1) == 15*mpc1.model.ny

mpc2 = LinMPC(model, Hc=4, Cwt=Inf)
@test size(mpc2.Ẽ,2) == 4*mpc2.model.nu

mpc3 = LinMPC(model, Hc=4, Cwt=1e5)
@test size(mpc3.Ẽ,2) == 4*mpc3.model.nu + 1

kf = KalmanFilter(model)
mpc4 = LinMPC(kf)
@test isa(mpc4.estim, KalmanFilter)

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
@test all((mpc.Umin, mpc.Umax) .≈ ([5, 9.9], [100,99]))
setconstraint!(mpc, Δumin=[-5,-10], Δumax=[6,11])
@test all((mpc.ΔŨmin, mpc.ΔŨmax) .≈ ([-5,-10,0], [6,11,Inf]))
setconstraint!(mpc, ŷmin=[5,10],ŷmax=[55, 35])
@test all((mpc.Ŷmin, mpc.Ŷmax) .≈ ([5,10], [55,35]))
setconstraint!(mpc, c_umin=[0.1,0.2], c_umax=[0.3,0.4])
@test all((mpc.c_Umin, mpc.c_Umax) .≈ ([0.1,0.2], [0.3,0.4]))
setconstraint!(mpc, c_Δumin=[0.05,0.15], c_Δumax=[0.25,0.35])
@test all((mpc.c_ΔUmin, mpc.c_ΔUmax) .≈ ([0.05,0.15], [0.25,0.35]))
setconstraint!(mpc, c_ŷmin=[1.0,1.1], c_ŷmax=[1.2,1.3])
@test all((mpc.c_Ŷmin, mpc.c_Ŷmax) .≈ ([1.0,1.1], [1.2,1.3]))

end


@testset "LinMPC moves" begin

mpc = LinMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1000, Hc=1)
r = [5]
u = moveinput!(mpc, r)
@test u ≈ [1] atol=1e-3
u = mpc(r)
@test u ≈ [1] atol=1e-3

end