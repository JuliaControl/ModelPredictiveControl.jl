# spell-checker: disable

using Pkg
using Revise
Pkg.activate(".")


using ModelPredictiveControl
using Preferences
set_preferences!(ModelPredictiveControl, "precompile_workload" => false; force=true)


#using JuMP, DAQP
#using JuMP, HiGHS
using JuMP, Ipopt
using LinearAlgebra
using ControlSystemsBase 
using MAT

vars_ml = matread("example/matlab.mat")

A   = vars_ml["mMPC"]["A"]
Bu  = vars_ml["mMPC"]["B"]
Bd  = vars_ml["mMPC"]["Bd"]
C   = vars_ml["mMPC"]["C"]
Du  = zeros(size(C,1),size(Bu,2))
Dd  = vars_ml["mMPC"]["Dd"]
Ts  = vars_ml["mMPC"]["Ts"]

linModel1 = LinModel(ss(A,Bu,C,0,Ts),Ts)
linModel2 = LinModel(ss(A,[Bu Bd],C,[Du Dd],Ts),Ts,i_d=[3])
sys = [   tf(1.90,[18.0,1])   tf(1.90,[18.0,1])   tf(1.90,[18.0,1]);
          tf(-0.74,[8.0,1])   tf(0.74,[8.0,1])    tf(-0.74,[8.0,1])   ]
linModel3 = LinModel(sys,Ts,i_d=[3])
linModel4 = LinModel(ss(A,[Bu Bd],C,[Du Dd],Ts),Ts,i_d=[3])
setop!(linModel4,uop=[10,10],yop=[50,30],dop=[5])
linModel5 = LinModel(ss(A,[Bu Bd],C,[Du Dd],Ts),Ts,i_u=1:2)
linModel6 = LinModel([delay(4) delay(8)]*sys,Ts,i_d=[3])

f(x,u,_) = A*x + Bu*u
h(x,_) = C*x
f2(x,u,d) = A*x + Bu*u + Bd*d
h2(x,_) = C*x

nonLinModel1 = NonLinModel(f,h,Ts,2,4,2)
nonLinModel2 = setop!(NonLinModel(f2,h2,Ts,2,4,2,1), uop=[10,10],yop=[50,30],dop=[5])

internalModel1 = InternalModel(linModel1)
internalModel2 = InternalModel(linModel1,stoch_ym=[tf([1,0],[1,-1],Ts) 0; 0 tf([1,0],[1,-1],Ts)])
internalModel3 = InternalModel(linModel1,i_ym=[2])

initstate!(internalModel1,[0,0],[1,1])

luenberger = Luenberger(linModel1)

updatestate!(luenberger, [0,0], [0,0])

mpcluen = LinMPC(luenberger)

kalmanFilter1 = KalmanFilter(linModel1)
kalmanFilter2 = KalmanFilter(linModel1,nint_ym=0)

updatestate!(kalmanFilter2,[1, 1],[1, 1])

initstate!(kalmanFilter1,[0,0],[2,1])

ssKalmanFilter1 = SteadyKalmanFilter(linModel1)
ssKalmanFilter2 = SteadyKalmanFilter(linModel1,nint_ym=0)


updatestate!(ssKalmanFilter2,[1, 1],[1,1])

initstate!(ssKalmanFilter1,[0,0],[2,1])

extKalmanFilter = ExtendedKalmanFilter(nonLinModel2)

initstate!(extKalmanFilter, [10,10], [50,30], [5]) 
updatestate!(extKalmanFilter, [10,11], [50,30], [5])



uscKalmanFilter1 = UnscentedKalmanFilter(linModel1)

updatestate!(uscKalmanFilter1,[0,0],[2,1])

initstate!(uscKalmanFilter1,[0,0],[2,1])

nmpc1 = NonLinMPC(uscKalmanFilter1)

nmpc2 = NonLinMPC(nonLinModel2, Hp=15, Hc=1, Mwt=[1, 1] , Nwt=[0.1, 0.1], Cwt=1e5)


setconstraint!(nmpc2, c_umin=[0,0], c_umax=[0,0])
setconstraint!(nmpc2, c_ŷmin=[1,1], c_ŷmax=[1,1])
setconstraint!(nmpc2, umin=[5, 9.9], umax=[Inf,Inf])
setconstraint!(nmpc2, ŷmin=[-Inf,-Inf], ŷmax=[55, 35])
setconstraint!(nmpc2, Δumin=[-Inf,-Inf],Δumax=[+Inf,+Inf])


nx = linModel4.nx
kf = KalmanFilter(linModel4, σP0=10*ones(nx), σQ=0.01*ones(nx), σR=[0.1, 0.1], σQ_int=0.05*ones(2), σP0_int=10*ones(2))

mpc = LinMPC(kf, Hp=15, Hc=1, Mwt=[1, 1] , Nwt=[0.1, 0.1], Cwt=1e5)#, optim=Model(DAQP.Optimizer))#, optim=Model(HiGHS.Optimizer))#, optim=Model(DAQP.Optimizer))

setconstraint!(mpc, c_umin=[0,0], c_umax=[0,0])
setconstraint!(mpc, c_ŷmin=[1,1], c_ŷmax=[1,1])
setconstraint!(mpc, umin=[5, 9.9], umax=[Inf,Inf])
setconstraint!(mpc, ŷmin=[-Inf,-Inf], ŷmax=[55, 35])
setconstraint!(mpc, Δumin=[-Inf,-Inf],Δumax=[+Inf,+Inf])

moveinput!(mpc, mpc.estim.model.yop + [0, 2], mpc.estim.model.dop)
display(getinfo(mpc)[2])

nmpc = NonLinMPC(kf, Hp=15, Hc=1, Mwt=[1, 1] , Nwt=[0.1, 0.1], Cwt=1e5, Ewt=0.0)

setconstraint!(nmpc, c_umin=[0,0], c_umax=[0,0])
setconstraint!(nmpc, c_ŷmin=[1,1], c_ŷmax=[1,1])
setconstraint!(nmpc, umin=[5, 9.9], umax=[Inf,Inf])
setconstraint!(nmpc, ŷmin=[-Inf,-Inf], ŷmax=[55, 35])
setconstraint!(nmpc, Δumin=[-Inf,-Inf],Δumax=[+Inf,+Inf])

moveinput!(nmpc, nmpc.estim.model.yop + [0, 2], nmpc.estim.model.dop)
display(getinfo(nmpc)[2])

function test_mpc(model, mpc)
    N = 200 
    u_data = zeros(2,N)
    y_data = zeros(2,N)
    r_data = zeros(2,N)
    d_data = zeros(1,N)
    u = model.uop
    d = model.dop
    r = [50,32]
    setstate!(model, zeros(model.nx))
    setstate!(mpc, zeros(mpc.estim.nx̂))
    initstate!(mpc,u,model(d),d)
    for k = 0:N-1
        if k == 40
            r[2] = 29
        end
        if k == 100
            r[2] = 36
        end
        if k == 150
            d = [3]
        end
        y = model(d)
        if k ≥ 180
            y[1] += 15
        end
        u = moveinput!(mpc, r, d)
        u_data[:,k+1] = u
        y_data[:,k+1] = y
        r_data[:,k+1] = r
        d_data[:,k+1] = d
        updatestate!(mpc, u, y, d)
        updatestate!(model, u, d)
    end
    return u_data, y_data, r_data, d_data
end

using PlotThemes, Plots
#theme(:default)
theme(:dark)
default(fontfamily="Computer Modern"); scalefontsizes(1.1)

test_mpc(linModel4 , mpc)
@time u_data, y_data, r_data, d_data = test_mpc(linModel4, mpc)

resM = sim!(nonLinModel2, mpc.Hp+10, [1,-1])
psM  = plot(resM)
display(psM)

res = sim!(mpc, mpc.Hp+10)
ps = plot(res, plotx=true)
display(ps)

res2 = sim!(uscKalmanFilter1, mpc.Hp+10)
ps2 = plot(res2, plotxwithx̂=true)
display(ps2)


test_mpc(linModel4, nmpc)
@time u_data, y_data, r_data, d_data = test_mpc(linModel4, nmpc)

test_mpc(nonLinModel2, nmpc2)
@time u_data, y_data, r_data, d_data = test_mpc(nonLinModel2, nmpc2)



N = size(r_data, 2) 
p1 = plot(0:N-1,y_data[1,:],label=raw"$y_1$")
plot!(0:N-1,r_data[1,:],label=raw"$r_1$",linestyle=:dash, linetype=:steppost)
p2 = plot(0:N-1,y_data[2,:],label=raw"$y_2$")
plot!(0:N-1,r_data[2,:],label=raw"$r_2$",linestyle=:dash, linetype=:steppost)
py = plot(p1,p2, layout=[1,1])

p1 = plot(0:N-1,u_data[1,:],label=raw"$u_1$",linetype=:steppost)
p2 = plot(0:N-1,u_data[2,:],label=raw"$u_2$",linetype=:steppost)
pu = plot(p1,p2, layout=[1,1])

pd = plot(0:N-1,d_data[1,:],label=raw"$d_1$")

display(pd)
display(pu)
display(py)

