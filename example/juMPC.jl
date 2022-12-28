# spell-checker: disable

using Pkg
using Revise
Pkg.activate(".")

using ModelPredictiveControl
#using DAQP
#using OSQP
#using JuMP 
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
setop!(linModel4,uop=[10,10],yop=[50,30],dop=[15])
linModel5 = LinModel(ss(A,[Bu Bd],C,[Du Dd],Ts),Ts,i_u=1:2)
linModel6 = LinModel([delay(4) delay(8)]*sys,Ts,i_d=[3])

f(x,u,_) = A*x + Bu*u
h(x,_) = C*x
f2(x,u,d) = A*x + Bu*u + Bd*d
h2(x,_) = C*x

nonLinModel1 = NonLinModel(f,h,Ts,2,4,2)
nonLinModel2 = NonLinModel(f2,h2,Ts,2,4,2,1)

internalModel1 = InternalModel(linModel1)
internalModel2 = InternalModel(linModel1,stoch_ym=[tf([1,0],[1,-1],Ts) 0; 0 tf([1,0],[1,-1],Ts)])
internalModel3 = InternalModel(linModel1,i_ym=[1])

initstate!(internalModel1,[0,0],[1,1])

kalmanFilter1 = KalmanFilter(linModel1)
kalmanFilter2 = KalmanFilter(linModel1,nint_ym=0)

updatestate!(kalmanFilter2,[1, 1],[1, 1])

initstate!(kalmanFilter1,[0,0],[2,1])

ssKalmanFilter1 = SteadyKalmanFilter(linModel1)
ssKalmanFilter2 = SteadyKalmanFilter(linModel1,nint_ym=0)


updatestate!(ssKalmanFilter2,[1, 1],[1,1])

initstate!(ssKalmanFilter1,[0,0],[2,1])

mpc = LinMPC(linModel4, Hp=15, Hc=1, Nwt=[0.1, 0.1], Cwt=1e6)

#setconstraint!(mpc, c_ŷmin=[1,1], c_ŷmax=[1,1])
#setconstraint!(mpc, umin=[5, 9.9])
#setconstraint!(mpc, ŷmax=[55, 35])


u = moveinput!(mpc, [50,31], mpc.model.dop)

updatestate!(mpc, u, [50,30], mpc.model.dop)

