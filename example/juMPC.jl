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
setop!(linModel4,uop=[10,50],yop=[50,30],dop=[15])
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

kalmanFilter1 = KalmanFilter(linModel1)
kalmanFilter1 = KalmanFilter(linModel1,nint_ym=0)

updatestate!(kalmanFilter1,[1, 1],[1, 1])

ssKalmanFilter1 = SteadyKalmanFilter(linModel1)
ssKalmanFilter1 = SteadyKalmanFilter(linModel1,nint_ym=0)

updatestate!(ssKalmanFilter1,[1, 1],[1, 1])

#=([
H_qp = vars_ml["mMPC"]["Hqp"]
f_qp = vec(vars_ml["fqp"])
 
A = vars_ml["A"]
b = vec(vars_ml["b"])

ΔUϵ_min = vec(vars_ml["mMPC"]["DU_min"])
ΔUϵ_max = vec(vars_ml["mMPC"]["DU_max"])

Hc = Int(vars_ml["mMPC"]["Hc"])
nu = Int(vars_ml["mMPC"]["nu"])
n_var = (Hc*nu)+1



model = Model(OSQP.Optimizer)

@variable(model, ΔUϵ_min[i] .<= ΔUϵ[i=1:n_var] .<= ΔUϵ_max[i])

@objective(model, Min, 0.5*(ΔUϵ)'H_qp*(ΔUϵ) + f_qp'*(ΔUϵ))
@constraint(model, A*ΔUϵ .<= b)
optimize!(model)

julia_ΔUϵ = value.(ΔUϵ)

ml_ΔUϵ = vec(vars_ml["deltaUhc"])
julia_ΔUϵ - ml_ΔUϵ 
=#