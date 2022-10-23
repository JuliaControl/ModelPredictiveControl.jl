using Pkg
using Revise
Pkg.activate(".")

using ModelPredictiveControl
using DAQP
using OSQP
using JuMP, LinearAlgebra
using ControlSystemsBase
using MAT


println(greet())

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
G = [tf(1.90,[18.0,1]) tf(1.90,[18.0,1]) tf(1.90,[18.0,1]);
    tf(-0.74,[8.0,1]) tf(0.74,[8.0,1]) tf(-0.74,[8.0,1])]
linModel3 = LinModel(G,Ts,i_d=[3])
linModel4 = LinModel(
    ss(A,[Bu Bd],C,[Du Dd],Ts),Ts,i_d=[3],
    u_op=[10,50],
    d_op=[5],
    y_op=[50,30])


f(x::Vector{ComplexF64},u::Vector{Float64}) = A*x + B*u
h(x::Vector{ComplexF64}) = C*x


nonLinModel1 = NonLinModel(f,h,Ts,2,4,2)


#TODO: trouver un moyen d'utiliser hasmethod avec simulfunc pour s'assurer que
# les arguments d'entrées sont O.K.

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