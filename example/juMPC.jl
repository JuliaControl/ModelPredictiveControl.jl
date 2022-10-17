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
Dd  = vars_ml["mMPC"]["Dd"]
Ts  = vars_ml["mMPC"]["Ts"]

linModel1 = LinModel(ss(A,Bu,C,0,Ts),Ts)
linModel2 = LinModel(ss(A,[Bu Bd],C,0,Ts),Ts,i_d=[3])
G = [tf(1.90,[18.0,1]) tf(1.90,[18.0,1]) tf(1.90,[18.0,1]);
    tf(-0.74,[8.0,1]) tf(0.74,[8.0,1]) tf(-0.74,[8.0,1])]
linModel3 = LinModel(G,Ts,i_d=[3])
linModel4 = LinModel(
    ss(A,[Bu Bd],C,0,Ts),Ts,i_d=[3],
    u_op=[10,50],
    d_op=[5],
    y_op=[50,30])
function MaSimulFunc(x,u_mat)
    Nx = size(u_mat,2) + 1
    x_mat = Matrix{Float64}(undef,4,Nx)
    y_mat = Matrix{Float64}(undef,2,Nx)
    x_mat[:,1] = x;
    for i in 1:Nx-1
        y_mat[:,i] = C*x_mat[:,i]
        x_mat[:,i+1] =  A*x_mat[:,i]+ Bu*u_mat[:,i]
    end
    y_mat[:,Nx] = C*x_mat[:,Nx]
    return (y_mat,x_mat)
end

nonLinModel1 = NonLinModel(Ts,4,2,2,0,MaSimulFunc)




#=
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