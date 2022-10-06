using Pkg, Revise
Pkg.activate(".")

using ModelPredictiveControl
using DAQP
using OSQP
using JuMP, LinearAlgebra
using MAT

println(greet())

vars_ml = matread("example/matlab.mat");

A   = vars_ml["mMPC"]["A"];
Bu  = vars_ml["mMPC"]["B"];
Bd  = vars_ml["mMPC"]["Bd"];
C   = vars_ml["mMPC"]["C"];
Dd  = vars_ml["mMPC"]["Dd"];

linModel1 = LinModel(A,Bu,C)
linModel2 = LinModel(A,Bu,C,Bd);
linModel3 = LinModel(A,Bu,C,Bd,[]);
linModel4 = LinModel(A,Bu,C,Bd,Dd);
 
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