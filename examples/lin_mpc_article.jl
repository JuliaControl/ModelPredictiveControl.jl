# ==========================================
# ========== GLOBAL SETTINGS ===============
# ==========================================
run_benchmarks = true

# ==========================================
# ========== FEEDBACK ======================
# ==========================================

using ModelPredictiveControl, ControlSystemsBase
G = [ tf(1.90, [18, 1]) tf(1.90, [18, 1]);
      tf(-0.74,[8, 1])  tf(0.74, [8, 1]) ]
uop, yop = [20, 20], [50, 30]
vu , vy  = ["\$u_c\$", "\$u_h\$"], ["\$y_L\$", "\$y_T\$"]
model = setop!(LinModel(G, 2.0); uop, yop)
model = setname!(model; u=vu, y=vy)

## =========================================
mpc = setconstraint!(LinMPC(model), ymin=[45, -Inf])

## =========================================
function test_mpc(mpc, plant)
    plant.x0 .= 0; y = plant() # or evaloutput(plant)
    initstate!(mpc, plant.uop, y)
    N = 75; ry = [50, 30]; ul = 0
    U, Y, Ry = zeros(2, N), zeros(2, N), zeros(2, N)
    for i = 1:N
        i == 26 && (ry = [48, 35])
        i == 51 && (ul = -10)
        y = plant() # simulated measurements
        preparestate!(mpc, y) # prepare mpc estimate
        u = mpc(ry) # or moveinput!(mpc, ry)
        U[:,i], Y[:,i], Ry[:,i] = u, y, ry
        updatestate!(mpc, u, y) # update mpc estimate
        updatestate!(plant, u+[0,ul]) # update simulator
    end
    return U, Y, Ry
end
U_data, Y_data, Ry_data = test_mpc(mpc, model)

## =========================================
res = SimResult(mpc, U_data, Y_data; Ry_data)
using Plots; plot(res)

## =========================================
## ========= Benchmark =====================
## =========================================
using BenchmarkTools
using JuMP, OSQP, DAQP

if run_benchmarks
    optim = JuMP.Model(OSQP.Optimizer, add_bridges=false)
    mpc_osqp = setconstraint!(LinMPC(model; optim), ymin=[45, -Inf])
    JuMP.unset_time_limit_sec(mpc_osqp.optim)
    bm = @benchmark test_mpc($mpc_osqp, $model) samples=500
    @show btime_solver_OS = median(bm)
    
    optim = JuMP.Model(DAQP.Optimizer, add_bridges=false)
    mpc_daqp = setconstraint!(LinMPC(model; optim), ymin=[45, -Inf])
    bm = @benchmark test_mpc($mpc_daqp, $model) samples=500
    @show btime_solver_AS = median(bm)
end

## =========================================
## ========= Feedforward ===================
## =========================================
model_d = LinModel([G G[1:2, 2]], 2.0; i_d=[3])
model_d = setop!(model_d; uop, yop, dop=[20])
model_d = setname!(model_d; u=vu, y=vy, d=["\$u_l\$"])

## =========================================
mpc_d = setconstraint!(LinMPC(model_d), ymin=[45, -Inf])
function test_mpc_d(mpc_d, plant)
    plant.x0 .= 0; y = plant(); d = [20]
    initstate!(mpc_d, plant.uop, y, d)
    N = 75; ry = [50, 30]; ul = 0
    U, Y, Ry = zeros(2, N), zeros(2, N), zeros(2, N)
    for i = 1:N
        i == 26 && (ry = [48, 35])
        i == 51 && (ul = -10)
        y, d = plant(), [20+ul] # simulated measurements
        preparestate!(mpc_d, y, d) # d in arguments
        u = mpc_d(ry, d) # d in arguments
        U[:,i], Y[:,i], Ry[:,i] = u, y, ry
        updatestate!(mpc_d, u, y, d) # d in arguments
        updatestate!(plant, u+[0,ul])
    end
    return U, Y, Ry
end
U_data, Y_data, Ry_data = test_mpc_d(mpc_d, model)
res = SimResult(mpc, U_data, Y_data; Ry_data)
plot(res)

## =========================================
## ========= Benchmark =====================
## =========================================
using BenchmarkTools
using JuMP, OSQP, DAQP

if run_benchmarks
    optim = JuMP.Model(OSQP.Optimizer, add_bridges=false)
    mpc_d_osqp = setconstraint!(LinMPC(model_d; optim), ymin=[45, -Inf])
    JuMP.unset_time_limit_sec(mpc_d_osqp.optim)
    bm = @benchmark test_mpc_d($mpc_d_osqp, $model) samples=500
    @show btime_solver_OS = median(bm)

    optim = JuMP.Model(DAQP.Optimizer, add_bridges=false)
    mpc_d_daqp = setconstraint!(LinMPC(model_d; optim), ymin=[45, -Inf])
    bm = @benchmark test_mpc_d($mpc_d_daqp, $model) samples=500
    @show btime_solver_AS = median(bm)
end