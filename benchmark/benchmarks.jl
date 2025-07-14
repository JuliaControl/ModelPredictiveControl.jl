using BenchmarkTools
using ModelPredictiveControl, ControlSystemsBase, LinearAlgebra
using JuMP, OSQP, DAQP, Ipopt, MadNLP

Ts = 400.0
sys = [ tf(1.90,[1800.0,1])   tf(1.90,[1800.0,1])   tf(1.90,[1800.0,1]);
        tf(-0.74,[800.0,1])   tf(0.74,[800.0,1])    tf(-0.74,[800.0,1])   ] 

const SUITE = BenchmarkGroup(["ModelPredictiveControl"])

## ==================================================================================
## ================== SimModel benchmarks ===========================================
## ==================================================================================
linmodel = setop!(LinModel(sys, Ts, i_d=[3]), uop=[10, 50], yop=[50, 30], dop=[5])
function f!(ẋ, x, u, d, p)
    mul!(ẋ, p.A, x)
    mul!(ẋ, p.Bu, u, 1, 1)
    mul!(ẋ, p.Bd, d, 1, 1)
    return nothing
end
function h!(y, x, d, p)
    mul!(y, p.C, x)
    mul!(y, p.Dd, d, 1, 1)
    return nothing
end
nonlinmodel = NonLinModel(f!, h!, Ts, 2, 4, 2, 1, p=linmodel, solver=nothing)
nonlinmodel = setop!(nonlinmodel, uop=[10, 50], yop=[50, 30], dop=[5])
u, d, y = [10, 50], [5], [50, 30]

## ----------------- Runtime benchmarks ---------------------------------------------
# TODO: Add runtime benchmarks for SimModel


## ----------------- Allocation benchmarks ------------------------------------------
samples, evals = 1, 1
SUITE["allocation"]["SimModel"]["LinModel"]["updatestate!"] = @benchmarkable(
    updatestate!($linmodel, $u, $d); samples=samples, evals=evals
)
SUITE["allocation"]["SimModel"]["LinModel"]["evaloutput"] = @benchmarkable(
    evaloutput($linmodel, $d); samples=samples, evals=evals
)
SUITE["allocation"]["SimModel"]["NonLinModel"]["updatestate!"] = @benchmarkable(
    updatestate!($nonlinmodel, $u, $d); samples=samples, evals=evals
)
SUITE["allocation"]["SimModel"]["NonLinModel"]["evaloutput"] = @benchmarkable(
    evaloutput($nonlinmodel, $d); samples=samples, evals=evals
)
SUITE["allocation"]["SimModel"]["NonLinModel"]["linearize!"] = @benchmarkable(
    linearize!($linmodel, $nonlinmodel); samples=samples, evals=evals
)

## ==================================================================================
## ================== StateEstimator benchmarks =====================================
## ==================================================================================


## ----------------- Runtime benchmarks ---------------------------------------------
# TODO: Add runtime benchmarks for StateEstimator


## ----------------- Allocation benchmarks ------------------------------------------
samples, evals = 1, 1

skf = SteadyKalmanFilter(linmodel)
SUITE["allocation"]["StateEstimator"]["SteadyKalmanFilter"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($skf, $y, $d),
        samples=samples, evals=evals
    )
SUITE["allocation"]["StateEstimator"]["SteadyKalmanFilter"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($skf, $u, $y, $d), 
        setup=preparestate!($skf, $y, $d),
        samples=samples, evals=evals
    )
SUITE["allocation"]["StateEstimator"]["SteadyKalmanFilter"]["evaloutput"] = 
    @benchmarkable(
        evaloutput($skf, $d),
        setup=preparestate!($skf, $y, $d),
        samples=samples, evals=evals
    )

kf = KalmanFilter(linmodel, nint_u=[1, 1], direct=false)
SUITE["allocation"]["StateEstimator"]["KalmanFilter"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($kf, $y, $d),
        samples=samples, evals=evals
    )
SUITE["allocation"]["StateEstimator"]["KalmanFilter"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($kf, $u, $y, $d),
        setup=preparestate!($kf, $y, $d),
        samples=samples, evals=evals
    )

lo = Luenberger(linmodel, nint_u=[1, 1])
SUITE["allocation"]["StateEstimator"]["Luenberger"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($lo, $y, $d),
        samples=samples, evals=evals
    )
SUITE["allocation"]["StateEstimator"]["Luenberger"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($lo, $u, $y, $d),
        setup=preparestate!($lo, $y, $d),
        samples=samples, evals=evals
    )

im = InternalModel(nonlinmodel)
SUITE["allocation"]["StateEstimator"]["InternalModel"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($im, $y, $d),
        samples=samples, evals=evals
    )
SUITE["allocation"]["StateEstimator"]["InternalModel"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($im, $u, $y, $d),
        setup=preparestate!($im, $y, $d),
        samples=samples, evals=evals
    )

ukf = UnscentedKalmanFilter(nonlinmodel)
SUITE["allocation"]["StateEstimator"]["UnscentedKalmanFilter"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($ukf, $y, $d),
        samples=samples, evals=evals
    )
SUITE["allocation"]["StateEstimator"]["UnscentedKalmanFilter"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($ukf, $u, $y,  $d),
        setup=preparestate!($ukf, $y, $d),
        samples=samples, evals=evals
    )
SUITE["allocation"]["StateEstimator"]["UnscentedKalmanFilter"]["evaloutput"] = 
    @benchmarkable(
        evaloutput($ukf, $d),
        setup=preparestate!($ukf, $y, $d),
        samples=samples, evals=evals
    )

ekf = ExtendedKalmanFilter(linmodel, nint_u=[1, 1], direct=false)
SUITE["allocation"]["StateEstimator"]["ExtendedKalmanFilter"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($ekf, $y, $d),
        samples=samples, evals=evals
    )
SUITE["allocation"]["StateEstimator"]["ExtendedKalmanFilter"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($ekf, $u, $y, $d),
        setup=preparestate!($ekf, $y, $d),
        samples=samples, evals=evals
    )

## ==================================================================================
## ================== PredictiveController benchmarks ===============================
## ==================================================================================
G = [ tf(1.90, [18, 1]) tf(1.90, [18, 1]);
      tf(-0.74,[8, 1])  tf(0.74, [8, 1]) ]
uop, yop = [20, 20], [50, 30]
model = setop!(LinModel(G, 2.0); uop, yop)
function test_mpc(mpc, plant)
    plant.x0 .= 0; y = plant() 
    initstate!(mpc, plant.uop, y)
    N = 75; ry = [50, 30]; ul = 0
    U, Y, Ry = zeros(2, N), zeros(2, N), zeros(2, N)
    for i = 1:N
        i == 26 && (ry = [48, 35])
        i == 51 && (ul = -10)
        y = plant() 
        preparestate!(mpc, y) 
        u = mpc(ry)
        U[:,i], Y[:,i], Ry[:,i] = u, y, ry
        updatestate!(mpc, u, y) 
        updatestate!(plant, u+[0,ul])
    end
    return U, Y, Ry
end

## ----------------- Runtime benchmarks ---------------------------------------------
optim = JuMP.Model(OSQP.Optimizer, add_bridges=false)
transcription = SingleShooting()
mpc_osqp_ss = setconstraint!(LinMPC(model; optim, transcription), ymin=[45, -Inf])
JuMP.unset_time_limit_sec(mpc_osqp_ss.optim)

optim = JuMP.Model(OSQP.Optimizer, add_bridges=false)
transcription = MultipleShooting()
mpc_osqp_ms = setconstraint!(LinMPC(model; optim, transcription), ymin=[45, -Inf])
JuMP.unset_time_limit_sec(mpc_osqp_ms.optim)

optim = JuMP.Model(DAQP.Optimizer, add_bridges=false)
transcription = SingleShooting()
mpc_daqp_ss = setconstraint!(LinMPC(model; optim, transcription), ymin=[45, -Inf])

optim = JuMP.Model(DAQP.Optimizer, add_bridges=false)
transcription = MultipleShooting()
mpc_daqp_ms = setconstraint!(LinMPC(model; optim, transcription), ymin=[45, -Inf])
# needed to solve Hessians with eigenvalues at zero, like in MultipleShooting transcription:
JuMP.set_attribute(mpc_daqp_ms.optim, "eps_prox", 1e-6) 

samples, evals = 500, 1
SUITE["runtime"]["PredictiveController"]["CSTR"]["LinMPC"]["OSQP"]["SingleShooting"] = 
    @benchmarkable(test_mpc($mpc_osqp_ss, $model); 
    samples=samples, evals=evals
)
SUITE["runtime"]["PredictiveController"]["CSTR"]["LinMPC"]["OSQP"]["MultipleShooting"] = 
    @benchmarkable(test_mpc($mpc_osqp_ms, $model); 
    samples=samples, evals=evals
)
SUITE["runtime"]["PredictiveController"]["CSTR"]["LinMPC"]["DAQP"]["SingleShooting"] =
    @benchmarkable(test_mpc($mpc_daqp_ss, $model); 
    samples=samples, evals=evals
)
SUITE["runtime"]["PredictiveController"]["CSTR"]["LinMPC"]["DAQP"]["MultipleShooting"] =
    @benchmarkable(test_mpc($mpc_daqp_ms, $model); 
    samples=samples, evals=evals
)


# ---------------------- Allocation benchmarks ------------------------------------------
empc = ExplicitMPC(linmodel, Mwt=[1, 1], Nwt=[0.1, 0.1], Lwt=[0.1, 0.1])

samples, evals = 1, 1
SUITE["allocation"]["PredictiveController"]["ExplicitMPC"]["moveinput!"] = 
    @benchmarkable(
        moveinput!($empc, $y, $d),
        setup=preparestate!($empc, $y, $d),
        samples=samples, evals=evals
    )
