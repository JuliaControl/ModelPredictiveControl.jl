using BenchmarkTools
using ModelPredictiveControl, ControlSystemsBase, LinearAlgebra

Ts = 400.0
sys = [ tf(1.90,[1800.0,1])   tf(1.90,[1800.0,1])   tf(1.90,[1800.0,1]);
        tf(-0.74,[800.0,1])   tf(0.74,[800.0,1])    tf(-0.74,[800.0,1])   ] 

const SUITE = BenchmarkGroup()

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

SUITE["SimModel"]["allocation"] = BenchmarkGroup(["allocation"])
SUITE["SimModel"]["allocation"]["LinModel_updatestate!"] = @benchmarkable(
    updatestate!($linmodel, $u, $d),
    samples=1
)
SUITE["SimModel"]["allocation"]["LinModel_evaloutput"] = @benchmarkable(
    evaloutput($linmodel, $d),
    samples=1
)
SUITE["SimModel"]["allocation"]["NonLinModel_updatestate!"] = @benchmarkable(
    updatestate!($nonlinmodel, $u, $d),
    samples=1
)
SUITE["SimModel"]["allocation"]["NonLinModel_evaloutput"] = @benchmarkable(
    evaloutput($nonlinmodel, $d),
    samples=1
)

SUITE["SimModel"]["allocation"]["NonLinModel_linearize!"] = @benchmarkable(
    linearize!($linmodel, $nonlinmodel),
    samples=1
)

## ==================================================================================
## ================== StateEstimator benchmarks =====================================
## ==================================================================================
skf = SteadyKalmanFilter(linmodel)
SUITE["StateEstimator"]["allocation"] = BenchmarkGroup(["allocation"])
SUITE["StateEstimator"]["allocation"]["SteadyKalmanFilter_preparestate!"] = @benchmarkable(
    preparestate!($skf, $y, $d),
    samples=1
)
SUITE["StateEstimator"]["allocation"]["SteadyKalmanFilter_updatestate!"] = @benchmarkable(
    updatestate!($skf, $u, $y, $d),
    setup=preparestate!($skf, $y, $d),
    samples=1
)
SUITE["StateEstimator"]["allocation"]["SteadyKalmanFilter_evaloutput"] = @benchmarkable(
    evaloutput($skf, $d),
    setup=preparestate!($skf, $y, $d),
    samples=1
)

kf = KalmanFilter(linmodel, nint_u=[1, 1], direct=false)
SUITE["StateEstimator"]["allocation"]["KalmanFilter_preparestate!"] = @benchmarkable(
    preparestate!($kf, $y, $d),
    samples=1
)
SUITE["StateEstimator"]["allocation"]["KalmanFilter_updatestate!"] = @benchmarkable(
    updatestate!($kf, $u, $y, $d),
    setup=preparestate!($kf, $y, $d),
    samples=1
)

lo = Luenberger(linmodel, nint_u=[1, 1])
SUITE["StateEstimator"]["allocation"]["Luenberger_preparestate!"] = @benchmarkable(
    preparestate!($lo, $y, $d),
    samples=1
)
SUITE["StateEstimator"]["allocation"]["Luenberger_updatestate!"] = @benchmarkable(
    updatestate!($lo, $u, $y, $d),
    setup=preparestate!($lo, $y, $d),
    samples=1
)

im = InternalModel(nonlinmodel)
SUITE["StateEstimator"]["allocation"]["InternalModel_preparestate!"] = @benchmarkable(
    preparestate!($im, $y, $d),
    samples=1
)
SUITE["StateEstimator"]["allocation"]["InternalModel_updatestate!"] = @benchmarkable(
    updatestate!($im, $u, $y, $d),
    setup=preparestate!($im, $y, $d),
    samples=1
)

ukf = UnscentedKalmanFilter(nonlinmodel)
SUITE["StateEstimator"]["allocation"]["UnscentedKalmanFilter_preparestate!"] = @benchmarkable(
    preparestate!($ukf, $y, $d),
    samples=1
)
SUITE["StateEstimator"]["allocation"]["UnscentedKalmanFilter_updatestate!"] = @benchmarkable(
    updatestate!($ukf, $u, $y,  $d),
    setup=preparestate!($ukf, $y, $d),
    samples=1
)
SUITE["StateEstimator"]["allocation"]["UnscentedKalmanFilter_evaloutput"] = @benchmarkable(
    evaloutput($ukf, $d),
    setup=preparestate!($ukf, $y, $d),
    samples=1
)

ekf = ExtendedKalmanFilter(linmodel, nint_u=[1, 1], direct=false)
SUITE["StateEstimator"]["allocation"]["ExtendedKalmanFilter_preparestate!"] = @benchmarkable(
    preparestate!($ekf, $y, $d),
    samples=1
)
SUITE["StateEstimator"]["allocation"]["ExtendedKalmanFilter_updatestate!"] = @benchmarkable(
    updatestate!($ekf, $u, $y, $d),
    setup=preparestate!($ekf, $y, $d),
    samples=1
)