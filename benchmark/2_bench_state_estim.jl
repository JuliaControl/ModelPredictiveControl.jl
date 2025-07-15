## ----------------- Unit tests (no allocation)  -----------------------------------------
const UNIT_ESTIM = SUITE["unit tests"]["StateEstimator"]

samples, evals = 10000, 1
skf = SteadyKalmanFilter(linmodel)
UNIT_ESTIM["SteadyKalmanFilter"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($skf, $y, $d),
        samples=samples, evals=evals
    )
UNIT_ESTIM["SteadyKalmanFilter"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($skf, $u, $y, $d), 
        setup=preparestate!($skf, $y, $d),
        samples=samples, evals=evals
    )
UNIT_ESTIM["SteadyKalmanFilter"]["evaloutput"] = 
    @benchmarkable(
        evaloutput($skf, $d),
        setup=preparestate!($skf, $y, $d),
        samples=samples, evals=evals
    )

kf = KalmanFilter(linmodel, nint_u=[1, 1], direct=false)
UNIT_ESTIM["KalmanFilter"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($kf, $y, $d),
        samples=samples, evals=evals
    )
UNIT_ESTIM["KalmanFilter"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($kf, $u, $y, $d),
        setup=preparestate!($kf, $y, $d),
        samples=samples, evals=evals
    )

lo = Luenberger(linmodel, nint_u=[1, 1])
UNIT_ESTIM["Luenberger"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($lo, $y, $d),
        samples=samples, evals=evals
    )
UNIT_ESTIM["Luenberger"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($lo, $u, $y, $d),
        setup=preparestate!($lo, $y, $d),
        samples=samples, evals=evals
    )

im = InternalModel(nonlinmodel)
UNIT_ESTIM["InternalModel"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($im, $y, $d),
        samples=samples, evals=evals
    )
UNIT_ESTIM["InternalModel"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($im, $u, $y, $d),
        setup=preparestate!($im, $y, $d),
        samples=samples, evals=evals
    )

ukf = UnscentedKalmanFilter(nonlinmodel)
UNIT_ESTIM["UnscentedKalmanFilter"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($ukf, $y, $d),
        samples=samples, evals=evals
    )
UNIT_ESTIM["UnscentedKalmanFilter"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($ukf, $u, $y,  $d),
        setup=preparestate!($ukf, $y, $d),
        samples=samples, evals=evals
    )
UNIT_ESTIM["UnscentedKalmanFilter"]["evaloutput"] = 
    @benchmarkable(
        evaloutput($ukf, $d),
        setup=preparestate!($ukf, $y, $d),
        samples=samples, evals=evals
    )

ekf = ExtendedKalmanFilter(linmodel, nint_u=[1, 1], direct=false)
UNIT_ESTIM["ExtendedKalmanFilter"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($ekf, $y, $d),
        samples=samples, evals=evals
    )
UNIT_ESTIM["ExtendedKalmanFilter"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($ekf, $u, $y, $d),
        setup=preparestate!($ekf, $y, $d),
        samples=samples, evals=evals
    )

## ----------------- Case studies ---------------------------------------------------
# TODO: Add case study benchmarks for StateEstimator


