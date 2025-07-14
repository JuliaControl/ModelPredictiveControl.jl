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