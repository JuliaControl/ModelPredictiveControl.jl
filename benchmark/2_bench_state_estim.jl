## ----------------- Runtime benchmarks ---------------------------------------------
# TODO: Add runtime benchmarks for StateEstimator


## ----------------- Allocation benchmarks ------------------------------------------
samples, evals = 1, 1

skf = SteadyKalmanFilter(linmodel)
ALLOC["StateEstimator"]["SteadyKalmanFilter"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($skf, $y, $d),
        samples=samples, evals=evals
    )
ALLOC["StateEstimator"]["SteadyKalmanFilter"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($skf, $u, $y, $d), 
        setup=preparestate!($skf, $y, $d),
        samples=samples, evals=evals
    )
ALLOC["StateEstimator"]["SteadyKalmanFilter"]["evaloutput"] = 
    @benchmarkable(
        evaloutput($skf, $d),
        setup=preparestate!($skf, $y, $d),
        samples=samples, evals=evals
    )

kf = KalmanFilter(linmodel, nint_u=[1, 1], direct=false)
ALLOC["StateEstimator"]["KalmanFilter"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($kf, $y, $d),
        samples=samples, evals=evals
    )
ALLOC["StateEstimator"]["KalmanFilter"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($kf, $u, $y, $d),
        setup=preparestate!($kf, $y, $d),
        samples=samples, evals=evals
    )

lo = Luenberger(linmodel, nint_u=[1, 1])
ALLOC["StateEstimator"]["Luenberger"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($lo, $y, $d),
        samples=samples, evals=evals
    )
ALLOC["StateEstimator"]["Luenberger"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($lo, $u, $y, $d),
        setup=preparestate!($lo, $y, $d),
        samples=samples, evals=evals
    )

im = InternalModel(nonlinmodel)
ALLOC["StateEstimator"]["InternalModel"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($im, $y, $d),
        samples=samples, evals=evals
    )
ALLOC["StateEstimator"]["InternalModel"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($im, $u, $y, $d),
        setup=preparestate!($im, $y, $d),
        samples=samples, evals=evals
    )

ukf = UnscentedKalmanFilter(nonlinmodel)
ALLOC["StateEstimator"]["UnscentedKalmanFilter"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($ukf, $y, $d),
        samples=samples, evals=evals
    )
ALLOC["StateEstimator"]["UnscentedKalmanFilter"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($ukf, $u, $y,  $d),
        setup=preparestate!($ukf, $y, $d),
        samples=samples, evals=evals
    )
ALLOC["StateEstimator"]["UnscentedKalmanFilter"]["evaloutput"] = 
    @benchmarkable(
        evaloutput($ukf, $d),
        setup=preparestate!($ukf, $y, $d),
        samples=samples, evals=evals
    )

ekf = ExtendedKalmanFilter(linmodel, nint_u=[1, 1], direct=false)
ALLOC["StateEstimator"]["ExtendedKalmanFilter"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($ekf, $y, $d),
        samples=samples, evals=evals
    )
ALLOC["StateEstimator"]["ExtendedKalmanFilter"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($ekf, $u, $y, $d),
        setup=preparestate!($ekf, $y, $d),
        samples=samples, evals=evals
    )