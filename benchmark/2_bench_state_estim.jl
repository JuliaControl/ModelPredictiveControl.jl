## ----------------- Unit tests -----------------------------------------------------------
const UNIT_ESTIM = SUITE["unit tests"]["StateEstimator"]

skf = SteadyKalmanFilter(linmodel)
UNIT_ESTIM["SteadyKalmanFilter"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($skf, $y, $d),
    )
UNIT_ESTIM["SteadyKalmanFilter"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($skf, $u, $y, $d), 
        setup=preparestate!($skf, $y, $d),
    )
UNIT_ESTIM["SteadyKalmanFilter"]["evaloutput"] = 
    @benchmarkable(
        evaloutput($skf, $d),
        setup=preparestate!($skf, $y, $d),
    )

kf = KalmanFilter(linmodel, nint_u=[1, 1], direct=false)
UNIT_ESTIM["KalmanFilter"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($kf, $y, $d),
    )
UNIT_ESTIM["KalmanFilter"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($kf, $u, $y, $d),
        setup=preparestate!($kf, $y, $d),
    )
UNIT_ESTIM["KalmanFilter"]["evaloutput"] =
    @benchmarkable(
        evaloutput($kf, $d),
        setup=preparestate!($kf, $y, $d),
    )

lo = Luenberger(linmodel, nint_u=[1, 1])
UNIT_ESTIM["Luenberger"]["preparestate!"] = 
    @benchmarkable(
        preparestate!($lo, $y, $d),
    )
UNIT_ESTIM["Luenberger"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($lo, $u, $y, $d),
        setup=preparestate!($lo, $y, $d),
    )
UNIT_ESTIM["Luenberger"]["evaloutput"] =
    @benchmarkable(
        evaloutput($lo, $d),
        setup=preparestate!($lo, $y, $d),
    )

im_lin    = InternalModel(linmodel)
im_nonlin = InternalModel(nonlinmodel)
UNIT_ESTIM["InternalModel"]["preparestate!"]["LinModel"] = 
    @benchmarkable(
        preparestate!($im_lin, $y, $d),
    )
UNIT_ESTIM["InternalModel"]["updatestate!"]["LinModel"] = 
    @benchmarkable(
        updatestate!($im_lin, $u, $y, $d),
        setup=preparestate!($im_lin, $y, $d),
    )
UNIT_ESTIM["InternalModel"]["evaloutput"]["LinModel"] =
    @benchmarkable(
        evaloutput($im_lin, $d),
        setup=preparestate!($im_lin, $y, $d),
    )
UNIT_ESTIM["InternalModel"]["preparestate!"]["NonLinModel"] = 
    @benchmarkable(
        preparestate!($im_nonlin, $y, $d),
    )
UNIT_ESTIM["InternalModel"]["updatestate!"]["NonLinModel"] = 
    @benchmarkable(
        updatestate!($im_nonlin, $u, $y, $d),
        setup=preparestate!($im_nonlin, $y, $d),
    )
UNIT_ESTIM["InternalModel"]["evaloutput"]["NonLinModel"] =
    @benchmarkable(
        evaloutput($im_nonlin, $d),
        setup=preparestate!($im_nonlin, $y, $d),
    )

ukf_lin    = UnscentedKalmanFilter(linmodel)
ukf_nonlin = UnscentedKalmanFilter(nonlinmodel)
UNIT_ESTIM["UnscentedKalmanFilter"]["preparestate!"]["LinModel"] =
    @benchmarkable(
        preparestate!($ukf_lin, $y, $d),
    )
UNIT_ESTIM["UnscentedKalmanFilter"]["updatestate!"]["LinModel"] =
    @benchmarkable(
        updatestate!($ukf_lin, $u, $y,  $d),
        setup=preparestate!($ukf_lin, $y, $d),
    )
UNIT_ESTIM["UnscentedKalmanFilter"]["evaloutput"]["LinModel"] =
    @benchmarkable(
        evaloutput($ukf_lin, $d),
        setup=preparestate!($ukf_lin, $y, $d),
    )
UNIT_ESTIM["UnscentedKalmanFilter"]["preparestate!"]["NonLinModel"] =
    @benchmarkable(
        preparestate!($ukf_nonlin, $y, $d),
    )
UNIT_ESTIM["UnscentedKalmanFilter"]["updatestate!"]["NonLinModel"] =
    @benchmarkable(
        updatestate!($ukf_nonlin, $u, $y,  $d),
        setup=preparestate!($ukf_nonlin, $y, $d),
    )
UNIT_ESTIM["UnscentedKalmanFilter"]["evaloutput"]["NonLinModel"] =
    @benchmarkable(
        evaloutput($ukf_nonlin, $d),
        setup=preparestate!($ukf_nonlin, $y, $d),
    )

ekf_lin    = ExtendedKalmanFilter(linmodel, nint_u=[1, 1], direct=false)
ekf_nonlin = ExtendedKalmanFilter(nonlinmodel, nint_u=[1, 1], direct=false)
UNIT_ESTIM["ExtendedKalmanFilter"]["preparestate!"]["LinModel"] =
    @benchmarkable(
        preparestate!($ekf_lin, $y, $d),
    )
UNIT_ESTIM["ExtendedKalmanFilter"]["updatestate!"]["LinModel"] = 
    @benchmarkable(
        updatestate!($ekf_lin, $u, $y, $d),
        setup=preparestate!($ekf_lin, $y, $d),
    )
UNIT_ESTIM["ExtendedKalmanFilter"]["evaloutput"]["LinModel"] =
    @benchmarkable(
        evaloutput($ekf_lin, $d),
        setup=preparestate!($ekf_lin, $y, $d),
    )
UNIT_ESTIM["ExtendedKalmanFilter"]["preparestate!"]["NonLinModel"] =
    @benchmarkable(
        preparestate!($ekf_nonlin, $y, $d),
    )
UNIT_ESTIM["ExtendedKalmanFilter"]["updatestate!"]["NonLinModel"] = 
    @benchmarkable( 
        updatestate!($ekf_nonlin, $u, $y, $d),
        setup=preparestate!($ekf_nonlin, $y, $d),
    )
UNIT_ESTIM["ExtendedKalmanFilter"]["evaloutput"]["NonLinModel"] =
    @benchmarkable(
        evaloutput($ekf_nonlin, $d),
        setup=preparestate!($ekf_nonlin, $y, $d),
    )

mhe_lin_direct = MovingHorizonEstimator(linmodel, He=10, direct=true)
mhe_lin_nondirect = MovingHorizonEstimator(linmodel, He=10, direct=false)
mhe_nonlin_direct = MovingHorizonEstimator(nonlinmodel, He=10, direct=true)
mhe_nonlin_nondirect = MovingHorizonEstimator(nonlinmodel, He=10, direct=false)


## ----------------- Case studies ---------------------------------------------------
# TODO: Add case study benchmarks for StateEstimator


