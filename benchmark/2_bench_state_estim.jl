## ----------------------------------------------------------------------------------------
## ----------------- UNIT TESTS -----------------------------------------------------------
## ----------------------------------------------------------------------------------------
const UNIT_ESTIM = SUITE["UNIT TESTS"]["StateEstimator"]

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

mhe_lin_curr    = MovingHorizonEstimator(linmodel, He=10, direct=true)
mhe_lin_pred    = MovingHorizonEstimator(linmodel, He=10, direct=false)
mhe_nonlin_curr = MovingHorizonEstimator(nonlinmodel, He=10, direct=true)
mhe_nonlin_pred = MovingHorizonEstimator(nonlinmodel, He=10, direct=false)

samples, evals, seconds = 10000, 1, 60
UNIT_ESTIM["MovingHorizonEstimator"]["preparestate!"]["LinModel"]["Current form"] =
    @benchmarkable(
        preparestate!($mhe_lin_curr, $y, $d),
        samples=samples, evals=evals, seconds=seconds,
    )
UNIT_ESTIM["MovingHorizonEstimator"]["updatestate!"]["LinModel"]["Current form"] = 
    @benchmarkable(
        updatestate!($mhe_lin_curr, $u, $y, $d),
        setup=preparestate!($mhe_lin_curr, $y, $d),
        samples=samples, evals=evals, seconds=seconds,
    )
UNIT_ESTIM["MovingHorizonEstimator"]["preparestate!"]["LinModel"]["Prediction form"] =
    @benchmarkable(
        preparestate!($mhe_lin_pred, $y, $d),
        samples=samples, evals=evals, seconds=seconds,
    )
UNIT_ESTIM["MovingHorizonEstimator"]["updatestate!"]["LinModel"]["Prediction form"] =
    @benchmarkable(
        updatestate!($mhe_lin_pred, $u, $y, $d),
        setup=preparestate!($mhe_lin_pred, $y, $d),
        samples=samples, evals=evals, seconds=seconds,
    )
UNIT_ESTIM["MovingHorizonEstimator"]["preparestate!"]["NonLinModel"]["Current form"] =
    @benchmarkable(
        preparestate!($mhe_nonlin_curr, $y, $d),
        samples=samples, evals=evals, seconds=seconds,
    )
UNIT_ESTIM["MovingHorizonEstimator"]["updatestate!"]["NonLinModel"]["Current form"] = 
    @benchmarkable(
        updatestate!($mhe_nonlin_curr, $u, $y, $d),
        setup=preparestate!($mhe_nonlin_curr, $y, $d),
        samples=samples, evals=evals, seconds=seconds,
    )
UNIT_ESTIM["MovingHorizonEstimator"]["preparestate!"]["NonLinModel"]["Prediction form"] =
    @benchmarkable(
        preparestate!($mhe_nonlin_pred, $y, $d),
        samples=samples, evals=evals, seconds=seconds,
    )
UNIT_ESTIM["MovingHorizonEstimator"]["updatestate!"]["NonLinModel"]["Prediction form"] =
    @benchmarkable(
        updatestate!($mhe_nonlin_pred, $u, $y, $d),
        setup=preparestate!($mhe_nonlin_pred, $y, $d),
        samples=samples, evals=evals, seconds=seconds,
    ) 

## ----------------------------------------------------------------------------------------
## ----------------- CASE STUDIES ---------------------------------------------------------
## ----------------------------------------------------------------------------------------
const CASE_ESTIM = SUITE["CASE STUDIES"]["StateEstimator"]

## ----------------- Case study: CSTR -----------------------------------------------------
model = CSTR_model
plant = deepcopy(model)
plant.A[diagind(plant.A)] .-= 0.1 # plant-model mismatch
function test_mhe(mhe, plant)
    plant.x0 .= 0.1; y = plant() 
    initstate!(mhe, plant.uop, y)
    N = 75; u = [20, 20]; ul = 0
    U, Y, Ŷ = zeros(2, N), zeros(2, N), zeros(2, N)
    for i = 1:N
        i == 26 && (u = [15, 25])
        i == 51 && (ul = -10)
        y = plant() 
        preparestate!(mhe, y) 
        ŷ = evaloutput(mhe)
        U[:,i], Y[:,i], Ŷ[:,i] = u, y, ŷ
        updatestate!(mhe, u, y) 
        updatestate!(plant, u+[0,ul])
    end
    return U, Y, Ŷ
end
He = 4; nint_u = [1, 1]; σQint_u = [1, 2]
v̂min, v̂max = [-1, -0.5], [+1, +0.5]

optim = JuMP.Model(OSQP.Optimizer, add_bridges=false)
direct = true
mhe_cstr_osqp_curr = MovingHorizonEstimator(model; He, nint_u, σQint_u, optim, direct)
mhe_cstr_osqp_curr = setconstraint!(mhe_cstr_osqp_curr; v̂min, v̂max)
JuMP.unset_time_limit_sec(mhe_cstr_osqp_curr.optim)

optim = JuMP.Model(OSQP.Optimizer, add_bridges=false)
direct = false
mhe_cstr_osqp_pred = MovingHorizonEstimator(model; He, nint_u, σQint_u, optim, direct)
mhe_cstr_osqp_pred = setconstraint!(mhe_cstr_osqp_pred; v̂min, v̂max)
JuMP.unset_time_limit_sec(mhe_cstr_osqp_pred.optim)

optim = JuMP.Model(DAQP.Optimizer, add_bridges=false)
direct = true
mhe_cstr_daqp_curr = MovingHorizonEstimator(model; He, nint_u, σQint_u, optim, direct)
mhe_cstr_daqp_curr = setconstraint!(mhe_cstr_daqp_curr; v̂min, v̂max)
JuMP.set_attribute(mhe_cstr_daqp_curr.optim, "eps_prox", 1e-6) # needed to support hessians H≥0

optim = JuMP.Model(DAQP.Optimizer, add_bridges=false)
direct = false
mhe_cstr_daqp_pred = MovingHorizonEstimator(model; He, nint_u, σQint_u, optim, direct)
mhe_cstr_daqp_pred = setconstraint!(mhe_cstr_daqp_pred; v̂min, v̂max)
JuMP.set_attribute(mhe_cstr_daqp_pred.optim, "eps_prox", 1e-6) # needed to support hessians H≥0

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
direct = true
mhe_cstr_ipopt_curr = MovingHorizonEstimator(model; He, nint_u, σQint_u, optim, direct)
mhe_cstr_ipopt_curr = setconstraint!(mhe_cstr_ipopt_curr; v̂min, v̂max)
JuMP.unset_time_limit_sec(mhe_cstr_ipopt_curr.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
direct = false
mhe_cstr_ipopt_pred = MovingHorizonEstimator(model; He, nint_u, σQint_u, optim, direct)
mhe_cstr_ipopt_pred = setconstraint!(mhe_cstr_ipopt_pred; v̂min, v̂max)
JuMP.unset_time_limit_sec(mhe_cstr_ipopt_pred.optim)

samples, evals = 10000, 1
CASE_ESTIM["CSTR"]["MovingHorizonEstimator"]["OSQP"]["Current form"] =
    @benchmarkable(test_mhe($mhe_cstr_osqp_curr, $plant); 
        samples=samples, evals=evals
    )
CASE_ESTIM["CSTR"]["MovingHorizonEstimator"]["OSQP"]["Prediction form"] =
    @benchmarkable(test_mhe($mhe_cstr_osqp_pred, $plant);
        samples=samples, evals=evals
    )
CASE_ESTIM["CSTR"]["MovingHorizonEstimator"]["DAQP"]["Current form"] =
    @benchmarkable(test_mhe($mhe_cstr_daqp_curr, $plant); 
        samples=samples, evals=evals
    )
CASE_ESTIM["CSTR"]["MovingHorizonEstimator"]["DAQP"]["Prediction form"] =
    @benchmarkable(test_mhe($mhe_cstr_daqp_pred, $plant);
        samples=samples, evals=evals
    )
CASE_ESTIM["CSTR"]["MovingHorizonEstimator"]["Ipopt"]["Current form"] =
    @benchmarkable(test_mhe($mhe_cstr_ipopt_curr, $plant); 
        samples=samples, evals=evals
    )
CASE_ESTIM["CSTR"]["MovingHorizonEstimator"]["Ipopt"]["Prediction form"] =
    @benchmarkable(test_mhe($mhe_cstr_ipopt_pred, $plant);
        samples=samples, evals=evals
    )

## ---------------------- Case study: pendulum -------------------------------------------
model, p = pendulum_model, pendulum_p
plant = deepcopy(model)
plant.p[3] = 1.25*p[3]  # plant-model mismatch
σQ = [0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
He = 3; v̂min, v̂max = [-5.0], [+5.0]
N = 35; 

x_0 = [0.1, 0.1]; x̂_0 = [0, 0, 0]; u = [0.5]

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
direct = true
mhe_pendulum_ipopt_curr = MovingHorizonEstimator(
    model; He, σQ, σR, nint_u, σQint_u, optim, direct
)
mhe_pendulum_ipopt_curr = setconstraint!(mhe_pendulum_ipopt_curr; v̂min, v̂max)
JuMP.unset_time_limit_sec(mhe_pendulum_ipopt_curr.optim)
JuMP.set_attribute(mhe_pendulum_ipopt_curr.optim, "tol", 1e-7)

hessian = true
mhe_pendulum_ipopt_currh = MovingHorizonEstimator(
    model; He, σQ, σR, nint_u, σQint_u, optim, direct, hessian
)
mhe_pendulum_ipopt_currh = setconstraint!(mhe_pendulum_ipopt_currh; v̂min, v̂max)
JuMP.unset_time_limit_sec(mhe_pendulum_ipopt_currh.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
direct = false
mhe_pendulum_ipopt_pred = MovingHorizonEstimator(
    model; He, σQ, σR, nint_u, σQint_u, optim, direct
)
mhe_pendulum_ipopt_pred = setconstraint!(mhe_pendulum_ipopt_pred; v̂min, v̂max)
JuMP.unset_time_limit_sec(mhe_pendulum_ipopt_pred.optim)
JuMP.set_attribute(mhe_pendulum_ipopt_pred.optim, "tol", 1e-7)

hessian = true
mhe_pendulum_ipopt_predh = MovingHorizonEstimator(
    model; He, σQ, σR, nint_u, σQint_u, optim, direct, hessian
)
mhe_pendulum_ipopt_predh = setconstraint!(mhe_pendulum_ipopt_predh; v̂min, v̂max)
JuMP.unset_time_limit_sec(mhe_pendulum_ipopt_predh.optim)
JuMP.set_attribute(mhe_pendulum_ipopt_predh.optim, "tol", 1e-7)

optim = JuMP.Model(MadNLP.Optimizer, add_bridges=false)
direct = true
mhe_pendulum_madnlp_curr = MovingHorizonEstimator(
    model; He, σQ, σR, nint_u, σQint_u, optim, direct
)
mhe_pendulum_madnlp_curr = setconstraint!(mhe_pendulum_madnlp_curr; v̂min, v̂max)
JuMP.unset_time_limit_sec(mhe_pendulum_madnlp_curr.optim)
JuMP.set_attribute(mhe_pendulum_madnlp_curr.optim, "tol", 1e-7)

optim = JuMP.Model(MadNLP.Optimizer, add_bridges=false)
direct = false
mhe_pendulum_madnlp_pred = MovingHorizonEstimator(
    model; He, σQ, σR, nint_u, σQint_u, optim, direct
)
mhe_pendulum_madnlp_pred = setconstraint!(mhe_pendulum_madnlp_pred; v̂min, v̂max)
JuMP.unset_time_limit_sec(mhe_pendulum_madnlp_pred.optim)
JuMP.set_attribute(mhe_pendulum_madnlp_pred.optim, "tol", 1e-7)

samples, evals, seconds = 25, 1, 15*60
CASE_ESTIM["Pendulum"]["MovingHorizonEstimator"]["Ipopt"]["Current form"] =
    @benchmarkable(
        sim!($mhe_pendulum_ipopt_curr, $N, $u; plant=$plant, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_ESTIM["Pendulum"]["MovingHorizonEstimator"]["Ipopt"]["Current form (Hessian)"] =
    @benchmarkable(
        sim!($mhe_pendulum_ipopt_currh, $N, $u; plant=$plant, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_ESTIM["Pendulum"]["MovingHorizonEstimator"]["Ipopt"]["Prediction form"] =
    @benchmarkable(
        sim!($mhe_pendulum_ipopt_pred, $N, $u; plant=$plant, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_ESTIM["Pendulum"]["MovingHorizonEstimator"]["Ipopt"]["Prediction form (Hessian)"] =
    @benchmarkable(
        sim!($mhe_pendulum_ipopt_predh, $N, $u; plant=$plant, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_ESTIM["Pendulum"]["MovingHorizonEstimator"]["MadNLP"]["Current form"] =
    @benchmarkable(
        sim!($mhe_pendulum_madnlp_curr, $N, $u; plant=$plant, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_ESTIM["Pendulum"]["MovingHorizonEstimator"]["MadNLP"]["Prediction form"] =
    @benchmarkable(
        sim!($mhe_pendulum_madnlp_pred, $N, $u; plant=$plant, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )