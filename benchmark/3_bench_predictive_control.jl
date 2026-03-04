# -----------------------------------------------------------------------------------------
# ---------------------- UNIT TESTS -------------------------------------------------------
# -----------------------------------------------------------------------------------------
const UNIT_MPC = SUITE["UNIT TESTS"]["PredictiveController"]

linmpc_ss = LinMPC(
    linmodel, transcription=SingleShooting(), 
    Mwt=[1, 1], Nwt=[0.1, 0.1], Lwt=[0.1, 0.1], Hp=10
)
linmpc_ms = LinMPC(
    linmodel, transcription=MultipleShooting(), 
    Mwt=[1, 1], Nwt=[0.1, 0.1], Lwt=[0.1, 0.1], Hp=10
)

samples, evals, seconds = 10000, 1, 60
UNIT_MPC["LinMPC"]["moveinput!"]["SingleShooting"] = 
    @benchmarkable(
        moveinput!($linmpc_ss, $y, $d),
        setup=preparestate!($linmpc_ss, $y, $d),
        samples=samples, evals=evals, seconds=seconds
    )
UNIT_MPC["LinMPC"]["moveinput!"]["MultipleShooting"] = 
    @benchmarkable(
        moveinput!($linmpc_ms, $y, $d),
        setup=preparestate!($linmpc_ms, $y, $d),
        samples=samples, evals=evals, seconds=seconds
    )
UNIT_MPC["LinMPC"]["getinfo!"] =
    @benchmarkable(
        getinfo($linmpc_ss),
        setup=(preparestate!($linmpc_ss, $y, $d); moveinput!($linmpc_ss, $y, $d)),
        samples=samples, evals=evals, seconds=seconds
    )

empc = ExplicitMPC(linmodel, Mwt=[1, 1], Nwt=[0.1, 0.1], Lwt=[0.1, 0.1], Hp=10)

UNIT_MPC["ExplicitMPC"]["moveinput!"] = 
    @benchmarkable(
        moveinput!($empc, $y, $d),
        setup=preparestate!($empc, $y, $d),
    )

nmpc_lin_ss = NonLinMPC(
    linmodel, transcription=SingleShooting(),
    Mwt=[1, 1], Nwt=[0.1, 0.1], Lwt=[0.1, 0.1], Hp=10
)
JuMP.set_attribute(nmpc_lin_ss.optim, "tol", 1e-7)
nmpc_lin_ms = NonLinMPC(
    linmodel, transcription=MultipleShooting(),
    Mwt=[1, 1], Nwt=[0.1, 0.1], Lwt=[0.1, 0.1], Hp=10
)
JuMP.set_attribute(nmpc_lin_ms.optim, "tol", 1e-7)
nmpc_nonlin_ss = NonLinMPC(
    nonlinmodel, transcription=SingleShooting(),
    Mwt=[1, 1], Nwt=[0.1, 0.1], Lwt=[0.1, 0.1], Hp=10    
)
JuMP.set_attribute(nmpc_nonlin_ss.optim, "tol", 1e-7)
nmpc_nonlin_ss_hess = NonLinMPC(
    nonlinmodel, transcription=SingleShooting(), hessian=true,
    Mwt=[1, 1], Nwt=[0.1, 0.1], Lwt=[0.1, 0.1], Hp=10   
)
JuMP.set_attribute(nmpc_nonlin_ss_hess.optim, "tol", 1e-7)
nmpc_nonlin_ms = NonLinMPC(
    nonlinmodel, transcription=MultipleShooting(),
    Mwt=[1, 1], Nwt=[0.1, 0.1], Lwt=[0.1, 0.1], Hp=10    
)
JuMP.set_attribute(nmpc_nonlin_ms.optim, "tol", 1e-7)
nmpc_nonlin_ms_hess = NonLinMPC(
    nonlinmodel, transcription=MultipleShooting(), hessian=true,
    Mwt=[1, 1], Nwt=[0.1, 0.1], Lwt=[0.1, 0.1], Hp=10 
)
JuMP.set_attribute(nmpc_nonlin_ms_hess.optim, "tol", 1e-7)
nmpc_nonlin_tc = NonLinMPC(
    nonlinmodel_c, transcription=TrapezoidalCollocation(),
    Mwt=[1], Nwt=[0.1], Lwt=[0.1], Hp=10    
)
JuMP.set_attribute(nmpc_nonlin_tc.optim, "tol", 1e-7)
nmpc_nonlin_oc = NonLinMPC(
    nonlinmodel_c, transcription=OrthogonalCollocation(),
    Mwt=[1], Nwt=[0.1], Lwt=[0.1], Hp=10    
)
JuMP.set_attribute(nmpc_nonlin_oc.optim, "tol", 1e-7)
nmpc_nonlin_oc_hess = NonLinMPC(
    nonlinmodel_c, transcription=OrthogonalCollocation(), hessian=true,
    Mwt=[1], Nwt=[0.1], Lwt=[0.1], Hp=10    
)
JuMP.set_attribute(nmpc_nonlin_oc_hess.optim, "tol", 1e-7)

samples, evals, seconds = 10000, 1, 60
UNIT_MPC["NonLinMPC"]["moveinput!"]["LinModel"]["SingleShooting"] =
    @benchmarkable(
        moveinput!($nmpc_lin_ss, $y, $d),
        setup=preparestate!($nmpc_lin_ss, $y, $d),
        samples=samples, evals=evals, seconds=seconds
    )
UNIT_MPC["NonLinMPC"]["moveinput!"]["LinModel"]["MultipleShooting"] =
    @benchmarkable(
        moveinput!($nmpc_lin_ms, $y, $d),
        setup=preparestate!($nmpc_lin_ms, $y, $d),
        samples=samples, evals=evals, seconds=seconds
    )
UNIT_MPC["NonLinMPC"]["getinfo!"]["LinModel"] =
    @benchmarkable(
        getinfo($nmpc_lin_ss),
        setup=(preparestate!($nmpc_lin_ss, $y, $d); moveinput!($nmpc_lin_ss, $y, $d)),
        samples=samples, evals=evals, seconds=seconds
    )
UNIT_MPC["NonLinMPC"]["moveinput!"]["NonLinModel"]["SingleShooting"] =
    @benchmarkable(
        moveinput!($nmpc_nonlin_ss, $y, $d),
        setup=preparestate!($nmpc_nonlin_ss, $y, $d),
        samples=samples, evals=evals, seconds=seconds
    )
UNIT_MPC["NonLinMPC"]["moveinput!"]["NonLinModel"]["SingleShootingHessian"] =
    @benchmarkable(
        moveinput!($nmpc_nonlin_ss, $y, $d),
        setup=preparestate!($nmpc_nonlin_ss_hess, $y, $d),
        samples=samples, evals=evals, seconds=seconds
    )
UNIT_MPC["NonLinMPC"]["moveinput!"]["NonLinModel"]["MultipleShooting"] =
    @benchmarkable(
        moveinput!($nmpc_nonlin_ms, $y, $d),
        setup=preparestate!($nmpc_nonlin_ms, $y, $d),
        samples=samples, evals=evals, seconds=seconds
    )
UNIT_MPC["NonLinMPC"]["moveinput!"]["NonLinModel"]["MultipleShootingHessian"] =
    @benchmarkable(
        moveinput!($nmpc_nonlin_ms, $y, $d),
        setup=preparestate!($nmpc_nonlin_ms_hess, $y, $d),
        samples=samples, evals=evals, seconds=seconds
    )
UNIT_MPC["NonLinMPC"]["moveinput!"]["NonLinModel"]["TrapezoidalCollocation"] =
    @benchmarkable(
        moveinput!($nmpc_nonlin_tc, $y_c, $d_c),
        setup=preparestate!($nmpc_nonlin_tc, $y_c, $d_c),
        samples=samples, evals=evals, seconds=seconds
    )
UNIT_MPC["NonLinMPC"]["moveinput!"]["NonLinModel"]["OrthogonalCollocation"] =
    @benchmarkable(
        moveinput!($nmpc_nonlin_oc, $y_c, $d_c),
        setup=preparestate!($nmpc_nonlin_oc, $y_c, $d_c),
        samples=samples, evals=evals, seconds=seconds
    )
UNIT_MPC["NonLinMPC"]["moveinput!"]["NonLinModel"]["OrthogonalCollocationHessian"] =
    @benchmarkable(
        moveinput!($nmpc_nonlin_oc_hess, $y_c, $d_c),
        setup=preparestate!($nmpc_nonlin_oc_hess, $y_c, $d_c),
        samples=samples, evals=evals, seconds=seconds
    )
UNIT_MPC["NonLinMPC"]["getinfo!"]["NonLinModel"] =
    @benchmarkable(
        getinfo($nmpc_nonlin_ss),
        setup=(preparestate!($nmpc_nonlin_ss, $y, $d); moveinput!($nmpc_nonlin_ss, $y, $d)),
        samples=samples, evals=evals, seconds=seconds
    )

## ----------------------------------------------------------------------------------------
## ---------------------- CASE STUDIES ----------------------------------------------------
## ----------------------------------------------------------------------------------------
const CASE_MPC = SUITE["CASE STUDIES"]["PredictiveController"]

## ----------------- Case study: CSTR without feedforward ---------------------------------
model = CSTR_model
plant = deepcopy(model)
plant.A[diagind(plant.A)] .-= 0.1 # plant-model mismatch
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

# # Skip DAQP with MultipleShooting, it is not designed for sparse Hessians. Kind of works 
# # with "eps_prox" configured to 1e-6, but not worth it.
# optim = JuMP.Model(DAQP.Optimizer, add_bridges=false)
# transcription = MultipleShooting()
# mpc_daqp_ms = setconstraint!(LinMPC(model; optim, transcription), ymin=[45, -Inf])
# JuMP.set_attribute(mpc_daqp_ms.optim, "eps_prox", 1e-6)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription = SingleShooting()
mpc_ipopt_ss = setconstraint!(LinMPC(model; optim, transcription), ymin=[45, -Inf])
JuMP.unset_time_limit_sec(mpc_ipopt_ss.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription = MultipleShooting()
mpc_ipopt_ms = setconstraint!(LinMPC(model; optim, transcription), ymin=[45, -Inf])
JuMP.unset_time_limit_sec(mpc_ipopt_ms.optim) 

samples, evals = 5000, 1
CASE_MPC["CSTR"]["LinMPC"]["Without feedforward"]["OSQP"]["SingleShooting"] = 
    @benchmarkable(test_mpc($mpc_osqp_ss, $plant); 
        samples=samples, evals=evals
    )
CASE_MPC["CSTR"]["LinMPC"]["Without feedforward"]["OSQP"]["MultipleShooting"] = 
    @benchmarkable(test_mpc($mpc_osqp_ms, $plant); 
        samples=samples, evals=evals
    )
CASE_MPC["CSTR"]["LinMPC"]["Without feedforward"]["DAQP"]["SingleShooting"] =
    @benchmarkable(test_mpc($mpc_daqp_ss, $plant); 
        samples=samples, evals=evals
    )
CASE_MPC["CSTR"]["LinMPC"]["Without feedforward"]["Ipopt"]["SingleShooting"] =
    @benchmarkable(test_mpc($mpc_ipopt_ss, $plant); 
    samples=samples, evals=evals
)
CASE_MPC["CSTR"]["LinMPC"]["Without feedforward"]["Ipopt"]["MultipleShooting"] =
    @benchmarkable(test_mpc($mpc_ipopt_ms, $plant); 
        samples=samples, evals=evals
    )

## ----------------- Case study: CSTR with feedforward -------------------------
model_d = CSTR_model_d
function test_mpc_d(mpc_d, plant)
    plant.x0 .= 0; y = plant(); d = [20]
    initstate!(mpc_d, plant.uop, y, d)
    N = 75; ry = [50, 30]; ul = 0
    U, Y, Ry = zeros(2, N), zeros(2, N), zeros(2, N)
    for i = 1:N
        i == 26 && (ry = [48, 35])
        i == 51 && (ul = -10)
        y, d = plant(), [20+ul]
        preparestate!(mpc_d, y, d)
        u = mpc_d(ry, d)
        U[:,i], Y[:,i], Ry[:,i] = u, y, ry
        updatestate!(mpc_d, u, y, d)
        updatestate!(plant, u+[0,ul])
    end
    return U, Y, Ry
end

optim = JuMP.Model(OSQP.Optimizer, add_bridges=false)
transcription = SingleShooting()
mpc_d_osqp_ss = setconstraint!(LinMPC(model_d; optim, transcription), ymin=[45, -Inf])
JuMP.unset_time_limit_sec(mpc_d_osqp_ss.optim)

optim = JuMP.Model(OSQP.Optimizer, add_bridges=false)
transcription = MultipleShooting()
mpc_d_osqp_ms = setconstraint!(LinMPC(model_d; optim, transcription), ymin=[45, -Inf])
JuMP.unset_time_limit_sec(mpc_d_osqp_ms.optim)

optim = JuMP.Model(DAQP.Optimizer, add_bridges=false)
transcription = SingleShooting()
mpc_d_daqp_ss = setconstraint!(LinMPC(model_d; optim, transcription), ymin=[45, -Inf])

# # Skip DAQP with MultipleShooting, it is not designed for sparse Hessians. Kind of works 
# # with "eps_prox" configured to 1e-6, but not worth it.
# optim = JuMP.Model(DAQP.Optimizer, add_bridges=false)
# transcription = MultipleShooting()
# mpc_d_daqp_ms = setconstraint!(LinMPC(model_d; optim, transcription), ymin=[45, -Inf])
# JuMP.set_attribute(mpc_d_daqp_ms.optim, "eps_prox", 1e-6)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription = SingleShooting()
mpc_d_ipopt_ss = setconstraint!(LinMPC(model_d; optim, transcription), ymin=[45, -Inf])
JuMP.unset_time_limit_sec(mpc_d_ipopt_ss.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription = MultipleShooting()
mpc_d_ipopt_ms = setconstraint!(LinMPC(model_d; optim, transcription), ymin=[45, -Inf])
JuMP.unset_time_limit_sec(mpc_d_ipopt_ms.optim)

samples, evals = 5000, 1
CASE_MPC["CSTR"]["LinMPC"]["With feedforward"]["OSQP"]["SingleShooting"] = 
    @benchmarkable(test_mpc_d($mpc_d_osqp_ss, $plant); 
        samples=samples, evals=evals
    )
CASE_MPC["CSTR"]["LinMPC"]["With feedforward"]["OSQP"]["MultipleShooting"] =
    @benchmarkable(test_mpc_d($mpc_d_osqp_ms, $plant); 
        samples=samples, evals=evals
    )
CASE_MPC["CSTR"]["LinMPC"]["With feedforward"]["DAQP"]["SingleShooting"] =
    @benchmarkable(test_mpc_d($mpc_d_daqp_ss, $plant); 
        samples=samples, evals=evals
    )
CASE_MPC["CSTR"]["LinMPC"]["With feedforward"]["Ipopt"]["SingleShooting"] =
    @benchmarkable(test_mpc_d($mpc_d_ipopt_ss, $plant); 
        samples=samples, evals=evals
    )
CASE_MPC["CSTR"]["LinMPC"]["With feedforward"]["Ipopt"]["MultipleShooting"] =
    @benchmarkable(test_mpc_d($mpc_d_ipopt_ms, $plant); 
        samples=samples, evals=evals
    )

# ----------------- Case study: Pendulum noneconomic -----------------------------
model, p = pendulum_model, pendulum_p
σQ = [0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
estim = UnscentedKalmanFilter(model; σQ, σR, nint_u, σQint_u)
plant = deepcopy(model)
plant.p[3] = 1.25*p[3]  # plant-model mismatch
N = 35; u = [0.5]; 

Hp, Hc, Mwt, Nwt, Cwt = 20, 2, [0.5], [2.5], Inf
umin, umax = [-1.5], [+1.5]
x_0 = [0, 0]; x̂_0 = [0, 0, 0]; ry = [180]

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription = SingleShooting()
nmpc_ipopt_ss = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt, optim, transcription)
nmpc_ipopt_ss = setconstraint!(nmpc_ipopt_ss; umin, umax)
JuMP.unset_time_limit_sec(nmpc_ipopt_ss.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription, hessian = SingleShooting(), true
nmpc_ipopt_ss_hess = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt, optim, transcription, hessian)
nmpc_ipopt_ss_hess = setconstraint!(nmpc_ipopt_ss_hess; umin, umax)
JuMP.unset_time_limit_sec(nmpc_ipopt_ss_hess.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription = MultipleShooting()
nmpc_ipopt_ms = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt, optim, transcription)
nmpc_ipopt_ms = setconstraint!(nmpc_ipopt_ms; umin, umax)
JuMP.unset_time_limit_sec(nmpc_ipopt_ms.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription, hessian = MultipleShooting(), true
nmpc_ipopt_ms_hess = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt, optim, transcription, hessian)
nmpc_ipopt_ms_hess = setconstraint!(nmpc_ipopt_ms_hess; umin, umax)
JuMP.unset_time_limit_sec(nmpc_ipopt_ms_hess.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription = MultipleShooting(f_threads=true)
nmpc_ipopt_mst = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt, optim, transcription)
nmpc_ipopt_mst = setconstraint!(nmpc_ipopt_mst; umin, umax)
JuMP.unset_time_limit_sec(nmpc_ipopt_mst.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription = TrapezoidalCollocation()
nmpc_ipopt_tc = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt, optim, transcription)
nmpc_ipopt_tc = setconstraint!(nmpc_ipopt_tc; umin, umax)
JuMP.unset_time_limit_sec(nmpc_ipopt_tc.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription, hessian = TrapezoidalCollocation(), true
nmpc_ipopt_tc_hess = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt, optim, transcription, hessian)
nmpc_ipopt_tc_hess = setconstraint!(nmpc_ipopt_tc_hess; umin, umax)
JuMP.unset_time_limit_sec(nmpc_ipopt_tc_hess.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription = TrapezoidalCollocation(f_threads=true)
nmpc_ipopt_tct = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt, optim, transcription)
nmpc_ipopt_tct = setconstraint!(nmpc_ipopt_tct; umin, umax)
JuMP.unset_time_limit_sec(nmpc_ipopt_tct.optim)

optim = JuMP.Model(MadNLP.Optimizer, add_bridges=false)
transcription = SingleShooting()
nmpc_madnlp_ss = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt, optim, transcription)
nmpc_madnlp_ss = setconstraint!(nmpc_madnlp_ss; umin, umax)
JuMP.unset_time_limit_sec(nmpc_madnlp_ss.optim)

optim = JuMP.Model(()->UnoSolver.Optimizer(preset="filtersqp"), add_bridges=false)
transcription, hessian = MultipleShooting(), true
nmpc_uno_ms_hess = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt, optim, transcription, hessian)
nmpc_uno_ms_hess = setconstraint!(nmpc_uno_ms_hess; umin, umax)
JuMP.unset_time_limit_sec(nmpc_uno_ms_hess.optim)

# skip MadNLP.jl with MultipleShooting and hessian=false, their LBFGS does not work well

optim = JuMP.Model(MadNLP.Optimizer)
transcription, hessian = MultipleShooting(), true
nmpc_madnlp_ms_hess = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt, optim, transcription, hessian)
nmpc_madnlp_ms_hess = setconstraint!(nmpc_madnlp_ms_hess; umin, umax)
JuMP.unset_time_limit_sec(nmpc_madnlp_ms_hess.optim)

samples, evals, seconds = 100, 1, 15*60
CASE_MPC["Pendulum"]["NonLinMPC"]["Noneconomic"]["Ipopt"]["SingleShooting"] = 
    @benchmarkable(
        sim!($nmpc_ipopt_ss, $N, $ry; plant=$plant, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_MPC["Pendulum"]["NonLinMPC"]["Noneconomic"]["Ipopt"]["SingleShooting (Hessian)"] = 
    @benchmarkable(
        sim!($nmpc_ipopt_ss_hess, $N, $ry; plant=$plant, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_MPC["Pendulum"]["NonLinMPC"]["Noneconomic"]["Ipopt"]["MultipleShooting"] =
    @benchmarkable(
        sim!($nmpc_ipopt_ms, $N, $ry; plant=$plant, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_MPC["Pendulum"]["NonLinMPC"]["Noneconomic"]["Ipopt"]["MultipleShooting (Hessian)"] =
    @benchmarkable(
        sim!($nmpc_ipopt_ms_hess, $N, $ry; plant=$plant, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_MPC["Pendulum"]["NonLinMPC"]["Noneconomic"]["Ipopt"]["MultipleShooting (threaded)"] =
    @benchmarkable(
        sim!($nmpc_ipopt_mst, $N, $ry; plant=$plant, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_MPC["Pendulum"]["NonLinMPC"]["Noneconomic"]["Ipopt"]["TrapezoidalCollocation"] =
    @benchmarkable(
        sim!($nmpc_ipopt_tc, $N, $ry; plant=$plant, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_MPC["Pendulum"]["NonLinMPC"]["Noneconomic"]["Ipopt"]["TrapezoidalCollocation (Hessian)"] =
    @benchmarkable(
        sim!($nmpc_ipopt_tc_hess, $N, $ry; plant=$plant, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_MPC["Pendulum"]["NonLinMPC"]["Noneconomic"]["Ipopt"]["TrapezoidalCollocation (threaded)"] =
    @benchmarkable(
        sim!($nmpc_ipopt_tct, $N, $ry; plant=$plant, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_MPC["Pendulum"]["NonLinMPC"]["Noneconomic"]["MadNLP"]["SingleShooting"] = 
    @benchmarkable(
        sim!($nmpc_madnlp_ss, $N, $ry; plant=$plant, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_MPC["Pendulum"]["NonLinMPC"]["Noneconomic"]["MadNLP"]["MultipleShooting (Hessian)"] = 
    @benchmarkable(
        sim!($nmpc_madnlp_ms_hess, $N, $ry; plant=$plant, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
# CASE_MPC["Pendulum"]["NonLinMPC"]["Noneconomic"]["Uno"]["MultipleShooting (Hessian)"] = 
#     @benchmarkable(
#         sim!($nmpc_uno_ms_hess, $N, $ry; plant=$plant, x_0=$x_0, x̂_0=$x̂_0, progress=false),
#        samples=samples, evals=evals, seconds=seconds
#     )    

# ----------------- Case study: Pendulum economic --------------------------------
model2, p = pendulum_model2, pendulum_p2
plant2 = deepcopy(model2)
plant2.p[3] = 1.25*p[3]  # plant-model mismatch
estim2 = UnscentedKalmanFilter(model2; σQ, σR, nint_u, σQint_u, i_ym=[1])
function JE(UE, ŶE, _ , p, _)
    Ts = p
    τ, ω = @views UE[1:end-1], ŶE[2:2:end-1]
    return Ts*dot(τ, ω)
end
p = Ts; Mwt2 = [Mwt; 0.0]; Ewt = 3.5e3
x_0 = [0, 0]; x̂_0 = [0, 0, 0]; ry = [180; 0]

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription = SingleShooting()
empc_ipopt_ss = NonLinMPC(estim2; Hp, Hc, Nwt, Mwt=Mwt2, Cwt, JE, Ewt, optim, transcription, p)
empc_ipopt_ss = setconstraint!(empc_ipopt_ss; umin, umax)
JuMP.unset_time_limit_sec(empc_ipopt_ss.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription, hessian = SingleShooting(), true
empc_ipopt_ss_hess = NonLinMPC(estim2; Hp, Hc, Nwt, Mwt=Mwt2, Cwt, JE, Ewt, optim, transcription, hessian, p)
empc_ipopt_ss_hess = setconstraint!(empc_ipopt_ss_hess; umin, umax)
JuMP.unset_time_limit_sec(empc_ipopt_ss_hess.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription = MultipleShooting()
empc_ipopt_ms = NonLinMPC(estim2; Hp, Hc, Nwt, Mwt=Mwt2, Cwt, JE, Ewt, optim, transcription, p)
empc_ipopt_ms = setconstraint!(empc_ipopt_ms; umin, umax)
JuMP.unset_time_limit_sec(empc_ipopt_ms.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription, hessian = MultipleShooting(), true
empc_ipopt_ms_hess = NonLinMPC(estim2; Hp, Hc, Nwt, Mwt=Mwt2, Cwt, JE, Ewt, optim, transcription, hessian, p)
empc_ipopt_ms_hess = setconstraint!(empc_ipopt_ms_hess; umin, umax)
JuMP.unset_time_limit_sec(empc_ipopt_ms_hess.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription = TrapezoidalCollocation()
empc_ipopt_tc = NonLinMPC(estim2; Hp, Hc, Nwt, Mwt=Mwt2, Cwt, JE, Ewt, optim, transcription, p)
empc_ipopt_tc = setconstraint!(empc_ipopt_tc; umin, umax)
JuMP.unset_time_limit_sec(empc_ipopt_tc.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription, hessian = TrapezoidalCollocation(), true
empc_ipopt_tc_hess = NonLinMPC(estim2; Hp, Hc, Nwt, Mwt=Mwt2, Cwt, JE, Ewt, optim, transcription, hessian, p)
empc_ipopt_tc_hess = setconstraint!(empc_ipopt_tc_hess; umin, umax)
JuMP.unset_time_limit_sec(empc_ipopt_tc_hess.optim)

optim = JuMP.Model(MadNLP.Optimizer, add_bridges=false)
transcription = SingleShooting()
empc_madnlp_ss = NonLinMPC(estim2; Hp, Hc, Nwt, Mwt=Mwt2, Cwt, JE, Ewt, optim, transcription, p)
empc_madnlp_ss = setconstraint!(empc_madnlp_ss; umin, umax)
JuMP.unset_time_limit_sec(empc_madnlp_ss.optim)

# TODO: test EMPC with MadNLP and MultipleShooting and TrapezoidalCollocation, see comment above.

samples, evals, seconds = 100, 1, 15*60
CASE_MPC["Pendulum"]["NonLinMPC"]["Economic"]["Ipopt"]["SingleShooting"] = 
    @benchmarkable(
        sim!($empc_ipopt_ss, $N, $ry; plant=$plant2, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_MPC["Pendulum"]["NonLinMPC"]["Economic"]["Ipopt"]["SingleShooting (Hessian)"] = 
    @benchmarkable(
        sim!($empc_ipopt_ss_hess, $N, $ry; plant=$plant2, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_MPC["Pendulum"]["NonLinMPC"]["Economic"]["Ipopt"]["MultipleShooting"] =
    @benchmarkable(
        sim!($empc_ipopt_ms, $N, $ry; plant=$plant2, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_MPC["Pendulum"]["NonLinMPC"]["Economic"]["Ipopt"]["MultipleShooting (Hessian)"] =
    @benchmarkable(
        sim!($empc_ipopt_ms_hess, $N, $ry; plant=$plant2, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_MPC["Pendulum"]["NonLinMPC"]["Economic"]["Ipopt"]["TrapezoidalCollocation"] =
    @benchmarkable(
        sim!($empc_ipopt_tc, $N, $ry; plant=$plant2, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_MPC["Pendulum"]["NonLinMPC"]["Economic"]["Ipopt"]["TrapezoidalCollocation (Hessian)"] =
    @benchmarkable(
        sim!($empc_ipopt_tc_hess, $N, $ry; plant=$plant2, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_MPC["Pendulum"]["NonLinMPC"]["Economic"]["MadNLP"]["SingleShooting"] = 
    @benchmarkable(
        sim!($empc_madnlp_ss, $N, $ry; plant=$plant2, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )

# -------------- Case study: Pendulum custom constraints --------------------------
function gc!(LHS, Ue, Ŷe, _, p, ϵ)
    Pmax = p
    i_τ, i_ω = 1, 2
    for i in eachindex(LHS)
        τ, ω = Ue[i_τ], Ŷe[i_ω]
        P = τ*ω
        LHS[i] = P - Pmax - ϵ
        i_τ += 1
        i_ω += 2
    end
    return nothing
end
Cwt, Pmax, nc = 1e5, 3, Hp+1
x_0 = [0, 0]; x̂_0 = [0, 0, 0]; ry = [180; 0]

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription = SingleShooting()
nmpc2_ipopt_ss = NonLinMPC(estim2; 
    Hp, Hc, Nwt=Nwt, Mwt=[0.5, 0], Cwt, gc!, nc, p=Pmax, optim, transcription
)
nmpc2_ipopt_ss = setconstraint!(nmpc2_ipopt_ss; umin, umax)
JuMP.unset_time_limit_sec(nmpc2_ipopt_ss.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription = MultipleShooting()
nmpc2_ipopt_ms = NonLinMPC(estim2; 
    Hp, Hc, Nwt=Nwt, Mwt=[0.5, 0], Cwt, gc!, nc, p=Pmax, optim, transcription
)
nmpc2_ipopt_ms = setconstraint!(nmpc2_ipopt_ms; umin, umax)
JuMP.unset_time_limit_sec(nmpc2_ipopt_ms.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription, hessian = MultipleShooting(), true
nmpc2_ipopt_ms_hess = NonLinMPC(estim2; 
    Hp, Hc, Nwt=Nwt, Mwt=[0.5, 0], Cwt, gc!, nc, p=Pmax, optim, transcription, hessian
)
nmpc2_ipopt_ms_hess = setconstraint!(nmpc2_ipopt_ms_hess; umin, umax)
JuMP.unset_time_limit_sec(nmpc2_ipopt_ms_hess.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription = TrapezoidalCollocation()
nmpc2_ipopt_tc = NonLinMPC(estim2; 
    Hp, Hc, Nwt=Nwt, Mwt=[0.5, 0], Cwt, gc!, nc, p=Pmax, optim, transcription
)
nmpc2_ipopt_tc = setconstraint!(nmpc2_ipopt_tc; umin, umax)
JuMP.unset_time_limit_sec(nmpc2_ipopt_tc.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription, hessian = TrapezoidalCollocation(), true
nmpc2_ipopt_tc_hess = NonLinMPC(estim2; 
    Hp, Hc, Nwt=Nwt, Mwt=[0.5, 0], Cwt, gc!, nc, p=Pmax, optim, transcription, hessian
)
nmpc2_ipopt_tc = setconstraint!(nmpc2_ipopt_tc_hess; umin, umax)
JuMP.unset_time_limit_sec(nmpc2_ipopt_tc_hess.optim)

samples, evals, seconds = 100, 1, 15*60
CASE_MPC["Pendulum"]["NonLinMPC"]["Custom constraints"]["Ipopt"]["SingleShooting"] = 
    @benchmarkable(
        sim!($nmpc2_ipopt_ss, $N, $ry; plant=$plant2, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_MPC["Pendulum"]["NonLinMPC"]["Custom constraints"]["Ipopt"]["MultipleShooting"] =
    @benchmarkable(
        sim!($nmpc2_ipopt_ms, $N, $ry; plant=$plant2, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_MPC["Pendulum"]["NonLinMPC"]["Custom constraints"]["Ipopt"]["MultipleShooting (Hessian)"] =
    @benchmarkable(
        sim!($nmpc2_ipopt_ms_hess, $N, $ry; plant=$plant2, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_MPC["Pendulum"]["NonLinMPC"]["Custom constraints"]["Ipopt"]["TrapezoidalCollocation"] =
    @benchmarkable(
        sim!($nmpc2_ipopt_tc, $N, $ry; plant=$plant2, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )
CASE_MPC["Pendulum"]["NonLinMPC"]["Custom constraints"]["Ipopt"]["TrapezoidalCollocation (Hessian)"] =
    @benchmarkable(
        sim!($nmpc2_ipopt_tc_hess, $N, $ry; plant=$plant2, x_0=$x_0, x̂_0=$x̂_0, progress=false),
        samples=samples, evals=evals, seconds=seconds
    )

# ----------------- Case study: Pendulum successive linearization -------------------------
linmodel = linearize(model, x=[0, 0], u=[0])
kf = KalmanFilter(linmodel; σQ, σR, nint_u, σQint_u)
function sim2!(mpc, nlmodel, N, ry, plant, x, 𝕩̂, y_step)
    U, Y, Ry = zeros(1, N), zeros(1, N), zeros(1, N)
    setstate!(plant, x); setstate!(mpc, 𝕩̂)
    initstate!(mpc, [0], plant())
    linmodel = linearize(nlmodel; u=[0], x=𝕩̂[1:2])
    setmodel!(mpc, linmodel)
    for i = 1:N
        y = plant() + y_step
        𝕩̂ = preparestate!(mpc, y)
        u = mpc(ry)
        linearize!(linmodel, nlmodel; u, x=𝕩̂[1:2])
        setmodel!(mpc, linmodel) 
        U[:,i], Y[:,i], Ry[:,i] = u, y, ry
        updatestate!(mpc, u, y)
        updatestate!(plant, u)
    end
    U_data, Y_data, Ry_data = U, Y, Ry
    return SimResult(mpc, U_data, Y_data; Ry_data)
end
x_0 = [0, 0]; x̂_0 = [0, 0, 0]; ry = [180]; y_step=[0]

optim = JuMP.Model(OSQP.Optimizer, add_bridges=false)
transcription = SingleShooting()
mpc3_osqp_ss = LinMPC(kf; Hp, Hc, Mwt, Nwt, Cwt, optim, transcription)
mpc3_osqp_ss = setconstraint!(mpc3_osqp_ss; umin, umax)
JuMP.unset_time_limit_sec(mpc3_osqp_ss.optim)
JuMP.set_attribute(mpc3_osqp_ss.optim, "polish", true) # needed to 
JuMP.set_attribute(mpc3_osqp_ss.optim, "sigma",  1e-9) # needed to 

optim = JuMP.Model(OSQP.Optimizer, add_bridges=false)
transcription = MultipleShooting()
mpc3_osqp_ms = LinMPC(kf; Hp, Hc, Mwt, Nwt, Cwt, optim, transcription)
mpc3_osqp_ms = setconstraint!(mpc3_osqp_ms; umin, umax)
JuMP.unset_time_limit_sec(mpc3_osqp_ms.optim)

optim = JuMP.Model(DAQP.Optimizer, add_bridges=false)
transcription = SingleShooting()
mpc3_daqp_ss = LinMPC(kf; Hp, Hc, Mwt, Nwt, Cwt, optim, transcription)
mpc3_daqp_ss = setconstraint!(mpc3_daqp_ss; umin, umax)

# skip DAQP with MultipleShooting, it is not designed for sparse Hessians
# did not found any settings that works well here (always reach the iteration limit).

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription = SingleShooting()
mpc3_ipopt_ss = LinMPC(kf; Hp, Hc, Mwt, Nwt, Cwt, optim, transcription)
mpc3_ipopt_ss = setconstraint!(mpc3_ipopt_ss; umin, umax)
JuMP.unset_time_limit_sec(mpc3_ipopt_ss.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription = MultipleShooting()
mpc3_ipopt_ms = LinMPC(kf; Hp, Hc, Mwt, Nwt, Cwt, optim, transcription)
mpc3_ipopt_ms = setconstraint!(mpc3_ipopt_ms; umin, umax)
JuMP.unset_time_limit_sec(mpc3_ipopt_ms.optim)

samples, evals = 10000, 1
CASE_MPC["Pendulum"]["LinMPC"]["Successive linearization"]["OSQP"]["SingleShooting"] = 
    @benchmarkable(
        sim2!($mpc3_osqp_ss, $model, $N, $ry, $plant, $x_0, $x̂_0, $y_step),
        samples=samples, evals=evals
    )
CASE_MPC["Pendulum"]["LinMPC"]["Successive linearization"]["OSQP"]["MultipleShooting"] = 
    @benchmarkable(
        sim2!($mpc3_osqp_ms, $model, $N, $ry, $plant, $x_0, $x̂_0, $y_step),
        samples=samples, evals=evals
    )
CASE_MPC["Pendulum"]["LinMPC"]["Successive linearization"]["DAQP"]["SingleShooting"] = 
    @benchmarkable(
        sim2!($mpc3_daqp_ss, $model, $N, $ry, $plant, $x_0, $x̂_0, $y_step),
        samples=samples, evals=evals
    )
CASE_MPC["Pendulum"]["LinMPC"]["Successive linearization"]["Ipopt"]["SingleShooting"] = 
    @benchmarkable(
        sim2!($mpc3_ipopt_ss, $model, $N, $ry, $plant, $x_0, $x̂_0, $y_step),
        samples=samples, evals=evals
    )
CASE_MPC["Pendulum"]["LinMPC"]["Successive linearization"]["Ipopt"]["MultipleShooting"] = 
    @benchmarkable(
        sim2!($mpc3_ipopt_ms, $model, $N, $ ry, $plant, $x_0, $x̂_0, $y_step),
        samples=samples, evals=evals
    )
