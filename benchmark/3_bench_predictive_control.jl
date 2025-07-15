## ----------------- Runtime benchmarks : CSTR ----------------------------------------
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
# needed to solve Hessians with eigenvalues at zero, like with MultipleShooting:
JuMP.set_attribute(mpc_daqp_ms.optim, "eps_prox", 1e-6) 

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription = SingleShooting()
mpc_ipopt_ss = setconstraint!(LinMPC(model; optim, transcription), ymin=[45, -Inf])
JuMP.unset_time_limit_sec(mpc_ipopt_ss.optim)

optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer,"sb"=>"yes"), add_bridges=false)
transcription = MultipleShooting()
mpc_ipopt_ms = setconstraint!(LinMPC(model; optim, transcription), ymin=[45, -Inf])
JuMP.unset_time_limit_sec(mpc_ipopt_ms.optim) 

samples, evals = 500, 1
RUNTIME["PredictiveController"]["CSTR"]["LinMPC"]["OSQP"]["SingleShooting"] = 
    @benchmarkable(test_mpc($mpc_osqp_ss, $model); 
    samples=samples, evals=evals
)
RUNTIME["PredictiveController"]["CSTR"]["LinMPC"]["OSQP"]["MultipleShooting"] = 
    @benchmarkable(test_mpc($mpc_osqp_ms, $model); 
    samples=samples, evals=evals
)
RUNTIME["PredictiveController"]["CSTR"]["LinMPC"]["DAQP"]["SingleShooting"] =
    @benchmarkable(test_mpc($mpc_daqp_ss, $model); 
    samples=samples, evals=evals
)
RUNTIME["PredictiveController"]["CSTR"]["LinMPC"]["DAQP"]["MultipleShooting"] =
    @benchmarkable(test_mpc($mpc_daqp_ms, $model); 
    samples=samples, evals=evals
)
RUNTIME["PredictiveController"]["CSTR"]["LinMPC"]["Ipopt"]["SingleShooting"] =
    @benchmarkable(test_mpc($mpc_ipopt_ss, $model); 
    samples=samples, evals=evals
)
RUNTIME["PredictiveController"]["CSTR"]["LinMPC"]["Ipopt"]["MultipleShooting"] =
    @benchmarkable(test_mpc($mpc_ipopt_ms, $model); 
    samples=samples, evals=evals
)

# ----------------- Runtime benchmarks : Pendulum ---------------------------------------
function f!(ẋ, x, u, _ , p)
    g, L, K, m = p       # [m/s²], [m], [kg/s], [kg]
    θ, ω = x[1], x[2]    # [rad], [rad/s]
    τ = u[1]             # [Nm]
    ẋ[1] = ω
    ẋ[2] = -g/L*sin(θ) - K/m*ω + τ/m/L^2
end
h!(y, x, _ , _ ) = (y[1] = 180/π*x[1])   # [°]
p = [9.8, 0.4, 1.2, 0.3]
nu = 1; nx = 2; ny = 1; Ts = 0.1
model = NonLinModel(f!, h!, Ts, nu, nx, ny; p)
σQ = [0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
estim = UnscentedKalmanFilter(model; σQ, σR, nint_u, σQint_u)
p_plant = copy(p); p_plant[3] = 1.25*p[3]
plant = NonLinModel(f!, h!, Ts, nu, nx, ny; p=p_plant)
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
transcription = MultipleShooting()
nmpc_ipopt_ms = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt, optim, transcription)
nmpc_ipopt_ms = setconstraint!(nmpc_ipopt_ms; umin, umax)
JuMP.unset_time_limit_sec(nmpc_ipopt_ms.optim)

optim = JuMP.Model(MadNLP.Optimizer, add_bridges=false)
transcription = SingleShooting()
nmpc_madnlp_ss = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt, optim, transcription)
nmpc_madnlp_ss = setconstraint!(nmpc_madnlp_ss; umin, umax)
JuMP.unset_time_limit_sec(nmpc_madnlp_ss.optim) 

optim = JuMP.Model(MadNLP.Optimizer)
transcription = MultipleShooting()
nmpc_madnlp_ms = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt, optim, transcription)
nmpc_madnlp_ms = setconstraint!(nmpc_madnlp_ms; umin, umax)
JuMP.unset_time_limit_sec(nmpc_madnlp_ms.optim)
# TODO: does not work well with MadNLP and MultipleShooting, figure out why. 
# Current theory: MadNLP LBFGS approximation is less robust than Ipopt version.
# Re-test when exact Hessians will be supported in ModelPredictiveControl.jl.
# The following attributes kinda work with the MadNLP LBFGS approximation but super slow:
JuMP.set_attribute(nmpc_madnlp_ms.optim, "hessian_approximation", MadNLP.CompactLBFGS)
MadNLP_QNopt = MadNLP.QuasiNewtonOptions(; max_history=42)
JuMP.set_attribute(nmpc_madnlp_ms.optim, "quasi_newton_options", MadNLP_QNopt)

samples, evals, seconds = 50, 1, 15*60
RUNTIME["PredictiveController"]["Pendulum"]["NonLinMPC"]["Ipopt"]["SingleShooting"] = 
    @benchmarkable(
        sim!($nmpc_ipopt_ss, $N, $ry; plant=$plant, x_0=$x_0, x̂_0=$x̂_0),
        samples=samples, evals=evals, seconds=seconds
    )
RUNTIME["PredictiveController"]["Pendulum"]["NonLinMPC"]["Ipopt"]["MultipleShooting"] =
    @benchmarkable(
        sim!($nmpc_ipopt_ms, $N, $ry; plant=$plant, x_0=$x_0, x̂_0=$x̂_0),
        samples=samples, evals=evals, seconds=seconds
    )
RUNTIME["PredictiveController"]["Pendulum"]["NonLinMPC"]["MadNLP"]["SingleShooting"] = 
    @benchmarkable(
        sim!($nmpc_madnlp_ss, $N, $ry; plant=$plant, x_0=$x_0, x̂_0=$x̂_0),
        samples=samples, evals=evals, seconds=seconds
    )
# TODO: way too slow, samples=1 (just an informative single point), might change this later.
RUNTIME["PredictiveController"]["Pendulum"]["NonLinMPC"]["MadNLP"]["MultipleShooting"] =
    @benchmarkable(
        sim!($nmpc_madnlp_ms, $N, $ry; plant=$plant, x_0=$x_0, x̂_0=$x̂_0),
        samples=1, evals=evals, seconds=seconds 
    )

# ---------------------- Allocation benchmarks ------------------------------------------
empc = ExplicitMPC(linmodel, Mwt=[1, 1], Nwt=[0.1, 0.1], Lwt=[0.1, 0.1])

samples, evals = 1, 1
ALLOC["PredictiveController"]["ExplicitMPC"]["moveinput!"] = 
    @benchmarkable(
        moveinput!($empc, $y, $d),
        setup=preparestate!($empc, $y, $d),
        samples=samples, evals=evals
    )
