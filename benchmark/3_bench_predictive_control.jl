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
