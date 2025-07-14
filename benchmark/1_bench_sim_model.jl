linmodel = setop!(LinModel(sys, Ts, i_d=[3]), uop=[10, 50], yop=[50, 30], dop=[5])
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
