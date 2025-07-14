## ----------------- Runtime benchmarks ---------------------------------------------
# TODO: Add runtime benchmarks for SimModel


## ----------------- Allocation benchmarks ------------------------------------------
samples, evals = 1, 1
ALLOC["SimModel"]["LinModel"]["updatestate!"] = @benchmarkable(
    updatestate!($linmodel, $u, $d); samples=samples, evals=evals
)
ALLOC["SimModel"]["LinModel"]["evaloutput"] = @benchmarkable(
    evaloutput($linmodel, $d); samples=samples, evals=evals
)
ALLOC["SimModel"]["NonLinModel"]["updatestate!"] = @benchmarkable(
    updatestate!($nonlinmodel, $u, $d); samples=samples, evals=evals
)
ALLOC["SimModel"]["NonLinModel"]["evaloutput"] = @benchmarkable(
    evaloutput($nonlinmodel, $d); samples=samples, evals=evals
)
ALLOC["SimModel"]["NonLinModel"]["linearize!"] = @benchmarkable(
    linearize!($linmodel, $nonlinmodel); samples=samples, evals=evals
)
