## ----------------- Unit tests (no allocation) -------------------------------------
const UNIT_MODEL = SUITE["unit tests"]["SimModel"]

samples, evals = 10000, 1
UNIT_MODEL["LinModel"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($linmodel, $u, $d); 
        samples=samples, evals=evals
    )
UNIT_MODEL["LinModel"]["evaloutput"] = 
    @benchmarkable(
        evaloutput($linmodel, $d); 
        samples=samples, evals=evals
    )
UNIT_MODEL["NonLinModel"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($nonlinmodel, $u, $d); 
        samples=samples, evals=evals
    )
UNIT_MODEL["NonLinModel"]["evaloutput"] = 
    @benchmarkable(
        evaloutput($nonlinmodel, $d); 
        samples=samples, evals=evals
    )
UNIT_MODEL["NonLinModel"]["linearize!"] = 
    @benchmarkable(
        linearize!($linmodel, $nonlinmodel); 
        samples=samples, evals=evals
    )

## ----------------- Case studies ---------------------------------------------------
# TODO: Add case study benchmarks for SimModel
