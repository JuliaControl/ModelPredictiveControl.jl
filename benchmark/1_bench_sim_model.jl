## ----------------------------------------------------------------------------------------
## ----------------- UNIT TESTS  ----------------------------------------------------------
## ----------------------------------------------------------------------------------------
const UNIT_MODEL = SUITE["UNIT TESTS"]["SimModel"]

UNIT_MODEL["LinModel"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($linmodel, $u, $d); 
    )
UNIT_MODEL["LinModel"]["evaloutput"] = 
    @benchmarkable(
        evaloutput($linmodel, $d); 
    )
UNIT_MODEL["NonLinModel"]["updatestate!"] = 
    @benchmarkable(
        updatestate!($nonlinmodel, $u, $d); 
    )
UNIT_MODEL["NonLinModel"]["evaloutput"] = 
    @benchmarkable(
        evaloutput($nonlinmodel, $d); 
    )
UNIT_MODEL["NonLinModel"]["linearize!"] = 
    @benchmarkable(
        linearize!($linmodel, $nonlinmodel); 
    )

## ----------------------------------------------------------------------------------------
## ----------------- CASE STUDIES ---------------------------------------------------------
## ----------------------------------------------------------------------------------------
const CASE_MODEL = SUITE["CASE STUDIES"]["SimModel"]
# TODO: Add case study benchmarks for SimModel