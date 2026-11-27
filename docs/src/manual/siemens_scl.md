# Siemens SCL example {#manual_scl}

The downloadable [CSTR SCL source](../assets/CSTR_LinMPC_S7-1515F.scl) is a fixed-size
example of the linear CSTR controller described in [Linear Design](@ref man_lin). It combines
the augmented steady-state Kalman observer and a condensed `Hp=10`, `Hc=2` linear MPC using
`Ts=2 s`, `Mwt=[1,1]`, `Nwt=[0.1,0.1]`, and the soft output bound `y₁ ≥ 48`. The controller
uses a bounded-iteration ADMM QP solver written in SCL; it is **not DAQP** and is not expected
to reproduce DAQP bit-for-bit.

Import the source in TIA Portal and create the optimized instance DB for
`FB_CSTR_LinMPC_S7`. Supply measurements, measured disturbance, setpoint, and an initialization
pulse with the initial state/control. Measurements, disturbance, setpoint, and controls use the
example's absolute CSTR units; the initialized observer state is the six-state estimate in the
example model. `SolverOK` is false unless the QP solver converges within its iteration limit; on
failure the block holds the last control while continuing the observer prediction.

This is a process-control example, not a safety function. Do not use it to implement or replace
F-CPU safety logic. The SCL has not been compiled or validated in TIA Portal or on an S7-1515F;
verify syntax, execution time, numerical behavior, limits, and fail-state behavior on the
intended hardware before any operational use.
