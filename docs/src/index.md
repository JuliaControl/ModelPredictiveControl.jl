# ModelPredictiveControl.jl

A [model predictive control](https://en.wikipedia.org/wiki/Model_predictive_control) package
for Julia. package for Julia.

The package depends on [`ControlSystemsBase.jl`](https://github.com/JuliaControl/ControlSystems.jl)
for the linear systems and [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl) for the solvers.

## Contents

```@contents
Pages = [
    "index.md",
    "manual.md",
    "public/sim_model.md",
    "public/state_estim.md",
    "public/predictive_control.md",
    "public/generic_func.md",
    "internals/sim_model.md",
    "internals/state_estim.md",
    "internals/predictive_control.md",
    "func_index.md"
]
```

## Features

### Legend

✅ implemented feature  
⬜ planned feature

### Model Predictive Control Features

- ✅ linear and nonlinear plant models exploiting multiple dispatch
- ⬜ model predictive controllers based on:
  - ✅ linear plant models
  - ⬜ nonlinear plant models
- ⬜ supported objective function terms:
  - ✅ output setpoint tracking
  - ✅ move suppression
  - ✅ input setpoint tracking
  - ⬜ additional custom penalty (e.g. economic costs)
  - ⬜ terminal cost to ensure nominal stability
- ✅ soft and hard constraints on:
  - ✅ output predictions
  - ✅ manipulated inputs
  - ✅ manipulated inputs increments
- ⬜ custom manipulated input constraints that are a function of the predictions
- ✅ supported feedback strategy:
  - ✅ state estimator (see State Estimation features)
  - ✅ internal model structure with a custom stochastic model
- ✅ offset-free tracking with a single or multiple integrators on measured outputs
- ✅ support for unmeasured model outputs
- ✅ feedforward action with measured disturbances that supports direct transmission
- ✅ custom predictions for:
  - ✅ output setpoints
  - ✅ measured disturbances
- ⬜ easy integration with `Plots.jl`
- ✅ optimization based on `JuMP.jl`:
  - ✅ quickly compare multiple optimizers
  - ⬜ nonlinear solvers relying on automatic differentiation (exact derivative)
- ⬜ additional information about the optimum to ease troubleshooting:
  - ✅ optimal input increments over control horizon
  - ✅ slack variable optimum
  - ✅ objective function optimum
  - ✅ output predictions at optimum
  - ✅ current stochastic output predictions
  - ⬜ custom penalty value at optimum

### State Estimation Features

- ⬜ supported state estimators/observers:
  - ✅ steady-state Kalman filter
  - ✅ Kalman filter
  - ⬜ Luenberger observer
  - ✅ internal model structure
  - ✅ unscented Kalman filter
  - ⬜ moving horizon estimator
- ✅ observers in predictor form to ease  control applications
- ⬜ moving horizon estimator that supports:
  - ⬜ inequality state constraints
  - ⬜ zero process noise equality constraint to reduce the problem size
