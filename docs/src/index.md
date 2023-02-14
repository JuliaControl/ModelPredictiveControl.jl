# ModelPredictiveControl.jl

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

### Model Predictive Control

- linear and nonlinear plant models using a unified structure and multiple dispatch
- support for linear model predictions based on matrix algebra in a nonlinear controller
  (e.g. economic optimization of a linear process model)
- supported criterion terms : output setpoint tracking, move suppression, input setpoint
  tracking and additional custom penalty (e.g. economic costs)
- constraints on output predictions, manipulated inputs and manipulated inputs increments
  that all support softening
- custom manipulated input constraints that are a function of the predictions
- supported feedback strategy : internal model structure with custom stochastic model or
  state estimator (see State Estimation features)
- offset-free tracking with a single or multiple integrators on each measured output
- support for unmeasured model outputs
- feedforward action with measured disturbances that supports direct transmission
- custom predictions for output setpoints and measured disturbances
- get additional information about the optimal input to ease troubleshooting : optimal
  output predictions, slack variable optimum, optimal input increments over control horizon,
  objective function minimum, custom penalty optimum, etc.

### State Estimation

- supported state estimators/observers :
  - steady-state Kalman filter
  - Kalman filter
  - Luenberger observer
  - internal model structure
  - unscented Kalman filter
  - moving horizon estimator
- observers in the predictor form to facilitate predictive control applications.
- moving horizon estimator with inequality state constraints, and equality constraints at 
  zero on process noise (to reduce the problem size)
