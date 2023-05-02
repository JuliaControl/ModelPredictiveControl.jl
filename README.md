# ModelPredictiveControl.jl

[![Build Status](https://github.com/franckgaga/ModelPredictiveControl.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/franckgaga/ModelPredictiveControl.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![doc-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://franckgaga.github.io/ModelPredictiveControl.jl/stable)
[![coc-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://franckgaga.github.io/ModelPredictiveControl.jl/dev)

A [model predictive control](https://en.wikipedia.org/wiki/Model_predictive_control) package
for Julia.

The package depends on [`ControlSystemsBase.jl`](https://github.com/JuliaControl/ControlSystems.jl)
for the linear systems and [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl) for the solvers.

## Installation

To install the `ModelPredictiveControl` package, run this command in the Julia REPL:

```julia
using Pkg; Pkg.add("ModelPredictiveControl")
```

## Features

### Legend

✅ implemented feature  
⬜ planned feature

### Model Predictive Control Features

- ✅ linear and nonlinear plant models exploiting multiple dispatch
- ✅ model predictive controllers based on:
  - ✅ linear plant models
  - ✅ nonlinear plant models
- ✅ supported objective function terms:
  - ✅ output setpoint tracking
  - ✅ move suppression
  - ✅ input setpoint tracking
  - ✅ economic costs (economic model predictive control)
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
  - ✅ nonlinear solvers relying on automatic differentiation (exact derivative)
- ⬜ additional information about the optimum to ease troubleshooting:
  - ⬜ optimal input increments over control horizon
  - ⬜ slack variable optimum
  - ⬜ objective function optimum
  - ⬜ output predictions at optimum
  - ⬜ current stochastic output predictions
  - ⬜ optimal economic costs

### State Estimation Features

- ⬜ supported state estimators/observers:
  - ✅ steady-state Kalman filter
  - ✅ Kalman filter
  - ⬜ Luenberger observer
  - ✅ internal model structure
  - ⬜ extended Kalman filter
  - ✅ unscented Kalman filter
  - ⬜ moving horizon estimator
- ✅ observers in predictor form to ease  control applications
- ⬜ moving horizon estimator that supports:
  - ⬜ inequality state constraints
  - ⬜ zero process noise equality constraint to reduce the problem size
