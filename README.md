# ModelPredictiveControl.jl

[![Build Status](https://github.com/franckgaga/ModelPredictiveControl.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/franckgaga/ModelPredictiveControl.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://franckgaga.github.io/ModelPredictiveControl.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://franckgaga.github.io/ModelPredictiveControl.jl/dev)

A model predictive control package for Julia.

## Installation

To install the `ModelPredictiveControl` package, run this command in the Julia REPL:

```julia
using Pkg; Pkg.add("ModelPredictiveControl")
```

## Features

✅ implemented feature  
⬜ planned feature

### Model Predictive Control

- ✅ linear and nonlinear plant models using a unified structure
- ⬜ model predictive controllers based on a :
  - ✅ linear plant model
  - ⬜ nonlinear plant model
- ⬜ support for linear model predictions using fast matrix algebra in a nonlinear
  controller (e.g. economic cost minimization of a linear plant model)
- ⬜ supported objective function terms :
  - ✅ output setpoint tracking
  - ✅ move suppression
  - ✅ input setpoint tracking
  - ⬜ additional custom penalty (e.g. economic costs)
  - ⬜ terminal cost to ensure nominal stability
- ✅ soft and hard constraints on :
  - ✅ output predictions
  - ✅ manipulated inputs
  - ✅ manipulated inputs increments
- ⬜ custom manipulated input constraints that are a function of the predictions
- ✅ supported feedback strategy :
  - ✅ internal model structure with custom stochastic model
  - ✅ state estimator (see State Estimation features)
- ✅ offset-free tracking with a single or multiple integrators on measured outputs
- ✅ support for unmeasured model outputs
- ✅ feedforward action with measured disturbances that supports direct transmission
- ✅ custom predictions for :
  - ✅ output setpoints
  - ✅ measured disturbances
- ⬜ get additional information about the optimum to ease troubleshooting :
  - ✅ optimal input increments over control horizon
  - ✅ slack variable optimum
  - ✅ objective function optimum
  - ✅ output predictions at optimum
  - ✅ current stochastic output predictions
  - ⬜ custom penalty value at optimum

### State Estimation

- ⬜ supported state estimators/observers :
  - ✅ steady-state Kalman filter
  - ✅ Kalman filter
  - ⬜ Luenberger observer
  - ✅ internal model structure
  - ⬜ unscented Kalman filter
  - ⬜ moving horizon estimator
- ✅ observers in the predictor form to facilitate predictive control applications.
- ⬜ moving horizon estimator that supports :
  - ⬜ inequality state constraints
  - ⬜ equality constraints at zero on process noise (to reduce the problem size)