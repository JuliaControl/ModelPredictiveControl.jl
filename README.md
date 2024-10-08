# ModelPredictiveControl.jl

[![Build Status](https://github.com/JuliaControl/ModelPredictiveControl.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaControl/ModelPredictiveControl.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/JuliaControl/ModelPredictiveControl.jl/branch/main/graph/badge.svg?token=K4V0L113M4)](https://codecov.io/gh/JuliaControl/ModelPredictiveControl.jl)
[![doc-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaControl.github.io/ModelPredictiveControl.jl/stable)
[![doc-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaControl.github.io/ModelPredictiveControl.jl/dev)

An open source [model predictive control](https://en.wikipedia.org/wiki/Model_predictive_control)
package for Julia.

The package depends on [`ControlSystemsBase.jl`](https://github.com/JuliaControl/ControlSystems.jl)
for the linear systems and [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl) for the solving.

## Installation

To install the `ModelPredictiveControl` package, run this command in the Julia REPL:

```julia
using Pkg; Pkg.add("ModelPredictiveControl")
```

## Getting Started

To construct model predictive controllers (MPCs), we must first specify a plant model that
is typically extracted from input-output data using [system identification](https://github.com/baggepinnen/ControlSystemIdentification.jl).
The model here is linear with one input, two outputs and a large time delay in the first
channel (a transfer function matrix, with $s$ as the Laplace variable):

```math
\mathbf{G}(s) = \frac{\mathbf{y}(s)}{\mathbf{u}(s)} = 
\begin{bmatrix}
    \frac{2e^{-20s}}{10s + 1} \\[3pt]
    \frac{10}{4s +1}
\end{bmatrix}
```

We first construct the plant model with a sample time $T_s = 1$ s:

```julia
using ModelPredictiveControl, ControlSystemsBase
G = [ tf( 2 , [10, 1])*delay(20)
      tf( 10, [4,  1]) ]
Ts = 1.0
model = LinModel(G, Ts)
```

Our goal is controlling the first output $y_1$, but the second one $y_2$ should never exceed
35:

```julia
mpc = LinMPC(model, Mwt=[1, 0], Nwt=[0.1])
mpc = setconstraint!(mpc, ymax=[Inf, 35])
```

The keyword arguments `Mwt` and `Nwt` are the output setpoint tracking and move suppression
weights, respectively. A setpoint step change of five tests `mpc` controller in closed-loop.
The result is displayed with [`Plots.jl`](https://github.com/JuliaPlots/Plots.jl):

```julia
using Plots
ry = [5, 0]
res = sim!(mpc, 40, ry)
plot(res, plotry=true, plotymax=true)
```

![StepChangeResponse](/docs/src/assets/readme_result.svg)

See the [manual](https://JuliaControl.github.io/ModelPredictiveControl.jl/stable/manual/linmpc/)
for more detailed examples.

## Features

### Legend

- [x] implemented feature  
- [ ] planned feature

### Model Predictive Control Features

- [x] linear and nonlinear plant models exploiting multiple dispatch
- [x] model linearization based on automatic differentiation (exact Jacobians)
- [x] supported objective function terms:
  - [x] output setpoint tracking
  - [x] move suppression
  - [x] input setpoint tracking
  - [x] terminal costs
  - [x] economic costs (economic model predictive control)
- [x] adaptive linear model predictive controller
  - [x] manual model modification
  - [x] automatic successive linearization of a nonlinear model
  - [x] objective function weights and covariance matrices modification
- [x] explicit predictive controller for problems without constraint
- [x] online-tunable soft and hard constraints on:
  - [x] output predictions
  - [x] manipulated inputs
  - [x] manipulated inputs increments
  - [x] terminal states to ensure nominal stability
- [ ] custom manipulated input constraints that are a function of the predictions
- [x] supported feedback strategy:
  - [x] state estimator (see State Estimation features)
  - [x] internal model structure with a custom stochastic model
- [x] automatic model augmentation with integrating states for offset-free tracking
- [x] support for unmeasured model outputs
- [x] feedforward action with measured disturbances that supports direct transmission
- [x] custom predictions for:
  - [x] output setpoints
  - [x] measured disturbances
- [x] easy integration with `Plots.jl`
- [x] optimization based on `JuMP.jl`:
  - [x] quickly compare multiple optimizers
  - [x] nonlinear solvers relying on automatic differentiation (exact derivative)
- [x] additional information about the optimum to ease troubleshooting
- [x] real-time control loop features:
  - [x] implementations that carefully limits the allocations
  - [x] simple soft real-time utilities

### State Estimation Features

- [x] supported state estimators/observers:
  - [x] steady-state Kalman filter
  - [x] Kalman filter
  - [x] Luenberger observer
  - [x] internal model structure
  - [x] extended Kalman filter
  - [x] unscented Kalman filter
  - [x] moving horizon estimator
- [x] easily estimate unmeasured disturbances by adding one or more integrators at the:
  - [x] manipulated inputs
  - [x] measured outputs
- [x] bumpless manual to automatic transfer for control with a proper initial estimate
- [x] estimators in two possible forms:
  - [x] filter (or current) form to improve accuracy and robustness
  - [x] predictor (or delayed) form to reduce computational load
- [x] moving horizon estimator in two formulations:
  - [x] linear plant models (quadratic optimization)
  - [x] nonlinear plant models (nonlinear optimization)
- [x] moving horizon estimator online-tunable soft and hard constraints on:
  - [x] state estimates
  - [x] process noise estimates
  - [x] sensor noise estimates
