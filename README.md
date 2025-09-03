# ModelPredictiveControl.jl

[![Build Status](https://github.com/JuliaControl/ModelPredictiveControl.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaControl/ModelPredictiveControl.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/JuliaControl/ModelPredictiveControl.jl/branch/main/graph/badge.svg?token=K4V0L113M4)](https://codecov.io/gh/JuliaControl/ModelPredictiveControl.jl)
[![doc-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaControl.github.io/ModelPredictiveControl.jl/stable)
[![doc-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaControl.github.io/ModelPredictiveControl.jl/dev)

An open source [model predictive control](https://en.wikipedia.org/wiki/Model_predictive_control)
package for Julia.

The package depends on [`ControlSystemsBase.jl`](https://github.com/JuliaControl/ControlSystems.jl)
for the linear systems, [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl) for the
optimization and [`DifferentiationInterface.jl`](https://github.com/JuliaDiff/DifferentiationInterface.jl)
for the derivatives.

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

### Model Predictive Control Features

- linear and nonlinear plant models exploiting multiple dispatch
- model linearization based on automatic differentiation (exact Jacobians)
- supported objective function terms:
  - output setpoint tracking
  - move suppression
  - input setpoint tracking
  - terminal costs
  - custom economic costs (economic model predictive control)
- control horizon distinct from prediction horizon and custom move blocking
- adaptive linear model predictive controller
  - manual model modification
  - automatic successive linearization of a nonlinear model
  - objective function weights and covariance matrices modification
- explicit predictive controller for problems without constraint
- online-tunable soft and hard constraints on:
  - output predictions
  - manipulated inputs
  - manipulated inputs increments
  - terminal states to ensure nominal stability
- custom nonlinear inequality constraints (soft or hard)
- supported feedback strategy:
  - state estimator (see State Estimation features)
  - internal model structure with a custom stochastic model
- automatic model augmentation with integrating states for offset-free tracking
- support for unmeasured model outputs
- feedforward action with measured disturbances that supports direct transmission
- custom predictions for (or preview):
  - output setpoints
  - measured disturbances
  - input setpoints
- easy integration with `Plots.jl`
- optimization based on `JuMP.jl` to quickly compare multiple optimizers:
  - many quadratic solvers for linear control
  - many nonlinear solvers for nonlinear control (local or global)
- derivatives based on `DifferentiationInterface.jl` to compare different approaches:
  - automatic differentiation (exact solution)
  - symbolic differentiation (exact solution)
  - finite difference (approximate solution)
- supported transcription methods of the optimization problem:
  - direct single shooting
  - direct multiple shooting
  - trapezoidal collocation
- additional information about the optimum to ease troubleshooting
- real-time control loop features:
  - implementations that carefully limits the allocations
  - simple soft real-time utilities

### State Estimation Features

- supported state estimators/observers:
  - steady-state Kalman filter
  - Kalman filter
  - Luenberger observer
  - internal model structure
  - extended Kalman filter
  - unscented Kalman filter
  - moving horizon estimator
- disable built-in observer to manually provide your own state estimate
- easily estimate unmeasured disturbances by adding one or more integrators at the:
  - manipulated inputs
  - measured outputs
- bumpless manual to automatic transfer for control with a proper initial estimate
- estimators in two possible forms:
  - filter (or current) form to improve accuracy and robustness
  - predictor (or delayed) form to reduce computational load
- moving horizon estimator in two formulations:
  - linear plant models (quadratic optimization)
  - nonlinear plant models (nonlinear optimization)
- moving horizon estimator online-tunable soft and hard constraints on:
  - state estimates
  - process noise estimates
  - sensor noise estimates
