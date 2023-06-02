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

## Getting Started

To construct model predictive controllers, we must first specify a plant model that is
typically extracted from input-output data using [system identification](https://github.com/baggepinnen/ControlSystemIdentification.jl).
The model here is linear with one input, two outputs and a large time delay in the first
channel:

```math
\mathbf{G}(s) = \frac{\mathbf{y}(s)}{\mathbf{u}(s)} = 
\begin{bmatrix}
    \frac{2e^{-20s}}{10s + 1} \\[3pt]
    \frac{10}{4s +1}
\end{bmatrix}
```

We first construct the plant model with a sample time ``T_s = 1`` s:

```julia
using ModelPredictiveControl, ControlSystemsBase
sys = [
    tf( 2 , [10, 1])*delay(20)
    tf( 10, [4,  1])
]
Ts = 1.0
model = LinModel(sys, Ts)
```

Our goal is controlling the first output, but the second one should never exceed 35:

```julia
mpc = LinMPC(model, Mwt=[1, 0], Nwt=[0.1], Hp=30, Hc=2)
mpc = setconstraint!(mpc, ŷmax=[Inf, 35])
```

The keyword arguments `Mwt` and `Nwt` are the setpoint tracking and move suppression
weights, respectively. We can now test `mpc` controller with a setpoint step change and
display the result using [`Plots.jl`](https://github.com/JuliaPlots/Plots.jl):

```julia
using Plots
ry = [5, 0]
plot(sim!(mpc, 40, ry), plotry=true, plotŷmax=true)
```

![StepChangeResponse](/example/readme_result.svg)

See the [manual](https://franckgaga.github.io/ModelPredictiveControl.jl/stable/manual/) for
more detailed examples.

## Features

### Legend

✅ implemented feature  
⬜ planned feature

### Model Predictive Control Features

- ✅ linear and nonlinear plant models exploiting multiple dispatch
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
- ✅ easy integration with `Plots.jl`
- ✅ optimization based on `JuMP.jl`:
  - ✅ quickly compare multiple optimizers
  - ✅ nonlinear solvers relying on automatic differentiation (exact derivative)
- ✅ additional information about the optimum to ease troubleshooting

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
