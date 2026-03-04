# ModelPredictiveControl.jl

[![Build Status](https://github.com/JuliaControl/ModelPredictiveControl.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaControl/ModelPredictiveControl.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/JuliaControl/ModelPredictiveControl.jl/branch/main/graph/badge.svg?token=K4V0L113M4)](https://codecov.io/gh/JuliaControl/ModelPredictiveControl.jl)
[![doc-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaControl.github.io/ModelPredictiveControl.jl/stable)
[![doc-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaControl.github.io/ModelPredictiveControl.jl/dev)
[![arXiv](https://img.shields.io/badge/arXiv-2411.09764-b31b1b.svg)](https://arxiv.org/abs/2411.09764)

An open source [model predictive control](https://en.wikipedia.org/wiki/Model_predictive_control)
package for Julia.

The package depends on [`ControlSystemsBase.jl`](https://github.com/JuliaControl/ControlSystems.jl)
for the linear systems, [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl) for the
optimization and [`DifferentiationInterface.jl`](https://github.com/JuliaDiff/DifferentiationInterface.jl)
for the derivatives.

## 🛠️ Installation

To install the `ModelPredictiveControl` package, run this command in the Julia REPL:

```julia
using Pkg; Pkg.add("ModelPredictiveControl")
```

## 🚀 Getting Started

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

## ✨ Features

### 🎯 Model Predictive Control Features

- 🏭️ **Plant Model**: Linear or nonlinear models exploiting multiple dispatch.
- ⛳️ **Objectives**: Tracking for inputs/outputs, move suppression, terminal costs, and economic costs.
- ⏳️ **Horizons**: Distinct prediction/control horizons with custom move blocking.
- 📸 **Linearization**: Auto-differentiation for exact Jacobians.
- ⚙️ **Adaptive MPC**: Manual model updates or automatic successive linearization.
- 🏎️ **Explicit MPC**: Specialized for unconstrained problems.
- 🚧 **Bounds**: Soft/hard limits on inputs, outputs, increments, and terminal states.
- 🚫 **Contraints**: Soft/hard custom linear and nonlinear inequality constraints.
- 🔁 **Feedback**: Internal model or state estimators (see features below).
- 📡 **Feedforward**: Integrated support for measured disturbances.
- 🔮 **Preview**: Custom predictions for setpoints and measured disturbances.
- 📈 **Offset-Free**: Automatic model augmentation with integrators.
- 📊 **Visuals**: Easy integration with `Plots.jl`.
- 🧩 **Solvers**: Optimization via `JuMP.jl` (quadratic & nonlinear) and derivatives via `DifferentiationInterface.jl`.
- 📝 **Transcription**: Direct single/multiple shooting and trapezoidal/orthogonal collocation.
- 🩺 **Troubleshooting**: Detailed diagnostic information about optimum.
- ⏱️ **Real-Time**: Optimized for low memory allocations with soft real-time utilities.
- 📟️ **Embedded**: Lightweight C code generation via `LinearMPC.jl`

### 🔭 State Estimation Features

- 🔍️ **Estimators**: Many Kalman filters, Luenberger, and Moving Horizon Estimator (MHE).
- 🎛️ **Customization**: Ability to use custom/external state estimates.
- 🌊 **Disturbances**: Estimate unmeasured disturbances via integrators on inputs/outputs.
- 🛣️ **Bumpless Transfer**: Smooth transitions from manual to automatic control.
- ⌚️ **Timing**: Estimators available in filter (current) or predictor (delayed) forms.
- 🏷️ **MHE Types**: Formulations for both linear (quadratic optimization) and nonlinear plants.
- 🛡️ **MHE Constraints**: Tunable soft/hard constraints on state and noise estimates.
