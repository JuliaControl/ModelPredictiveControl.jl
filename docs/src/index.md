# ModelPredictiveControl.jl

An open source [model predictive control](https://en.wikipedia.org/wiki/Model_predictive_control)
package for Julia.

The package depends on [`ControlSystemsBase.jl`](https://github.com/JuliaControl/ControlSystems.jl)
for the linear systems, [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl) for the
optimization and [`DifferentiationInterface.jl`](https://github.com/JuliaDiff/DifferentiationInterface.jl)
for the derivatives.

The objective is to provide a simple, clear and modular framework to quickly design model
predictive controllers (MPCs) in Julia, while preserving the flexibility for advanced
real-time optimization. Modern MPCs based on closed-loop state estimators are the main focus
of the package, but classical approaches that rely on internal models are also possible. The
`JuMP` and `DifferentiationInterface` dependencies allows the user to test different
optimizers and automatic differentiation (AD) backends easily if the performances of the
default settings are not satisfactory.

The documentation is divided in two parts:

- **[Manual](@ref man_lin)** — This section includes step-by-step guides to design
  predictive controllers on multiple case studies.
- **[Functions](@ref func_sim_model)** — Documentation of methods and types exported by the
  package. The "Internals" section provides implementation details of functions that are
  not exported.

## Manual

```@contents
Depth = 2
Pages = [
    "manual/installation.md",
    "manual/linmpc.md",
    "manual/nonlinmpc.md",
    "manual/mtk.md"
]
```

## Functions: Public

```@contents
Depth = 2
Pages = [
    "public/sim_model.md",
    "public/state_estim.md",
    "public/predictive_control.md",
    "public/generic_func.md",
    "public/plot_sim.md",
]
```

## Functions: Internals

```@contents
Depth = 1
Pages = [
    "internals/sim_model.md",
    "internals/state_estim.md",
    "internals/predictive_control.md",
]
```
