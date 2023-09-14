# ModelPredictiveControl.jl

A [model predictive control](https://en.wikipedia.org/wiki/Model_predictive_control) package
for Julia.

The package depends on [`ControlSystemsBase.jl`](https://github.com/JuliaControl/ControlSystems.jl)
for the linear systems and [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl) for the solving.

The objective is to provide a simple and clear framework to quickly design model predictive
controllers (MPCs) in Julia, while preserving the flexibility for advanced real-time
optimization. Modern MPCs based on closed-loop state estimators are the main focus of the
package, but classical approaches that rely on internal models are also possible. The
`JuMP.jl` interface allows the user to test different solvers easily if the performance of
the default settings is not satisfactory.

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
