# [Functions: Plant Models](@id func_sim_model)

```@contents
Pages = ["sim_model.md"]
```

The [`SimModel`](@ref) types represents discrete state-space models that can be used to
construct [`StateEstimator`](@ref)s and [`PredictiveController`](@ref)s, or as plant
simulators by calling [`evaloutput`](@ref) and [`updatestate!`](@ref) methods on
[`SimModel`](@ref) instances (to test estimator/controller designs). For time simulations,
the states ``\mathbf{x}`` are stored inside [`SimModel`](@ref) instances. Use [`setstate!`](@ref)
method to manually modify them.

!!! info
    The nomenclature in this page introduces the model manipulated input ``\mathbf{u}``,
    measured disturbances ``\mathbf{d}``, state ``\mathbf{x}`` and output ``\mathbf{y}``,
    four vectors of `nu`, `nd`, `nx` and `ny` elements, respectively. The ``\mathbf{s}``
    vector combines the elements of ``\mathbf{u}`` and ``\mathbf{d}``.

## SimModel

```@docs
SimModel
```

## LinModel

```@docs
LinModel
```

## NonLinModel

```@docs
NonLinModel
```

## Set Variable Names

```@docs
setname!
```

## Set Operating Points

```@docs
setop!
```

## Linearize

```@docs
linearize
linearize!
```

## Differential Equation Solvers

### DiffSolver

```@docs
ModelPredictiveControl.DiffSolver
```

### RungeKutta

```@docs
RungeKutta
```

### ForwardEuler

```@docs
ForwardEuler
```
