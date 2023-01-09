# Specifying Plant Models

```@contents
Pages = ["sim_model.md"]
```

The [`SimModel`](@ref) types represents discrete state-space models that can be used to
construct [`StateEstimator`](@ref) and [`PredictiveController`](@ref) objects, or as plant
simulators by calling [`evaloutput`](@ref) and [`updatestate!`](@ref) methods on
[`SimModel`](@ref) objects (to test estimator/controller designs). For time simulations, the
states `x` are stored inside [`SimModel`](@ref) objects. Use [`setstate!`](@ref) method
to manually modify them.

## Abstract Types

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

## Generic Functions

```@docs
setop!
```
