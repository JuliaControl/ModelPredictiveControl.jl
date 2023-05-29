# Plant Models

```@contents
Pages = ["sim_model.md"]
```

The [`SimModel`](@ref) types represents discrete state-space models that can be used to
construct [`StateEstimator`](@ref)s and [`PredictiveController`](@ref)s, or as plant
simulators by calling [`evaloutput`](@ref) and [`updatestate!`](@ref) methods on
[`SimModel`](@ref) instances (to test estimator/controller designs). For time simulations,
the states `x` are stored inside [`SimModel`](@ref) instances. Use [`setstate!`](@ref)
method to manually modify them.

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

## Set Operating Points

```@docs
setop!
```
