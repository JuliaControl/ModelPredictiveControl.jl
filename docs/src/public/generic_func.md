# Functions: Generic Functions

```@contents
Pages = ["generic_func.md"]
```

This page contains the documentation of functions that are generic to [`SimModel`](@ref),
[`StateEstimator`](@ref) and [`PredictiveController`](@ref) types.

## Set Constraint

```@docs
setconstraint!
```

## Evaluate Output y

```@docs
evaloutput
```

## Change State x

### Prepare State x

```@docs
preparestate!
```

### Update State x

```@docs
updatestate!
```

### Init State x

```@docs
initstate!
```

### Set State x

```@docs
setstate!
```

## Set Model and Weights

```@docs
setmodel!
```

## Get Additional Information

```@docs
getinfo
```

## Simulate/Control in Real-Time

!!!danger "Disclaimer"
    These utilities are for soft real-time applications. They are not suitable for hard
    real-time environnement like safety-critical processes.

### Save current time t

```@docs
savetime!
```

### Period Sleep

```@docs
periodsleep
```
