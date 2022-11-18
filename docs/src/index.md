# ModelPredictiveControl.jl Documentation

```@contents
```

# Tutorial


# Specifying models

adsasd

## SimModel functions

```@docs
LinModel
NonLinModel
setop!
updatestate!
evaloutput
```

# State estimator design

!!! info 
    All the state estimators support measured ``\mathbf{y^m}`` and unmeasured 
    ``\mathbf{y^u}`` model outputs, where ``\mathbf{y}`` refers to all of them.

## StateEstimator functions

```@docs
InternalModel
```

# Advanced Topics

## Internals

```@docs
ModelPredictiveControl.init_internalmodel
```


# API

```@index
```