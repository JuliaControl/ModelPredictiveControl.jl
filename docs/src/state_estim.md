# State estimator design

!!! info 
    All the state estimators support measured ``\mathbf{y^m}`` and unmeasured 
    ``\mathbf{y^u}`` model outputs, where ``\mathbf{y}`` refers to all of them.

## StateEstimator functions

### InternalModel

```@docs
InternalModel
updatestate!(::InternalModel,::Any, ::Any, ::Any)
evaloutput(::InternalModel, ::Any, ::Any)
```

### Asymptotic Kalman filter

### Kalman filter

# Advanced Topics

## Internals

```@docs
ModelPredictiveControl.init_internalmodel
```