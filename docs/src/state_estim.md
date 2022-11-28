# State estimator design

!!! info 
    All the state estimators support measured ``\mathbf{y^m}`` and unmeasured 
    ``\mathbf{y^u}`` model outputs, where ``\mathbf{y}`` refers to all of them.

## StateEstimator functions and types


```@docs
StateEstimator
```

### InternalModel

```@docs
InternalModel
updatestate!(::InternalModel,::Any, ::Any, ::Any)
evaloutput(::InternalModel, ::Any, ::Any)
```

### Luenburger

### SteadyKalmanFilter

### KalmanFilter

```@docs
KalmanFilter
updatestate!(::KalmanFilter,::Any, ::Any, ::Any)
evaloutput(::KalmanFilter, ::Any)
```

### UnscentedKalmanFilter

### MovingHorizonEstimator

# Advanced Topics

## Internals

```@docs
ModelPredictiveControl.init_internalmodel
ModelPredictiveControl.init_estimstoch
ModelPredictiveControl.augment_model
```