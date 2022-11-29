# State estimator design

This package includes many state estimators (or state observer), both for deterministic
and stochastic systems. 

The estimator are all implemented in the predictor form (a.k.a. observer form), that is, 
they all estimates at each discrete time ``k`` the states of the next period 
``\mathbf{x̂}_k(k+1)``. This form comes in handy for control applications since the 
estimations do not delay the controller computations. In contrast, the filter form that
estimates ``\mathbf{x̂}_k(k)`` is sometimes slightly more accurate.

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