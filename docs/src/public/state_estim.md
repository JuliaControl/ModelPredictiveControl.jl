# State Estimator Design

This package includes many state estimators (or state observer), both for deterministic
and stochastic systems.

The estimator are all implemented in the predictor form (a.k.a. observer form), that is,
they all estimates at each discrete time ``k`` the states of the next period
``\mathbf{x̂}_k(k+1)``. This form comes in handy for control applications since the
estimations come after the controller computations, without introducing any additional
delays. In contrast, the filter form that estimates ``\mathbf{x̂}_k(k)`` is sometimes
slightly more accurate.

!!! info
    All the state estimators support measured ``\mathbf{y^m}`` and unmeasured
    ``\mathbf{y^u}`` model outputs, where ``\mathbf{y}`` refers to all of them.

## Abstract Types

```@docs
StateEstimator
```

## SteadyKalmanFilter

```@docs
SteadyKalmanFilter
```

## KalmanFilter

```@docs
KalmanFilter
```

## InternalModel

```@docs
InternalModel
```

## Luenberger

## UnscentedKalmanFilter

## MovingHorizonEstimator

## Generic Functions

```@docs
initstate!
```
