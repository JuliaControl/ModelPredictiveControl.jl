# Functions: State Estimators

```@contents
Pages = ["state_estim.md"]
```

This module includes many state estimators (or state observer), both for deterministic
and stochastic systems. The implementations focus on control applications, that is, relying
on the estimates to compute a [full state feedback](https://en.wikipedia.org/wiki/Full_state_feedback)
(predictive controllers, in this package). They all incorporates some kind of
integral action by default, since it is generally desired to eliminate the steady-state
error with closed-loop control (offset-free tracking).

!!! warning
    If you plan to use the estimators for other contexts than this specific package (e.g. :
    filter, parameter estimation, etc.), careful must be taken at construction since the
    integral action is not necessarily desired. The options `nint_u=0` and `nint_ym=0`
    disable it.

The estimators are all implemented in the predictor form (a.k.a. observer form), that is,
they all estimates at each discrete time ``k`` the states of the next period
``\mathbf{x̂}_k(k+1)``[^1]. In contrast, the filter form that estimates ``\mathbf{x̂}_k(k)``
is sometimes slightly more accurate.

[^1]: also denoted ``\mathbf{x̂}_{k+1|k}`` [elsewhere](https://en.wikipedia.org/wiki/Kalman_filter).

The predictor form comes in handy for control applications since the estimations come after
the controller computations, without introducing any additional delays. Moreover, the
[`moveinput!`](@ref) method of the predictive controllers does not automatically update the
estimates with [`updatestate!`](@ref). This allows applying the calculated inputs on the
real plant before starting the potentially expensive estimator computations (see
[Manual](@ref man_lin) for examples).

!!! info
    All the estimators support measured ``\mathbf{y^m}`` and unmeasured ``\mathbf{y^u}``
    model outputs, where ``\mathbf{y}`` refers to all of them.

## StateEstimator

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

## Luenberger

```@docs
Luenberger
```

## UnscentedKalmanFilter

```@docs
UnscentedKalmanFilter
```

## ExtendedKalmanFilter

```@docs
ExtendedKalmanFilter
```

## MovingHorizonEstimator

```@docs
MovingHorizonEstimator
```

## InternalModel

```@docs
InternalModel
```

## Default Model Augmentation

```@docs
default_nint
```
