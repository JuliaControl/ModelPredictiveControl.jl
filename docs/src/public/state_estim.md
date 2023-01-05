# State Estimator Design

This module includes many state estimators (or state observer), both for deterministic
and stochastic systems. The implementations focus on control applications, that is, relying
on the estimates to compute a [full state feedback](https://en.wikipedia.org/wiki/Full_state_feedback).
(model predictive controllers, for this package). They all incorporates some kind of
integral action by default, since it is generally desired to eleminate the steady-state
error in closed-loop (a.k.a. offset-free tracking).

!!! warning
    The addition of the integral action is not necessarily desired for other applications
    than control (e.g. filter, soft-sensor, etc.). Careful must be taken at the estimator
    construction in such a case, for example, by using `nint_ym=0` to add 0 integrator.

The estimators are all implemented in the predictor form (a.k.a. observer form), that is,
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
