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

The estimators are all implemented in the current form (a.k.a. as filter form) by default
to improve accuracy and robustness, that is, they all estimates at each discrete time ``k``
the states of the current period ``\mathbf{x̂}_k(k)``[^1] (using the newest measurements, see
[Manual](@ref man_lin) for examples). The predictor form (a.k.a. delayed form) is also
available with the option `direct=false`. This allow moving the estimator computations after
solving the MPC problem with [`moveinput!`](@ref), for when the estimations are expensive
(for instance, with the [`MovingHorizonEstimator`](@ref)).

[^1]: also denoted ``\mathbf{x̂}_{k|k}`` [elsewhere](https://en.wikipedia.org/wiki/Kalman_filter).

!!! info
    The nomenclature in this page introduces the estimated state ``\mathbf{x̂}`` and output
    ``\mathbf{ŷ}`` vectors of respectively `nx̂` and `ny` elements. Also, all the estimators
    support measured ``\mathbf{y^m}`` (`nym` elements) and unmeasured ``\mathbf{y^u}``
    (`nyu` elements) model output, where ``\mathbf{y}`` refers to all of them.

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
