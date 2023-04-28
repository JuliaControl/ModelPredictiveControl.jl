# Predictive Controllers

```@contents
Pages = ["predictive_control.md"]
```

All the predictive controllers in this module rely on a state estimator to compute the
predictions. The default [`LinMPC`](@ref) estimator is a [`SteadyKalmanFilter`](@ref), and
[`NonLinMPC`](@ref), an [`UnscentedKalmanFilter`](@ref). For simpler and more classical
designs, an [`InternalModel`](@ref) structure is also available, that assumes by default
that the current model mismatch estimation is constant in the future (same approach than
dynamic matrix control, DMC).

!!! info
    The nomenclature uses hats for the predictions (or estimations, for the state
    estimators) e.g. ``\mathbf{Ŷ}`` encompasses the future model outputs ``\mathbf{y}`` over
    the prediction horizon ``H_p``.

## PredictiveController

```@docs
PredictiveController
```

## LinMPC

```@docs
LinMPC
```

## NonLinMPC

```@docs
NonLinMPC
```

## Set Constraint

```@docs
setconstraint!
```

## Move Manipulated Input

```@docs
moveinput!
```
