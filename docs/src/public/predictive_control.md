# Predictive Controller Design

```@contents
Pages = ["predictive_control.md"]
```

All the predictive controllers in this module rely on a state estimator to compute the
predictions. The default [`LinMPC`](@ref) estimator is a [`SteadyKalmanFilter`](@ref), and
[`NonLinMPC`](@ref), an `UnscentedKalmanFilter`. For simpler designs, an
[`InternalModel`](@ref) structure is also available, that assumes by default that the
current model mismatch estimation is constant in the future (same approach than dynamic
matrix control, DMC).

## Abstract Types

```@docs
PredictiveController
```

## LinMPC

```@docs
LinMPC
```

## Set Constraint

```@docs
setconstraint!
```

## Move Manipulated Input

```@docs
moveinput!
```
