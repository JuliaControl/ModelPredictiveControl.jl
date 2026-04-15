# Functions: Direct Transcription Methods

```@contents
Pages = ["transcription.md"]
```

This page contains the documentation of the direct transcription methods used to
construct [`MovingHorizonEstimator`](@ref), [`LinMPC`](@ref) and [`NonLinMPC`](@ref) types.
They represent ways to discretize the continuous-time optimal control or estimation problem
into a finite-dimensional quadratic (QP) or nonlinear program (NLP), which can then be
solved using an appropriate optimizer.

## TranscriptionMethod

```@docs
ModelPredictiveControl.TranscriptionMethod
```

## SingleShooting

```@docs
SingleShooting
```

## MultipleShooting

```@docs
MultipleShooting
```

## TrapezoidalCollocation

```@docs
TrapezoidalCollocation
```

## OrthogonalCollocation

```@docs
OrthogonalCollocation
```
