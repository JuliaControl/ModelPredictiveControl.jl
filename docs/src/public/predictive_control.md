# Predictive Controller Design

All the predictive controllers in this module rely on a state estimator to compute the
predictions (a.k.a. [full state feedback](https://en.wikipedia.org/wiki/Full_state_feedback)
control).


## PredictiveController functions and types

```@docs
PredictiveController
LinMPC
```

## Generic Functions

```@docs
setconstraint!
```
