# ModelPredictiveControl.jl Documentation

```@docs
LinModel
NonLinModel
setop!
updatestate
evaloutput
```

All the state estimators support measured ``\mathbf{y^m}`` and unmeasured ``\mathbf{y^u}``
outputs, where ``\mathbf{y}`` refers to all of them.

```@docs
InternalModel
init_internalmodel
```