# Specifying plant models

The [`SimModel`](@ref) types represents discrete state-space models that can be used to 
construct [`StateEstimator`](@ref) and [`PredictiveController`](@ref) objects, or as plant 
simulators by calling [`evaloutput`](@ref) and [`updatestate!`](@ref) methods on 
[`SimModel`](@ref) objects (to test estimator/controller designs). For simulations, the 
states `x` are stored inside [`SimModel`](@ref) objects. Use [`setstate!`](@ref) method 
to modify them.  

## SimModel functions and types

```@docs
LinModel
NonLinModel
SimModel
setop!
setstate!(::SimModel,::Any)
updatestate!(::SimModel,::Any)
evaloutput(::SimModel)
```