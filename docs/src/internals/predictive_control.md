# Functions: PredictiveController Internals

```@contents
Pages = ["predictive_control.md"]
```

The prediction methodology of this module is mainly based on Maciejowski textbook [^1].

[^1]: Maciejowski, J. 2000, "Predictive control : with constraints", 1st ed., Prentice Hall,
     ISBN 978-0201398236.

## Controller Construction

```@docs
ModelPredictiveControl.init_ZtoΔU   
ModelPredictiveControl.init_ZtoU
ModelPredictiveControl.init_predmat
ModelPredictiveControl.init_defectmat
ModelPredictiveControl.relaxU
ModelPredictiveControl.relaxΔU
ModelPredictiveControl.relaxŶ
ModelPredictiveControl.relaxterminal
ModelPredictiveControl.init_quadprog
ModelPredictiveControl.init_stochpred
ModelPredictiveControl.init_matconstraint_mpc
ModelPredictiveControl.init_nonlincon!
ModelPredictiveControl.get_optim_functions(::NonLinMPC, ::JuMP.GenericModel)
```

## Update Quadratic Optimization

```@docs
ModelPredictiveControl.initpred!(::PredictiveController, ::LinModel, ::Any, ::Any, ::Any, ::Any)
ModelPredictiveControl.linconstraint!(::PredictiveController, ::LinModel, ::TranscriptionMethod)
ModelPredictiveControl.linconstrainteq!
```

## Solve Optimization Problem

```@docs
ModelPredictiveControl.optim_objective!(::PredictiveController)
ModelPredictiveControl.set_warmstart!
ModelPredictiveControl.getinput
```
