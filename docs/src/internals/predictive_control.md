# Functions: PredictiveController Internals

```@contents
Pages = ["predictive_control.md"]
```

The prediction methodology of this module is mainly based on Maciejowski textbook [^1].

[^1]: Maciejowski, J. 2000, "Predictive control : with constraints", 1st ed., Prentice Hall,
     ISBN 978-0201398236.

## Controller Construction

```@docs
ModelPredictiveControl.init_ΔUtoU
ModelPredictiveControl.init_predmat
ModelPredictiveControl.init_defectmat
ModelPredictiveControl.relaxU
ModelPredictiveControl.relaxΔU
ModelPredictiveControl.relaxŶ
ModelPredictiveControl.relaxterminal
ModelPredictiveControl.init_quadprog
ModelPredictiveControl.init_stochpred
ModelPredictiveControl.init_matconstraint_mpc
```

## Update Quadratic Optimization

```@docs
ModelPredictiveControl.initpred!(::PredictiveController, ::LinModel, ::Any, ::Any, ::Any, ::Any)
ModelPredictiveControl.linconstraint!(::PredictiveController, ::LinModel)
```

## Solve Optimization Problem

```@docs
ModelPredictiveControl.optim_objective!(::PredictiveController)
ModelPredictiveControl.getinput
```
