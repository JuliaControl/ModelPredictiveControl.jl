# Functions: PredictiveController Internals

```@contents
Pages = ["predictive_control.md"]
```

The prediction methodology of this module is mainly based on Maciejowski textbook [^1].

[^1]: Maciejowski, J. 2000, "Predictive control : with constraints", 1st ed., Prentice Hall,
     ISBN 978-0201398236.

## Controller Construction

```@docs
ModelPredictiveControl.init_predmat
ModelPredictiveControl.init_ΔUtoU
ModelPredictiveControl.init_quadprog
ModelPredictiveControl.init_stochpred
ModelPredictiveControl.init_matconstraint_mpc
```

## Constraint Relaxation

```@docs
ModelPredictiveControl.relaxU
ModelPredictiveControl.relaxΔU
ModelPredictiveControl.relaxŶ
ModelPredictiveControl.relaxterminal
```

## Constraints

```@docs
ModelPredictiveControl.linconstraint!(::PredictiveController, ::LinModel)
```
