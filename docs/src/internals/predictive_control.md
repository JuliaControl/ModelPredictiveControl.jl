# Functions: PredictiveController Internals

The prediction methodology of this module is mainly based on Maciejowski textbook [^1].

[^1]: Maciejowski, J. 2000, "Predictive control : with constraints", 1st ed., Prentice Hall,
     ISBN 978-0201398236.

## Controller Initialization

```@docs
ModelPredictiveControl.init_deterpred
ModelPredictiveControl.init_ΔUtoU
ModelPredictiveControl.init_quadprog
ModelPredictiveControl.init_stochpred
ModelPredictiveControl.init_linconstraint
```

## Constraint Relaxation

```@docs
ModelPredictiveControl.relaxU
ModelPredictiveControl.relaxΔU
ModelPredictiveControl.relaxŶ
```

## Get Estimates

```@docs
ModelPredictiveControl.getestimates!
```

## Predictions

```@docs
ModelPredictiveControl.initpred!
```

## Constraints

```@docs
ModelPredictiveControl.linconstraint!
```
