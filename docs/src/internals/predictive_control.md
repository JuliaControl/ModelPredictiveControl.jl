# PredictiveController Internals

The prediction methodology of this module is mainly based on Maciejowski textbook [^1].

[^1]: Maciejowski, J. 2000, "Predictive control : with constraints", 1st ed., Prentice Hall,
     ISBN 978-0201398236.

```@docs
ModelPredictiveControl.init_deterpred
ModelPredictiveControl.init_ΔUtoU
ModelPredictiveControl.relaxU
ModelPredictiveControl.relaxΔU
ModelPredictiveControl.relaxŶ
ModelPredictiveControl.init_quadprog
ModelPredictiveControl.init_stochpred
ModelPredictiveControl.init_linconstraint
ModelPredictiveControl.getestimates!
ModelPredictiveControl.initpred!
ModelPredictiveControl.linconstraint!
```
