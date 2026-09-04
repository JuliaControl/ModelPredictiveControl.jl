# Functions: SimModel Internals

```@contents
Pages = ["sim_model.md"]
```

## Abstract Types

```@docs
ModelPredictiveControl.SimModelODE
ModelPredictiveControl.SimModelDAE
```

## Model Construction

```@docs
ModelPredictiveControl.init_defectmat_dae
```

## State-Space Functions

```@docs
ModelPredictiveControl.f!
ModelPredictiveControl.h!
```

## Steady-State Calculation

```@docs
ModelPredictiveControl.steadystate!
```
