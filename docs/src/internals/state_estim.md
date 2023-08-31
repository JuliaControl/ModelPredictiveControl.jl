# Functions: StateEstimator Internals

## Estimator Initialization

```@docs
ModelPredictiveControl.init_estimstoch
ModelPredictiveControl.augment_model
ModelPredictiveControl.default_nint
ModelPredictiveControl.init_ukf
ModelPredictiveControl.init_internalmodel
```

## Augmented Model

```@docs
ModelPredictiveControl.f̂
ModelPredictiveControl.ĥ
```

## Remove Operating Points

```@docs
ModelPredictiveControl.remove_op!
```

## Update Estimate

!!! info
    All these methods assume that the operating points are already removed in `u`, `ym`
    and `d` arguments. Strickly speaking, the arguments should be called `u0`, `ym0` and
    `d0`, following [`setop!`](@ref) notation. The `0` is dropped to simplify the notation.

```@docs
ModelPredictiveControl.update_estimate!
```
