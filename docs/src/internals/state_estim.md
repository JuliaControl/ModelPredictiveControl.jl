# Functions: StateEstimator Internals

```@contents
Pages = ["state_estim.md"]
```

## Augmented Model

```@docs
ModelPredictiveControl.f̂
ModelPredictiveControl.ĥ
```

## Estimator Construction

```@docs
ModelPredictiveControl.init_estimstoch
ModelPredictiveControl.init_integrators
ModelPredictiveControl.augment_model
ModelPredictiveControl.init_ukf
ModelPredictiveControl.init_internalmodel
ModelPredictiveControl.init_predmat_mhe
ModelPredictiveControl.init_matconstraint_mhe
```

## Constraint Relaxation

```@docs
ModelPredictiveControl.relaxarrival
ModelPredictiveControl.relaxX̂
ModelPredictiveControl.relaxŴ
ModelPredictiveControl.relaxV̂
```

## Constraints

```@docs
ModelPredictiveControl.linconstraint!(::MovingHorizonEstimator, ::LinModel)
```

## Evaluate Estimated Output

```@docs
ModelPredictiveControl.evalŷ
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

## Init Estimate

!!! info
    Same as above: the arguments should be called `u0`, `ym0` and `d0`, strickly speaking.

```@docs
ModelPredictiveControl.init_estimate!
```
