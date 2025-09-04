# Functions: StateEstimator Internals

```@contents
Pages = ["state_estim.md"]
```

## Estimator Construction

```@docs
ModelPredictiveControl.init_estimstoch
ModelPredictiveControl.init_integrators
ModelPredictiveControl.augment_model
ModelPredictiveControl.init_ukf
ModelPredictiveControl.init_internalmodel
ModelPredictiveControl.init_predmat_mhe
ModelPredictiveControl.relaxarrival
ModelPredictiveControl.relaxX̂
ModelPredictiveControl.relaxŴ
ModelPredictiveControl.relaxV̂
ModelPredictiveControl.init_matconstraint_mhe
ModelPredictiveControl.get_optim_functions(::MovingHorizonEstimator, ::ModelPredictiveControl.GenericModel)
```

## Augmented Model

```@docs
ModelPredictiveControl.f̂!
ModelPredictiveControl.ĥ!
```

## Update Quadratic Optimization

```@docs
ModelPredictiveControl.initpred!(::MovingHorizonEstimator, ::LinModel)
ModelPredictiveControl.linconstraint!(::MovingHorizonEstimator, ::LinModel)
```

## Solve Optimization Problem

```@docs
ModelPredictiveControl.optim_objective!(::MovingHorizonEstimator)
ModelPredictiveControl.set_warmstart_mhe!
ModelPredictiveControl.predict_mhe!
ModelPredictiveControl.con_nonlinprog_mhe!
```

## Remove Operating Points

```@docs
ModelPredictiveControl.remove_op!
```

## Init Estimate

```@docs
ModelPredictiveControl.init_estimate!
```

## Correct Estimate

```@docs
ModelPredictiveControl.correct_estimate!
```

## Update Estimate

!!! info
    All these methods assume that the `u0`, `y0m` and `d0` arguments are deviation vectors
    from their respective operating points (see [`setop!`](@ref)). The associated equations
    in the documentation drops the ``\mathbf{0}`` in subscript to simplify the notation.
    Strictly speaking, the manipulated inputs, measured outputs, measured disturbances and
    estimated states should be denoted with ``\mathbf{u_0, y_0^m, d_0}`` and
    ``\mathbf{x̂_0}``, respectively.

```@docs
ModelPredictiveControl.update_estimate!
```
