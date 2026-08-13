# Functions: StateEstimator Internals

```@contents
Pages = ["state_estim.md"]
```

Similarly to [Function: Predictive Controllers](@ref func_predictive_control) page, the
various data windows of the [`MovingHorizonEstimator`](@ref) are explicitly defined here
as a preamble, to help the development of its internals. Note that they are not needed for
the other [`StateEstimator`](@ref) types.

At the ``k``th control period, the vectors that encompass the historical deviation values of
the manipulated input ``\mathbf{u_0}``, estimated state ``\mathbf{x̂_0}`` and measured
output ``\mathbf{y_0^m}`` over the window length ``N_k`` are:

```math
    \mathbf{U_0} = \begin{bmatrix}
        \mathbf{u_0}(k-N_k+p+0)   \\ \mathbf{u_0}(k-N_k+p+1)   \\ \vdots  \\ \mathbf{u_0}(k+p-1)
    \end{bmatrix} \: , \quad
    \mathbf{X̂_0} = \begin{bmatrix}
        \mathbf{x̂_0}(k-N_k+p+1)   \\ \mathbf{x̂_0}(k-N_k+p+2)   \\ \vdots  \\ \mathbf{x̂_0}(k+p)
    \end{bmatrix} \quad \text{and} \quad
    \mathbf{Y_0^m} = \begin{bmatrix}
        \mathbf{y_0^m}(k-N_k+1)   \\ \mathbf{y_0^m}(k-N_k+2)   \\ \vdots  \\ \mathbf{y_0^m}(k)
    \end{bmatrix} 
```

in which ``\mathbf{U_0}``, ``\mathbf{X̂_0}`` and  ``\mathbf{Y_0^m}`` are vectors of `nu*Nk`,
`nx̂*Nk` and `nym*Nk` elements, respectively. Notice that ``\mathbf{U_0}`` and
``\mathbf{X̂_0}`` vectors are always shifted by one time step. Additionally, ``\mathbf{U_0}``
and ``\mathbf{Y_0^m}`` are aligned only if ``p=1`` (`direct=false`). Lastly it is worth
noting that the arrival state estimate ``\mathbf{x̂_0}(k-N_k+p)`` is left out of the
``\mathbf{X̂_0}`` vector. The historical deviation values of the measured disturbance
``\mathbf{d_0}`` always include one additional data point compared to the other windows:

```math
    \mathbf{D_0} = \begin{bmatrix}
        \mathbf{d_0}(k-N_k+0)   \\ \mathbf{d_0}(k-N_k+1)   \\ \vdots  \\ \mathbf{d_0}(k) 
    \end{bmatrix}
```

See the Extended Help of the [`MovingHorizonEstimator`](@ref) for the definition of the
with the estimated process noises ``\mathbf{Ŵ}`` and sensor noises ``\mathbf{V̂}`` windows.

## Estimator Construction

```@docs
ModelPredictiveControl.init_estimstoch
ModelPredictiveControl.init_integrators
ModelPredictiveControl.augment_model
ModelPredictiveControl.init_ukf
ModelPredictiveControl.init_internalmodel
ModelPredictiveControl.init_ZtoŴ
ModelPredictiveControl.init_predmat_mhe
ModelPredictiveControl.init_defectmat_mhe
ModelPredictiveControl.relaxarrival
ModelPredictiveControl.relaxX̂
ModelPredictiveControl.relaxŴ
ModelPredictiveControl.relaxV̂
ModelPredictiveControl.init_matconstraint_mhe
ModelPredictiveControl.get_nonlinobj_op(::MovingHorizonEstimator, ::ModelPredictiveControl.GenericModel)
ModelPredictiveControl.get_nonlincon_oracle(::MovingHorizonEstimator, ::ModelPredictiveControl.GenericModel)
```

## Augmented Model

```@docs
ModelPredictiveControl.f̂!
ModelPredictiveControl.ĥ!
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

## Update Quadratic Optimization

```@docs
ModelPredictiveControl.initpred!(::MovingHorizonEstimator, ::LinModel)
ModelPredictiveControl.linconstraint!(::MovingHorizonEstimator, ::LinModel, ::TranscriptionMethod)
```

## Solve Optimization Problem

```@docs
ModelPredictiveControl.optim_objective!(::MovingHorizonEstimator)
ModelPredictiveControl.set_warmstart_mhe!
ModelPredictiveControl.disturbedinput!
ModelPredictiveControl.predict_mhe!
ModelPredictiveControl.con_nonlinprog_mhe!
ModelPredictiveControl.con_nonlinprogeq_mhe!
ModelPredictiveControl.getstate!
```
