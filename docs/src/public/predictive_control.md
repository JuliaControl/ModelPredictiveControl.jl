# Functions: Predictive Controllers

```@contents
Pages = ["predictive_control.md"]
```

All the predictive controllers in this module rely on a state estimator to compute the
predictions. The default [`LinMPC`](@ref) estimator is a [`SteadyKalmanFilter`](@ref), and
[`NonLinMPC`](@ref) with nonlinear models, an [`UnscentedKalmanFilter`](@ref). For simpler
and more classical designs, an [`InternalModel`](@ref) structure is also available, that
assumes by default that the current model mismatch estimation is constant in the future
(same approach as dynamic matrix control, DMC).

!!! info
    The nomenclature uses boldfaces for vectors or matrices, capital boldface letters for
    vectors representing time series (and also for matrices), and hats for the predictions
    (and also for the observer estimations).

To be precise, at the ``k``th control period, the vectors that encompass the future measured
disturbances ``\mathbf{d̂}``, model outputs ``\mathbf{ŷ}`` and setpoints ``\mathbf{r̂_y}``
over the prediction horizon ``H_p`` are defined as:

```math
    \mathbf{D̂} = \begin{bmatrix}
        \mathbf{d̂}(k+1)   \\ \mathbf{d̂}(k+2)   \\ \vdots  \\ \mathbf{d̂}(k+H_p)
    \end{bmatrix} \: , \quad
    \mathbf{Ŷ} = \begin{bmatrix}
        \mathbf{ŷ}(k+1)   \\ \mathbf{ŷ}(k+2)   \\ \vdots  \\ \mathbf{ŷ}(k+H_p)
    \end{bmatrix} \quad \text{and} \quad
    \mathbf{R̂_y} = \begin{bmatrix}
        \mathbf{r̂_y}(k+1) \\ \mathbf{r̂_y}(k+2) \\ \vdots  \\ \mathbf{r̂_y}(k+H_p)
    \end{bmatrix}
```

in which ``\mathbf{D̂}``, ``\mathbf{Ŷ}`` and  ``\mathbf{R̂_y}`` are vectors of `nd*Hp`, `ny*Hp`
and `ny*Hp` elements, respectively. The vectors for the manipulated input ``\mathbf{u}``
are shifted by one time step:

```math
    \mathbf{U} = \begin{bmatrix}
        \mathbf{u}(k+0)   \\ \mathbf{u}(k+1)   \\ \vdots  \\ \mathbf{u}(k+H_p-1)
    \end{bmatrix} \quad \text{and} \quad
    \mathbf{R̂_u} = \begin{bmatrix}
        \mathbf{r̂_u}(k+0) \\ \mathbf{r̂_u}(k+1) \\ \vdots  \\ \mathbf{r̂_u}(k+H_p-1)
    \end{bmatrix}
```

in which ``\mathbf{U}`` and ``\mathbf{R̂_u}`` are both vectors of `nu*Hp` elements. Defining
the manipulated input increment as ``\mathbf{Δu}(k+j) = \mathbf{u}(k+j) - \mathbf{u}(k+j-1)``,
the control horizon ``H_c`` enforces that ``\mathbf{Δu}(k+j) = \mathbf{0}`` for ``j ≥ H_c``.
For this reason, the vector that collects them is truncated up to ``k+H_c-1`` (without
custom move blocking):

```math
    \mathbf{ΔU} =
    \begin{bmatrix}
        \mathbf{Δu}(k+0) \\ \mathbf{Δu}(k+1) \\ \vdots  \\ \mathbf{Δu}(k+H_c-1)
    \end{bmatrix}
```

in which ``\mathbf{ΔU}`` is a vector of `nu*Hc` elements.

## PredictiveController

```@docs
PredictiveController
```

## LinMPC

```@docs
LinMPC
```

## ExplicitMPC

```@docs
ExplicitMPC
```

## NonLinMPC

```@docs
NonLinMPC
```

## Move Manipulated Input u

```@docs
moveinput!
```

## Direct Transcription Methods

### TranscriptionMethod

```@docs
ModelPredictiveControl.TranscriptionMethod
```

### SingleShooting

```@docs
SingleShooting
```

### MultipleShooting

```@docs
MultipleShooting
```
