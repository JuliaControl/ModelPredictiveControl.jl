# Functions: Predictive Controllers

```@contents
Pages = ["predictive_control.md"]
```

All the predictive controllers in this module rely on a state estimator to compute the
predictions. The default [`LinMPC`](@ref) estimator is a [`SteadyKalmanFilter`](@ref), and
[`NonLinMPC`](@ref) with nonlinear models, an [`UnscentedKalmanFilter`](@ref). For simpler
and more classical designs, an [`InternalModel`](@ref) structure is also available, that
assumes by default that the current model mismatch estimation is constant in the future
(same approach than dynamic matrix control, DMC).

!!! info
    The nomenclature use capital letters for time series (and matrices) and hats for the
    predictions (and estimations, for state estimators).

To be precise, at the ``k``th control period, the vectors that encompass the future measured
disturbances ``\mathbf{d̂}``, model outputs ``\mathbf{ŷ}`` and setpoints ``\mathbf{r̂_y}``
over the prediction horizon ``H_p`` are defined as:

```math
    \mathbf{D̂} = \begin{bmatrix}
        \mathbf{d̂}(k+1)   \\ \mathbf{d̂}(k+2)   \\ \vdots  \\ \mathbf{d̂}(k+H_p)
    \end{bmatrix} \: , \quad
    \mathbf{Ŷ} = \begin{bmatrix}
        \mathbf{ŷ}(k+1)   \\ \mathbf{ŷ}(k+2)   \\ \vdots  \\ \mathbf{ŷ}(k+H_p)
    \end{bmatrix} \: \text{and} \quad
    \mathbf{R̂_y} = \begin{bmatrix}
        \mathbf{r̂_y}(k+1) \\ \mathbf{r̂_y}(k+2) \\ \vdots  \\ \mathbf{r̂_y}(k+H_p)
    \end{bmatrix}
```

The vectors for the manipulated input ``\mathbf{u}`` are shifted by one time step:

```math
    \mathbf{U} = \begin{bmatrix}
        \mathbf{u}(k+0) \\ \mathbf{u}(k+1) \\ \vdots  \\ \mathbf{u}(k+H_p-1)
    \end{bmatrix} \: \text{and} \quad
    \mathbf{R̂_u} = \begin{bmatrix}
        \mathbf{r_u}    \\ \mathbf{r_u}    \\ \vdots  \\ \mathbf{r_u}
    \end{bmatrix}
```

assuming constant input setpoints at ``\mathbf{r_u}``. Defining the manipulated input
increment as ``\mathbf{Δu}(k+j) = \mathbf{u}(k+j) - \mathbf{u}(k+j-1)``, the control horizon
``H_c`` enforces that ``\mathbf{Δu}(k+j) = \mathbf{0}`` for ``j ≥ H_c``. For this reason,
the vector that collects them is truncated up to ``k+H_c-1``:

```math
    \mathbf{ΔU} =
    \begin{bmatrix}
        \mathbf{Δu}(k+0) \\ \mathbf{Δu}(k+1) \\ \vdots  \\ \mathbf{Δu}(k+H_c-1)
    \end{bmatrix}
```

## PredictiveController

```@docs
PredictiveController
```

## LinMPC

```@docs
LinMPC
```

## NonLinMPC

```@docs
NonLinMPC
```

## Set Constraint

```@docs
setconstraint!
```

## Move Manipulated Input u

```@docs
moveinput!
```

## Get Additional Information

```@docs
getinfo
```
