# Functions: Simulations and Plots

```@contents
Pages = ["plot_sim.md"]
```

This page documents the functions for quick plotting of open- and closed-loop
simulations. They are generic to [`SimModel`](@ref), [`StateEstimator`](@ref) and
[`PredictiveController`](@ref) types. A [`SimResult`](@ref) instance must be created first
with its constructor or by calling [`sim!`](@ref). The results are then visualized with
`plot` function from [`Plots.jl`](https://github.com/JuliaPlots/Plots.jl).

## Quick Simulations

```@docs
sim!
```

## Simulation Results

```@docs
SimResult
```

## Plots

```@docs
ModelPredictiveControl.plot
```
