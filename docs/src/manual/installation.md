# Manual: Installation

To install the `ModelPredictiveControl` package, run this command in the Julia REPL:

```julia
using Pkg; Pkg.add("ModelPredictiveControl")
```

It will also automatically install all the dependencies, including [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl)
and [`ControlSystemsBase.jl`](https://github.com/JuliaControl/ControlSystems.jl). Note that
that the construction of linear models typically requires `ss` or `tf` functions, it is thus
recommended to load the package with:

```julia
using ModelPredictiveControl, ControlSystemsBase
```
