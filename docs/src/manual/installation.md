# Manual: Installation

To install the `ModelPredictiveControl` package, run this command in the Julia REPL:

```julia
using Pkg; Pkg.add("ModelPredictiveControl")
```

Note that that the construction of linear models typically requires `ss` or `tf` functions,
it is thus recommended to load the package with:

```julia
using ModelPredictiveControl, ControlSystemsBase
```
