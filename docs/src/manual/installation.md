# Manual: Installation

To install the `ModelPredictiveControl` package, run this command in the Julia REPL:

```julia
using Pkg; Pkg.activate(); Pkg.add("ModelPredictiveControl")
```

Doing so will install the package to default Julia environnement, that is, accessible
anywhere. To facilitate sharing of code and reproducibility of results, it is recommended to
install packages in a project environnement. To generate a new project named `MPCproject`
with this package in the current working directory, write this in the REPL:

```julia
using Pkg; Pkg.generate("MPCproject"); Pkg.activate("."); Pkg.add("ModelPredictiveControl")
```

Note that that the construction of linear models typically requires `ss` or `tf` functions,
it is thus advised to load the package with:

```julia
using ModelPredictiveControl, ControlSystemsBase
```
