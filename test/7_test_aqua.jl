@testitem "Aqua ambiguities" begin
    using Aqua
    Aqua.test_ambiguities(ModelPredictiveControl)
end

### All the functions defined inside `ModelPredictiveControl.get_nonlinobj_op` with the 
### `Vararg`s have unbound type parameters. This is necessary for the splatting syntax of
### `JuMP.@operator`, and JuMP will never call these functions with 0 argument, so
### defining zero-argument methods would be useless, so I disable this check here.
# @testitem "Aqua unbound args" begin
#     using Aqua
#     Aqua.test_unbound_args(ModelPredictiveControl)
# end

@testitem "Aqua undefined exports" begin
    using Aqua
    Aqua.test_undefined_exports(ModelPredictiveControl)
end

@testitem "Aqua project extras" begin
    using Aqua
    Aqua.test_project_extras(ModelPredictiveControl)
end

@testitem "Aqua stale deps" begin
    using Aqua
    Aqua.test_stale_deps(ModelPredictiveControl)
end

@testitem "Aqua deps compat" begin
    using Aqua
    Aqua.test_deps_compat(ModelPredictiveControl)
end

@testitem "Aqua piracies" begin
    using Aqua
    Aqua.test_piracies(ModelPredictiveControl)
end

@testitem "Aqua persistent tasks" begin
    using Aqua
    Aqua.test_persistent_tasks(ModelPredictiveControl)
end

@testitem "Aqua undocumented names" begin
    using Aqua
    Aqua.test_undocumented_names(ModelPredictiveControl)
end