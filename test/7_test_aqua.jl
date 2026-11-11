@testitem "Aqua ambiguities" begin
    using Aqua
    Aqua.test_ambiguities(ModelPredictiveControl)
end

@testitem "Aqua unbound args" begin
    using Aqua
    Aqua.test_unbound_args(ModelPredictiveControl)
end

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

### This package does not use a __init__ function, and will very likely never use one,
### so the persistent task test is useless (and slow, because of the long precompilation)
# @testitem "Aqua persistent tasks" begin
#     using Aqua
#     Aqua.test_persistent_tasks(ModelPredictiveControl)
# end

@testitem "Aqua undocumented names" begin
    using Aqua
    Aqua.test_undocumented_names(ModelPredictiveControl)
end