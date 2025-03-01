@testitem "DocTest" begin
    using Documenter
    old_debug_level = get(ENV, "JULIA_DEBUG", "")
    DocMeta.setdocmeta!(
        ModelPredictiveControl, 
        :DocTestSetup, 
        :(
            using ModelPredictiveControl, ControlSystemsBase;
            ENV["JULIA_DEBUG"] = ""; # temporarily disable @debug logging for the doctests
        ); 
        recursive=true,
        warn=false
    )
    doctest(ModelPredictiveControl, testset="DocTest")
    ENV["JULIA_DEBUG"] = old_debug_level
end