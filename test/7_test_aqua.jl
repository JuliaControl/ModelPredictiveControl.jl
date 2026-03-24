@testitem "Aqua" begin
    using Aqua
    # All the functions defined inside `ModelPredictiveControl.get_nonlinobj_op` with the 
    # `Vararg`s have unbound type parameters. This is necessary for the splatting syntax of
    # `JuMP.@operator`, and JuMP will never call these functions with 0 argument, so
    # defining zero-argument methods would be useless, so I disable this check here.
    unbound_args = false
    Aqua.test_all(ModelPredictiveControl; unbound_args)
end