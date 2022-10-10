using ModelPredictiveControl
using Test

@testset "ModelPredictiveControl.jl" begin
    @test ModelPredictiveControl.greet() == "Hello World!"
end
