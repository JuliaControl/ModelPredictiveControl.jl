module LinearMPCext

using ModelPredictiveControl, LinearMPC


ModelPredictiveControl.hi(::ModelPredictiveControl.Ext) = println("hello world!")

end # LinearMPCext