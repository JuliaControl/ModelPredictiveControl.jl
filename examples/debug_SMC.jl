using ModelPredictiveControl
using DifferentiationInterface, SparseConnectivityTracer, SparseMatrixColorings
import ForwardDiff
hessian = AutoSparse(
    AutoForwardDiff(); 
    sparsity_detector  = TracerSparsityDetector(), 
    coloring_algorithm = GreedyColoringAlgorithm(; postprocessing=true)
)
f(x,u,_,_) = 0.5*x + 0.5*u
h(x,_,_)   = x
model = NonLinModel(f, h, 10.0, 1, 1, 1, solver=nothing)
nmpc = NonLinMPC(model; Hp=5, direct=false, hessian)
moveinput!(nmpc)