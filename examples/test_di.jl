using DifferentiationInterface, SparseConnectivityTracer, SparseMatrixColorings
import ForwardDiff
using LinearAlgebra, SparseArrays

cons(x) = map(abs2, x)
lag(x, μ) = dot(μ, cons(x))

backend = AutoSparse(
    AutoForwardDiff();
    sparsity_detector = TracerSparsityDetector(),
    coloring_algorithm = GreedyColoringAlgorithm(),
)

typical_x = zeros(10)
typical_μ1 = zeros(10)
typical_μ2 = spzeros(10)

prep1 = prepare_hessian(lag, backend, typical_x, Constant(typical_μ1));
prep2 = prepare_hessian(lag, backend, typical_x, Constant(typical_μ2));

display(sparsity_pattern(prep1))
display(sparsity_pattern(prep2))