# spell-checker: disable

using Revise

using ModelPredictiveControl

using LinearAlgebra
using ControlSystemsBase
using BenchmarkTools
using DifferentiationInterface

#=
using DifferentiationInterface
using SparseConnectivityTracer
using SparseMatrixColorings

f1(x::AbstractVector) = diff(x .^ 2) + diff(reverse(x .^ 2))
f2(x::AbstractVector) = sum(f1(x) .^ 2)

dense_forward_backend = AutoForwardDiff()
sparse_forward_backend = AutoSparse(
    dense_forward_backend;  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

x = rand(1000)
jac_prep_sparse = prepare_jacobian(f1, sparse_forward_backend, zero(x))

J = similar(sparsity_pattern(jac_prep_sparse), eltype(x))
jacobian!(f1, J, jac_prep_sparse, sparse_forward_backend, x)
=#


function f3!(dx, x, u, d, p)
    mul!(dx, p.A, x)
    mul!(dx, p.Bu, u, 1, 1)
    mul!(dx, p.Bd, d, 1, 1)
    return nothing
end
function h3!(y, x, d, p)
    mul!(y, p.C, x)
    mul!(y, p.Dd, d, 1, 1)
    return nothing
end

A=[0 0.5; -0.2 -0.1]
Bu=[0; 0.5]
Bd=[0; 0.5]
C=[0.4 0]
Dd=[0]
nonLinModel3 = NonLinModel(f3!, h3!, 1.0, 1, 2, 1, 1, solver=RungeKutta(4, supersample=2), p=(;A, Bu, Bd, C, Dd))


linearizeModel = linearize(nonLinModel3, x=[1,1], u=[1], d=[1])

@benchmark linearize!($linearizeModel, $nonLinModel3, x=$[2,2], u=$[1], d=$[1])

linearize2!(linemodel, model) = (linearize!(linemodel, model); nothing)

#=
linfunc! = nonLinModel3.linfunc!
xnext = linearizeModel.buffer.x
y = linearizeModel.buffer.y
A = linearizeModel.A
Bu = linearizeModel.Bu
Bd = linearizeModel.Bd
C = linearizeModel.C
Dd = linearizeModel.Dd
x = [2.0, 2.0]
u = [1.0]
d = [1.0]
cx = Constant(x)
cu = Constant(u)
cd = Constant(d)
backend = nonLinModel3.jacobian


@code_warntype linfunc!(xnext, y, A, Bu, C, Bd, Dd, backend, x, u, d, cx, cu, cd)
=#