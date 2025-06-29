using BenchmarkTools

using ModelPredictiveControl, ControlSystemsBase, LinearAlgebra
Ts = 400.0
sys = [ tf(1.90,[1800.0,1])   tf(1.90,[1800.0,1])   tf(1.90,[1800.0,1]);
        tf(-0.74,[800.0,1])   tf(0.74,[800.0,1])    tf(-0.74,[800.0,1])   ] 
linmodel = setop!(LinModel(sys, Ts, i_d=[3]), uop=[10,50], yop=[50,30], dop=[5])
function f!(ẋ, x, u, d, p)
    mul!(ẋ, p.A, x)
    mul!(ẋ, p.Bu, u, 1, 1)
    mul!(ẋ, p.Bd, d, 1, 1)
    return nothing
end
function h!(y, x, d, p)
    mul!(y, p.C, x)
    mul!(y, p.Dd, d, 1, 1)
    return nothing
end
nonlinmodel = NonLinModel(f!, h!, Ts, 2, 4, 2, 1, p=linmodel, solver=nothing)
nonlinmodel = setop!(nonlinmodel, uop=[10,50], yop=[50,30], dop=[5])
u, d = [10, 50], [5]

const SUITE = BenchmarkGroup()

## ================== SimModel benchmarks =========================================
SUITE["SimModel"]["allocation"] = BenchmarkGroup(["allocation"])
SUITE["SimModel"]["allocation"]["LinModel_updatestate!"] = @benchmarkable(
    updatestate!($linmodel, $u, $d),
    samples=1
)
SUITE["SimModel"]["allocation"]["LinModel_evaloutput"] = @benchmarkable(
    evaloutput($linmodel, $d),
    samples=1
)
SUITE["SimModel"]["allocation"]["NonLinModel_updatestate!"] = @benchmarkable(
    updatestate!($nonlinmodel, $u, $d),
    samples=1
)
SUITE["SimModel"]["allocation"]["NonLinModel_evaloutput"] = @benchmarkable(
    evaloutput($nonlinmodel, $d),
    samples=1
)

SUITE["SimModel"]["allocation"]["linearize!"] = @benchmarkable(
    linearize!($linmodel, $nonlinmodel),
    samples=1
)

## ================== StateEstimator benchmarks ================================
