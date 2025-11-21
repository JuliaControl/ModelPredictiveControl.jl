using ControlSystemsBase, LinearAlgebra, ModelPredictiveControl

sys = [ 
    tf(1.90, [1800, 1]) tf(1.90, [1800, 1]);
    tf(-0.74,[800, 1])  tf(0.74, [800, 1]) 
]
Ts = 400.0
model = setop!(LinModel(sys, Ts), uop=[10, 10], yop=[50, 30])
y = model()

sys2 = minreal(ss(sys))

function f!(xnext, x, u, _ , p)
    A, B, _ = p
    mul!(xnext, A , x)
    mul!(xnext, B, u, 1, 1)
    return nothing
end
function h!(y, x, _ , p)
    _, _, C = p
    mul!(y, C, x)
    return nothing
end

nlmodel = setop!(
    NonLinModel(f!, h!, Ts, 2, 2, 2, solver=RungeKutta(4), p=(sys2.A, sys2.B, sys2.C)), 
    uop=[10, 10], yop=[50, 30]
)
y = nlmodel()

function JE( _ , Ŷe, _ , R̂y)
    Ŷ = @views Ŷe[3:end]
    Ȳ = R̂y - Ŷ
    return dot(Ȳ, Ȳ)
end
R̂y = repeat([55; 30], 10)
empc = setconstraint!(
    NonLinMPC(nlmodel, Mwt=[0, 0], Hp=10, Cwt=Inf, Ewt=1, JE=JE, p=R̂y), ymin=[45, -Inf]
)
preparestate!(empc, [55, 30])
u = empc()
sim!(empc, 2)