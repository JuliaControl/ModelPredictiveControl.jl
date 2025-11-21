using ModelPredictiveControl, ControlSystemsBase
using JuMP, Ipopt, MadNLP
using BenchmarkTools

Ts = 4.0
A =  [  0.800737  0.0       0.0  0.0
        0.0       0.606531  0.0  0.0
        0.0       0.0       0.8  0.0
        0.0       0.0       0.0  0.6    ]
Bu = [  0.378599  0.378599
        -0.291167  0.291167
        0.0       0.0
        0.0       0.0                   ]
Bd = [  0; 0; 0.5; 0.5;;                ]
C =  [  1.0  0.0  0.684   0.0
        0.0  1.0  0.0    -0.4736        ]
Dd = [  0.19; -0.148;;                  ]
Du = zeros(2,2)
#model = LinModel(ss(A,[Bu Bd],C,[Du Dd],Ts),Ts,i_d=[3])
#model = setop!(model, uop=[10,10], yop=[50,30], dop=[5])
#mpc = LinMPC(model, transcription=MultipleShooting())#, Cwt=Inf)
#mpc = setconstraint!(mpc, ymax=[Inf,30.5])

using LinearAlgebra

function f!(xnext, x, u, d, p)
    mul!(xnext, p.A, x)
    mul!(xnext, p.Bu, u, 1, 1)
    mul!(xnext, p.Bd, d, 1, 1)
    return nothing
end

function h!(y, x, d, p)
    mul!(y, p.C, x)
    mul!(y, p.Dd, d, 1, 1)
    return nothing
end

model = NonLinModel(f!, h!, Ts, 2, 4, 2, 1, solver=nothing, p=(;A,Bu,Bd,C,Dd))
model = setop!(model, uop=[10,10], yop=[50,30], dop=[5])

ekf = ExtendedKalmanFilter(model, nint_u=[1, 1])

y = [50.0, 30.0]
u = [10.0, 10.0]
d = [5.0]
preparestate!(ekf, y, d)
updatestate!(ekf, u, y, d)

@benchmark (preparestate!($ekf, $y, $d); updatestate!($ekf, $u, $y, $d))
