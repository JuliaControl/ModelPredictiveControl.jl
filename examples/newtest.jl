using ModelPredictiveControl, Preferences
set_preferences!(ModelPredictiveControl, "precompile_workload" => false; force=true)

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
model = LinModel(ss(A,[Bu Bd],C,[Du Dd],Ts),Ts,i_d=[3])
model = setop!(model, uop=[10,10], yop=[50,30], dop=[5])

using BenchmarkTools

updatestate!(model, [0.0, 0.0], [0.0])
#@btime updatestate!($model, $[0.0, 0.0], $[0.0])
#@ballocations updatestate!($model, $[0.0, 0.0], $[0.0])
#@btime evaloutput($model, $[0.0])

#mpc = LinMPC(model, Hp=10, Hc=[1, 3, 3, 1, 2], transcription=MultipleShooting())#, Cwt=Inf)
#mpc = setconstraint!(mpc, ymin=[48,29],ymax=[52,30.5])

f(x,u,d,p) = p.A*x + p.Bu*u + p.Bd*d
h(x,d,p)   = p.C*x + p.Dd*d
model = NonLinModel(f, h, Ts, 2, 4, 2, 1, solver=nothing, p=(;A,Bu,Bd,C,Dd))
model = setop!(model, uop=[10,10], yop=[50,30], dop=[5])

#using Logging; debuglogger = ConsoleLogger(stderr, Logging.Debug)
# = with_logger(debuglogger) do
#end

using DifferentiationInterface, SparseConnectivityTracer, SparseMatrixColorings
import ForwardDiff, Symbolics

# using JuMP, MadNLP
# optim = Model(MadNLP.Optimizer)

hessian = false

mpc = NonLinMPC(model; Hp=10, transcription=MultipleShooting(), Cwt=Inf, hessian)
mpc = setconstraint!(mpc, ymin=[48,29],ymax=[52,30.5])
#unset_silent(mpc.optim)
using JuMP; unset_time_limit_sec(mpc.optim)
res = sim!(mpc, 15, x̂_0=zeros(6), x_0=zeros(4))
obj = mpc

# mhe = MovingHorizonEstimator(model; He=10, hessian)#, optim)
# using JuMP; unset_time_limit_sec(mhe.optim)
# mhe = setconstraint!(mhe, x̂max=[0.0,Inf,Inf,Inf,Inf,Inf])
# #unset_silent(mhe.optim)
# res = sim!(mhe, 15, x̂_0=zeros(6))
# obj = mhe

using PlotThemes, Plots
#theme(:default)
theme(:dark)
default(fontfamily="Computer Modern"); scalefontsizes(1.1)
plot(res) |> display

#using Logging; debuglogger = ConsoleLogger(stderr, Logging.Debug)
#with_logger(debuglogger) do
@benchmark sim!($obj, 15, x̂_0=$zeros(6), x_0=$zeros(4)) seconds = 30
#end

