using ModelPredictiveControl
using JuMP, KNITRO, Ipopt

A =  [  0.800737  0.0       
        0.0       0.606531  ]
Bu = [  0.378599  0.378599
        -0.291167  0.291167 ]
C =  [  1.0  0.0  
        0.0  1.0            ]
f(x,u,_,_) = A*x + Bu*u
h(x,_,_)   = C*x
model = NonLinModel(f, h, 4.0, 2, 2, 2, solver=nothing)

optim = Model(KNITRO.Optimizer) # Model(Ipopt.Optimizer) # Ipopt does work here
oracle = true # true leads to `LOCALLY_INFEASIBLE` on KNITRO dev version, false works

nmpc = NonLinMPC(model; Hp=10, direct=false, Cwt=Inf, optim, oracle)
nmpc = setconstraint!(nmpc, ymin=[-2,-2],ymax=[+2,+2])
unset_time_limit_sec(nmpc.optim)
unset_silent(nmpc.optim)
set_optimizer_attribute(nmpc.optim, "outlev", 6)
u = moveinput!(nmpc, [1, 1])
#solution_summary(nmpc.optim)