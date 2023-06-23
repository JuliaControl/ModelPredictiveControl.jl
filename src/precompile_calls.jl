sys = [ tf(1.90, [18, 1]) tf(1.90, [18, 1]);
        tf(-0.74,[8, 1])  tf(0.74, [8, 1]) ]
Ts = 4.0
model = setop!(LinModel(sys, Ts), uop=[10, 10], yop=[50, 30])
y = model()

mpc_im = setconstraint!(LinMPC(InternalModel(model)), ŷmin=[45, -Inf])
initstate!(mpc_im, model.uop, y)
u = mpc_im([50, 30], ym=y)

mpc_kf = setconstraint!(LinMPC(KalmanFilter(model)), ŷmin=[45, -Inf])
initstate!(mpc_kf, model.uop, model())
u = mpc_kf([50, 30])
sim!(mpc_kf, 15)
mpc_lo = setconstraint!(LinMPC(Luenberger(model)), ŷmin=[45, -Inf])
initstate!(mpc_lo, model.uop, model())
u = mpc_lo([50, 30])
sim!(mpc_lo, 15)
mpc_ukf = setconstraint!(LinMPC(UnscentedKalmanFilter(model)), ŷmin=[45, -Inf])
initstate!(mpc_ukf, model.uop, model())
u = mpc_ukf([50, 30])
sim!(mpc_ukf, 15)
mpc_skf = setconstraint!(LinMPC(SteadyKalmanFilter(model)), ŷmin=[45, -Inf])
initstate!(mpc_skf, model.uop, model())
u = mpc_skf([50, 30])
sim!(mpc_skf, 15)

sim!(model, 15)

f(x,u,_) = model.A*x + model.Bu*u
h(x,_) = model.C*x

nlmodel = setop!(NonLinModel(f, h, Ts, 2, 2, 2))
y = nlmodel()
nmpc_im = setconstraint!(NonLinMPC(InternalModel(nlmodel)), ŷmin=[45, -Inf])
initstate!(nmpc_im, nlmodel.uop, y)
u = nmpc_im([50, 30], ym=y)
sim!(nmpc_im, 15)
nmpc_ukf = setconstraint!(NonLinMPC(UnscentedKalmanFilter(nlmodel)), ŷmin=[45, -Inf])
initstate!(nmpc_ukf, nlmodel.uop, y)
u = nmpc_ukf([50, 30])
sim!(nmpc_ukf, 15)