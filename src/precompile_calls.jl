sys = [ tf(1.90, [18, 1]) tf(1.90, [18, 1]);
        tf(-0.74,[8, 1])  tf(0.74, [8, 1]) ]
Ts = 4.0
model = setop!(LinModel(sys, Ts), uop=[10, 10], yop=[50, 30])
y = model()

mpc_im = setconstraint!(LinMPC(InternalModel(model)), ŷmin=[45, -Inf])
initstate!(mpc_im, model.uop, y)
u = mpc_im([50, 30], ym=y)
updatestate!(mpc_im, u, y)
mpc_kf = setconstraint!(LinMPC(KalmanFilter(model)), ŷmin=[45, -Inf])
initstate!(mpc_kf, model.uop, model())
u = mpc_kf([50, 30])
updatestate!(mpc_kf, u, y)
mpc_ukf = setconstraint!(LinMPC(UnscentedKalmanFilter(model)), ŷmin=[45, -Inf])
initstate!(mpc_ukf, model.uop, model())
u = mpc_ukf([50, 30])
updatestate!(mpc_ukf, u, y)
mpc_skf = setconstraint!(LinMPC(SteadyKalmanFilter(model)), ŷmin=[45, -Inf])
initstate!(mpc_skf, model.uop, model())
u = mpc_skf([50, 30])
updatestate!(mpc_skf, u, y)
updatestate!(model, u) 

f(x,u,_) = model.A*x + model.Bu*u
h(x,_) = model.C*x

nlmodel = setop!(NonLinModel(f, h, Ts, 2, 2, 2))
y = nlmodel()
nmpc_im = setconstraint!(NonLinMPC(InternalModel(nlmodel)), ŷmin=[45, -Inf])
initstate!(nmpc_im, nlmodel.uop, y)
u = nmpc_im([50, 30], ym=y)
updatestate!(nmpc_im, u, y)
nmpc_ukf = setconstraint!(NonLinMPC(UnscentedKalmanFilter(nlmodel)), ŷmin=[45, -Inf])
initstate!(nmpc_ukf, nlmodel.uop, y)
u = nmpc_ukf([50, 30])
updatestate!(nmpc_ukf, u, y)