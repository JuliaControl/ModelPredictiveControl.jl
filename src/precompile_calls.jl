sys = [ 
    tf(1.90, [18, 1]) tf(1.90, [18, 1]);
    tf(-0.74,[8, 1])  tf(0.74, [8, 1]) 
]
Ts = 4.0
model = setop!(LinModel(sys, Ts), uop=[10, 10], yop=[50, 30])
y = model()

mpc_im = setconstraint!(LinMPC(InternalModel(model)), ymin=[45, -Inf])
initstate!(mpc_im, model.uop, y)
u = mpc_im([55, 30], ym=y)
sim!(mpc_im, 3)

mpc_kf = setconstraint!(LinMPC(KalmanFilter(model)), ymin=[45, -Inf])
initstate!(mpc_kf, model.uop, model())
mpc_kf.estim()
u = mpc_kf([55, 30])
sim!(mpc_kf, 3, [55, 30])

mpc_lo = setconstraint!(LinMPC(Luenberger(model)), ymin=[45, -Inf])
initstate!(mpc_lo, model.uop, model())
mpc_lo.estim()
u = mpc_lo([55, 30])
sim!(mpc_lo, 3, [55, 30])

mpc_ukf = setconstraint!(LinMPC(UnscentedKalmanFilter(model)), ymin=[45, -Inf])
initstate!(mpc_ukf, model.uop, model())
mpc_ukf.estim()
u = mpc_ukf([55, 3])
sim!(mpc_ukf, 3, [55, 30])

mpc_ekf = setconstraint!(LinMPC(ExtendedKalmanFilter(model)), ymin=[45, -Inf])
initstate!(mpc_ekf, model.uop, model())
mpc_ekf.estim()
u = mpc_ekf([55, 30])
sim!(mpc_ekf, 3, [55, 30])

mpc_skf = setconstraint!(LinMPC(SteadyKalmanFilter(model)), ymin=[45, -Inf])
initstate!(mpc_skf, model.uop, model())
mpc_skf.estim()
u = mpc_skf([55, 30])
sim!(mpc_skf, 3, [55, 30])

nmpc_skf = setconstraint!(NonLinMPC(SteadyKalmanFilter(model), Cwt=Inf), ymin=[45, -Inf])
initstate!(nmpc_skf, model.uop, model())
nmpc_skf.estim()
u = nmpc_skf([55, 30])
sim!(nmpc_skf, 3, [55, 30])

res = sim!(model, 3)

res_man = SimResult(model, res.U_data, res.Y_data; X_data=res.X_data)

exmpc = ExplicitMPC(model)
initstate!(exmpc, model.uop, model())
exmpc.estim()
u = exmpc([55, 30])
sim!(exmpc, 3, [55, 30])

f(x,u,_) = model.A*x + model.Bu*u
h(x,_) = model.C*x

nlmodel = setop!(NonLinModel(f, h, Ts, 2, 2, 2), uop=[10, 10], yop=[50, 30])
y = nlmodel()
nmpc_im = setconstraint!(NonLinMPC(InternalModel(nlmodel), Hp=10, Cwt=Inf), ymin=[45, -Inf])
initstate!(nmpc_im, nlmodel.uop, y)
u = nmpc_im([55, 30], ym=y)
sim!(nmpc_im, 3, [55, 30])

nmpc_ukf = setconstraint!(NonLinMPC(UnscentedKalmanFilter(nlmodel), Hp=10, Cwt=1e3), ymin=[45, -Inf])
initstate!(nmpc_ukf, nlmodel.uop, y)
u = nmpc_ukf([55, 30])
sim!(nmpc_ukf, 3, [55, 30])

nmpc_ekf = setconstraint!(NonLinMPC(ExtendedKalmanFilter(model), Cwt=Inf), ymin=[45, -Inf])
initstate!(nmpc_ekf, model.uop, model())
u = nmpc_ekf([55, 30])
sim!(nmpc_ekf, 3, [55, 30])

function JE( _ , ŶE, _ )
    Ŷ  = ŶE[3:end]
    Eŷ = repeat([55; 30], 10) - Ŷ
    return dot(Eŷ, I, Eŷ)
end
empc = setconstraint!(NonLinMPC(nlmodel, Mwt=[0, 0], Hp=10, Cwt=Inf, Ewt=1, JE=JE), ymin=[45, -Inf])
u = empc()
sim!(empc, 3)
