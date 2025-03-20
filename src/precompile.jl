sys = [ 
    tf(1.90, [1800, 1]) tf(1.90, [1800, 1]);
    tf(-0.74,[800, 1])  tf(0.74, [800, 1]) 
]
Ts = 400.0
model = setop!(LinModel(sys, Ts), uop=[10, 10], yop=[50, 30])
y = model()

mpc_im = setconstraint!(LinMPC(InternalModel(model)), ymin=[45, -Inf])
initstate!(mpc_im, model.uop, y)
preparestate!(mpc_im, [55, 30])
mpc_im.estim()
u = mpc_im([55, 30])
sim!(mpc_im, 2)

mpc_kf = setconstraint!(LinMPC(KalmanFilter(model)), ymin=[45, -Inf])
initstate!(mpc_kf, model.uop, model())
preparestate!(mpc_kf, [55, 30])
mpc_kf.estim()
u = mpc_kf([55, 30])
sim!(mpc_kf, 2, [55, 30])

mpc_lo = setconstraint!(LinMPC(Luenberger(model)), ymin=[45, -Inf])
initstate!(mpc_lo, model.uop, model())
preparestate!(mpc_lo, [55, 30])
mpc_lo.estim()
u = mpc_lo([55, 30])
sim!(mpc_lo, 2, [55, 30])

mpc_ukf = setconstraint!(LinMPC(UnscentedKalmanFilter(model)), ymin=[45, -Inf])
initstate!(mpc_ukf, model.uop, model())
preparestate!(mpc_ukf, [55, 30])
mpc_ukf.estim()
u = mpc_ukf([55, 3])
sim!(mpc_ukf, 2, [55, 30])

mpc_ekf = setconstraint!(LinMPC(ExtendedKalmanFilter(model)), ymin=[45, -Inf])
initstate!(mpc_ekf, model.uop, model())
preparestate!(mpc_ekf, [55, 30])
mpc_ekf.estim()
u = mpc_ekf([55, 30])
sim!(mpc_ekf, 2, [55, 30])

mpc_skf = setconstraint!(LinMPC(SteadyKalmanFilter(model)), ymin=[45, -Inf])
initstate!(mpc_skf, model.uop, model())
preparestate!(mpc_skf, [55, 30])
mpc_skf.estim()
u = mpc_skf([55, 30])
sim!(mpc_skf, 2, [55, 30])

mpc_mhe = setconstraint!(LinMPC(MovingHorizonEstimator(model, He=2)), ymin=[45, -Inf])
setconstraint!(mpc_mhe.estim, x̂min=[-50,-50,-50,-50], x̂max=[50,50,50,50])
initstate!(mpc_mhe, model.uop, model())
preparestate!(mpc_mhe, [55, 30])
mpc_mhe.estim()
u = mpc_mhe([55, 30])
sim!(mpc_mhe, 2, [55, 30])

nmpc_skf = setconstraint!(NonLinMPC(SteadyKalmanFilter(model), Cwt=Inf), ymin=[45, -Inf])
initstate!(nmpc_skf, model.uop, model())
preparestate!(nmpc_skf, [55, 30])
nmpc_skf.estim()
u = nmpc_skf([55, 30])
sim!(nmpc_skf, 2, [55, 30])

res = sim!(model, 2)

res_man = SimResult(model, res.U_data, res.Y_data; X_data=res.X_data)

exmpc = ExplicitMPC(model)
initstate!(exmpc, model.uop, model())
preparestate!(exmpc, [55, 30])
exmpc.estim()
u = exmpc([55, 30])
sim!(exmpc, 2, [55, 30])

f(x,u,_,model) = model.A*x + model.Bu*u
h(x,_,model) = model.C*x

nlmodel = setop!(
    NonLinModel(f, h, Ts, 2, 2, 2, solver=nothing, p=model), 
    uop=[10, 10], yop=[50, 30]
)
y = nlmodel()
nmpc_im = setconstraint!(NonLinMPC(InternalModel(nlmodel), Hp=10, Cwt=Inf), ymin=[45, -Inf])
initstate!(nmpc_im, nlmodel.uop, y)
preparestate!(nmpc_im, [55, 30])
nmpc_im.estim()
u = nmpc_im([55, 30])
sim!(nmpc_im, 2, [55, 30])

nmpc_ukf = setconstraint!(
    NonLinMPC(UnscentedKalmanFilter(nlmodel), Hp=10, Cwt=1e3), ymin=[45, -Inf]
)
initstate!(nmpc_ukf, nlmodel.uop, y)
preparestate!(nmpc_ukf, [55, 30])
u = nmpc_ukf([55, 30])
sim!(nmpc_ukf, 2, [55, 30])

nmpc_ekf = setconstraint!(NonLinMPC(ExtendedKalmanFilter(model), Cwt=Inf), ymin=[45, -Inf])
initstate!(nmpc_ekf, model.uop, model())
preparestate!(nmpc_ekf, [55, 30])
u = nmpc_ekf([55, 30])
sim!(nmpc_ekf, 2, [55, 30])

nmpc_mhe = setconstraint!(
    NonLinMPC(MovingHorizonEstimator(nlmodel, He=2), Hp=10, Cwt=Inf), ymin=[45, -Inf]
)
setconstraint!(nmpc_mhe.estim, x̂min=[-50,-50,-50,-50], x̂max=[50,50,50,50])
initstate!(nmpc_mhe, nlmodel.uop, y)
preparestate!(nmpc_mhe, [55, 30])
u = nmpc_mhe([55, 30])
sim!(nmpc_mhe, 2, [55, 30])

function JE( _ , Ŷe, _ , R̂y)
    Ŷ = Ŷe[3:end]
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

linearizemodel = linearize(nlmodel)
setmodel!(mpc_kf, linearizemodel)