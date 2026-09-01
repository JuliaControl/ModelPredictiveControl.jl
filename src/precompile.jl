# Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the
# size of the precompile file and potentially make loading faster.
@setup_workload begin

sys = [ 
    tf(1.90, [1800, 1]) tf(1.90, [1800, 1]);
    tf(-0.74,[800, 1])  tf(0.74, [800, 1]) 
]
Ts = 400.0
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
p = (sys2.A, sys2.B, sys2.C)

function JE( _ , Ŷe, _ , R̂y , _ )
    Ŷ = @views Ŷe[3:end]
    Ȳ = R̂y - Ŷ
    return dot(Ȳ, Ȳ)
end
R̂y = repeat([55; 30], 3)

# all calls in this block will be precompiled, regardless of whether
# they belong to your package or not (on Julia 1.8 and higher)
@compile_workload begin

    model = setop!(LinModel(sys, Ts), uop=[10, 10], yop=[50, 30])
    y = model()

    mpc_skf = setconstraint!(LinMPC(SteadyKalmanFilter(model)), ymin=[45, -Inf])
    initstate!(mpc_skf, model.uop, model())
    preparestate!(mpc_skf, [55, 30])
    mpc_skf.estim()
    u = mpc_skf([55, 30])
    sim!(mpc_skf, 2, [55, 30])

    transcription = MultipleShooting()
    mpc_kf = setconstraint!(LinMPC(KalmanFilter(model); transcription), ymin=[45, -Inf])
    initstate!(mpc_kf, model.uop, model())
    preparestate!(mpc_kf, [55, 30])
    mpc_kf.estim()
    u = mpc_kf([55, 30])
    sim!(mpc_kf, 2, [55, 30])

    mpc_im = setconstraint!(LinMPC(InternalModel(model)), ymin=[45, -Inf])
    initstate!(mpc_im, model.uop, y)
    preparestate!(mpc_im, [55, 30])
    mpc_im.estim()
    u = mpc_im([55, 30])

    mhe = MovingHorizonEstimator(model, He=2)
    mhe = setconstraint!(mhe, x̂min=[-50,-50,-50,-50], x̂max=[50,50,50,50])
    initstate!(mhe, model.uop, model())
    preparestate!(mhe, [55, 30])
    mhe()

    res = sim!(model, 2)
    res_man = SimResult(model, res.U_data, res.Y_data; X_data=res.X_data)

    exmpc = ExplicitMPC(model)
    initstate!(exmpc, model.uop, model())
    preparestate!(exmpc, [55, 30])
    exmpc.estim()
    exmpc([55, 30])

    nlmodel = NonLinModel(f!, h!, Ts, 2, 2, 2; solver=RungeKutta(4), p)
    nlmodel = setop!(nlmodel, uop=[10, 10], yop=[50, 30])
    y = nlmodel()

    transcription = MultipleShooting(f_threads=true)
    nmpc_ukf = setconstraint!(
        NonLinMPC(nlmodel; Hp=3, Mwt=[0, 0], Ewt=1, JE=JE, p=R̂y, transcription),
        ymin=[45, -Inf]
    )
    sim!(nmpc_ukf, 2, [55, 30])

    transcription = OrthogonalCollocation(0, 2)
    nmpc_mhe = setconstraint!(NonLinMPC(
        MovingHorizonEstimator(nlmodel; He=2, transcription); transcription, Hp=3, Cwt=Inf), 
        ymin=[45, -Inf]
    )
    setconstraint!(nmpc_mhe.estim, x̂min=[-50,-50,-50,-50], x̂max=[50,50,50,50])
    initstate!(nmpc_mhe, nlmodel.uop, y)
    preparestate!(nmpc_mhe, [55, 30])
    nmpc_mhe([55, 30])

    linearizemodel = linearize(nlmodel)
    setmodel!(mpc_kf, linearizemodel)

end

end # @setup_workload