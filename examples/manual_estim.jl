using ModelPredictiveControl, ControlSystemsBase

function man_sim()
    f(x,u,_,_) = 0.5*sin.(x + u)
    h(x,_,_) = x
    model = NonLinModel(f, h, 10.0, 1, 1, 1, solver=nothing)
    linModel = linearize(model, x=[0], u=[0])
    man = ManualEstimator(linModel, nint_u=[1])
    mpc = LinMPC(man)
    estim = MovingHorizonEstimator(model, nint_u=[1], He=5)
    estim = setconstraint!(estim, v̂min=[-0.001], v̂max=[0.001])
    initstate!(estim, [0], [0])
    y_data, ŷ_data = zeros(5), zeros(5)
    for i=1:5
        y = model()                         # simulated measurement
        x̂ = preparestate!(estim, y)         # correct nonlinear MHE state estimate
        ŷ = estim()                         # nonlinear MHE estimated output
        setstate!(mpc, x̂)                   # update MPC with the MHE corrected state 
        u = moveinput!(mpc, [0])
        y_data[i], ŷ_data[i] = y[1], ŷ[1]
        updatestate!(estim, u, y)           # update nonlinear MHE estimation
        updatestate!(model, u .+ 0.5)       # update plant simulator with load disturbance
    end
    return collect([y_data ŷ_data]')
end
YandŶ = round.(man_sim(), digits=6)