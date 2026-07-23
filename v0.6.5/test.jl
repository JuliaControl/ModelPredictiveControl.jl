using ModelPredictiveControl

function pendulum(par, x, u)
    g, L, K, m = par    # [m/s], [m], [kg/s], [kg]
    θ, ω = x[1], x[2]   # [rad], [rad/s]
    τ  = u[1]           # [N m]
    dθ = ω
    dω = -g/L*sin(θ) - K/m*ω + τ/m/L^2
    return [dθ, dω]
end

Ts = 0.1    # [s]
par = (9.8, 0.4, 1.2, 0.3)
f(x, u, _) = x + Ts*pendulum(par, x, u)
h(x, _ ) = [180/π*x[1]]      # [rad] to [°]
nu, nx, ny = 1, 2, 1
model = NonLinModel(f, h, Ts, nu, nx, ny)

using Plots
p1 = plot(sim!(model, 50, [0.5])) # τ = 0.5 N m
display(p1)

par_plant = (par[1], par[2], par[3] + 0.25, par[4])
f_plant(x, u, _) = x + Ts*pendulum(par_plant, x, u)
plant = NonLinModel(f_plant, h, Ts, nu, nx, ny)

estim = UnscentedKalmanFilter(model, σQ=[0.5, 2.5], σQ_int=[0.5])


res = sim!(estim, 30, [0.5], plant=plant, y_noise=[0.5]) # τ = 0.5 N m
p2 = plot(res, plotu=false, plotx=true, plotx̂=true)
display(p2)


mpc = NonLinMPC(estim, Hp=20, Hc=2, Mwt=[0.1], Nwt=[1.0], Cwt=Inf)
mpc = setconstraint!(mpc, umin=[-1.5], umax=[+1.5])

res = sim!(mpc, 30, [180.0], x̂=zeros(mpc.estim.nx̂), plant=plant)
plot(res, plotŷ=true)
