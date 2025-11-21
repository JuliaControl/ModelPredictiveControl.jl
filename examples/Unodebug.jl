using ModelPredictiveControl, JuMP
using UnoSolver

N = 35 # number of JuMP.optimize! calls

function f!(ẋ, x, u, _ , p)
    g, L, K, m = p          # [m/s²], [m], [kg/s], [kg]
    θ, ω = x[1], x[2]       # [rad], [rad/s]
    τ  = u[1]               # [Nm]
    ẋ[1] = ω
    ẋ[2] = -g/L*sin(θ) - K/m*ω + τ/m/L^2
    return nothing
end
h!(y, x, _ , _ ) = (y[1] = 180/π*x[1]; nothing) # [°]
p = [9.8, 0.4, 1.2, 0.3]
nu, nx, ny, Ts = 1, 2, 1, 0.1
model = NonLinModel(f!, h!, Ts, nu, nx, ny; p)
p_plant = copy(p); p_plant[3] = p[3]*1.25
plant = NonLinModel(f!, h!, Ts, nu, nx, ny; p=p_plant)
Hp, Hc, Mwt, Nwt = 20, 2, [0.5], [2.5]
α=0.01; σQ=[0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
σQint_ym = zeros(0)
umin, umax = [-1.5], [+1.5]
transcription = MultipleShooting()
optim = Model(()->UnoSolver.Optimizer(preset="filtersqp"))
oracle = true
hessian = true
nmpc = NonLinMPC(model; 
    Hp, Hc, Mwt, Nwt, Cwt=Inf, transcription, oracle, hessian, optim,
    α, σQ, σR, nint_u, σQint_u, σQint_ym
)
nmpc = setconstraint!(nmpc; umin, umax)
unset_time_limit_sec(nmpc.optim)
sim!(nmpc, N, [180.0]; plant=plant, x_0=[0, 0], x̂_0=[0, 0, 0])
@time sim!(nmpc, N, [180.0]; plant=plant, x_0=[0, 0], x̂_0=[0, 0, 0])
@profview sim!(nmpc, N, [180.0]; plant=plant, x_0=[0, 0], x̂_0=[0, 0, 0])