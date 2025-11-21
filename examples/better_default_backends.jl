using ModelPredictiveControl, JuMP
using Ipopt, UnoSolver

N = 35*10 # number of solves/optimize! calls

function f!(ẋ, x, u, _ , p)
    g, L, K, m = p          # [m/s²], [m], [kg/s], [kg]
    θ, ω = x[1], x[2]       # [rad], [rad/s]
    τ  = u[1]               # [Nm]
    ẋ[1] = ω
    ẋ[2] = -g/L*sin(θ) - K/m*ω + τ/m/L^2
    return nothing
end
p = [9.8, 0.4, 1.2, 0.3]
nu, nx, ny, Ts = 1, 2, 1, 0.1

p_plant = copy(p)
p_plant[3] = 1.25*p[3]

h(x, _ , _ ) = [180/π*x[1], x[2]]
nu, nx, ny = 1, 2, 2
model = NonLinModel(f!, h, Ts, nu, nx, ny; p=p)
plant = NonLinModel(f!, h, Ts, nu, nx, ny; p=p_plant)

Hp, Hc, Mwt, Nwt = 20, 2, [0.5, 0], [2.5]
α=0.01; σQ=[0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
σQint_ym = zeros(0)

estim = UnscentedKalmanFilter(model; σQ, σR, nint_u, σQint_u, i_ym=[1])

umin, umax = [-1.5], [+1.5]
transcription = MultipleShooting()
oracle = true
hessian = true
optim = JuMP.Model(()->UnoSolver.Optimizer(preset="filtersqp"))

function gc!(LHS, Ue, Ŷe, _, p, ϵ)
    Pmax = p
    i_τ, i_ω = 1, 2
    for i in eachindex(LHS)
        τ, ω = Ue[i_τ], Ŷe[i_ω]
        P = τ*ω
        LHS[i] = P - Pmax - ϵ
        i_τ += 1
        i_ω += 2
    end
    return nothing
end
Cwt, Pmax, nc = 1e5, 3, Hp+1

nmpc = NonLinMPC(estim; 
    Hp, Hc, Mwt, Nwt, 
    #Cwt, gc!, nc, p=Pmax,
    transcription, oracle, hessian,
    optim
)
nmpc = setconstraint!(nmpc; umin, umax)
unset_time_limit_sec(nmpc.optim)
#unset_silent(nmpc.optim)
sim!(nmpc, N, [180.0, 0]; plant=plant, x_0=[0, 0], x̂_0=[0, 0, 0])
@time sim!(nmpc, N, [180.0, 0]; plant=plant, x_0=[0, 0], x̂_0=[0, 0, 0])
@profview sim!(nmpc, N, [180.0, 0]; plant=plant, x_0=[0, 0], x̂_0=[0, 0, 0])