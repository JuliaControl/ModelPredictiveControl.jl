using ModelPredictiveControl, Plots
function f!(ẋ, x, u, _ , p)
    g, L, K, m = p       # [m/s²], [m], [kg/s], [kg]
    θ, ω = x[1], x[2]    # [rad], [rad/s]
    τ = u[1]             # [Nm]
    ẋ[1] = ω
    ẋ[2] = -g/L*sin(θ) - K/m*ω + τ/m/L^2
end
h!(y, x, _ , _ ) = (y[1] = 180/π*x[1])   # [°]
p = [9.8, 0.4, 1.2, 0.3]
nu = 1; nx = 2; ny = 1; Ts = 0.1
model = NonLinModel(f!, h!, Ts, nu, nx, ny; p)
vu = ["\$τ\$ (Nm)"]
vx = ["\$θ\$ (rad)", "\$ω\$ (rad/s)"]
vy = ["\$θ\$ (°)"]
model = setname!(model; u=vu, x=vx, y=vy)

## =========================================
σQ = [0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]

## =========================================
p_plant = copy(p); p_plant[3] = 1.25*p[3]
N = 35; u = [0.5]; 

## =========================================
Hp, Hc, Mwt, Nwt, Cwt = 20, 2, [0.5], [2.5], Inf
umin, umax = [-1.5], [+1.5]

h2!(y, x, _ , _ ) = (y[1] = 180/π*x[1]; y[2]=x[2])
nu, nx, ny = 1, 2, 2
model2 = NonLinModel(f!, h2!, Ts, nu, nx, ny; p)
plant2 = NonLinModel(f!, h2!, Ts, nu, nx, ny; p=p_plant)
model2 = setname!(model2, u=vu, x=vx, y=[vy; vx[2]])
plant2 = setname!(plant2, u=vu, x=vx, y=[vy; vx[2]])
estim2 = UnscentedKalmanFilter(model2; σQ, σR, 
                               nint_u, σQint_u, i_ym=[1])

function JE(UE, ŶE, _ , p)
    Ts = p
    τ, ω = UE[1:end-1], ŶE[2:2:end-1]
    return Ts*sum(τ.*ω)
end
p = Ts; Mwt2 = [Mwt; 0.0]; Ewt = 3.5e3
empc = NonLinMPC(estim2; Hp, Hc, 
                 Nwt, Mwt=Mwt2, Cwt, JE, Ewt, p, oracle=true, transcription=MultipleShooting(), hessian=true)
empc = setconstraint!(empc; umin, umax)

## =========================================
using JuMP; unset_time_limit_sec(empc.optim)

## =========================================
x_0 = [0, 0]; x̂_0 = [0, 0, 0]; ry = [180; 0]
res2_ry = sim!(empc, N, ry; plant=plant2, x_0, x̂_0)
plot(res2_ry, ploty=[1])


## =========================================
## ========= Benchmark =====================
## =========================================
using BenchmarkTools
using JuMP, Ipopt, KNITRO

optim = JuMP.Model(Ipopt.Optimizer, add_bridges=false)
empc_ipopt = NonLinMPC(estim2; Hp, Hc, Nwt, Mwt=Mwt2, Cwt, JE, Ewt, optim, p, oracle=true, transcription=MultipleShooting(), hessian=true)
empc_ipopt = setconstraint!(empc_ipopt; umin, umax)
JuMP.unset_time_limit_sec(empc_ipopt.optim)
sim!(empc_ipopt, N, ry; plant=plant2, x_0=x_0, x̂_0=x̂_0)
@profview sim!(empc_ipopt, N, ry; plant=plant2, x_0=x_0, x̂_0=x̂_0)

y_step = [10.0, 0.0]

bm = @benchmark(
        sim!($empc_ipopt, $N, $ry; plant=$plant2, x_0=$x_0, x̂_0=$x̂_0),
        samples=50, 
        seconds=10*60
    )
@show btime_EMPC_track_solver_IP = median(bm)


bm = @benchmark(
        sim!($empc_ipopt, $N, $ry; plant=$plant2, x_0=$x_0, x̂_0=$x̂_0, y_step=$y_step),
        samples=50,
        seconds=10*60
    )
@show btime_EMPC_regul_solver_IP = median(bm)

