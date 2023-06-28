# [Manual: Nonlinear Design](@id man_nonlin)

## Nonlinear Model

In this example, the goal is to control the angular position ``θ`` of a pendulum
attached to a motor. Knowing that the manipulated input is the motor torque ``τ``, the I/O
vectors are:

```math
\begin{aligned}
    \mathbf{u} &= \begin{bmatrix} τ \end{bmatrix} \\
    \mathbf{y} &= \begin{bmatrix} θ \end{bmatrix}
\end{aligned}
```

The plant model is nonlinear:

```math
\begin{aligned}
    \dot{θ}(t) &= ω(t)                                                                    \\
    \dot{ω}(t) &= -\frac{g}{L}\sin\big( θ(t) \big) - \frac{K}{m} ω(t) + \frac{1}{m L^2} τ(t)
\end{aligned}
```

in which ``g`` is the gravitational acceleration, ``L``, the pendulum length, ``K``, the
friction coefficient at the pivot point, and ``m``, the mass attached at the end of the
pendulum. Here, the explicit Euler method discretizes the system to construct a
[`NonLinModel`](@ref):

```@example 1
using ModelPredictiveControl
function pendulum(par, x, u)
    g, L, K, m = par        # [m/s], [m], [kg/s], [kg]
    θ, ω = x[1], x[2]       # [rad], [rad/s]
    τ  = u[1]               # [N m]
    dθ = ω
    dω = -g/L*sin(θ) - K/m*ω + τ/m/L^2
    return [dθ, dω]
end
Ts  = 0.1                   # [s]
par = (9.8, 0.4, 1.2, 0.3)
f(x, u, _ ) = x + Ts*pendulum(par, x, u) # Euler method
h(x, _ )    = [180/π*x[1]]  # [°]
nu, nx, ny = 1, 2, 1
model = NonLinModel(f, h, Ts, nu, nx, ny)
```

The output function ``\mathbf{h}`` converts the ``θ`` angle to degrees. It is good practice
to first simulate `model` using [`sim!`](@ref) as a quick sanity check:

```@example 1
using Plots
u = [0.5]
plot(sim!(model, 60, u), plotu=false)
```

## Nonlinear Predictive Controller

An [`UnscentedKalmanFilter`](@ref) estimates the plant state :

```@example 1
estim = UnscentedKalmanFilter(model, σQ=[0.1, 0.5], σQ_int=[5.0], σR=[0.5])
```

The standard deviation of the angular velocity ``ω`` is higher here (`σQ` second value)
since ``\dot{ω}(t)`` equation includes an uncertain parameter: the friction coefficient
``K``. The estimator tuning is tested on a plant simulated with a different ``K``:

```@example 1
par_plant = (par[1], par[2], par[3] + 0.25, par[4])
f_plant(x, u, _) = x + Ts*pendulum(par_plant, x, u)
plant = NonLinModel(f_plant, h, Ts, nu, nx, ny)
res = sim!(estim, 60, [0.5], plant=plant, y_noise=[0.5])
plot(res, plotu=false, plotxwithx̂=true)
```

The estimate ``x̂_3`` is the integrator state that compensates for static errors (`nint_ym`
parameter of [`UnscentedKalmanFilter`](@ref)). The Kalman filter performance seems
sufficient for control. As the motor torque is limited to -1.5 to 1.5 N m, we incorporate
the input constraints in a [`NonLinMPC`](@ref):

```@example 1
mpc = NonLinMPC(estim, Hp=20, Hc=4, Mwt=[0.05], Nwt=[2.5], Cwt=Inf)
mpc = setconstraint!(mpc, umin=[-1.5], umax=[+1.5])
```

We test `mpc` performance on `plant` by imposing an angular setpoint of 180° (inverted
position):

```@example 1
res = sim!(mpc, 60, [180.0], plant=plant, x0=zeros(plant.nx), x̂0=zeros(mpc.estim.nx̂))
plot(res)
```

The controller seems robust enough to variations on ``K`` coefficient. Starting from this
inverted position, the closed-loop response to a step disturbances of 10° is also
satisfactory:

```@example 1
res = sim!(mpc, 60, [180.0], plant=plant, x0=[π, 0], x̂0=[π, 0, 0], y_step=[10])
plot(res)
```
