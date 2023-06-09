# Manual

## Installation

To install the `ModelPredictiveControl` package, run this command in the Julia REPL:

```text
using Pkg; Pkg.add("ModelPredictiveControl")
```

## Predictive Controller Design

### Linear Model

The considered plant is well-stirred tank with a cold and hot water inlet. The water
flows out of an opening at the bottom of the tank. The manipulated inputs are the cold
``u_c`` and hot ``u_h`` water flow rate, and the measured outputs are the liquid level
``y_L`` and temperature ``y_T``:

```math
\begin{aligned}
    \mathbf{u} &= \begin{bmatrix} u_c \\ u_h \end{bmatrix} \\
    \mathbf{y} &= \begin{bmatrix} y_L \\ y_T \end{bmatrix}
\end{aligned}
```

At the steady-state operating points:

```math
\begin{aligned}
    \mathbf{u_{op}} &= \begin{bmatrix} 10 \\ 10 \end{bmatrix} \\
    \mathbf{y_{op}} &= \begin{bmatrix} 50 \\ 30 \end{bmatrix} 
\end{aligned}
```

the following linear model accurately describes the plant dynamics:

```math
\begin{bmatrix}
    y_L(s) \\ y_T(s)
\end{bmatrix} = 
\begin{bmatrix}
    \frac{1.90}{18s + 1} & \frac{1.90}{18s + 1} \\[3pt]
    \frac{-0.74}{8s + 1} & \frac{0.74}{8s + 1}
\end{bmatrix}
\begin{bmatrix}
    u_c(s) \\ u_h(s)
\end{bmatrix}
```

We want to design a predictive feedback that controls both the water level ``y_L`` and
temperature ``y_T`` in the tank, at a sampling time of 4 s. The tank level should also never
fall below 45:

```math
y_L ≥ 45
```

We first need to construct a [`LinModel`](@ref) objet with [`setop!`](@ref) to handle the
operating points:

```@example 1
using ModelPredictiveControl, ControlSystemsBase
sys = [ tf(1.90, [18, 1]) tf(1.90, [18, 1]);
        tf(-0.74,[8, 1])  tf(0.74, [8, 1]) ]
Ts = 4.0
model = setop!(LinModel(sys, Ts), uop=[10, 10], yop=[50, 30])
```

The `model` object will be used for two purposes : to construct our controller, and as a
plant simulator to test the design. We design our [`LinMPC`](@ref) controllers by including
the level constraint with [`setconstraint!`](@ref):

```@example 1
mpc = setconstraint!(LinMPC(model, Hp=15, Hc=2), ŷmin=[45, -Inf])
```

By default, [`LinMPC`](@ref) controllers use a [`SteadyKalmanFilter`](@ref) to estimate the
plant states. Before closing the loop, we call [`initstate!`](@ref) with the actual plant
inputs and measurements to ensure a bumpless transfer. Since `model` simulates our plant
here, its output will initialize the states. [`LinModel`](@ref) objects are callable for
this purpose (an alias for [`evaloutput`](@ref)):

```@example 1
u = model.uop
y = model() # or equivalently : y = evaloutput(model)
initstate!(mpc, u, y)
nothing # hide
```

We can then close the loop and test `mpc` performance on the simulator by imposing step
changes on output setpoints ``\mathbf{r_y}`` and on a load disturbance ``\mathbf{u_d}``:

```@example 1
function test_mpc(mpc, model)
    N = 100
    ry, ud = [50, 30], [0, 0]
    u_data  = zeros(model.nu, N)
    y_data  = zeros(model.ny, N)
    ry_data = zeros(model.ny, N)
    for k = 0:N-1
        y = model() # simulated measurements
        k == 25 && (ry = [50, 35])
        k == 50 && (ry = [55, 30])
        k == 75 && (ud = [-15, 0])
        u = mpc(ry) # or equivalently : u = moveinput!(mpc, ry)
        u_data[:,k+1]  = u
        y_data[:,k+1]  = y
        ry_data[:,k+1] = ry 
        updatestate!(mpc, u, y) # update mpc state estimate
        updatestate!(model, u + ud) # update simulator with disturbance
    end
    return u_data, y_data, ry_data
end
u_data, y_data, ry_data = test_mpc(mpc, model)
t_data = Ts*(0:(size(y_data,2)-1))
nothing # hide
```

The [`LinMPC`](@ref) objects are also callable as an alternative syntax for
[`moveinput!`](@ref). Calling [`updatestate!`](@ref) on the `mpc` object updates its
internal state for the *NEXT* control period (this is by design, see
[State Estimators](@ref) for justifications). That is why the call is done at the end of the
`for` loop. The same logic applies for `model`.

Lastly, we plot the closed-loop test with the `Plots` package:

```@example 1
using Plots
p1 = plot(t_data, y_data[1,:], label="level"); ylabel!("level")
plot!(t_data, ry_data[1,:], label="setpoint", linestyle=:dash, linetype=:steppost)
plot!(t_data, fill(45,size(t_data)), label="min", linestyle=:dot, linetype=:steppost)
p2 = plot(t_data, y_data[2,:], label="temp."); ylabel!("temp.")
plot!(t_data, ry_data[2,:],label="setpoint", linestyle=:dash, linetype=:steppost)
p3 = plot(t_data,u_data[1,:],label="cold", linetype=:steppost); ylabel!("flow rate")
plot!(t_data,u_data[2,:],label="hot", linetype=:steppost); xlabel!("time (s)")
p = plot(p1, p2, p3, layout=(3,1), fmt=:svg)
```

### Nonlinear Model

In this example, the goal is to control the angular position ``θ`` of a pendulum
attached to a motor. If the manipulated input is the motor torque ``τ``, the vectors
are:

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

```@example 2
using ModelPredictiveControl
function pendulum(par, x, u)
    g, L, K, m = par        # [m/s], [m], [kg/s], [kg]
    θ, ω = x[1], x[2]       # [rad], [rad/s]
    τ  = u[1]               # [N m]
    dθ = ω
    dω = -g/L*sin(θ) - k/m*ω + τ/m/L^2
    return [dθ, dω]
end
Ts  = 0.1                   # [s]
par = (9.8, 0.4, 1.2, 0.3)
f(x, u, _ ) = x + Ts*pendulum(par, x, u) # Euler method
h(x, _ )    = [180/π*x[1]]  # [°]
nu, nx, ny = 1, 2, 1
model = NonLinModel(f, h, Ts, nu, nx, ny)
```

The output function ``\mathbf{h}`` converts the angular position ``θ`` to degrees. It
is good practice to first simulate `model` using [`sim!`](@ref) as a quick sanity check:

```@example 2
using Plots
u = [0.5] # τ = 0.5 N m
plot(sim!(model, 60, u), plotu=false)
```

An [`UnscentedKalmanFilter`](@ref) estimates the plant state :

```@example 2
estim = UnscentedKalmanFilter(model, σQ=[0.5, 2.5], σQ_int=[0.5])
```

The standard deviation of the angular velocity ``ω`` is higher here (`σQ` second value)
since ``\dot{ω}(t)`` equation includes an uncertain parameter: the friction coefficient
``K``. The estimator tuning is tested on a plant simulated with a different ``K``:

```@example 2
par_plant = (par[1], par[2], par[3] + 0.25, par[4])
f_plant(x, u, _) = x + Ts*pendulum(par_plant, x, u)
plant = NonLinModel(f_plant, h, Ts, nu, nx, ny)
res = sim!(estim, 30, [0.5], plant=plant, y_noise=[0.5]) # τ = 0.5 N m
p2 = plot(res, plotu=false, plotx=true, plotx̂=true)
```

The Kalman filter performance seems sufficient for control. As the motor torque is limited
to -1.5 to 1.5 N m, we incorporate the input constraints in a [`NonLinMPC`](@ref):

```@example 2
mpc = NonLinMPC(estim, Hp=20, Hc=2, Mwt=[0.1], Nwt=[1.0], Cwt=Inf)
mpc = setconstraint!(mpc, umin=[-1.5], umax=[+1.5])
```

We test `mpc` performance on `plant` by imposing an angular setpoint of 180° (inverted
position):

```@example 2
res = sim!(mpc, 30, [180.0], x̂0=zeros(mpc.estim.nx̂), plant=plant, x0=zeros(plant.nx))
plot(res, plotŷ=true)
```

The controller seems robust enough to variations on ``K`` coefficient.
