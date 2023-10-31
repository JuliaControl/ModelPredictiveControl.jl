# [Manual: Nonlinear Design](@id man_nonlin)

```@contents
Pages = ["nonlinmpc.md"]
```

## Nonlinear Model

In this example, the goal is to control the angular position ``θ`` of a pendulum
attached to a motor. Knowing that the manipulated input is the motor torque ``τ``, the I/O
vectors are:

```math
\begin{aligned}
    \mathbf{u} &= τ \\
    \mathbf{y} &= θ
\end{aligned}
```

The following figure presents the system:

```@raw html
<img src="../../assets/pendulum.svg" alt="pendulum" width=200 style="background-color:white; 
    border:20px solid white; display: block; margin-left: auto; margin-right: auto;"/>
```

The plant model is nonlinear:

```math
\begin{aligned}
    \dot{θ}(t) &= ω(t) \\
    \dot{ω}(t) &= -\frac{g}{L}\sin\big( θ(t) \big) - \frac{K}{m} ω(t) + \frac{1}{m L^2} τ(t)
\end{aligned}
```

in which ``g`` is the gravitational acceleration in m/s², ``L``, the pendulum length in m,
``K``, the friction coefficient at the pivot point in kg/s, and ``m``, the mass attached at
the end of the pendulum in kg. Here, the explicit Euler method discretizes the system to
construct a [`NonLinModel`](@ref):

```@example 1
using ModelPredictiveControl
function pendulum(par, x, u)
    g, L, K, m = par        # [m/s²], [m], [kg/s], [kg]
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

The output function ``\mathbf{h}`` converts the ``θ`` angle to degrees. Note that special
characters like ``θ`` can be typed in the Julia REPL or VS Code by typing `\theta` and
pressing the `<TAB>` key. It is good practice to first simulate `model` using [`sim!`](@ref)
as a quick sanity check:

```@example 1
using Plots
u = [0.5]
N = 35
plot(sim!(model, N, u), plotu=false)
savefig(ans, "plot1_NonLinMPC.svg"); nothing # hide
```

![plot1_NonLinMPC](plot1_NonLinMPC.svg)

## Nonlinear Model Predictive Controller

An [`UnscentedKalmanFilter`](@ref) estimates the plant state :

```@example 1
σQ=[0.1, 0.5]; σR=[0.5]; nint_u=[1]; σQint_u=[0.1]
estim = UnscentedKalmanFilter(model; σQ, σR, nint_u, σQint_u)
```

The vectors `σQ` and σR `σR` are the standard deviations of the process and sensor noises,
respectively. The value for the velocity ``ω`` is higher here (`σQ` second value) since
``\dot{ω}(t)`` equation includes an uncertain parameter: the friction coefficient ``K``.
Also, the argument `nint_u` explicitly adds one integrating state at the model input, the
motor torque ``τ`` , with an associated standard deviation `σQint_u` of 0.1 N m. The
estimator tuning is tested on a plant with a 25 % larger friction coefficient ``K``:

```@example 1
par_plant = (par[1], par[2], 1.25*par[3], par[4])
f_plant(x, u, _) = x + Ts*pendulum(par_plant, x, u)
plant = NonLinModel(f_plant, h, Ts, nu, nx, ny)
res = sim!(estim, N, [0.5], plant=plant, y_noise=[0.5])
plot(res, plotu=false, plotxwithx̂=true)
savefig(ans, "plot2_NonLinMPC.svg"); nothing # hide
```

![plot2_NonLinMPC](plot2_NonLinMPC.svg)

The estimate ``x̂_3`` is the integrating state on the torque ``τ`` that compensates for
static errors. The Kalman filter performance seems sufficient for control.

As the motor torque is limited to -1.5 to 1.5 N m, we incorporate the input constraints in
a [`NonLinMPC`](@ref):

```@example 1
nmpc = NonLinMPC(estim, Hp=20, Hc=2, Mwt=[0.5], Nwt=[2.5])
nmpc = setconstraint!(nmpc, umin=[-1.5], umax=[+1.5])
```

We test `mpc` performance on `plant` by imposing an angular setpoint of 180° (inverted
position):

```@example 1
res_ry = sim!(nmpc, N, [180.0], plant=plant, x0=[0, 0], x̂0=[0, 0, 0])
plot(res_ry)
savefig(ans, "plot3_NonLinMPC.svg"); nothing # hide
```

![plot3_NonLinMPC](plot3_NonLinMPC.svg)

The controller seems robust enough to variations on ``K`` coefficient. Starting from this
inverted position, the closed-loop response to a step disturbances of 10° is also
satisfactory:

```@example 1
res_yd = sim!(nmpc, N, [180.0], plant=plant, x0=[π, 0], x̂0=[π, 0, 0], y_step=[10])
plot(res_yd)
savefig(ans, "plot4_NonLinMPC.svg"); nothing # hide
```

![plot4_NonLinMPC](plot4_NonLinMPC.svg)

## Economic Model Predictive Controller

Economic MPC can achieve the same objective but with lower economical costs. For this case
study, the controller will aim to reduce the energy consumed by the motor. The power (W)
transmitted by the motor to the pendulum is:

```math
Ẇ(t) = τ(t) ω(t)
```

Thus, the work (J) done by the motor from ``t = t_0`` to ``t_{end}`` is:

```math
W = \int_{t_0}^{t_{end}} Ẇ(t) \mathrm{d}t = \int_{t_0}^{t_{end}} τ(t) ω(t) dt
```

With the sampling time ``T_s`` in s, the prediction horizon ``H_p``, the limits defined as
``t_0 = k T_s`` and ``t_{end} = (k+H_p) T_s``, and the left-endpoint rectangle method for
the integral, we get:

```math
W ≈ T_s \sum_{j=0}^{H_p-1} τ(k + j) ω(k + j)
```

The objective function will now include an additive term that penalizes the work done by the
motor ``W`` to reduce the energy consumption. Notice that ``W`` is a function of the
manipulated input ``τ`` and the angular speed ``ω``, a state that is not measured (only the
angle ``θ`` is measured here). As the arguments of [`NonLinMPC`](@ref) economic function
`JE` do not include the states, the speed is now defined as an unmeasured output to design a
Kalman Filter similar to the previous one (``\mathbf{y^m} = θ`` and ``\mathbf{y^u} = ω``):

```@example 1
h2(x, _ ) = [180/π*x[1], x[2]]
nu, nx, ny = 1, 2, 2
model2 = NonLinModel(f, h2, Ts, nu, nx, ny)
estim2 = UnscentedKalmanFilter(model2; σQ, σR, nint_u, σQint_u, i_ym=[1])
```

We can now define the ``J_E`` function and the `empc` controller:

```@example 1
function JE(UE, ŶE, _ )
    τ, ω = UE[1:end-1], ŶE[2:2:end-1]
    return Ts*sum(τ.*ω)
end
empc = NonLinMPC(estim2, Hp=20, Hc=2, Mwt=[0.5, 0], Nwt=[2.5], Ewt=4e3, JE=JE)
empc = setconstraint!(empc, umin=[-1.5], umax=[+1.5])
```

The keyword argument `Ewt` weights the economic costs relative to the other terms in the
objective function. The second element of `Mwt` is zero since the speed ``ω`` is not
requested to track a setpoint. The closed-loop response to a 180° setpoint is similar:

```@example 1
res2_ry = sim!(empc, N, [180.0, 0], plant=plant, x0=[0, 0], x̂0=[0, 0, 0])
plot(res2_ry)
savefig(ans, "plot5_NonLinMPC.svg"); nothing # hide
```

![plot5_NonLinMPC](plot5_NonLinMPC.svg)

And the energy consumption is almost identical here:

```@example 1
function calcW(res)
    τ = res.U_data[1, 1:end-1]
    ω = res.X_data[2, 1:end-1]
    return Ts*sum(τ.*ω)
end
Dict(:nmpc => calcW(res_ry), :empc => calcW(res2_ry))
```

But, for the 10° step disturbance:

```@example 1
res2_yd = sim!(empc, N, [180.0; 0]; plant, x0=[π, 0], x̂0=[π, 0, 0], y_step=[10])
plot(res2_yd)
savefig(ans, "plot6_NonLinMPC.svg"); nothing # hide
```

![plot6_NonLinMPC](plot6_NonLinMPC.svg)

the new controller is able to recuperate more energy from the pendulum (i.e. negative work):

```@example 1
Dict(:nmpc => calcW(res_yd), :empc => calcW(res2_yd))
```

Of course, this gain is only exploitable if the motor electronic includes some kind of
regenerative circuitry.
