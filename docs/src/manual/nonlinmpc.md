# [Manual: Nonlinear Design](@id man_nonlin)

```@contents
Pages = ["nonlinmpc.md"]
```

```@setup 1
using Logging; errlogger = ConsoleLogger(stderr, Error);
old_logger = global_logger(); global_logger(errlogger);
```

## Nonlinear Model

In this example, the goal is to control the angular position ``θ`` of a pendulum
attached to a motor. Knowing that the manipulated input is the motor torque ``τ`` in Nm, the
I/O vectors are:

```math
\begin{aligned}
    \mathbf{u} &= τ \\
    \mathbf{y} &= θ
\end{aligned}
```

The following figure presents the system:

```@raw html
<p><img src="../../assets/pendulum.svg" alt="pendulum" width=200 style="background-color:white; 
    border:20px solid white; display: block; margin-left: auto; margin-right: auto;"/></p>
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
the end of the pendulum in kg, all bundled in the parameter vector ``\mathbf{p} =
[\begin{smallmatrix} g & L & K & m \end{smallmatrix}]'``. The [`NonLinModel`](@ref)
constructor assumes by default that the state function `f` is continuous in time, that is,
an ordinary differential equation system (like here):

```@example 1
using ModelPredictiveControl
function f(x, u, _ , p)
    g, L, K, m = p          # [m/s²], [m], [kg/s], [kg]
    θ, ω = x[1], x[2]       # [rad], [rad/s]
    τ  = u[1]               # [Nm]
    dθ = ω
    dω = -g/L*sin(θ) - K/m*ω + τ/m/L^2
    return [dθ, dω]
end
h(x, _ , _ ) = [180/π*x[1]] # [°]
p_model = [9.8, 0.4, 1.2, 0.3]
nu, nx, ny, Ts = 1, 2, 1, 0.1
vu, vx, vy = ["\$τ\$ (Nm)"], ["\$θ\$ (rad)", "\$ω\$ (rad/s)"], ["\$θ\$ (°)"]
model = setname!(NonLinModel(f, h, Ts, nu, nx, ny; p=p_model); u=vu, x=vx, y=vy)
```

The output function ``\mathbf{h}`` converts the ``θ`` angle to degrees. Note that special
characters like ``θ`` can be typed in the Julia REPL or VS Code by typing `\theta` and
pressing the `<TAB>` key. Note that the model parameter `p` can be any 
 A 4th order [`RungeKutta`](@ref) method solves the differential
equations by default. It is good practice to first simulate `model` using [`sim!`](@ref) as
a quick sanity check:

```@example 1
using Plots
u = [0.5]
N = 35
res = sim!(model, N, u)
plot(res, plotu=false)
savefig("plot1_NonLinMPC.svg"); nothing # hide
```

![plot1_NonLinMPC](plot1_NonLinMPC.svg)

The [`setname!`](@ref) function allows customized Y-axis labels. The available plotting
options are detailed in the documentation of the corresponding [`plot`](@ref ModelPredictiveControl.plot)
method.

## Nonlinear Model Predictive Controller

An [`UnscentedKalmanFilter`](@ref) estimates the plant state :

```@example 1
α=0.01; σQ=[0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
estim = UnscentedKalmanFilter(model; α, σQ, σR, nint_u, σQint_u)
```

The vectors `σQ` and `σR` are the standard deviations of the process and sensor noises,
respectively. The value for the velocity ``ω`` is higher here (`σQ` second value) since
``\dot{ω}(t)`` equation includes an uncertain parameter: the friction coefficient ``K``.
Also, the argument `nint_u` explicitly adds one integrating state at the model input, the
motor torque ``τ``, with an associated standard deviation `σQint_u` of 0.1 N m. The
estimator tuning is tested on a plant with a 25 % larger friction coefficient ``K``:

```@example 1
p_plant = copy(p_model)
p_plant[3] = 1.25*p_model[3]
plant = setname!(NonLinModel(f, h, Ts, nu, nx, ny; p=p_plant); u=vu, x=vx, y=vy)
res = sim!(estim, N, [0.5], plant=plant, y_noise=[0.5])
plot(res, plotu=false, plotxwithx̂=true)
savefig("plot2_NonLinMPC.svg"); nothing # hide
```

![plot2_NonLinMPC](plot2_NonLinMPC.svg)

The estimate ``x̂_3`` is the integrating state on the torque ``τ`` that compensates for
static errors. The Kalman filter performance seems sufficient for control.

As the motor torque is limited to -1.5 to 1.5 N m, we incorporate the input constraints in
a [`NonLinMPC`](@ref):

```@example 1
Hp, Hc, Mwt, Nwt = 20, 2, [0.5], [2.5]
nmpc = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt=Inf)
umin, umax = [-1.5], [+1.5]
nmpc = setconstraint!(nmpc; umin, umax)
```

The option `Cwt=Inf` disables the slack variable `ϵ` for constraint softening. We test `mpc`
performance on `plant` by imposing an angular setpoint of 180° (inverted position):

```@example 1
using JuMP; unset_time_limit_sec(nmpc.optim) # hide
res_ry = sim!(nmpc, N, [180.0], plant=plant, x_0=[0, 0], x̂_0=[0, 0, 0])
plot(res_ry)
savefig("plot3_NonLinMPC.svg"); nothing # hide
```

![plot3_NonLinMPC](plot3_NonLinMPC.svg)

The controller seems robust enough to variations on ``K`` coefficient. Starting from this
inverted position, the closed-loop response to a step disturbances of 10° is also
satisfactory:

```@example 1
res_yd = sim!(nmpc, N, [180.0], plant=plant, x_0=[π, 0], x̂_0=[π, 0, 0], y_step=[10])
plot(res_yd)
savefig("plot4_NonLinMPC.svg"); nothing # hide
```

![plot4_NonLinMPC](plot4_NonLinMPC.svg)

See [`sim!`](@ref) documentation for details on the possible simulation scenarios.

## Economic Model Predictive Controller

Economic MPC can achieve the same objective but with lower economical costs. For this case
study, the controller will aim to reduce the energy consumed by the motor. The power in watt
transmitted by the motor to the pendulum is:

```math
P(t) = τ(t) ω(t)
```

Thus, the work in joule done by the motor from ``t = t_0`` to ``t_{end}`` is:

```math
W = \int_{t_0}^{t_{end}} P(t) \mathrm{d}t = \int_{t_0}^{t_{end}} τ(t) ω(t) \mathrm{d}t
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
h2(x, _ , _ ) = [180/π*x[1], x[2]]
nu, nx, ny = 1, 2, 2
model2 = setname!(NonLinModel(f, h2, Ts, nu, nx, ny; p=p_model), u=vu, x=vx, y=[vy; vx[2]])
plant2 = setname!(NonLinModel(f, h2, Ts, nu, nx, ny; p=p_plant), u=vu, x=vx, y=[vy; vx[2]])
estim2 = UnscentedKalmanFilter(model2; σQ, σR, nint_u, σQint_u, i_ym=[1])
```

The `plant2` object based on `h2` is also required since [`sim!`](@ref) expects that the
output vector of `plant` argument corresponds to the model output vector in `mpc` argument.
We can now define the ``J_E`` function and the `empc` controller:

```@example 1
function JE(UE, ŶE, _ , Ts)
    τ, ω = UE[1:end-1], ŶE[2:2:end-1]
    return Ts*sum(τ.*ω)
end
empc = NonLinMPC(estim2; Hp, Hc, Nwt, Mwt=[0.5, 0], Cwt=Inf, Ewt=3.5e3, JE=JE, p=Ts)
empc = setconstraint!(empc; umin, umax)
```

The keyword argument `Ewt` weights the economic costs relative to the other terms in the
objective function. The term must be large enough to be significant but a too high value can
lead to a static error on the angle setpoint. The second element of `Mwt` is zero since the
speed ``ω`` is not requested to track a setpoint. The closed-loop response to a 180°
setpoint is similar:

```@example 1
unset_time_limit_sec(empc.optim)    # hide
res2_ry = sim!(empc, N, [180, 0], plant=plant2, x_0=[0, 0], x̂_0=[0, 0, 0])
plot(res2_ry, ploty=[1])
savefig("plot5_NonLinMPC.svg"); nothing # hide
```

![plot5_NonLinMPC](plot5_NonLinMPC.svg)

and the energy consumption is slightly lower:

```@example 1
function calcW(res)
    τ, ω = res.U_data[1, 1:end-1], res.X_data[2, 1:end-1]
    return Ts*sum(τ.*ω)
end
Dict(:W_nmpc => calcW(res_ry), :W_empc => calcW(res2_ry))
```

Also, for a 10° step disturbance:

```@example 1
res2_yd = sim!(empc, N, [180; 0]; plant=plant2, x_0=[π, 0], x̂_0=[π, 0, 0], y_step=[10, 0])
plot(res2_yd, ploty=[1])
savefig("plot6_NonLinMPC.svg"); nothing # hide
```

![plot6_NonLinMPC](plot6_NonLinMPC.svg)

the new controller is able to recuperate a little more energy from the pendulum (i.e.
negative work):

```@example 1
Dict(:W_nmpc => calcW(res_yd), :W_empc => calcW(res2_yd))
```

Of course, this gain is only exploitable if the motor electronic includes some kind of
regenerative circuitry.

## Model Linearization

Nonlinear MPC is more computationally expensive than [`LinMPC`](@ref). Solving the problem
should always be faster than the sampling time ``T_s = 0.1`` s for real-time operation. This
requirement is sometimes hard to meet on electronics or mechanical systems because of the
fast dynamics. To ease the design and comparison with [`LinMPC`](@ref), the [`linearize`](@ref)
function allows automatic linearization of [`NonLinModel`](@ref) based on [`ForwardDiff.jl`](https://juliadiff.org/ForwardDiff.jl/stable/).
We first linearize `model` at the point ``θ = π`` rad and ``ω = τ = 0`` (inverted position):

```@example 1
linmodel = linearize(model, x=[π, 0], u=[0])
```

A [`SteadyKalmanFilter`](@ref) and a [`LinMPC`](@ref) are designed from `linmodel`:

```@example 1

skf = SteadyKalmanFilter(linmodel; σQ, σR, nint_u, σQint_u)
mpc = LinMPC(skf; Hp, Hc, Mwt, Nwt, Cwt=Inf)
mpc = setconstraint!(mpc, umin=[-1.5], umax=[+1.5])
```

The linear controller satisfactorily rejects the 10° step disturbance:

```@example 1
res_lin = sim!(mpc, N, [180.0]; plant, x_0=[π, 0], y_step=[10])
plot(res_lin)
savefig("plot7_NonLinMPC.svg"); nothing # hide
```

![plot7_NonLinMPC](plot7_NonLinMPC.svg)

Solving the optimization problem of `mpc` with [`DAQP`](https://darnstrom.github.io/daqp/)
optimizer instead of the default `OSQP` solver can improve the performance here. It is
indeed documented that `DAQP` can perform better on small/medium dense matrices and unstable
poles[^1], which is obviously the case here (absolute value of unstable poles are greater
than one):

[^1]: Arnström, D., Bemporad, A., and Axehill, D. (2022). A dual active-set solver for
    embedded quadratic programming using recursive LDLᵀ updates. IEEE Trans. Autom. Contr.,
    67(8). <https://doi.org/doi:10.1109/TAC.2022.3176430>.

```@example 1
using LinearAlgebra; poles = eigvals(linmodel.A)
```

To install the solver, run:

```text
using Pkg; Pkg.add("DAQP")
```

Constructing a [`LinMPC`](@ref) with `DAQP`:

```@example 1
using JuMP, DAQP
daqp = Model(DAQP.Optimizer, add_bridges=false)
mpc2 = LinMPC(skf; Hp, Hc, Mwt, Nwt, Cwt=Inf, optim=daqp)
mpc2 = setconstraint!(mpc2; umin, umax)
```

does slightly improve the rejection of the step disturbance:

```@example 1
res_lin2 = sim!(mpc2, N, [180.0]; plant, x_0=[π, 0], y_step=[10])
plot(res_lin2)
savefig("plot8_NonLinMPC.svg"); nothing # hide
```

![plot8_NonLinMPC](plot8_NonLinMPC.svg)

The closed-loop performance is still lower than the nonlinear controller, as expected, but
computations are about 210 times faster (0.000071 s versus 0.015 s per time steps, on
average). However, remember that `linmodel` is only valid for angular positions near 180°.
For example, the 180° setpoint response from 0° is unsatisfactory since the predictions are
poor in the first quadrant:

```@example 1
res_lin3 = sim!(mpc2, N, [180.0]; plant, x_0=[0, 0])
plot(res_lin3)
savefig("plot9_NonLinMPC.svg"); nothing # hide
```

![plot9_NonLinMPC](plot9_NonLinMPC.svg)

Multiple linearized model and controller objects are required for large deviations from this
operating point. This is known as gain scheduling. Another approach is adapting the model of
the [`LinMPC`](@ref) instance based on repeated online linearization.

## Adapting the Model via Successive Linearization

The [`setmodel!`](@ref) method allows online adaptation of a linear plant model. Combined
with the automatic linearization of [`linearize`](@ref), a successive linearization MPC can
be designed with minimal efforts. The [`SteadyKalmanFilter`](@ref) does not support
[`setmodel!`](@ref) so we need to use the time-varying [`KalmanFilter`](@ref), and we
initialize it with a linearization at ``θ = ω = τ = 0``:

```@example 1
linmodel = linearize(model, x=[0, 0], u=[0])
kf = KalmanFilter(linmodel; σQ, σR, nint_u, σQint_u)
mpc3 = LinMPC(kf; Hp, Hc, Mwt, Nwt, Cwt=Inf, optim=daqp)
mpc3 = setconstraint!(mpc3; umin, umax)
```

We create a function that simulates the plant and the adaptive controller:

```@example 1
function sim_adapt!(mpc, nonlinmodel, N, ry, plant, x_0, x̂_0, y_step=[0])
    U_data, Y_data, Ry_data = zeros(plant.nu, N), zeros(plant.ny, N), zeros(plant.ny, N)
    setstate!(plant, x_0)
    initstate!(mpc, [0], plant())
    setstate!(mpc, x̂_0)
    for i = 1:N
        y = plant() + y_step
        x̂ = preparestate!(mpc, y)
        u = moveinput!(mpc, ry)
        linmodel = linearize(nonlinmodel; u, x=x̂[1:2])
        setmodel!(mpc, linmodel)
        U_data[:,i], Y_data[:,i], Ry_data[:,i] = u, y, ry
        updatestate!(mpc, u, y) # update mpc state estimate
        updatestate!(plant, u)  # update plant simulator
    end
    res = SimResult(mpc, U_data, Y_data; Ry_data)
    return res
end
nothing # hide
```

The [`setmodel!`](@ref) method must be called after solving the optimization problem with
[`moveinput!`](@ref) and before updating the state estimate with [`updatestate!`](@ref),
that is, when both ``\mathbf{u}(k)`` and ``\mathbf{x̂}_{k}(k)`` are available as the new
operating point. The [`SimResult`](@ref) object is for plotting purposes only. The adaptive
[`LinMPC`](@ref) performances are similar to the nonlinear MPC, both for the 180° setpoint:

```@example 1
x_0 = [0, 0]; x̂_0 = [0, 0, 0]; ry = [180]
res_slin = sim_adapt!(mpc3, model, N, ry, plant, x_0, x̂_0)
plot(res_slin)
savefig("plot10_NonLinMPC.svg"); nothing # hide
```

![plot10_NonLinMPC](plot10_NonLinMPC.svg)

and the 10° step disturbance:

```@example 1
x_0 = [π, 0]; x̂_0 = [π, 0, 0]; y_step = [10]
res_slin = sim_adapt!(mpc3, model, N, ry, plant, x_0, x̂_0, y_step)
plot(res_slin)
savefig("plot11_NonLinMPC.svg"); nothing # hide
```

![plot11_NonLinMPC](plot11_NonLinMPC.svg)

The computations of the successive linearization MPC are about 125 times faster than the
nonlinear MPC (0.00012 s per time steps versus 0.015 s per time steps, on average), an
impressive gain for similar closed-loop performances!

```@setup 1
global_logger(old_logger);
```
