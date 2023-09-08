# [Manual: Linear Design](@id man_lin)

## Linear Model

The example considers a continuously stirred-tank reactor (CSTR) with a cold and hot water
inlet as a plant. The water flows out of an opening at the bottom of the tank. The
manipulated inputs are the cold ``u_c`` and hot ``u_h`` water flow rates, and the measured
outputs are the liquid level ``y_L`` and temperature ``y_T``:

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

We first need to construct a [`LinModel`](@ref) objet with [`setop!`](@ref) to handle the
operating points:

```julia
using ModelPredictiveControl, ControlSystemsBase
sys = [ tf(1.90, [18, 1]) tf(1.90, [18, 1]);
        tf(-0.74,[8, 1])  tf(0.74, [8, 1]) ]
Ts = 2.0
model = setop!(LinModel(sys, Ts), uop=[10, 10], yop=[50, 30])
```

The `model` object will be used for two purposes : to construct our controller, and as a
plant simulator to test the design.

## Linear Model Predictive Controller

A linear model predictive controller (MPC) will control both the water level ``y_L`` and
temperature ``y_T`` in the tank, at a sampling time of 4 s. The tank level should also never
fall below 45:

```math
y_L ≥ 45
```

We design our [`LinMPC`](@ref) controllers by including the linear level constraint with
[`setconstraint!`](@ref) (`±Inf` values should be used when there is no bound):

```julia
mpc = setconstraint!(LinMPC(model, Hp=15, Hc=2), ŷmin=[45, -Inf])
```

in which `Hp` and `Hc` keyword arguments are the predictive and control horizons
respectively. By default, [`LinMPC`](@ref) controllers use [`OSQP`](https://osqp.org/) to
solve the problem, soft constraints on output predictions ``\mathbf{ŷ}`` to ensure
feasibility, and a [`SteadyKalmanFilter`](@ref) to estimate the plant states. An attentive
reader will also notice that the Kalman filter estimates two additional states compared to
the plant model. These are the integrating states for the unmeasured plant disturbances, and
they are automatically added the model outputs if feasible (see [`SteadyKalmanFilter`](@ref)
for details).

Before closing the loop, we call [`initstate!`](@ref) with the actual plant inputs and
measurements to ensure a bumpless transfer. Since `model` simulates our plant here, its
output will initialize the states. [`LinModel`](@ref) objects are callable for this purpose
(an alias for [`evaloutput`](@ref)):

```julia
u = model.uop
y = model() # or equivalently : y = evaloutput(model)
initstate!(mpc, u, y)
nothing # hide
```

We can then close the loop and test `mpc` performance on the simulator by imposing step
changes on output setpoints ``\mathbf{r_y}`` and on a load disturbance ``\mathbf{u_d}``:

```julia
function test_mpc(mpc, model)
    N = 200
    ry, ud = [50, 30], [0, 0]
    u_data  = zeros(model.nu, N)
    y_data  = zeros(model.ny, N)
    ry_data = zeros(model.ny, N)
    for k = 0:N-1
        y = model() # simulated measurements
        k == 50  && (ry = [50, 35])
        k == 100 && (ry = [55, 30])
        k == 150 && (ud = [-25, 0])
        u = mpc(ry) # or equivalently : u = moveinput!(mpc, ry)
        u_data[:,k+1]  = u
        y_data[:,k+1]  = y
        ry_data[:,k+1] = ry 
        updatestate!(mpc, u, y) # update mpc state estimate
        updatestate!(model, u + ud) # update simulator with disturbance
        println(getinfo(mpc)[2][:Ŷs])
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
[Functions: State Estimators](@ref) for justifications). That is why the call is done at the
end of the `for` loop. The same logic applies for `model`.

Lastly, we plot the closed-loop test with the `Plots` package:

```julia
using Plots
function plot_data(t_data, u_data, y_data, ry_data)
    p1 = plot(t_data, y_data[1,:], label="meas."); ylabel!("level")
    plot!(t_data, ry_data[1,:], label="setpoint", linestyle=:dash, linetype=:steppost)
    plot!(t_data, fill(45,size(t_data)), label="min", linestyle=:dot, linewidth=1.5)
    p2 = plot(t_data, y_data[2,:], label="meas."); ylabel!("temp.")
    plot!(t_data, ry_data[2,:],label="setpoint", linestyle=:dash, linetype=:steppost)
    p3 = plot(t_data,u_data[1,:],label="cold", linetype=:steppost); ylabel!("flow rate")
    plot!(t_data,u_data[2,:],label="hot", linetype=:steppost); xlabel!("time (s)")
    return plot(p1, p2, p3, layout=(3,1), fmt=:svg)
end
plot_data(t_data, u_data, y_data, ry_data)
```

For some situations, when [`LinMPC`](@ref) matrices are small/medium and dense, [`DAQP`](https://darnstrom.github.io/daqp/)
optimizer may be more efficient. To install it, run:

```text
using Pkg; Pkg.add("DAQP")
```

Also, compared to the default setting, adding the integrating states at the model inputs may
improve the closed-loop performance. Load disturbances are indeed very frequent in real-life
control problems. Constructing a [`LinMPC`](@ref) with `DAQP` and input integrators:

```julia
using JuMP, DAQP
daqp  = Model(DAQP.Optimizer)
estim = SteadyKalmanFilter(model, nint_u=[1, 1])
mpc2  = setconstraint!(LinMPC(estim, Hp=15, Hc=2, optim=daqp), ŷmin=[45, -Inf])
```

leads to identical results here:

```julia
setstate!(model, zeros(model.nx))
initstate!(mpc2, model.uop, model())
u_data2, y_data2, ry_data2 = test_mpc(mpc2, model)
plot_data(t_data, u_data2, y_data2, ry_data2)
```
