# [Manual: Linear Design](@id man_lin)

```@contents
Pages = ["linmpc.md"]
```

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
    \mathbf{u_{op}} &= \begin{bmatrix} 20 \\ 20 \end{bmatrix} \\
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

```@example 1
using ModelPredictiveControl, ControlSystemsBase
sys = [ tf(1.90, [18, 1]) tf(1.90, [18, 1]);
        tf(-0.74,[8, 1])  tf(0.74, [8, 1]) ]
Ts = 2.0
model = setop!(LinModel(sys, Ts), uop=[20, 20], yop=[50, 30])
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

```@example 1
mpc = LinMPC(model, Hp=15, Hc=2, Mwt=[1, 1], Nwt=[0.1, 0.1])
mpc = setconstraint!(mpc, ymin=[45, -Inf])
```

in which `Hp` and `Hc` keyword arguments are respectively the predictive and control
horizons, and `Mwt` and `Nwt`, the output setpoint tracking and move suppression weights. By
default, [`LinMPC`](@ref) controllers use [`OSQP`](https://osqp.org/) to solve the problem,
soft constraints on output predictions ``\mathbf{ŷ}`` to ensure feasibility, and a
[`SteadyKalmanFilter`](@ref) to estimate the plant states[^1]. An attentive reader will also
notice that the Kalman filter estimates two additional states compared to the plant model.
These are the integrating states for the unmeasured plant disturbances, and they are
automatically added to the model outputs by default if feasible (see [`SteadyKalmanFilter`](@ref)
for details).

[^1]: To avoid observer design, we could have use an [`InternalModel`](@ref) structure with
    `mpc = LinMPC(InternalModel(model), Hp=15, Hc=2, Mwt=[1, 1], Nwt=[0.1, 0.1])` . It was
    tested on the example of this page and it gives similar results.

Before closing the loop, we call [`initstate!`](@ref) with the actual plant inputs and
measurements to ensure a bumpless transfer. Since `model` simulates our plant here, its
output will initialize the states. [`LinModel`](@ref) objects are callable for this purpose
(an alias for [`evaloutput`](@ref)):

```@example 1
u, y = model.uop, model() # or equivalently : y = evaloutput(model)
initstate!(mpc, u, y)
nothing # hide
```

We can then close the loop and test `mpc` performance on the simulator by imposing step
changes on output setpoints ``\mathbf{r_y}`` and on a load disturbance ``u_l``:

```@example 1
function test_mpc(mpc, model)
    N = 200
    ry, ul = [50, 30], 0
    u_data  = zeros(model.nu, N)
    y_data  = zeros(model.ny, N)
    ry_data = zeros(model.ny, N)
    for k = 0:N-1
        k == 50  && (ry = [50, 35])
        k == 100 && (ry = [54, 30])
        k == 150 && (ul = -20)
        y = model() # simulated measurements
        u = mpc(ry) # or equivalently : u = moveinput!(mpc, ry)
        u_data[:,k+1]  = u
        y_data[:,k+1]  = y
        ry_data[:,k+1] = ry 
        updatestate!(mpc, u, y) # update mpc state estimate
        updatestate!(model, u + [0; ul]) # update simulator with the load disturbance
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

```@example 1
using Plots
function plot_data(t_data, u_data, y_data, ry_data)
    p1 = plot(t_data, y_data[1,:], label="meas."); ylabel!("level")
    plot!(t_data, ry_data[1,:], label="setpoint", linestyle=:dash, linetype=:steppost)
    plot!(t_data, fill(45,size(t_data)), label="min", linestyle=:dot, linewidth=1.5)
    p2 = plot(t_data, y_data[2,:], label="meas.", legend=:topleft); ylabel!("temp.")
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
improve the closed-loop performance. Load disturbances are indeed very frequent in many
real-life control problems. Constructing a [`LinMPC`](@ref) with `DAQP` and input integrators:

```@example 1
using JuMP, DAQP
daqp = Model(DAQP.Optimizer)
mpc2 = LinMPC(model, Hp=15, Hc=2, Mwt=[1, 1], Nwt=[0.1, 0.1], optim=daqp, nint_u=[1, 1])
mpc2 = setconstraint!(mpc2, ymin=[45, -Inf])
```

leads to similar computational times, but it does accelerate the rejection of the load
disturbance and eliminates the level constraint violation:

```@example 1
setstate!(model, zeros(model.nx))
u, y = model.uop, model()
initstate!(mpc2, u, y)
u_data, y_data, ry_data = test_mpc(mpc2, model)
plot_data(t_data, u_data, y_data, ry_data)
```

## Adding Feedforward Compensation

Suppose that the load disturbance ``u_l`` of the last section is in fact caused by a
separate hot water pipe that discharges into the tank. Measuring this flow rate allows us to
incorporate feedforward compensation in the controller. The new plant model is:

```math
\begin{bmatrix}
    y_L(s) \\ y_T(s)
\end{bmatrix} = 
\begin{bmatrix}
    \frac{1.90}{18s + 1} & \frac{1.90}{18s + 1} & \frac{1.90}{18s + 1} \\[3pt]
    \frac{-0.74}{8s + 1} & \frac{0.74}{8s + 1}  & \frac{0.74}{8s + 1}
\end{bmatrix}
\begin{bmatrix}
    u_c(s) \\ u_h(s) \\ u_l(s)
\end{bmatrix}
```

We need to construct a new [`LinModel`](@ref) that includes the measured disturbance
``\mathbf{d} = [u_l]`` and the operating point ``\mathbf{d_{op}} = [20]``:

```@example 1
sys_ff   = [sys sys[1:2, 2]]
model_ff = setop!(LinModel(sys_ff, Ts, i_d=[3]), uop=[20, 20], yop=[50, 30], dop=[20])
```

A [`LinMPC`](@ref) controller is constructed on this model:

```@example 1
mpc_ff = LinMPC(model_ff, Hp=15, Hc=2, Mwt=[1, 1], Nwt=[0.1, 0.1], nint_u=[1, 1])
mpc_ff = setconstraint!(mpc_ff, ymin=[45, -Inf])
```

A new test function that feeds the measured disturbance ``\mathbf{d}`` to the controller is
also required:

```@example 1
function test_mpc_ff(mpc_ff, model)
    N = 200
    ry, ul = [50, 30], 0
    dop = mpc_ff.estim.model.dop
    u_data  = zeros(model.nu, N)
    y_data  = zeros(model.ny, N)
    ry_data = zeros(model.ny, N)
    for k = 0:N-1
        k == 50  && (ry = [50, 35])
        k == 100 && (ry = [54, 30])
        k == 150 && (ul = -20)
        d = ul .+ dop   # simulated measured disturbance
        y = model()     # simulated measurements
        u = mpc_ff(ry, d) # also feed the measured disturbance d to the controller
        u_data[:,k+1]  = u
        y_data[:,k+1]  = y
        ry_data[:,k+1] = ry 
        updatestate!(mpc_ff, u, y, d)    # update estimate with the measured disturbance d
        updatestate!(model, u + [0; ul]) # update simulator
    end
    return u_data, y_data, ry_data
end
nothing # hide
```

The new feedforward compensation is able to almost perfectly reject the load disturbance:

```@example 1
setstate!(model, zeros(model.nx))
u, y, d = model.uop, model(), mpc_ff.estim.model.dop
initstate!(mpc_ff, u, y, d)
u_data, y_data, ry_data = test_mpc_ff(mpc_ff, model)
plot_data(t_data, u_data, y_data, ry_data)
```
