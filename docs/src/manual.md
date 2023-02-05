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

At the operating points:

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
    \frac{1.90}{18s + 1} & \frac{1.90}{18s + 1} \\[2pt]
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

```julia
using ControlSystemsBase
using ModelPredictiveControl
sys = [ tf(1.90, [18, 1]) tf(1.90, [18, 1]);
        tf(-0.74,[8, 1])  tf(0.74, [8, 1]) ]
Ts = 4.0
model = setop!(LinModel(sys, Ts), uop=[10, 10], yop=[50, 30])
```

The `model` object will be used for two purposes : to construct our controller, and as a
plant simulator to test the design. We design our [`LinMPC`](@ref) controllers by including
the level constraint with [`setconstraint`](@ref):

```julia
mpc = setconstraint!(LinMPC(model, Hp=15, Hc=2), ŷmin=[45, -Inf])
```

Before closing the loop, we call [`initstate!`](@ref) with the actual plant inputs and
measurements to ensure a bumpless transfer. Since `model` simulates our plant here, we
evaluate its outputs to initialize the states. [`LinModel`](@ref) objects are callable for
this purpose (an alias for [`evaloutput`](@ref)):

```julia
u = model.uop
y = model() # or equivalently : y = evaloutput(model)
initstate!(mpc, u, y)
```

We can then close the loop and test `mpc` performance on the simulator by imposing step
changes on output setpoints ``\mathbf{r_y}`` and on a load disturbance ``\mathbf{u_d}``:

```julia
N = 100
t_data  = Ts*(0:N-1)
ry, ud  = [50, 30], [0, 0]
u_data  = zeros(model.nu, N)
y_data  = zeros(model.ny, N)
ry_data = zeros(model.ny, N)
for k = 0:N-1
    k == 0  && (ry = [50, 35])
    k == 25 && (ry = [55, 35])
    k == 50 && (ry = [50, 30])
    k == 75 && (ud = [-10, 0])
    y = model() # simulated measurements
    u = mpc(ry) # or equivalently : u = moveinput!(mpc, ry)
    u_data[:,k+1]  = u
    y_data[:,k+1]  = y
    ry_data[:,k+1] = ry 
    updatestate!(mpc, u, y) # update mpc state estimate
    updatestate!(model, u + ud)  # update simulated plant + load disturbance
end
```

The [`LinMPC`](@ref) objects are also callable to provide an alternative syntax for
the [`moveinput!`] method. Calling [updatestate!](@ref) on the `mpc` object updates its
internal state for the **NEXT** control period. That is why the call is done at the end
of the `for` loop. The same logic applies for the `model` object. Lastly, we plot the
closed-loop test with the `Plots` package:

```julia
using Plots
p1 = plot(t_data, y_data[1,:], label=raw"$y_L$")
plot!(t_data,r_data[1,:],label=raw"$r_L$", linestyle=:dash)
p2 = plot(t_data,y_data[2,:],label=raw"$y_T$")
plot!(t_data,r_data[2,:],label=raw"$r_T$", linestyle=:dash)
py = plot(p1, p2, layout=[1,1])
p1 = plot(t_data,u_data[1,:],label=raw"$u_1$", linetype=:steppost)
p2 = plot(t_data,u_data[2,:],label=raw"$u_2$", linetype=:steppost)
pu = plot(p1,p2, layout=[1,1])
display(pu)
display(py)
```
