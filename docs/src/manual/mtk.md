# [Manual: ModelingToolkit Integration](@id man_mtk)

```@contents
Pages = ["mtk.md"]
```

```@setup 1
using Logging; errlogger = ConsoleLogger(stderr, Error);
old_logger = global_logger(); global_logger(errlogger);
```

## Pendulum Model

This example integrates the simple pendulum model of the [last section](@ref man_nonlin) in the
[`ModelingToolkit.jl`](https://docs.sciml.ai/ModelingToolkit/stable/) (MTK) framework and
extracts appropriate `f!` and `h!` functions to construct a [`NonLinModel`](@ref). An
[`NonLinMPC`](@ref) is designed from this model and simulated to reproduce the results of
the last section.

!!! danger "Disclaimer"
    This simple example is not an official interface to `ModelingToolkit.jl`. It is provided
    as a basic starting template to combine both packages. There is no guarantee that it
    will work for all corner cases.

We first construct and instantiate the pendulum model:

```@example 1
using ModelPredictiveControl, ModelingToolkit
using ModelingToolkit: D_nounits as D, t_nounits as t, varmap_to_vars
@mtkmodel Pendulum begin
    @parameters begin
        g = 9.8
        L = 0.4
        K = 1.2
        m = 0.3
    end
    @variables begin
        θ(t) # state
        ω(t) # state
        τ(t) # input
        y(t) # output
    end
    @equations begin
        D(θ)    ~ ω
        D(ω)    ~ -g/L*sin(θ) - K/m*ω + τ/m/L^2
        y       ~ θ * 180 / π
    end
end
@named mtk_model = Pendulum()
mtk_model = complete(mtk_model)
```

We than convert the MTK model to an [input-output system](https://docs.sciml.ai/ModelingToolkit/stable/basics/InputOutput/):

```@example 1
function generate_f_h(model, inputs, outputs)
    (_, f_ip), dvs, psym, io_sys = ModelingToolkit.generate_control_function(
        model, inputs, split=false; outputs
    )
    if any(ModelingToolkit.is_alg_equation, equations(io_sys)) 
        error("Systems with algebraic equations are not supported")
    end
    h_ = ModelingToolkit.build_explicit_observed_function(io_sys, outputs; inputs = inputs)
    nx = length(dvs)
    vx = string.(dvs)
    par = varmap_to_vars(defaults(io_sys), psym)
    function f!(ẋ, x, u, _ , _ )
        f_ip(ẋ, x, u, par, 1)
        nothing
    end
    function h!(y, x, _ , _ )
        y .= h_(x, 1, par, 1)
        nothing
    end
    return f!, h!, nx, vx
end
inputs, outputs = [mtk_model.τ], [mtk_model.y]
f!, h!, nx, vx = generate_f_h(mtk_model, inputs, outputs)
nu, ny, Ts = length(inputs), length(outputs), 0.1
vu, vy = ["\$τ\$ (Nm)"], ["\$θ\$ (°)"]
nothing # hide
```

A [`NonLinModel`](@ref) can now be constructed:

```@example 1
model = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny); u=vu, x=vx, y=vy)
```

We also instantiate a plant model with a 25 % larger friction coefficient ``K``:

```@example 1
mtk_model.K = defaults(mtk_model)[mtk_model.K] * 1.25
f_plant, h_plant, _, _ = generate_f_h(mtk_model, inputs, outputs)
plant = setname!(NonLinModel(f_plant, h_plant, Ts, nu, nx, ny); u=vu, x=vx, y=vy)
```

## Controller Design

We can than reproduce the Kalman filter and the controller design of the [last section](@ref man_nonlin):

```@example 1
α=0.01; σQ=[0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
estim = UnscentedKalmanFilter(model; α, σQ, σR, nint_u, σQint_u)
Hp, Hc, Mwt, Nwt = 20, 2, [0.5], [2.5]
nmpc = NonLinMPC(estim; Hp, Hc, Mwt, Nwt, Cwt=Inf)
umin, umax = [-1.5], [+1.5]
nmpc = setconstraint!(nmpc; umin, umax)
```

The 180° setpoint response is identical:

```@example 1
using Plots
N = 35
using JuMP; unset_time_limit_sec(nmpc.optim) # hide
res_ry = sim!(nmpc, N, [180.0], plant=plant, x_0=[0, 0], x̂_0=[0, 0, 0])
plot(res_ry)
savefig("plot1_MTK.svg"); nothing # hide
```

![plot1_MTK](plot1_MTK.svg)

and also the output disturbance rejection:

```@example 1
res_yd = sim!(nmpc, N, [180.0], plant=plant, x_0=[π, 0], x̂_0=[π, 0, 0], y_step=[10])
plot(res_yd)
savefig("plot2_MTK.svg"); nothing # hide
```

![plot2_MTK](plot2_MTK.svg)

## Acknowledgement

Authored by `1-Bart-1` and `baggepinnen`, thanks for the contribution.

```@setup 1
global_logger(old_logger);
```
