# [Manual: ModelingToolkit Integration](@id man_mtk)

```@contents
Pages = ["mtk.md"]
```

```@setup 1
using Logging; errlogger = ConsoleLogger(stderr, Error);
old_logger = global_logger(); global_logger(errlogger);
```

## Pendulum Model

This example integrates the simple pendulum model of the [last section](@ref man_nonlin) in
[`ModelingToolkit`](https://docs.sciml.ai/ModelingToolkit/stable/) (MTK) framework and
extracts appropriate `f!` and `h!` functions to construct a [`NonLinModel`](@ref). An
[`NonLinMPC`](@ref) is designed from this model and simulated to reproduce the results of
the last section.

!!! danger "Disclaimer"
    This simple example is not an official interface to `ModelingToolkit.jl`. It is provided
    as a basic starting template to combine both packages. There is no guarantee that it
    will work for all corner cases.

!!! compat
    The example works on `ModelingToolkit.jl` v9.50 to 9.76 (corresponding to the following
    `[compat]` entry: `ModelingToolkit = "9.50 - 9.76"`).

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
    nu, nx, ny = length(inputs), length(dvs), length(outputs)
    vx = string.(dvs)
    p = varmap_to_vars(defaults(io_sys), psym)
    function f!(ẋ, x, u, _ , p)
        try
            f_ip(ẋ, x, u, p, nothing)
        catch err
            if err isa MethodError
                error("NonLinModel does not support a time argument t in the f function, "*
                      "see the constructor docstring for a workaround.")
            else
                rethrow()
            end
        end
        return nothing
    end
    (_, h_ip) = ModelingToolkit.build_explicit_observed_function(
        io_sys, outputs; inputs, return_inplace = true
    )
    u_nothing = fill(nothing, nu)
    function h!(y, x, _ , p)
        try
            # MTK.jl supports a `u` argument in `h_ip` function but not this package. We set
            # `u` as a vector of nothing and `h_ip` function will presumably throw an
            # MethodError it this argument is used inside the function
            h_ip(y, x, u_nothing, p, nothing)
        catch err
            if err isa MethodError
                error("NonLinModel only support strictly proper systems (no manipulated "*
                      "input argument u in the output function h)")
            else
                rethrow()
            end
        end
        return nothing
    end
    return f!, h!, p, nu, nx, ny, vx
end
inputs, outputs = [mtk_model.τ], [mtk_model.y]
f!, h!, p, nu, nx, ny, vx = generate_f_h(mtk_model, inputs, outputs)
Ts = 0.1
vu, vy = ["\$τ\$ (Nm)"], ["\$θ\$ (°)"]
nothing # hide
```

A [`NonLinModel`](@ref) can now be constructed:

```@example 1
model = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny; p); u=vu, x=vx, y=vy)
```

We also instantiate a plant model with a 25 % larger friction coefficient ``K``:

```@example 1
mtk_model.K = defaults(mtk_model)[mtk_model.K] * 1.25
f_plant, h_plant, p = generate_f_h(mtk_model, inputs, outputs)
plant = setname!(NonLinModel(f_plant, h_plant, Ts, nu, nx, ny; p); u=vu, x=vx, y=vy)
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
