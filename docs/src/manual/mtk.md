# [Manual: ModelingToolkit Integration](@id man_mtk)

```@contents
Pages = ["mtk.md"]
```

## Introduction

This example integrates the simple pendulum and pH neutralization models of the
[penultimate section](@ref man_nonlin) and [ultimate section](@ref man_dae), respectively,
in the [`ModelingToolkit`](https://docs.sciml.ai/ModelingToolkit/stable/) (MTK) framework.
It then extracts functions with appropriate signature to construct [`SimModel`](@ref)
objects. This object is then used to design controllers similar to the previous sections.

!!! danger "Disclaimer"
    This simple example is not an official interface to `ModelingToolkit.jl`. It is provided
    as a basic starting template to combine both packages. There is no guarantee that it
    will work for all corner cases.

```@codeblocks
line_counter = :continue
```

## MTK Model Conversion

We first define two methods that convert MTK model to either a [`NonLinModel`](@ref) or a
[`NonLinModelDAE`](@ref), depending on the presence of algebraic equations. The methods rely
on [input-output systems](https://docs.sciml.ai/ModelingToolkit/stable/basics/InputOutput/).
The `generate_f_h_mtk` method extracts the `f_mtk!` and `h_mtk!` functions from the MTK
model, with combined differential states and algebraic variables in a single vector. The
`i_alg` boolean vector indicates which elements of the state vector are algebraic variables:

```@example 1
using ModelPredictiveControl, ModelingToolkit
using ModelingToolkit: D_nounits as D, t_nounits as t

function generate_f_h_mtk(model, inputs, outputs, disturbances=nothing)
    known_disturbance_inputs = disturbances
    (_, f_ip), x_sym, p_sym, io_sys = ModelingToolkit.generate_control_function(
        model, inputs; known_disturbance_inputs, split=false, simplify=true
    )
    i_alg = ModelingToolkit.is_alg_equation.(equations(io_sys))
    nd = isnothing(disturbances) ? 0 : length(disturbances) 
    f_ip_with_d = (nd < 1) ? (ẋ, x, u, p, t, _ ) -> f_ip(ẋ, x, u, p, t) : f_ip
    function f_mtk!(ẋ, x, u, d, p)
        try
            f_ip_with_d(ẋ, x, u, p, nothing, d)
        catch err
            if err isa MethodError
                error("NonLinModel does not support a time argument t in the f "*
                    "function, see the constructor docstring for a workaround.")
            else
                rethrow()
            end
        end
        return nothing
    end
    (_, h_ip) = ModelingToolkit.build_explicit_observed_function(
        io_sys, outputs; inputs, known_disturbance_inputs, return_inplace = true
    )
    h_ip_with_d = (nd < 1) ? (y, x, u, p, t, _ ) -> h_ip(y, x, u, p, t) : h_ip
    u_nothing = fill(nothing, length(inputs))
    function h_mtk!(y, x, d, p)
        try
            # MTK.jl supports a `u` argument in `h_ip` function but not this package. We
            # set `u` as a vector of nothing and `h_ip` function will presumably throw
            # an MethodError if this argument is used inside the function
            h_ip_with_d(y, x, u_nothing, p, nothing, d)
        catch err
            if err isa MethodError
                error("NonLinModel only support strictly proper systems (no "*
                      "manipulated input argument u in the output function h)")
            else
                rethrow()
            end
        end
        return nothing
    end
    ic = initial_conditions(io_sys)
    p_map, x_map = try
        (
            Dict(sym => ic[sym] for sym in p_sym), 
            Dict(sym => ic[sym] for sym in x_sym)
        )
    catch err 
        if err isa KeyError # the key presumably appears in `bindings(io_sys)`:
            error("Non-constant parameter values are not supported (a.k.a. bindings)")
        else
            rethrow()
        end
    end
    p = ModelingToolkit.varmap_to_vars(p_map, p_sym)
    x = ModelingToolkit.varmap_to_vars(x_map, x_sym)
    return f_mtk!, h_mtk!, p_sym, p, x_sym, x, i_alg
end
nothing # hide
```

The `generate_model` method then constructs an adequate [`SimModel`](@ref) object that
internally separates the differential states and algebraic variables, if any, to comply with
this package's interface:

```@example 1
function generate_model(
    f_mtk!::F, h_mtk!::H, Ts, i_alg, inputs, outputs, disturbances=nothing;
    p=nothing, xs_mtk=zeros(length(i_alg)), unit="s", transcription=OrthogonalCollocation()
) where {F<:Function, H<:Function}
    nu, ny = length(inputs), length(outputs)
    nd = isnothing(disturbances) ? 0 : length(disturbances)
    nx_mtk::Int = length(i_alg)
    na::Int     = sum(i_alg)
    nx::Int     = nx_mtk - na
    i_x::Vector{Int} = findall(iszero, i_alg)
    i_a::Vector{Int} = findall(isone, i_alg)
    if na > 0 
        function fq!(ẋ, res, x, a, u, d, p)
            T = promote_type(eltype(x), eltype(a), eltype(u))
            ẋ_mtk, x_mtk = similar(ẋ, T, nx_mtk), similar(x, T, nx_mtk)
            x_mtk[i_x] .= x
            x_mtk[i_a] .= a 
            f_mtk!(ẋ_mtk, x_mtk, u, d, p)
            ẋ   .= @views ẋ_mtk[i_x]
            res .= @views ẋ_mtk[i_a]
            return nothing
        end
        function h!(y, x, a, d, p)
            T = promote_type(eltype(x), eltype(a))
            x_mtk = similar(x, T, nx_mtk)
            x_mtk[i_x] .= x
            x_mtk[i_a] .= a 
            h_mtk!(y, x_mtk, d, p)
            return nothing
        end
        xs_0, as_0 = xs_mtk[i_x], xs_mtk[i_a]
        model = NonLinModelDAE(
            fq!, h!, Ts, nu, nx, na, ny, nd; p, unit, xs_0, as_0, transcription
        )
    else
        f!, h! = f_mtk!, h_mtk!
        xs_0 = xs_mtk
        model = NonLinModel(f!, h!, Ts, nu, nx, ny, nd; p, xs_0, unit)
    end
    return model # deliberately type unstable function
end
nothing # hide
```

## Pendulum Model

We first construct and instantiate the pendulum model as a MTK `System` object:

```@example 1
@parameters g=9.8 L=0.4 K=1.2 m=0.3
@variables θ(t)=0 ω(t)=0 τ(t)=0 y(t)
eqs = [
    D(θ) ~ ω
    D(ω) ~ -g/L*sin(θ) - K/m*ω + τ/m/L^2
    y ~ θ * 180 / π
]
@named mtk_model = System(eqs, t)
```

We can now convert the MTK model:

```@example 1
inputs, outputs = [τ], [y]
f_mtk!, h_mtk!, p_sym, p, x_sym, x, i_alg = generate_f_h_mtk(mtk_model, inputs, outputs)
[x_sym x]
```

Since MTK is an acausal modeling framework, we do not have control over the state
realization chosen by the package. The content of `x_sym` above shows it settled for the
state vector ``\mathbf{x}(t) = [\begin{smallmatrix}ω(t) && θ(t)\end{smallmatrix}]'``,
that is, the states of the [pendulum section](@ref man_nonlin) in the reverse order. As the
same also applies to the parameters, the `p_sym` object informs on how the `p` vector is
sorted:

```@example 1
[p_sym p]
```

We can now construct a [`SimModel`](@ref) with this specific state realization:

```@example 1
Ts = 0.1
model = generate_model(f_mtk!, h_mtk!, Ts, i_alg, inputs, outputs; p)
vu, vx, vy = ["\$τ\$ (Nm)"], ["\$ω\$ (rad/s)", "\$θ\$ (rad)"], ["\$θ\$ (°)"]
model = setname!(model; u=vu, x=vx, y=vy)
```

As expected, the returned model is a [`NonLinModel`](@ref) object, since the pendulum model
has no algebraic equations. We also instantiate a plant model with a 25 % larger friction
coefficient ``K``:

```@example 1
i_K = findfirst(==("K"), string.(p_sym))
p2 = copy(p)
p2[i_K] = 1.25*p[i_K]
plant = generate_model(f_mtk!, h_mtk!, Ts, i_alg, inputs, outputs; p=p2)
plant = setname!(plant; u=vu, x=vx, y=vy)
```

## Pendulum Controller

We can then reproduce the Kalman filter and the controller design of the [penultimate section](@ref man_nonlin)
by reversing the order of `σQ` vector, because of the different state realization:

```@example 1
α=0.01; σQ=[1.0, 0.1]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
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
import JuMP; JuMP.unset_time_limit_sec(nmpc.optim) # hide
using Logging # hide
res_ry = with_logger(ConsoleLogger(stderr, Error)) do # hide
res_ry = sim!(nmpc, N, [180.0], plant=plant, x_0=[0, 0], x̂_0=[0, 0, 0])
end #
p = plot(res_ry)
savefig(p, "plot1_MTK.svg") # hide
display(p) # hide
```

![plot1_MTK](plot1_MTK.svg)

and also the output disturbance rejection:

```@example 1
res_yd = with_logger(ConsoleLogger(stderr, Error)) do # hide
res_yd = sim!(nmpc, N, [180.0], plant=plant, x_0=[0, π], x̂_0=[0, π, 0], y_step=[10])
end # hide
p = plot(res_yd)
savefig(p, "plot2_MTK.svg") # hide
display(p) # hide
```

![plot2_MTK](plot2_MTK.svg)

## Neutralization Reactor Model

This example repeats the neutralization reactor setup from the previous section but with
MTK. The observer and controller are non-adaptive to simplify the design. Instead of
estimating the acid feed concentration, an integrating state is added at the manipulated
input ``q_{Bin}``.

The MTK model of the neutralization reactor from the [last section](@ref man_dae) is
constructed as follows:

```@example 1
@parameters c_Ain=0.1 c_Bin=0.1 Kw=1e-14 Ka=1.75e-5 V=1000
@variables q_Ain(t)=10
@variables c_A(t)=0.025 c_B(t)=0.025 q_Bin(t)=10 pH(t)=7.0
@variables q_out(t) c_Aout(t) c_Bout(t)
eqs = [
    D(c_A) ~ 60/V*(q_Ain * c_Ain - q_out * c_Aout)
    D(c_B) ~ 60/V*(q_Bin * c_Bin - q_out * c_Bout)
    0      ~ 10^(-pH) + c_B - (Kw / 10^(-pH)) - (Ka * c_A / (Ka + 10^(-pH)))
    q_out  ~ q_Ain + q_Bin
    c_Aout ~ c_A
    c_Bout ~ c_B
]
@named mtk_model = System(eqs, t)
```

The substitution ``a_H = 10^{-\mathrm{pH}}`` is explicitly applied in the algebraic equation
to improve the numerical conditioning of the system. The converted model is:

```@example 1
inputs, outputs, disturbances = [q_Bin], [pH], [q_Ain]
f_mtk!, h_mtk!, p_sym, p, x_sym, x, i_alg = generate_f_h_mtk(
    mtk_model, inputs, outputs, disturbances
)
[x_sym x]
```

The content of `x_sym` shows that it chose for the combined state and algebraic vector this:
``[\begin{smallmatrix}c_B(t) && c_A(t) && \mathrm{pH}(t)\end{smallmatrix}]'``. The states
are thus in the reverse order of the [last section](@ref man_dae). The order of the
parameters is:

```@example 1
[p_sym p]
```

We can now construct a [`SimModel`](@ref) with this realization:

```@example 1
Ts, unit = 0.5, "h"
xs_mtk = x
transcription = TrapezoidalCollocation()
model = generate_model(
    f_mtk!, h_mtk!, Ts, i_alg, inputs, outputs, disturbances; 
    p, xs_mtk, unit, transcription
)
vu, vd = [raw"$q_{Bin}$ (L/min)"], [raw"$q_{Ain}$ (L/min)"]
vx, vy = [raw"$c_B$ (mol/L)", raw"$c_A$ (mol/L)"], [raw"$\mathrm{pH}$"]
model = setname!(model, u=vu, d=vd, y=vy, x=vx)
```

The returned value is a [`NonLinModelDAE`](@ref) object, as expected. The transient response
is identical to the first 10 hours of the open-loop simulations in the [last section](@ref man_dae):

```@example 1
using Plots
u, d, x_0 = [10.0], [10.0], [0.0495, 0.0505]
N = 21
using Logging # hide
res = with_logger(ConsoleLogger(stderr, Error)) do # hide
res = sim!(model, N, u, d; x_0)
end # hide
p = plot(res)
savefig(p, "plot3_MTK.svg") # hide
display(p) # hide
```

![plot3_MTK](plot3_MTK.svg)

## Neutralization Reactor Controller

We next configure the MHE, including an integrating state at the manipulated input:

```@example 1
nint_u=[1] # integrating state at the q_Bin input 
He = 8; hessian = true
σQ = [0.5e-3, 0.5e-3]; σR=[0.05]; σP_0 = [0.05, 0.05]; σQint_u=[0.15]; σPint_u_0=[0.075]
mhe = MovingHorizonEstimator(model; nint_u, He, hessian, σQ, σR, σP_0, σQint_u, σPint_u_0)
using JuMP; unset_time_limit_sec(mhe.optim)     # no wall time limit during optimization
mhe = setconstraint!(mhe, x̂min=[0, 0, -Inf])
```

Using the MHE, we configure an NMPC to keep the pH within the desired range:

```@example 1
nmpc = NonLinMPC(mhe, Hp=8, Hc=2, Nwt=[5.0], hessian=true)
using JuMP; unset_time_limit_sec(nmpc.optim)  # no wall time limit during optimization
nmpc = setconstraint!(nmpc, ymin=[6], ymax=[8], umin=[0], umax=[20])
```

Finally, we simulate the closed-loop response with a pH setpoint of 7:

```@example 1
ry = [7.0]
x_0, x̂_0 = [0.04, 0.07], [0.025, 0.025, 0]
using Logging # hide
res = with_logger(ConsoleLogger(stderr, Error)) do # hide
res = sim!(nmpc, N, ry, d; x_0, x̂_0)
end # hide
p = plot(res, plotd=false, plotxwithx̂=true, plotx̂min=false, plotŷ=true)
savefig(p, "plot4_MTK.svg") # hide
display(p) # hide
```

![plot4_MTK](plot4_MTK.svg)

As expected, the close-loop response is similar to the one of the [last section](@ref man_dae),
but not identical since the stochastic model of the unmeasured disturbances is different.

## Acknowledgement

Authored by `1-Bart-1` and `baggepinnen`, thanks for the contribution.
