# [Manual: Nonlinear Design (DAE)](@id man_dae)

```@contents
Pages = ["nonlinmpc2.md"]
```

!!! todo "Under Construction"
    This tutorial is currently under construction. Only the modeling and the estimation
    parts are written for now.

## Nonlinear Model (DAE)

In this example, the goal is to control the pH of a solution in a continuously stirred
tank reactor (CSTR) for neutralization. The manipulated input is the inlet flow rate of
a strong base in L/min, while the inlet flow rate of a weak acid, also in L/min, is a
measured disturbance:

```math
\begin{aligned}
    \mathbf{u} &= q_{Bin}                                   \\
    \mathbf{y} &= \mathrm{pH}                               \\
    \mathbf{d} &= q_{Ain}                                   
\end{aligned}
```

An overflow weir draws the neutralized solution, effectively keeping a constant volume of
solution inside the tank. The following figure depicts the pH neutralization process:

```@raw html
<p><img src="../../assets/ph_neutralization.svg" alt="ph_neutralization" width=250 
    style="background-color:white; border:20px solid white; display: block; 
    margin-left: auto; margin-right: auto;"/></p>
```

### Instantaneous Charge Balance

The solution must remain electrically neutral, i.e. the sum of the charges of all ions must
equal zero. The ions in this case study are:

- Hydrogen ``[\mathrm{H}^+]``
- Sodium ``[\mathrm{Na}^+]``
- Hydroxide ``[\mathrm{OH}^-]``
- Acetate ``[\mathrm{Ac}^-]``

The instantaneous charge balance leads to:

```math
0 = [\mathrm{H}^+] + [\mathrm{Na}^+] - [\mathrm{OH}^-] - [\mathrm{Ac}^-]
```

The water dissociation constant and the weak acid equilibrium constant are respectively
defined by:

```math
\begin{aligned}
K_w &= [\mathrm{H}^+][\mathrm{OH}^-]                                        \\
K_a &= \frac{[\mathrm{H}^+][\mathrm{Ac}^-]}{[\mathrm{H}\mathrm{Ac}]}
\end{aligned}
```

We respectively denote the algebraic variable and the two states with:

```math
\begin{aligned}
    a_H &= [\mathrm{H}^+]                                                       \\
    c_A &= [\mathrm{H}\mathrm{Ac}] + [\mathrm{Ac}^-]                            \\
    c_B &= [\mathrm{Na}^+]
\end{aligned}
```

in which ``c_A`` represents the total concentration of the acid species in the reactor
(mol/L), composed of an undissociated acid ``\mathrm{H}\mathrm{Ac}`` and the acetate ion
``\mathrm{Ac}^-``. Substituting the constants, algebraic and state variables into the charge
balance leads the algebraic equation:

```math
0 = a_H + c_B - \frac{K_w}{a_H} - \frac{K_a c_A}{K_a + a_H}
```

The pH is computed with:

```math
\mathrm{pH} = -10 \log_{10}(a_H)
```

!!! details "Reduction to an ODE"
    The algebraic equation can be further manipulated to produce this cubic expression:
    ```math
    0 = a_H^3 + (c_B + K_a) a_H^2 + \big(K_a(c_A + c_B) + K_w \big) a_H - K_w K_a
    ```
    We could extract the positive real root of this expression inside the output function
    `h!` to transform the system to an ODE, effectively avoiding the complexity of DAEs.
    When possible, plant model should be constructed with the specialized [`NonLinModel`](@ref)
    for ODEs. This tutorial will still treat the system as a DAE to illustrate its API.

### Mass Balance

Applying a mass balance on the weak acid ``A`` and the strong base ``B`` invariants leads
to the differential equations:

```math
\begin{aligned}
    \dot{c}_A(t) &= \frac{60}{V}(q_{Ain} c_{Ain} - q_{out} c_{Aout})         \\
    \dot{c}_B(t) &= \frac{60}{V}(q_{Bin} c_{Bin} - q_{out} c_{Bout})
\end{aligned}
```

in which the concentrations ``c`` are in mol/L, the tank volume ``V`` in L and the
volumetric flow rates ``q`` in L/min. The accumulation terms ``\dot{c}`` are in mol/(L h)
because of the 60 factors. By assuming a perfectly mixed reactor and a constant volume
because of the weir, the following relations compute the outflow terms:

```math
\begin{aligned}
    c_{Aout} &= c_{A}                       \\
    c_{Bout} &= c_{B}                       \\
    q_{out}  &= q_{Ain} + q_{Bin}           
\end{aligned}
```

The [`NonLinModelDAE`](@ref) constructor expects that the state dynamics and the algebraic
equation is combined into a single `fq!(ẋ, res, x, a, u, d, p) -> nothing` function that
modifies both `ẋ` and `res` arguments in-place (an out-of-place option is also available),
with the state dynamics and the residual of the algebraic equation, respectively:

```@example 1
using ModelPredictiveControl

calc_ċ_A(c_Ain, q_Ain, c_Aout, q_out) = (60/V)*(q_Ain * c_Ain - q_out * c_Aout)
calc_ċ_B(c_Bin, q_Bin, c_Bout, q_out) = (60/V)*(q_Bin * c_Bin - q_out * c_Bout)
calc_res(a_H, c_A, c_B, Kw, Ka) = a_H + c_B - (Kw / a_H) - (Ka * c_A / (Ka + a_H))
function fq!(ẋ, res, x, a, u, d, p)
    c_Ain, c_Bin, Kw, Ka, V = p
    q_Ain, q_Bin = d[1], u[1] # [L/min], [L/min]
    c_A, c_B     = x[1], x[2] # [mol/L], [mol/L]
    a_H = a[1]                # [mol/L]
    q_out = q_Ain + q_Bin     # [L/min]
    c_Aout = c_A              # [mol/L]
    c_Bout = c_B              # [mol/L]
    ẋ[1]   = calc_ċ_A(c_Ain, q_Ain, c_Aout, q_out)
    ẋ[2]   = calc_ċ_B(c_Bin, q_Bin, c_Bout, q_out)
    res[1] = calc_res(a_H, c_A, c_B, Kw, Ka)
    return nothing
end
```

A similar in-place function is expected for the model output:

```@example 1
calc_pH(a_H) = -log10(a_H)
function h!(y, _ , a, _ , _ ) 
    a_H = a[1]
    pH = try
        calc_pH(a_H)
    catch myerror
        myerror isa DomainError ? NaN : rethrow()
    end
    y[1] = pH
    return nothing
end
```

By default, Julia throws a `DomainError` if `log10` is called with a negative number. The
`try` blocks is necessary to let the optimizer explores undefined domains by a returning a
`NaN` value in such cases. For similar reasons, providing an initial guess for the algebraic
variable `as_0` is crucial here to prioritize positive ``a_H`` concentration and ensure a
defined ``\mathrm{pH}`` value:

```@example 1
V = 1000.0      # reactor volume [L]
c_Ain = 0.1     # feed concentration of weak acid [mol/L]
c_Bin = 0.1     # feed concentration of strong base [mol/L]
Kw = 1.0e-14    # water dissociation constant [mol^2/L^2]
Ka = 1.75e-5    # acid dissociation constant [mol/L]

Ts = 0.5        # Sample time [h]
nu, nx, na, ny, nd = 1, 2, 1, 1, 1
p = [c_Ain, c_Bin, Kw, Ka, V]

vu, vd = [raw"$q_{Bin}$ (L/min)"], [raw"$q_{Ain}$ (L/min)"]
vx, vy = [raw"$c_A$ (mol/L)", raw"$c_B$ (mol/L)"], [raw"$\mathrm{pH}$"]

model = NonLinModelDAE(fq!, h!, Ts, nu, nx, na, ny, nd; p, as_0=[1e-5])
model = setname!(model, u=vu, x=vx, y=vy, d=vd)
```

By default, an [`OrthogonalCollocation`](@ref) with 3 collocation points transcribes the
state dynamics and the algebraic equations into an optimization problem. A simple open-loop
simulation of `model` with:

1. a bump on the base flow rate ``\mathbf{u} = q_{Bin}``
2. a bump on the acid flow rare ``\mathbf{d} = q_{Ain}``
3. a bump on the acid feed concentration ``c_{Ain}`` (as an unmeasured disturbance)

validates that our DAE is well-posed:

```@example 1
function simDAE(model, N; x_0)
    ny, ny, nd, nx = model.ny, model.ny, model.nd, model.nx
    Y_data, U_data, D_data, X_data = zeros(ny, N), zeros(nu, N), zeros(nd, N), zeros(nx, N)
    C_Ain_0 = model.p[1]
    setstate!(model, x_0)
    x = x_0
    for i=1:N
        u     = i ≤ (1N÷4) ? [10.0]  : [9.7]
        d     = i ≤ (2N÷4) ? [10.0]  : [9.8]
        c_Ain = i ≤ (3N÷4) ? C_Ain_0 : (C_Ain_0 + 0.05)
        model.p[1] = c_Ain
        y = model(d)
        Y_data[:, i] = y
        U_data[:, i] = u
        D_data[:, i] = d
        X_data[:, i] = x
        x = updatestate!(model, u, d)
    end
    model.p[1] = C_Ain_0
    return SimResult(model, U_data, Y_data, D_data; X_data)
end
x_0 = [0.0505, 0.0495]
N = 101
res = simDAE(model, N; x_0)
```

We plot the results by modifying the x-axis label to substitute the default time units
to hours:

```@example 1
using Plots
plot(res, plotu=true, plotd=true, xlabel="Time (h)")
savefig("plot1_DAEpH.svg"); nothing # hide
```

![plot1_DAEpH](plot1_DAEpH.svg)

## Adaptive Moving Horizon Estimation

The default settings of the [`MovingHorizonEstimator`](@ref) assume that the measured
output is disturbed by a random-walk stochastic process (the pH). This is generally enough
to estimate the unmeasured disturbances in steady-state (the acid feed concentration).
To improve the interpretability of the results and the estimation performances, we can
instead disable the default stochastic model and construct an adaptive estimator. We first
need to augment the dynamics with our estimated parameter, the acid feed concentration
``c_{Ain}``:

```@example 1
calc_ċ_Ain( _ ) = 0
function f̂q!(ẋ, res, x, a, u, d, p̂)
    c_Bin, Kw, Ka, V = p̂
    q_Ain, q_Bin = d[1], u[1]
    c_A, c_B     = x[1], x[2]
    c_Ain        = x[3]
    a_H = a[1]               
    q_out = q_Ain + q_Bin     
    c_Aout = c_A             
    c_Bout = c_B             
    ẋ[1]   = calc_ċ_A(c_Ain, q_Ain, c_Aout, q_out)
    ẋ[2]   = calc_ċ_B(c_Bin, q_Bin, c_Bout, q_out)
    ẋ[3]   = calc_ċ_Ain(c_Ain)
    res[1] = calc_res(a_H, c_A, c_B, Kw, Ka)
    return nothing
end
ĥ!(y, x, a, d, p̂) = h!(y, x, a, d, p)
p̂ = [c_Bin, Kw, Ka, V]
nx̂ = 3
vx̂ = [vx; raw"$c_{Ain}$ (mol/L)"]
model_aug = NonLinModelDAE(f̂q!, ĥ!, Ts, nu, nx̂, na, ny, nd; p, as_0=[1e-5])
model_aug = setname!(model_aug, u=vu, x=vx̂, y=vy, d=vd)
```

Since `calc_ċ_Ain` returns `0`, the ``c_{Ain}`` parameter is assumed to be time-invariant.
More precisely, this feed property is assumed to be disturbed by a random-walk, instead of
the measured output directly.

```@example 1
mhe = MovingHorizonEstimator(model_aug, He=5, nint_ym=0)
```
