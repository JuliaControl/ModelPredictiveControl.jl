# [Manual: Nonlinear Design (DAE)](@id man_ade)

```@contents
Pages = ["nonlinmpc2.md"]
```

The solution must remain electrically neutral, i.e. the sum of the charges of all ions must
equal zero. Here, the ions are:

- Hydrogen ``[\mathrm{H}^+]``
- Sodium ``[\mathrm{Na}^+]``
- Hydroxide ``[\mathrm{OH}^-]``
- Acetate ``[\mathrm{Ac}^-]``

The charge balance leads to:

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

We have one algebraic variable and two states, respectively denoted with:

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

The code is:

```@example 1
using ModelPredictiveControl

# Process parameters
V = 1000.0      # Reactor volume [L]
c_Ain = 0.1     # Feed concentration of weak acid [mol/L]
c_Bin = 0.1     # Feed concentration of strong base [mol/L]
Kw = 1.0e-14    # Water dissociation constant [mol^2/L^2]
Ka = 1.75e-5    # Acid dissociation constant [mol/L]

function fq!(ẋ, res, x, a, u, d, p)
    c_Ain, c_Bin, Kw, Ka, V = p
    q_Ain, q_Bin = d[1], u[1] # [L/min], [L/min]
    c_A, c_B     = x[1], x[2] # [mol/L], [mol/L]
    a_H = a[1]                # [mol/L]
    q_out = q_Ain + q_Bin     # overflow weir
    c_Aout = c_A              # perfect mixing assumption
    c_Bout = c_B              # perfect mixing assumption
    ẋ[1]   = (q_Ain * c_Ain - q_out * c_Aout) / V
    ẋ[2]   = (q_Bin * c_Bin - q_out * c_Bout) / V
    res[1] = a_H + c_B - (Kw / a_H) - (Ka * c_A / (Ka + a_H))
    return nothing
end

function h!(y, _, a, _ , _ ) 
    a_H = a[1]
    y[1] = try
        -log10(a_H) # y = pH
    catch myerror
        if myerror isa DomainError
            NaN
        else
            rethrow()
        end
    end
    return nothing
end

Ts = 15.0 # Sample time [min]
nu = 1    # Manipulated input: base flow rate q_B
nx = 2    # Differential states: total acid W_a, total base W_b
na = 1    # Algebraic variable: Hydrogen ion concentration [H+]
ny = 1    # Measured output: pH
nd = 1    # Measured disturbance: acid flow rate q_A
p = [c_Ain, c_Bin, Kw, Ka, V]

model = NonLinModelDAE(fq!, h!, Ts, nu, nx, na, ny, nd; p)
model = setname!(model, u=["\$q_B\$"], x=["\$c_a\$", "\$c_b\$"], y=["\$\\mathrm{pH}\$"], d=["\$q_A\$"])

model.a0 .= 1e-5 # initial guess for algebraic variable (positive value because of log10)
model.Z  .= 1e-5 # initial guess for collocation points (positive value because of log10)

u = [10.0]
d = [10.0]
x_0 = [0.05556, 0.04444]
#res = sim!(model, 50, u, d; x_0)
N = 100
Y_data, U_data, D_data, X_data = zeros(ny, N), zeros(nu, N), zeros(nd, N), zeros(nx, N)
x = x_0
let x=x, u=u, d=d
    setstate!(model, x)
    for i=1:N
        #@show x
        d = [10.0]
        y = model(d)
        u = i < N/2 ? [10.0] : [9.5]
        Y_data[:, i] = y
        U_data[:, i] = u
        D_data[:, i] = d
        X_data[:, i] = x
        x = updatestate!(model, u, d)
    end
end
res = SimResult(model, U_data, Y_data, D_data; X_data)

using Plots
theme(:default)
#theme(:dark)
default(fontfamily="Computer Modern"); scalefontsizes(1.1)
plot(res, plotx=true, plotd=false)
```
