## -------------------------- linear CSTR model ----------------------------------

Ts = 400.0
sys = [ tf(1.90,[1800.0,1])   tf(1.90,[1800.0,1])   tf(1.90,[1800.0,1]);
        tf(-0.74,[800.0,1])   tf(0.74,[800.0,1])    tf(-0.74,[800.0,1])   ] 
function f_lin!(ẋ, x, u, d, p)
    mul!(ẋ, p.A, x)
    mul!(ẋ, p.Bu, u, 1, 1)
    mul!(ẋ, p.Bd, d, 1, 1)
    return nothing
end
function h_lin!(y, x, d, p)
    mul!(y, p.C, x)
    mul!(y, p.Dd, d, 1, 1)
    return nothing
end
linmodel = setop!(LinModel(sys, Ts, i_d=[3]), uop=[10, 50], yop=[50, 30], dop=[5])
nonlinmodel = NonLinModel(f_lin!, h_lin!, Ts, 2, 4, 2, 1, p=linmodel, solver=nothing)
nonlinmodel = setop!(nonlinmodel, uop=[10, 50], yop=[50, 30], dop=[5])
u, d, y = [10, 50], [5], [50, 30]
nonlinmodel_c = NonLinModel(
    (ẋ,x,u,d,_) -> ẋ .= -0.001x .+ u .+ d , 
    (y,x,d,_) -> y .= x .+ 0.001d, 500, 1, 1, 1, 1
)
u_c, d_c, y_c = [1], [0], [0]

G = [ tf(1.90, [18, 1]) tf(1.90, [18, 1]);
      tf(-0.74,[8, 1])  tf(0.74, [8, 1]) ]
uop, yop, dop = [20, 20], [50, 30], [20]
CSTR_model   = setop!(LinModel(G, 2.0); uop, yop)
CSTR_model_d = setop!(LinModel([G G[1:2, 2]], 2.0; i_d=[3]); uop, yop, dop)

## ------------------------ pendulum model -------------------------------------
function f!(ẋ, x, u, _ , p)
    g, L, K, m = p
    θ, ω = x[1], x[2]
    τ = u[1]
    ẋ[1] = ω
    ẋ[2] = -g/L*sin(θ) - K/m*ω + τ/m/L^2
end
h!(y, x, _ , _ ) = (y[1] = 180/π*x[1])
p = [9.8, 0.4, 1.2, 0.3]
nu = 1; nx = 2; ny = 1; Ts = 0.1
pendulum_model = NonLinModel(f!, h!, Ts, nu, nx, ny; p)
pendulum_p = p

h2!(y, x, _ , _ ) = (y[1] = 180/π*x[1]; y[2]=x[2])
nu, nx, ny = 1, 2, 2
pendulum_model2 = NonLinModel(f!, h2!, Ts, nu, nx, ny; p)
pendulum_p2 = p

# --------------------------- pH neutralization reactor model ----------------------------
function fq_ph!(dx̂, res, x̂, a, u, d, p̂)
    c_Bin, Kw, Ka, V = p̂
    q_Ain, q_Bin = d[1], u[1]
    c_A, c_B     = x̂[1], x̂[2]
    c_Ain        = x̂[3]
    pH           = a[1]       
    q_out  = q_Ain + q_Bin     
    c_Aout = c_A             
    c_Bout = c_B     
    a_H    = 10.0^(-pH)
    dx̂[1]  = (60/V)*(q_Ain * c_Ain - q_out * c_Aout)
    dx̂[2]  = (60/V)*(q_Bin * c_Bin - q_out * c_Bout)
    dx̂[3]  = 0
    res[1] = a_H + c_B - (Kw / a_H) - (Ka * c_A / (Ka + a_H))
    return nothing
end
h_ph!(y, _ , a , _ , _ ) = (y .= a; nothing)
p = [0.1, 1.0e-14, 1.75e-5, 1000.0]
Ts = 0.5           
nu, nx, na, ny, nd = 1, 2+1, 1, 1, 1
xs_0, as_0 = [0.025, 0.025, 0.1], [7]
transcription = TrapezoidalCollocation()
ph_model = NonLinModelDAE(
    fq_ph!, h_ph!, Ts, nu, nx, na, ny, nd; p, xs_0, as_0, transcription, unit="h"
)
ph_p = p
ph_x_0 = [0.0, 0.0, 0.1]