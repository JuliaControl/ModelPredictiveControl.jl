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

G = [ tf(1.90, [18, 1]) tf(1.90, [18, 1]);
      tf(-0.74,[8, 1])  tf(0.74, [8, 1]) ]
uop, yop, dop = [20, 20], [50, 30], [20]
CSTR_model   = setop!(LinModel(G, 2.0); uop, yop)
CSTR_model_d = setop!(LinModel([G G[1:2, 2]], 2.0; i_d=[3]); uop, yop, dop)

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