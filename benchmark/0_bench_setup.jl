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