using ControlSystemsBase
using LinearAlgebra

Ts = 4.0
sys = [ tf(1.90,[18.0,1])   tf(1.90,[18.0,1])   tf(1.90,[18.0,1]);
        tf(-0.74,[8.0,1])   tf(0.74,[8.0,1])    tf(-0.74,[8.0,1])   ]        

Gss = c2d(minreal(ss(sys))[:,1:2], Ts, :zoh)

A = I(3);

B = Diagonal([3,4,5])

poles = eigvals(Gss.A)