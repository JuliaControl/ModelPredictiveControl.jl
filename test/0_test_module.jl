@testmodule SetupMPCtests begin
    using ControlSystemsBase
    Ts = 400.0
    sys = [ tf(1.90,[1800.0,1])   tf(1.90,[1800.0,1])   tf(1.90,[1800.0,1]);
            tf(-0.74,[800.0,1])   tf(0.74,[800.0,1])    tf(-0.74,[800.0,1])   ] 
    sys_ss = minreal(ss(sys))
    Gss = c2d(sys_ss[:,1:2], Ts, :zoh)
    Gss2 = c2d(sys_ss[:,1:2], 0.5Ts, :zoh)
    export Ts, sys, sys_ss, Gss, Gss2
end