@testmodule SetupMPCtests begin
    using ControlSystemsBase
    Ts = 400.0
    sys = [ tf(1.90,[1800.0,1])   tf(1.90,[1800.0,1])   tf(1.90,[1800.0,1]);
            tf(-0.74,[800.0,1])   tf(0.74,[800.0,1])    tf(-0.74,[800.0,1])   ] 
    sys_ss = ss(sys)
    sys_ss_u = sminreal(sys_ss[:,1:2])
    Gss  = minreal(c2d(sys_ss_u, Ts, :zoh))
    Gss2 = minreal(c2d(sys_ss_u, 0.5Ts, :zoh))
    export Ts, sys, sys_ss, Gss, Gss2
end