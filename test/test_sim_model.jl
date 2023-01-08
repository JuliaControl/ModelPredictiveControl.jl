Ts = 4.0
sys = [ tf(1.90,[18.0,1])   tf(1.90,[18.0,1])   tf(1.90,[18.0,1]);
        tf(-0.74,[8.0,1])   tf(0.74,[8.0,1])    tf(-0.74,[8.0,1])   ] 
sys_ss = minreal(ss(sys))
Gss = c2d(sys_ss[:,1:2], Ts, :zoh)
Gss2 = c2d(sys_ss[:,1:2], 0.5Ts, :zoh)

@testset "LinModel construction" begin
linmodel1 = LinModel(sys, Ts, i_u=1:2)
@test linmodel1.nx == 2
@test linmodel1.nu == 2
@test linmodel1.nd == 0
@test linmodel1.ny == 2
@test linmodel1.A   ≈ Gss.A
@test linmodel1.Bu  ≈ Gss.B
@test linmodel1.Bd  ≈ zeros(2,0)
@test linmodel1.C   ≈ Gss.C
@test linmodel1.Dd  ≈ zeros(2,0)

linmodel2 = LinModel(Gss)
setop!(linmodel2, uop=[10,50], yop=[50,30])
@test linmodel2.A   ≈ Gss.A
@test linmodel2.Bu  ≈ Gss.B
@test linmodel2.Bd  ≈ zeros(2,0)
@test linmodel2.C   ≈ Gss.C
@test linmodel2.Dd  ≈ zeros(2,0)
@test linmodel2.uop ≈ [10,50]
@test linmodel2.yop ≈ [50,30]
@test linmodel2.dop ≈ zeros(0,1)

linmodel3 = LinModel(Gss, 0.5Ts)
@test linmodel3.Ts == 2.0
@test linmodel3.A ≈ Gss2.A
@test linmodel3.C ≈ Gss2.C

linmodel4 = LinModel(Gss)
setstate!(linmodel4, [1;-1])
@test linmodel4.x ≈ [1;-1]

linmodel5 = LinModel(sys,Ts,i_d=[3])
@test linmodel5.nx == 4
@test linmodel5.nu == 2
@test linmodel5.nd == 1
@test linmodel5.ny == 2
sysu_ss = sminreal(c2d(minreal(ss(sys))[:,1:2], Ts, :zoh))
sysd_ss = sminreal(c2d(minreal(ss(sys))[:,3],   Ts, :tustin))
sys_ss = [sysu_ss sysd_ss]
@test linmodel5.A   ≈ sys_ss.A
@test linmodel5.Bu  ≈ sys_ss.B[:,1:2]
@test linmodel5.Bd  ≈ sys_ss.B[:,3]
@test linmodel5.C   ≈ sys_ss.C
@test linmodel5.Dd  ≈ sys_ss.D[:,3]

linmodel6 = LinModel([delay(4) delay(4)]*sys,Ts,i_d=[3])
@test linmodel6.nx == 6
@test sum(eigvals(linmodel6.A) .≈ 0) == 2

@test_throws ErrorException LinModel(sys)
@test_throws ErrorException LinModel(sys,-Ts)
@test_throws ErrorException LinModel(sys,Ts,i_u=[1,1])
@test_throws ErrorException LinModel(sys_ss,Ts+1)
@test_throws ErrorException setop!(linmodel5, uop=[0,0,0,0,0])
@test_throws ErrorException setop!(linmodel5, yop=[0,0,0,0,0])
@test_throws ErrorException setop!(linmodel5, dop=[0,0,0,0,0])
sys_ss.D .= 1
@test_throws ErrorException LinModel(sys_ss,Ts)
end

@testset "$(rpad("LinModel sim methods", testset_titlelen))" begin
linmodel1 = setop!(LinModel(Gss), uop=[10,50], yop=[50,30])

@test updatestate!(linmodel1, [10, 50]) ≈ zeros(2) 
@test updatestate!(linmodel1, [10, 50], Float64[]) ≈ zeros(2)
@test linmodel1.x ≈ zeros(2)
@test evaloutput(linmodel1) ≈ linmodel1() ≈ [50,30] 
@test evaloutput(linmodel1, Float64[]) ≈ linmodel1(Float64[]) ≈ [50,30] 

@test_throws DimensionMismatch updatestate!(linmodel1, zeros(2), zeros(1))
@test_throws DimensionMismatch evaloutput(linmodel1, zeros(1))
end


@testset "NonLinModel construction" begin
linmodel1 = LinModel(sys,Ts,i_u=[1,2])
f1(x,u,_) = linmodel1.A*x + linmodel1.Bu*u
h1(x,_)   = linmodel1.C*x
nonlinmodel1 = NonLinModel(f1,h1,Ts,2,2,2)
@test nonlinmodel1.nx == 2
@test nonlinmodel1.nu == 2
@test nonlinmodel1.nd == 0
@test nonlinmodel1.ny == 2
@test nonlinmodel1.f([0,0],[0,0],[1]) ≈ zeros(2,)
@test nonlinmodel1.h([0,0],[1]) ≈ zeros(2,)

linmodel2 = LinModel(sys,Ts,i_d=[3])
f2(x,u,d) = linmodel2.A*x + linmodel2.Bu*u + linmodel2.Bd*d
h2(x,_)   = linmodel2.C*x 
nonlinmodel2 = NonLinModel(f2,h2,Ts,2,4,2,1)

@test nonlinmodel2.nx == 4
@test nonlinmodel2.nu == 2
@test nonlinmodel2.nd == 1
@test nonlinmodel2.ny == 2
@test nonlinmodel2.f([0,0,0,0],[0,0],[0]) ≈ zeros(4,)
@test nonlinmodel2.h([0,0,0,0],[0]) ≈ zeros(2,)

@test_throws ErrorException NonLinModel(
    (x,u)->linmodel1.A*x + linmodel1.Bu*u,
    (x,_)->linmodel1.C*x, Ts, 2, 4, 2, 1)
@test_throws ErrorException NonLinModel(
    (x,u,_)->linmodel1.A*x + linmodel1.Bu*u,
    (x)->linmodel1.C*x, Ts, 2, 4, 2, 1)
end

@testset "$(rpad("NonLinModel sim methods", testset_titlelen))" begin
linmodel1 = LinModel(sys,Ts,i_u=[1,2])
f1(x,u,_) = linmodel1.A*x + linmodel1.Bu*u
h1(x,_)   = linmodel1.C*x
nonlinmodel = NonLinModel(f1,h1,Ts,2,2,2)

@test updatestate!(nonlinmodel, zeros(2,)) ≈ zeros(2) 
@test updatestate!(nonlinmodel, zeros(2,), Float64[]) ≈ zeros(2)
@test nonlinmodel.x ≈ zeros(2)
@test evaloutput(nonlinmodel) ≈ nonlinmodel() ≈ zeros(2)
@test evaloutput(nonlinmodel, Float64[]) ≈ nonlinmodel(Float64[]) ≈ zeros(2)

@test_throws DimensionMismatch updatestate!(nonlinmodel, zeros(2), zeros(1))
@test_throws DimensionMismatch evaloutput(nonlinmodel, zeros(1))
end