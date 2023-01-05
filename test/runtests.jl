# spell-checker: disable

using ControlSystemsBase
using Documenter
using LinearAlgebra
using ModelPredictiveControl
using Test

testset_titlelen = 40

Ts = 4.0
sys = [ tf(1.90,[18.0,1])   tf(1.90,[18.0,1])   tf(1.90,[18.0,1]);
        tf(-0.74,[8.0,1])   tf(0.74,[8.0,1])    tf(-0.74,[8.0,1])   ] 
sys_ss = minreal(ss(sys))
Gss = c2d(sys_ss[:,1:2], Ts, :zoh)
Gss2 = c2d(sys_ss[:,1:2], 0.5Ts, :zoh)

@testset "$(rpad("LinModel construction", testset_titlelen))" begin
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

@testset "$(rpad("NonLinModel construction", testset_titlelen))" begin
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

@testset "$(rpad("SteadyKalmanFilter construction", testset_titlelen))" begin
    linmodel1 = LinModel(sys,Ts,i_u=[1,2])
    skalmanfilter1 = SteadyKalmanFilter(linmodel1)
    @test skalmanfilter1.nym == 2
    @test skalmanfilter1.nyu == 0
    @test skalmanfilter1.nxs == 2
    @test skalmanfilter1.nx̂ == 4

    linmodel2 = LinModel(sys,Ts,i_d=[3])
    skalmanfilter2 = SteadyKalmanFilter(linmodel2, i_ym=[2])
    @test skalmanfilter2.nym == 1
    @test skalmanfilter2.nyu == 1
    @test skalmanfilter2.nxs == 1
    @test skalmanfilter2.nx̂ == 5

    skalmanfilter3 = SteadyKalmanFilter(linmodel1, nint_ym=0)
    @test skalmanfilter3.nxs == 0
    @test skalmanfilter3.nx̂ == 2

    skalmanfilter4 = SteadyKalmanFilter(linmodel1, nint_ym=[2,2])
    @test skalmanfilter4.nxs == 4
    @test skalmanfilter4.nx̂ == 6

    @test_throws ErrorException SteadyKalmanFilter(linmodel1, nint_ym=[1,1,1])
    @test_throws ErrorException SteadyKalmanFilter(linmodel1, nint_ym=[-1,0])
    @test_throws ErrorException SteadyKalmanFilter(linmodel1, nint_ym=0, σQ=[1])
    @test_throws ErrorException SteadyKalmanFilter(linmodel1, nint_ym=0, σR=[1,1,1])
    @test_throws ErrorException SteadyKalmanFilter(LinModel(tf(1,[1,0]),1),nint_ym=[10])
end    
    
@testset "$(rpad("SteadyKalmanFilter estimator methods", testset_titlelen))" begin
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    skalmanfilter1 = SteadyKalmanFilter(linmodel1)
    @test updatestate!(skalmanfilter1, [10, 50], [50, 30]) ≈ zeros(4)
    @test updatestate!(skalmanfilter1, [10, 50], [50, 30], Float64[]) ≈ zeros(4)
    @test skalmanfilter1.x̂ ≈ zeros(4)
    @test evaloutput(skalmanfilter1) ≈ skalmanfilter1() ≈ [50, 30]
    @test evaloutput(skalmanfilter1, Float64[]) ≈ skalmanfilter1(Float64[]) ≈ [50, 30]
end   
    
@testset "$(rpad("KalmanFilter construction", testset_titlelen))" begin
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    kalmanfilter1 = KalmanFilter(linmodel1)
    @test kalmanfilter1.nym == 2
    @test kalmanfilter1.nyu == 0
    @test kalmanfilter1.nxs == 2
    @test kalmanfilter1.nx̂ == 4

    linmodel2 = LinModel(sys,Ts,i_d=[3])
    kalmanfilter2 = KalmanFilter(linmodel2, i_ym=[2])
    @test kalmanfilter2.nym == 1
    @test kalmanfilter2.nyu == 1
    @test kalmanfilter2.nxs == 1
    @test kalmanfilter2.nx̂ == 5

    kalmanfilter3 = KalmanFilter(linmodel1, nint_ym=0)
    @test kalmanfilter3.nxs == 0
    @test kalmanfilter3.nx̂ == 2

    kalmanfilter4 = KalmanFilter(linmodel1, nint_ym=[2,2])
    @test kalmanfilter4.nxs == 4
    @test kalmanfilter4.nx̂ == 6

    @test_throws ErrorException KalmanFilter(linmodel1, nint_ym=0, σP0=[1])
end

@testset "$(rpad("KalmanFilter estimator methods", testset_titlelen))" begin
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    kalmanfilter1 = KalmanFilter(linmodel1)
    @test updatestate!(kalmanfilter1, [10, 50], [50, 30]) ≈ zeros(4)
    @test updatestate!(kalmanfilter1, [10, 50], [50, 30], Float64[]) ≈ zeros(4)
    @test kalmanfilter1.x̂ ≈ zeros(4)
    @test evaloutput(kalmanfilter1) ≈ kalmanfilter1() ≈ [50, 30]
    @test evaloutput(kalmanfilter1, Float64[]) ≈ kalmanfilter1(Float64[]) ≈ [50, 30]
end   

@testset "$(rpad("InternalModel construction", testset_titlelen))" begin
    linmodel1 = LinModel(sys,Ts,i_u=[1,2])
    internalmodel1 = InternalModel(linmodel1)
    @test internalmodel1.nym == 2
    @test internalmodel1.nyu == 0
    @test internalmodel1.nxs == 2
    @test internalmodel1.nx̂ == 2

    linmodel2 = LinModel(sys,Ts,i_d=[3])
    internalmodel2 = InternalModel(linmodel2,i_ym=[2])
    @test internalmodel2.nym == 1
    @test internalmodel2.nyu == 1
    @test internalmodel2.nxs == 1
    @test internalmodel2.nx̂ == 4

    stoch_ym_tf = tf([1, -0.3],[1, -0.5],Ts)*tf([1,0],[1,-1],Ts).*I(2)
    internalmodel3 = InternalModel(linmodel2,stoch_ym=stoch_ym_tf)
    @test internalmodel3.nym == 2
    @test internalmodel3.nyu == 0
    @test internalmodel3.nxs == 4
    @test internalmodel3.nx̂ == 4

    stoch_ym_ss=minreal(ss(stoch_ym_tf))
    internalmodel4 = InternalModel(linmodel2,stoch_ym=stoch_ym_ss)
    @test internalmodel4.nym == 2
    @test internalmodel4.nyu == 0
    @test internalmodel4.nxs == 4
    @test internalmodel4.nx̂ == 4
    @test internalmodel4.As == stoch_ym_ss.A
    @test internalmodel4.Bs == stoch_ym_ss.B
    @test internalmodel4.Cs == stoch_ym_ss.C
    @test internalmodel4.Ds == stoch_ym_ss.D

    unstablemodel = LinModel(ss(diagm([0.5, -0.5, 1.5]), ones(3,1), I, 0, 1))
    @test_throws ErrorException InternalModel(unstablemodel)
    @test_throws ErrorException InternalModel(linmodel1, i_ym=[1,4])
    @test_throws ErrorException InternalModel(linmodel1, i_ym=[2,2])
    @test_throws ErrorException InternalModel(linmodel1, stoch_ym=ss(1,1,1,1,Ts))
    @test_throws ErrorException InternalModel(linmodel1, stoch_ym=ss(1,1,1,0,Ts).*I(2))
end    
    
@testset "$(rpad("InternalModel estimator methods", testset_titlelen))" begin
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]) , uop=[10,50], yop=[50,30])
    internalmodel1 = InternalModel(linmodel1)
    @test updatestate!(internalmodel1, [10, 50], [50, 30] .+ 1) ≈ zeros(2)
    @test updatestate!(internalmodel1, [10, 50], [50, 30] .+ 1, Float64[]) ≈ zeros(2)
    @test internalmodel1.x̂d ≈ internalmodel1.x̂ ≈ zeros(2)
    @test internalmodel1.x̂s ≈ ones(2)
    @test evaloutput(internalmodel1, [51,31]) ≈ internalmodel1([51,31]) ≈ [51,31] 
    @test evaloutput(internalmodel1, [51,31], Float64[]) ≈ internalmodel1([51,31], Float64[]) ≈ [51,31]
end
    
@testset "$(rpad("DocTest", testset_titlelen))" begin
    DocMeta.setdocmeta!(
        ModelPredictiveControl, 
        :DocTestSetup, 
        :(using ModelPredictiveControl, ControlSystemsBase); 
        recursive=true
    )
    doctest(ModelPredictiveControl)
end
