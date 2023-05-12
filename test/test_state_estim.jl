Ts = 4.0
sys = [ tf(1.90,[18.0,1])   tf(1.90,[18.0,1])   tf(1.90,[18.0,1]);
        tf(-0.74,[8.0,1])   tf(0.74,[8.0,1])    tf(-0.74,[8.0,1])   ] 

@testset "SteadyKalmanFilter construction" begin
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

    skalmanfilter5 = SteadyKalmanFilter(linmodel2, σQ=[1,2,3,4], σQ_int=[5, 6],  σR=[7, 8])
    @test skalmanfilter5.Q̂ ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test skalmanfilter5.R̂ ≈ Hermitian(diagm(Float64[49, 64]))

    @test_throws ErrorException SteadyKalmanFilter(linmodel1, nint_ym=[1,1,1])
    @test_throws ErrorException SteadyKalmanFilter(linmodel1, nint_ym=[-1,0])
    @test_throws ErrorException SteadyKalmanFilter(linmodel1, nint_ym=0, σQ=[1])
    @test_throws ErrorException SteadyKalmanFilter(linmodel1, nint_ym=0, σR=[1,1,1])
    @test_throws ErrorException SteadyKalmanFilter(LinModel(tf(1,[1,0]),1),nint_ym=[10])
end    
    
@testset "SteadyKalmanFilter estimator methods" begin
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    skalmanfilter1 = SteadyKalmanFilter(linmodel1)
    @test updatestate!(skalmanfilter1, [10, 50], [50, 30]) ≈ zeros(4)
    @test updatestate!(skalmanfilter1, [10, 50], [50, 30], Float64[]) ≈ zeros(4)
    @test skalmanfilter1.x̂ ≈ zeros(4)
    @test evaloutput(skalmanfilter1) ≈ skalmanfilter1() ≈ [50, 30]
    @test evaloutput(skalmanfilter1, Float64[]) ≈ skalmanfilter1(Float64[]) ≈ [50, 30]
    @test initstate!(skalmanfilter1, [10, 50], [50, 30+1]) ≈ [zeros(3); [1]]
    setstate!(skalmanfilter1, [1,2,3,4])
    @test skalmanfilter1.x̂ ≈ [1,2,3,4]
end   
    
@testset "KalmanFilter construction" begin
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

    kalmanfilter5 = KalmanFilter(linmodel2, σQ=[1,2,3,4], σQ_int=[5, 6],  σR=[7, 8])
    @test kalmanfilter5.Q̂ ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test kalmanfilter5.R̂ ≈ Hermitian(diagm(Float64[49, 64]))

    kalmanfilter6 = KalmanFilter(linmodel2, σP0=[1,2,3,4], σP0_int=[5,6])
    @test kalmanfilter6.P̂0 ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test kalmanfilter6.P̂  ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test kalmanfilter6.P̂0 !== kalmanfilter6.P̂

    @test_throws ErrorException KalmanFilter(linmodel1, nint_ym=0, σP0=[1])
end

@testset "KalmanFilter estimator methods" begin
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    kalmanfilter1 = KalmanFilter(linmodel1)
    @test updatestate!(kalmanfilter1, [10, 50], [50, 30]) ≈ zeros(4)
    @test updatestate!(kalmanfilter1, [10, 50], [50, 30], Float64[]) ≈ zeros(4)
    @test kalmanfilter1.x̂ ≈ zeros(4)
    @test evaloutput(kalmanfilter1) ≈ kalmanfilter1() ≈ [50, 30]
    @test evaloutput(kalmanfilter1, Float64[]) ≈ kalmanfilter1(Float64[]) ≈ [50, 30]
    @test initstate!(kalmanfilter1, [10, 50], [50, 30+1]) ≈ [zeros(3); [1]]
    setstate!(kalmanfilter1, [1,2,3,4])
    @test kalmanfilter1.x̂ ≈ [1,2,3,4]
end   

@testset "InternalModel construction" begin
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

    stoch_ym_resample = c2d(d2c(ss(1,1,1,1,linmodel2.Ts), :tustin), 2linmodel2.Ts, :tustin)
    internalmodel5 = InternalModel(linmodel2, i_ym=[2], stoch_ym=stoch_ym_resample)
    @test internalmodel5.As ≈ internalmodel2.As
    @test internalmodel5.Bs ≈ internalmodel2.Bs
    @test internalmodel5.Cs ≈ internalmodel2.Cs
    @test internalmodel5.Ds ≈ internalmodel2.Ds

    unstablemodel = LinModel(ss(diagm([0.5, -0.5, 1.5]), ones(3,1), I, 0, 1))
    @test_throws ErrorException InternalModel(unstablemodel)
    @test_throws ErrorException InternalModel(linmodel1, i_ym=[1,4])
    @test_throws ErrorException InternalModel(linmodel1, i_ym=[2,2])
    @test_throws ErrorException InternalModel(linmodel1, stoch_ym=ss(1,1,1,1,Ts))
    @test_throws ErrorException InternalModel(linmodel1, stoch_ym=ss(1,1,1,0,Ts).*I(2))
end    
    
@testset "InternalModel estimator methods" begin
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]) , uop=[10,50], yop=[50,30])
    internalmodel1 = InternalModel(linmodel1)
    @test updatestate!(internalmodel1, [10, 50], [50, 30] .+ 1) ≈ zeros(2)
    @test updatestate!(internalmodel1, [10, 50], [50, 30] .+ 1, Float64[]) ≈ zeros(2)
    @test internalmodel1.x̂d ≈ internalmodel1.x̂ ≈ zeros(2)
    @test internalmodel1.x̂s ≈ ones(2)
    @test evaloutput(internalmodel1, [51,31]) ≈ internalmodel1([51,31]) ≈ [51,31] 
    @test evaloutput(internalmodel1, [51,31], Float64[]) ≈ internalmodel1([51,31], Float64[]) ≈ [51,31]
    @test initstate!(internalmodel1, [10, 50], [50, 30+1]) ≈ zeros(2)
    @test internalmodel1.x̂s ≈ zeros(2)
    setstate!(internalmodel1, [1,2])
    @test internalmodel1.x̂ ≈ [1,2]
end
 
@testset "UnscentedKalmanFilter construction" begin
    linmodel1 = LinModel(sys,Ts,i_d=[3])
    f(x,u,d) = linmodel1.A*x + linmodel1.Bu*u + linmodel1.Bd*d
    h(x,d)   = linmodel1.C*x + linmodel1.Du*d
    nonlinmodel = NonLinModel(f, h, Ts, 2, 4, 2, 1)

    ukf1 = UnscentedKalmanFilter(linmodel1)
    @test ukf1.nym == 2
    @test ukf1.nyu == 0
    @test ukf1.nxs == 2
    @test ukf1.nx̂ == 6

    ukf2 = UnscentedKalmanFilter(nonlinmodel)
    @test ukf2.nym == 2
    @test ukf2.nyu == 0
    @test ukf2.nxs == 2
    @test ukf2.nx̂ == 6

    ukf3 = UnscentedKalmanFilter(nonlinmodel, i_ym=[2])
    @test ukf3.nym == 1
    @test ukf3.nyu == 1
    @test ukf3.nxs == 1
    @test ukf3.nx̂ == 5

    ukf4 = UnscentedKalmanFilter(nonlinmodel, σQ=[1,2,3,4], σQ_int=[5, 6],  σR=[7, 8])
    @test ukf4.Q̂ ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test ukf4.R̂ ≈ Hermitian(diagm(Float64[49, 64]))
    
    ukf5 = UnscentedKalmanFilter(nonlinmodel, nint_ym=[2,2])
    @test ukf5.nxs == 4
    @test ukf5.nx̂ == 8

    ukf6 = UnscentedKalmanFilter(nonlinmodel, σP0=[1,2,3,4], σP0_int=[5,6])
    @test ukf6.P̂0 ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test ukf6.P̂  ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test ukf6.P̂0 !== ukf6.P̂

    ukf7 = UnscentedKalmanFilter(nonlinmodel, α=0.1, β=4, κ=0.2)
    @test ukf7.γ ≈ 0.1*√(ukf7.nx̂+0.2)
    @test ukf7.Ŝ[1, 1] ≈ 2 - 0.1^2 + 4 - ukf7.nx̂/(ukf7.γ^2)
end

@testset "UnscentedKalmanFilter estimator methods" begin
    linmodel1 = LinModel(sys,Ts,i_u=[1,2])
    f(x,u,_) = linmodel1.A*x + linmodel1.Bu*u
    h(x,_)   = linmodel1.C*x
    nonlinmodel = setop!(NonLinModel(f, h, Ts, 2, 2, 2), uop=[10,50], yop=[50,30])
    ukf1 = UnscentedKalmanFilter(nonlinmodel)
    @test updatestate!(ukf1, [10, 50], [50, 30]) ≈ zeros(4) atol=1e-9
    @test updatestate!(ukf1, [10, 50], [50, 30], Float64[]) ≈ zeros(4) atol=1e-9
    @test ukf1.x̂ ≈ zeros(4) atol=1e-9
    @test evaloutput(ukf1) ≈ ukf1() ≈ [50, 30]
    @test evaloutput(ukf1, Float64[]) ≈ ukf1(Float64[]) ≈ [50, 30]
    @test initstate!(ukf1, [10, 50], [50, 30+1]) ≈ [zeros(3); [1]]
    setstate!(ukf1, [1,2,3,4])
    @test ukf1.x̂ ≈ [1,2,3,4]
end