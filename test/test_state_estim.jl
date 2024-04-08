Ts = 400.0
sys = [ tf(1.90,[1800.0,1])   tf(1.90,[1800.0,1])   tf(1.90,[1800.0,1]);
        tf(-0.74,[800.0,1])   tf(0.74,[800.0,1])    tf(-0.74,[800.0,1])   ] 

@testset "SteadyKalmanFilter construction" begin
    linmodel1 = LinModel(sys,Ts,i_u=[1,2])
    skalmanfilter1 = SteadyKalmanFilter(linmodel1)
    @test skalmanfilter1.nym == 2
    @test skalmanfilter1.nyu == 0
    @test skalmanfilter1.nxs == 2
    @test skalmanfilter1.nx̂ == 4
    @test skalmanfilter1.nint_ym == [1, 1]

    linmodel2 = LinModel(sys,Ts,i_d=[3])
    skalmanfilter2 = SteadyKalmanFilter(linmodel2, i_ym=[2])
    @test skalmanfilter2.nym == 1
    @test skalmanfilter2.nyu == 1
    @test skalmanfilter2.nxs == 1
    @test skalmanfilter2.nx̂ == 5
    @test skalmanfilter2.nint_ym == [1]

    skalmanfilter3 = SteadyKalmanFilter(linmodel1, nint_ym=0)
    @test skalmanfilter3.nxs == 0
    @test skalmanfilter3.nx̂ == 2
    @test skalmanfilter3.nint_ym == [0, 0]

    skalmanfilter4 = SteadyKalmanFilter(linmodel1, nint_ym=[2,2])
    @test skalmanfilter4.nxs == 4
    @test skalmanfilter4.nx̂ == 6

    skalmanfilter5 = SteadyKalmanFilter(linmodel2, σQ=[1,2,3,4], σQint_ym=[5, 6],  σR=[7, 8])
    @test skalmanfilter5.Q̂ ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test skalmanfilter5.R̂ ≈ Hermitian(diagm(Float64[49, 64]))

    linmodel3 = LinModel(append(tf(1,[1, 0]),tf(1,[10, 1]),tf(1,[-1, 1])), 0.1)
    skalmanfilter6 = SteadyKalmanFilter(linmodel3)
    @test skalmanfilter6.nxs == 2
    @test skalmanfilter6.nx̂ == 5
    @test skalmanfilter6.nint_ym == [0, 1, 1]

    skalmanfilter7 = SteadyKalmanFilter(linmodel1, nint_u=[1,1])
    @test skalmanfilter7.nxs == 2
    @test skalmanfilter7.nx̂  == 4
    @test skalmanfilter7.nint_u  == [1, 1]
    @test skalmanfilter7.nint_ym == [0, 0]

    linmodel2 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    skalmanfilter8 = SteadyKalmanFilter(linmodel2)
    @test isa(skalmanfilter8, SteadyKalmanFilter{Float32})

    skalmanfilter9 = SteadyKalmanFilter(linmodel1, 1:2, 0, [1, 1], I(4), I(2))
    @test skalmanfilter9.Q̂ ≈ I(4)
    @test skalmanfilter9.R̂ ≈ I(2)

    @test_throws ErrorException SteadyKalmanFilter(linmodel1, nint_ym=[1,1,1])
    @test_throws ErrorException SteadyKalmanFilter(linmodel1, nint_ym=[-1,0])
    @test_throws ErrorException SteadyKalmanFilter(linmodel1, nint_ym=0, σQ=[1])
    @test_throws ErrorException SteadyKalmanFilter(linmodel1, nint_ym=0, σR=[1,1,1])
    @test_throws ErrorException SteadyKalmanFilter(linmodel3, nint_ym=[1, 0, 0])
    model_unobs = LinModel([1 0;0 1.5], [1; 0], [1 0], zeros(2,0), zeros(1,0), 1.0)
    @test_throws ErrorException SteadyKalmanFilter(model_unobs, nint_ym=[1])
    @test_throws ErrorException SteadyKalmanFilter(LinModel(tf(1,[10, 1]), 1.0), 1:1, 0, 0, [-1], [1])
    @test_throws ErrorException SteadyKalmanFilter(LinModel(tf(1, [1,0]), 1), nint_ym=[1])
    @test_throws ErrorException SteadyKalmanFilter(linmodel1, nint_u=[1,1], nint_ym=[1,1])
end

@testset "SteadyKalmanFilter estimator methods" begin
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    skalmanfilter1 = SteadyKalmanFilter(linmodel1, nint_ym=[1, 1])
    @test updatestate!(skalmanfilter1, [10, 50], [50, 30]) ≈ zeros(4)
    @test updatestate!(skalmanfilter1, [10, 50], [50, 30], Float64[]) ≈ zeros(4)
    @test skalmanfilter1.x̂ ≈ zeros(4)
    @test evaloutput(skalmanfilter1) ≈ skalmanfilter1() ≈ [50, 30]
    @test evaloutput(skalmanfilter1, Float64[]) ≈ skalmanfilter1(Float64[]) ≈ [50, 30]
    @test initstate!(skalmanfilter1, [10, 50], [50, 30+1]) ≈ [zeros(3); [1]]
    linmodel2 = LinModel(append(tf(1, [1, 0]), tf(2, [10, 1])), 1.0)
    skalmanfilter2 = SteadyKalmanFilter(linmodel2, nint_u=[1, 1])
    x = initstate!(skalmanfilter2, [10, 3], [0.5, 6+0.1])
    @test evaloutput(skalmanfilter2) ≈ [0.5, 6+0.1]
    @test updatestate!(skalmanfilter2, [0, 3], [0.5, 6+0.1]) ≈ x
    setstate!(skalmanfilter1, [1,2,3,4])
    @test skalmanfilter1.x̂ ≈ [1,2,3,4]
    for i in 1:100
        updatestate!(skalmanfilter1, [11, 52], [50, 30])
    end
    @test skalmanfilter1() ≈ [50, 30] atol=1e-3
    for i in 1:100
        updatestate!(skalmanfilter1, [10, 50], [51, 32])
    end
    @test skalmanfilter1() ≈ [51, 32] atol=1e-3
    skalmanfilter2 = SteadyKalmanFilter(linmodel1, nint_u=[1, 1])
    for i in 1:100
        updatestate!(skalmanfilter2, [11, 52], [50, 30])
    end
    @test skalmanfilter2() ≈ [50, 30] atol=1e-3
    for i in 1:100
        updatestate!(skalmanfilter2, [10, 50], [51, 32])
    end
    @test skalmanfilter2() ≈ [51, 32] atol=1e-3
    linmodel3 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    skalmanfilter3 = SteadyKalmanFilter(linmodel3)
    x̂ = updatestate!(skalmanfilter3, [0], [0])
    @test x̂ ≈ [0, 0]
    @test isa(x̂, Vector{Float32})
    @test_throws ArgumentError updatestate!(skalmanfilter1, [10, 50])
end   
    
@testset "KalmanFilter construction" begin
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    kalmanfilter1 = KalmanFilter(linmodel1)
    @test kalmanfilter1.nym == 2
    @test kalmanfilter1.nyu == 0
    @test kalmanfilter1.nxs == 2
    @test kalmanfilter1.nx̂ == 4
    @test kalmanfilter1.nint_ym == [1, 1]

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

    kalmanfilter5 = KalmanFilter(linmodel2, σQ=[1,2,3,4], σQint_ym=[5, 6],  σR=[7, 8])
    @test kalmanfilter5.Q̂ ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test kalmanfilter5.R̂ ≈ Hermitian(diagm(Float64[49, 64]))

    kalmanfilter6 = KalmanFilter(linmodel2, σP0=[1,2,3,4], σP0int_ym=[5,6])
    @test kalmanfilter6.P̂0 ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test kalmanfilter6.P̂  ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test kalmanfilter6.P̂0 !== kalmanfilter6.P̂

    kalmanfilter7 = KalmanFilter(linmodel1, nint_u=[1,1])
    @test kalmanfilter7.nxs == 2
    @test kalmanfilter7.nx̂  == 4
    @test kalmanfilter7.nint_u  == [1, 1]
    @test kalmanfilter7.nint_ym == [0, 0]

    kalmanfilter8 = KalmanFilter(linmodel1, 1:2, 0, [1, 1], I(4), I(4), I(2))
    @test kalmanfilter8.P̂0 ≈ I(4)
    @test kalmanfilter8.Q̂ ≈ I(4)
    @test kalmanfilter8.R̂ ≈ I(2)

    linmodel2 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    kalmanfilter8 = KalmanFilter(linmodel2)
    @test isa(kalmanfilter8, KalmanFilter{Float32})

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
    for i in 1:1000
        updatestate!(kalmanfilter1, [11, 52], [50, 30])
    end
    @test kalmanfilter1() ≈ [50, 30] atol=1e-3
    for i in 1:100
        updatestate!(kalmanfilter1, [10, 50], [51, 32])
    end
    @test kalmanfilter1() ≈ [51, 32] atol=1e-3
    kalmanfilter2 = KalmanFilter(linmodel1, nint_u=[1, 1])
    for i in 1:100
        updatestate!(kalmanfilter2, [11, 52], [50, 30])
    end
    @test kalmanfilter2() ≈ [50, 30] atol=1e-3
    for i in 1:100
        updatestate!(kalmanfilter2, [10, 50], [51, 32])
    end
    @test kalmanfilter2() ≈ [51, 32] atol=1e-3
    linmodel3 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    kalmanfilter3 = KalmanFilter(linmodel3)
    x̂ = updatestate!(kalmanfilter3, [0], [0])
    @test x̂ ≈ [0, 0]
    @test isa(x̂, Vector{Float32})
    @test_throws ArgumentError updatestate!(kalmanfilter1, [10, 50])
end   

@testset "Luenberger construction" begin
    linmodel1 = LinModel(sys,Ts,i_u=[1,2])
    lo1 = Luenberger(linmodel1)
    @test lo1.nym == 2
    @test lo1.nyu == 0
    @test lo1.nxs == 2
    @test lo1.nx̂ == 4

    linmodel2 = LinModel(sys,Ts,i_d=[3])
    lo2 = Luenberger(linmodel2, i_ym=[2])
    @test lo2.nym == 1
    @test lo2.nyu == 1
    @test lo2.nxs == 1
    @test lo2.nx̂ == 5

    lo3 = Luenberger(linmodel1, nint_ym=0)
    @test lo3.nxs == 0
    @test lo3.nx̂ == 2

    lo4 = Luenberger(linmodel1, nint_ym=[2,2])
    @test lo4.nxs == 4
    @test lo4.nx̂ == 6

    lo5 = Luenberger(linmodel1, nint_u=[1,1])
    @test lo5.nxs == 2
    @test lo5.nx̂  == 4
    @test lo5.nint_u  == [1, 1]
    @test lo5.nint_ym == [0, 0]

    linmodel2 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    lo6 = Luenberger(linmodel2)
    @test isa(lo6, Luenberger{Float32})

    @test_throws ErrorException Luenberger(linmodel1, nint_ym=[1,1,1])
    @test_throws ErrorException Luenberger(linmodel1, nint_ym=[-1,0])
    @test_throws ErrorException Luenberger(linmodel1, p̂=[0.5])
    @test_throws ErrorException Luenberger(linmodel1, p̂=fill(1.5, lo1.nx̂))
    @test_throws ErrorException Luenberger(LinModel(tf(1,[1, 0]),0.1), p̂=[0.5,0.6])
end
    
@testset "Luenberger estimator methods" begin
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    ukf1 = Luenberger(linmodel1, nint_ym=[1, 1])
    @test updatestate!(ukf1, [10, 50], [50, 30]) ≈ zeros(4)
    @test updatestate!(ukf1, [10, 50], [50, 30], Float64[]) ≈ zeros(4)
    @test ukf1.x̂ ≈ zeros(4)
    @test evaloutput(ukf1) ≈ ukf1() ≈ [50, 30]
    @test evaloutput(ukf1, Float64[]) ≈ ukf1(Float64[]) ≈ [50, 30]
    @test initstate!(ukf1, [10, 50], [50, 30+1]) ≈ [zeros(3); [1]]
    setstate!(ukf1, [1,2,3,4])
    @test ukf1.x̂ ≈ [1,2,3,4]
    for i in 1:100
        updatestate!(ukf1, [11, 52], [50, 30])
    end
    @test ukf1() ≈ [50, 30] atol=1e-3
    for i in 1:100
        updatestate!(ukf1, [10, 50], [51, 32])
    end
    @test ukf1() ≈ [51, 32] atol=1e-3
    lo2 = Luenberger(linmodel1, nint_u=[1, 1])
    for i in 1:100
        updatestate!(lo2, [11, 52], [50, 30])
    end
    @test lo2() ≈ [50, 30] atol=1e-3
    for i in 1:100
        updatestate!(lo2, [10, 50], [51, 32])
    end
    @test lo2() ≈ [51, 32] atol=1e-3
    linmodel3 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    lo3 = Luenberger(linmodel3)
    x̂ = updatestate!(lo3, [0], [0])
    @test x̂ ≈ [0, 0]
    @test isa(x̂, Vector{Float32})
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

    f(x,u,d) = linmodel2.A*x + linmodel2.Bu*u + linmodel2.Bd*d
    h(x,d)   = linmodel2.C*x + linmodel2.Dd*d
    nonlinmodel = NonLinModel(f, h, Ts, 2, 4, 2, 2, solver=nothing)
    internalmodel3 = InternalModel(nonlinmodel)
    @test internalmodel3.nym == 2
    @test internalmodel3.nyu == 0
    @test internalmodel3.nxs == 2
    @test internalmodel3.nx̂  == 4

    stoch_ym_tf = tf([1, -0.3],[1, -0.5],Ts)*tf([1,0],[1,-1],Ts).*I(2)
    internalmodel4 = InternalModel(linmodel2,stoch_ym=stoch_ym_tf)
    @test internalmodel4.nym == 2
    @test internalmodel4.nyu == 0
    @test internalmodel4.nxs == 4
    @test internalmodel4.nx̂ == 4

    stoch_ym_ss=minreal(ss(stoch_ym_tf))
    internalmodel5 = InternalModel(linmodel2,stoch_ym=stoch_ym_ss)
    @test internalmodel5.nym == 2
    @test internalmodel5.nyu == 0
    @test internalmodel5.nxs == 4
    @test internalmodel5.nx̂ == 4
    @test internalmodel5.As ≈ stoch_ym_ss.A
    @test internalmodel5.Bs ≈ stoch_ym_ss.B
    @test internalmodel5.Cs ≈ stoch_ym_ss.C
    @test internalmodel5.Ds ≈ stoch_ym_ss.D

    stoch_ym_resample = c2d(d2c(ss(1,1,1,1,linmodel2.Ts), :tustin), 2linmodel2.Ts, :tustin)
    internalmodel6 = InternalModel(linmodel2, i_ym=[2], stoch_ym=stoch_ym_resample)
    @test internalmodel6.As ≈ internalmodel2.As
    @test internalmodel6.Bs ≈ internalmodel2.Bs
    @test internalmodel6.Cs ≈ internalmodel2.Cs
    @test internalmodel6.Ds ≈ internalmodel2.Ds

    stoch_ym_cont = ss(zeros(2,2), I(2), I(2), zeros(2,2))
    stoch_ym_disc = c2d(stoch_ym_cont, linmodel2.Ts, :tustin)
    internalmodel7 = InternalModel(linmodel2, stoch_ym=stoch_ym_cont)
    @test internalmodel7.As ≈ stoch_ym_disc.A
    @test internalmodel7.Bs ≈ stoch_ym_disc.B
    @test internalmodel7.Cs ≈ stoch_ym_disc.C
    @test internalmodel7.Ds ≈ stoch_ym_disc.D

    linmodel3 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    internalmodel8 = InternalModel(linmodel3)
    @test isa(internalmodel8, InternalModel{Float32})

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
    @test ModelPredictiveControl.evalŷ(internalmodel1, [51,31], Float64[]) ≈ [51,31]
    @test initstate!(internalmodel1, [10, 50], [50, 30]) ≈ zeros(2)
    linmodel2 = LinModel(append(tf(3, [5, 1]), tf(2, [10, 1])), 1.0)
    stoch_ym = append(tf([2.5, 1],[1.2, 1, 0]),tf([1.5, 1], [1.3, 1, 0]))
    estim = InternalModel(linmodel2; stoch_ym)
    initstate!(estim, [1, 2], [3+0.1, 4+0.5])
    @test estim.x̂d ≈ estim.Â*estim.x̂d + estim.B̂u*[1, 2]
    ŷs = [3+0.1, 4+0.5] - estim()
    @test estim.x̂s ≈ estim.Âs*estim.x̂s + estim.B̂s*ŷs
    @test internalmodel1.x̂s ≈ zeros(2)
    setstate!(internalmodel1, [1,2])
    @test internalmodel1.x̂ ≈ [1,2]

    linmodel3 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    internalmodel3 = InternalModel(linmodel3)
    x̂ = updatestate!(internalmodel3, [0], [0])
    @test x̂ ≈ [0]
    @test isa(x̂, Vector{Float32})
end
 
@testset "UnscentedKalmanFilter construction" begin
    linmodel1 = LinModel(sys,Ts,i_d=[3])
    f(x,u,d) = linmodel1.A*x + linmodel1.Bu*u + linmodel1.Bd*d
    h(x,d)   = linmodel1.C*x + linmodel1.Du*d
    nonlinmodel = NonLinModel(f, h, Ts, 2, 4, 2, 1, solver=nothing)

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

    ukf4 = UnscentedKalmanFilter(nonlinmodel, σQ=[1,2,3,4], σQint_ym=[5, 6],  σR=[7, 8])
    @test ukf4.Q̂ ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test ukf4.R̂ ≈ Hermitian(diagm(Float64[49, 64]))
    
    ukf5 = UnscentedKalmanFilter(nonlinmodel, nint_ym=[2,2])
    @test ukf5.nxs == 4
    @test ukf5.nx̂ == 8

    ukf6 = UnscentedKalmanFilter(nonlinmodel, σP0=[1,2,3,4], σP0int_ym=[5,6])
    @test ukf6.P̂0 ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test ukf6.P̂  ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test ukf6.P̂0 !== ukf6.P̂

    ukf7 = UnscentedKalmanFilter(nonlinmodel, α=0.1, β=4, κ=0.2)
    @test ukf7.γ ≈ 0.1*√(ukf7.nx̂+0.2)
    @test ukf7.Ŝ[1, 1] ≈ 2 - 0.1^2 + 4 - ukf7.nx̂/(ukf7.γ^2)

    ukf8 = UnscentedKalmanFilter(nonlinmodel, nint_u=[1, 1], nint_ym=[0, 0])
    @test ukf8.nxs == 2
    @test ukf8.nx̂  == 6
    @test ukf8.nint_u  == [1, 1]
    @test ukf8.nint_ym == [0, 0]

    ukf9 = UnscentedKalmanFilter(nonlinmodel, 1:2, 0, [1, 1], I(6), I(6), I(2), 0.1, 2, 0)
    @test ukf9.P̂0 ≈ I(6)
    @test ukf9.Q̂ ≈ I(6)
    @test ukf9.R̂ ≈ I(2)

    linmodel2 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    ukf10 = UnscentedKalmanFilter(linmodel2)
    @test isa(ukf10, UnscentedKalmanFilter{Float32})
end

@testset "UnscentedKalmanFilter estimator methods" begin
    linmodel1 = LinModel(sys,Ts,i_u=[1,2])
    f(x,u,_) = linmodel1.A*x + linmodel1.Bu*u
    h(x,_)   = linmodel1.C*x
    nonlinmodel = setop!(NonLinModel(f, h, Ts, 2, 2, 2, solver=nothing), uop=[10,50], yop=[50,30])
    ukf1 = UnscentedKalmanFilter(nonlinmodel)
    @test updatestate!(ukf1, [10, 50], [50, 30]) ≈ zeros(4) atol=1e-9
    @test updatestate!(ukf1, [10, 50], [50, 30], Float64[]) ≈ zeros(4) atol=1e-9
    @test ukf1.x̂ ≈ zeros(4) atol=1e-9
    @test evaloutput(ukf1) ≈ ukf1() ≈ [50, 30]
    @test evaloutput(ukf1, Float64[]) ≈ ukf1(Float64[]) ≈ [50, 30]
    @test initstate!(ukf1, [10, 50], [50, 30+1]) ≈ zeros(4) atol=1e-9
    setstate!(ukf1, [1,2,3,4])
    @test ukf1.x̂ ≈ [1,2,3,4]
    for i in 1:100
        updatestate!(ukf1, [11, 52], [50, 30])
    end
    @test ukf1() ≈ [50, 30] atol=1e-3
    for i in 1:100
        updatestate!(ukf1, [10, 50], [51, 32])
    end
    @test ukf1() ≈ [51, 32] atol=1e-3
    ukf2 = UnscentedKalmanFilter(linmodel1, nint_u=[1, 1], nint_ym=[0, 0])
    for i in 1:100
        updatestate!(ukf2, [11, 52], [50, 30])
    end
    @test ukf2() ≈ [50, 30] atol=1e-3
    for i in 1:100
        updatestate!(ukf2, [10, 50], [51, 32])
    end
    @test ukf2() ≈ [51, 32] atol=1e-3
    linmodel3 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    ukf3 = UnscentedKalmanFilter(linmodel3)
    x̂ = updatestate!(ukf3, [0], [0])
    @test x̂ ≈ [0, 0] atol=1e-3
    @test isa(x̂, Vector{Float32})
end

@testset "ExtendedKalmanFilter construction" begin
    linmodel1 = LinModel(sys,Ts,i_d=[3])
    f(x,u,d) = linmodel1.A*x + linmodel1.Bu*u + linmodel1.Bd*d
    h(x,d)   = linmodel1.C*x + linmodel1.Du*d
    nonlinmodel = NonLinModel(f, h, Ts, 2, 4, 2, 1, solver=nothing)

    ekf1 = ExtendedKalmanFilter(linmodel1)
    @test ekf1.nym == 2
    @test ekf1.nyu == 0
    @test ekf1.nxs == 2
    @test ekf1.nx̂ == 6

    ekf2 = ExtendedKalmanFilter(nonlinmodel)
    @test ekf2.nym == 2
    @test ekf2.nyu == 0
    @test ekf2.nxs == 2
    @test ekf2.nx̂ == 6

    ekf3 = ExtendedKalmanFilter(nonlinmodel, i_ym=[2])
    @test ekf3.nym == 1
    @test ekf3.nyu == 1
    @test ekf3.nxs == 1
    @test ekf3.nx̂ == 5

    ekf4 = ExtendedKalmanFilter(nonlinmodel, σQ=[1,2,3,4], σQint_ym=[5, 6],  σR=[7, 8])
    @test ekf4.Q̂ ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test ekf4.R̂ ≈ Hermitian(diagm(Float64[49, 64]))
    
    ekf5 = ExtendedKalmanFilter(nonlinmodel, nint_ym=[2,2])
    @test ekf5.nxs == 4
    @test ekf5.nx̂ == 8

    ekf6 = ExtendedKalmanFilter(nonlinmodel, σP0=[1,2,3,4], σP0int_ym=[5,6])
    @test ekf6.P̂0 ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test ekf6.P̂  ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test ekf6.P̂0 !== ekf6.P̂

    ekf7 = ExtendedKalmanFilter(nonlinmodel, nint_u=[1,1], nint_ym=[0,0])
    @test ekf7.nxs == 2
    @test ekf7.nx̂  == 6
    @test ekf7.nint_u  == [1, 1]
    @test ekf7.nint_ym == [0, 0]

    ekf8 = ExtendedKalmanFilter(nonlinmodel, 1:2, 0, [1, 1], I(6), I(6), I(2))
    @test ekf8.P̂0 ≈ I(6)
    @test ekf8.Q̂ ≈ I(6)
    @test ekf8.R̂ ≈ I(2)

    linmodel2 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    ekf8 = ExtendedKalmanFilter(linmodel2)
    @test isa(ekf8, ExtendedKalmanFilter{Float32})
end

@testset "ExtendedKalmanFilter estimator methods" begin
    linmodel1 = LinModel(sys,Ts,i_u=[1,2])
    f(x,u,_) = linmodel1.A*x + linmodel1.Bu*u
    h(x,_)   = linmodel1.C*x
    nonlinmodel = setop!(NonLinModel(f, h, Ts, 2, 2, 2, solver=nothing), uop=[10,50], yop=[50,30])
    ekf1 = ExtendedKalmanFilter(nonlinmodel)
    @test updatestate!(ekf1, [10, 50], [50, 30]) ≈ zeros(4) atol=1e-9
    @test updatestate!(ekf1, [10, 50], [50, 30], Float64[]) ≈ zeros(4) atol=1e-9
    @test ekf1.x̂ ≈ zeros(4) atol=1e-9
    @test evaloutput(ekf1) ≈ ekf1() ≈ [50, 30]
    @test evaloutput(ekf1, Float64[]) ≈ ekf1(Float64[]) ≈ [50, 30]
    @test initstate!(ekf1, [10, 50], [50, 30+1]) ≈ zeros(4);
    setstate!(ekf1, [1,2,3,4])
    @test ekf1.x̂ ≈ [1,2,3,4]
    for i in 1:100
        updatestate!(ekf1, [11, 52], [50, 30])
    end
    @test ekf1() ≈ [50, 30] atol=1e-3
    for i in 1:100
        updatestate!(ekf1, [10, 50], [51, 32])
    end
    @test ekf1() ≈ [51, 32] atol=1e-3
    ekf2 = ExtendedKalmanFilter(linmodel1, nint_u=[1, 1], nint_ym=[0, 0])
    for i in 1:100
        updatestate!(ekf2, [11, 52], [50, 30])
    end
    @test ekf2() ≈ [50, 30] atol=1e-3
    for i in 1:100
        updatestate!(ekf2, [10, 50], [51, 32])
    end
    @test ekf2() ≈ [51, 32] atol=1e-3
    linmodel3 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    ekf3 = ExtendedKalmanFilter(linmodel3)
    x̂ = updatestate!(ekf3, [0], [0])
    @test x̂ ≈ [0, 0]
    @test isa(x̂, Vector{Float32})
end

@testset "MovingHorizonEstimator construction" begin
    linmodel1 = LinModel(sys,Ts,i_d=[3])
    f(x,u,d) = linmodel1.A*x + linmodel1.Bu*u + linmodel1.Bd*d
    h(x,d)   = linmodel1.C*x + linmodel1.Du*d
    nonlinmodel = NonLinModel(f, h, Ts, 2, 4, 2, 1, solver=nothing)

    mhe1 = MovingHorizonEstimator(linmodel1, He=5)
    @test mhe1.nym == 2
    @test mhe1.nyu == 0
    @test mhe1.nxs == 2
    @test mhe1.nx̂ == 6
    @test size(mhe1.Ẽ, 2) == 6*mhe1.nx̂

    mhe2 = MovingHorizonEstimator(nonlinmodel, He=5)
    @test mhe2.nym == 2
    @test mhe2.nyu == 0
    @test mhe2.nxs == 2
    @test mhe2.nx̂ == 6
    @test size(mhe1.Ẽ, 2) == 6*mhe1.nx̂

    mhe3 = MovingHorizonEstimator(nonlinmodel, He=5, i_ym=[2])
    @test mhe3.nym == 1
    @test mhe3.nyu == 1
    @test mhe3.nxs == 1
    @test mhe3.nx̂ == 5

    mhe4 = MovingHorizonEstimator(nonlinmodel, He=5, σQ=[1,2,3,4], σQint_ym=[5, 6], σR=[7, 8])
    @test mhe4.Q̂ ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test mhe4.R̂ ≈ Hermitian(diagm(Float64[49, 64]))
    
    mhe5 = MovingHorizonEstimator(nonlinmodel, He=5, nint_ym=[2,2])
    @test mhe5.nxs == 4
    @test mhe5.nx̂ == 8

    mhe6 = MovingHorizonEstimator(nonlinmodel, He=5, σP0=[1,2,3,4], σP0int_ym=[5,6])
    @test mhe6.P̂0       ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test mhe6.P̂arr_old ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test mhe6.P̂0 !== mhe6.P̂arr_old

    mhe7 = MovingHorizonEstimator(nonlinmodel, He=10)
    @test mhe7.He == 10
    @test length(mhe7.X̂)  == 10*6
    @test length(mhe7.Ym) == 10*2
    @test length(mhe7.U)  == 10*2
    @test length(mhe7.D)  == 10*1
    @test length(mhe7.Ŵ)  == 10*6

    mhe8 = MovingHorizonEstimator(nonlinmodel, He=5, nint_u=[1, 1], nint_ym=[0, 0])
    @test mhe8.nxs == 2
    @test mhe8.nx̂  == 6
    @test mhe8.nint_u  == [1, 1]
    @test mhe8.nint_ym == [0, 0]

    I_6 = Matrix{Float64}(I, 6, 6)
    I_2 = Matrix{Float64}(I, 2, 2)
    optim = Model(Ipopt.Optimizer)
    mhe9 = MovingHorizonEstimator(nonlinmodel, 5, 1:2, 0, [1, 1], I_6, I_6, I_2, 1e5 ,optim)
    @test mhe9.P̂0 ≈ I(6)
    @test mhe9.Q̂ ≈ I(6)
    @test mhe9.R̂ ≈ I(2)

    optim = Model(Ipopt.Optimizer)
    covestim = ExtendedKalmanFilter(nonlinmodel, 1:2, 0, [1, 1], I_6, I_6, I_2)
    mhe10 = MovingHorizonEstimator(
        nonlinmodel, 5, 1:2, 0, [1, 1], I_6, I_6, I_2, Inf, optim, covestim
    )

    mhe11 = MovingHorizonEstimator(nonlinmodel, He=5, optim=Model(OSQP.Optimizer))
    @test solver_name(mhe11.optim) == "OSQP"

    mhe12 = MovingHorizonEstimator(nonlinmodel, He=5, Cwt=1e3)
    @test size(mhe12.Ẽ, 2) == 6*mhe12.nx̂ + 1
    @test mhe12.C == 1e3

    linmodel2 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    mhe13 = MovingHorizonEstimator(linmodel2, He=5)
    @test isa(mhe13, MovingHorizonEstimator{Float32})

    @test_throws ArgumentError MovingHorizonEstimator(linmodel1)
    @test_throws ArgumentError MovingHorizonEstimator(linmodel1, He=0)
    @test_throws ArgumentError MovingHorizonEstimator(linmodel1, Cwt=-1)
end

@testset "MovingHorizonEstimator estimation and getinfo" begin
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2], i_d=[3]), uop=[10,50], yop=[50,30], dop=[5])
    f(x,u,d) = linmodel1.A*x + linmodel1.Bu*u + linmodel1.Bd*d
    h(x,d)   = linmodel1.C*x + linmodel1.Dd*d
    nonlinmodel = setop!(NonLinModel(f, h, Ts, 2, 4, 2, 1, solver=nothing), uop=[10,50], yop=[50,30], dop=[5])
    mhe1 = MovingHorizonEstimator(nonlinmodel, He=2)
    x̂ = updatestate!(mhe1, [10, 50], [50, 30], [5])
    @test x̂ ≈ zeros(6) atol=1e-9
    @test mhe1.x̂ ≈ zeros(6) atol=1e-9
    @test evaloutput(mhe1, [5]) ≈ mhe1([5]) ≈ [50, 30]
    info = getinfo(mhe1)
    @test info[:x̂] ≈ x̂ atol=1e-9
    @test info[:Ŷ][end-1:end] ≈ [50, 30] atol=1e-9

    @test initstate!(mhe1, [10, 50], [50, 30+1], [5]) ≈ zeros(6) atol=1e-9
    setstate!(mhe1, [1,2,3,4,5,6])
    @test mhe1.x̂ ≈ [1,2,3,4,5,6]
    for i in 1:100
        updatestate!(mhe1, [11, 52], [50, 30], [5])
    end
    @test mhe1([5]) ≈ [50, 30] atol=1e-3
    for i in 1:100
        updatestate!(mhe1, [10, 50], [51, 32], [5])
    end
    @test mhe1([5]) ≈ [51, 32] atol=1e-3
    
    mhe2 = MovingHorizonEstimator(linmodel1, He=2, nint_u=[1, 1], nint_ym=[0, 0])
    x̂ = updatestate!(mhe2, [10, 50], [50, 30], [5])
    @test x̂ ≈ zeros(6) atol=1e-9
    @test mhe2.x̂ ≈ zeros(6) atol=1e-9
    @test evaloutput(mhe2, [5]) ≈ mhe2([5]) ≈ [50, 30]
    info = getinfo(mhe2)
    @test info[:x̂] ≈ x̂ atol=1e-9
    @test info[:Ŷ][end-1:end] ≈ [50, 30] atol=1e-9
    for i in 1:100
        updatestate!(mhe2, [11, 52], [50, 30], [5])
    end
    @test mhe2([5]) ≈ [50, 30] atol=1e-3
    for i in 1:100
        updatestate!(mhe2, [10, 50], [51, 32], [5])
    end
    @test mhe2([5]) ≈ [51, 32] atol=1e-3
    linmodel3 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    mhe3 = MovingHorizonEstimator(linmodel3, He=1)
    x̂ = updatestate!(mhe3, [0], [0])
    @test x̂ ≈ [0, 0] atol=1e-3
    @test isa(x̂, Vector{Float32})

    mhe4 = setconstraint!(MovingHorizonEstimator(nonlinmodel, He=1, nint_ym=0), x̂max=[50,50,50,50])
    g_X̂max_end = mhe4.optim.nlp_model.operators.registered_multivariate_operators[end].f
    # test gfunc_i(i,::NTuple{N, Float64}):
    @test g_X̂max_end(
        (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)) ≤ 0.0 
    # test gfunc_i(i,::NTuple{N, ForwardDiff.Dual}): 
    @test ForwardDiff.gradient(
        g_X̂max_end, [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]) ≈ [0, 0, 0, 0, 0, 0, 0, 0]
    Q̂ = diagm([1/4, 1/4, 1/4, 1/4].^2) 
    R̂ = diagm([1, 1].^2)
    optim = Model(Ipopt.Optimizer)
    covestim = ExtendedKalmanFilter(nonlinmodel, 1:2, 0, 0, Q̂, Q̂, R̂)
    mhe5 = MovingHorizonEstimator(nonlinmodel, 1, 1:2, 0, 0, Q̂, Q̂, R̂, Inf, optim, covestim)
    x̂ = updatestate!(mhe5, [10, 50], [50, 30], [5])
    @test x̂ ≈ zeros(4) atol=1e-9
    @test mhe5.x̂ ≈ zeros(4) atol=1e-9
    @test evaloutput(mhe5, [5]) ≈ mhe5([5]) ≈ [50, 30]
    info = getinfo(mhe5)
    @test info[:x̂] ≈ x̂ atol=1e-9
    @test info[:Ŷ][end-1:end] ≈ [50, 30] atol=1e-9
end

@testset "MovingHorizonEstimator set constraints" begin
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    mhe1 = MovingHorizonEstimator(linmodel1, He=1, nint_ym=0, Cwt=1e3)
    setconstraint!(mhe1, x̂min=[-51,-52], x̂max=[53,54])
    @test all((mhe1.con.X̂min, mhe1.con.X̂max) .≈ ([-51,-52], [53,54]))
    @test all((mhe1.con.x̃min[2:end], mhe1.con.x̃max[2:end]) .≈ ([-51,-52], [53,54]))
    setconstraint!(mhe1, ŵmin=[-55,-56], ŵmax=[57,58])
    @test all((mhe1.con.Ŵmin, mhe1.con.Ŵmax) .≈ ([-55,-56], [57,58]))
    setconstraint!(mhe1, v̂min=[-59,-60], v̂max=[61,62])
    @test all((mhe1.con.V̂min, mhe1.con.V̂max) .≈ ([-59,-60], [61,62]))
    setconstraint!(mhe1, c_x̂min=[0.01,0.02], c_x̂max=[0.03,0.04])
    @test all((-mhe1.con.A_X̂min[:, end], -mhe1.con.A_X̂max[:, end]) .≈ ([0.01, 0.02], [0.03,0.04]))
    @test all((-mhe1.con.A_x̃min[2:end, end], -mhe1.con.A_x̃max[2:end, end]) .≈ ([0.01,0.02], [0.03,0.04]))
    setconstraint!(mhe1, c_ŵmin=[0.05,0.06], c_ŵmax=[0.07,0.08])
    @test all((-mhe1.con.A_Ŵmin[:, end], -mhe1.con.A_Ŵmax[:, end]) .≈ ([0.05, 0.06], [0.07,0.08]))
    setconstraint!(mhe1, c_v̂min=[0.09,0.10], c_v̂max=[0.11,0.12])
    @test all((-mhe1.con.A_V̂min[:, end], -mhe1.con.A_V̂max[:, end]) .≈ ([0.09, 0.10], [0.11,0.12]))

    mhe2 = MovingHorizonEstimator(linmodel1, He=4, nint_ym=0, Cwt=1e3)
    setconstraint!(mhe2, X̂min=-1(1:10), X̂max=1(1:10))
    @test all((mhe2.con.X̂min, mhe2.con.X̂max) .≈ (-1(3:10), 1(3:10)))
    @test all((mhe2.con.x̃min[2:end], mhe2.con.x̃max[2:end]) .≈ (-1(1:2),  1(1:2)))
    setconstraint!(mhe2, Ŵmin=-1(11:18), Ŵmax=1(11:18))
    @test all((mhe2.con.Ŵmin, mhe2.con.Ŵmax) .≈ (-1(11:18), 1(11:18)))
    setconstraint!(mhe2, V̂min=-1(31:38), V̂max=1(31:38))
    @test all((mhe2.con.V̂min, mhe2.con.V̂max) .≈ (-1(31:38), 1(31:38)))
    setconstraint!(mhe2, C_x̂min=0.01(1:10), C_x̂max=0.02(1:10))
    @test all((-mhe2.con.A_X̂min[:, end], -mhe2.con.A_X̂max[:, end]) .≈ (0.01(3:10), 0.02(3:10)))
    @test all((-mhe2.con.A_x̃min[2:end, end], -mhe2.con.A_x̃max[2:end, end]) .≈ (0.01(1:2), 0.02(1:2)))
    setconstraint!(mhe2, C_ŵmin=0.03(11:18), C_ŵmax=0.04(11:18))
    @test all((-mhe2.con.A_Ŵmin[:, end], -mhe2.con.A_Ŵmax[:, end]) .≈ (0.03(11:18), 0.04(11:18)))
    setconstraint!(mhe2, C_v̂min=0.05(31:38), C_v̂max=0.06(31:38))
    @test all((-mhe2.con.A_V̂min[:, end], -mhe2.con.A_V̂max[:, end]) .≈ (0.05(31:38), 0.06(31:38)))

    f(x,u,d) = linmodel1.A*x + linmodel1.Bu*u
    h(x,d)   = linmodel1.C*x 
    nonlinmodel = setop!(NonLinModel(f, h, Ts, 2, 2, 2, solver=nothing), uop=[10,50], yop=[50,30])

    mhe3 = MovingHorizonEstimator(nonlinmodel, He=4, nint_ym=0, Cwt=1e3)
    setconstraint!(mhe3, C_x̂min=0.01(1:10), C_x̂max=0.02(1:10))
    @test all((mhe3.con.C_x̂min, mhe3.con.C_x̂max) .≈ (0.01(3:10), 0.02(3:10)))
    setconstraint!(mhe3, C_v̂min=0.03(11:18), C_v̂max=0.04(11:18))
    @test all((mhe3.con.C_v̂min, mhe3.con.C_v̂max) .≈ (0.03(11:18), 0.04(11:18)))

    @test_throws ArgumentError setconstraint!(mhe2, x̂min=[-1])
    @test_throws ArgumentError setconstraint!(mhe2, x̂max=[+1])
    @test_throws ArgumentError setconstraint!(mhe2, ŵmin=[-1])
    @test_throws ArgumentError setconstraint!(mhe2, ŵmax=[+1])
    @test_throws ArgumentError setconstraint!(mhe2, v̂min=[-1])
    @test_throws ArgumentError setconstraint!(mhe2, v̂max=[+1])
    @test_throws ArgumentError setconstraint!(mhe2, c_x̂min=[-1])
    @test_throws ArgumentError setconstraint!(mhe2, c_x̂max=[+1])
    @test_throws ArgumentError setconstraint!(mhe2, c_ŵmin=[-1])
    @test_throws ArgumentError setconstraint!(mhe2, c_ŵmax=[+1])
    @test_throws ArgumentError setconstraint!(mhe2, c_v̂min=[-1])
    @test_throws ArgumentError setconstraint!(mhe2, c_v̂max=[+1])

    updatestate!(mhe1, [10, 50], [50, 30])
    @test_throws ErrorException setconstraint!(mhe1, x̂min=[-Inf,-Inf])
    @test_throws ErrorException setconstraint!(mhe1, x̂max=[+Inf,+Inf])
    @test_throws ErrorException setconstraint!(mhe1, ŵmin=[-Inf,-Inf])
    @test_throws ErrorException setconstraint!(mhe1, ŵmax=[+Inf,+Inf])
    @test_throws ErrorException setconstraint!(mhe1, v̂min=[-Inf,-Inf])
    @test_throws ErrorException setconstraint!(mhe1, v̂max=[+Inf,+Inf])
    @test_throws ErrorException setconstraint!(mhe1, c_x̂min=[100,100])
    @test_throws ErrorException setconstraint!(mhe1, c_x̂max=[200,200])
    @test_throws ErrorException setconstraint!(mhe1, c_ŵmin=[300,300])
    @test_throws ErrorException setconstraint!(mhe1, c_ŵmax=[400,400])
    @test_throws ErrorException setconstraint!(mhe1, c_v̂min=[500,500])
    @test_throws ErrorException setconstraint!(mhe1, c_v̂max=[600,600])

    mhe4 = MovingHorizonEstimator(nonlinmodel, He=1, nint_ym=0, Cwt=Inf)
    @test_throws ArgumentError setconstraint!(mhe4, c_x̂min=[1,1])
    @test_throws ArgumentError setconstraint!(mhe4, c_x̂max=[1,1])
    @test_throws ArgumentError setconstraint!(mhe4, c_ŵmin=[1,1])
    @test_throws ArgumentError setconstraint!(mhe4, c_ŵmax=[1,1])
    @test_throws ArgumentError setconstraint!(mhe4, c_v̂min=[1,1])
    @test_throws ArgumentError setconstraint!(mhe4, c_v̂max=[1,1])
end

@testset "MovingHorizonEstimator constraint violation" begin
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    mhe = MovingHorizonEstimator(linmodel1, He=1, nint_ym=0)

    setconstraint!(mhe, x̂min=[-100,-100], x̂max=[100,100])
    setconstraint!(mhe, ŵmin=[-100,-100], ŵmax=[100,100])
    setconstraint!(mhe, v̂min=[-100,-100], v̂max=[100,100])

    setconstraint!(mhe, x̂min=[1,1], x̂max=[100,100])
    x̂ = updatestate!(mhe, [10, 50], [50, 30])
    @test x̂ ≈ [1, 1] atol=1e-2

    setconstraint!(mhe, x̂min=[-100,-100], x̂max=[-1,-1])
    x̂ = updatestate!(mhe, [10, 50], [50, 30])
    @test x̂ ≈ [-1, -1] atol=1e-2

    setconstraint!(mhe, x̂min=[-100,-100], x̂max=[100,100])
    setconstraint!(mhe, ŵmin=[-100,-100], ŵmax=[100,100])
    setconstraint!(mhe, v̂min=[-100,-100], v̂max=[100,100])

    setconstraint!(mhe, ŵmin=[1,1], ŵmax=[100,100])
    x̂ = updatestate!(mhe, [10, 50], [50, 30])
    @test mhe.Ŵ ≈ [1,1] atol=1e-2

    setconstraint!(mhe, ŵmin=[-100,-100], ŵmax=[-1,-1])
    x̂ = updatestate!(mhe, [10, 50], [50, 30])
    @test mhe.Ŵ ≈ [-1,-1] atol=1e-2

    setconstraint!(mhe, x̂min=[-100,-100], x̂max=[100,100])
    setconstraint!(mhe, ŵmin=[-100,-100], ŵmax=[100,100])
    setconstraint!(mhe, v̂min=[-100,-100], v̂max=[100,100])

    setconstraint!(mhe, v̂min=[1,1], v̂max=[100,100])
    x̂ = updatestate!(mhe, [10, 50], [50, 30])
    info = getinfo(mhe)
    @test info[:V̂] ≈ [1,1] atol=1e-2

    setconstraint!(mhe, v̂min=[-100,-100], v̂max=[-1,-1])
    x̂ = updatestate!(mhe, [10, 50], [50, 30])
    info = getinfo(mhe)
    @test info[:V̂] ≈ [-1,-1] atol=1e-2

    f(x,u,_) = linmodel1.A*x + linmodel1.Bu*u
    h(x,_)   = linmodel1.C*x
    nonlinmodel = setop!(NonLinModel(f, h, Ts, 2, 2, 2, solver=nothing), uop=[10,50], yop=[50,30])
    mhe2 = MovingHorizonEstimator(nonlinmodel, He=1, nint_ym=0)

    setconstraint!(mhe2, x̂min=[-100,-100], x̂max=[100,100])
    setconstraint!(mhe2, ŵmin=[-100,-100], ŵmax=[100,100])
    setconstraint!(mhe2, v̂min=[-100,-100], v̂max=[100,100])

    setconstraint!(mhe2, x̂min=[1,1], x̂max=[100,100])
    x̂ = updatestate!(mhe2, [10, 50], [50, 30])
    @test x̂ ≈ [1, 1] atol=1e-2

    setconstraint!(mhe2, x̂min=[-100,-100], x̂max=[-1,-1])
    x̂ = updatestate!(mhe2, [10, 50], [50, 30])
    @test x̂ ≈ [-1, -1] atol=1e-2

    setconstraint!(mhe2, x̂min=[-100,-100], x̂max=[100,100])
    setconstraint!(mhe2, ŵmin=[-100,-100], ŵmax=[100,100])
    setconstraint!(mhe2, v̂min=[-100,-100], v̂max=[100,100])

    setconstraint!(mhe2, ŵmin=[1,1], ŵmax=[100,100])
    x̂ = updatestate!(mhe2, [10, 50], [50, 30])
    @test mhe2.Ŵ ≈ [1,1] atol=1e-2

    setconstraint!(mhe2, ŵmin=[-100,-100], ŵmax=[-1,-1])
    x̂ = updatestate!(mhe2, [10, 50], [50, 30])
    @test mhe2.Ŵ ≈ [-1,-1] atol=1e-2

    setconstraint!(mhe2, x̂min=[-100,-100], x̂max=[100,100])
    setconstraint!(mhe2, ŵmin=[-100,-100], ŵmax=[100,100])
    setconstraint!(mhe2, v̂min=[-100,-100], v̂max=[100,100])

    setconstraint!(mhe2, v̂min=[1,1], v̂max=[100,100])
    x̂ = updatestate!(mhe2, [10, 50], [50, 30])
    info = getinfo(mhe2)
    @test info[:V̂] ≈ [1,1] atol=1e-2

    setconstraint!(mhe2, v̂min=[-100,-100], v̂max=[-1,-1])
    x̂ = updatestate!(mhe2, [10, 50], [50, 30])
    info = getinfo(mhe2)
    @test info[:V̂] ≈ [-1,-1] atol=1e-2
end
