@testitem "SteadyKalmanFilter construction" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
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

@testitem "SteadyKalmanFilter estimator methods" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    skalmanfilter1 = SteadyKalmanFilter(linmodel1, nint_ym=[1, 1])
    preparestate!(skalmanfilter1, [50, 30])
    @test updatestate!(skalmanfilter1, [10, 50], [50, 30]) ≈ zeros(4)
    preparestate!(skalmanfilter1, [50, 30])
    @test updatestate!(skalmanfilter1, [10, 50], [50, 30], Float64[]) ≈ zeros(4)
    @test skalmanfilter1.x̂0 ≈ zeros(4)
    preparestate!(skalmanfilter1, [50, 30])
    @test evaloutput(skalmanfilter1) ≈ skalmanfilter1() ≈ [50, 30]
    @test evaloutput(skalmanfilter1, Float64[]) ≈ skalmanfilter1(Float64[]) ≈ [50, 30]
    @test initstate!(skalmanfilter1, [10, 50], [50, 30+1]) ≈ [zeros(3); [1]]
    linmodel2 = LinModel(append(tf(1, [1, 0]), tf(2, [10, 1])), 1.0)
    skalmanfilter2 = SteadyKalmanFilter(linmodel2, nint_u=[1, 1], direct=false)
    x = initstate!(skalmanfilter2, [10, 3], [0.5, 6+0.1])
    @test evaloutput(skalmanfilter2) ≈ [0.5, 6+0.1]
    @test updatestate!(skalmanfilter2, [10, 3], [0.5, 6+0.1]) ≈ x
    setstate!(skalmanfilter1, [1,2,3,4])
    @test skalmanfilter1.x̂0 ≈ [1,2,3,4]
    for i in 1:40
        preparestate!(skalmanfilter1, [50, 30])
        updatestate!(skalmanfilter1, [11, 52], [50, 30])
    end
    preparestate!(skalmanfilter1, [50, 30])
    @test skalmanfilter1() ≈ [50, 30] atol=1e-3
    for i in 1:40
        preparestate!(skalmanfilter1, [51, 32])
        updatestate!(skalmanfilter1, [10, 50], [51, 32])
    end
    preparestate!(skalmanfilter1, [51, 32])
    @test skalmanfilter1() ≈ [51, 32] atol=1e-3
    skalmanfilter2 = SteadyKalmanFilter(linmodel1, nint_u=[1, 1], direct=false)
    for i in 1:40
        preparestate!(skalmanfilter2, [50, 30])
        updatestate!(skalmanfilter2, [11, 52], [50, 30])
    end
    @test skalmanfilter2() ≈ [50, 30] atol=1e-3
    for i in 1:40
        preparestate!(skalmanfilter2, [51, 32])
        updatestate!(skalmanfilter2, [10, 50], [51, 32])
    end
    @test skalmanfilter2() ≈ [51, 32] atol=1e-3
    linmodel3 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    skalmanfilter3 = SteadyKalmanFilter(linmodel3)
    preparestate!(skalmanfilter3, [0])
    x̂ = updatestate!(skalmanfilter3, [0], [0])
    @test x̂ ≈ [0, 0]
    @test isa(x̂, Vector{Float32})
    @test_throws ArgumentError updatestate!(skalmanfilter1, [10, 50])
end 

@testitem "SteadyKalmanFilter set model" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel = LinModel(ss(0.5, 0.3, 1.0, 0, 10.0))
    linmodel = setop!(linmodel, uop=[2.0], yop=[50.0], xop=[3.0], fop=[3.0])
    skalmanfilter = SteadyKalmanFilter(linmodel, nint_ym=0)
    @test_nowarn setmodel!(skalmanfilter, linmodel)
    @test_throws ErrorException setmodel!(skalmanfilter, deepcopy(linmodel))
    @test_throws ErrorException setmodel!(skalmanfilter, linmodel, Q̂=[0.01])
    @test_throws ErrorException setmodel!(skalmanfilter, linmodel, R̂=[0.01])
end

@testitem "SteadyKalmanFilter real-time simulations" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel1 = LinModel(tf(2, [10, 1]), 0.25)
    skalmanfilter1 = SteadyKalmanFilter(linmodel1)
    times1 = zeros(5)
    for i=1:5
        times1[i] = savetime!(skalmanfilter1)
        preparestate!(skalmanfilter1, [1])
        updatestate!(skalmanfilter1, [1], [1])
        periodsleep(skalmanfilter1, true)
    end
    @test all(isapprox.(diff(times1[2:end]), 0.25, atol=0.01))
end
    
@testitem "KalmanFilter construction" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
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

    kalmanfilter6 = KalmanFilter(linmodel2, σP_0=[1,2,3,4], σPint_ym_0=[5,6])
    @test kalmanfilter6.P̂_0 ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test kalmanfilter6.P̂  ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test kalmanfilter6.P̂_0 !== kalmanfilter6.P̂

    kalmanfilter7 = KalmanFilter(linmodel1, nint_u=[1,1])
    @test kalmanfilter7.nxs == 2
    @test kalmanfilter7.nx̂  == 4
    @test kalmanfilter7.nint_u  == [1, 1]
    @test kalmanfilter7.nint_ym == [0, 0]

    kalmanfilter8 = KalmanFilter(linmodel1, 1:2, 0, [1, 1], I(4), I(4), I(2))
    @test kalmanfilter8.P̂_0 ≈ I(4)
    @test kalmanfilter8.Q̂ ≈ I(4)
    @test kalmanfilter8.R̂ ≈ I(2)

    linmodel2 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    kalmanfilter8 = KalmanFilter(linmodel2)
    @test isa(kalmanfilter8, KalmanFilter{Float32})

    @test_throws ErrorException KalmanFilter(linmodel1, nint_ym=0, σP_0=[1])
end

@testitem "KalmanFilter estimator methods" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    kalmanfilter1 = KalmanFilter(linmodel1)
    preparestate!(kalmanfilter1, [50, 30])
    @test updatestate!(kalmanfilter1, [10, 50], [50, 30]) ≈ zeros(4)
    preparestate!(kalmanfilter1, [50, 30])
    @test updatestate!(kalmanfilter1, [10, 50], [50, 30], Float64[]) ≈ zeros(4)
    @test kalmanfilter1.x̂0 ≈ zeros(4)
    preparestate!(kalmanfilter1, [50, 30])
    @test evaloutput(kalmanfilter1) ≈ kalmanfilter1() ≈ [50, 30]
    @test evaloutput(kalmanfilter1, Float64[]) ≈ kalmanfilter1(Float64[]) ≈ [50, 30]
    @test initstate!(kalmanfilter1, [10, 50], [50, 30+1]) ≈ [zeros(3); [1]]
    setstate!(kalmanfilter1, [1,2,3,4])
    @test kalmanfilter1.x̂0 ≈ [1,2,3,4]
    for i in 1:40
        preparestate!(kalmanfilter1, [50, 30])
        updatestate!(kalmanfilter1, [11, 52], [50, 30])
    end
    preparestate!(kalmanfilter1, [50, 30])
    @test kalmanfilter1() ≈ [50, 30] atol=1e-3
    for i in 1:40
        preparestate!(kalmanfilter1, [51, 32])
        updatestate!(kalmanfilter1, [10, 50], [51, 32])
    end
    preparestate!(kalmanfilter1, [51, 32])
    @test kalmanfilter1() ≈ [51, 32] atol=1e-3
    kalmanfilter2 = KalmanFilter(linmodel1, nint_u=[1, 1], direct=false)
    for i in 1:40
        preparestate!(kalmanfilter2, [50, 30])
        updatestate!(kalmanfilter2, [11, 52], [50, 30])
    end
    @test kalmanfilter2() ≈ [50, 30] atol=1e-3
    for i in 1:40
        preparestate!(kalmanfilter2, [51, 32])
        updatestate!(kalmanfilter2, [10, 50], [51, 32])
    end
    @test kalmanfilter2() ≈ [51, 32] atol=1e-3
    linmodel3 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    kalmanfilter3 = KalmanFilter(linmodel3)
    preparestate!(kalmanfilter3, [0])
    x̂ = updatestate!(kalmanfilter3, [0], [0])
    @test x̂ ≈ [0, 0]
    @test isa(x̂, Vector{Float32})
    @test_throws ArgumentError updatestate!(kalmanfilter1, [10, 50])
end

@testitem "KalmanFilter set model" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel = LinModel(ss(0.5, 0.3, 1.0, 0, 10.0))
    linmodel = setop!(linmodel, uop=[2.0], yop=[50.0], xop=[3.0], fop=[3.0])
    kalmanfilter = KalmanFilter(linmodel, nint_ym=0)
    @test kalmanfilter.Â ≈ [0.5]
    preparestate!(kalmanfilter, [50.0])
    @test evaloutput(kalmanfilter) ≈ [50.0]
    preparestate!(kalmanfilter, [50.0])
    x̂ = updatestate!(kalmanfilter, [2.0], [50.0])
    @test x̂ ≈ [3.0]
    newlinmodel = LinModel(ss(0.2, 0.3, 1.0, 0, 10.0))
    newlinmodel = setop!(newlinmodel, uop=[3.0], yop=[55.0], xop=[3.0], fop=[3.0])
    setmodel!(kalmanfilter, newlinmodel)
    @test kalmanfilter.Â ≈ [0.2]
    preparestate!(kalmanfilter, [55.0])
    @test evaloutput(kalmanfilter) ≈ [55.0]
    @test kalmanfilter.lastu0 ≈ [2.0 - 3.0]
    preparestate!(kalmanfilter, [55.0])
    x̂ = updatestate!(kalmanfilter, [3.0], [55.0])
    @test x̂ ≈ [3.0]
    newlinmodel = setop!(newlinmodel, uop=[3.0], yop=[55.0], xop=[8.0], fop=[8.0])
    setmodel!(kalmanfilter, newlinmodel)
    @test kalmanfilter.x̂0 ≈ [3.0 - 8.0]
    setmodel!(kalmanfilter, Q̂=[1e-3], R̂=[1e-6])
    @test kalmanfilter.Q̂ ≈ [1e-3]
    @test kalmanfilter.R̂ ≈ [1e-6]
end

@testitem "Luenberger construction" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
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
    @test_throws ErrorException Luenberger(linmodel1, poles=[0.5])
    @test_throws ErrorException Luenberger(linmodel1, poles=fill(1.5, lo1.nx̂))
    @test_throws ErrorException Luenberger(LinModel(tf(1,[1, 0]),0.1), poles=[0.5,0.6])
end
    
@testitem "Luenberger estimator methods" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    lo1 = Luenberger(linmodel1, nint_ym=[1, 1])
    preparestate!(lo1, [50, 30])
    @test updatestate!(lo1, [10, 50], [50, 30]) ≈ zeros(4)
    preparestate!(lo1, [50, 30])
    @test updatestate!(lo1, [10, 50], [50, 30], Float64[]) ≈ zeros(4)
    @test lo1.x̂0 ≈ zeros(4)
    preparestate!(lo1, [50, 30])
    @test evaloutput(lo1) ≈ lo1() ≈ [50, 30]
    @test evaloutput(lo1, Float64[]) ≈ lo1(Float64[]) ≈ [50, 30]
    @test initstate!(lo1, [10, 50], [50, 30+1]) ≈ [zeros(3); [1]]
    setstate!(lo1, [1,2,3,4])
    @test lo1.x̂0 ≈ [1,2,3,4]
    for i in 1:40
        preparestate!(lo1, [50, 30])
        updatestate!(lo1, [11, 52], [50, 30])
    end
    preparestate!(lo1, [50, 30])
    @test lo1() ≈ [50, 30] atol=1e-3
    for i in 1:40
        preparestate!(lo1, [51, 32])
        updatestate!(lo1, [10, 50], [51, 32])
    end
    preparestate!(lo1, [51, 32])
    @test lo1() ≈ [51, 32] atol=1e-3
    lo2 = Luenberger(linmodel1, nint_u=[1, 1], direct=false)
    for i in 1:40
        preparestate!(lo2, [50, 30])
        updatestate!(lo2, [11, 52], [50, 30])
    end
    @test lo2() ≈ [50, 30] atol=1e-3
    for i in 1:40
        preparestate!(lo2, [51, 32])
        updatestate!(lo2, [10, 50], [51, 32])
    end
    @test lo2() ≈ [51, 32] atol=1e-3
    linmodel3 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    lo3 = Luenberger(linmodel3)
    preparestate!(lo3, [0])
    x̂ = updatestate!(lo3, [0], [0])
    @test x̂ ≈ [0, 0]
    @test isa(x̂, Vector{Float32})
end

@testitem "Luenberger set model" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel = LinModel(ss(0.5, 0.3, 1.0, 0, 10.0))
    linmodel = setop!(linmodel, uop=[2.0], yop=[50.0], xop=[3.0], fop=[3.0])
    lo = Luenberger(linmodel, nint_ym=0)
    @test_nowarn setmodel!(lo, linmodel)
    @test_throws ErrorException setmodel!(lo, deepcopy(linmodel))
end

@testitem "InternalModel construction" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
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

    f(x,u,d,_) = linmodel2.A*x + linmodel2.Bu*u + linmodel2.Bd*d
    h(x,d,_)   = linmodel2.C*x + linmodel2.Dd*d
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
    
@testitem "InternalModel estimator methods" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]) , uop=[10,50], yop=[50,30])
    internalmodel1 = InternalModel(linmodel1)
    preparestate!(internalmodel1, [50, 30] .+ 1)
    @test updatestate!(internalmodel1, [10, 50], [50, 30] .+ 1) ≈ zeros(2)
    preparestate!(internalmodel1, [50, 30] .+ 1)
    @test updatestate!(internalmodel1, [10, 50], [50, 30] .+ 1, Float64[]) ≈ zeros(2)
    @test internalmodel1.x̂d ≈ internalmodel1.x̂0 ≈ zeros(2)
    @test internalmodel1.x̂s ≈ ones(2)
    preparestate!(internalmodel1, [51, 31])
    @test evaloutput(internalmodel1, Float64[]) ≈ [51,31]
    @test initstate!(internalmodel1, [10, 50], [50, 30]) ≈ zeros(2)
    @test internalmodel1.x̂s ≈ zeros(2)
    setstate!(internalmodel1, [1,2])
    @test internalmodel1.x̂0 ≈ [1,2]
    linmodel2 = LinModel(append(tf(3, [5, 1]), tf(2, [10, 1])), 1.0)
    stoch_ym = append(tf([2.5, 1],[1.2, 1, 0]),tf([1.5, 1], [1.3, 1, 0]))
    internalmodel2 = InternalModel(linmodel2; stoch_ym)
    initstate!(internalmodel2, [1, 2], [3+0.1, 4+0.5])
    @test internalmodel2.x̂d ≈ internalmodel2.Â*internalmodel2.x̂d + internalmodel2.B̂u*[1, 2]
    preparestate!(internalmodel2, [3+0.1, 4+0.5])
    ŷ = evaloutput(internalmodel2)
    @test ŷ ≈ [3+0.1, 4+0.5]
    x̂s = internalmodel2.x̂s
    @test x̂s ≈ internalmodel2.Âs*x̂s + internalmodel2.B̂s*internalmodel2.ŷs
    updatestate!(internalmodel2, [1, 2], [-13, -14])
    preparestate!(internalmodel2, [13, 14])
    ŷ = internalmodel2()
    @test ŷ ≈ [13, 14]
    linmodel3 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    internalmodel3 = InternalModel(linmodel3)
    preparestate!(internalmodel3, [0])
    x̂ = updatestate!(internalmodel3, [0], [0])
    @test x̂ ≈ [0]
    @test isa(x̂, Vector{Float32})
end

@testitem "InternalModel set model" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel = LinModel(ss(0.5, 0.3, 1.0, 0, 10.0))
    linmodel = setop!(linmodel, uop=[2.0], yop=[50.0], xop=[3.0], fop=[3.0])
    internalmodel = InternalModel(linmodel)
    @test internalmodel.Â ≈ [0.5]
    preparestate!(internalmodel, [50.0])
    @test evaloutput(internalmodel) ≈ [50.0]
    preparestate!(internalmodel, [50.0])
    x̂ = updatestate!(internalmodel, [2.0], [50.0])
    @test x̂ ≈ [3.0]
    newlinmodel = LinModel(ss(0.2, 0.3, 1.0, 0, 10.0))
    newlinmodel = setop!(newlinmodel, uop=[3.0], yop=[55.0], xop=[3.0], fop=[3.0])
    setmodel!(internalmodel, newlinmodel)
    @test internalmodel.Â ≈ [0.2]
    preparestate!(internalmodel, [55.0])
    @test evaloutput(internalmodel) ≈ [55.0]
    @test internalmodel.lastu0 ≈ [2.0 - 3.0]
    preparestate!(internalmodel, [55.0])
    x̂ = updatestate!(internalmodel, [3.0], [55.0])
    @test x̂ ≈ [3.0]
    newlinmodel = setop!(newlinmodel, uop=[3.0], yop=[55.0], xop=[8.0], fop=[8.0])
    setmodel!(internalmodel, newlinmodel)
    @test internalmodel.x̂0 ≈ [3.0 - 8.0]
end
 
@testitem "UnscentedKalmanFilter construction" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel1 = LinModel(sys,Ts,i_d=[3])
    f(x,u,d,_) = linmodel1.A*x + linmodel1.Bu*u + linmodel1.Bd*d
    h(x,d,_)   = linmodel1.C*x + linmodel1.Du*d
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

    ukf6 = UnscentedKalmanFilter(nonlinmodel, σP_0=[1,2,3,4], σPint_ym_0=[5,6])
    @test ukf6.P̂_0 ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test ukf6.P̂  ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test ukf6.P̂_0 !== ukf6.P̂

    ukf7 = UnscentedKalmanFilter(nonlinmodel, α=0.1, β=4, κ=0.2)
    @test ukf7.γ ≈ 0.1*√(ukf7.nx̂+0.2)
    @test ukf7.Ŝ[1, 1] ≈ 2 - 0.1^2 + 4 - ukf7.nx̂/(ukf7.γ^2)

    ukf8 = UnscentedKalmanFilter(nonlinmodel, nint_u=[1, 1], nint_ym=[0, 0])
    @test ukf8.nxs == 2
    @test ukf8.nx̂  == 6
    @test ukf8.nint_u  == [1, 1]
    @test ukf8.nint_ym == [0, 0]

    ukf9 = UnscentedKalmanFilter(nonlinmodel, 1:2, 0, [1, 1], I(6), I(6), I(2), 0.1, 2, 0)
    @test ukf9.P̂_0 ≈ I(6)
    @test ukf9.Q̂ ≈ I(6)
    @test ukf9.R̂ ≈ I(2)

    linmodel2 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    ukf10 = UnscentedKalmanFilter(linmodel2)
    @test isa(ukf10, UnscentedKalmanFilter{Float32})
end

@testitem "UnscentedKalmanFilter estimator methods" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel1 = LinModel(sys,Ts,i_u=[1,2])
    f(x,u,_,_) = linmodel1.A*x + linmodel1.Bu*u
    h(x,_,_)   = linmodel1.C*x
    nonlinmodel = setop!(NonLinModel(f, h, Ts, 2, 2, 2, solver=nothing), uop=[10,50], yop=[50,30])
    ukf1 = UnscentedKalmanFilter(nonlinmodel)
    preparestate!(ukf1, [50, 30])
    @test updatestate!(ukf1, [10, 50], [50, 30]) ≈ zeros(4) atol=1e-9
    preparestate!(ukf1, [50, 30])
    @test updatestate!(ukf1, [10, 50], [50, 30], Float64[]) ≈ zeros(4) atol=1e-9
    @test ukf1.x̂0 ≈ zeros(4) atol=1e-9
    preparestate!(ukf1, [50, 30])
    @test evaloutput(ukf1) ≈ ukf1() ≈ [50, 30]
    @test evaloutput(ukf1, Float64[]) ≈ ukf1(Float64[]) ≈ [50, 30]
    @test initstate!(ukf1, [10, 50], [50, 30+1]) ≈ zeros(4) atol=1e-9
    setstate!(ukf1, [1,2,3,4])
    @test ukf1.x̂0 ≈ [1,2,3,4]
    for i in 1:40
        preparestate!(ukf1, [50, 30])
        updatestate!(ukf1, [11, 52], [50, 30])
    end
    preparestate!(ukf1, [50, 30])
    @test ukf1() ≈ [50, 30] atol=1e-3
    for i in 1:40
        preparestate!(ukf1, [51, 32])
        updatestate!(ukf1, [10, 50], [51, 32])
    end
    preparestate!(ukf1, [51, 32])
    @test ukf1() ≈ [51, 32] atol=1e-3
    ukf2 = UnscentedKalmanFilter(linmodel1, nint_u=[1, 1], nint_ym=[0, 0], direct=false)
    for i in 1:40
        preparestate!(ukf2, [50, 30])
        updatestate!(ukf2, [11, 52], [50, 30])
    end
    @test ukf2() ≈ [50, 30] atol=1e-3
    for i in 1:40
        preparestate!(ukf2, [51, 32])
        updatestate!(ukf2, [10, 50], [51, 32])
    end
    @test ukf2() ≈ [51, 32] atol=1e-3
    linmodel3 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    ukf3 = UnscentedKalmanFilter(linmodel3)
    preparestate!(ukf3, [0])
    x̂ = updatestate!(ukf3, [0], [0])
    @test x̂ ≈ [0, 0] atol=1e-3
    @test isa(x̂, Vector{Float32})
end

@testitem "UnscentedKalmanFilter set model" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel = LinModel(ss(0.5, 0.3, 1.0, 0, 10.0))
    linmodel = setop!(linmodel, uop=[2.0], yop=[50.0], xop=[3.0], fop=[3.0])
    ukf1 = UnscentedKalmanFilter(linmodel, nint_ym=0)
    @test ukf1.Â ≈ [0.5]
    preparestate!(ukf1, [50.0])
    @test evaloutput(ukf1) ≈ [50.0]
    preparestate!(ukf1, [50.0])
    x̂ = updatestate!(ukf1, [2.0], [50.0])
    @test x̂ ≈ [3.0]
    newlinmodel = LinModel(ss(0.2, 0.3, 1.0, 0, 10.0))
    newlinmodel = setop!(newlinmodel, uop=[3.0], yop=[55.0], xop=[3.0], fop=[3.0])
    setmodel!(ukf1, newlinmodel)
    @test ukf1.Â ≈ [0.2]
    preparestate!(ukf1, [55.0])
    @test evaloutput(ukf1) ≈ [55.0]
    @test ukf1.lastu0 ≈ [2.0 - 3.0]
    preparestate!(ukf1, [55.0])
    x̂ = updatestate!(ukf1, [3.0], [55.0])
    @test x̂ ≈ [3.0]
    newlinmodel = setop!(newlinmodel, uop=[3.0], yop=[55.0], xop=[8.0], fop=[8.0])
    setmodel!(ukf1, newlinmodel)
    @test ukf1.x̂0 ≈ [3.0 - 8.0]
    setmodel!(ukf1, Q̂=[1e-3], R̂=[1e-6])
    @test ukf1.Q̂ ≈ [1e-3]
    @test ukf1.R̂ ≈ [1e-6]
    f(x,u,d,_) = linmodel.A*x + linmodel.Bu*u + linmodel.Bd*d
    h(x,d,_)   = linmodel.C*x + linmodel.Du*d
    nonlinmodel = NonLinModel(f, h, 10.0, 1, 1, 1)
    ukf2 = UnscentedKalmanFilter(nonlinmodel, nint_ym=0)
    setmodel!(ukf2, Q̂=[1e-3], R̂=[1e-6])
    @test ukf2.Q̂ ≈ [1e-3]
    @test ukf2.R̂ ≈ [1e-6]
    @test_throws ErrorException setmodel!(ukf2, deepcopy(nonlinmodel))
end

@testitem "ExtendedKalmanFilter construction" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel1 = LinModel(sys,Ts,i_d=[3])
    f(x,u,d,_) = linmodel1.A*x + linmodel1.Bu*u + linmodel1.Bd*d
    h(x,d,_)   = linmodel1.C*x + linmodel1.Du*d
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

    ekf6 = ExtendedKalmanFilter(nonlinmodel, σP_0=[1,2,3,4], σPint_ym_0=[5,6])
    @test ekf6.P̂_0 ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test ekf6.P̂  ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test ekf6.P̂_0 !== ekf6.P̂

    ekf7 = ExtendedKalmanFilter(nonlinmodel, nint_u=[1,1], nint_ym=[0,0])
    @test ekf7.nxs == 2
    @test ekf7.nx̂  == 6
    @test ekf7.nint_u  == [1, 1]
    @test ekf7.nint_ym == [0, 0]

    ekf8 = ExtendedKalmanFilter(nonlinmodel, 1:2, 0, [1, 1], I(6), I(6), I(2))
    @test ekf8.P̂_0 ≈ I(6)
    @test ekf8.Q̂ ≈ I(6)
    @test ekf8.R̂ ≈ I(2)

    linmodel2 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    ekf8 = ExtendedKalmanFilter(linmodel2)
    @test isa(ekf8, ExtendedKalmanFilter{Float32})
end

@testitem "ExtendedKalmanFilter estimator methods" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel1 = LinModel(sys,Ts,i_u=[1,2])
    f(x,u,_,_) = linmodel1.A*x + linmodel1.Bu*u
    h(x,_,_)   = linmodel1.C*x
    nonlinmodel = setop!(NonLinModel(f, h, Ts, 2, 2, 2, solver=nothing), uop=[10,50], yop=[50,30])
    ekf1 = ExtendedKalmanFilter(nonlinmodel)
    preparestate!(ekf1, [50, 30])
    @test updatestate!(ekf1, [10, 50], [50, 30]) ≈ zeros(4) atol=1e-9
    preparestate!(ekf1, [50, 30])
    @test updatestate!(ekf1, [10, 50], [50, 30], Float64[]) ≈ zeros(4) atol=1e-9
    @test ekf1.x̂0 ≈ zeros(4) atol=1e-9
    preparestate!(ekf1, [50, 30])
    @test evaloutput(ekf1) ≈ ekf1() ≈ [50, 30]
    @test evaloutput(ekf1, Float64[]) ≈ ekf1(Float64[]) ≈ [50, 30]
    @test initstate!(ekf1, [10, 50], [50, 30+1]) ≈ zeros(4);
    setstate!(ekf1, [1,2,3,4])
    @test ekf1.x̂0 ≈ [1,2,3,4]
    for i in 1:40
        preparestate!(ekf1, [50, 30])
        updatestate!(ekf1, [11, 52], [50, 30])
    end
    preparestate!(ekf1, [50, 30])
    @test ekf1() ≈ [50, 30] atol=1e-3
    for i in 1:40
        preparestate!(ekf1, [51, 32])
        updatestate!(ekf1, [10, 50], [51, 32])
    end
    preparestate!(ekf1, [51, 32])
    @test ekf1() ≈ [51, 32] atol=1e-3
    ekf2 = ExtendedKalmanFilter(linmodel1, nint_u=[1, 1], nint_ym=[0, 0], direct=false)
    for i in 1:40
        preparestate!(ekf2, [50, 30])
        updatestate!(ekf2, [11, 52], [50, 30])
    end
    @test ekf2() ≈ [50, 30] atol=1e-3
    for i in 1:40
        preparestate!(ekf2, [51, 32])
        updatestate!(ekf2, [10, 50], [51, 32])
    end
    @test ekf2() ≈ [51, 32] atol=1e-3
    linmodel3 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    ekf3 = ExtendedKalmanFilter(linmodel3)
    preparestate!(ekf3, [0])
    x̂ = updatestate!(ekf3, [0], [0])
    @test x̂ ≈ [0, 0]
    @test isa(x̂, Vector{Float32})
end

@testitem "ExtendedKalmanFilter set model" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel = LinModel(ss(0.5, 0.3, 1.0, 0, 10.0))
    linmodel = setop!(linmodel, uop=[2.0], yop=[50.0], xop=[3.0], fop=[3.0])
    ekf1 = ExtendedKalmanFilter(linmodel, nint_ym=0)
    @test ekf1.Â ≈ [0.5]
    preparestate!(ekf1, [50.0])
    @test evaloutput(ekf1) ≈ [50.0]
    preparestate!(ekf1, [50.0])
    x̂ = updatestate!(ekf1, [2.0], [50.0])
    @test x̂ ≈ [3.0]
    newlinmodel = LinModel(ss(0.2, 0.3, 1.0, 0, 10.0))
    newlinmodel = setop!(newlinmodel, uop=[3.0], yop=[55.0], xop=[3.0], fop=[3.0])
    setmodel!(ekf1, newlinmodel)
    @test ekf1.Â ≈ [0.2]
    preparestate!(ekf1, [55.0])
    @test evaloutput(ekf1) ≈ [55.0]
    @test ekf1.lastu0 ≈ [2.0 - 3.0]
    preparestate!(ekf1, [55.0])
    x̂ = updatestate!(ekf1, [3.0], [55.0])
    @test x̂ ≈ [3.0]
    newlinmodel = setop!(newlinmodel, uop=[3.0], yop=[55.0], xop=[8.0], fop=[8.0])
    setmodel!(ekf1, newlinmodel)
    @test ekf1.x̂0 ≈ [3.0 - 8.0]
    setmodel!(ekf1, Q̂=[1e-3], R̂=[1e-6])
    @test ekf1.Q̂ ≈ [1e-3]
    @test ekf1.R̂ ≈ [1e-6]
    f(x,u,d,_) = linmodel.A*x + linmodel.Bu*u + linmodel.Bd*d
    h(x,d,_)   = linmodel.C*x + linmodel.Du*d
    nonlinmodel = NonLinModel(f, h, 10.0, 1, 1, 1)
    ekf2 = ExtendedKalmanFilter(nonlinmodel, nint_ym=0)
    setmodel!(ekf2, Q̂=[1e-3], R̂=[1e-6])
    @test ekf2.Q̂ ≈ [1e-3]
    @test ekf2.R̂ ≈ [1e-6]
    @test_throws ErrorException setmodel!(ekf2, deepcopy(nonlinmodel))
end

@testitem "MovingHorizonEstimator construction" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra, JuMP, Ipopt
    linmodel1 = LinModel(sys,Ts,i_d=[3])
    f(x,u,d,_) = linmodel1.A*x + linmodel1.Bu*u + linmodel1.Bd*d
    h(x,d,_)   = linmodel1.C*x + linmodel1.Du*d
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

    mhe6 = MovingHorizonEstimator(nonlinmodel, He=5, σP_0=[1,2,3,4], σPint_ym_0=[5,6])
    @test mhe6.P̂_0       ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test mhe6.P̂arr_old ≈ Hermitian(diagm(Float64[1, 4, 9 ,16, 25, 36]))
    @test mhe6.P̂_0 !== mhe6.P̂arr_old

    mhe7 = MovingHorizonEstimator(nonlinmodel, He=10)
    @test mhe7.He == 10
    @test length(mhe7.X̂0)  == mhe7.He*6
    @test length(mhe7.Y0m) == mhe7.He*2
    @test length(mhe7.U0)  == mhe7.He*2
    @test length(mhe7.D0)  == (mhe7.He+mhe7.direct)*1
    @test length(mhe7.Ŵ)   == mhe7.He*6

    mhe8 = MovingHorizonEstimator(nonlinmodel, He=5, nint_u=[1, 1], nint_ym=[0, 0])
    @test mhe8.nxs == 2
    @test mhe8.nx̂  == 6
    @test mhe8.nint_u  == [1, 1]
    @test mhe8.nint_ym == [0, 0]

    I_6 = Matrix{Float64}(I, 6, 6)
    I_2 = Matrix{Float64}(I, 2, 2)
    optim = Model(Ipopt.Optimizer)
    mhe9 = MovingHorizonEstimator(nonlinmodel, 5, 1:2, 0, [1, 1], I_6, I_6, I_2, 1e5; optim)
    @test mhe9.P̂_0 ≈ I(6)
    @test mhe9.Q̂ ≈ I(6)
    @test mhe9.R̂ ≈ I(2)

    optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer, "nlp_scaling_max_gradient"=>1.0))
    covestim = ExtendedKalmanFilter(nonlinmodel, 1:2, 0, [1, 1], I_6, I_6, I_2)
    mhe10 = MovingHorizonEstimator(
        nonlinmodel, 5, 1:2, 0, [1, 1], I_6, I_6, I_2, Inf; optim, covestim
    )
    @test solver_name(mhe10.optim) == "Ipopt"

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

@testitem "MovingHorizonEstimator estimation and getinfo" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra, JuMP, Ipopt, ForwardDiff
    linmodel1 = LinModel(sys,Ts,i_u=[1,2], i_d=[3])
    linmodel1 = setop!(linmodel1, uop=[10,50], yop=[50,30], dop=[5])
    f(x,u,d,model) = model.A*x + model.Bu*u + model.Bd*d
    h(x,d,model)   = model.C*x + model.Dd*d
    nonlinmodel = NonLinModel(f, h, Ts, 2, 4, 2, 1, solver=nothing, p=linmodel1)
    nonlinmodel = setop!(nonlinmodel, uop=[10,50], yop=[50,30], dop=[5])
    
    mhe1 = MovingHorizonEstimator(nonlinmodel, He=2)
    preparestate!(mhe1, [50, 30], [5])
    x̂ = updatestate!(mhe1, [10, 50], [50, 30], [5])
    @test x̂ ≈ zeros(6) atol=1e-9
    @test mhe1.x̂0 ≈ zeros(6) atol=1e-9
    preparestate!(mhe1, [50, 30], [5])
    @test evaloutput(mhe1, [5]) ≈ mhe1([5]) ≈ [50, 30]
    info = getinfo(mhe1)
    @test info[:x̂] ≈ x̂ atol=1e-9
    @test info[:Ŷ][end-1:end] ≈ [50, 30] atol=1e-9

    @test initstate!(mhe1, [10, 50], [50, 30+1], [5]) ≈ zeros(6) atol=1e-9
    setstate!(mhe1, [1,2,3,4,5,6])
    @test mhe1.x̂0 ≈ [1,2,3,4,5,6]
    for i in 1:40
        preparestate!(mhe1, [50, 30], [5])
        updatestate!(mhe1, [11, 52], [50, 30], [5])
    end
    preparestate!(mhe1, [50, 30], [5])
    @test mhe1([5]) ≈ [50, 30] atol=1e-3
    for i in 1:40
        preparestate!(mhe1, [51, 32], [5])
        updatestate!(mhe1, [10, 50], [51, 32], [5])
    end
    preparestate!(mhe1, [51, 32], [5])
    @test mhe1([5]) ≈ [51, 32] atol=1e-3

    mhe1 = MovingHorizonEstimator(nonlinmodel, He=2, nint_u=[1, 1], nint_ym=[0, 0], direct=false)
    preparestate!(mhe1, [50, 30], [5])
    x̂ = updatestate!(mhe1, [10, 50], [50, 30], [5])
    @test x̂ ≈ zeros(6) atol=1e-9
    @test mhe1.x̂0 ≈ zeros(6) atol=1e-9
    @test evaloutput(mhe1, [5]) ≈ mhe1([5]) ≈ [50, 30]
    info = getinfo(mhe1)
    @test info[:x̂] ≈ x̂ atol=1e-9
    @test info[:Ŷ][end-1:end] ≈ [50, 30] atol=1e-9

    @test initstate!(mhe1, [10, 50], [50, 30+1], [5]) ≈ zeros(6) atol=1e-9
    setstate!(mhe1, [1,2,3,4,5,6])
    @test mhe1.x̂0 ≈ [1,2,3,4,5,6]
    for i in 1:40
        preparestate!(mhe1, [50, 30], [5])
        updatestate!(mhe1, [11, 52], [50, 30], [5])
    end
    @test mhe1([5]) ≈ [50, 30] atol=1e-3
    for i in 1:40
        preparestate!(mhe1, [51, 32], [5])
        updatestate!(mhe1, [10, 50], [51, 32], [5])
    end
    @test mhe1([5]) ≈ [51, 32] atol=1e-3

    mhe2 = MovingHorizonEstimator(linmodel1, He=2)
    preparestate!(mhe2, [50, 30], [5])
    x̂ = updatestate!(mhe2, [10, 50], [50, 30], [5])
    @test x̂ ≈ zeros(6) atol=1e-9
    @test mhe2.x̂0 ≈ zeros(6) atol=1e-9
    preparestate!(mhe2, [50, 30], [5])
    @test evaloutput(mhe2, [5]) ≈ mhe2([5]) ≈ [50, 30]
    info = getinfo(mhe2)
    @test info[:x̂] ≈ x̂ atol=1e-9
    @test info[:Ŷ][end-1:end] ≈ [50, 30] atol=1e-9
    for i in 1:40
        preparestate!(mhe2, [50, 30], [5])
        updatestate!(mhe2, [11, 52], [50, 30], [5])
    end
    preparestate!(mhe2, [50, 30], [5])
    @test mhe2([5]) ≈ [50, 30] atol=1e-3
    for i in 1:40
        preparestate!(mhe2, [51, 32], [5])
        updatestate!(mhe2, [10, 50], [51, 32], [5])
    end
    preparestate!(mhe2, [51, 32], [5])
    @test mhe2([5]) ≈ [51, 32] atol=1e-3

    mhe2 = MovingHorizonEstimator(linmodel1, He=2, nint_u=[1, 1], nint_ym=[0, 0], direct=false)
    preparestate!(mhe2, [50, 30], [5])
    x̂ = updatestate!(mhe2, [10, 50], [50, 30], [5])
    @test x̂ ≈ zeros(6) atol=1e-9
    @test mhe2.x̂0 ≈ zeros(6) atol=1e-9
    @test evaloutput(mhe2, [5]) ≈ mhe2([5]) ≈ [50, 30]
    info = getinfo(mhe2)
    @test info[:x̂] ≈ x̂ atol=1e-9
    @test info[:Ŷ][end-1:end] ≈ [50, 30] atol=1e-9
    for i in 1:40
        preparestate!(mhe2, [50, 30], [5])
        updatestate!(mhe2, [11, 52], [50, 30], [5])
    end
    @test mhe2([5]) ≈ [50, 30] atol=1e-2
    for i in 1:40
        preparestate!(mhe2, [51, 32], [5])
        updatestate!(mhe2, [10, 50], [51, 32], [5])
    end
    @test mhe2([5]) ≈ [51, 32] atol=1e-2
    linmodel3 = LinModel{Float32}(0.5*ones(1,1), ones(1,1), ones(1,1), zeros(1,0), zeros(1,0), 1.0)
    mhe3 = MovingHorizonEstimator(linmodel3, He=1)
    preparestate!(mhe3, [0])
    x̂ = updatestate!(mhe3, [0], [0])
    @test x̂ ≈ [0, 0] atol=1e-3
    @test isa(x̂, Vector{Float32})
    mhe4 = setconstraint!(MovingHorizonEstimator(nonlinmodel, He=1, nint_ym=0), v̂max=[50,50])
    g_V̂max_end = mhe4.optim[:g_V̂max_2].func
    # execute update_predictions! branch in `gfunc_i` for coverage:
    @test_nowarn g_V̂max_end(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) ≤ 0.0 

    Q̂ = diagm([1/4, 1/4, 1/4, 1/4].^2) 
    R̂ = diagm([1, 1].^2)
    optim = Model(Ipopt.Optimizer)
    covestim = ExtendedKalmanFilter(nonlinmodel, 1:2, 0, 0, Q̂, Q̂, R̂)
    mhe5 = MovingHorizonEstimator(nonlinmodel, 1, 1:2, 0, 0, Q̂, Q̂, R̂, Inf; optim, covestim)
    preparestate!(mhe5, [50, 30], [5])
    x̂ = updatestate!(mhe5, [10, 50], [50, 30], [5])
    @test x̂ ≈ zeros(4) atol=1e-9
    @test mhe5.x̂0 ≈ zeros(4) atol=1e-9
    preparestate!(mhe5, [50, 30], [5])
    @test evaloutput(mhe5, [5]) ≈ mhe5([5]) ≈ [50, 30]
    info = getinfo(mhe5)
    @test info[:x̂] ≈ x̂ atol=1e-9
    @test info[:Ŷ][end-1:end] ≈ [50, 30] atol=1e-9
    # for coverage of NLP functions, the univariate syntax of JuMP.@operator
    mhe6 = MovingHorizonEstimator(nonlinmodel, He=1, Cwt=Inf)
    setconstraint!(mhe6, v̂min=[-51,-52], v̂max=[53,54])
    x̂ = preparestate!(mhe6, [50, 30], [5])
    @test x̂ ≈ zeros(6) atol=1e-9
    @test_nowarn ModelPredictiveControl.info2debugstr(info)
end

@testitem "MovingHorizonEstimator fallbacks for arrival covariance estimation" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel = setop!(LinModel(sys,Ts,i_u=[1,2], i_d=[3]), uop=[10,50], yop=[50,30], dop=[5])
    f(x,u,d,_) = linmodel.A*x + linmodel.Bu*u + linmodel.Bd*d
    h(x,d,_)   = linmodel.C*x + linmodel.Dd*d
    nonlinmodel = setop!(NonLinModel(f, h, Ts, 2, 4, 2, 1, solver=nothing), uop=[10,50], yop=[50,30], dop=[5])
    mhe = MovingHorizonEstimator(nonlinmodel, nint_ym=0, He=1)
    preparestate!(mhe, [50, 30], [5])
    updatestate!(mhe, [10, 50], [50, 30], [5])
    mhe.P̂arr_old[1, 1] = -1e-3 # negative eigenvalue to trigger fallback
    P̂arr_old_copy = deepcopy(mhe.P̂arr_old)
    invP̄_copy = deepcopy(mhe.invP̄)
    @test_logs(
        (:error, "Arrival covariance P̄ is not positive definite: keeping the old one"), 
        preparestate!(mhe, [50, 30], [5])
    )
    @test mhe.P̂arr_old ≈ P̂arr_old_copy
    @test mhe.invP̄ ≈ invP̄_copy
    @test_logs(
        (:error, "Arrival covariance P̄ is not positive definite: keeping the old one"), 
        updatestate!(mhe, [10, 50], [50, 30], [5])
    )
    @test mhe.P̂arr_old ≈ P̂arr_old_copy
    @test mhe.invP̄ ≈ invP̄_copy
    @test_logs(
        (:error, "Arrival covariance P̄ is not invertible: keeping the old one"), 
        ModelPredictiveControl.invert_cov!(mhe, Hermitian(zeros(mhe.nx̂, mhe.nx̂),:L))
    )
    mhe.P̂arr_old[1, 1] = Inf # Inf to trigger fallback
    P̂arr_old_copy = deepcopy(mhe.P̂arr_old)
    invP̄_copy = deepcopy(mhe.invP̄)
    @test_logs(
        (:error, "Arrival covariance P̄ is not finite: keeping the old one"), 
        preparestate!(mhe, [50, 30], [5])
    )
    @test mhe.P̂arr_old ≈ P̂arr_old_copy
    @test mhe.invP̄ ≈ invP̄_copy
    @test_logs(
        (:error, "Arrival covariance P̄ is not finite: keeping the old one"), 
        updatestate!(mhe, [10, 50], [50, 30], [5])   
    )
    @test mhe.P̂arr_old ≈ P̂arr_old_copy
    @test mhe.invP̄ ≈ invP̄_copy
end

@testitem "MovingHorizonEstimator set constraints" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    mhe1 = MovingHorizonEstimator(linmodel1, He=1, nint_ym=0, Cwt=1e3)
    setconstraint!(mhe1, x̂min=[-51,-52], x̂max=[53,54])
    @test all((mhe1.con.X̂0min, mhe1.con.X̂0max) .≈ ([-51,-52], [53,54]))
    @test all((mhe1.con.x̃0min[2:end], mhe1.con.x̃0max[2:end]) .≈ ([-51,-52], [53,54]))
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
    @test all((mhe2.con.X̂0min, mhe2.con.X̂0max) .≈ (-1(3:10), 1(3:10)))
    @test all((mhe2.con.x̃0min[2:end], mhe2.con.x̃0max[2:end]) .≈ (-1(1:2),  1(1:2)))
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

    f(x,u,d,_) = linmodel1.A*x + linmodel1.Bu*u
    h(x,d,_)   = linmodel1.C*x 
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

    preparestate!(mhe1, [50, 30])
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

@testitem "MovingHorizonEstimator constraint violation" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel1 = setop!(LinModel(sys,Ts,i_u=[1,2]), uop=[10,50], yop=[50,30])
    mhe = MovingHorizonEstimator(linmodel1, He=1, nint_ym=0)

    setconstraint!(mhe, x̂min=[-100,-100], x̂max=[100,100])
    setconstraint!(mhe, ŵmin=[-100,-100], ŵmax=[100,100])
    setconstraint!(mhe, v̂min=[-100,-100], v̂max=[100,100])

    setconstraint!(mhe, x̂min=[1,1], x̂max=[100,100])
    preparestate!(mhe, [50, 30])
    x̂ = updatestate!(mhe, [10, 50], [50, 30])
    @test x̂ ≈ [1, 1] atol=5e-2

    setconstraint!(mhe, x̂min=[-100,-100], x̂max=[-1,-1])
    preparestate!(mhe, [50, 30])
    x̂ = updatestate!(mhe, [10, 50], [50, 30])
    @test x̂ ≈ [-1, -1] atol=5e-2

    setconstraint!(mhe, x̂min=[-100,-100], x̂max=[100,100])
    setconstraint!(mhe, ŵmin=[-100,-100], ŵmax=[100,100])
    setconstraint!(mhe, v̂min=[-100,-100], v̂max=[100,100])

    setconstraint!(mhe, ŵmin=[1,1], ŵmax=[100,100])
    preparestate!(mhe, [50, 30])
    x̂ = updatestate!(mhe, [10, 50], [50, 30])
    @test mhe.Ŵ ≈ [1,1] atol=5e-2

    setconstraint!(mhe, ŵmin=[-100,-100], ŵmax=[-1,-1])
    preparestate!(mhe, [50, 30])
    x̂ = updatestate!(mhe, [10, 50], [50, 30])
    @test mhe.Ŵ ≈ [-1,-1] atol=5e-2

    setconstraint!(mhe, x̂min=[-100,-100], x̂max=[100,100])
    setconstraint!(mhe, ŵmin=[-100,-100], ŵmax=[100,100])
    setconstraint!(mhe, v̂min=[-100,-100], v̂max=[100,100])

    setconstraint!(mhe, v̂min=[1,1], v̂max=[100,100])
    preparestate!(mhe, [50, 30])
    x̂ = updatestate!(mhe, [10, 50], [50, 30])
    info = getinfo(mhe)
    @test info[:V̂] ≈ [1,1] atol=5e-2

    setconstraint!(mhe, v̂min=[-100,-100], v̂max=[-1,-1])
    preparestate!(mhe, [50, 30])
    x̂ = updatestate!(mhe, [10, 50], [50, 30])
    info = getinfo(mhe)
    @test info[:V̂] ≈ [-1,-1] atol=5e-2

    f(x,u,_,_) = linmodel1.A*x + linmodel1.Bu*u
    h(x,_,_)   = linmodel1.C*x
    nonlinmodel = setop!(NonLinModel(f, h, Ts, 2, 2, 2, solver=nothing), uop=[10,50], yop=[50,30])
    mhe2 = MovingHorizonEstimator(nonlinmodel, He=1, nint_ym=0)

    setconstraint!(mhe2, x̂min=[-100,-100], x̂max=[100,100])
    setconstraint!(mhe2, ŵmin=[-100,-100], ŵmax=[100,100])
    setconstraint!(mhe2, v̂min=[-100,-100], v̂max=[100,100])

    setconstraint!(mhe2, x̂min=[1,1], x̂max=[100,100])
    preparestate!(mhe2, [50, 30])
    x̂ = updatestate!(mhe2, [10, 50], [50, 30])
    @test x̂ ≈ [1, 1] atol=5e-2

    setconstraint!(mhe2, x̂min=[-100,-100], x̂max=[-1,-1])
    preparestate!(mhe2, [50, 30])
    x̂ = updatestate!(mhe2, [10, 50], [50, 30])
    @test x̂ ≈ [-1, -1] atol=5e-2

    setconstraint!(mhe2, x̂min=[-100,-100], x̂max=[100,100])
    setconstraint!(mhe2, ŵmin=[-100,-100], ŵmax=[100,100])
    setconstraint!(mhe2, v̂min=[-100,-100], v̂max=[100,100])

    setconstraint!(mhe2, ŵmin=[1,1], ŵmax=[100,100])
    preparestate!(mhe2, [50, 30])
    x̂ = updatestate!(mhe2, [10, 50], [50, 30])
    @test mhe2.Ŵ ≈ [1,1] atol=5e-2

    setconstraint!(mhe2, ŵmin=[-100,-100], ŵmax=[-1,-1])
    preparestate!(mhe2, [50, 30])
    x̂ = updatestate!(mhe2, [10, 50], [50, 30])
    @test mhe2.Ŵ ≈ [-1,-1] atol=5e-2

    setconstraint!(mhe2, x̂min=[-100,-100], x̂max=[100,100])
    setconstraint!(mhe2, ŵmin=[-100,-100], ŵmax=[100,100])
    setconstraint!(mhe2, v̂min=[-100,-100], v̂max=[100,100])

    setconstraint!(mhe2, v̂min=[1,1], v̂max=[100,100])
    preparestate!(mhe2, [50, 30])
    x̂ = updatestate!(mhe2, [10, 50], [50, 30])
    info = getinfo(mhe2)
    @test info[:V̂] ≈ [1,1] atol=5e-2

    setconstraint!(mhe2, v̂min=[-100,-100], v̂max=[-1,-1])
    preparestate!(mhe2, [50, 30])
    x̂ = updatestate!(mhe2, [10, 50], [50, 30])
    info = getinfo(mhe2)
    @test info[:V̂] ≈ [-1,-1] atol=5e-2
end

@testitem "MovingHorizonEstimator set model" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel = LinModel(ss(0.5, 0.3, 1.0, 0, 10.0))
    linmodel = setop!(linmodel, uop=[2.0], yop=[50.0], xop=[3.0], fop=[3.0])
    mhe = MovingHorizonEstimator(linmodel, He=1, nint_ym=0, direct=false)
    setconstraint!(mhe, x̂min=[-1000], x̂max=[1000])
    @test mhe.Â ≈ [0.5]
    @test evaloutput(mhe) ≈ [50.0]
    preparestate!(mhe, [50.0])
    x̂ = updatestate!(mhe, [2.0], [50.0])
    @test x̂ ≈ [3.0]
    newlinmodel = LinModel(ss(0.2, 0.3, 1.0, 0, 10.0))
    newlinmodel = setop!(newlinmodel, uop=[3.0], yop=[55.0], xop=[3.0], fop=[3.0])
    setmodel!(mhe, newlinmodel)
    @test mhe.Â ≈ [0.2]
    @test evaloutput(mhe) ≈ [55.0]
    @test mhe.lastu0 ≈ [2.0 - 3.0]
    @test mhe.U0 ≈ [2.0 - 3.0]
    @test mhe.Y0m ≈ [50.0 - 55.0]
    preparestate!(mhe, [55.0])
    x̂ = updatestate!(mhe, [3.0], [55.0])
    @test x̂ ≈ [3.0]
    newlinmodel = setop!(newlinmodel, uop=[3.0], yop=[55.0], xop=[8.0], fop=[8.0])
    setmodel!(mhe, newlinmodel)
    @test mhe.x̂0   ≈ [3.0 - 8.0]
    @test mhe.Z̃[1] ≈ 3.0 - 8.0
    @test mhe.X̂0   ≈ [3.0 - 8.0]
    @test mhe.x̂0arr_old ≈ [3.0 - 8.0]
    @test mhe.con.X̂0min ≈ [-1000 - 8.0]
    @test mhe.con.X̂0max ≈ [+1000 - 8.0]
    @test mhe.con.x̃0min ≈ [-1000 - 8.0]
    @test mhe.con.x̃0max ≈ [+1000 - 8.0]
    setmodel!(mhe, Q̂=[1e-3], R̂=[1e-6])
    @test mhe.Q̂ ≈ [1e-3]
    @test mhe.R̂ ≈ [1e-6]
    f(x,u,d,_) = linmodel.A*x + linmodel.Bu*u + linmodel.Bd*d
    h(x,d,_)   = linmodel.C*x + linmodel.Du*d
    nonlinmodel = NonLinModel(f, h, 10.0, 1, 1, 1)
    mhe2 = MovingHorizonEstimator(nonlinmodel, He=1, nint_ym=0)
    setmodel!(mhe2, Q̂=[1e-3], R̂=[1e-6])
    @test mhe2.Q̂ ≈ [1e-3]
    @test mhe2.R̂ ≈ [1e-6]
    @test_throws ErrorException setmodel!(mhe2, deepcopy(nonlinmodel))
end

@testitem "MovingHorizonEstimator v.s. Kalman filters" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel1 = setop!(LinModel(sys,Ts,i_d=[3]), uop=[10,50], yop=[50,30], dop=[20])
    mhe = MovingHorizonEstimator(linmodel1, He=3, nint_ym=0, direct=false)
    kf  = KalmanFilter(linmodel1, nint_ym=0, direct=false)
    X̂_mhe = zeros(4, 6)
    X̂_kf  = zeros(4, 6)
    for i in 1:6
        y = [50,31] + randn(2)
        x̂_mhe = preparestate!(mhe, y, [25])
        x̂_kf  = preparestate!(kf,  y, [25])
        X̂_mhe[:,i] = x̂_mhe
        X̂_kf[:,i]  = x̂_kf
        updatestate!(mhe, [11, 50], y, [25])
        updatestate!(kf,  [11, 50], y, [25])
    end
    @test X̂_mhe ≈ X̂_kf atol=1e-3 rtol=1e-3
    mhe = MovingHorizonEstimator(linmodel1, He=3, nint_ym=0, direct=true)
    kf  = KalmanFilter(linmodel1, nint_ym=0, direct=true)
    X̂_mhe = zeros(4, 6)
    X̂_kf  = zeros(4, 6)
    for i in 1:6
        y = [50,31] + randn(2)
        x̂_mhe = preparestate!(mhe, y, [25])
        x̂_kf  = preparestate!(kf,  y, [25])
        X̂_mhe[:,i] = x̂_mhe
        X̂_kf[:,i]  = x̂_kf
        updatestate!(mhe, [11, 50], y, [25])
        updatestate!(kf,  [11, 50], y, [25])
    end
    @test X̂_mhe ≈ X̂_kf atol=1e-3 rtol=1e-3
    f = (x,u,d,_) -> linmodel1.A*x + linmodel1.Bu*u + linmodel1.Bd*d
    h = (x,d,_)   -> linmodel1.C*x + linmodel1.Dd*d
    nonlinmodel = NonLinModel(f, h, Ts, 2, 4, 2, 1, solver=nothing)
    nonlinmodel = setop!(nonlinmodel, uop=[10,50], yop=[50,30], dop=[20])
    mhe = MovingHorizonEstimator(nonlinmodel, He=5, nint_ym=0, direct=false)
    ukf = UnscentedKalmanFilter(nonlinmodel, nint_ym=0, direct=false)
    ekf = ExtendedKalmanFilter(nonlinmodel, nint_ym=0, direct=false)
    X̂_mhe = zeros(4, 6)
    X̂_ukf = zeros(4, 6)
    X̂_ekf = zeros(4, 6)
    for i in 1:6
        y = [50,31] + randn(2)
        x̂_mhe = preparestate!(mhe, y, [25])
        x̂_ukf = preparestate!(ukf,  y, [25])
        x̂_ekf = preparestate!(ekf,  y, [25])
        X̂_mhe[:,i] = x̂_mhe
        X̂_ukf[:,i] = x̂_ukf
        X̂_ekf[:,i] = x̂_ekf
        updatestate!(mhe, [11, 50], y, [25])
        updatestate!(ukf, [11, 50], y, [25])
        updatestate!(ekf, [11, 50], y, [25])
    end
    @test X̂_mhe ≈ X̂_ukf atol=1e-3 rtol=1e-3
    @test X̂_mhe ≈ X̂_ekf atol=1e-3 rtol=1e-3
    mhe = MovingHorizonEstimator(nonlinmodel, He=5, nint_ym=0, direct=true)
    ukf = UnscentedKalmanFilter(nonlinmodel, nint_ym=0, direct=true)
    ekf = ExtendedKalmanFilter(nonlinmodel, nint_ym=0, direct=true)
    X̂_mhe = zeros(4, 6)
    X̂_ukf = zeros(4, 6)
    X̂_ekf = zeros(4, 6)
    for i in 1:6
        y = [50,31] + randn(2)
        x̂_mhe = preparestate!(mhe, y, [25])
        x̂_ukf = preparestate!(ukf,  y, [25])
        x̂_ekf = preparestate!(ekf,  y, [25])
        X̂_mhe[:,i] = x̂_mhe
        X̂_ukf[:,i]  = x̂_ukf
        X̂_ekf[:,i]  = x̂_ekf
        updatestate!(mhe, [11, 50], y, [25])
        updatestate!(ukf, [11, 50], y, [25])
        updatestate!(ekf, [11, 50], y, [25])
    end
    @test X̂_mhe ≈ X̂_ukf atol=1e-3 rtol=1e-3
    @test X̂_mhe ≈ X̂_ekf atol=1e-3 rtol=1e-3 
end

@testitem "MovingHorizonEstimator LinModel v.s. NonLinModel" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra, JuMP, Ipopt
    linmodel = setop!(LinModel(sys,Ts,i_d=[3]), uop=[10,50], yop=[50,30], dop=[20])
    f = (x,u,d,_) -> linmodel.A*x + linmodel.Bu*u + linmodel.Bd*d
    h = (x,d,_)   -> linmodel.C*x + linmodel.Dd*d
    nonlinmodel = NonLinModel(f, h, Ts, 2, 4, 2, 1, solver=nothing)
    nonlinmodel = setop!(nonlinmodel, uop=[10,50], yop=[50,30], dop=[20])
    optim = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer, "sb" => "yes"))
    mhe_lin = MovingHorizonEstimator(linmodel, He=5, nint_ym=0, direct=true, optim=optim)
    mhe_lin = setconstraint!(
        mhe_lin, x̂min=[-100, -100, -100, -100], ŵmin=[10,10,10,10]
    )
    mhe_nonlin = MovingHorizonEstimator(nonlinmodel, He=5, nint_ym=0, direct=true)
    mhe_nonlin = setconstraint!(
        mhe_nonlin, x̂min=[-100, -100, -100, -100], ŵmin=[10,10,10,10]
    )
    X̂_lin = zeros(4, 6)
    X̂_nonlin = zeros(4, 6)
    for i in 1:6
        y = [50,31] + randn(2)
        x̂_lin = preparestate!(mhe_lin, y, [25])
        x̂_nonlin = preparestate!(mhe_nonlin, y, [25])
        X̂_lin[:,i] = x̂_lin
        X̂_nonlin[:,i] = x̂_nonlin
        updatestate!(mhe_lin, [11, 50], y, [25])
        updatestate!(mhe_nonlin, [11, 50], y, [25])
    end
    @test X̂_lin ≈ X̂_nonlin atol=1e-3 rtol=1e-3
    mhe2_lin = MovingHorizonEstimator(linmodel, He=5, nint_ym=0, direct=false, optim=optim)
    mhe2_lin = setconstraint!(
        mhe2_lin, x̂min=[-100, -100, -100, -100], ŵmin=[10,10,10,10]
    )
    mhe2_nonlin = MovingHorizonEstimator(nonlinmodel, He=5, nint_ym=0, direct=false)
    mhe2_nonlin = setconstraint!(
        mhe2_nonlin, x̂min=[-100, -100, -100, -100], ŵmin=[10,10,10,10]
    )
    X̂_lin = zeros(4, 6)
    X̂_nonlin = zeros(4, 6)
    for i in 1:6
        y = [50,31] + randn(2)
        x̂_lin = preparestate!(mhe2_lin, y, [25])
        x̂_nonlin = preparestate!(mhe2_nonlin, y, [25])
        X̂_lin[:,i] = x̂_lin
        X̂_nonlin[:,i] = x̂_nonlin
        updatestate!(mhe2_lin, [11, 50], y, [25])
        updatestate!(mhe2_nonlin, [11, 50], y, [25])
    end
    @test X̂_lin ≈ X̂_nonlin atol=1e-3 rtol=1e-3
end
