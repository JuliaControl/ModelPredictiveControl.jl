@testitem "LinModel construction" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
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
    @test linmodel3.Ts == 200.0
    @test linmodel3.A ≈ Gss2.A
    @test linmodel3.C ≈ Gss2.C

    linmodel4 = LinModel(Gss)
    setstate!(linmodel4, [1;-1])
    @test linmodel4.x0 ≈ [1;-1]

    linmodel5 = LinModel(sys,Ts,i_d=[3])
    setop!(linmodel5, uop=[10,50], yop=[50,30], dop=[20])
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
    @test linmodel5.uop ≈ [10,50]
    @test linmodel5.yop ≈ [50,30]
    @test linmodel5.dop ≈ [20]

    linmodel6 = LinModel([delay(Ts) delay(Ts)]*sys,Ts,i_d=[3])
    @test linmodel6.nx == 3
    @test sum(eigvals(linmodel6.A) .≈ 0) == 1

    linmodel7 = LinModel(
        ss(diagm( .1: .1: .3), I(3), diagm( .4: .1: .6), 0, 1.0), 
        i_u=[1, 2],
        i_d=[3])
    @test linmodel7.A ≈ diagm( .1: .1: .3)
    @test linmodel7.C ≈ diagm( .4: .1: .6)

    linmodel8 = LinModel(Gss.A, Gss.B, Gss.C, zeros(Float32, 2, 0), zeros(Float32, 2, 0), Ts)
    @test isa(linmodel8, LinModel{Float64})

    linmodel10 = LinModel(Gss.A, Gss.B, Gss.C, 0, 0.0, Ts)
    @test isa(linmodel10, LinModel{Float64})
    @test linmodel10.nd == 0

    linmodel11 = LinModel(Gss.A, Gss.B, I, 0, 0, Ts)
    @test linmodel11.ny == linmodel11.nx

    linmodel12 = LinModel{Float32}(Gss.A, Gss.B, Gss.C, zeros(2, 0), zeros(2, 0), Ts)
    @test isa(linmodel12, LinModel{Float32})

    linmodel13 = LinModel(sys,Ts,i_d=[3])
    linmodel13 = setname!(linmodel13, 
        u=["u_c", "u_h"], 
        y=["y_L", "y_T"], 
        d=["u_l"],
        x=["X_1", "X_2", "X_3", "X_4"]
    )
    @test all(linmodel13.uname .== ["u_c", "u_h"])
    @test all(linmodel13.yname .== ["y_L", "y_T"])
    @test all(linmodel13.dname .== ["u_l"])
    @test all(linmodel13.xname .== ["X_1", "X_2", "X_3", "X_4"])

    @test_throws ErrorException LinModel(sys)
    @test_throws ErrorException LinModel(sys,-Ts)
    @test_throws ErrorException LinModel(sys,Ts,i_u=[1,1])
    @test_throws ErrorException LinModel(sys,Ts,i_d=[3,3])
    @test_throws ErrorException LinModel(sys_ss,Ts+1)
    @test_throws ErrorException setop!(linmodel5, uop=[0,0,0,0,0])
    @test_throws ErrorException setop!(linmodel5, yop=[0,0,0,0,0])
    @test_throws ErrorException setop!(linmodel5, dop=[0,0,0,0,0])
    sys_ss.D .= 1
    @test_throws ErrorException LinModel(sys_ss,Ts)
end

@testitem "LinModel sim methods" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase
    linmodel1 = setop!(LinModel(Gss), uop=[10,50], yop=[50,30])
    @test updatestate!(linmodel1, [10, 50]) ≈ zeros(2)
    @test updatestate!(linmodel1, [10, 50], Float64[]) ≈ zeros(2)
    @test linmodel1.x0 ≈ zeros(2)
    @test evaloutput(linmodel1) ≈ linmodel1() ≈ [50,30]
    @test evaloutput(linmodel1, Float64[]) ≈ linmodel1(Float64[]) ≈ [50,30]
    x = initstate!(linmodel1, [10, 60])
    @test evaloutput(linmodel1) ≈ [50 + 19.0, 30 + 7.4]
    @test preparestate!(linmodel1, [10, 60]) ≈ x
    @test updatestate!(linmodel1,  [10, 60]) ≈ x
    linmodel2 = LinModel(append(tf(1, [1, 0]), tf(2, [10, 1])), 1.0)
    x = initstate!(linmodel2, [10, 3])
    @test evaloutput(linmodel2) ≈ [0, 6]
    @test updatestate!(linmodel2, [0, 3]) ≈ x

    @test_throws DimensionMismatch updatestate!(linmodel1, zeros(2), zeros(1))
    @test_throws DimensionMismatch evaloutput(linmodel1, zeros(1))
end

@testitem "LinModel real time simulations" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel1 = LinModel(tf(2, [10, 1]), 0.1)
    times1 = zeros(5)
    for i=1:5
        times1[i] = savetime!(linmodel1)
        updatestate!(linmodel1, [1])
        periodsleep(linmodel1)
    end
    @test all(isapprox.(diff(times1[2:end]), 0.1, atol=0.01))
    linmodel2 = LinModel(tf(2, [0.1, 1]), 0.001)
    times2 = zeros(5)
    for i=1:5
        times2[i] = savetime!(linmodel2)
        updatestate!(linmodel2, [1])
        periodsleep(linmodel2, true)
    end
    @test all(isapprox.(diff(times2[2:end]), 0.001, atol=0.0001))
end

@testitem "NonLinModel construction" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel1 = LinModel(sys,Ts,i_u=[1,2])
    f1(x,u,_,model) = model.A*x + model.Bu*u
    h1(x,_,model)   = model.C*x
    nonlinmodel1 = NonLinModel(f1,h1,Ts,2,2,2,solver=nothing,p=linmodel1)
    @test nonlinmodel1.nx == 2
    @test nonlinmodel1.nu == 2
    @test nonlinmodel1.nd == 0
    @test nonlinmodel1.ny == 2
    xnext, y = similar(nonlinmodel1.x0), similar(nonlinmodel1.yop)
    nonlinmodel1.f!(xnext,[0,0],[0,0],[1],nonlinmodel1.p)
    @test xnext ≈ zeros(2,)
    nonlinmodel1.h!(y,[0,0],[1],nonlinmodel1.p)
    @test y ≈ zeros(2,)

    linmodel2 = LinModel(sys,Ts,i_d=[3])
    f2(x,u,d,_) = linmodel2.A*x + linmodel2.Bu*u + linmodel2.Bd*d
    h2(x,d,_)   = linmodel2.C*x + linmodel2.Dd*d
    nonlinmodel2 = NonLinModel(f2,h2,Ts,2,4,2,1,solver=nothing)

    @test nonlinmodel2.nx == 4
    @test nonlinmodel2.nu == 2
    @test nonlinmodel2.nd == 1
    @test nonlinmodel2.ny == 2
    xnext, y = similar(nonlinmodel2.x0), similar(nonlinmodel2.yop)
    nonlinmodel2.f!(xnext,[0,0,0,0],[0,0],[0],nonlinmodel2.p)
    @test xnext ≈ zeros(4,)
    nonlinmodel2.h!(y,[0,0,0,0],[0],nonlinmodel2.p)
    @test y ≈ zeros(2,)

    nonlinmodel3 = NonLinModel{Float32}(f2,h2,Ts,2,4,2,1,solver=nothing)
    @test isa(nonlinmodel3, NonLinModel{Float32})

    function f1!(xnext, x, u, d,_)
        mul!(xnext, linmodel2.A,  x)
        mul!(xnext, linmodel2.Bu, u, 1, 1)
        mul!(xnext, linmodel2.Bd, d, 1, 1)
        return nothing
    end 
    function h1!(y, x, d,_)
        mul!(y, linmodel2.C,  x)
        mul!(y, linmodel2.Dd, d, 1, 1)
        return nothing
    end
    nonlinmodel4 = NonLinModel(f1!, h1!, Ts, 2, 4, 2, 1, solver=nothing)
    xnext, y = similar(nonlinmodel4.x0), similar(nonlinmodel4.yop)
    nonlinmodel4.f!(xnext,[0,0,0,0],[0,0],[0],nonlinmodel4.p)
    @test xnext ≈ zeros(4)
    nonlinmodel4.h!(y,[0,0,0,0],[0],nonlinmodel4.p)
    @test y ≈ zeros(2)

    A  = [0 0.5; -0.2 -0.1]
    Bu = reshape([0; 0.5], 2, 1)
    Bd = reshape([0; 0.5], 2, 1)
    C  = [0.4 0]
    Dd = reshape([0], 1, 1)
    f3(x, u, d, _) = A*x + Bu*u+ Bd*d
    h3(x, d, _) = C*x + Dd*d
    solver=RungeKutta(4)
    @test string(solver) == 
        "4th order Runge-Kutta differential equation solver with 1 supersamples."
    nonlinmodel5 = NonLinModel(f3, h3, 1.0, 1, 2, 1, 1, solver=solver)
    xnext, y = similar(nonlinmodel5.x0), similar(nonlinmodel5.yop)
    nonlinmodel5.f!(xnext, [0; 0], [0], [0], nonlinmodel5.p)
    @test xnext ≈ zeros(2)
    nonlinmodel5.h!(y, [0; 0], [0], nonlinmodel5.p)
    @test y ≈ zeros(1)

    function f2!(ẋ, x, u , d, _)
        mul!(ẋ, A, x)
        mul!(ẋ, Bu, u, 1, 1)
        mul!(ẋ, Bd, d, 1, 1)
        return nothing
    end
    function h2!(y, x, d, _)
        mul!(y, C, x)
        mul!(y, Dd, d, 1, 1)
        return nothing
    end
    nonlinmodel6 = NonLinModel(f2!, h2!, 1.0, 1, 2, 1, 1, solver=RungeKutta())
    xnext, y = similar(nonlinmodel6.x0), similar(nonlinmodel6.yop)
    nonlinmodel6.f!(xnext, [0; 0], [0], [0], nonlinmodel6.p)
    @test xnext ≈ zeros(2)
    nonlinmodel6.h!(y, [0; 0], [0], nonlinmodel6.p)
    @test y ≈ zeros(1)
    nonlinemodel7 = NonLinModel(f2!, h2!, 1.0, 1, 2, 1, 1, solver=ForwardEuler())
    xnext, y = similar(nonlinemodel7.x0), similar(nonlinemodel7.yop)
    nonlinemodel7.f!(xnext, [0; 0], [0], [0], nonlinemodel7.p)
    @test xnext ≈ zeros(2)
    nonlinemodel7.h!(y, [0; 0], [0], nonlinemodel7.p)
    @test y ≈ zeros(1)

    
    @test_throws ErrorException NonLinModel(
        (x,u)->linmodel1.A*x + linmodel1.Bu*u,
        (x,_,_)->linmodel1.C*x, Ts, 2, 4, 2, 1, solver=nothing)
    @test_throws ErrorException NonLinModel(
        (x,u,_)->linmodel1.A*x + linmodel1.Bu*u,
        (x,_,_)->linmodel1.C*x, Ts, 2, 4, 2, 1, solver=nothing)
    @test_throws ErrorException NonLinModel(
        (x,u,_,_)->linmodel1.A*x + linmodel1.Bu*u,
        (x)->linmodel1.C*x, Ts, 2, 4, 2, 1, solver=nothing)
    @test_throws ErrorException NonLinModel(
        (x,u,_,_)->linmodel1.A*x + linmodel1.Bu*u,
        (x,_)->linmodel1.C*x, Ts, 2, 4, 2, 1, solver=nothing)
end

@testitem "NonLinModel sim methods" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel1 = LinModel(sys,Ts,i_u=[1,2])
    f1(x,u,_,model) = model.A*x + model.Bu*u
    h1(x,_,model)   = model.C*x
    nonlinmodel = NonLinModel(f1,h1,Ts,2,2,2,p=linmodel1,solver=nothing)

    @test updatestate!(nonlinmodel, zeros(2,)) ≈ zeros(2) 
    @test updatestate!(nonlinmodel, zeros(2,), Float64[]) ≈ zeros(2)
    @test nonlinmodel.x0 ≈ zeros(2)
    @test evaloutput(nonlinmodel) ≈ nonlinmodel() ≈ zeros(2)
    @test evaloutput(nonlinmodel, Float64[]) ≈ nonlinmodel(Float64[]) ≈ zeros(2)

    x = initstate!(nonlinmodel, [0, 10]) # do nothing for NonLinModel
    @test evaloutput(nonlinmodel) ≈ [0, 0]

    @test_throws DimensionMismatch updatestate!(nonlinmodel, zeros(2), zeros(1))
    @test_throws DimensionMismatch evaloutput(nonlinmodel, zeros(1))
end

@testitem "NonLinModel linearization" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra, ForwardDiff
    Ts = 1.0
    f1(x,u,d,_) = x.^5 + u.^4 + d.^3
    h1(x,d,_)   = x.^2 + d
    nonlinmodel1 = NonLinModel(f1,h1,Ts,1,1,1,1,solver=nothing)
    x, u, d = [2.0], [3.0], [4.0]
    linmodel1 = linearize(nonlinmodel1; x, u, d)
    @test linmodel1.A  ≈ 5*x.^4
    @test linmodel1.Bu ≈ 4*u.^3
    @test linmodel1.Bd ≈ 3*d.^2
    @test linmodel1.C  ≈ 2*x.^1
    @test linmodel1.Dd ≈ 1*d.^0
    linmodel2 = LinModel(nonlinmodel1; x, u, d)
    @test linmodel1.A  ≈ linmodel2.A
    @test linmodel1.Bu ≈ linmodel2.Bu
    @test linmodel1.Bd ≈ linmodel2.Bd
    @test linmodel1.C  ≈ linmodel2.C
    @test linmodel1.Dd ≈ linmodel2.Dd 

    f1!(ẋ, x, u, d, _) = (ẋ .= x.^5 + u.^4 + d.^3; nothing)
    h1!(y, x, d, _) = (y .= x.^2 + d; nothing)
    nonlinmodel3 = NonLinModel(f1!,h1!,Ts,1,1,1,1,solver=RungeKutta())
    linmodel3 = linearize(nonlinmodel3; x, u, d)
    u0, d0 = u - nonlinmodel3.uop, d - nonlinmodel3.dop
    xnext, y = similar(nonlinmodel3.x0), similar(nonlinmodel3.yop)
    A  = ForwardDiff.jacobian((xnext, x)  -> nonlinmodel3.f!(xnext, x, u0, d0, nonlinmodel3.p), xnext, x)
    Bu = ForwardDiff.jacobian((xnext, u0) -> nonlinmodel3.f!(xnext, x, u0, d0, nonlinmodel3.p), xnext, u0)
    Bd = ForwardDiff.jacobian((xnext, d0) -> nonlinmodel3.f!(xnext, x, u0, d0, nonlinmodel3.p), xnext, d0)
    C  = ForwardDiff.jacobian((y, x)  -> nonlinmodel3.h!(y, x, d0, nonlinmodel3.p), y, x)
    Dd = ForwardDiff.jacobian((y, d0) -> nonlinmodel3.h!(y, x, d0, nonlinmodel3.p), y, d0)
    @test linmodel3.A  ≈ A
    @test linmodel3.Bu ≈ Bu
    @test linmodel3.Bd ≈ Bd
    @test linmodel3.C  ≈ C
    @test linmodel3.Dd ≈ Dd

    # test `linearize` at a non-equilibrium point:
    N = 5
    x, u, d = [0.2], [0.0], [0.0]
    Ynl = zeros(N)
    Yl  = zeros(N)
    setstate!(nonlinmodel3, x)
    linmodel3 = linearize(nonlinmodel3; x, u, d)
    for i=1:N
        ynl = nonlinmodel3(d)
        global yl  = linmodel3(d)
        Ynl[i] = ynl[1]
        Yl[i]  = yl[1]
        global linmodel3 = linearize(nonlinmodel3; u, d)
        updatestate!(nonlinmodel3, u, d)
        updatestate!(linmodel3, u, d)
    end
    @test all(isapprox.(Ynl, Yl, atol=1e-6))
end

@testitem "NonLinModel real time simulations" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel1 = LinModel(tf(2, [10, 1]), 0.1)
    nonlinmodel1 = NonLinModel(
        (x,u,_,_)->linmodel1.A*x + linmodel1.Bu*u,
        (x,_,_)->linmodel1.C*x,
        linmodel1.Ts, 1, 1, 1, 0, solver=nothing
    )
    times1 = zeros(5)
    for i=1:5
        times1[i] = savetime!(nonlinmodel1)
        updatestate!(nonlinmodel1, [1])
        periodsleep(nonlinmodel1)
    end
    @test all(isapprox.(diff(times1[2:end]), 0.1, atol=0.01))
    linmodel2 = LinModel(tf(2, [0.1, 1]), 0.001)
    nonlinmodel2 = NonLinModel(
        (x,u,_,_)->linmodel2.A*x + linmodel2.Bu*u,
        (x,_,_)->linmodel2.C*x,
        linmodel2.Ts, 1, 1, 1, 0, solver=nothing
    )
    times2 = zeros(5)
    for i=1:5
        times2[i] = savetime!(nonlinmodel2)
        updatestate!(nonlinmodel2, [1])
        periodsleep(nonlinmodel2, true)
    end
    @test all(isapprox.(diff(times2[2:end]), 0.001, atol=0.0001))
end