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
    sysu_ss = c2d(sminreal(ss(sys)[:,1:2]), Ts, :zoh)
    sysd_ss = c2d(sminreal(ss(sys)[:,3]),   Ts, :tustin)
    sys_ss = minreal([sysu_ss sysd_ss])
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
    @test sum(isapprox.(eigvals(linmodel6.A), 0, atol=1e-15)) == 1

    A = diagm( .1: .1: .3)
    Bu = [I(2); zeros(1,2)]
    C = diagm( .4: .1: .6)
    Bd = [zeros(2,1); I(1)]
    Dd = 0;
    linmodel7 = LinModel(A, Bu, C, Bd, Dd, 1.0)
    @test linmodel7.A ≈ A
    @test linmodel7.Bu ≈ Bu
    @test linmodel7.Bd ≈ Bd
    @test linmodel7.C ≈ C
    @test linmodel7.Dd ≈ zeros(3,1)

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
    u, d = [10, 50], Float64[]
    @test updatestate!(linmodel1, u) ≈ zeros(2)
    @test updatestate!(linmodel1, u, d) ≈ zeros(2)
    @test linmodel1.x0 ≈ zeros(2)
    @test evaloutput(linmodel1) ≈ linmodel1() ≈ [50,30]
    @test evaloutput(linmodel1, Float64[]) ≈ linmodel1(Float64[]) ≈ [50,30]
    x = initstate!(linmodel1, [10, 60])
    @test evaloutput(linmodel1) ≈ [50 + 19.0, 30 + 7.4]
    @test preparestate!(linmodel1) ≈ x           
    @test updatestate!(linmodel1,  [10, 60]) ≈ x
    linmodel2 = LinModel(append(tf(1, [1, 0]), tf(2, [10, 1])), 1.0)
    x = initstate!(linmodel2, [10, 3])
    @test evaloutput(linmodel2) ≈ [0, 6]
    @test updatestate!(linmodel2, [0, 3]) ≈ x

    @test_throws DimensionMismatch updatestate!(linmodel1, zeros(2), zeros(1))
    @test_throws DimensionMismatch evaloutput(linmodel1, zeros(1))
end

@testitem "LinModel linearization" setup=[SetupMPCtests] begin
    using .SetupMPCtests
    linmodel1 = LinModel(sys, Ts, i_d=[3])
    linmodel2 = linearize(linmodel1, x=[1,2,3,4], u=[5,6], d=[7])
    @test linmodel2.A ≈ linmodel1.A
    @test linmodel2.Bu ≈ linmodel1.Bu
    @test linmodel2.Bd ≈ linmodel1.Bd
    @test linmodel2.C ≈ linmodel1.C
    @test linmodel2.Dd ≈ linmodel1.Dd
end

@testitem "LinModel real time simulations" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel1 = LinModel(tf(2, [10, 1]), 0.25)
    times1 = zeros(5)
    for i=1:5
        times1[i] = savetime!(linmodel1)
        updatestate!(linmodel1, [1])
        periodsleep(linmodel1)
    end
    @test all(isapprox.(diff(times1[2:end]), 0.25, atol=0.05))
    linmodel2 = LinModel(tf(2, [0.1, 1]), 0.25)
    times2 = zeros(5)
    for i=1:5
        times2[i] = savetime!(linmodel2)
        updatestate!(linmodel2, [1])
        periodsleep(linmodel2, true)
    end
    @test all(isapprox.(diff(times2[2:end]), 0.25, atol=0.05))
end

@testitem "NonLinModel construction" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    using DifferentiationInterface
    import FiniteDiff
    linmodel1 = LinModel(sys,Ts,i_u=[1,2])
    f1!(x,u,_,model) = model.A*x + model.Bu*u
    h1!(x,_,model)   = model.C*x
    daemodel = NonLinModel(f1!,h1!,Ts,2,2,2,solver=nothing,p=linmodel1)
    @test daemodel.nx == 2
    @test daemodel.nu == 2
    @test daemodel.nd == 0
    @test daemodel.ny == 2
    ẋ, y = daemodel.buffer.x, daemodel.buffer.y
    daemodel.f!(ẋ, [0,0],[0,0],[1],daemodel.p)
    @test ẋ ≈ zeros(2,)
    daemodel.h!(y,[0,0],[1],daemodel.p)
    @test y ≈ zeros(2,)

    linmodel2 = LinModel(sys,Ts,i_d=[3])
    f2(x,u,d,model) = model.A*x + model.Bu*u + model.Bd*d
    h2(x,d,model)   = model.C*x + model.Dd*d
    nonlinmodel2 = NonLinModel(f2,h2,Ts,2,4,2,1,solver=nothing,p=linmodel2)

    @test nonlinmodel2.nx == 4
    @test nonlinmodel2.nu == 2
    @test nonlinmodel2.nd == 1
    @test nonlinmodel2.ny == 2
    ẋ, y = nonlinmodel2.buffer.x, nonlinmodel2.buffer.y
    nonlinmodel2.f!(ẋ,[0,0,0,0],[0,0],[0],nonlinmodel2.p)
    @test ẋ ≈ zeros(4,)
    nonlinmodel2.h!(y,[0,0,0,0],[0],nonlinmodel2.p)
    @test y ≈ zeros(2,)

    nonlinmodel3 = NonLinModel{Float32}(f2,h2,Ts,2,4,2,1,solver=nothing)
    @test isa(nonlinmodel3, NonLinModel{Float32})

    function f1!(xnext, x, u, d, model)
        mul!(xnext, model.A,  x)
        mul!(xnext, model.Bu, u, 1, 1)
        mul!(xnext, model.Bd, d, 1, 1)
        return nothing
    end 
    function h1!(y, x, d, model)
        mul!(y, model.C,  x)
        mul!(y, model.Dd, d, 1, 1)
        return nothing
    end
    nonlinmodel4 = NonLinModel(f1!, h1!, Ts, 2, 4, 2, 1, solver=nothing, p=linmodel2)
    ẋ, y = nonlinmodel4.buffer.x, nonlinmodel4.buffer.y
    nonlinmodel4.f!(ẋ,[0,0,0,0],[0,0],[0],nonlinmodel4.p)
    @test ẋ ≈ zeros(4)
    nonlinmodel4.h!(y,[0,0,0,0],[0],nonlinmodel4.p)
    @test y ≈ zeros(2)

    A  = [0 0.5; -0.2 -0.1]
    Bu = reshape([0; 0.5], 2, 1)
    Bd = reshape([0; 0.5], 2, 1)
    C  = [0.4 0]
    Dd = reshape([0], 1, 1)
    p=(; A, Bu, Bd, C, Dd)
    f3(x, u, d, p) = p.A*x + p.Bu*u+ p.Bd*d
    h3(x, d, p) = p.C*x + p.Dd*d
    solver=RungeKutta(4)
    @test string(solver) == 
        "4th order Runge-Kutta differential equation solver with 1 supersamples."
    nonlinmodel5 = NonLinModel(f3, h3, 1.0, 1, 2, 1, 1, solver=solver, p=p)
    ẋ, k̄, y = nonlinmodel5.buffer.x, nonlinmodel5.buffer.k̄, nonlinmodel5.buffer.y
    ModelPredictiveControl.f!(ẋ, k̄, nonlinmodel5, [0; 0], [0], [0], nonlinmodel5.p)
    @test ẋ ≈ zeros(2)
    ModelPredictiveControl.h!(y, nonlinmodel5, [0; 0], [0], nonlinmodel5.p)
    @test y ≈ zeros(1)

    function f2!(ẋ, x, u , d, p)
        mul!(ẋ, p.A, x)
        mul!(ẋ, p.Bu, u, 1, 1)
        mul!(ẋ, p.Bd, d, 1, 1)
        return nothing
    end
    function h2!(y, x, d, p)
        mul!(y, p.C, x)
        mul!(y, p.Dd, d, 1, 1)
        return nothing
    end
    nonlinmodel6 = NonLinModel(f2!, h2!, 1.0, 1, 2, 1, 1, solver=RungeKutta(), p=p)
    ẋ, k̄, y = nonlinmodel6.buffer.x, nonlinmodel6.buffer.k̄, nonlinmodel6.buffer.y
    ModelPredictiveControl.f!(ẋ, k̄, nonlinmodel6, [0; 0], [0], [0], nonlinmodel6.p)
    @test ẋ ≈ zeros(2)
    ModelPredictiveControl.h!(y, nonlinmodel6, [0; 0], [0], nonlinmodel6.p)
    @test y ≈ zeros(1)
    nonlinmodel7 = NonLinModel(f2!, h2!, 1.0, 1, 2, 1, 1, solver=ForwardEuler(), p=p)
    ẋ, k̄, y = nonlinmodel7.buffer.x, nonlinmodel7.buffer.k̄, nonlinmodel7.buffer.y
    ModelPredictiveControl.f!(ẋ, k̄, nonlinmodel7, [0; 0], [0], [0], nonlinmodel7.p)
    @test ẋ ≈ zeros(2)
    ModelPredictiveControl.h!(y, nonlinmodel7, [0; 0], [0], nonlinmodel7.p)
    @test y ≈ zeros(1)
    nonlinmodel8 = NonLinModel(f2!, h2!, 1.0, 1, 2, 1, 1, p=p, jacobian=AutoFiniteDiff())
    @test nonlinmodel8.jacobian == AutoFiniteDiff()

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

    @test_throws ArgumentError RungeKutta(2)
    @test_throws ArgumentError RungeKutta(0)
    @test_throws ArgumentError RungeKutta(4, supersample=0)
end

@testitem "NonLinModel sim methods" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel1 = LinModel(sys,Ts,i_u=[1,2])
    function f1!(xnext, x, u, _ , model)
        mul!(xnext, model.A, x)
        mul!(xnext, model.Bu, u, 1, 1)
    end
    h1!(y, x , _ , model) = mul!(y, model.C, x)
    nonlinmodel = NonLinModel(f1!,h1!,Ts,2,2,2,p=linmodel1,solver=nothing)

    u, d = zeros(2), Float64[]
    @test updatestate!(nonlinmodel, u) ≈ zeros(2) 
    @test updatestate!(nonlinmodel, u, d) ≈ zeros(2)
    @test nonlinmodel.x0 ≈ zeros(2)
    @test evaloutput(nonlinmodel) ≈ nonlinmodel() ≈ zeros(2)
    @test evaloutput(nonlinmodel, d) ≈ nonlinmodel(Float64[]) ≈ zeros(2)

    x = initstate!(nonlinmodel, [0, 10]) # do nothing for NonLinModel
    @test evaloutput(nonlinmodel) ≈ [0, 0]

    @test_throws DimensionMismatch updatestate!(nonlinmodel, zeros(2), zeros(1))
    @test_throws DimensionMismatch evaloutput(nonlinmodel, zeros(1))
end

@testitem "NonLinModel linearization" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    using DifferentiationInterface
    import ForwardDiff, FiniteDiff
    Ts = 1.0
    f1!(x,u,d,_) = x.^5 .+ u.^4 .+ d.^3
    h1!(x,d,_)   = x.^2 .+ d
    daemodel = NonLinModel(f1!,h1!,Ts,1,1,1,1,solver=nothing)
    x, u, d = [2.0], [3.0], [4.0]
    linmodel1 = linearize(daemodel; x, u, d)
    @test linmodel1.A  ≈ 5*x.^4
    @test linmodel1.Bu ≈ 4*u.^3
    @test linmodel1.Bd ≈ 3*d.^2
    @test linmodel1.C  ≈ 2*x.^1
    @test linmodel1.Dd ≈ 1*d.^0
    linmodel1b = LinModel(daemodel; x, u, d)
    @test linmodel1.A  ≈ linmodel1b.A
    @test linmodel1.Bu ≈ linmodel1b.Bu
    @test linmodel1.Bd ≈ linmodel1b.Bd
    @test linmodel1.C  ≈ linmodel1b.C
    @test linmodel1.Dd ≈ linmodel1b.Dd  

    nonlinmodel2 = NonLinModel(f1!,h1!,Ts,1,1,1,1,solver=nothing, jacobian=AutoFiniteDiff()) 
    linmodel2 = linearize(nonlinmodel2; x, u, d)
    @test linmodel2.A  ≈ 5*x.^4 atol=1e-3
    @test linmodel2.Bu ≈ 4*u.^3 atol=1e-3
    @test linmodel2.Bd ≈ 3*d.^2 atol=1e-3
    @test linmodel2.C  ≈ 2*x.^1 atol=1e-3
    @test linmodel2.Dd ≈ 1*d.^0 atol=1e-3

    f1!(ẋ, x, u, d, _) = (ẋ .= x.^5 .+ u.^4 .+ d.^3; nothing)
    h1!(y, x, d, _) = (y .= x.^2 .+ d; nothing)
    nonlinmodel3 = NonLinModel(f1!,h1!,Ts,1,1,1,1,solver=RungeKutta())
    linmodel3 = linearize(nonlinmodel3; x, u, d)
    x0, u0, d0 = x - nonlinmodel3.xop, u - nonlinmodel3.uop, d - nonlinmodel3.dop
    x0next, k, y0 = nonlinmodel3.buffer.x, nonlinmodel3.buffer.k̄, nonlinmodel3.buffer.y
    backend = AutoForwardDiff()
    f_A(x0next, x0, k)  = ModelPredictiveControl.f!(x0next, k, nonlinmodel3, x0, u0, d0, nonlinmodel3.p)
    f_Bu(x0next, u0, k) = ModelPredictiveControl.f!(x0next, k, nonlinmodel3, x0, u0, d0, nonlinmodel3.p)
    f_Bd(x0next, d0, k) = ModelPredictiveControl.f!(x0next, k, nonlinmodel3, x0, u0, d0, nonlinmodel3.p)
    h_C(y0, x0)  = ModelPredictiveControl.h!(y0, nonlinmodel3, x0, d0, nonlinmodel3.p)
    h_Dd(y0, d0) = ModelPredictiveControl.h!(y0, nonlinmodel3, x0, d0, nonlinmodel3.p)
    A  = jacobian(f_A,  x0next, backend, x0, Cache(k))
    Bu = jacobian(f_Bu, x0next, backend, u0, Cache(k))
    Bd = jacobian(f_Bd, x0next, backend, d0, Cache(k))
    C  = jacobian(h_C,  y0, backend, x0)
    Dd = jacobian(h_Dd, y0, backend, d0)
    @test linmodel3.A  ≈ A
    @test linmodel3.Bu ≈ Bu
    @test linmodel3.Bd ≈ Bd
    @test linmodel3.C  ≈ C
    @test linmodel3.Dd ≈ Dd

    # test `linearize` at a non-equilibrium point:
    Ynl, Yl = let nonlinmodel3=nonlinmodel3
        N = 5
        Ynl = zeros(N)
        Yl = zeros(N)
        x, u, d = [0.2], [0.0], [0.0]
        setstate!(nonlinmodel3, x)
        linmodel3 = linearize(nonlinmodel3; x, u, d)
        for i=1:N
            ynl = nonlinmodel3(d)
            yl  = linmodel3(d)
            Ynl[i] = ynl[1]
            Yl[i]  = yl[1]
            linearize!(linmodel3, nonlinmodel3; u, d)
            updatestate!(nonlinmodel3, u, d)
            updatestate!(linmodel3, u, d)
        end
        Ynl, Yl
    end
    @test all(isapprox.(Ynl, Yl, atol=1e-6))

    # test nd==0 also works with AutoFiniteDiff (does not support empty matrices):
    f2!(x0next, x0, u0, _, _) = (x0next .= x0 .+ u0)
    h2!(y0, x0, _, _) = (y0 .= x0)
    nonlinmodel4 = NonLinModel(f2!,h2!,Ts,1,1,1,0,solver=nothing,jacobian=AutoFiniteDiff())
    @test_nowarn linearize(nonlinmodel4, x=[1], u=[2])

    # test linearization with nonzero operating points in the NonLinModel object:
    linmodel_op = LinModel(tf(2, [10, 1]),1)
    linmodel_op = setop!(linmodel_op, uop=[10], yop=[20], xop=[-2], fop=[5])
    f3!(xnext, x, u, _, p) = (xnext .= p.A*x .+ p.Bu*u)
    h3!(y, x, _, p) = (y .= p.C*x)
    nonlinmodel_op = NonLinModel(f3!, h3!, 1.0, 1, 1, 1, p=linmodel_op, solver=nothing)
    nonlinmodel_op = setop!(linmodel_op, uop=[10], yop=[20], xop=[-2], fop=[5])
    linmodel_op_2 = linearize(nonlinmodel_op)
    @test linmodel_op.A   ≈ linmodel_op_2.A
    @test linmodel_op.Bu  ≈ linmodel_op_2.Bu
    @test linmodel_op.C   ≈ linmodel_op_2.C
    @test linmodel_op.uop ≈ linmodel_op_2.uop
    @test linmodel_op.yop ≈ linmodel_op_2.yop
    @test linmodel_op.xop ≈ linmodel_op_2.xop
    @test linmodel_op.fop ≈ linmodel_op_2.fop
end

@testitem "NonLinModel real time simulations" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    linmodel1 = LinModel(tf(2, [10, 1]), 0.25)
    daemodel = NonLinModel(
        (x,u,_,_)->linmodel1.A*x + linmodel1.Bu*u,
        (x,_,_)->linmodel1.C*x,
        linmodel1.Ts, 1, 1, 1, 0, solver=nothing
    )
    times1 = zeros(5)
    for i=1:5
        times1[i] = savetime!(daemodel)
        updatestate!(daemodel, [1])
        periodsleep(daemodel)
    end
    @test all(isapprox.(diff(times1[2:end]), 0.25, atol=0.05))
    linmodel2 = LinModel(tf(2, [0.1, 1]), 0.25)
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
    @test all(isapprox.(diff(times2[2:end]), 0.25, atol=0.05))
end

@testitem "NonLinModelDAE construction" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    using JuMP, Ipopt
    using DifferentiationInterface
    import FiniteDiff

    function fq!(ẋ, res, x, a, u, _ , p)
        ẋ[] = -p[] * (x[] - 0.2 * u[])
        res[] = x[] - a[]
        return nothing
    end
    function fq(x, a, u, _ , p)
        ẋ = -p .* (x .- 0.2 .* u)
        res = x .- a
        return ẋ, res
    end
    function h!(y, x, a, _ , _ )
        y[] = 2*x[]+ a[]
    end
    function h(x, a, _ , _ )
        y = 2 .* x .+  a
        return y
    end
    nu, nx, na, ny = 1, 1, 1, 1
    Ts = 1
    p = [0.5]

    dae = NonLinModelDAE(fq!, h!, Ts, nu, nx, na, ny; p)
    @test dae.nx == nx
    @test dae.na == na
    @test dae.nu == nu
    @test dae.nd == 0
    @test dae.ny == ny
    @test dae.iszero_Ha == false 
    ẋ, q, y = dae.buffer.x, dae.buffer.a, dae.buffer.y
    dae.fq!(ẋ, q, [0], [0], [0], [0], dae.p)
    @test ẋ ≈ zeros(1)
    @test q ≈ zeros(1)
    dae.h!(y, [0], [0], [0], dae.p)
    @test y ≈ zeros(1)

    dae_oop = NonLinModelDAE(fq, h, Ts, nu, nx, na, ny; p)
    dae_oop.fq!(ẋ, q, [0], [0], [0], [0], dae_oop.p)
    @test ẋ ≈ zeros(1)
    @test q ≈ zeros(1)
    dae_oop.h!(y, [0], [0], [0], dae_oop.p)
    @test y ≈ zeros(1)

    transcription = TrapezoidalCollocation()
    dae2 = NonLinModelDAE(fq!, h!, Ts, nu, nx, na, ny; p, transcription)
    @test dae2.transcription isa TrapezoidalCollocation
    @test length(dae2.Z) == 3
    @test size(dae2.Aeq, 1) == 0

    transcription = OrthogonalCollocation(0, 4, roots=:gausslegendre)
    dae3 = NonLinModelDAE(fq!, h!, Ts, nu, nx, na, ny; p, transcription)
    @test dae3.transcription isa OrthogonalCollocation
    @test length(dae3.Z) == 1 + 1 + 4 + 4
    @test size(dae3.Aeq, 1) == 1

    optim_state  = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer, "sb"=>"yes"))
    optim_output = JuMP.Model(optimizer_with_attributes(Ipopt.Optimizer, "sb"=>"yes"))
    dae4 = NonLinModelDAE(fq!, h!, Ts, nu, nx, na, ny; p, optim_state, optim_output)
    @test solver_name(dae4.optim_state)  == "Ipopt"
    @test solver_name(dae4.optim_output) == "Ipopt"

    jacobian = AutoFiniteDiff()
    hessian  = true
    dae5 = NonLinModelDAE(fq!, h!, Ts, nu, nx, na, ny; p, jacobian, hessian)
    @test dae5.jacobian isa AutoFiniteDiff
    @test dae5.hessian  isa AutoForwardDiff 

    function h_no_a!(y, x, _ , _ , _ )
        y .= 2 .* x
    end
    dae6 = NonLinModelDAE(fq!, h_no_a!, Ts, nu, nx, na, ny; p)
    @test dae6.iszero_Ha == true

    dae7 = NonLinModelDAE{Float32}(fq!,h!, Ts, nu, nx, na, ny; p)
    @test isa(dae7, NonLinModelDAE{Float32})
    
    @test_throws ErrorException NonLinModelDAE(
        (x,u,p)->(x+u+p, 0.0),
        (x,a,d,p)->(x+a+d+p), Ts, 1, 1, 1, 1)
    @test_throws ErrorException NonLinModelDAE(
        (x,a,u,d,p)->(x+u+p, a-x),
        (x,a,d)->(x+a+d), Ts, 1, 1, 1, 1)
    @test_throws ArgumentError NonLinModelDAE(
        fq!, h!, Ts, 1, 1, 1, 1; p, optim_state, optim_output=optim_state
    )

    # DAE with direct transmission from input to output:
    function fq_dt!(ẋ, res, x, a, u, _ , p) 
        ẋ[] = -p[] * (x[] - 0.2 * u[])
        res[] = a[] - u[]
    end
    h_dt!(y, x, a, _ , _ ) = (y[] = 2*x[] + a[])
    @test_throws ErrorException NonLinModelDAE(
        fq_dt!, h_dt!, Ts, nu, nx, na, ny; p
    )
end

@testitem "NonLinModelDAE sim methods" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra
    using DifferentiationInterface
    import FiniteDiff

    function fq!(ẋ, res, x, a, u, _ , p)
        ẋ[] = -p[] * (x[] - 0.2 * u[])
        res[] = x[] - a[]
        return nothing
    end
    function h!(y, x, a, _ , _ )
        y[] = 2*x[]+ a[]
    end
    nu, nx, na, ny = 1, 1, 1, 1
    Ts = 1
    p = [1.0]

    transcription = OrthogonalCollocation(0, 4, roots=:gausslegendre)
    dae = NonLinModelDAE(fq!, h!, Ts, nu, nx, na, ny; p, transcription)
    u = [0.0]
    d = Float64[]

    @test updatestate!(dae, u) ≈ zeros(1)
    @test updatestate!(dae, u, d) ≈ zeros(1)
    @test dae.x0 ≈ zeros(1)
    @test evaloutput(dae) ≈ dae() ≈ zeros(1)

    transcription = TrapezoidalCollocation()
    dae2 = NonLinModelDAE(fq!, h!, Ts, nu, nx, na, ny; p, transcription)
    @test updatestate!(dae2, u) ≈ zeros(1)
    @test updatestate!(dae2, u, d) ≈ zeros(1)
    @test dae2.x0 ≈ zeros(1)
    @test evaloutput(dae2) ≈ dae2() ≈ zeros(1)

    x = initstate!(dae, [10]) # do nothing for NonLinModelDAE
    @test evaloutput(dae) ≈ [0]

    @test_throws DimensionMismatch updatestate!(dae, zeros(2))
    @test_throws DimensionMismatch updatestate!(dae, zeros(1), zeros(1))
    @test_throws DimensionMismatch evaloutput(dae, zeros(1))

end