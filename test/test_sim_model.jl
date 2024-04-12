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

    linmodel6 = LinModel([delay(4) delay(4)]*sys,Ts,i_d=[3])
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

    linmodel11 = LinModel{Float32}(Gss.A, Gss.B, Gss.C, zeros(2, 0), zeros(2, 0), Ts)
    @test isa(linmodel11, LinModel{Float32})


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

@testset "LinModel sim methods" begin
    linmodel1 = setop!(LinModel(Gss), uop=[10,50], yop=[50,30])
    @test updatestate!(linmodel1, [10, 50]) ≈ zeros(2)
    @test updatestate!(linmodel1, [10, 50], Float64[]) ≈ zeros(2)
    @test linmodel1.x ≈ zeros(2)
    @test evaloutput(linmodel1) ≈ linmodel1() ≈ [50,30]
    @test evaloutput(linmodel1, Float64[]) ≈ linmodel1(Float64[]) ≈ [50,30]
    x = initstate!(linmodel1, [10, 60])
    @test evaloutput(linmodel1) ≈ [50 + 19.0, 30 + 7.4]
    @test updatestate!(linmodel1, [10, 60]) ≈ x
    linmodel2 = LinModel(append(tf(1, [1, 0]), tf(2, [10, 1])), 1.0)
    x = initstate!(linmodel2, [10, 3])
    @test evaloutput(linmodel2) ≈ [0, 6]
    @test updatestate!(linmodel2, [0, 3]) ≈ x

    @test_throws DimensionMismatch updatestate!(linmodel1, zeros(2), zeros(1))
    @test_throws DimensionMismatch evaloutput(linmodel1, zeros(1))
end

@testset "NonLinModel construction" begin
    linmodel1 = LinModel(sys,Ts,i_u=[1,2])
    f1(x,u,_) = linmodel1.A*x + linmodel1.Bu*u
    h1(x,_)   = linmodel1.C*x
    nonlinmodel1 = NonLinModel(f1,h1,Ts,2,2,2,solver=nothing)
    @test nonlinmodel1.nx == 2
    @test nonlinmodel1.nu == 2
    @test nonlinmodel1.nd == 0
    @test nonlinmodel1.ny == 2
    xnext, y = similar(nonlinmodel1.x), similar(nonlinmodel1.yop)
    nonlinmodel1.f!(xnext,[0,0],[0,0],[1])
    @test xnext ≈ zeros(2,)
    nonlinmodel1.h!(y,[0,0],[1])
    @test y ≈ zeros(2,)

    linmodel2 = LinModel(sys,Ts,i_d=[3])
    f2(x,u,d) = linmodel2.A*x + linmodel2.Bu*u + linmodel2.Bd*d
    h2(x,d)   = linmodel2.C*x + linmodel2.Dd*d
    nonlinmodel2 = NonLinModel(f2,h2,Ts,2,4,2,1,solver=nothing)

    @test nonlinmodel2.nx == 4
    @test nonlinmodel2.nu == 2
    @test nonlinmodel2.nd == 1
    @test nonlinmodel2.ny == 2
    xnext, y = similar(nonlinmodel2.x), similar(nonlinmodel2.yop)
    nonlinmodel2.f!(xnext,[0,0,0,0],[0,0],[0])
    @test xnext ≈ zeros(4,)
    nonlinmodel2.h!(y,[0,0,0,0],[0])
    @test y ≈ zeros(2,)

    nonlinmodel3 = NonLinModel{Float32}(f2,h2,Ts,2,4,2,1,solver=nothing)
    @test isa(nonlinmodel3, NonLinModel{Float32})

    function f1!(xnext, x, u, d)
        mul!(xnext, linmodel2.A,  x)
        mul!(xnext, linmodel2.Bu, u, 1, 1)
        mul!(xnext, linmodel2.Bd, d, 1, 1)
        return nothing
    end 
    function h1!(y, x, d)
        mul!(y, linmodel2.C,  x)
        mul!(y, linmodel2.Dd, d, 1, 1)
        return nothing
    end
    nonlinmodel4 = NonLinModel(f1!, h1!, Ts, 2, 4, 2, 1, solver=nothing)
    xnext, y = similar(nonlinmodel4.x), similar(nonlinmodel4.yop)
    nonlinmodel4.f!(xnext,[0,0,0,0],[0,0],[0])
    @test xnext ≈ zeros(4)
    nonlinmodel4.h!(y,[0,0,0,0],[0])
    @test y ≈ zeros(2)

    A  = [0 0.5; -0.2 -0.1]
    Bu = reshape([0; 0.5], 2, 1)
    Bd = reshape([0; 0.5], 2, 1)
    C  = [0.4 0]
    Dd = reshape([0], 1, 1)
    f3(x, u, d) = A*x + Bu*u+ Bd*d
    h3(x, d) = C*x + Dd*d
    nonlinmodel5 = NonLinModel(f3, h3, 1.0, 1, 2, 1, 1, solver=RungeKutta())
    xnext, y = similar(nonlinmodel5.x), similar(nonlinmodel5.yop)
    nonlinmodel5.f!(xnext, [0; 0], [0], [0])
    @test xnext ≈ zeros(2)
    nonlinmodel5.h!(y, [0; 0], [0])
    @test y ≈ zeros(1)

    function f2!(ẋ, x, u , d)
        mul!(ẋ, A, x)
        mul!(ẋ, Bu, u, 1, 1)
        mul!(ẋ, Bd, d, 1, 1)
        return nothing
    end
    function h2!(y, x, d)
        mul!(y, C, x)
        mul!(y, Dd, d, 1, 1)
        return nothing
    end
    nonlinmodel6 = NonLinModel(f2!, h2!, 1.0, 1, 2, 1, 1, solver=RungeKutta())
    xnext, y = similar(nonlinmodel6.x), similar(nonlinmodel6.yop)
    nonlinmodel6.f!(xnext, [0; 0], [0], [0])
    @test xnext ≈ zeros(2)
    nonlinmodel6.h!(y, [0; 0], [0])
    @test y ≈ zeros(1)
    
    @test_throws ErrorException NonLinModel(
        (x,u)->linmodel1.A*x + linmodel1.Bu*u,
        (x,_)->linmodel1.C*x, Ts, 2, 4, 2, 1, solver=nothing)
    @test_throws ErrorException NonLinModel(
        (x,u,_)->linmodel1.A*x + linmodel1.Bu*u,
        (x)->linmodel1.C*x, Ts, 2, 4, 2, 1, solver=nothing)
end

@testset "NonLinModel sim methods" begin
    linmodel1 = LinModel(sys,Ts,i_u=[1,2])
    f1(x,u,_) = linmodel1.A*x + linmodel1.Bu*u
    h1(x,_)   = linmodel1.C*x
    nonlinmodel = NonLinModel(f1,h1,Ts,2,2,2,solver=nothing)

    @test updatestate!(nonlinmodel, zeros(2,)) ≈ zeros(2) 
    @test updatestate!(nonlinmodel, zeros(2,), Float64[]) ≈ zeros(2)
    @test nonlinmodel.x ≈ zeros(2)
    @test evaloutput(nonlinmodel) ≈ nonlinmodel() ≈ zeros(2)
    @test evaloutput(nonlinmodel, Float64[]) ≈ nonlinmodel(Float64[]) ≈ zeros(2)

    x = initstate!(nonlinmodel, [0, 10]) # do nothing for NonLinModel
    @test evaloutput(nonlinmodel) ≈ [0, 0]

    @test_throws DimensionMismatch updatestate!(nonlinmodel, zeros(2), zeros(1))
    @test_throws DimensionMismatch evaloutput(nonlinmodel, zeros(1))
end

@testset "NonLinModel linearization" begin
    Ts = 1.0
    f1(x,u,d) = x.^5 + u.^4 + d.^3
    h1(x,d)   = x.^2 + d
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

    f1!(ẋ, x, u, d) = (ẋ .= x.^5 + u.^4 + d.^3; nothing)
    h1!(y, x, d) = (y .= x.^2 + d; nothing)
    nonlinmodel2 = NonLinModel(f1!,h1!,Ts,1,1,1,1,solver=RungeKutta())
    linmodel3 = linearize(nonlinmodel2; x, u, d)
    u0, d0 = u - nonlinmodel2.uop, d - nonlinmodel2.dop
    xnext, y = similar(nonlinmodel2.x), similar(nonlinmodel2.yop)
    A  = ForwardDiff.jacobian((xnext, x)  -> nonlinmodel2.f!(xnext, x, u0, d0), xnext, x)
    Bu = ForwardDiff.jacobian((xnext, u0) -> nonlinmodel2.f!(xnext, x, u0, d0), xnext, u0)
    Bd = ForwardDiff.jacobian((xnext, d0) -> nonlinmodel2.f!(xnext, x, u0, d0), xnext, d0)
    C  = ForwardDiff.jacobian((y, x)  -> nonlinmodel2.h!(y, x, d0), y, x)
    Dd = ForwardDiff.jacobian((y, d0) -> nonlinmodel2.h!(y, x, d0), y, d0)
    @test linmodel3.A  ≈ A
    @test linmodel3.Bu ≈ Bu
    @test linmodel3.Bd ≈ Bd
    @test linmodel3.C  ≈ C
    @test linmodel3.Dd ≈ Dd
end