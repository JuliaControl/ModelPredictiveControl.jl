# spell-checker: disable

using Revise

using ModelPredictiveControl

using JuMP, DAQP, Ipopt, OSQP
using LinearAlgebra
using ControlSystemsBase
using BenchmarkTools

using PlotThemes, Plots
#theme(:default)
theme(:dark)
default(fontfamily="Computer Modern"); scalefontsizes(1.1)


Ts = 4.0
A =  [  0.800737  0.0       0.0  0.0
        0.0       0.606531  0.0  0.0
        0.0       0.0       0.8  0.0
        0.0       0.0       0.0  0.6    ]
Bu = [  0.378599  0.378599
        -0.291167  0.291167
        0.0       0.0
        0.0       0.0                   ]
Bd = [  0; 0; 0.5; 0.5;;                ]
C =  [  1.0  0.0  0.684   0.0
        0.0  1.0  0.0    -0.4736        ]
Dd = [  0.19; -0.148;;                  ]
Du = zeros(2, 2)

linModel1 = LinModel(ss(A,Bu,C,0,Ts),Ts)
linModel2 = LinModel(ss(A,[Bu Bd],C,[Du Dd],Ts),Ts,i_d=[3])
sys = [   tf(1.90,[18.0,1])   tf(1.90,[18.0,1])   tf(1.90,[18.0,1]);
          tf(-0.74,[8.0,1])   tf(0.74,[8.0,1])    tf(-0.74,[8.0,1])   ]
linModel3 = LinModel(sys,Ts,i_d=[3])
linModel4 = setop!(LinModel(A, Bu, C, Bd, Dd, Ts), uop=[10,10],yop=[50,30],dop=[5])
linModel5 = LinModel([delay(4) delay(8)]*sys,Ts,i_d=[3])

f(x,u,_,_) = A*x + Bu*u
h(x,_,_) = C*x

function f2!(xnext, x, u, d, (A, Bu, Bd, C, Dd))
    mul!(xnext, A,  x)
    mul!(xnext, Bu, u, 1, 1)
    mul!(xnext, Bd, d, 1, 1)
    return xnext
end
function h2!(y, x, d, (A, Bu, Bd, C, Dd))
    mul!(y, C,  x)
    mul!(y, Dd, d, 1, 1)
    return y
end

nonLinModel1 = NonLinModel{Float32}(f,h,Ts,2,4,2,solver=nothing)
nonLinModel2 = setop!(
    NonLinModel(f2!,h2!,Ts,2,4,2,1,p=(A, Bu, Bd, C, Dd), solver=nothing), 
    uop=[10,10],yop=[50,30],dop=[5]
    )
#=
nonLinModel2 = setop!(NonLinModel(
    (x,u,d)->A*x+Bu*u+Bd*d, 
    (x,d)  ->C*x+Dd*d, 
    Ts , 2, 4, 2, 1), uop=[10,10], yop=[50,30], dop=[5]
)
=#
f3!, h3! = let A=[0 0.5; -0.2 -0.1], Bu=[0; 0.5], Bd=[0; 0.5], C=[0.4 0], Dd=[0]
    function f!(dx, x, u, d, _ )
        mul!(dx, A, x)
        mul!(dx, Bu, u, 1, 1)
        mul!(dx, Bd, d, 1, 1)
        return nothing
    end
    function h!(y, x, d, _ )
        mul!(y, C, x)
        mul!(y, Dd, d, 1, 1)
        return nothing
    end
    f!, h!
end
nonLinModel3 = NonLinModel(f3!, h3!, 1.0, 1, 2, 1, 1, solver=RungeKutta(4, supersample=2))

plot(sim!(nonLinModel3, 100))

linearizeModel = linearize(nonLinModel3, x=[1,1], u=[1], d=[1])


# @btime linearize!($linearizeModel, $nonLinModel3, x=$[2,2], u=$[1], d=$[1])
@profview for i=1:10000; linearize!(linearizeModel, nonLinModel3, x=[2,2], u=[1], d=[1]); end;


internalModel1 = InternalModel(linModel1)
internalModel2 = InternalModel(linModel1,stoch_ym=[tf([1,0],[1,-1],Ts) 0; 0 tf([1,0],[1,-1],Ts)])
internalModel3 = InternalModel(linModel1,i_ym=[2])
internalModel3 = InternalModel(nonLinModel2)


initstate!(internalModel1,[0,0],[1,1])

preparestate!(internalModel1, [0, 0])
updatestate!(internalModel1, [1, 1], [1, 1])

luenberger = Luenberger(linModel1)

preparestate!(luenberger, [0,0])
updatestate!(luenberger, [1,1], [1,1])


mpcluen = LinMPC(
    luenberger, 
    M_Hp=Diagonal(collect(1.01:0.01:1.2)), 
    N_Hc=Diagonal([0.1,0.11,0.12,0.13]), 
    L_Hp=Diagonal(collect(0.001:0.001:0.02))
)

initstate!(mpcluen, [0,0], [0,0])

preparestate!(mpcluen, [0,0])
moveinput!(mpcluen, [10, 10])

mpc_ms = LinMPC(linModel4, transcription=MultipleShooting(), Hp=3, Hc=3)




mhe = MovingHorizonEstimator(nonLinModel2, He=5, Cwt=1e5, direct=true)

unset_time_limit_sec(mhe.optim)
#set_attribute(mhe.optim, "nlp_scaling_max_gradient", 1/mhe.C*10.0)# 5*1e-5)
#set_attribute(mhe.optim, "nlp_scaling_constr_target_gradient", 1e-4)
#set_attribute(mhe.optim, "nlp_scaling_obj_target_gradient"   , 100*1e-4)
mhe = setconstraint!(mhe, x̂min=[-10,-10,-10,-10,-10,-10], X̂max=[1;1;1;1;10;10;fill(1.5,6*5)])
mhe = setconstraint!(mhe, V̂min=-1.1:-0.1:-2.0,v̂max=[0.1,0.1])
#mhe = setconstraint!(mhe, ŵmin=fill(-0.1,6))

initstate!(mhe, [11, 11], [50, 30], [5])

res_mhe1 = sim!(mhe, 30, x_0=[0,0,0,0], x̂_0=[0,0,0,0,0,0])
@time sim!(mhe, 30, x_0=[0,0,0,0], x̂_0=[0,0,0,0,0,0])
info = getinfo(mhe)
p = plot(res_mhe1, plotd=false, plotu=false, plotxwithx̂=true, plotx̂min=false)
display(p)


mhe2 = MovingHorizonEstimator(linModel4, He=5, direct=true)

setmodel!(mhe2, linModel4)

unset_time_limit_sec(mhe2.optim)
mhe2 = setconstraint!(mhe2, x̂min=[-10,-10,-10,-10,-10,-10], X̂max=[1;1;1;1;10;10;fill(1.5,6*5)])
mhe2 = setconstraint!(mhe2, V̂min=-1.1:-0.1:-2.0,v̂max=[0.1,0.1])
#mhe2 = setconstraint!(mhe2, ŵmin=fill(-0.1,6))

preparestate!(mhe2, [50, 30], [5])
updatestate!(mhe2, [10, 11], [50, 30], [5])



res_mhe2 = sim!(mhe2, 30, x_0=[0,0,0,0], x̂_0=[0,0,0,0,0,0])
@time sim!(mhe2, 30, x_0=[0,0,0,0], x̂_0=[0,0,0,0,0,0])
display(getinfo(mhe2))
p = plot(res_mhe2, plotd=false, plotu=false, plotxwithx̂=true, plotx̂min=false)
display(p)



mpcexplicit = ExplicitMPC(LinModel(append(tf(3,[2, 1]), tf(2, [6, 1])), 0.1), Hp=10000, Hc=1)
preparestate!(mpcexplicit, [0, 0])
moveinput!(mpcexplicit, [10, 10])

kalmanFilter1 = KalmanFilter(linModel1)
setmodel!(kalmanFilter1, linModel1)


kalmanFilter2 = KalmanFilter(linModel1,nint_ym=0,direct=false)

updatestate!(kalmanFilter2,[1, 1],[1, 1])

initstate!(kalmanFilter1,[0,0],[2,1])

ssKalmanFilter1 = SteadyKalmanFilter(linModel1)

preparestate!(ssKalmanFilter1, [0, 0])
updatestate!(ssKalmanFilter1, [1, 1], [1, 1])


ssKalmanFilter2 = SteadyKalmanFilter(linModel1,nint_ym=0,direct=false)
snxX̂sKalmanFilter3 = SteadyKalmanFilter(linModel1,i_ym=[2],nint_u=[0, 1], nint_ym=[1])

updatestate!(ssKalmanFilter2,[1, 1],[1,1])

initstate!(ssKalmanFilter1,[0,0],[2,1])

extKalmanFilter = ExtendedKalmanFilter(nonLinModel2, direct=false)

initstate!(extKalmanFilter, [10,10], [50,30], [5]) 
updatestate!(extKalmanFilter, [10,11], [50,30], [5])

mpc_t = setconstraint!(LinMPC(LinModel(ss(0.5, 0.5, 1, 0, 1.0)), Hp=5), x̂min=[-1e-3,-Inf], x̂max=[1e-3,+Inf])
preparestate!(mpc_t, [0])
moveinput!(mpc_t, [100])
println(getinfo(mpc_t)[:Ŷ])

f_t(x,u,_,_) = 0.5*x + 0.5*u
h_t(x,_,_)   = x
nmpc_t = setconstraint!(
    NonLinMPC(NonLinModel(f_t,h_t,0.001, 1, 1, 1, solver=nothing), Cwt=Inf, Hp=5), 
    x̂min=[-1e-3,-Inf], 
    x̂max=[1e-3,+Inf]
)

preparestate!(nmpc_t, [0])
moveinput!(nmpc_t, [100])
println(getinfo(nmpc_t)[:Ŷ])


uscKalmanFilter1 = UnscentedKalmanFilter(linModel1, nint_u=[1, 1])

preparestate!(uscKalmanFilter1, [0, 0])
updatestate!(uscKalmanFilter1,[0,0],[2,1])

initstate!(uscKalmanFilter1,[0,0],[2,1])

nmpc1 = NonLinMPC(uscKalmanFilter1)


nmpc2 = NonLinMPC(nonLinModel2, Hp=15, Hc=1, Mwt=[1, 1] , Nwt=[0.1, 0.1], Cwt=1e5)


setconstraint!(nmpc2, c_umin=[0,0], c_umax=[0,0])
setconstraint!(nmpc2, c_ymin=[1,1], c_ymax=[1,1])
setconstraint!(nmpc2, umin=[5, 9.9], umax=[Inf,Inf])
setconstraint!(nmpc2, ymin=[-Inf,-Inf], ymax=[55, 35])
setconstraint!(nmpc2, Δumin=[-Inf,-Inf],Δumax=[+Inf,+Inf])


nx = linModel4.nx
kf = KalmanFilter(linModel2, σP_0=10*ones(nx), σQ=0.01*ones(nx), σR=[0.1, 0.1], σQint_ym=0.05*ones(2), σPint_ym_0=10*ones(2))

setmodel!(kf, linModel4)

mpc = LinMPC(kf, Hp=15, Hc=2, Mwt=[1, 1], Cwt=1e5)#, optim=Model(DAQP.Optimizer, add_bridges=false))


setconstraint!(mpc, c_umin=[0,0], c_umax=[0,0])
setconstraint!(mpc, c_ymin=[1,1], c_ymax=[1,1])
setconstraint!(mpc, umin=[5, 9.9], umax=[Inf,Inf])
setconstraint!(mpc, Δumin=[-Inf,-Inf],Δumax=[+Inf,+Inf])
setconstraint!(mpc, ymin=[-Inf,-Inf], ymax=[1000, 1000])

setmodel!(mpc, linModel4)

preparestate!(mpc, mpc.estim.model.yop, mpc.estim.model.dop)
moveinput!(mpc, [60, 40], mpc.estim.model.dop)
info = getinfo(mpc)

setconstraint!(mpc, ymax=[55, 35])
moveinput!(mpc, [60, 40], mpc.estim.model.dop)
info = getinfo(mpc)


empc = ExplicitMPC(linModel4, Hp=15, Hc=1, Mwt=[1, 1] , Nwt=[0.1, 0.1])
sim!(empc, empc.Hp+10, x_0=[0,0,0,0])
@time sim!(empc, empc.Hp+10, x_0=[0,0,0,0])

nmpc = NonLinMPC(kf, Hp=15, Hc=1, Mwt=[1, 1] , Nwt=[0.1, 0.1], Cwt=1e5, Ewt=0.0)

setconstraint!(nmpc, c_umin=[0,0], c_umax=[0,0])
setconstraint!(nmpc, c_ymin=[1,1], c_ymax=[1,1])
setconstraint!(nmpc, umin=[5, 9.9], umax=[Inf,Inf])
setconstraint!(nmpc, ymin=[-Inf,-Inf], ymax=[55, 35])
setconstraint!(nmpc, Δumin=[-Inf,-Inf],Δumax=[+Inf,+Inf])

preparestate!(nmpc, nmpc.estim.model.yop, nmpc.estim.model.dop)
moveinput!(nmpc, nmpc.estim.model.yop + [0, 2], nmpc.estim.model.dop)
display(getinfo(nmpc))

if Sys.free_memory()/2^30 < 6.0
    GC.gc()
end

function test_mpc(model, mpc, Nrepeat=1)
    N = 200 
    u_data = zeros(model.nu,N)
    y_data = zeros(model.ny,N)
    r_data = zeros(model.ny,N)
    d_data = zeros(model.nd,N)
    for j=1:Nrepeat
        u = model.uop
        d = model.dop
        r = [50,32]
        setstate!(model, zeros(model.nx))
        setstate!(mpc, zeros(mpc.estim.nx̂))
        initstate!(mpc,u,model(d),d)
        for k = 0:N-1
            if k == 40
                r[2] = 29
            end
            if k == 100
                r[2] = 36
            end
            if k == 150
                d = [3]
            end
            y = model(d)
            if k ≥ 180
                y[1] += 15
            end
            preparestate!(mpc, y, d)
            u = moveinput!(mpc, r, d)
            u_data[:,k+1] = u
            y_data[:,k+1] = y
            r_data[:,k+1] = r
            d_data[:,k+1] = d
            updatestate!(mpc, u, y, d)
            updatestate!(model, u, d)
        end
    end
    return u_data, y_data, r_data, d_data
end

test_mpc(linModel4 , mpc)
@btime test_mpc($linModel4 , $mpc, 100)

@profview test_mpc(linModel4, mpc, 2500)
@profview test_mpc(nonLinModel2, nmpc2, 3)

resM = sim!(nonLinModel2, mpc.Hp+10, [1,-1])
psM  = plot(resM)
display(psM)

res = sim!(mpc, mpc.Hp+10, x_0=[0,0,0,0])
ps = plot(res, plotx=true)
display(ps)

res_manual = SimResult(res.obj, res.U_data, res.Y_data; Ry_data=res.Ry_data)
ps_manual = plot(res_manual)
display(ps_manual)

res2 = sim!(uscKalmanFilter1, mpc.Hp+10)
# @time sim!(uscKalmanFilter1, mpc.Hp+10)
ps2 = plot(res2, plotxwithx̂=true)
display(ps2)

test_mpc(linModel4, nmpc)
@time u_data, y_data, r_data, d_data = test_mpc(linModel4, nmpc)

test_mpc(nonLinModel2, nmpc2)
@time u_data, y_data, r_data, d_data = test_mpc(nonLinModel2, nmpc2)

N = size(r_data, 2) 
p1 = plot(0:N-1,y_data[1,:],label=raw"$y_1$")
plot!(0:N-1,r_data[1,:],label=raw"$r_1$",linestyle=:dash, linetype=:steppost)
p2 = plot(0:N-1,y_data[2,:],label=raw"$y_2$")
plot!(0:N-1,r_data[2,:],label=raw"$r_2$",linestyle=:dash, linetype=:steppost)
py = plot(p1,p2, layout=[1,1])

p1 = plot(0:N-1,u_data[1,:],label=raw"$u_1$",linetype=:steppost)
p2 = plot(0:N-1,u_data[2,:],label=raw"$u_2$",linetype=:steppost)
pu = plot(p1,p2, layout=[1,1])

pd = plot(0:N-1,d_data[1,:],label=raw"$d_1$")

display(pd)
display(pu)
display(py)




using ModelPredictiveControl, JuMP, Plots
function f!(ẋ, x, u, _ , p)
    g, L, K, m = p          # [m/s²], [m], [kg/s], [kg]
    θ, ω = x[1], x[2]       # [rad], [rad/s]
    τ  = u[1]               # [Nm]
    ẋ[1] = ω
    ẋ[2] = -g/L*sin(θ) - K/m*ω + τ/m/L^2
    return nothing
end
h!(y, x, _ , _ ) = (y[1] = 180/π*x[1]; nothing) # [°]
p = [9.8, 0.4, 1.2, 0.3]
nu, nx, ny, Ts = 1, 2, 1, 0.1
vu, vx, vy = ["\$τ\$ (Nm)"], ["\$θ\$ (rad)", "\$ω\$ (rad/s)"], ["\$θ\$ (°)"]

model = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny; p); u=vu, x=vx, y=vy)

p_plant = copy(p); p_plant[3] = p[3]*1.25
plant = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny; p=p_plant); u=vu, x=vx, y=vy)

N = 35

sim!(model, N) |> plot |> display


Hp, Hc, Mwt, Nwt = 20, 2, [0.5], [2.5]
α=0.01; σQ=[0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
σQint_ym = zeros(0)

function gc!(LHS, Ue, Ŷe, D̂e, p, ϵ)
    Hp, umax = p
    for i = 1:Hp+1
        LHS[i] = Ue[i] - umax[1]
    end
    return nothing
end


function gc(Ue, Ŷe, D̂e, p, ϵ)
    LHS = similar(Ue)
    Hp, umax = p
    for i = 1:Hp+1
        LHS[i] = Ue[i] - umax[1]
    end
    return LHS
end


umin, umax = [-1.5], [+1.5]
nc = Hp+1
transcription = MultipleShooting()#SingleShooting()#TrapezoidalCollocation(f_threads=true)
using UnoSolver
optim = Model(()->UnoSolver.Optimizer(preset="filtersqp")); 
oracle = true
using DifferentiationInterface, SparseConnectivityTracer, SparseMatrixColorings
import ForwardDiff
hessian = AutoSparse(
    AutoForwardDiff(); 
    sparsity_detector  = TracerSparsityDetector(), 
    coloring_algorithm = GreedyColoringAlgorithm( decompression=:substitution)
)
#nint_u = 0
#σQint_u = zeros(0)
#σQint_ym = [0.1]

nmpc = NonLinMPC(model; 
    Hp, Hc, Mwt, Nwt, Cwt=Inf, p=(Hp,umax), transcription, oracle, hessian, optim,
    α, σQ, σR, nint_u, σQint_u, σQint_ym
)
nmpc = setconstraint!(nmpc; umin, umax)
unset_time_limit_sec(nmpc.optim)

res_ry = sim!(nmpc, N, [180.0]; plant, x_0=[0, 0], x̂_0=[0, 0, 0])
display(plot(res_ry))
#@benchmark sim!($nmpc, $N, $[180.0]; plant=$plant, x_0=$[0, 0], x̂_0=$[0, 0, 0]) samples=50 seconds=1*60
@profview sim!(nmpc, 1, [180.0]; plant=plant, x_0=[0, 0], x̂_0=[0, 0, 0])


#TODO: investigate the numerical error with debug log and find if its related to UKF


#=
mhe = MovingHorizonEstimator(model, He=2, nint_ym=0)
unset_time_limit_sec(mhe.optim)
mhe = setconstraint!(mhe, v̂min=[-1], v̂max=[1])
using Logging; debuglogger = ConsoleLogger(stderr, Logging.Debug)
res_mhe = #with_logger(debuglogger) do
sim!(mhe, N, [1.0]; plant=plant, x_0=[0, 0], x̂_0=[0, 0])
#end
@time sim!(mhe, N, [1.0]; plant=plant, x_0=[0, 0], x̂_0=[0, 0])
display(plot(res_mhe))



preparestate!(nmpc, [0])
u = moveinput!(nmpc, [180])
updatestate!(nmpc, u, [0])
preparestate!(nmpc, [0])



linModelA = LinModel(ss(0.5, 0.5, 1, 0, 1.0))


function gcLin!(LHS, Ue, Ŷe, D̂e, p, ϵ)
    Hp = p
    ymax = 1.0
    for i = 1:Hp+1
        LHS[i] = Ŷe[i] - ymax[1]
    end
    return nothing
end

Hp=5
nmpcA = NonLinMPC(linModelA, Hp=Hp, p=Hp, gc=gcLin!, nc=Hp+1)
unset_time_limit_sec(nmpcA.optim)

preparestate!(nmpcA, [0])
u = moveinput!(nmpcA, [1.0])
updatestate!(nmpcA, u, [0])
preparestate!(nmpcA, [0])

res_ryA = sim!(nmpcA, N, [2.0])
display(plot(res_ryA))

#@btime sim!($nmpcA, $N, $[2.0]) samples=50 seconds=10*60

nothing
=#