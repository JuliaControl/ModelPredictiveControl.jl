using LinearMPC
using ModelPredictiveControl

using LinearMPC
# Continuous time system dx = A x + B u, y = C x
A = [0 1 0 0; 0 -10 9.81 0; 0 0 0 1; 0 -20 39.24 0]; 
B = 100*[0;1.0;0;2.0;;];
C = [1.0 0 0 0; 0 0 1.0 0];


# create an MPC control with sample time 0.01, prediction/control horizon 50/5
Ts = 0.01
mpc = LinearMPC.MPC(A,B,Ts;C,Np=50,Nc=5);

# ...existing code...
using LinearMPC
using ModelPredictiveControl

# Continuous time system dx = A x + B u, y = C x
A = [0 1 0 0; 0 -10 9.81 0; 0 0 0 1; 0 -20 39.24 0]; 
B = 100*[0;1.0;0;2.0];
C = [1.0 0 0 0; 0 0 1.0 0];

# create an MPC control with sample time 0.01, prediction/control horizon 50/5
Ts = 0.01
mpc = LinearMPC.MPC(A,B,Ts; C=C, Np=50, Nc=5);

# set the objective functions weights
set_objective!(mpc; Q=[1.2^2,1], R=[0.0], Rr=[1.0])

# set actuator limits
set_bounds!(mpc; umin=[-2.0],umax=[2.0])

u = compute_control(mpc,[0,0,0,0], r = [1, 0])


Base.convert(::Type{LinearMPC.MPC}, mpc::ModelPredictiveControl.LinMPC) = begin
    model, weights = mpc.estim.model, mpc.weights

    A = mpc.estim.Â
    B = mpc.estim.B̂u
    Ts = model.Ts

    C = mpc.estim.Ĉ
    Np = mpc.Hp
    Nc = mpc.Hc


    newmpc = LinearMPC.MPC(A, B, Ts; C, Np, Nc)


    Q = weights.M_Hp[1:model.ny, 1:model.ny]
    Rr = weights.Ñ_Hc[1:model.nu, 1:model.nu]
    R = weights.L_Hp[1:model.nu, 1:model.nu]
    # Qf = weights.M_Hp[end-model.ny+1:end, end-model.ny+1:end]

    LinearMPC.set_objective!(newmpc; Q, Rr, R)


    Umin, Umax = mpc.con.U0min + mpc.Uop, mpc.con.U0max + mpc.Uop
    Ymin, Ymax = mpc.con.Y0min + mpc.Yop, mpc.con.Y0max + mpc.Yop
    
    umin, umax = Umin[1:model.nu], Umax[1:model.nu]
    ymin, ymax = Ymin[1:model.ny], Ymax[1:model.ny]

    LinearMPC.set_bounds!(newmpc; umin, umax, ymin, ymax)

    return newmpc
end

using ModelPredictiveControl, ControlSystemsBase
G = [ tf( 2 , [10, 1])*delay(20)
      tf( 10, [4,  1]) ]
Ts = 1.0
model = LinModel(G, Ts)

mpc = LinMPC(model, Mwt=[1, 0], Nwt=[0.1])
mpc = setconstraint!(mpc, ymax=[Inf, 35])

mpc2 = convert(LinearMPC.MPC, mpc)

sim = LinearMPC.Simulation(mpc2;x0=zeros(24),r=[5,0],N=10)

using PlotThemes, Plots
#theme(:default)
theme(:dark)
default(fontfamily="Computer Modern"); scalefontsizes(1.1)
plot(sim)
