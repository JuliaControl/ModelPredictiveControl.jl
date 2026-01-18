@testitem "LinearMPCext extension" setup=[SetupMPCtests] begin
    using .SetupMPCtests, ControlSystemsBase, LinearAlgebra, JuMP, DAQP
    import LinearMPC
    model = LinModel(sys, Ts, i_u=1:2)
    model = setop!(model, uop=[20, 20], yop=[50, 30])
    optim = JuMP.Model(DAQP.Optimizer)
    mpc1 = LinMPC(model, Hp=15, Hc=[2, 3, 10], optim=optim)
    mpc1 = setconstraint!(mpc1, ymin=[48, -Inf], umax=[Inf, 30])
    mpc2 = LinearMPC.MPC(mpc1)
    function sim_both(model, mpc1, mpc2, N)
        r = [55.0; 30.0]
        u1 = [20.0, 20.0]
        u2 = [20.0, 20.0]
        model.x0 .= 0
        u_data1, u_data2 = zeros(model.nu, N), zeros(model.nu, N)
        for k in 0:N-1
            k == 10 && (r .= [45; 30.0])
            k == 25 && (r .= [50; 45.0])
            y = model()
            preparestate!(mpc1, y)
            x̂ = LinearMPC.correct_state!(mpc2, y)
            u1 = moveinput!(mpc1, r)
            u2 = LinearMPC.compute_control(mpc2, x̂, r=r, uprev=u2)
            u_data1[:, k+1], u_data2[:, k+1] = u1, u2
            updatestate!(model, u1)
            updatestate!(mpc1, u1, y)
            LinearMPC.predict_state!(mpc2, u2)
        end
        return u_data1, u_data2
    end
    N = 50
    u_data1, u_data2 = sim_both(model, mpc1, mpc2, N)
    @test u_data1 ≈ u_data2 atol=1e-3 rtol=1e-3 # looser tol due to different softening

    mpc1_hard = LinMPC(model, Hp=15, Cwt=Inf, optim=optim)
    mpc1_hard = setconstraint!(mpc1_hard, ymin=[48, -Inf], umax=[Inf, 30])
    mpc2_hard = LinearMPC.MPC(mpc1_hard)
    u_data1_hard, u_data2_hard = sim_both(
        model, mpc1_hard, mpc2_hard, N
    )
    @test u_data1_hard ≈ u_data2_hard atol=1e-12 # exact match for hard constraints

    mpc_ms = LinMPC(model; transcription=MultipleShooting(), optim)
    @test_throws ErrorException LinearMPC.MPC(mpc_ms)
    mpc_kf = LinMPC(KalmanFilter(model, direct=false); optim)
    @test_throws ErrorException LinearMPC.MPC(mpc_kf)
    mpc_osqp = LinMPC(model)
    @test_logs(
        (:warn, "LinearMPC relies on DAQP, and the solver in the mpc object is currently "*
        "OSQP.\nThe results in closed-loop may be different."),
        LinearMPC.MPC(mpc_osqp)
    )
end
