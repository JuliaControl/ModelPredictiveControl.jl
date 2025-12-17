module LinearMPCext

using ModelPredictiveControl, LinearMPC
using LinearAlgebra
using JuMP

function Base.convert(::Type{LinearMPC.MPC}, mpc::ModelPredictiveControl.LinMPC)
    model, estim, weights = mpc.estim.model, mpc.estim, mpc.weights
    nu, ny, nd = model.nu, model.ny, model.nd
    validate_compatibility(mpc)

    F, G, Gd = estim.Â, estim.B̂u, estim.B̂d
    C, Dd = estim.Ĉ, estim.D̂d
    Np = Hp = mpc.Hp
    Nc = Hc = mpc.Hc

    newmpc = LinearMPC.MPC(F, G; Gd, C, Dd, Np, Nc)

    Q, R = estim.cov.Q̂, estim.cov.R̂
    set_state_observer!(newmpc; C=estim.Ĉm, Q, R)

    Q  = weights.M_Hp[1:ny, 1:ny]
    Qf = weights.M_Hp[end-ny+1:end, end-ny+1:end]
    Rr = weights.Ñ_Hc[1:nu, 1:nu]
    R  = weights.L_Hp[1:nu, 1:nu]
    soft_weight = weights.Ñ_Hc[end, end]
    
    LinearMPC.set_objective!(newmpc; Q, Rr, R, Qf)
    !weights.isinf_C && (newmpc.settings.soft_weight = soft_weight)

    Umin, Umax = mpc.con.U0min + mpc.Uop, mpc.con.U0max + mpc.Uop
    Ymin, Ymax = mpc.con.Y0min + mpc.Yop, mpc.con.Y0max + mpc.Yop
    C_u = -mpc.con.A_Umin[:, end]
    C_y = -mpc.con.A_Ymin[:, end]
    # ymin_k, y_max_k = Ymin[(k-1)*ny+1:k*ny], Ymax[(k-1)*ny+1:k*ny]
    for k in 0:Hp-1    
        umin_k, u_max_k = Umin[k*nu+1:(k+1)*nu], Umax[k*nu+1:(k+1)*nu]
        c_u_k = C_u[k*nu+1:(k+1)*nu]
        ks = [k]
        Au_k = Matrix{Float64}(I, nu, nu)
        for i in 1:nu
            lb, ub = [umin_k[i]], [u_max_k[i]]
            soft = (c_u_k[i] ≈ 1)
            Au = Au_k[i:i, :]
            add_constraint!(newmpc; Au, lb, ub, ks, soft)
        end
    end


    #umin, umax = Umin[1:model.nu], Umax[1:model.nu]
    #ymin, ymax = Ymin[1:model.ny], Ymax[1:model.ny]

    #LinearMPC.set_bounds!(newmpc; umin, umax, ymin, ymax)

    return newmpc
end

function validate_compatibility(mpc::ModelPredictiveControl.LinMPC)
    if mpc.transcription isa MultipleShooting
        error("LinearMPC.MPC only supports SingleShooting transcription.")
    end
    if !(mpc.estim isa SteadyKalmanFilter) || !mpc.estim.direct
        error("LinearMPC.MPC only supports SteadyKalmanFilter with direct=true option.")
    end
    if JuMP.solver_name(mpc.optim) != "DAQP"
        @warn "LinearMPC.MPC relies on DAQP, and the solver in the mpc object "*
              "is currently $(JuMP.solver_name(mpc.optim)).\n"*
              "The results in closed-loop may be different."
    end
    validate_weights(mpc)
    validate_constraints(mpc)
    return nothing
end

function validate_weights(mpc::ModelPredictiveControl.LinMPC)
    ny, nu = mpc.estim.model.ny, mpc.estim.model.nu
    M_Hp, N_Hc, L_Hp = mpc.weights.M_Hp, mpc.weights.Ñ_Hc, mpc.weights.L_Hp
    M_1, N_1, L_1 = M_Hp[1:ny, 1:ny], N_Hc[1:nu, 1:nu], L_Hp[1:nu, 1:nu]
    for i in 2:mpc.Hp-1 # last block is terminal weight, can be different
        M_i = M_Hp[(i-1)*ny+1:i*ny, (i-1)*ny+1:i*ny]
        if !isapprox(M_i, M_1)
            error("LinearMPC.MPC only supports identical weights for each stages in M_Hp.")
        end
    end
    for i in 2:mpc.Hc
        N_i = N_Hc[(i-1)*nu+1:i*nu, (i-1)*nu+1:i*nu]
        if !isapprox(N_i, N_1)
            error("LinearMPC.MPC only supports identical weights for each stages in Ñ_Hc.")
        end
    end
    for i in 2:mpc.Hp
        L_i = L_Hp[(i-1)*nu+1:i*nu, (i-1)*nu+1:i*nu]
        if !isapprox(L_i, L_1)
            error("LinearMPC.MPC only supports identical weights for each stages in L_Hp.")
        end
    end
    return nothing
end

function validate_constraints(mpc::ModelPredictiveControl.LinMPC)
    ΔŨmin, ΔŨmax = mpc.con.ΔŨmin, mpc.con.ΔŨmax
    C_umin, C_umax = -mpc.con.A_Umin[:, end], -mpc.con.A_Umax[:, end]
    C_ymin, C_ymax = -mpc.con.A_Ymin[:, end], -mpc.con.A_Ymax[:, end]
    is0or1(C) = all(x -> x ≈ 0 || x ≈ 1, C)
    if !is0or1(C_umin) || !is0or1(C_umax) || !is0or1(C_ymin) || !is0or1(C_ymax) 
        error("LinearMPC.MPC does not support softness parameters c ≠ 0 or 1.")
    end
    if !isapprox(C_umin, C_umax) || !isapprox(C_ymin, C_ymax)
        error("LinearMPC.MPC does not support different softness parameters for lower and upper bounds.")
    end
    nΔU = mpc.Hc*mpc.estim.model.nu
    if any(isfinite, ΔŨmin[1:nΔU]) || any(isfinite, ΔŨmax[1:nΔU])
        error("LinearMPC.MPC does not support constraints on input increments Δu")
    end
    return nothing
end

LinearMPC.MPC(mpc::ModelPredictiveControl.LinMPC) = convert(LinearMPC.MPC, mpc)

end # LinearMPCext