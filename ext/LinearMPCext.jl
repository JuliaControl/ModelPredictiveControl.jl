module LinearMPCext

using ModelPredictiveControl, LinearMPC

using JuMP

function Base.convert(::Type{LinearMPC.MPC}, mpc::ModelPredictiveControl.LinMPC)
    model, estim, weights = mpc.estim.model, mpc.estim, mpc.weights
    validate_compatibility(mpc)

    F, G, Gd = estim.Â, estim.B̂u, estim.B̂d
    C, Dd = estim.Ĉ, estim.D̂d
    Np = mpc.Hp
    Nc = mpc.Hc


    newmpc = LinearMPC.MPC(F, G; Gd, C, Dd, Np, Nc)


    Q, R = estim.cov.Q̂, estim.cov.R̂
    C = estim.Ĉm
    set_state_observer!(newmpc; C, Q, R)

    Q  = weights.M_Hp[1:model.ny, 1:model.ny]
    Qf = weights.M_Hp[end-model.ny+1:end, end-model.ny+1:end]
    Rr = weights.Ñ_Hc[1:model.nu, 1:model.nu]
    R  = weights.L_Hp[1:model.nu, 1:model.nu]
    soft_weight = weights.Ñ_Hc[end, end]
    
    LinearMPC.set_objective!(newmpc; Q, Rr, R, Qf)
    !weights.isinf_C && (newmpc.settings.soft_weight = soft_weight)

    Umin, Umax = mpc.con.U0min + mpc.Uop, mpc.con.U0max + mpc.Uop
    Ymin, Ymax = mpc.con.Y0min + mpc.Yop, mpc.con.Y0max + mpc.Yop
    
    umin, umax = Umin[1:model.nu], Umax[1:model.nu]
    ymin, ymax = Ymin[1:model.ny], Ymax[1:model.ny]

    LinearMPC.set_bounds!(newmpc; umin, umax, ymin, ymax)

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
    nΔU = mpc.Hc*mpc.estim.model.nu
    if any(isfinite, mpc.con.ΔŨmin[1:nΔU]) || any(isfinite, mpc.con.ΔŨmin[1:nΔU])
        error("LinearMPC.MPC does not support constraints on input increments Δu")
    end
    return nothing
end

LinearMPC.MPC(mpc::ModelPredictiveControl.LinMPC) = convert(LinearMPC.MPC, mpc)


end # LinearMPCext