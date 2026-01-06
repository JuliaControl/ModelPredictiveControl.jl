module LinearMPCext

using ModelPredictiveControl, LinearMPC
using LinearAlgebra, SparseArrays
using JuMP

import ModelPredictiveControl: isblockdiag

function Base.convert(::Type{LinearMPC.MPC}, mpc::ModelPredictiveControl.LinMPC)
    model, estim, weights = mpc.estim.model, mpc.estim, mpc.weights
    nu, ny, nd, nx̂ = model.nu, model.ny, model.nd, estim.nx̂
    validate_compatibility(mpc)
    # --- Model parameters ---
    F, G, Gd = estim.Â, estim.B̂u, estim.B̂d
    C, Dd = estim.Ĉ, estim.D̂d
    Np = Hp = mpc.Hp
    Nc = Hc = mpc.Hc
    newmpc = LinearMPC.MPC(F, G; Gd, C, Dd, Np, Nc)
    # --- Operating points ---
    xo = estim.x̂op
    uo = model.uop
    yo = model.yop
    !iszero(yo) && error("LinearMPC does not support non-zero output operating points yop.")
    if !iszero(model.dop)
        @warn "LinearMPC does not support measured disturbance operating points dop.\n" *
              "Ensure to subtract the operating point from the measurement at each time "*
              "step before solving the MPC problem."
    LinearMPC.set_operating_point!(newmpc; xo, uo, relinearize=false)
    # --- State observer parameters ---
    Q, R = estim.cov.Q̂, estim.cov.R̂
    set_state_observer!(newmpc; C=estim.Ĉm, Q, R)
    # --- Objective function weights ---
    Q = weights.M_Hp[1:ny, 1:ny]
    Qf = weights.M_Hp[end-ny+1:end, end-ny+1:end]
    Rr = weights.Ñ_Hc[1:nu, 1:nu]
    R = weights.L_Hp[1:nu, 1:nu]
    LinearMPC.set_objective!(newmpc; Q, Rr, R, Qf)
    if !weights.isinf_C
        Cwt = weights.Ñ_Hc[end, end]
        newmpc.settings.soft_weight = Cwt
    end
    # --- Custom move blocking ---
    LinearMPC.move_block!(newmpc, mpc.nb) # un-comment when debugged
    # --- Manipulated inputs constraints ---
    Umin, Umax = mpc.con.U0min + mpc.Uop, mpc.con.U0max + mpc.Uop
    C_u = -mpc.con.A_Umin[:, end]
    I_u = Matrix{Float64}(I, nu, nu)
    # add_constraint! does not support u bounds pass the control horizon Hc
    # so we compute the extremum bounds from k=Hc-1 to Hp, and apply them at k=Hc-1
    Umin_finals = reshape(Umin[nu*(Hc-1)+1:end], nu, :)
    Umax_finals = reshape(Umax[nu*(Hc-1)+1:end], nu, :)
    umin_end = mapslices(maximum, Umin_finals; dims=2)
    umax_end = mapslices(minimum, Umax_finals; dims=2)
    for k in 0:Hc-1
        if k < Hc - 1
            umin_k, umax_k = Umin[k*nu+1:(k+1)*nu], Umax[k*nu+1:(k+1)*nu]
        else
            umin_k, umax_k = umin_end, umax_end
        end
        c_u_k = C_u[k*nu+1:(k+1)*nu]
        ks = [k + 1] # a `1` in ks argument corresponds to the present time step k+0
        for i in 1:nu
            lb = isfinite(umin_k[i]) ? [umin_k[i]] : zeros(0)
            ub = isfinite(umax_k[i]) ? [umax_k[i]] : zeros(0)
            soft = (c_u_k[i] > 0)
            Au = I_u[i:i, :]
            add_constraint!(newmpc; Au, lb, ub, ks, soft)
        end
    end
    # --- Output constraints ---
    Ymin, Ymax = mpc.con.Y0min + mpc.Yop, mpc.con.Y0max + mpc.Yop
    C_y = -mpc.con.A_Ymin[:, end]
    for k in 1:Hp
        ymin_k, ymax_k = Ymin[(k-1)*ny+1:k*ny], Ymax[(k-1)*ny+1:k*ny]
        c_y_k = C_y[(k-1)*ny+1:k*ny]
        ks = [k + 1] # a `1` in ks argument corresponds to the present time step k+0
        for i in 1:ny
            lb = isfinite(ymin_k[i]) ? [ymin_k[i]] : zeros(0)
            ub = isfinite(ymax_k[i]) ? [ymax_k[i]] : zeros(0)
            soft = (c_y_k[i] > 0)
            Ax, Ad = C[i:i, :], Dd[i:i, :]
            add_constraint!(newmpc; Ax, Ad, lb, ub, ks, soft)
        end
    end
    # --- Terminal constraints ---
    x̂min, x̂max = mpc.con.x̂0min + estim.x̂op, mpc.con.x̂0max + estim.x̂op
    c_x̂ = -mpc.con.A_x̂min[:, end]
    I_x̂ = Matrix{Float64}(I, nx̂, nx̂)
    ks = [Hp + 1] # a `1` in ks argument corresponds to the present time step k+0
    for i in 1:nx̂
        lb = isfinite(x̂min[i]) ? [x̂min[i]] : zeros(0)
        ub = isfinite(x̂max[i]) ? [x̂max[i]] : zeros(0)
        soft = (c_x̂[i] > 0)
        Ax = I_x̂[i:i, :]
        add_constraint!(newmpc; Ax, lb, ub, ks, soft)
    end
    return newmpc
end

function validate_compatibility(mpc::ModelPredictiveControl.LinMPC)
    if mpc.transcription isa MultipleShooting
        error("LinearMPC only supports SingleShooting transcription.")
    end
    if !(mpc.estim isa SteadyKalmanFilter) || !mpc.estim.direct
        error("LinearMPC only supports SteadyKalmanFilter with direct=true option.")
    end
    if JuMP.solver_name(mpc.optim) != "DAQP"
        @warn "LinearMPC relies on DAQP, and the solver in the mpc object " *
              "is currently $(JuMP.solver_name(mpc.optim)).\n" *
              "The results in closed-loop may be different."
    end
    validate_weights(mpc)
    validate_constraints(mpc)
    return nothing
end

function validate_weights(mpc::ModelPredictiveControl.LinMPC)
    ny, nu = mpc.estim.model.ny, mpc.estim.model.nu
    Hp, Hc = mpc.Hp, mpc.Hc
    nΔU = Hc * nu
    M_Hp, N_Hc, L_Hp = mpc.weights.M_Hp, mpc.weights.Ñ_Hc[1:nΔU, 1:nΔU], mpc.weights.L_Hp
    M_1, N_1, L_1 = M_Hp[1:ny, 1:ny], N_Hc[1:nu, 1:nu], L_Hp[1:nu, 1:nu]
    for i in 2:mpc.Hp-1 # last block is terminal weight, can be different
        M_i = M_Hp[(i-1)*ny+1:i*ny, (i-1)*ny+1:i*ny]
        if !isapprox(M_i, M_1)
            error("LinearMPC only supports identical weights for each stages in M_Hp.")
        end
    end
    isblockdiag(M_Hp, ny, Hp) || error("M_Hp must be block diagonal.")
    for i in 2:mpc.Hc
        N_i = N_Hc[(i-1)*nu+1:i*nu, (i-1)*nu+1:i*nu]
        if !isapprox(N_i, N_1)
            error("LinearMPC only supports identical weights for each stages in Ñ_Hc.")
        end
    end
    isblockdiag(N_Hc, nu, Hc) || error("Ñ_Hc must be block diagonal.")
    for i in 2:mpc.Hp
        L_i = L_Hp[(i-1)*nu+1:i*nu, (i-1)*nu+1:i*nu]
        if !isapprox(L_i, L_1)
            error("LinearMPC only supports identical weights for each stages in L_Hp.")
        end
    end
    isblockdiag(L_Hp, nu, Hp) || error("L_Hp must be block diagonal.")
    return nothing
end

function validate_constraints(mpc::ModelPredictiveControl.LinMPC)
    ΔŨmin, ΔŨmax = mpc.con.ΔŨmin, mpc.con.ΔŨmax
    C_umin, C_umax = -mpc.con.A_Umin[:, end], -mpc.con.A_Umax[:, end]
    C_ymin, C_ymax = -mpc.con.A_Ymin[:, end], -mpc.con.A_Ymax[:, end]
    C_x̂min, C_x̂max = -mpc.con.A_x̂min[:, end], -mpc.con.A_x̂max[:, end]
    is0or1(C) = all(x -> x ≈ 0 || x ≈ 1, C)
    if !is0or1(C_umin) || !is0or1(C_umax) || !is0or1(C_ymin) || !is0or1(C_ymax)
        error("LinearMPC does not support softness parameters c ≠ 0 or 1.")
    end
    if !isapprox(C_umin, C_umax) || !isapprox(C_ymin, C_ymax) || !isapprox(C_x̂min, C_x̂max)
        error("LinearMPC does not support different softness parameters for lower and upper bounds.")
    end
    nΔU = mpc.Hc * mpc.estim.model.nu
    if any(isfinite, ΔŨmin[1:nΔU]) || any(isfinite, ΔŨmax[1:nΔU])
        error("LinearMPC does not support constraints on input increments Δu")
    end
    return nothing
end

@doc raw"""
    LinearMPC.MPC(mpc::LinMPC)

Convert a `ModelPredictiveControl.LinMPC` object to a `LinearMPC.MPC` object.

The `LinearMPC` package needs to be installed and available in the activated Julia
environment. The converted object can be used to generate lightweight C-code for embedded
applications using the `LinearMPC.codegen` function. Note that not all features of [`LinMPC`](@ref)
are supported, including these restrictions:

- the solver is limited to [`DAQP`](https://darnstrom.github.io/daqp/).
- the transcription method must be [`SingleShooting`](@ref).
- the state estimator must be a [`SteadyKalmanFilter`](@ref) with `direct=true`.
- only block-diagonal weights are allowed.
- input increment constraints ``\mathbf{Δu_{min}}`` and ``\mathbf{Δu_{max}}`` are not
  supported for now.

But the package has also several exclusive functionalities, such as pre-stabilization,
constrained explicit MPC, and binary manipulated inputs. See the [`LinearMPC.jl`](@extref LinearMPC)
documentation for more details on the supported features and how to generate code. 

# Examples
```jldoctest
julia> import LinearMPC, JuMP, DAQP;

julia> mpc1 = LinMPC(LinModel(tf(2, [10, 1]), 1.0); optim=JuMP.Model(DAQP.Optimizer));

julia> preparestate!(mpc1, [1.0]);

julia> u1 = moveinput!(mpc1, [10.0]); round.(u1, digits=6)
1-element Vector{Float64}:
 17.577311

julia> mpc2 = LinearMPC.MPC(mpc1);

julia> x̂2 = LinearMPC.correct_state!(mpc2, [1.0]);

julia> u2 = LinearMPC.compute_control(mpc2, x̂2, r=[10.0]); round.(u2, digits=6)
1-element Vector{Float64}:
 17.577311
```
"""
LinearMPC.MPC(mpc::ModelPredictiveControl.LinMPC) = convert(LinearMPC.MPC, mpc)


end # LinearMPCext