module LinearMPCext

using ModelPredictiveControl
using LinearAlgebra, SparseArrays
using JuMP

import LinearMPC
import ModelPredictiveControl: isblockdiag

function Base.convert(::Type{LinearMPC.MPC}, mpc::ModelPredictiveControl.LinMPC)
    model, estim, weights = mpc.estim.model, mpc.estim, mpc.weights
    nu, ny, nx̂ = model.nu, model.ny, estim.nx̂
    Hp, Hc = mpc.Hp, mpc.Hc
    nΔU = Hc * nu
    validate_compatibility(mpc)
    # --- Model parameters ---
    F, G, Gd = estim.Â, estim.B̂u, estim.B̂d
    C, Dd = estim.Ĉ, estim.D̂d
    Np = Hp
    Nc = Hc
    newmpc = LinearMPC.MPC(F, G; Gd, C, Dd, Np, Nc)
    # --- Operating points ---
    uoff = model.uop
    doff = model.dop
    yoff = model.yop
    xoff = estim.x̂op
    foff = estim.f̂op
    LinearMPC.set_offset!(newmpc; uo=uoff, ho=yoff, doff=doff, xo=xoff, fo=foff)
    # --- State observer parameters ---
    Q, R = estim.cov.Q̂, estim.cov.R̂
    LinearMPC.set_state_observer!(newmpc; C=estim.Ĉm, Q, R)
    # --- Objective function weights ---
    Q = weights.M_Hp[1:ny, 1:ny]
    Qf = weights.M_Hp[end-ny+1:end, end-ny+1:end]
    Rr = weights.Ñ_Hc[1:nu, 1:nu]
    R = weights.L_Hp[1:nu, 1:nu]
    LinearMPC.set_objective!(newmpc; Q, Rr, R, Qf)
    # --- Custom move blocking ---
    LinearMPC.move_block!(newmpc, mpc.nb) # un-comment when debugged
    # ---- Constraint softening ---
    only_hard = weights.isinf_C
    if !only_hard
        # LinearMPC relies on a different softening mechanism (new implicit slacks for each
        # softened bounds), so we apply an approximate conversion factor on the Cwt weight:
        Cwt = weights.Ñ_Hc[end, end]
        nsoft = sum((mpc.con.A[:,end] .< 0) .& (mpc.con.i_b)) - 1
        newmpc.settings.soft_weight = 10*sqrt(nsoft*Cwt)
        C_u  = -mpc.con.A_Umin[:, end]
        C_Δu = -mpc.con.A_ΔŨmin[1:nΔU, end]
        C_y  = -mpc.con.A_Ymin[:, end]
        c_x̂  = -mpc.con.A_x̂min[:, end]
    else
        C_u  = zeros(nu*Hp)
        C_Δu = zeros(nu*Hc)
        C_y  = zeros(ny*Hp)
        c_x̂  = zeros(nx̂)
    end
    # --- Manipulated inputs constraints ---
    Umin, Umax = mpc.con.U0min + mpc.Uop, mpc.con.U0max + mpc.Uop
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
            soft = !only_hard && c_u_k[i] > 0
            Au = I_u[i:i, :]
            LinearMPC.add_constraint!(newmpc; Au, lb, ub, ks, soft)
        end
    end
    # --- Input increment constraints ---
    ΔUmin, ΔUmax = mpc.con.ΔŨmin[1:nΔU], mpc.con.ΔŨmax[1:nΔU]
    I_Δu = Matrix{Float64}(I, nu, nu)
    for k in 0:Hc-1
        Δumin_k, Δumax_k = ΔUmin[k*nu+1:(k+1)*nu], ΔUmax[k*nu+1:(k+1)*nu]
        c_Δu_k = C_Δu[k*nu+1:(k+1)*nu]
        ks = [k + 1] # a `1` in ks argument corresponds to the present time step k+0
        for i in 1:nu
            lb = isfinite(Δumin_k[i]) ? [Δumin_k[i]] : zeros(0)
            ub = isfinite(Δumax_k[i]) ? [Δumax_k[i]] : zeros(0)
            soft = !only_hard && c_Δu_k[i] > 0
            Au, Aup = I_Δu[i:i, :], -I_Δu[i:i, :]
            LinearMPC.add_constraint!(newmpc; Au, Aup, lb, ub, ks, soft)
        end
    end
    # --- Output constraints ---
    Y0min, Y0max = mpc.con.Y0min, mpc.con.Y0max
    for k in 1:Hp
        ymin_k, ymax_k = Y0min[(k-1)*ny+1:k*ny], Y0max[(k-1)*ny+1:k*ny]
        c_y_k = C_y[(k-1)*ny+1:k*ny]
        ks = [k + 1] # a `1` in ks argument corresponds to the present time step k+0
        for i in 1:ny
            lb = isfinite(ymin_k[i]) ? [ymin_k[i]] : zeros(0)
            ub = isfinite(ymax_k[i]) ? [ymax_k[i]] : zeros(0)
            soft = !only_hard && c_y_k[i] > 0
            Ax, Ad = C[i:i, :], Dd[i:i, :]
            LinearMPC.add_constraint!(newmpc; Ax, Ad, lb, ub, ks, soft)
        end
    end
    # --- Terminal constraints ---
    x̂0min, x̂0max = mpc.con.x̂0min, mpc.con.x̂0max
    I_x̂ = Matrix{Float64}(I, nx̂, nx̂)
    ks = [Hp + 1] # a `1` in ks argument corresponds to the present time step k+0
    for i in 1:nx̂
        lb = isfinite(x̂0min[i]) ? [x̂0min[i]] : zeros(0)
        ub = isfinite(x̂0max[i]) ? [x̂0max[i]] : zeros(0)
        soft = !only_hard && c_x̂[i] > 0
        Ax = I_x̂[i:i, :]
        LinearMPC.add_constraint!(newmpc; Ax, lb, ub, ks, soft)
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
    nΔU = mpc.Hc * mpc.estim.model.nu
    mpc.weights.isinf_C && return nothing # only hard constraints are entirely supported
    C_umin, C_umax   = -mpc.con.A_Umin[:, end], -mpc.con.A_Umax[:, end]
    C_Δumin, C_Δumax = -mpc.con.A_ΔŨmin[1:nΔU, end], -mpc.con.A_ΔŨmax[1:nΔU, end]
    C_ymin, C_ymax   = -mpc.con.A_Ymin[:, end], -mpc.con.A_Ymax[:, end]
    C_x̂min, C_x̂max   = -mpc.con.A_x̂min[:, end], -mpc.con.A_x̂max[:, end]
    is0or1(C) = all(x -> x ≈ 0 || x ≈ 1, C)
    if (
        !is0or1(C_umin)  || !is0or1(C_umax)  || 
        !is0or1(C_Δumin) || !is0or1(C_Δumax) ||
        !is0or1(C_ymin)  || !is0or1(C_ymax)  || 
        !is0or1(C_x̂min)  || !is0or1(C_x̂max) 
        
    )
        error("LinearMPC only supports softness parameters c = 0 or 1.")
    end
    if (
        !isapprox(C_umin, C_umax)   || 
        !isapprox(C_Δumin, C_Δumax) ||
        !isapprox(C_ymin, C_ymax)   || 
        !isapprox(C_x̂min, C_x̂max)   
    )
        error("LinearMPC only supports identical softness parameters for lower and upper bounds.")
    end
    issoft(C) = any(x -> x > 0, C)
    if !mpc.weights.isinf_C && sum(mpc.con.i_b) > 1 # ignore the slack variable ϵ bound
        if issoft(C_umin) || issoft(C_Δumin) || issoft(C_ymin) || issoft(C_x̂min)
            @warn "The LinearMPC conversion applies an approximate conversion " *
                  "of the soft constraints.\n You may need to adjust the soft_weight "*
                  "field of the LinearMPC.MPC object to replicate behaviors."
        end
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
- the constraint relaxation mechanism is different, so a 1-on-1 conversion of the soft 
  constraints is impossible (use `Cwt=Inf` to disable relaxation).

But the package has also several exclusive functionalities, such as pre-stabilization,
constrained explicit MPC, and binary manipulated inputs. See the [`LinearMPC.jl`](@extref LinearMPC)
documentation for more details on the supported features and how to generate code. 

# Examples
```jldoctest
julia> import LinearMPC, JuMP, DAQP;

julia> mpc1 = LinMPC(LinModel(tf(2, [10, 1]), 1.0); optim=JuMP.Model(DAQP.Optimizer));

julia> preparestate!(mpc1, [1.0]);

julia> u = moveinput!(mpc1, [10.0]); round.(u, digits=6)
1-element Vector{Float64}:
 17.577311

julia> mpc2 = LinearMPC.MPC(mpc1);

julia> x̂ = LinearMPC.correct_state!(mpc2, [1.0]);

julia> u = LinearMPC.compute_control(mpc2, x̂, r=[10.0]); round.(u, digits=6)
1-element Vector{Float64}:
 17.577311
```
"""
LinearMPC.MPC(mpc::ModelPredictiveControl.LinMPC) = convert(LinearMPC.MPC, mpc)


end # LinearMPCext