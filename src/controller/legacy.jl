# TODO: Deprecated constraint splatting syntax (legacy), delete get_optim_functions later.

"""
    get_optim_functions(mpc::NonLinMPC, optim)

Get the legacy nonlinear optimization functions for MPC (all based on the splatting syntax).

See [`get_nonlinops`](@ref) for additional details.
"""
function get_optim_functions(mpc::NonLinMPC, ::JuMP.GenericModel{JNT}) where JNT<:Real
    # ----------- common cache for Jfunc, gfuncs and geqfuncs  ----------------------------
    model = mpc.estim.model
    transcription = mpc.transcription
    grad, jac = mpc.gradient, mpc.jacobian
    nu, ny, nx̂, nϵ = model.nu, model.ny, mpc.estim.nx̂, mpc.nϵ
    nk = get_nk(model, transcription)
    Hp, Hc = mpc.Hp, mpc.Hc
    ng, nc, neq = length(mpc.con.i_g), mpc.con.nc, mpc.con.neq
    nZ̃, nU, nŶ, nX̂, nK = length(mpc.Z̃), Hp*nu, Hp*ny, Hp*nx̂, Hp*nk
    nΔŨ, nUe, nŶe = nu*Hc + nϵ, nU + nu, nŶ + ny  
    strict = Val(true)
    myNaN  = convert(JNT, NaN)
    J::Vector{JNT}                   = zeros(JNT, 1)
    ΔŨ::Vector{JNT}                  = zeros(JNT, nΔŨ)
    x̂0end::Vector{JNT}               = zeros(JNT, nx̂)
    K0::Vector{JNT}                  = zeros(JNT, nK)
    Ue::Vector{JNT}, Ŷe::Vector{JNT} = zeros(JNT, nUe), zeros(JNT, nŶe)
    U0::Vector{JNT}, Ŷ0::Vector{JNT} = zeros(JNT, nU),  zeros(JNT, nŶ)
    Û0::Vector{JNT}, X̂0::Vector{JNT} = zeros(JNT, nU),  zeros(JNT, nX̂)
    gc::Vector{JNT}, g::Vector{JNT}  = zeros(JNT, nc),  zeros(JNT, ng)
    geq::Vector{JNT}                 = zeros(JNT, neq)
    # ---------------------- objective function ------------------------------------------- 
    function Jfunc!(Z̃, ΔŨ, x̂0end, Ue, Ŷe, U0, Ŷ0, Û0, K0, X̂0, gc, g, geq)
        update_predictions!(ΔŨ, x̂0end, Ue, Ŷe, U0, Ŷ0, Û0, K0, X̂0, gc, g, geq, mpc, Z̃)
        return obj_nonlinprog!(Ŷ0, U0, mpc, model, Ue, Ŷe, ΔŨ)
    end
    Z̃_∇J = fill(myNaN, nZ̃)      # NaN to force update_predictions! at first call
    ∇J_context = (
        Cache(ΔŨ), Cache(x̂0end), Cache(Ue), Cache(Ŷe), Cache(U0), Cache(Ŷ0), 
        Cache(Û0), Cache(K0), Cache(X̂0), 
        Cache(gc), Cache(g), Cache(geq),
    )
    ∇J_prep = prepare_gradient(Jfunc!, grad, Z̃_∇J, ∇J_context...; strict)
    ∇J = Vector{JNT}(undef, nZ̃)
    function update_objective!(J, ∇J, Z̃_∇J, Z̃arg)
        if isdifferent(Z̃arg, Z̃_∇J)
            Z̃_∇J .= Z̃arg
            J[], _ = value_and_gradient!(Jfunc!, ∇J, ∇J_prep, grad, Z̃_∇J, ∇J_context...)
        end
    end    
    function J_func(Z̃arg::Vararg{T, N}) where {N, T<:Real}
        update_objective!(J, ∇J, Z̃_∇J, Z̃arg)
        return J[]::T
    end
    ∇J_func! = if nZ̃ == 1        # univariate syntax (see JuMP.@operator doc):
        function (Z̃arg)
            update_objective!(J, ∇J, Z̃_∇J, Z̃arg)
            return ∇J[begin]
        end
    else                        # multivariate syntax (see JuMP.@operator doc):
        function (∇Jarg::AbstractVector{T}, Z̃arg::Vararg{T, N}) where {N, T<:Real}
            update_objective!(J, ∇J, Z̃_∇J, Z̃arg)
            return ∇Jarg .= ∇J
        end
    end
    # --------------------- inequality constraint functions -------------------------------
    function gfunc!(g, Z̃, ΔŨ, x̂0end, Ue, Ŷe, U0, Ŷ0, Û0, K0, X̂0, gc, geq)
        update_predictions!(ΔŨ, x̂0end, Ue, Ŷe, U0, Ŷ0, Û0, K0, X̂0, gc, g, geq, mpc, Z̃)
        return g
    end
    Z̃_∇g = fill(myNaN, nZ̃)      # NaN to force update_predictions! at first call
    ∇g_context = (
        Cache(ΔŨ), Cache(x̂0end), Cache(Ue), Cache(Ŷe), Cache(U0), Cache(Ŷ0), 
        Cache(Û0), Cache(K0), Cache(X̂0), 
        Cache(gc), Cache(geq),
    )
    # temporarily enable all the inequality constraints for sparsity detection:
    mpc.con.i_g[1:end-nc] .= true
    ∇g_prep  = prepare_jacobian(gfunc!, g, jac, Z̃_∇g, ∇g_context...; strict)
    mpc.con.i_g[1:end-nc] .= false
    ∇g = init_diffmat(JNT, jac, ∇g_prep, nZ̃, ng)
    function update_con!(g, ∇g, Z̃_∇g, Z̃arg)
        if isdifferent(Z̃arg, Z̃_∇g)
            Z̃_∇g .= Z̃arg
            value_and_jacobian!(gfunc!, g, ∇g, ∇g_prep, jac, Z̃_∇g, ∇g_context...)
        end
    end
    g_funcs = Vector{Function}(undef, ng)
    for i in eachindex(g_funcs)
        gfunc_i = function (Z̃arg::Vararg{T, N}) where {N, T<:Real}
            update_con!(g, ∇g, Z̃_∇g, Z̃arg)
            return g[i]::T
        end
        g_funcs[i] = gfunc_i
    end
    ∇g_funcs! = Vector{Function}(undef, ng)
    for i in eachindex(∇g_funcs!)
        ∇gfuncs_i! = if nZ̃ == 1     # univariate syntax (see JuMP.@operator doc):
            function (Z̃arg::T) where T<:Real
                update_con!(g, ∇g, Z̃_∇g, Z̃arg)
                return ∇g[i, begin]
            end
        else                        # multivariate syntax (see JuMP.@operator doc):
            function (∇g_i, Z̃arg::Vararg{T, N}) where {N, T<:Real}
                update_con!(g, ∇g, Z̃_∇g, Z̃arg)
                return ∇g_i .= @views ∇g[i, :] 
            end
        end
        ∇g_funcs![i] = ∇gfuncs_i!
    end
    # --------------------- equality constraint functions ---------------------------------
    function geqfunc!(geq, Z̃, ΔŨ, x̂0end, Ue, Ŷe, U0, Ŷ0, Û0, K0, X̂0, gc, g) 
        update_predictions!(ΔŨ, x̂0end, Ue, Ŷe, U0, Ŷ0, Û0, K0, X̂0, gc, g, geq, mpc, Z̃)
        return geq
    end
    Z̃_∇geq = fill(myNaN, nZ̃)    # NaN to force update_predictions! at first call
    ∇geq_context = (
        Cache(ΔŨ), Cache(x̂0end), Cache(Ue), Cache(Ŷe), Cache(U0), Cache(Ŷ0),
        Cache(Û0), Cache(K0),   Cache(X̂0),
        Cache(gc), Cache(g)
    )
    ∇geq_prep = prepare_jacobian(geqfunc!, geq, jac, Z̃_∇geq, ∇geq_context...; strict)
    ∇geq = init_diffmat(JNT, jac, ∇geq_prep, nZ̃, neq)
    function update_con_eq!(geq, ∇geq, Z̃_∇geq, Z̃arg)
        if isdifferent(Z̃arg, Z̃_∇geq)
            Z̃_∇geq .= Z̃arg
            value_and_jacobian!(geqfunc!, geq, ∇geq, ∇geq_prep, jac, Z̃_∇geq, ∇geq_context...)
        end
    end
    geq_funcs = Vector{Function}(undef, neq)
    for i in eachindex(geq_funcs)
        geqfunc_i = function (Z̃arg::Vararg{T, N}) where {N, T<:Real}
            update_con_eq!(geq, ∇geq, Z̃_∇geq, Z̃arg)
            return geq[i]::T
        end
        geq_funcs[i] = geqfunc_i          
    end
    ∇geq_funcs! = Vector{Function}(undef, neq)
    for i in eachindex(∇geq_funcs!)
        # only multivariate syntax, univariate is impossible since nonlinear equality
        # constraints imply MultipleShooting, thus input increment ΔU and state X̂0 in Z̃:
        ∇geqfuncs_i! = 
            function (∇geq_i, Z̃arg::Vararg{T, N}) where {N, T<:Real}
                update_con_eq!(geq, ∇geq, Z̃_∇geq, Z̃arg)
                return ∇geq_i .= @views ∇geq[i, :]
            end
        ∇geq_funcs![i] = ∇geqfuncs_i!
    end
    return J_func, ∇J_func!, g_funcs, ∇g_funcs!, geq_funcs, ∇geq_funcs!
end

# TODO: Deprecated constraint splatting syntax (legacy), delete init_nonlincon_leg! later.

"""
    init_nonlincon_leg!(
        mpc::PredictiveController, model::LinModel, transcription::TranscriptionMethod, 
        gfuncs  , ∇gfuncs!,   
        geqfuncs, ∇geqfuncs!
    )

Init nonlinear constraints for [`LinModel`](@ref) for all [`TranscriptionMethod`](@ref).

The only nonlinear constraints are the custom inequality constraints `gc`.
"""
function init_nonlincon_leg!(
    mpc::PredictiveController, ::LinModel, ::TranscriptionMethod,
    gfuncs, ∇gfuncs!, 
    _ , _    
) 
    optim, con = mpc.optim, mpc.con
    nZ̃ = length(mpc.Z̃)
    if length(con.i_g) ≠ 0
        i_base = 0
        for i in 1:con.nc
            name = Symbol("g_c_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_base+i], ∇gfuncs![i_base+i]; name
            )
        end
    end
    return nothing
end

"""
    init_nonlincon_leg!(
        mpc::PredictiveController, model::NonLinModel, ::SingleShooting, 
        gfuncs,   ∇gfuncs!,
        geqfuncs, ∇geqfuncs!
    )

Init nonlinear constraints for [`NonLinModel`](@ref) and [`SingleShooting`](@ref).

The nonlinear constraints are the custom inequality constraints `gc`, the output
prediction `Ŷ` bounds and the terminal state `x̂end` bounds.
"""
function init_nonlincon_leg!(
    mpc::PredictiveController, ::NonLinModel, ::SingleShooting, gfuncs, ∇gfuncs!, _ , _
)
    optim, con = mpc.optim, mpc.con
    ny, nx̂, Hp, nZ̃ = mpc.estim.model.ny, mpc.estim.nx̂, mpc.Hp, length(mpc.Z̃)
    if length(con.i_g) ≠ 0
        i_base = 0
        for i in eachindex(con.Y0min)
            name = Symbol("g_Y0min_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_base+i], ∇gfuncs![i_base+i]; name
            )
        end
        i_base = 1Hp*ny
        for i in eachindex(con.Y0max)
            name = Symbol("g_Y0max_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_base+i], ∇gfuncs![i_base+i]; name
            )
        end
        i_base = 2Hp*ny
        for i in eachindex(con.x̂0min)
            name = Symbol("g_x̂0min_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_base+i], ∇gfuncs![i_base+i]; name
            )
        end
        i_base = 2Hp*ny + nx̂
        for i in eachindex(con.x̂0max)
            name = Symbol("g_x̂0max_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_base+i], ∇gfuncs![i_base+i]; name
            )
        end
        i_base = 2Hp*ny + 2nx̂
        for i in 1:con.nc
            name = Symbol("g_c_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_base+i], ∇gfuncs![i_base+i]; name
            )
        end
    end
    return nothing
end

"""
    init_nonlincon_leg!(
        mpc::PredictiveController, model::NonLinModel, transcription::TranscriptionMethod, 
        gfuncs,   ∇gfuncs!,
        geqfuncs, ∇geqfuncs!
    )
    
Init nonlinear constraints for [`NonLinModel`](@ref) and other [`TranscriptionMethod`](@ref).

The nonlinear constraints are the output prediction `Ŷ` bounds, the custom inequality
constraints `gc` and all the nonlinear equality constraints `geq`.
"""
function init_nonlincon_leg!(
    mpc::PredictiveController, ::NonLinModel, ::TranscriptionMethod, 
    gfuncs,     ∇gfuncs!,
    geqfuncs,   ∇geqfuncs!
)
    optim, con = mpc.optim, mpc.con
    ny, nx̂, Hp, nZ̃ = mpc.estim.model.ny, mpc.estim.nx̂, mpc.Hp, length(mpc.Z̃)
    # --- nonlinear inequality constraints ---
    if length(con.i_g) ≠ 0
        i_base = 0
        for i in eachindex(con.Y0min)
            name = Symbol("g_Y0min_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_base+i], ∇gfuncs![i_base+i]; name
            )
        end
        i_base = 1Hp*ny
        for i in eachindex(con.Y0max)
            name = Symbol("g_Y0max_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_base+i], ∇gfuncs![i_base+i]; name
            )
        end
        i_base = 2Hp*ny
        for i in 1:con.nc
            name = Symbol("g_c_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, gfuncs[i_base+i], ∇gfuncs![i_base+i]; name
            )
        end
    end
    # --- nonlinear equality constraints ---
    Z̃var = optim[:Z̃var]
    for i in eachindex(geqfuncs)
        name = Symbol("geq_$i")
        geqfunc_i = optim[name] = JuMP.add_nonlinear_operator(
            optim, nZ̃, geqfuncs[i], ∇geqfuncs![i]; name
        )
        # set with @constrains here instead of set_nonlincon!, since the number of nonlinear 
        # equality constraints is known and constant (±Inf are impossible):
        @constraint(optim, geqfunc_i(Z̃var...) == 0)
    end
    return nothing
end

# TODO: Deprecated constraint splatting syntax (legacy), delete set_nonlincon_leg! later.

"By default, there is no nonlinear constraint, thus do nothing."
function set_nonlincon_leg!(
    ::PredictiveController, ::SimModel, ::TranscriptionMethod, ::JuMP.GenericModel, 
    )
    return nothing
end

"""
    set_nonlincon_leg!(mpc::PredictiveController, ::LinModel, ::TranscriptionMethod, optim)

Set the custom nonlinear inequality constraints for `LinModel`.
"""
function set_nonlincon_leg!(
    mpc::PredictiveController, ::LinModel, ::TranscriptionMethod, ::JuMP.GenericModel{JNT}
) where JNT<:Real
    optim = mpc.optim
    Z̃var = optim[:Z̃var]
    con = mpc.con
    nonlin_constraints = JuMP.all_constraints(optim, JuMP.NonlinearExpr, MOI.LessThan{JNT})
    map(con_ref -> JuMP.delete(optim, con_ref), nonlin_constraints)
    for i in 1:con.nc
        gfunc_i = optim[Symbol("g_c_$i")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    return nothing
end

"""
    set_nonlincon_leg!(mpc::PredictiveController, ::NonLinModel, ::MultipleShooting, optim)

Also set output prediction `Ŷ` constraints for `NonLinModel` and non-`SingleShooting`.
"""
function set_nonlincon_leg!(
    mpc::PredictiveController, ::NonLinModel, ::TranscriptionMethod, ::JuMP.GenericModel{JNT}
) where JNT<:Real
    optim = mpc.optim
    Z̃var = optim[:Z̃var]
    con = mpc.con
    nonlin_constraints = JuMP.all_constraints(optim, JuMP.NonlinearExpr, MOI.LessThan{JNT})
    map(con_ref -> JuMP.delete(optim, con_ref), nonlin_constraints)
    for i in findall(.!isinf.(con.Y0min))
        gfunc_i = optim[Symbol("g_Y0min_$(i)")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    for i in findall(.!isinf.(con.Y0max))
        gfunc_i = optim[Symbol("g_Y0max_$(i)")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    for i in 1:con.nc
        gfunc_i = optim[Symbol("g_c_$i")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    return nothing
end

"""
    set_nonlincon_leg!(mpc::PredictiveController, ::NonLinModel, ::SingleShooting, optim)

Also set output prediction `Ŷ` and terminal state `x̂end` constraint for `SingleShooting`.
"""
function set_nonlincon_leg!(
    mpc::PredictiveController, ::NonLinModel, ::SingleShooting, ::JuMP.GenericModel{JNT}
) where JNT<:Real
    optim = mpc.optim
    Z̃var = optim[:Z̃var]
    con = mpc.con
    nonlin_constraints = JuMP.all_constraints(optim, JuMP.NonlinearExpr, MOI.LessThan{JNT})
    map(con_ref -> JuMP.delete(optim, con_ref), nonlin_constraints)
    for i in findall(.!isinf.(con.Y0min))
        gfunc_i = optim[Symbol("g_Y0min_$(i)")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    for i in findall(.!isinf.(con.Y0max))
        gfunc_i = optim[Symbol("g_Y0max_$(i)")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    for i in findall(.!isinf.(con.x̂0min))
        gfunc_i = optim[Symbol("g_x̂0min_$(i)")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    for i in findall(.!isinf.(con.x̂0max))
        gfunc_i = optim[Symbol("g_x̂0max_$(i)")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    for i in 1:con.nc
        gfunc_i = optim[Symbol("g_c_$i")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    return nothing
end