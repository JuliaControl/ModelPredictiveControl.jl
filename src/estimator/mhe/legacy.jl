# TODO: Deprecated constraint splatting syntax (legacy), delete get_optim_functions later.

"""
    get_optim_functions(estim::MovingHorizonEstimator, optim)

Get the legacy nonlinear optimization functions for MHE (all based on the splatting syntax).

See [`get_nonlinops`](@ref) for additional details.
"""
function get_optim_functions(
    estim::MovingHorizonEstimator, ::JuMP.GenericModel{JNT},
) where {JNT <: Real}
    # ----------- common cache for Jfunc and gfuncs  --------------------------------------
    model, con = estim.model, estim.con
    grad, jac = estim.gradient, estim.jacobian
    nx̂, nym, nŷ, nu, nε, nk = estim.nx̂, estim.nym, model.ny, model.nu, estim.nε, model.nk
    He = estim.He
    nV̂, nX̂, ng, nZ̃ = He*nym, He*nx̂, length(con.i_g), length(estim.Z̃)
    strict = Val(true)
    myNaN = convert(JNT, NaN)
    J::Vector{JNT}                   = zeros(JNT, 1)
    V̂::Vector{JNT},  X̂0::Vector{JNT} = zeros(JNT, nV̂), zeros(JNT, nX̂)
    k0::Vector{JNT}                  = zeros(JNT, nk)
    û0::Vector{JNT}, ŷ0::Vector{JNT} = zeros(JNT, nu), zeros(JNT, nŷ)
    g::Vector{JNT}                   = zeros(JNT, ng)
    x̄::Vector{JNT}                   = zeros(JNT, nx̂)
    # --------------------- objective functions -------------------------------------------
    function Jfunc!(Z̃, V̂, X̂0, û0, k0, ŷ0, g, x̄)
        update_prediction!(V̂, X̂0, û0, k0, ŷ0, g, estim, Z̃)
        return obj_nonlinprog!(x̄, estim, model, V̂, Z̃)
    end
    Z̃_∇J = fill(myNaN, nZ̃)      # NaN to force update_predictions! at first call
    ∇J_context = (
        Cache(V̂),  Cache(X̂0), Cache(û0), Cache(k0), Cache(ŷ0),
        Cache(g),
        Cache(x̄),
    )
    # temporarily "fill" the estimation window for the preparation of the gradient: 
    estim.Nk[] = He
    ∇J_prep = prepare_gradient(Jfunc!, grad, Z̃_∇J, ∇J_context...; strict)
    estim.Nk[] = 0
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
    ∇J_func! = function (∇Jarg::AbstractVector{T}, Z̃arg::Vararg{T, N}) where {N, T<:Real}
        # only the multivariate syntax of JuMP.@operator, univariate is impossible for MHE
        # since Z̃ comprises the arrival state estimate AND the estimated process noise
        update_objective!(J, ∇J, Z̃_∇J, Z̃arg)
        return ∇Jarg .= ∇J
    end
    # --------------------- inequality constraint functions -------------------------------
    function gfunc!(g, Z̃, V̂, X̂0, û0, k0, ŷ0)
        return update_prediction!(V̂, X̂0, û0, k0, ŷ0, g, estim, Z̃)
    end
    Z̃_∇g = fill(myNaN, nZ̃)      # NaN to force update_predictions! at first call
    ∇g_context = (
        Cache(V̂), Cache(X̂0), Cache(û0), Cache(k0), Cache(ŷ0),
    )
    # temporarily enable all the inequality constraints for sparsity detection:
    estim.con.i_g .= true  
    estim.Nk[] = He
    ∇g_prep  = prepare_jacobian(gfunc!, g, jac, Z̃_∇g, ∇g_context...; strict)
    estim.con.i_g .= false
    estim.Nk[] = 0
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
        ∇g_funcs![i] = function (∇g_i, Z̃arg::Vararg{T, N}) where {N, T<:Real}
            # only the multivariate syntax of JuMP.@operator, see above for the explanation
            update_con!(g, ∇g, Z̃_∇g, Z̃arg)
            return ∇g_i .= @views ∇g[i, :]
        end
    end
    return J_func, ∇J_func!, g_funcs, ∇g_funcs!
end

# TODO: Deprecated constraint splatting syntax (legacy), delete init_nonlincon_leg! later.

function init_nonlincon_leg!(estim::MovingHorizonEstimator, g_funcs, ∇g_funcs!)
    optim, con = estim.optim, estim.con
    nV̂, nX̂ = estim.He*estim.nym, estim.He*estim.nx̂
    nZ̃ = length(estim.Z̃)
    if length(con.i_g) ≠ 0
        i_base = 0
        for i in eachindex(con.X̂0min)
            name = Symbol("g_X̂0min_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, g_funcs[i_base + i], ∇g_funcs![i_base + i]; name
            )
        end
        i_base = nX̂
        for i in eachindex(con.X̂0max)
            name = Symbol("g_X̂0max_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, g_funcs[i_base + i], ∇g_funcs![i_base + i]; name
            )
        end
        i_base = 2*nX̂
        for i in eachindex(con.V̂min)
            name = Symbol("g_V̂min_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, g_funcs[i_base + i], ∇g_funcs![i_base + i]; name
            )
        end
        i_base = 2*nX̂ + nV̂
        for i in eachindex(con.V̂max)
            name = Symbol("g_V̂max_$i")
            optim[name] = JuMP.add_nonlinear_operator(
                optim, nZ̃, g_funcs[i_base + i], ∇g_funcs![i_base + i]; name
            )
        end
    end
    return nothing
end

# TODO: Deprecated constraint splatting syntax (legacy), delete set_nonlincon_leg! later.

"By default, no nonlinear constraints in the MHE, thus return nothing."
set_nonlincon_leg!(::MovingHorizonEstimator, ::SimModel, ::JuMP.GenericModel) = nothing

"Set the nonlinear constraints on the output predictions `Ŷ` and terminal states `x̂end`."
function set_nonlincon_leg!(
    estim::MovingHorizonEstimator, ::NonLinModel, optim::JuMP.GenericModel{JNT}
) where JNT<:Real
    optim, con = estim.optim, estim.con
    Z̃var = optim[:Z̃var]
    nonlin_constraints = JuMP.all_constraints(optim, JuMP.NonlinearExpr, MOI.LessThan{JNT})
    map(con_ref -> JuMP.delete(optim, con_ref), nonlin_constraints)
    for i in findall(.!isinf.(con.X̂0min))
        gfunc_i = optim[Symbol("g_X̂0min_$(i)")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    for i in findall(.!isinf.(con.X̂0max))
        gfunc_i = optim[Symbol("g_X̂0max_$(i)")]
        @constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    for i in findall(.!isinf.(con.V̂min))
        gfunc_i = optim[Symbol("g_V̂min_$(i)")]
        JuMP.@constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    for i in findall(.!isinf.(con.V̂max))
        gfunc_i = optim[Symbol("g_V̂max_$(i)")]
        JuMP.@constraint(optim, gfunc_i(Z̃var...) <= 0)
    end
    return nothing
end