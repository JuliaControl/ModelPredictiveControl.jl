@doc raw"""
Abstract supertype of all predictive controllers.

---

    (mpc::PredictiveController)(ry, d=Float64[]; kwargs...)

Functor allowing callable `PredictiveController` object as an alias for [`moveinput!`](@ref).

# Examples
```jldoctest
julia> mpc = LinMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1000, Hc=1);

julia> u = mpc([5]); round.(u, digits=3)
1-element Vector{Float64}:
 1.0
```

"""
abstract type PredictiveController end

"Type alias for vector of linear inequality constraints."
const LinConVector = Vector{ConstraintRef{
    Model, 
    MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}, 
    ScalarShape
}}

const InfoDictType = Union{Vector{Float64}, Float64}

"Include all the data for the constraints of [`PredictiveController`](@ref)"
struct ControllerConstraint
    Umin   ::Vector{Float64}
    Umax   ::Vector{Float64}
    ΔŨmin  ::Vector{Float64}
    ΔŨmax  ::Vector{Float64}
    Ŷmin   ::Vector{Float64}
    Ŷmax   ::Vector{Float64}
    A_Umin ::Matrix{Float64}
    A_Umax ::Matrix{Float64}
    A_ΔŨmin::Matrix{Float64}
    A_ΔŨmax::Matrix{Float64}
    A_Ŷmin ::Matrix{Float64}
    A_Ŷmax ::Matrix{Float64}
    A      ::Matrix{Float64}
    b      ::Vector{Float64}
    i_b    ::BitVector
    c_Ŷmin ::Vector{Float64}
    c_Ŷmax ::Vector{Float64}
end

@doc raw"""
    setconstraint!(mpc::PredictiveController; <keyword arguments>)

Set the constraint parameters of `mpc` predictive controller.

The predictive controllers support both soft and hard constraints, defined by:
```math 
\begin{alignat*}{3}
    \mathbf{u_{min}  - c_{u_{min}}}  ϵ &≤ \mathbf{u}(k+j)  &&≤ \mathbf{u_{max}  + c_{u_{max}}}  ϵ &&\qquad j = 0, 1 ,..., H_c - 1 \\
    \mathbf{Δu_{min} - c_{Δu_{min}}} ϵ &≤ \mathbf{Δu}(k+j) &&≤ \mathbf{Δu_{max} + c_{Δu_{max}}} ϵ &&\qquad j = 0, 1 ,..., H_c - 1 \\
    \mathbf{ŷ_{min}  - c_{ŷ_{min}}}  ϵ &≤ \mathbf{ŷ}(k+j)  &&≤ \mathbf{ŷ_{max}  + c_{ŷ_{max}}}  ϵ &&\qquad j = 1, 2 ,..., H_p \\
\end{alignat*}
```
and also ``ϵ ≥ 0``. All the constraint parameters are vector. Use `±Inf` values when there 
is no bound. The constraint softness parameters ``\mathbf{c}``, also called equal concern 
for relaxation, are non-negative values that specify the softness of the associated bound. 
Use `0.0` values for hard constraints. The predicted output constraints ``\mathbf{ŷ_{min}}`` 
and ``\mathbf{ŷ_{max}}`` are soft by default.

# Arguments
!!! info
    The default constraints are mentioned here for clarity but omitting a keyword argument 
    will not re-assign to its default value (defaults are set at construction only).

- `umin=fill(-Inf,nu)` : manipulated input lower bounds ``\mathbf{u_{min}}`` 
- `umax=fill(+Inf,nu)` : manipulated input upper bounds ``\mathbf{u_{max}}`` 
- `Δumin=fill(-Inf,nu)` : manipulated input increment lower bounds ``\mathbf{Δu_{min}}`` 
- `Δumax=fill(+Inf,nu)` : manipulated input increment upper bounds ``\mathbf{Δu_{max}}`` 
- `ŷmin=fill(-Inf,ny)` : predicted output lower bounds ``\mathbf{ŷ_{min}}`` 
- `ŷmax=fill(+Inf,ny)` : predicted output upper bounds ``\mathbf{ŷ_{max}}`` 
- `c_umin=fill(0.0,nu)` : `umin` softness weights ``\mathbf{c_{u_{min}}}`` 
- `c_umax=fill(0.0,nu)` : `umax` softness weights ``\mathbf{c_{u_{max}}}`` 
- `c_Δumin=fill(0.0,nu)` : `Δumin` softness weights ``\mathbf{c_{Δu_{min}}}`` 
- `c_Δumax=fill(0.0,nu)` : `Δumax` softness weights ``\mathbf{c_{Δu_{max}}}`` 
- `c_ŷmin=fill(1.0,ny)` : `ŷmin` softness weights ``\mathbf{c_{ŷ_{min}}}`` 
- `c_ŷmax=fill(1.0,ny)` : `ŷmax` softness weights ``\mathbf{c_{ŷ_{max}}}``

# Examples
```jldoctest
julia> mpc = LinMPC(setop!(LinModel(tf(3, [30, 1]), 4), uop=[50], yop=[25]));

julia> mpc = setconstraint!(mpc, umin=[0], umax=[100], c_umin=[0.0], c_umax=[0.0]);

julia> mpc = setconstraint!(mpc, Δumin=[-10], Δumax=[+10], c_Δumin=[1.0], c_Δumax=[1.0])
LinMPC controller with a sample time Ts = 4.0 s, SteadyKalmanFilter estimator and:
 10 prediction steps Hp
  2 control steps Hc
  1 manipulated inputs u
  2 states x̂
  1 measured outputs ym
  0 unmeasured outputs yu
  0 measured disturbances d
```
"""
function setconstraint!(
    mpc::PredictiveController; 
    umin    = nothing, umax    = nothing,
    Δumin   = nothing, Δumax   = nothing,
    ŷmin    = nothing, ŷmax    = nothing,
    c_umin  = nothing, c_umax  = nothing,
    c_Δumin = nothing, c_Δumax = nothing,
    c_ŷmin  = nothing, c_ŷmax  = nothing
)
    model = mpc.estim.model
    con = mpc.con
    nu, ny = model.nu, model.ny
    Hp, Hc = mpc.Hp, mpc.Hc
    C, E = mpc.C, mpc.Ẽ[:, 1:nu*Hc]
    if !isnothing(umin)
        size(umin)   == (nu,) || error("umin size must be $((nu,))")
        Umin  = repeat(umin, Hc)
        con.Umin[:] = Umin
    end
    if !isnothing(umax)
        size(umax)   == (nu,) || error("umax size must be $((nu,))")
        Umax  = repeat(umax, Hc)
        con.Umax[:] = Umax
    end
    if !isnothing(Δumin)
        size(Δumin)  == (nu,) || error("Δumin size must be $((nu,))")
        ΔUmin = repeat(Δumin, Hc)
        con.ΔŨmin[1:nu*Hc] = ΔUmin
    end
    if !isnothing(Δumax)
        size(Δumax)  == (nu,) || error("Δumax size must be $((nu,))")
        ΔUmax = repeat(Δumax, Hc)
        con.ΔŨmax[1:nu*Hc] = ΔUmax
    end
    if !isnothing(ŷmin)
        size(ŷmin)   == (ny,) || error("ŷmin size must be $((ny,))")
        Ŷmin  = repeat(ŷmin, Hp)
        con.Ŷmin[:] = Ŷmin
    end
    if !isnothing(ŷmax)
        size(ŷmax)   == (ny,) || error("ŷmax size must be $((ny,))")
        Ŷmax  = repeat(ŷmax, Hp)
        con.Ŷmax[:] = Ŷmax
    end
    if !isnothing(c_umin)
        !isinf(C) || error("Slack variable Cwt must be finite to set softness parameters") 
        size(c_umin) == (nu,) || error("c_umin size must be $((nu,))")
        any(c_umin .< 0) && error("c_umin weights should be non-negative")
        c_Umin  = repeat(c_umin, Hc)
        con.A_Umin[:, end] = -c_Umin
    end
    if !isnothing(c_umax)
        !isinf(C) || error("Slack variable Cwt must be finite to set softness parameters") 
        size(c_umax) == (nu,) || error("c_umax size must be $((nu,))")
        any(c_umax .< 0) && error("c_umax weights should be non-negative")
        c_Umax  = repeat(c_umax, Hc)
        con.A_Umax[:, end] = -c_Umax
    end
    if !isnothing(c_Δumin)
        !isinf(C) || error("Slack variable Cwt must be finite to set softness parameters") 
        size(c_Δumin) == (nu,) || error("c_Δumin size must be $((nu,))")
        any(c_Δumin .< 0) && error("c_Δumin weights should be non-negative")
        c_ΔUmin  = repeat(c_Δumin, Hc)
        con.A_ΔŨmin[1:end-1, end] = -c_ΔUmin 
    end
    if !isnothing(c_Δumax)
        !isinf(C) || error("Slack variable Cwt must be finite to set softness parameters") 
        size(c_Δumax) == (nu,) || error("c_Δumax size must be $((nu,))")
        any(c_Δumax .< 0) && error("c_Δumax weights should be non-negative")
        c_ΔUmax  = repeat(c_Δumax, Hc)
        con.A_ΔŨmax[1:end-1, end] = -c_ΔUmax
    end
    if !isnothing(c_ŷmin)
        !isinf(C) || error("Slack variable Cwt must be finite to set softness parameters") 
        size(c_ŷmin) == (ny,) || error("c_ŷmin size must be $((ny,))")
        any(c_ŷmin .< 0) && error("c_ŷmin weights should be non-negative")
        c_Ŷmin  = repeat(c_ŷmin, Hp)
        con.c_Ŷmin[:] = c_Ŷmin
        A_Ŷmin ,_ = relaxŶ(model, C, con.c_Ŷmin, con.c_Ŷmax, E)
        con.A_Ŷmin[:] = A_Ŷmin
    end
    if !isnothing(c_ŷmax)
        !isinf(C) || error("Slack variable Cwt must be finite to set softness parameters") 
        size(c_ŷmax) == (ny,) || error("c_ŷmax size must be $((ny,))")
        any(c_ŷmax .< 0) && error("c_ŷmax weights should be non-negative")
        c_Ŷmax  = repeat(c_ŷmax, Hp)
        con.c_Ŷmax[:] = c_Ŷmax
        _, A_Ŷmax = relaxŶ(model, C, con.c_Ŷmin, con.c_Ŷmax, E)
        con.A_Ŷmax[:] = A_Ŷmax
    end
    i_Umin,  i_Umax  = .!isinf.(con.Umin),  .!isinf.(con.Umax)
    i_ΔŨmin, i_ΔŨmax = .!isinf.(con.ΔŨmin), .!isinf.(con.ΔŨmin)
    i_Ŷmin,  i_Ŷmax  = .!isinf.(con.Ŷmin),  .!isinf.(con.Ŷmax)
    con.A[:], con.i_b[:] = init_linconstraint(model, 
        con.A_Umin, con.A_Umax, con.A_ΔŨmin, con.A_ΔŨmax, con.A_Ŷmin, con.A_Ŷmax,
        i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ŷmin, i_Ŷmax
    )
    A = con.A[con.i_b, :]
    b = con.b[con.i_b]
    ΔŨvar = mpc.optim[:ΔŨvar]
    delete(mpc.optim, mpc.optim[:linconstraint])
    unregister(mpc.optim, :linconstraint)
    @constraint(mpc.optim, linconstraint, A*ΔŨvar .≤ b)
    setnonlincon!(mpc, model)
    return mpc
end

"By default, there is no nonlinear constraint, thus do nothing."
setnonlincon!(::PredictiveController, ::SimModel) = nothing

@doc raw"""
    moveinput!(
        mpc::PredictiveController, 
        ry = mpc.estim.model.yop, 
        d  = Float64[];
        R̂y = repeat(ry, mpc.Hp), 
        D̂  = repeat(d,  mpc.Hp), 
        ym = nothing
    )

Compute the optimal manipulated input value `u` for the current control period.

Solve the optimization problem of `mpc` [`PredictiveController`](@ref) and return the 
results ``\mathbf{u}(k)``. Following the receding horizon principle, the algorithm discards 
the optimal future manipulated inputs ``\mathbf{u}(k+1), \mathbf{u}(k+2), ...`` The 
arguments `ry` and `d` are current output setpoints ``\mathbf{r_y}(k)`` and measured 
disturbances ``\mathbf{d}(k)``.

The keyword arguments `R̂y` and `D̂` are the predicted output setpoints ``\mathbf{R̂_y}`` and
measured disturbances ``\mathbf{D̂}``. They are assumed constant in the future by default, 
that is ``\mathbf{r̂_y}(k+j) = \mathbf{r_y}(k)`` and ``\mathbf{d̂}(k+j) = \mathbf{d}(k)`` for
``j=1`` to ``H_p``. Current measured output `ym` is only required if `mpc.estim` is a 
[`InternalModel`](@ref).

Calling a [`PredictiveController`](@ref) object calls this method.

See also [`LinMPC`](@ref), [`NonLinMPC`](@ref).

# Examples
```jldoctest
julia> mpc = LinMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1000, Hc=1);

julia> ry = [5]; u = moveinput!(mpc, ry); round.(u, digits=3)
1-element Vector{Float64}:
 1.0
```
"""
function moveinput!(
    mpc::PredictiveController, 
    ry::Vector = mpc.estim.model.yop, 
    d ::Vector = Float64[];
    R̂y::Vector = repeat(ry, mpc.Hp),
    D̂ ::Vector = repeat(d,  mpc.Hp),
    ym::Union{Vector, Nothing} = nothing
)   
    validate_setpointdist(mpc, ry, d, R̂y, D̂)
    getestimates!(mpc, mpc.estim, ym, d)
    predictstoch!(mpc, mpc.estim, d, ym)
    initpred!(mpc, mpc.estim.model, d, D̂, R̂y)
    linconstraint!(mpc, mpc.estim.model)
    ΔŨ = optim_objective!(mpc)
    Δu = ΔŨ[1:mpc.estim.model.nu] # receding horizon principle: only Δu(k) is used (1st one)
    u = mpc.estim.lastu0 + mpc.estim.model.uop + Δu
    return u
end

@doc raw"""
    getinfo(mpc::PredictiveController)

Get additional information about `mpc` controller optimum to ease troubleshooting.

Return the optimizer solution summary that can be printed, `sol_summary`, and the dictionary 
`info` with the following fields:

- `:ΔU` : optimal manipulated input increments over `Hc` ``(\mathbf{ΔU})``
- `:ϵ`  : optimal slack variable ``(ϵ)``
- `:J`  : objective value optimum ``(J)``
- `:U`  : optimal manipulated inputs over `Hp` ``(\mathbf{U})``
- `:u`  : current optimal manipulated input ``(\mathbf{u})``
- `:d`  : current measured disturbance ``(\mathbf{d})``
- `:D̂`  : predicted measured disturbances over `Hp` ``(\mathbf{D̂})``
- `:ŷ`  : current estimated output ``(\mathbf{ŷ})``
- `:Ŷ`  : optimal predicted outputs over `Hp` ``(\mathbf{Ŷ = Ŷ_d + Ŷ_s})``
- `:Ŷd` : optimal predicted deterministic output over `Hp` ``(\mathbf{Ŷ_d})``
- `:Ŷs` : predicted stochastic output over `Hp` ``(\mathbf{Ŷ_s})``
- `:R̂y` : predicted output setpoint over `Hp` ``(\mathbf{R̂_y})``
- `:R̂u` : predicted manipulated input setpoint over `Hp` ``(\mathbf{R̂_u})``

# Examples
```jldoctest
julia> mpc = LinMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1, Hc=1);

julia> u = moveinput!(mpc, [10]);

julia> sol_summary, info = getinfo(mpc); round.(info[:Ŷ], digits=3)
1-element Vector{Float64}:
 10.0
```
"""
function getinfo(mpc::PredictiveController)
    sol_summary = solution_summary(mpc.optim) 
    info = Dict{Symbol, InfoDictType}()
    info[:ΔU]  = mpc.ΔŨ[1:mpc.Hc*mpc.estim.model.nu]
    info[:ϵ]   = isinf(mpc.C) ? NaN : mpc.ΔŨ[end]
    info[:J]   = objective_value(mpc.optim) + mpc.p[begin]
    info[:U]   = mpc.S̃_Hp*mpc.ΔŨ + mpc.T_Hp*(mpc.estim.lastu0 + mpc.estim.model.uop)
    info[:u]   = info[:U][1:mpc.estim.model.nu]
    info[:d]   = mpc.d
    info[:D̂]   = mpc.D̂
    info[:ŷ]   = mpc.ŷ
    info[:Ŷ]   = predict(mpc, mpc.estim.model, mpc.ΔŨ)
    info[:Ŷs]  = mpc.Ŷs
    info[:Ŷd]  = info[:Ŷ] - info[:Ŷs]
    info[:R̂y]  = mpc.R̂y
    info[:R̂u]  = mpc.R̂u
    return sol_summary, info
end

"""
    setstate!(mpc::PredictiveController, x̂)

Set the estimate at `mpc.estim.x̂`.
"""
setstate!(mpc::PredictiveController, x̂) = (setstate!(mpc.estim, x̂); return mpc)

"""
    initstate!(mpc::PredictiveController, u, ym, d=Float64[])

Init `mpc.ΔŨ` for warm-starting and the states of `mpc.estim` [`StateEstimator`](@ref).
"""
function initstate!(mpc::PredictiveController, u, ym, d=Float64[])
    mpc.ΔŨ .= 0
    return initstate!(mpc.estim, u, ym, d)
end


"""
    updatestate!(mpc::PredictiveController, u, ym, d=Float64[])

Call [`updatestate!`](@ref) on `mpc.estim` [`StateEstimator`](@ref).
"""
updatestate!(mpc::PredictiveController, u, ym, d=Float64[]) = updatestate!(mpc.estim,u,ym,d)

function validate_setpointdist(mpc::PredictiveController, ry, d, R̂y, D̂)
    ny, nd, Hp = mpc.estim.model.ny, mpc.estim.model.nd, mpc.Hp
    size(ry) ≠ (ny,)    && error("ry size $(size(ry)) ≠ output size ($ny,)")
    size(d)  ≠ (nd,)    && error("d size $(size(d)) ≠ measured dist. size ($nd,)")
    size(R̂y) ≠ (ny*Hp,) && error("R̂y size $(size(R̂y)) ≠ output size × Hp ($(ny*Hp),)")
    size(D̂)  ≠ (nd*Hp,) && error("D̂ size $(size(D̂)) ≠ measured dist. size × Hp ($(nd*Hp),)")
end

"""
    getestimates!(mpc::PredictiveController, estim::StateEstimator)

Get estimator output and split `x̂` into the deterministic `x̂d` and stochastic `x̂s` states.
"""
function getestimates!(mpc::PredictiveController, estim::StateEstimator, ym , d)
    nx = estim.model.nx
    mpc.x̂d[:] = estim.x̂[1:nx]
    mpc.x̂s[:] = estim.x̂[nx+1:end]
    mpc.ŷ[:]  = evalŷ(estim, ym, d)
    return mpc.x̂d, mpc.x̂s, mpc.ŷ
end

"""
    getestimates!(mpc::PredictiveController, estim::InternalModel)

Get the internal model deterministic `estim.x̂d` and stochastic `estim.x̂s` states.
"""
function getestimates!(mpc::PredictiveController, estim::InternalModel, ym, d)
    isnothing(ym) && error("Predictive controllers with InternalModel need the measured "*
                           "outputs ym in keyword argument to compute control actions u")
    mpc.x̂d[:] = estim.x̂d
    mpc.x̂s[:] = estim.x̂s
    mpc.ŷ[:]  = evalŷ(estim, ym, d)
    return mpc.x̂d, mpc.x̂s, mpc.ŷ
end

"""
    predictstoch!(mpc, estim::StateEstimator, x̂s, d, _ )

Predict the future `Ŷs` stochastic model outputs over `Hp`. 

See [`init_stochpred`](@ref) for details on `Ŷs` and `Ks` matrices.
"""
function predictstoch!(mpc, estim::StateEstimator, _ , _ )
    mpc.Ŷs[:] = mpc.Ks*mpc.x̂s 
    return mpc.Ŷs
end

"""
    predictstoch!(mpc, estim::InternalModel, x̂s, d, ym )

Use current measured outputs `ym` for prediction when `estim` is a [`InternalModel`](@ref).
"""
function predictstoch!(mpc, estim::InternalModel, d, ym )
    ŷd = h(estim.model, mpc.x̂d, d - estim.model.dop) + estim.model.yop 
    ŷs = zeros(estim.model.ny)
    ŷs[estim.i_ym] = ym - ŷd[estim.i_ym]  # ŷs=0 for unmeasured outputs
    mpc.Ŷs[:] = mpc.Ks*mpc.x̂s + mpc.Ps*ŷs
    return mpc.Ŷs
end


@doc raw"""
    initpred!(mpc, model::LinModel, d, D̂, R̂y)

Init linear model prediction matrices `F`, `q̃` and `p`.

See [`init_deterpred`](@ref) and [`init_quadprog`](@ref) for the definition of the matrices.
"""
function initpred!(mpc::PredictiveController, model::LinModel, d, D̂, R̂y)
    mpc.F[:] = mpc.Kd*mpc.x̂d + mpc.Q*mpc.estim.lastu0 + mpc.Yop + mpc.Ŷs 
    if model.nd ≠ 0
        mpc.d[:], mpc.D̂[:] = d, D̂
        mpc.F[:] = mpc.F + mpc.G*(mpc.d - model.dop) + mpc.J*(mpc.D̂ - mpc.Dop)
    end
    mpc.R̂y[:] = R̂y
    Ẑ = mpc.F - R̂y
    mpc.q̃[:] = 2(mpc.M_Hp*mpc.Ẽ)'*Ẑ
    mpc.p[:] = [Ẑ'*mpc.M_Hp*Ẑ]
    if ~isempty(mpc.R̂u)
        lastu = mpc.estim.lastu0 + model.uop
        V̂ = mpc.T_Hp*lastu - mpc.R̂u
        mpc.q̃[:] += 2(mpc.L_Hp*mpc.S̃_Hp)'*V̂
        mpc.p[:] += [V̂'*mpc.L_Hp*V̂]
    end
    return nothing
end

@doc raw"""
    initpred!(mpc::PredictiveController, model::SimModel, d, D̂, R̂y)

Init `d0` and `D̂0` matrices when model is not a [`LinModel`](@ref).

`d0` and `D̂0` are the measured disturbances and its predictions without the operating points
``\mathbf{d_{op}}``.
"""
function initpred!(mpc::PredictiveController, model::SimModel, d, D̂, R̂y)
    if model.nd ≠ 0
        mpc.d[:], mpc.D̂[:] = d, D̂
    end
    mpc.R̂y[:] = R̂y
    return nothing
end

"""
    predict(mpc::PredictiveController, model::LinModel, ΔŨ)

Evaluate the outputs predictions ``\\mathbf{Ŷ}`` when `model` is a [`LinModel`](@ref).
"""
function predict(mpc::PredictiveController, ::LinModel, ΔŨ::Vector{T}) where {T<:Real}
    return mpc.Ẽ*ΔŨ + mpc.F
end

"""
    predict(mpc::PredictiveController, model::SimModel, ΔŨ)

Evaluate  ``\\mathbf{Ŷ}`` when `model` is not a [`LinModel`](@ref).
"""
function predict(mpc::PredictiveController, model::SimModel, ΔŨ::Vector{T}) where {T<:Real}
    nu, ny, nd, Hp = model.nu, model.ny, model.nd, mpc.Hp
    yop, dop = model.yop, model.dop
    U0 = mpc.S̃_Hp*ΔŨ + mpc.T_Hp*(mpc.estim.lastu0)
    Ŷd::Vector{T} = Vector{T}(undef, ny*Hp)
    u0::Vector{T} = Vector{T}(undef, nu)
    x̂d::Vector{T} = copy(mpc.x̂d)
    d0 = mpc.d - dop
    for j=1:Hp
        u0[:] = @views U0[(1 + nu*(j-1)):(nu*j)]
        x̂d[:] = f(model, x̂d, u0, d0)
        d0[:] = @views mpc.D̂[(1 + nd*(j-1)):(nd*j)] - dop
        Ŷd[(1 + ny*(j-1)):(ny*j)] = h(model, x̂d, d0) + yop
    end
    return Ŷd + mpc.Ŷs
end

@doc raw"""
    linconstraint!(mpc::PredictiveController, model::LinModel)

Set `b` vector for the linear model inequality constraints (``\mathbf{A ΔŨ ≤ b}``).
"""
function linconstraint!(mpc::PredictiveController, model::LinModel)
    mpc.con.b[:] = [
        -mpc.con.Umin + mpc.T_Hc*(mpc.estim.lastu0 + model.uop)
        +mpc.con.Umax - mpc.T_Hc*(mpc.estim.lastu0 + model.uop)
        -mpc.con.ΔŨmin
        +mpc.con.ΔŨmax 
        -mpc.con.Ŷmin + mpc.F
        +mpc.con.Ŷmax - mpc.F
    ]
    lincon::LinConVector = mpc.optim[:linconstraint]
    set_normalized_rhs.(lincon, mpc.con.b[mpc.con.i_b])
end

"Set `b` excluding predicted output constraints when `model` is not a [`LinModel`](@ref)."
function linconstraint!(mpc::PredictiveController, model::SimModel)
    mpc.con.b[:] = [
        -mpc.con.Umin + mpc.T_Hc*(mpc.estim.lastu0 + model.uop)
        +mpc.con.Umax - mpc.T_Hc*(mpc.estim.lastu0 + model.uop)
        -mpc.con.ΔŨmin
        +mpc.con.ΔŨmax 
    ]
    lincon::LinConVector = mpc.optim[:linconstraint]
    set_normalized_rhs.(lincon, mpc.con.b[mpc.con.i_b])
end

"""
    optim_objective!(mpc::PredictiveController, b, p)

Optimize the objective function ``J`` of `mpc` controller. 
"""
function optim_objective!(mpc::PredictiveController)
    optim = mpc.optim
    model = mpc.estim.model
    ΔŨvar::Vector{VariableRef} = optim[:ΔŨvar]
    lastΔŨ = mpc.ΔŨ
    # initial ΔŨ (warm-start): [Δu_{k-1}(k); Δu_{k-1}(k+1); ... ; 0_{nu × 1}]
    ΔŨ0 = [lastΔŨ[(model.nu+1):(mpc.Hc*model.nu)]; zeros(model.nu)]
    # if soft constraints, append the last slack value ϵ_{k-1}:
    !isinf(mpc.C) && (ΔŨ0 = [ΔŨ0; lastΔŨ[end]])
    set_start_value.(ΔŨvar, ΔŨ0)
    set_objective_linear_coef!(mpc, ΔŨvar)
    try
        optimize!(optim)
    catch err
        if isa(err, MOI.UnsupportedAttribute{MOI.VariablePrimalStart})
            # reset_optimizer to unset warm-start, set_start_value.(nothing) seems buggy
            MOIU.reset_optimizer(optim)
            optimize!(optim)
        else
            rethrow(err)
        end
    end
    status = termination_status(optim)
    if !(status == OPTIMAL || status == LOCALLY_SOLVED)
        @warn "MPC termination status not OPTIMAL or LOCALLY_SOLVED ($status)"
        @debug solution_summary(optim)
    end
    mpc.ΔŨ[:] = isfatal(status) ? ΔŨ0 : value.(ΔŨvar) # fatal status : use last value
    return mpc.ΔŨ
end

"By default, no need to modify the objective function."
set_objective_linear_coef!(::PredictiveController, _ ) = nothing

@doc raw"""
    init_ΔUtoU(nu, Hp, Hc, C, c_Umin, c_Umax)

Init manipulated input increments to inputs conversion matrices.

The conversion from the input increments ``\mathbf{ΔU}`` to manipulated inputs over ``H_p`` 
and ``H_c`` are calculated by:
```math
\begin{aligned}
\mathbf{U} = 
    \mathbf{U}_{H_p} &= \mathbf{S}_{H_p} \mathbf{ΔU} + \mathbf{T}_{H_p} \mathbf{u}(k-1) \\
    \mathbf{U}_{H_c} &= \mathbf{S}_{H_c} \mathbf{ΔU} + \mathbf{T}_{H_c} \mathbf{u}(k-1)
\end{aligned}
```
"""
function init_ΔUtoU(nu, Hp, Hc)
    S_Hc = LowerTriangular(repeat(I(nu), Hc, Hc))
    T_Hc = repeat(I(nu), Hc)
    S_Hp = [S_Hc; repeat(I(nu), Hp - Hc, Hc)]
    T_Hp = [T_Hc; repeat(I(nu), Hp - Hc, 1)]
    return S_Hp, T_Hp, S_Hc, T_Hc
end


@doc raw"""
    init_deterpred(model::LinModel, Hp, Hc)

Construct deterministic prediction matrices for [`LinModel`](@ref) `model`.

The linear model predictions are evaluated by :
```math
\begin{aligned}
    \mathbf{Ŷ} &= \mathbf{E ΔU} + \mathbf{G d}(k) + \mathbf{J D̂} + \mathbf{K_d x̂_d}(k) 
                                                  + \mathbf{Q u}(k-1) + \mathbf{Ŷ_s}     \\
               &= \mathbf{E ΔU} + \mathbf{F}
\end{aligned}
```
where predicted outputs ``\mathbf{Ŷ}``, stochastic outputs ``\mathbf{Ŷ_s}``, and 
measured disturbances ``\mathbf{D̂}`` are from ``k + 1`` to ``k + H_p``. Input increments 
``\mathbf{ΔU}`` are from ``k`` to ``k + H_c - 1``. Deterministic state estimates 
``\mathbf{x̂_d}(k)`` are extracted from current estimates ``\mathbf{x̂}_{k-1}(k)`` with
[`getestimates!`](@ref). Operating points on ``\mathbf{u}``, ``\mathbf{d}`` and ``\mathbf{y}`` 
are omitted in above equations.

# Extended Help
Using the ``\mathbf{A, B_u, C, B_d, D_d}`` matrices in `model` and the equation
``\mathbf{W}_j = \mathbf{C} ( ∑_{i=0}^j \mathbf{A}^i ) \mathbf{B_u}``, the prediction 
matrices are computed by :
```math
\begin{aligned}
\mathbf{E} &= \begin{bmatrix}
\mathbf{W}_{0}      & \mathbf{0}         & \cdots & \mathbf{0}              \\
\mathbf{W}_{1}      & \mathbf{W}_{0}     & \cdots & \mathbf{0}              \\
\vdots              & \vdots             & \ddots & \vdots                  \\
\mathbf{W}_{H_p-1}  & \mathbf{W}_{H_p-2} & \cdots & \mathbf{W}_{H_p-H_c+1}
\end{bmatrix}
\\
\mathbf{G} &= \begin{bmatrix}
\mathbf{C}\mathbf{A}^{0} \mathbf{B_d}     \\ 
\mathbf{C}\mathbf{A}^{1} \mathbf{B_d}     \\ 
\vdots                                    \\
\mathbf{C}\mathbf{A}^{H_p-1} \mathbf{B_d}
\end{bmatrix}
\\
\mathbf{J} &= \begin{bmatrix}
\mathbf{D_d}                              & \mathbf{0}                                & \cdots & \mathbf{0}   \\ 
\mathbf{C}\mathbf{A}^{0} \mathbf{B_d}     & \mathbf{D_d}                              & \cdots & \mathbf{0}   \\ 
\vdots                                    & \vdots                                    & \ddots & \vdots       \\
\mathbf{C}\mathbf{A}^{H_p-2} \mathbf{B_d} & \mathbf{C}\mathbf{A}^{H_p-3} \mathbf{B_d} & \cdots & \mathbf{D_d}
\end{bmatrix}
\\
\mathbf{K_d} &= \begin{bmatrix}
\mathbf{C}\mathbf{A}^{0}      \\
\mathbf{C}\mathbf{A}^{1}      \\
\vdots                        \\
\mathbf{C}\mathbf{A}^{H_p-1}
\end{bmatrix}
\\
\mathbf{Q} &= \begin{bmatrix}
\mathbf{W}_0        \\
\mathbf{W}_1        \\
\vdots              \\
\mathbf{W}_{H_p-1}
\end{bmatrix}
\end{aligned}
```
!!! note
    Stochastic predictions ``\mathbf{Ŷ_s}`` are calculated separately (see 
    [`init_stochpred`](@ref)) and added to the ``\mathbf{F}`` matrix to support internal 
    model structure and reduce [`NonLinMPC`](@ref) computational costs. That is also why the 
    prediction matrices are built on ``\mathbf{A, B_u, C, B_d, D_d}`` instead of the 
    augmented model ``\mathbf{Â, B̂_u, Ĉ, B̂_d, D̂_d}``.
"""
function init_deterpred(model::LinModel, Hp, Hc)
    A, Bu, C, Bd, Dd = model.A, model.Bu, model.C, model.Bd, model.Dd
    nu, nx, ny, nd = model.nu, model.nx, model.ny, model.nd
    # Apow 3D array : Apow[:,:,1] = A^0, Apow[:,:,2] = A^1, ...
    Apow = Array{Float64}(undef, size(A,1), size(A,2), Hp+1)
    Apow[:,:,1] = I(nx)
    Kd = Matrix{Float64}(undef, Hp*ny, nx)
    for i=1:Hp
        Apow[:,:,i+1] = A^i
        iRow = (1:ny) .+ ny*(i-1)
        Kd[iRow,:] = C*Apow[:,:,i+1]
    end 
    # Apow_csum 3D array : Apow_csum[:,:,1] = A^0, Apow_csum[:,:,2] = A^1 + A^0, ...
    Apow_csum  = cumsum(Apow, dims=3)
    # --- manipulated inputs u ---
    Q = Matrix{Float64}(undef, Hp*ny, nu)
    for i=1:Hp
        iRow = (1:ny) .+ ny*(i-1)
        Q[iRow,:] = C*Apow_csum[:,:,i]*Bu
    end
    E = zeros(Hp*ny, Hc*nu) 
    for i=1:Hc # truncated with control horizon
        iRow = (ny*(i-1)+1):(ny*Hp)
        iCol = (1:nu) .+ nu*(i-1)
        E[iRow,iCol] = Q[iRow .- ny*(i-1),:]
    end
    # --- measured disturbances d ---
    G = Matrix{Float64}(undef, Hp*ny, nd)
    J = repeatdiag(Dd, Hp)
    if nd ≠ 0
        for i=1:Hp
            iRow = (1:ny) .+ ny*(i-1)
            G[iRow,:] = C*Apow[:,:,i]*Bd
        end
        for i=1:Hp
            iRow = (ny*i+1):(ny*Hp)
            iCol = (1:nd) .+ nd*(i-1)
            J[iRow,iCol] = G[iRow .- ny*i,:]
        end
    end
    F = zeros(ny*Hp) # dummy value (updated just before optimization)
    return E, F, G, J, Kd, Q
end

"Return empty matrices if `model` is not a [`LinModel`](@ref)"
function init_deterpred(model::SimModel, Hp, Hc)
    nu, nx, ny, nd = model.nu, model.nx, model.ny, model.nd
    E  = zeros(0, nu*Hc)
    G  = zeros(0, nd)
    J  = zeros(0, nd*Hp)
    Kd = zeros(0, nx)
    Q  = zeros(0, nu)
    F  = zeros(0)
    return E, F, G, J, Kd, Q
end

@doc raw"""
    init_quadprog(model::LinModel, Ẽ, S_Hp, M_Hp, N_Hc, L_Hp)

Init the quadratic programming optimization matrix `P̃` and `q̃`.

The matrices appear in the quadratic general form :
```math
    J = \min_{\mathbf{ΔŨ}} \frac{1}{2}\mathbf{(ΔŨ)'P̃(ΔŨ)} + \mathbf{q̃'(ΔŨ)} + p 
```
``\mathbf{P̃}`` is constant if the model and weights are linear and time invariant (LTI). The 
vector ``\mathbf{q̃}`` and scalar ``p`` need recalculation each control period ``k`` (see
[`initpred!`](@ref) method). ``p`` does not impact the minima position. It is thus 
useless at optimization but required to evaluate the minimal ``J`` value.
"""
function init_quadprog(::LinModel, Ẽ, S_Hp, M_Hp, N_Hc, L_Hp)
    P̃ = 2*Hermitian(Ẽ'*M_Hp*Ẽ + N_Hc + S_Hp'*L_Hp*S_Hp)
    q̃ = zeros(size(P̃, 1))   # dummy value (updated just before optimization)
    p = zeros(1)            # dummy value (updated just before optimization)
    return P̃, q̃, p
end
"Return empty matrices if `model` is not a [`LinModel`](@ref)."
function init_quadprog(::SimModel, Ẽ, S_Hp, M_Hp, N_Hc, L_Hp)
    P̃ = Hermitian(zeros(0, 0))
    q̃ = zeros(0)
    p = zeros(1)            # dummy value (updated just before optimization)
    return P̃, q̃, p
end

"Return the quadratic programming objective function, see [`init_quadprog`](@ref)."
obj_quadprog(ΔŨ, P̃, q̃) = 0.5*ΔŨ'*P̃*ΔŨ + q̃'*ΔŨ

"""
    init_defaultcon(model, C, S_Hp, S_Hc, N_Hc, E)

Init `ControllerConstraint` struct with default parameters.

Also return `S̃_Hp`, `Ñ_Hc` and `Ẽ` matrices for the the augmented decision vector `ΔŨ`.
"""
function init_defaultcon(model, Hp, Hc, C, S_Hp, S_Hc, N_Hc, E)
    nu, ny = model.nu, model.ny
    umin,       umax    = fill(-Inf, nu), fill(+Inf, nu)
    Δumin,      Δumax   = fill(-Inf, nu), fill(+Inf, nu)
    ŷmin,       ŷmax    = fill(-Inf, ny), fill(+Inf, ny)
    c_umin,     c_umax  = fill(0.0, nu),  fill(0.0, nu)
    c_Δumin,    c_Δumax = fill(0.0, nu),  fill(0.0, nu)
    c_ŷmin,     c_ŷmax  = fill(1.0, ny),  fill(1.0, ny)
    Umin, Umax, ΔUmin, ΔUmax, Ŷmin, Ŷmax = 
        repeat_constraints(Hp, Hc, umin, umax, Δumin, Δumax, ŷmin, ŷmax)
    c_Umin, c_Umax, c_ΔUmin, c_ΔUmax, c_Ŷmin, c_Ŷmax = 
        repeat_constraints(Hp, Hc, c_umin, c_umax, c_Δumin, c_Δumax, c_ŷmin, c_ŷmax)
    A_Umin, A_Umax, S̃_Hp = relaxU(C, c_Umin, c_Umax, S_Hp, S_Hc)
    A_ΔŨmin, A_ΔŨmax, ΔŨmin, ΔŨmax, Ñ_Hc = relaxΔU(C, c_ΔUmin, c_ΔUmax, ΔUmin, ΔUmax, N_Hc)
    A_Ŷmin, A_Ŷmax, Ẽ = relaxŶ(model, C, c_Ŷmin, c_Ŷmax, E)
    i_Umin,  i_Umax  = .!isinf.(Umin),  .!isinf.(Umax)
    i_ΔŨmin, i_ΔŨmax = .!isinf.(ΔŨmin), .!isinf.(ΔŨmax)
    i_Ŷmin,  i_Ŷmax  = .!isinf.(Ŷmin),  .!isinf.(Ŷmax)
    A, i_b, b = init_linconstraint(
        model, 
        A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, A_Ŷmin, A_Ŷmax,
        i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ŷmin, i_Ŷmax
    )
    con = ControllerConstraint(
        Umin    , Umax  , ΔŨmin  , ΔŨmax    , Ŷmin  , Ŷmax,
        A_Umin  , A_Umax, A_ΔŨmin, A_ΔŨmax  , A_Ŷmin, A_Ŷmax,
        A       , b     , i_b    , c_Ŷmin   , c_Ŷmax 
    )
    return con, S̃_Hp, Ñ_Hc, Ẽ
end

"Repeat predictive controller constraints over prediction `Hp` and control `Hc` horizons."
function repeat_constraints(Hp, Hc, umin, umax, Δumin, Δumax, ŷmin, ŷmax)
    Umin  = repeat(umin, Hc)
    Umax  = repeat(umax, Hc)
    ΔUmin = repeat(Δumin, Hc)
    ΔUmax = repeat(Δumax, Hc)
    Ŷmin  = repeat(ŷmin, Hp)
    Ŷmax  = repeat(ŷmax, Hp)
    return Umin, Umax, ΔUmin, ΔUmax, Ŷmin, Ŷmax
end

@doc raw"""
    relaxU(C, c_Umin, c_Umax, S_Hp, S_Hc)

Augment manipulated inputs constraints with slack variable ϵ for softening.

Denoting the input increments augmented with the slack variable
``\mathbf{ΔŨ} = [\begin{smallmatrix} \mathbf{ΔU} \\ ϵ \end{smallmatrix}]``, it returns the
augmented conversion matrix ``\mathbf{S̃}_{H_p}``, similar to the one described at
[`init_ΔUtoU`](@ref). It also returns the ``\mathbf{A}`` matrices for the inequality
constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{U_{min}}} \\ 
    \mathbf{A_{U_{max}}} 
\end{bmatrix} \mathbf{ΔŨ} ≤
\begin{bmatrix}
    - \mathbf{U_{min}} + \mathbf{T}_{H_c} \mathbf{u}(k-1) \\
    + \mathbf{U_{max}} - \mathbf{T}_{H_c} \mathbf{u}(k-1)
\end{bmatrix}
```
"""
function relaxU(C, c_Umin, c_Umax, S_Hp, S_Hc)
    if !isinf(C) # ΔŨ = [ΔU; ϵ]
        # ϵ impacts ΔU → U conversion for constraint calculations:
        A_Umin, A_Umax = -[S_Hc  c_Umin],  [S_Hc -c_Umax] 
        # ϵ has no impact on ΔU → U conversion for prediction calculations:
        S̃_Hp = [S_Hp falses(size(S_Hp, 1))]
    else # ΔŨ = ΔU (only hard constraints)
        A_Umin, A_Umax = -S_Hc,  S_Hc
        S̃_Hp = S_Hp
    end
    return A_Umin, A_Umax, S̃_Hp
end

@doc raw"""
    relaxΔU(C, c_ΔUmin, c_ΔUmax, ΔUmin, ΔUmax, N_Hc)

Augment input increments constraints with slack variable ϵ for softening.

Denoting the input increments augmented with the slack variable 
``\mathbf{ΔŨ} = [\begin{smallmatrix} \mathbf{ΔU} \\ ϵ \end{smallmatrix}]``, it returns the
augmented input increment weights ``\mathbf{Ñ}_{H_c}`` (that incorporate ``C``). It also  
returns the augmented constraints ``\mathbf{ΔŨ_{min}}`` and ``\mathbf{ΔŨ_{max}}`` and the 
``\mathbf{A}`` matrices for the inequality constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{ΔŨ_{min}}} \\ 
    \mathbf{A_{ΔŨ_{max}}}
\end{bmatrix} \mathbf{ΔŨ} ≤
\begin{bmatrix}
    - \mathbf{ΔŨ_{min}} \\
    + \mathbf{ΔŨ_{max}}
\end{bmatrix}
```
"""
function relaxΔU(C, c_ΔUmin, c_ΔUmax, ΔUmin, ΔUmax, N_Hc)
    if !isinf(C) # ΔŨ = [ΔU; ϵ]
        # 0 ≤ ϵ ≤ ∞  
        ΔŨmin, ΔŨmax = [ΔUmin; 0.0], [ΔUmax; Inf]
        A_ϵ = [zeros(1, length(ΔUmin)) [1]]
        A_ΔŨmin, A_ΔŨmax = -[I  c_ΔUmin; A_ϵ], [I -c_ΔUmax; A_ϵ]
        Ñ_Hc = Diagonal([diag(N_Hc); C])
    else # ΔŨ = ΔU (only hard constraints)
        ΔŨmin, ΔŨmax = ΔUmin, ΔUmax
        I_Hc = Matrix{Float64}(I, size(N_Hc))
        A_ΔŨmin, A_ΔŨmax = -I_Hc,  I_Hc
        Ñ_Hc = N_Hc
    end
    return A_ΔŨmin, A_ΔŨmax, ΔŨmin, ΔŨmax, Ñ_Hc
end

@doc raw"""
    relaxŶ(::LinModel, C, c_Ŷmin, c_Ŷmax, E)

Augment linear output prediction constraints with slack variable ϵ for softening.

Denoting the input increments augmented with the slack variable 
``\mathbf{ΔŨ} = [\begin{smallmatrix} \mathbf{ΔU} \\ ϵ \end{smallmatrix}]``, it returns the 
``\mathbf{Ẽ}`` matrix that appears in the linear model prediction equation 
``\mathbf{Ŷ = Ẽ ΔŨ + F}``, and the ``\mathbf{A}`` matrices for the inequality constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{Ŷ_{min}}} \\ 
    \mathbf{A_{Ŷ_{max}}}
\end{bmatrix} \mathbf{ΔŨ} ≤
\begin{bmatrix}
    - \mathbf{Ŷ_{min}} + \mathbf{F} \\
    + \mathbf{Ŷ_{max}} - \mathbf{F} 
\end{bmatrix}
```
"""
function relaxŶ(::LinModel, C, c_Ŷmin, c_Ŷmax, E)
    if !isinf(C) # ΔŨ = [ΔU; ϵ]
        # ϵ impacts predicted output constraint calculations:
        A_Ŷmin, A_Ŷmax = -[E  c_Ŷmin], [E -c_Ŷmax] 
        # ϵ has not impact on output predictions
        Ẽ = [E zeros(size(E, 1), 1)] 
    else # ΔŨ = ΔU (only hard constraints)
        Ẽ = E
        A_Ŷmin, A_Ŷmax = -E,  E
    end
    return A_Ŷmin, A_Ŷmax, Ẽ
end

"Return empty matrices if model is not a [`LinModel`](@ref)"
function relaxŶ(::SimModel, C, c_Ŷmin, c_Ŷmax, E)
    Ẽ = !isinf(C) ? [E zeros(0, 1)] : E
    A_Ŷmin, A_Ŷmax = Ẽ, Ẽ 
    return A_Ŷmin, A_Ŷmax, Ẽ
end

@doc raw"""
    init_stochpred(estim::StateEstimator, Hp)

Init the stochastic prediction matrix `Ks` from `estim` estimator for predictive control.

``\mathbf{K_s}`` is the prediction matrix of the stochastic model (composed exclusively of 
integrators):
```math
    \mathbf{Ŷ_s} = \mathbf{K_s x̂_s}(k)
```
The stochastic predictions ``\mathbf{Ŷ_s}`` are the integrator outputs from ``k+1`` to 
``k+H_p``. ``\mathbf{x̂_s}(k)`` is extracted from current estimates ``\mathbf{x̂}_{k-1}(k)``
with [`getestimates!`](@ref). The method also returns an empty ``\mathbf{P_s}`` matrix, since 
it is useless except for [`InternalModel`](@ref) estimators.
"""
function init_stochpred(estim::StateEstimator, Hp)
    As, Cs = estim.As, estim.Cs
    nxs = estim.nxs
    Ms = Matrix{Float64}(undef, Hp*nxs, nxs) 
    for i = 1:Hp
        iRow = (1:nxs) .+ nxs*(i-1)
        Ms[iRow, :] = As^i
    end
    Js = repeatdiag(Cs, Hp)
    Ks = Js*Ms
    Ps = zeros(estim.model.ny*Hp, 0)
    return Ks, Ps
end

@doc raw"""
    init_stochpred(estim::InternalModel, Hp)

Init the stochastic prediction matrices for [`InternalModel`](@ref).

`Ks` and `Ps` matrices are defined as:
```math
    \mathbf{Ŷ_s} = \mathbf{K_s x̂_s}(k) + \mathbf{P_s ŷ_s}(k)
```
Current stochastic outputs ``\mathbf{ŷ_s}(k)`` comprises the measured outputs 
``\mathbf{ŷ_s^m}(k) = \mathbf{y^m}(k) - \mathbf{ŷ_d^m}(k)`` and unmeasured 
``\mathbf{ŷ_s^u}(k) = \mathbf{0}``. See [^2].

[^2]: Desbiens, A., D. Hodouin & É. Plamondon. 2000, "Global predictive control : a unified
    control structure for decoupling setpoint tracking, feedforward compensation and 
    disturbance rejection dynamics", *IEE Proceedings - Control Theory and Applications*, 
    vol. 147, no 4, https://doi.org/10.1049/ip-cta:20000443, p. 465–475, ISSN 1350-2379.
"""
function init_stochpred(estim::InternalModel, Hp) 
    As, B̂s, Cs = estim.As, estim.B̂s, estim.Cs
    ny  = estim.model.ny
    nxs = estim.nxs
    Ks = Matrix{Float64}(undef, ny*Hp, nxs)
    Ps = Matrix{Float64}(undef, ny*Hp, ny)
    for i = 1:Hp
        iRow = (1:ny) .+ ny*(i-1)
        Ms = Cs*As^(i-1)*B̂s
        Ks[iRow,:] = Cs*As^i - Ms*Cs
        Ps[iRow,:] = Ms
    end
    return Ks, Ps 
end

@doc raw"""
    init_linconstraint(model::LinModel, 
        A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, A_Ŷmin, A_Ŷmax,
        i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ŷmin, i_Ŷmax
    )

Init `A`, `b` and `i_b` for the linear inequality constraints (``\mathbf{A ΔŨ ≤ b}``).

`i_b` is a `BitVector` including the indices of ``\mathbf{b}`` that are finite numbers.
"""
function init_linconstraint(::LinModel, 
    A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, A_Ŷmin, A_Ŷmax,
    i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ŷmin, i_Ŷmax
)
    A   = [A_Umin; A_Umax; A_ΔŨmin; A_ΔŨmax; A_Ŷmin; A_Ŷmax]
    i_b = [i_Umin; i_Umax; i_ΔŨmin; i_ΔŨmax; i_Ŷmin; i_Ŷmax]
    b   = zeros(size(A, 1)) # dummy b vector (updated just before optimization)
    return A, i_b, b
end

"Init values without predicted output constraints if `model` is not a [`LinModel`](@ref)."
function init_linconstraint(::SimModel,
    A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, _ , _ ,
    i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, _ , _ 
)
    A   = [A_Umin; A_Umax; A_ΔŨmin; A_ΔŨmax]
    i_b = [i_Umin; i_Umax; i_ΔŨmin; i_ΔŨmax]
    b   = zeros(size(A, 1)) # dummy b vector (updated just before optimization)
    return A, i_b, b
end

"Validate predictive controller weight and horizon specified values."
function validate_weights(model, Hp, Hc, Mwt, Nwt, Lwt, Cwt, ru, Ewt=nothing)
    nu, ny = model.nu, model.ny
    Hp < 1  && error("Prediction horizon Hp should be ≥ 1")
    Hc < 1  && error("Control horizon Hc should be ≥ 1")
    Hc > Hp && error("Control horizon Hc should be ≤ prediction horizon Hp")
    size(Mwt) ≠ (ny,) && error("Mwt size $(size(Mwt)) ≠ output size ($ny,)")
    size(Nwt) ≠ (nu,) && error("Nwt size $(size(Nwt)) ≠ manipulated input size ($nu,)")
    size(Lwt) ≠ (nu,) && error("Lwt size $(size(Lwt)) ≠ manipulated input size ($nu,)")
    size(ru)  ≠ (nu,) && error("ru size $(size(ru)) ≠ manipulated input size ($nu,)")
    size(Cwt) ≠ ()    && error("Cwt should be a real scalar")
    any(Mwt.<0) && error("Mwt weights should be ≥ 0")
    any(Nwt.<0) && error("Nwt weights should be ≥ 0")
    any(Lwt.<0) && error("Lwt weights should be ≥ 0")
    Cwt < 0     && error("Cwt weight should be ≥ 0")
    !isnothing(Ewt) && size(Ewt) ≠ () && error("Ewt should be a real scalar")
end

"Generate a block diagonal matrix repeating `n` times the matrix `A`."
repeatdiag(A, n::Int) = kron(I(n), A)

function Base.show(io::IO, mpc::PredictiveController)
    Hp, Hc = mpc.Hp, mpc.Hc
    nu, nd = mpc.estim.model.nu, mpc.estim.model.nd
    nx̂, nym, nyu = mpc.estim.nx̂, mpc.estim.nym, mpc.estim.nyu
    n = maximum(ndigits.((Hp, Hc, nu, nx̂, nym, nyu, nd))) + 1
    println(io, "$(typeof(mpc).name.name) controller with a sample time Ts = "*
                "$(mpc.estim.model.Ts) s, $(typeof(mpc.estim).name.name) estimator and:")
    println(io, "$(lpad(Hp, n)) prediction steps Hp")
    println(io, "$(lpad(Hc, n)) control steps Hc")
    println(io, "$(lpad(nu, n)) manipulated inputs u")
    println(io, "$(lpad(nx̂, n)) states x̂")
    println(io, "$(lpad(nym, n)) measured outputs ym")
    println(io, "$(lpad(nyu, n)) unmeasured outputs yu")
    print(io,   "$(lpad(nd, n)) measured disturbances d")
end

"Verify that the solver termination status means 'no solution available'."
function isfatal(status::TerminationStatusCode)
    fatalstatuses = [
        INFEASIBLE, DUAL_INFEASIBLE, LOCALLY_INFEASIBLE, INFEASIBLE_OR_UNBOUNDED, 
        SLOW_PROGRESS, NUMERICAL_ERROR, INVALID_MODEL, INVALID_OPTION, INTERRUPTED, 
        OTHER_ERROR
    ]
    return any(status .== fatalstatuses)
end

"Functor allowing callable `PredictiveController` object as an alias for `moveinput!`."
function (mpc::PredictiveController)(
    ry::Vector = mpc.estim.model.yop, 
    d ::Vector = Float64[];
    kwargs...
)
    return moveinput!(mpc, ry, d; kwargs...)
end

include("controller/linmpc.jl")
include("controller/nonlinmpc.jl")
