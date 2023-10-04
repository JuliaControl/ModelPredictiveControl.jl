@doc raw"""
Abstract supertype of all predictive controllers.

---

    (mpc::PredictiveController)(ry, d=[]; kwargs...) -> u

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

const DEFAULT_HP0 = 10
const DEFAULT_HC  = 2
const DEFAULT_MWT = 1.0
const DEFAULT_NWT = 0.1
const DEFAULT_LWT = 0.0
const DEFAULT_CWT = 1e5
const DEFAULT_EWT = 0.0

"Type alias for vector of linear inequality constraints."
const LinConVector = Vector{ConstraintRef{
    Model, 
    MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}, 
    ScalarShape
}}

const InfoDictType = Union{JuMP._SolutionSummary, Vector{Float64}, Float64}

"Include all the data for the constraints of [`PredictiveController`](@ref)"
struct ControllerConstraint
    Umin   ::Vector{Float64}
    Umax   ::Vector{Float64}
    ΔŨmin  ::Vector{Float64}
    ΔŨmax  ::Vector{Float64}
    Ymin   ::Vector{Float64}
    Ymax   ::Vector{Float64}
    A_Umin ::Matrix{Float64}
    A_Umax ::Matrix{Float64}
    A_ΔŨmin::Matrix{Float64}
    A_ΔŨmax::Matrix{Float64}
    A_Ymin ::Matrix{Float64}
    A_Ymax ::Matrix{Float64}
    A      ::Matrix{Float64}
    b      ::Vector{Float64}
    i_b    ::BitVector
    c_Ymin ::Vector{Float64}
    c_Ymax ::Vector{Float64}
end

@doc raw"""
    setconstraint!(mpc::PredictiveController; <keyword arguments>) -> mpc

Set the constraint parameters of `mpc` predictive controller.

The predictive controllers support both soft and hard constraints, defined by:
```math 
\begin{alignat*}{3}
    \mathbf{u_{min}  - c_{u_{min}}}  ϵ ≤&&\       \mathbf{u}(k+j) &≤ \mathbf{u_{max}  + c_{u_{max}}}  ϵ &&\qquad  j = 0, 1 ,..., H_p - 1 \\
    \mathbf{Δu_{min} - c_{Δu_{min}}} ϵ ≤&&\      \mathbf{Δu}(k+j) &≤ \mathbf{Δu_{max} + c_{Δu_{max}}} ϵ &&\qquad  j = 0, 1 ,..., H_c - 1 \\
    \mathbf{y_{min}  - c_{y_{min}}}  ϵ ≤&&\       \mathbf{ŷ}(k+j) &≤ \mathbf{y_{max}  + c_{y_{max}}}  ϵ &&\qquad  j = 1, 2 ,..., H_p     \\
    \mathbf{x̂_{min}  - c_{x̂_{min}}}  ϵ ≤&&\ \mathbf{x̂}_{k-1}(k+j) &≤ \mathbf{x̂_{max}  + c_{x̂_{max}}}  ϵ &&\qquad  j = H_p
\end{alignat*}
```
and also ``ϵ ≥ 0``. The last line is the terminal constraints applied on the states at the
end of the horizon only (see Extended Help). All the constraint parameters are vector. Use
`±Inf` values when there is no bound. The constraint softness parameters ``\mathbf{c}``,
also called equal concern for relaxation, are non-negative values that specify the softness
of the associated bound. Use `0.0` values for hard constraints. The output and terminal 
constraints are all soft by default. See Extended Help for time-varying constraints.

# Arguments
!!! info
    The default constraints are mentioned here for clarity but omitting a keyword argument 
    will not re-assign to its default value (defaults are set at construction only).

- `mpc::PredictiveController` : predictive controller to set constraints.
- `umin  = fill(-Inf,nu)` : manipulated input lower bounds ``\mathbf{u_{min}}``.
- `umax  = fill(+Inf,nu)` : manipulated input upper bounds ``\mathbf{u_{max}}``.
- `Δumin = fill(-Inf,nu)` : manipulated input increment lower bounds ``\mathbf{Δu_{min}}``.
- `Δumax = fill(+Inf,nu)` : manipulated input increment upper bounds ``\mathbf{Δu_{max}}``.
- `ymin  = fill(-Inf,ny)` : predicted output lower bounds ``\mathbf{y_{min}}``.
- `ymax  = fill(+Inf,ny)` : predicted output upper bounds ``\mathbf{y_{max}}``.
- `x̂min  = fill(-Inf,nx̂)` : terminal constraint lower bounds ``\mathbf{x̂_{min}}``.
- `x̂max  = fill(+Inf,nx̂)` : terminal constraint upper bounds ``\mathbf{x̂_{max}}``.
- `c_umin  = fill(0.0,nu)` : `umin` softness weights ``\mathbf{c_{u_{min}}}``.
- `c_umax  = fill(0.0,nu)` : `umax` softness weights ``\mathbf{c_{u_{max}}}``.
- `c_Δumin = fill(0.0,nu)` : `Δumin` softness weights ``\mathbf{c_{Δu_{min}}}``.
- `c_Δumax = fill(0.0,nu)` : `Δumax` softness weights ``\mathbf{c_{Δu_{max}}}``.
- `c_ymin  = fill(1.0,ny)` : `ymin` softness weights ``\mathbf{c_{y_{min}}}``.
- `c_ymax  = fill(1.0,ny)` : `ymax` softness weights ``\mathbf{c_{y_{max}}}``.
- `c_x̂min  = fill(1.0,nx̂)` : `x̂min` softness weights ``\mathbf{c_{x̂_{min}}}``.
- `c_x̂max  = fill(1.0,nx̂)` : `x̂max` softness weights ``\mathbf{c_{x̂_{max}}}``.
- all the keyword arguments above but with a capital letter, except for the terminal
  constraints, e.g. `Ymax` or `c_ΔUmin` : for time-varying constraints (see Extended Help).

# Examples
```jldoctest
julia> mpc = LinMPC(setop!(LinModel(tf(3, [30, 1]), 4), uop=[50], yop=[25]));

julia> mpc = setconstraint!(mpc, umin=[0], umax=[100], c_umin=[0.0], c_umax=[0.0]);

julia> mpc = setconstraint!(mpc, Δumin=[-10], Δumax=[+10], c_Δumin=[1.0], c_Δumax=[1.0])
LinMPC controller with a sample time Ts = 4.0 s, OSQP optimizer, SteadyKalmanFilter estimator and:
 10 prediction steps Hp
  2 control steps Hc
  1 manipulated inputs u (0 integrating states)
  2 states x̂
  1 measured outputs ym (1 integrating states)
  0 unmeasured outputs yu
  0 measured disturbances d
```

# Extended Help
Terminal constraints provide closed-loop stailibility guarantees on the nominal plant
model. They can render an unfeasible problem however. In practice, a sufficiently large
prediction horizon ``H_p`` is typically enough for stability. Note that terminal constraints
are applied on the augmented state vector ``\mathbf{x̂}`` (see [`SteadyKalmanFilter`](@ref)
for details on augmentation).

For variable constraints, the bounds can be modified after calling [`moveinput!`](@ref),
that is, at runtime, but not the softness parameters ``\mathbf{c}``. It is not possible to
modify `±Inf` bounds at runtime.

!!! tip
    To keep a variable unconstrained while maintaining the ability to add a constraint later
    at runtime, set the bound to an absolute value sufficiently large when you create the
    controller (but different than `±Inf`).

It is also possible to specify time-varying constraints over ``H_p`` and ``H_c`` horizons. 
In such a case, they are defined by:
```math 
\begin{alignat*}{3}
    \mathbf{U_{min}  - c_{U_{min}}}  ϵ ≤&&\ \mathbf{U}  &≤ \mathbf{U_{max}  + c_{U_{max}}}  ϵ \\
    \mathbf{ΔU_{min} - c_{ΔU_{min}}} ϵ ≤&&\ \mathbf{ΔU} &≤ \mathbf{ΔU_{max} + c_{ΔY_{max}}} ϵ \\
    \mathbf{Y_{min}  - c_{Y_{min}}}  ϵ ≤&&\ \mathbf{Ŷ}  &≤ \mathbf{Y_{max}  + c_{Y_{max}}}  ϵ
\end{alignat*}
```
For this, use the same keyword arguments as above but with a capital letter:
- `Umin`  / `Umax`  / `c_Umin`  / `c_Umax`  : ``\mathbf{U}`` constraints `(nu*Hp,)`.
- `ΔUmin` / `ΔUmax` / `c_ΔUmin` / `c_ΔUmax` : ``\mathbf{ΔU}`` constraints `(nu*Hc,)`.
- `Ymin`  / `Ymax`  / `c_Ymin`  / `c_Ymax`  : ``\mathbf{Ŷ}`` constraints `(ny*Hp,)`.
"""
function setconstraint!(
    mpc::PredictiveController; 
    umin    = nothing, umax    = nothing,
    Δumin   = nothing, Δumax   = nothing,
    ymin    = nothing, ymax    = nothing,
    x̂min    = nothing, x̂max    = nothing,
    c_umin  = nothing, c_umax  = nothing,
    c_Δumin = nothing, c_Δumax = nothing,
    c_ymin  = nothing, c_ymax  = nothing,
    c_x̂min  = nothing, c_x̂max  = nothing,
    Umin    = nothing, Umax    = nothing,
    ΔUmin   = nothing, ΔUmax   = nothing,
    Ymin    = nothing, Ymax    = nothing,
    c_Umax  = nothing, c_Umin  = nothing,
    c_ΔUmax = nothing, c_ΔUmin = nothing,
    c_Ymax  = nothing, c_Ymin  = nothing,
    # ------------ will be deleted in the future ---------------
    ŷmin    = nothing, ŷmax    = nothing,
    c_ŷmin  = nothing, c_ŷmax  = nothing,
    # ----------------------------------------------------------
)
    # ----- these 4 `if`s will be deleted in the future --------
    if !isnothing(ŷmin)
        Base.depwarn("keyword arg ŷmin is deprecated, use ymin instead", :setconstraint!)
        ymin = ŷmin
    end
    if !isnothing(ŷmax)
        Base.depwarn("keyword arg ŷmax is deprecated, use ymax instead", :setconstraint!)
        ymax = ŷmax
    end
    if !isnothing(c_ŷmin)
        Base.depwarn("keyword arg ŷmin is deprecated, use ymin instead", :setconstraint!)
        c_ymin = c_ŷmin
    end
    if !isnothing(c_ŷmax)
        Base.depwarn("keyword arg ŷmax is deprecated, use ymax instead", :setconstraint!)
        c_ymax = c_ŷmax
    end
    # ----------------------------------------------------------
    model, con, optim = mpc.estim.model, mpc.con, mpc.optim
    nu, ny, Hp, Hc = model.nu, model.ny, mpc.Hp, mpc.Hc
    notSolvedYet = (termination_status(optim) == OPTIMIZE_NOT_CALLED)
    C, E = mpc.C, mpc.Ẽ[:, 1:nu*Hc]
    isnothing(Umin)     && !isnothing(umin)     && (Umin    = repeat(umin,    Hp))
    isnothing(Umax)     && !isnothing(umax)     && (Umax    = repeat(umax,    Hp))
    isnothing(ΔUmin)    && !isnothing(Δumin)    && (ΔUmin   = repeat(Δumin,   Hc))
    isnothing(ΔUmax)    && !isnothing(Δumax)    && (ΔUmax   = repeat(Δumax,   Hc))
    isnothing(Ymin)     && !isnothing(ymin)     && (Ymin    = repeat(ymin,    Hp))
    isnothing(Ymax)     && !isnothing(ymax)     && (Ymax    = repeat(ymax,    Hp))
    isnothing(c_Umin)   && !isnothing(c_umin)   && (c_Umin  = repeat(c_umin,  Hp))
    isnothing(c_Umax)   && !isnothing(c_umax)   && (c_Umax  = repeat(c_umax,  Hp))
    isnothing(c_ΔUmin)  && !isnothing(c_Δumin)  && (c_ΔUmin = repeat(c_Δumin, Hc))
    isnothing(c_ΔUmax)  && !isnothing(c_Δumax)  && (c_ΔUmax = repeat(c_Δumax, Hc))
    isnothing(c_Ymin)   && !isnothing(c_ymin)   && (c_Ymin  = repeat(c_ymin,  Hp))
    isnothing(c_Ymax)   && !isnothing(c_ymax)   && (c_Ymax  = repeat(c_ymax,  Hp))
    if !all(isnothing.([c_Umin, c_Umax, c_ΔUmin, c_ΔUmax, c_Ymin, c_Ymax, c_x̂min, c_x̂max]))
        !isinf(C) || throw(ArgumentError("Slack variable Cwt must be finite to set softness parameters"))
        notSolvedYet || error("Cannot set softness parameters after calling moveinput!")
    end
    if !isnothing(Umin)
        size(Umin)   == (nu*Hp,) || throw(ArgumentError("Umin size must be $((nu*Hp,))"))
        con.Umin[:] = Umin
    end
    if !isnothing(Umax)
        size(Umax)   == (nu*Hp,) || throw(ArgumentError("Umax size must be $((nu*Hp,))"))
        con.Umax[:] = Umax
    end
    if !isnothing(ΔUmin)
        size(ΔUmin)  == (nu*Hc,) || throw(ArgumentError("ΔUmin size must be $((nu*Hc,))"))
        con.ΔŨmin[1:nu*Hc] = ΔUmin
    end
    if !isnothing(ΔUmax)
        size(ΔUmax)  == (nu*Hc,) || throw(ArgumentError("ΔUmax size must be $((nu*Hc,))"))
        con.ΔŨmax[1:nu*Hc] = ΔUmax
    end
    if !isnothing(Ymin)
        size(Ymin)   == (ny*Hp,) || throw(ArgumentError("Ymin size must be $((ny*Hp,))"))
        con.Ymin[:] = Ymin
    end
    if !isnothing(Ymax)
        size(Ymax)   == (ny*Hp,) || throw(ArgumentError("Ymax size must be $((ny*Hp,))"))
        con.Ymax[:] = Ymax
    end
    if !isnothing(c_Umin)
        size(c_Umin) == (nu*Hp,) || throw(ArgumentError("c_Umin size must be $((nu*Hp,))"))
        any(c_Umin .< 0) && error("c_Umin weights should be non-negative")
        con.A_Umin[:, end] = -c_Umin
    end
    if !isnothing(c_Umax)
        size(c_Umax) == (nu*Hp,) || throw(ArgumentError("c_Umax size must be $((nu*Hp,))"))
        any(c_Umax .< 0) && error("c_Umax weights should be non-negative")
        con.A_Umax[:, end] = -c_Umax
    end
    if !isnothing(c_ΔUmin)
        size(c_ΔUmin) == (nu*Hc,) || throw(ArgumentError("c_ΔUmin size must be $((nu*Hc,))"))
        any(c_ΔUmin .< 0) && error("c_ΔUmin weights should be non-negative")
        con.A_ΔŨmin[1:end-1, end] = -c_ΔUmin 
    end
    if !isnothing(c_ΔUmax)
        size(c_ΔUmax) == (nu*Hc,) || throw(ArgumentError("c_ΔUmax size must be $((nu*Hc,))"))
        any(c_ΔUmax .< 0) && error("c_ΔUmax weights should be non-negative")
        con.A_ΔŨmax[1:end-1, end] = -c_ΔUmax
    end
    if !isnothing(c_Ymin)
        size(c_Ymin) == (ny*Hp,) || throw(ArgumentError("c_Ymin size must be $((ny*Hp,))"))
        any(c_Ymin .< 0) && error("c_Ymin weights should be non-negative")
        con.c_Ymin[:] = c_Ymin
        A_Ymin ,_ = relaxŶ(model, C, con.c_Ymin, con.c_Ymax, E)
        con.A_Ymin[:] = A_Ymin
    end
    if !isnothing(c_Ymax)
        size(c_Ymax) == (ny*Hp,) || throw(ArgumentError("c_Ymax size must be $((ny*Hp,))"))
        any(c_Ymax .< 0) && error("c_Ymax weights should be non-negative")
        con.c_Ymax[:] = c_Ymax
        _, A_Ymax = relaxŶ(model, C, con.c_Ymin, con.c_Ymax, E)
        con.A_Ymax[:] = A_Ymax
    end
    i_Umin,  i_Umax  = .!isinf.(con.Umin),  .!isinf.(con.Umax)
    i_ΔŨmin, i_ΔŨmax = .!isinf.(con.ΔŨmin), .!isinf.(con.ΔŨmin)
    i_Ymin,  i_Ymax  = .!isinf.(con.Ymin),  .!isinf.(con.Ymax)
    if notSolvedYet
        con.i_b[:], con.A[:] = init_linconstraint(model,
            i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ymin, i_Ymax,
            con.A_Umin, con.A_Umax, con.A_ΔŨmin, con.A_ΔŨmax, con.A_Ymin, con.A_Ymax
        )
        A = con.A[con.i_b, :]
        b = con.b[con.i_b]
        ΔŨvar = mpc.optim[:ΔŨvar]
        delete(mpc.optim, mpc.optim[:linconstraint])
        unregister(mpc.optim, :linconstraint)
        @constraint(mpc.optim, linconstraint, A*ΔŨvar .≤ b)
        setnonlincon!(mpc, model)
    else
        i_b, _ = init_linconstraint(model, i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ymin, i_Ymax)
        i_b == con.i_b || error("Cannot modify ±Inf constraints after calling moveinput!")
    end
    return mpc
end



"By default, there is no nonlinear constraint, thus do nothing."
setnonlincon!(::PredictiveController, ::SimModel) = nothing

@doc raw"""
    moveinput!(mpc::PredictiveController, ry=mpc.estim.model.yop, d=[]; <keyword args>) -> u

Compute the optimal manipulated input value `u` for the current control period.

Solve the optimization problem of `mpc` [`PredictiveController`](@ref) and return the
results ``\mathbf{u}(k)``. Following the receding horizon principle, the algorithm discards
the optimal future manipulated inputs ``\mathbf{u}(k+1), \mathbf{u}(k+2), ...`` Note that
the method mutates `mpc` internal data but it does not modifies `mpc.estim` states. Call
[`updatestate!(mpc, u, ym, d)`](@ref) to update `mpc` state estimates.

Calling a [`PredictiveController`](@ref) object calls this method.

See also [`LinMPC`](@ref), [`ExplicitMPC`](@ref), [`NonLinMPC`](@ref).

# Arguments
- `mpc::PredictiveController` : solve optimization problem of `mpc`.
- `ry=mpc.estim.model.yop` : current output setpoints ``\mathbf{r_y}(k)``.
- `d=[]` : current measured disturbances ``\mathbf{d}(k)``.
- `D̂=repeat(d, mpc.Hp)` : predicted measured disturbances ``\mathbf{D̂}``, constant in the
  future by default or ``\mathbf{d̂}(k+j)=\mathbf{d}(k)`` for ``j=1`` to ``H_p``.
- `R̂y=repeat(ry, mpc.Hp)` : predicted output setpoints ``\mathbf{R̂_y}``, constant in the
  future by default or ``\mathbf{r̂_y}(k+j)=\mathbf{r_y}(k)`` for ``j=1`` to ``H_p``.
- `R̂u=repeat(mpc.estim.model.uop, mpc.Hp)` : predicted manipulated input setpoints, constant
  in the future by default or ``\mathbf{r̂_u}(k+j)=\mathbf{u_{op}}`` for ``j=0`` to ``H_p-1``.
- `ym=nothing` : current measured outputs ``\mathbf{y^m}(k)``, only required if `mpc.estim` 
   is an [`InternalModel`](@ref).

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
    d ::Vector = empty(mpc.estim.x̂);
    D̂ ::Vector = repeat(d,  mpc.Hp),
    R̂y::Vector = repeat(ry, mpc.Hp),
    R̂u::Vector = repeat(mpc.estim.model.uop, mpc.Hp),
    ym::Union{Vector, Nothing} = nothing
)
    validate_setpointdist(mpc, ry, d, D̂, R̂y, R̂u)
    initpred!(mpc, mpc.estim.model, d, ym, D̂, R̂y, R̂u)
    linconstraint!(mpc, mpc.estim.model)
    ΔŨ = optim_objective!(mpc)
    Δu = ΔŨ[1:mpc.estim.model.nu] # receding horizon principle: only Δu(k) is used (1st one)
    u = mpc.estim.lastu0 + mpc.estim.model.uop + Δu
    return u
end

@doc raw"""
    getinfo(mpc::PredictiveController) -> info

Get additional information about `mpc` controller optimum to ease troubleshooting.

The function should be called after calling [`moveinput!`](@ref). It returns the dictionary
`info` with the following fields:

- `:ΔU` : optimal manipulated input increments over `Hc` ``(\mathbf{ΔU})``
- `:ϵ`  : optimal slack variable ``(ϵ)``
- `:J`  : objective value optimum ``(J)``
- `:U`  : optimal manipulated inputs over `Hp` ``(\mathbf{U})``
- `:u`  : current optimal manipulated input ``(\mathbf{u})``
- `:d`  : current measured disturbance ``(\mathbf{d})``
- `:D̂`  : predicted measured disturbances over `Hp` ``(\mathbf{D̂})``
- `:ŷ`  : current estimated output ``(\mathbf{ŷ})``
- `:Ŷ`  : optimal predicted outputs over `Hp` ``(\mathbf{Ŷ})``
- `:Ŷs` : predicted stochastic output over `Hp` of [`InternalModel`](@ref) ``(\mathbf{Ŷ_s})``
- `:R̂y` : predicted output setpoint over `Hp` ``(\mathbf{R̂_y})``
- `:R̂u` : predicted manipulated input setpoint over `Hp` ``(\mathbf{R̂_u})``

For [`LinMPC`](@ref) and [`NonLinMPC`](@ref), the field `:sol` also contains the optimizer
solution summary that can be printed. Lastly, the optimal economic cost `:JE` is also
available for [`NonLinMPC`](@ref).

# Examples
```jldoctest
julia> mpc = LinMPC(LinModel(tf(5, [2, 1]), 3), Nwt=[0], Hp=1, Hc=1);

julia> u = moveinput!(mpc, [10]);

julia> round.(getinfo(mpc)[:Ŷ], digits=3)
1-element Vector{Float64}:
 10.0
```
"""
function getinfo(mpc::PredictiveController)
    info = Dict{Symbol, InfoDictType}()
    Ŷ = similar(mpc.Ŷop)
    Ŷ = predict!(Ŷ, mpc, mpc.estim.model, mpc.ΔŨ)
    info[:ΔU]  = mpc.ΔŨ[1:mpc.Hc*mpc.estim.model.nu]
    info[:ϵ]   = isinf(mpc.C) ? NaN : mpc.ΔŨ[end]
    info[:J]   = obj_nonlinprog(mpc, mpc.estim.model, Ŷ, mpc.ΔŨ) + mpc.p[]
    info[:U]   = mpc.S̃*mpc.ΔŨ + mpc.T*(mpc.estim.lastu0 + mpc.estim.model.uop)
    info[:u]   = info[:U][1:mpc.estim.model.nu]
    info[:d]   = mpc.d0 + mpc.estim.model.dop
    info[:D̂]   = mpc.D̂0 + mpc.Dop
    info[:ŷ]   = mpc.ŷ
    info[:Ŷ]   = Ŷ
    info[:Ŷs]  = mpc.Ŷop - repeat(mpc.estim.model.yop, mpc.Hp) # Ŷop = Ŷs + Yop
    info[:R̂y]  = mpc.R̂y
    info[:R̂u]  = mpc.R̂u
    info = addinfo!(info, mpc)
    return info
end

"""
    addinfo!(info, mpc::PredictiveController) -> info

By default, add the solution summary `:sol` that can be printed to `info`.
"""
function addinfo!(info, mpc::PredictiveController)
    info[:sol] = solution_summary(mpc.optim, verbose=true)
    return info
end


"""
    setstate!(mpc::PredictiveController, x̂)

Set the estimate at `mpc.estim.x̂`.
"""
setstate!(mpc::PredictiveController, x̂) = (setstate!(mpc.estim, x̂); return mpc)

@doc raw"""
    initstate!(mpc::PredictiveController, u, ym, d=[]) -> x̂

Init the states of `mpc.estim` [`StateEstimator`](@ref) and warm start `mpc.ΔŨ` at zero.
"""
function initstate!(mpc::PredictiveController, u, ym, d=empty(mpc.estim.x̂))
    mpc.ΔŨ .= 0
    return initstate!(mpc.estim, u, ym, d)
end


"""
    updatestate!(mpc::PredictiveController, u, ym, d=[]) -> x̂

Call [`updatestate!`](@ref) on `mpc.estim` [`StateEstimator`](@ref).
"""
updatestate!(mpc::PredictiveController, u, ym, d=empty(mpc.estim.x̂)) = updatestate!(mpc.estim,u,ym,d)
updatestate!(::PredictiveController, _ ) = throw(ArgumentError("missing measured outputs ym"))

"""
    default_Hp(model::LinModel, Hp)

Estimate the default prediction horizon `Hp` with a security margin for [`LinModel`](@ref).
"""
function default_Hp(model::LinModel, Hp)
    poles = eigvals(model.A)
    # atol=1e-3 to overestimate the number of delays : for closed-loop stability, it is
    # better to overestimate the default value of Hp, as a security margin.
    nk = sum(isapprox.(abs.(poles), 0.0, atol=1e-3)) # number of delays
    if isnothing(Hp)
        Hp = DEFAULT_HP0 + nk
    end
    if Hp ≤ nk
        @warn("prediction horizon Hp ($Hp) ≤ estimated number of delays in model "*
              "($nk), the closed-loop system may be unstable or zero-gain (unresponsive)")
    end
    return Hp
end

"""
    default_Hp(model::SimModel, Hp)

Throw an error if `isnothing(Hp)` when model is not a [`LinModel`](@ref).
"""
function default_Hp(::SimModel, Hp)
    if isnothing(Hp)
        Hp = 0
        throw(ArgumentError("Prediction horizon Hp must be explicitly specified if "*
                            "model is not a LinModel."))
    end
    return Hp
end

function validate_setpointdist(mpc::PredictiveController, ry, d, D̂, R̂y, R̂u)
    ny, nd, nu, Hp = mpc.estim.model.ny, mpc.estim.model.nd, mpc.estim.model.nu, mpc.Hp
    size(ry) ≠ (ny,)    && throw(ArgumentError("ry size $(size(ry)) ≠ output size ($ny,)"))
    size(d)  ≠ (nd,)    && throw(ArgumentError("d size $(size(d)) ≠ measured dist. size ($nd,)"))
    size(D̂)  ≠ (nd*Hp,) && throw(ArgumentError("D̂ size $(size(D̂)) ≠ measured dist. size × Hp ($(nd*Hp),)"))
    size(R̂y) ≠ (ny*Hp,) && throw(ArgumentError("R̂y size $(size(R̂y)) ≠ output size × Hp ($(ny*Hp),)"))
    size(R̂u) ≠ (nu*Hp,) && throw(ArgumentError("R̂u size $(size(R̂u)) ≠ manip. input size × Hp ($(nu*Hp),)"))
end

@doc raw"""
    initpred!(mpc, model::LinModel, d, ym, D̂, R̂y, R̂u)

Init linear model prediction matrices `F`, `q̃` and `p`.

See [`init_predmat`](@ref) and [`init_quadprog`](@ref) for the definition of the matrices.
"""
function initpred!(mpc::PredictiveController, model::LinModel, d, ym, D̂, R̂y, R̂u)
    predictstoch!(mpc, mpc.estim, d, ym) # init mpc.Ŷop for InternalModel
    mpc.F[:] = mpc.K*mpc.estim.x̂ + mpc.Q*mpc.estim.lastu0 + mpc.Ŷop
    if model.nd ≠ 0
        mpc.d0[:], mpc.D̂0[:] = d - model.dop, D̂ - mpc.Dop
        mpc.F[:] = mpc.F + mpc.G*mpc.d0 + mpc.J*mpc.D̂0
    end
    mpc.R̂y[:], mpc.R̂u[:] = R̂y, R̂u
    Ẑ = mpc.F - R̂y
    mpc.q̃[:] = 2(mpc.M_Hp*mpc.Ẽ)'*Ẑ
    mpc.p[]  = Ẑ'*mpc.M_Hp*Ẑ
    if ~mpc.noR̂u
        lastu = mpc.estim.lastu0 + model.uop
        V̂ = mpc.T*lastu - mpc.R̂u
        mpc.q̃[:] = mpc.q̃ + 2(mpc.L_Hp*mpc.S̃)'*V̂
        mpc.p[]  = mpc.p[] + V̂'*mpc.L_Hp*V̂
    end
    return nothing
end

@doc raw"""
    initpred!(mpc::PredictiveController, model::SimModel, d, ym, D̂, R̂y, R̂u)

Init `Ŷop`, `d0` and `D̂0` matrices when model is not a [`LinModel`](@ref).

`d0` and `D̂0` are the measured disturbances and its predictions without the operating points
``\mathbf{d_{op}}``. The vector `Ŷop` is kept unchanged if `mpc.estim` is not an
[`InternalModel`](@ref).
"""
function initpred!(mpc::PredictiveController, model::SimModel, d, ym, D̂, R̂y, R̂u)
    predictstoch!(mpc, mpc.estim, d, ym) # init mpc.Ŷop for InternalModel
    if model.nd ≠ 0
        mpc.d0[:], mpc.D̂0[:] = d - model.dop, D̂ - mpc.Dop
    end
    mpc.R̂y[:], mpc.R̂u[:] = R̂y, R̂u
    return nothing
end

@doc raw"""
    predictstoch!(mpc::PredictiveController, estim::InternalModel, x̂s, d, ym)

Init `Ŷop` vector when if `estim` is an [`InternalModel`](@ref).

The vector combines the output operating points and the stochastic predictions:
``\mathbf{Ŷ_{op} = Ŷ_{s} + Y_{op}}`` (both values are constant between the nonlinear 
programming iterations).
"""
function predictstoch!(mpc::PredictiveController, estim::InternalModel, d, ym)
    isnothing(ym) && error("Predictive controllers with InternalModel need the measured "*
                           "outputs ym in keyword argument to compute control actions u")
    ŷd = h(estim.model, estim.x̂d, d - estim.model.dop) + estim.model.yop 
    ŷs = zeros(estim.model.ny)
    ŷs[estim.i_ym] = ym - ŷd[estim.i_ym]  # ŷs=0 for unmeasured outputs
    Ŷs = mpc.Ks*mpc.estim.x̂s + mpc.Ps*ŷs
    mpc.Ŷop[:] = Ŷs + repeat(estim.model.yop, mpc.Hp)
    return nothing
end
"Separate stochastic predictions are not needed if `estim` is not [`InternalModel`](@ref)."
predictstoch!(::PredictiveController, ::StateEstimator, _ , _ ) = nothing

@doc raw"""
    predict!(Ŷ, mpc::PredictiveController, model::LinModel, ΔŨ) -> Ŷ

Evaluate the outputs predictions ``\mathbf{Ŷ}`` when `model` is a [`LinModel`](@ref).
"""
function predict!(Ŷ, mpc::PredictiveController, ::LinModel, ΔŨ::Vector{T}) where {T<:Real}
    return mul!(Ŷ, mpc.Ẽ, ΔŨ) + mpc.F # in-place operations to reduce allocations
end

@doc raw"""
    predict!(Ŷ, mpc::PredictiveController, model::SimModel, ΔŨ) -> Ŷ

Evaluate  ``\mathbf{Ŷ}`` when `model` is not a [`LinModel`](@ref).
"""
function predict!(Ŷ, mpc::PredictiveController, model::SimModel, ΔŨ::Vector{T}) where {T<:Real}
    nu, ny, nd, Hp, Hc = model.nu, model.ny, model.nd, mpc.Hp, mpc.Hc
    x̂ ::Vector{T} = copy(mpc.estim.x̂)
    u0::Vector{T} = copy(mpc.estim.lastu0)
    d0::Vector{T} = copy(mpc.d0)
    for j=1:Hp
        if j ≤ Hc
            u0[:] = u0 + ΔŨ[(1 + nu*(j-1)):(nu*j)]
        end
        x̂[:]  = f̂(mpc.estim, x̂, u0, d0)
        d0[:] = mpc.D̂0[(1 + nd*(j-1)):(nd*j)]
        Ŷ[(1 + ny*(j-1)):(ny*j)] = ĥ(mpc.estim, x̂, d0)
    end
    Ŷ[:] = Ŷ + mpc.Ŷop # Ŷop = Ŷs + Yop, and Ŷs=0 if mpc.estim is not an InternalModel
    return Ŷ
end

@doc raw"""
    linconstraint!(mpc::PredictiveController, model::LinModel)

Set `b` vector for the linear model inequality constraints (``\mathbf{A ΔŨ ≤ b}``).
"""
function linconstraint!(mpc::PredictiveController, model::LinModel)
    mpc.con.b[:] = [
        -mpc.con.Umin + mpc.T*(mpc.estim.lastu0 + model.uop)
        +mpc.con.Umax - mpc.T*(mpc.estim.lastu0 + model.uop)
        -mpc.con.ΔŨmin
        +mpc.con.ΔŨmax 
        -mpc.con.Ymin + mpc.F
        +mpc.con.Ymax - mpc.F
    ]
    lincon::LinConVector = mpc.optim[:linconstraint]
    set_normalized_rhs.(lincon, mpc.con.b[mpc.con.i_b])
end

"Set `b` excluding predicted output constraints when `model` is not a [`LinModel`](@ref)."
function linconstraint!(mpc::PredictiveController, model::SimModel)
    mpc.con.b[:] = [
        -mpc.con.Umin + mpc.T*(mpc.estim.lastu0 + model.uop)
        +mpc.con.Umax - mpc.T*(mpc.estim.lastu0 + model.uop)
        -mpc.con.ΔŨmin
        +mpc.con.ΔŨmax 
    ]
    lincon::LinConVector = mpc.optim[:linconstraint]
    set_normalized_rhs.(lincon, mpc.con.b[mpc.con.i_b])
end

"""
    optim_objective!(mpc::PredictiveController)

Optimize the objective function ``J`` of `mpc` controller and return the solution `ΔŨ`.
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
    mpc.ΔŨ[:] = isfatal(status) ? value.(ΔŨvar) : value.(ΔŨvar) # fatal status : use last value
    return mpc.ΔŨ
end

"By default, no need to modify the objective function."
set_objective_linear_coef!(::PredictiveController, _ ) = nothing

@doc raw"""
    init_ΔUtoU(nu, Hp) -> S, T

Init manipulated input increments to inputs conversion matrices.

The conversion from the input increments ``\mathbf{ΔU}`` to manipulated inputs over ``H_p`` 
are calculated by:
```math
\mathbf{U} = \mathbf{S} \mathbf{ΔU} + \mathbf{T} \mathbf{u}(k-1) \\
```
"""
function init_ΔUtoU(nu, Hp, Hc)
    S_Hc = LowerTriangular(repeat(I(nu), Hc, Hc))
    S = [S_Hc; repeat(I(nu), Hp - Hc, Hc)]
    T = repeat(I(nu), Hp)
    return S, T
end


@doc raw"""
    init_predmat(estim::StateEstimator, ::LinModel, Hp, Hc) -> E, G, J, K, Q

Construct the prediction matrices for [`LinModel`](@ref) `model`.

The linear model predictions are evaluated by :
```math
\begin{aligned}
    \mathbf{Ŷ} &= \mathbf{E ΔU} + \mathbf{G d}(k) + \mathbf{J D̂} + \mathbf{K x̂}_{k-1}(k) 
                                                  + \mathbf{Q u}(k-1) \\
               &= \mathbf{E ΔU} + \mathbf{F}
\end{aligned}
```
where predicted outputs ``\mathbf{Ŷ}``, stochastic outputs ``\mathbf{Ŷ_s}``, and measured
disturbances ``\mathbf{D̂}`` are from ``k + 1`` to ``k + H_p``. Input increments 
``\mathbf{ΔU}`` are from ``k`` to ``k + H_c - 1``. The vector ``\mathbf{x̂}_{k-1}(k)`` is the
state estimated at the last control period. Operating points on ``\mathbf{u}``, ``\mathbf{d}``
and ``\mathbf{y}`` are omitted in above equations.

# Extended Help
Using the augmented matrices ``\mathbf{Â, B̂_u, Ĉ, B̂_d, D̂_d}`` in `estim` and the equation
``\mathbf{W}_j = \mathbf{Ĉ} ( ∑_{i=0}^j \mathbf{Â}^i ) \mathbf{B̂_u}``, the prediction 
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
\mathbf{Ĉ}\mathbf{Â}^{0} \mathbf{B̂_d}     \\ 
\mathbf{Ĉ}\mathbf{Â}^{1} \mathbf{B̂_d}     \\ 
\vdots                                    \\
\mathbf{Ĉ}\mathbf{Â}^{H_p-1} \mathbf{B̂_d}
\end{bmatrix}
\\
\mathbf{J} &= \begin{bmatrix}
\mathbf{D̂_d}                              & \mathbf{0}                                & \cdots & \mathbf{0}   \\ 
\mathbf{Ĉ}\mathbf{Â}^{0} \mathbf{B̂_d}     & \mathbf{D̂_d}                              & \cdots & \mathbf{0}   \\ 
\vdots                                    & \vdots                                    & \ddots & \vdots       \\
\mathbf{Ĉ}\mathbf{Â}^{H_p-2} \mathbf{B̂_d} & \mathbf{Ĉ}\mathbf{Â}^{H_p-3} \mathbf{B̂_d} & \cdots & \mathbf{D̂_d}
\end{bmatrix}
\\
\mathbf{K} &= \begin{bmatrix}
\mathbf{Ĉ}\mathbf{Â}^{1}      \\
\mathbf{Ĉ}\mathbf{Â}^{2}      \\
\vdots                        \\
\mathbf{Ĉ}\mathbf{Â}^{H_p}
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
"""
function init_predmat(estim::StateEstimator, model::LinModel, Hp, Hc)
    Â, B̂u, Ĉ, B̂d, D̂d = estim.Â, estim.B̂u, estim.Ĉ, estim.B̂d, estim.D̂d
    nu, nx̂, ny, nd = model.nu, estim.nx̂, model.ny, model.nd
    # Apow 3D array : Apow[:,:,1] = A^0, Apow[:,:,2] = A^1, ...
    Âpow = Array{Float64}(undef, size(Â,1), size(Â,2), Hp+1)
    Âpow[:,:,1] = I(nx̂)
    K = Matrix{Float64}(undef, Hp*ny, nx̂)
    for i=1:Hp
        Âpow[:,:,i+1] = Â^i
        iRow = (1:ny) .+ ny*(i-1)
        K[iRow,:] = Ĉ*Âpow[:,:,i+1]
    end 
    # Apow_csum 3D array : Apow_csum[:,:,1] = A^0, Apow_csum[:,:,2] = A^1 + A^0, ...
    Apow_csum  = cumsum(Âpow, dims=3)
    # --- manipulated inputs u ---
    Q = Matrix{Float64}(undef, Hp*ny, nu)
    for i=1:Hp
        iRow = (1:ny) .+ ny*(i-1)
        Q[iRow,:] = Ĉ*Apow_csum[:,:,i]*B̂u
    end
    E = zeros(Hp*ny, Hc*nu) 
    for i=1:Hc # truncated with control horizon
        iRow = (ny*(i-1)+1):(ny*Hp)
        iCol = (1:nu) .+ nu*(i-1)
        E[iRow,iCol] = Q[iRow .- ny*(i-1),:]
    end
    # --- measured disturbances d ---
    G = Matrix{Float64}(undef, Hp*ny, nd)
    J = repeatdiag(D̂d, Hp)
    if nd ≠ 0
        for i=1:Hp
            iRow = (1:ny) .+ ny*(i-1)
            G[iRow,:] = Ĉ*Âpow[:,:,i]*B̂d
        end
        for i=1:Hp
            iRow = (ny*i+1):(ny*Hp)
            iCol = (1:nd) .+ nd*(i-1)
            J[iRow,iCol] = G[iRow .- ny*i,:]
        end
    end
    F = zeros(ny*Hp) # dummy value (updated just before optimization)
    return E, F, G, J, K, Q
end

"Return empty matrices if `model` is not a [`LinModel`](@ref)"
function init_predmat(estim::StateEstimator, model::SimModel, Hp, Hc)
    nu, nx̂, nd = model.nu, estim.nx̂, model.nd
    E = zeros(0, nu*Hc)
    G = zeros(0, nd)
    J = zeros(0, nd*Hp)
    K = zeros(0, nx̂)
    Q = zeros(0, nu)
    F = zeros(0)
    return E, F, G, J, K, Q
end

@doc raw"""
    init_quadprog(model::LinModel, Ẽ, S, M_Hp, N_Hc, L_Hp) -> P̃, q̃, p

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
function init_quadprog(::LinModel, Ẽ, S, M_Hp, N_Hc, L_Hp)
    P̃ = 2*Hermitian(Ẽ'*M_Hp*Ẽ + N_Hc + S'*L_Hp*S)
    q̃ = zeros(size(P̃, 1))   # dummy value (updated just before optimization)
    p = zeros(1)            # dummy value (updated just before optimization)
    return P̃, q̃, p
end
"Return empty matrices if `model` is not a [`LinModel`](@ref)."
function init_quadprog(::SimModel, Ẽ, S, M_Hp, N_Hc, L_Hp)
    P̃ = Hermitian(zeros(0, 0))
    q̃ = zeros(0)
    p = zeros(1)            # dummy value (updated just before optimization)
    return P̃, q̃, p
end

"Return the quadratic programming objective function, see [`init_quadprog`](@ref)."
obj_quadprog(ΔŨ, P̃, q̃) = 0.5*ΔŨ'*P̃*ΔŨ + q̃'*ΔŨ

"""
    obj_nonlinprog(mpc::PredictiveController, model::LinModel, ΔŨ::Vector{Real})

Nonlinear programming objective function when `model` is a [`LinModel`](@ref).

The function is called by the nonlinear optimizer of [`NonLinMPC`](@ref) controllers. It can
also be called on any [`PredictiveController`](@ref)s to evaluate the objective function `J`
at specific input increments `ΔŨ` and predictions `Ŷ` values.
"""
function obj_nonlinprog(
    mpc::PredictiveController, model::LinModel, Ŷ, ΔŨ::Vector{T}
) where {T<:Real}
    J = obj_quadprog(ΔŨ, mpc.P̃, mpc.q̃)
    if !iszero(mpc.E)
        U = mpc.S̃*ΔŨ + mpc.T*(mpc.estim.lastu0 + model.uop)
        UE = [U; U[(end - model.nu + 1):end]]
        ŶE = [mpc.ŷ; Ŷ]
        D̂E = [mpc.d0 + model.dop; mpc.D̂0 + mpc.Dop]
        J += mpc.E*mpc.JE(UE, ŶE, D̂E)
    end
    return J
end

"""
    obj_nonlinprog(mpc::PredictiveController, model::SimModel, ΔŨ::Vector{Real})

Nonlinear programming objective function when `model` is not a [`LinModel`](@ref).
"""
function obj_nonlinprog(
    mpc::PredictiveController, model::SimModel, Ŷ, ΔŨ::Vector{T}
) where {T<:Real}
    # --- output setpoint tracking term ---
    êy = mpc.R̂y - Ŷ
    JR̂y = êy'*mpc.M_Hp*êy  
    # --- move suppression and slack variable term ---
    JΔŨ = ΔŨ'*mpc.Ñ_Hc*ΔŨ
    # --- input over prediction horizon ---
    if !mpc.noR̂u || !iszero(mpc.E)
        U = mpc.S̃*ΔŨ + mpc.T*(mpc.estim.lastu0 + model.uop)
    end
    # --- input setpoint tracking term ---
    if !mpc.noR̂u
        êu = mpc.R̂u - U
        JR̂u = êu'*mpc.L_Hp*êu
    else
        JR̂u = 0.0
    end
    # --- economic term ---
    if !iszero(mpc.E)
        UE = [U; U[(end - model.nu + 1):end]]
        ŶE = [mpc.ŷ; Ŷ]
        D̂E = [mpc.d0 + model.dop; mpc.D̂0 + mpc.Dop]
        E_JE = mpc.E*mpc.JE(UE, ŶE, D̂E)
    else
        E_JE = 0.0
    end
    return JR̂y + JΔŨ + JR̂u + E_JE
end

"""
    init_defaultcon(model, C, S, N_Hc, E) -> con, S̃, Ñ_Hc, Ẽ

Init `ControllerConstraint` struct with default parameters.

Also return `S̃`, `Ñ_Hc` and `Ẽ` matrices for the the augmented decision vector `ΔŨ`.
"""
function init_defaultcon(model, Hp, Hc, C, S, N_Hc, E)
    nu, ny = model.nu, model.ny
    umin,       umax    = fill(-Inf, nu), fill(+Inf, nu)
    Δumin,      Δumax   = fill(-Inf, nu), fill(+Inf, nu)
    ymin,       ymax    = fill(-Inf, ny), fill(+Inf, ny)
    c_umin,     c_umax  = fill(0.0, nu),  fill(0.0, nu)
    c_Δumin,    c_Δumax = fill(0.0, nu),  fill(0.0, nu)
    c_ymin,     c_ymax  = fill(1.0, ny),  fill(1.0, ny)
    Umin, Umax, ΔUmin, ΔUmax, Ymin, Ymax = 
        repeat_constraints(Hp, Hc, umin, umax, Δumin, Δumax, ymin, ymax)
    c_Umin, c_Umax, c_ΔUmin, c_ΔUmax, c_Ymin, c_Ymax = 
        repeat_constraints(Hp, Hc, c_umin, c_umax, c_Δumin, c_Δumax, c_ymin, c_ymax)
    A_Umin, A_Umax, S̃ = relaxU(C, c_Umin, c_Umax, S)
    A_ΔŨmin, A_ΔŨmax, ΔŨmin, ΔŨmax, Ñ_Hc = relaxΔU(C, c_ΔUmin, c_ΔUmax, ΔUmin, ΔUmax, N_Hc)
    A_Ymin, A_Ymax, Ẽ = relaxŶ(model, C, c_Ymin, c_Ymax, E)
    i_Umin,  i_Umax  = .!isinf.(Umin),  .!isinf.(Umax)
    i_ΔŨmin, i_ΔŨmax = .!isinf.(ΔŨmin), .!isinf.(ΔŨmax)
    i_Ymin,  i_Ymax  = .!isinf.(Ymin),  .!isinf.(Ymax)
    i_b, A = init_linconstraint(
        model, 
        i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ymin, i_Ymax,
        A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, A_Ymin, A_Ymax
    )
    b = zeros(size(A, 1)) # dummy b vector (updated just before optimization)
    con = ControllerConstraint(
        Umin    , Umax  , ΔŨmin  , ΔŨmax    , Ymin  , Ymax,
        A_Umin  , A_Umax, A_ΔŨmin, A_ΔŨmax  , A_Ymin, A_Ymax,
        A       , b     , i_b    , c_Ymin   , c_Ymax 
    )
    return con, S̃, Ñ_Hc, Ẽ
end

"Repeat predictive controller constraints over prediction `Hp` and control `Hc` horizons."
function repeat_constraints(Hp, Hc, umin, umax, Δumin, Δumax, ymin, ymax)
    Umin  = repeat(umin, Hp)
    Umax  = repeat(umax, Hp)
    ΔUmin = repeat(Δumin, Hc)
    ΔUmax = repeat(Δumax, Hc)
    Ymin  = repeat(ymin, Hp)
    Ymax  = repeat(ymax, Hp)
    return Umin, Umax, ΔUmin, ΔUmax, Ymin, Ymax
end

@doc raw"""
    relaxU(C, c_Umin, c_Umax, S) -> A_Umin, A_Umax, S̃

Augment manipulated inputs constraints with slack variable ϵ for softening.

Denoting the input increments augmented with the slack variable
``\mathbf{ΔŨ} = [\begin{smallmatrix} \mathbf{ΔU} \\ ϵ \end{smallmatrix}]``, it returns the
augmented conversion matrix ``\mathbf{S̃}``, similar to the one described at
[`init_ΔUtoU`](@ref). It also returns the ``\mathbf{A}`` matrices for the inequality
constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{U_{min}}} \\ 
    \mathbf{A_{U_{max}}} 
\end{bmatrix} \mathbf{ΔŨ} ≤
\begin{bmatrix}
    - \mathbf{U_{min}} + \mathbf{T} \mathbf{u}(k-1) \\
    + \mathbf{U_{max}} - \mathbf{T} \mathbf{u}(k-1)
\end{bmatrix}
```
"""
function relaxU(C, c_Umin, c_Umax, S)
    if !isinf(C) # ΔŨ = [ΔU; ϵ]
        # ϵ impacts ΔU → U conversion for constraint calculations:
        A_Umin, A_Umax = -[S  c_Umin],  [S -c_Umax] 
        # ϵ has no impact on ΔU → U conversion for prediction calculations:
        S̃ = [S falses(size(S, 1))]
    else # ΔŨ = ΔU (only hard constraints)
        A_Umin, A_Umax = -S,  S
        S̃ = S
    end
    return A_Umin, A_Umax, S̃
end

@doc raw"""
    relaxΔU(C, c_ΔUmin, c_ΔUmax, ΔUmin, ΔUmax, N_Hc) -> A_ΔŨmin, A_ΔŨmax, ΔŨmin, ΔŨmax, Ñ_Hc

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
    relaxŶ(::LinModel, C, c_Ymin, c_Ymax, E) -> A_Ymin, A_Ymax, Ẽ

Augment linear output prediction constraints with slack variable ϵ for softening.

Denoting the input increments augmented with the slack variable 
``\mathbf{ΔŨ} = [\begin{smallmatrix} \mathbf{ΔU} \\ ϵ \end{smallmatrix}]``, it returns the 
``\mathbf{Ẽ}`` matrix that appears in the linear model prediction equation 
``\mathbf{Ŷ = Ẽ ΔŨ + F}``, and the ``\mathbf{A}`` matrices for the inequality constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{Y_{min}}} \\ 
    \mathbf{A_{Y_{max}}}
\end{bmatrix} \mathbf{ΔŨ} ≤
\begin{bmatrix}
    - \mathbf{Y_{min}} + \mathbf{F} \\
    + \mathbf{Y_{max}} - \mathbf{F} 
\end{bmatrix}
```
"""
function relaxŶ(::LinModel, C, c_Ymin, c_Ymax, E)
    if !isinf(C) # ΔŨ = [ΔU; ϵ]
        # ϵ impacts predicted output constraint calculations:
        A_Ymin, A_Ymax = -[E  c_Ymin], [E -c_Ymax] 
        # ϵ has not impact on output predictions
        Ẽ = [E zeros(size(E, 1), 1)] 
    else # ΔŨ = ΔU (only hard constraints)
        Ẽ = E
        A_Ymin, A_Ymax = -E,  E
    end
    return A_Ymin, A_Ymax, Ẽ
end

"Return empty matrices if model is not a [`LinModel`](@ref)"
function relaxŶ(::SimModel, C, c_Ymin, c_Ymax, E)
    Ẽ = !isinf(C) ? [E zeros(0, 1)] : E
    A_Ymin, A_Ymax = Ẽ, Ẽ 
    return A_Ymin, A_Ymax, Ẽ
end

@doc raw"""
    init_stochpred(estim::InternalModel, Hp) -> Ks, Ps

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
"Return empty matrices if `estim` is not a [`InternalModel`](@ref)."
init_stochpred(estim::StateEstimator, _ ) = zeros(0, estim.nxs), zeros(0, estim.model.ny)


@doc raw"""
    init_linconstraint(::LinModel,
        i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ymin, i_Ymax, args...
    ) -> i_b, A

Init `i_b` and `A` for the linear inequality constraints (``\mathbf{A ΔŨ ≤ b}``).

If provided, the arguments in `args` should be all the inequality constraint matrices:
`A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, A_Ymin, A_Ymax`. If not provided, it returns an empty `A`
matrix. `i_b` is a `BitVector` including the indices of ``\mathbf{b}`` that are finite
numbers.
"""
function init_linconstraint(::LinModel, 
    i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ymin, i_Ymax, args...
)
    i_b = [i_Umin; i_Umax; i_ΔŨmin; i_ΔŨmax; i_Ymin; i_Ymax]
    if isempty(args)
        A = zeros(length(i_b), 0)
    else
        A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, A_Ymin, A_Ymax = args
        A = [A_Umin; A_Umax; A_ΔŨmin; A_ΔŨmax; A_Ymin; A_Ymax]
    end
    return i_b, A
end

"Init values without predicted output constraints if `model` is not a [`LinModel`](@ref)."
function init_linconstraint(::SimModel,
    i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, _ , _ , args...
)
    i_b = [i_Umin; i_Umax; i_ΔŨmin; i_ΔŨmax]
    if isempty(args)
        A = zeros(length(i_b), 0)
    else
        A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, _ , _ = args
        A = [A_Umin; A_Umax; A_ΔŨmin; A_ΔŨmax]
    end
    return i_b, A
end

"Validate predictive controller weight and horizon specified values."
function validate_weights(model, Hp, Hc, Mwt, Nwt, Lwt, Cwt, Ewt=nothing)
    nu, ny = model.nu, model.ny
    Hp < 1  && throw(ArgumentError("Prediction horizon Hp should be ≥ 1"))
    Hc < 1  && throw(ArgumentError("Control horizon Hc should be ≥ 1"))
    Hc > Hp && throw(ArgumentError("Control horizon Hc should be ≤ prediction horizon Hp"))
    size(Mwt) ≠ (ny,) && throw(ArgumentError("Mwt size $(size(Mwt)) ≠ output size ($ny,)"))
    size(Nwt) ≠ (nu,) && throw(ArgumentError("Nwt size $(size(Nwt)) ≠ manipulated input size ($nu,)"))
    size(Lwt) ≠ (nu,) && throw(ArgumentError("Lwt size $(size(Lwt)) ≠ manipulated input size ($nu,)"))
    size(Cwt) ≠ ()    && throw(ArgumentError("Cwt should be a real scalar"))
    any(Mwt.<0) && throw(ArgumentError("Mwt weights should be ≥ 0"))
    any(Nwt.<0) && throw(ArgumentError("Nwt weights should be ≥ 0"))
    any(Lwt.<0) && throw(ArgumentError("Lwt weights should be ≥ 0"))
    Cwt < 0     && throw(ArgumentError("Cwt weight should be ≥ 0"))
    !isnothing(Ewt) && size(Ewt) ≠ () && throw(ArgumentError("Ewt should be a real scalar"))
end

function Base.show(io::IO, mpc::PredictiveController)
    Hp, Hc = mpc.Hp, mpc.Hc
    nu, nd = mpc.estim.model.nu, mpc.estim.model.nd
    nx̂, nym, nyu = mpc.estim.nx̂, mpc.estim.nym, mpc.estim.nyu
    n = maximum(ndigits.((Hp, Hc, nu, nx̂, nym, nyu, nd))) + 1
    println(io, "$(typeof(mpc).name.name) controller with a sample time Ts = "*
                "$(mpc.estim.model.Ts) s, $(solver_name(mpc.optim)) optimizer, "*
                "$(typeof(mpc.estim).name.name) estimator and:")
    println(io, "$(lpad(Hp, n)) prediction steps Hp")
    println(io, "$(lpad(Hc, n)) control steps Hc")
    print_estim_dim(io, mpc.estim, n)
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
    d ::Vector = empty(mpc.estim.x̂);
    kwargs...
)
    return moveinput!(mpc, ry, d; kwargs...)
end

include("controller/explicitmpc.jl")
include("controller/linmpc.jl")
include("controller/nonlinmpc.jl")
