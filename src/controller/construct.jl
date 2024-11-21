"Include all the data for the constraints of [`PredictiveController`](@ref)"
struct ControllerConstraint{NT<:Real, GCfunc<:Function}
    ẽx̂      ::Matrix{NT}
    fx̂      ::Vector{NT}
    gx̂      ::Matrix{NT}
    jx̂      ::Matrix{NT}
    kx̂      ::Matrix{NT}
    vx̂      ::Matrix{NT}
    bx̂      ::Vector{NT}
    U0min   ::Vector{NT}
    U0max   ::Vector{NT}
    ΔŨmin   ::Vector{NT}
    ΔŨmax   ::Vector{NT}
    Y0min   ::Vector{NT}
    Y0max   ::Vector{NT}
    x̂0min   ::Vector{NT}
    x̂0max   ::Vector{NT}
    A_Umin  ::Matrix{NT}
    A_Umax  ::Matrix{NT}
    A_ΔŨmin ::Matrix{NT}
    A_ΔŨmax ::Matrix{NT}
    A_Ymin  ::Matrix{NT}
    A_Ymax  ::Matrix{NT}
    A_x̂min  ::Matrix{NT}
    A_x̂max  ::Matrix{NT}
    A       ::Matrix{NT}
    b       ::Vector{NT}
    i_b     ::BitVector
    C_ymin  ::Vector{NT}
    C_ymax  ::Vector{NT}
    c_x̂min  ::Vector{NT}
    c_x̂max  ::Vector{NT}
    i_g     ::BitVector
    gc!     ::GCfunc
    nc      ::Int
end

@doc raw"""
    setconstraint!(mpc::PredictiveController; <keyword arguments>) -> mpc

Set the bound constraint parameters of the [`PredictiveController`](@ref) `mpc`.

The predictive controllers support both soft and hard constraints, defined by:
```math 
\begin{alignat*}{3}
    \mathbf{u_{min}  - c_{u_{min}}}  ϵ ≤&&\       \mathbf{u}(k+j) &≤ \mathbf{u_{max}  + c_{u_{max}}}  ϵ &&\qquad  j = 0, 1 ,..., H_p - 1 \\
    \mathbf{Δu_{min} - c_{Δu_{min}}} ϵ ≤&&\      \mathbf{Δu}(k+j) &≤ \mathbf{Δu_{max} + c_{Δu_{max}}} ϵ &&\qquad  j = 0, 1 ,..., H_c - 1 \\
    \mathbf{y_{min}  - c_{y_{min}}}  ϵ ≤&&\       \mathbf{ŷ}(k+j) &≤ \mathbf{y_{max}  + c_{y_{max}}}  ϵ &&\qquad  j = 1, 2 ,..., H_p     \\
    \mathbf{x̂_{min}  - c_{x̂_{min}}}  ϵ ≤&&\     \mathbf{x̂}_i(k+j) &≤ \mathbf{x̂_{max}  + c_{x̂_{max}}}  ϵ &&\qquad  j = H_p
\end{alignat*}
```
and also ``ϵ ≥ 0``. The last line is the terminal constraints applied on the states at the
end of the horizon (see Extended Help). See [`MovingHorizonEstimator`](@ref) constraints
for details on bounds and softness parameters ``\mathbf{c}``. The output and terminal 
constraints are all soft by default. See Extended Help for time-varying constraints.

# Arguments
!!! info
    The keyword arguments `Δumin`, `Δumax`, `c_Δumin`, `c_Δumax`, `x̂min`, `x̂max`, `c_x̂min`,
    `c_x̂max` and their capital letter versions have non-Unicode alternatives e.g. 
    *`Deltaumin`*, *`xhatmax`* and *`C_Deltaumin`*

    The default constraints are mentioned here for clarity but omitting a keyword argument 
    will not re-assign to its default value (defaults are set at construction only).

- `mpc::PredictiveController` : predictive controller to set constraints
- `umin=fill(-Inf,nu)` / `umax=fill(+Inf,nu)` : manipulated input bound ``\mathbf{u_{min/max}}``
- `Δumin=fill(-Inf,nu)` / `Δumax=fill(+Inf,nu)` : manipulated input increment bound ``\mathbf{Δu_{min/max}}``
- `ymin=fill(-Inf,ny)` / `ymax=fill(+Inf,ny)` : predicted output bound ``\mathbf{y_{min/max}}``
- `x̂min=fill(-Inf,nx̂)` / `x̂max=fill(+Inf,nx̂)` : terminal constraint bound ``\mathbf{x̂_{min/max}}``
- `c_umin=fill(0.0,nu)` / `c_umax=fill(0.0,nu)` : `umin` / `umax` softness weight ``\mathbf{c_{u_{min/max}}}``
- `c_Δumin=fill(0.0,nu)` / `c_Δumax=fill(0.0,nu)` : `Δumin` / `Δumax` softness weight ``\mathbf{c_{Δu_{min/max}}}``
- `c_ymin=fill(1.0,ny)` / `c_ymax=fill(1.0,ny)` : `ymin` / `ymax` softness weight ``\mathbf{c_{y_{min/max}}}``
- `c_x̂min=fill(1.0,nx̂)` / `c_x̂max=fill(1.0,nx̂)` : `x̂min` / `x̂max` softness weight ``\mathbf{c_{x̂_{min/max}}}``
- all the keyword arguments above but with a first capital letter, except for the terminal
  constraints, e.g. `Ymax` or `C_Δumin`: for time-varying constraints (see Extended Help)

# Examples
```jldoctest
julia> mpc = LinMPC(setop!(LinModel(tf(3, [30, 1]), 4), uop=[50], yop=[25]));

julia> mpc = setconstraint!(mpc, umin=[0], umax=[100], Δumin=[-10], Δumax=[+10])
LinMPC controller with a sample time Ts = 4.0 s, OSQP optimizer, SteadyKalmanFilter estimator and:
 10 prediction steps Hp
  2 control steps Hc
  1 slack variable ϵ (control constraints)
  1 manipulated inputs u (0 integrating states)
  2 estimated states x̂
  1 measured outputs ym (1 integrating states)
  0 unmeasured outputs yu
  0 measured disturbances d
```

# Extended Help
!!! details "Extended Help"
    Terminal constraints provide closed-loop stability guarantees on the nominal plant model.
    They can render an unfeasible problem however. In practice, a sufficiently large
    prediction horizon ``H_p`` without terminal constraints is typically enough for 
    stability. If `mpc.estim.direct==true`, the estimator computes the states at ``i = k`` 
    (the current time step), otherwise at ``i = k - 1``. Note that terminal constraints are
    applied on the augmented state vector ``\mathbf{x̂}`` (see [`SteadyKalmanFilter`](@ref)
    for details on augmentation).

    For variable constraints, the bounds can be modified after calling [`moveinput!`](@ref),
    that is, at runtime, but not the softness parameters ``\mathbf{c}``. It is not possible
    to modify `±Inf` bounds at runtime.

    !!! tip
        To keep a variable unconstrained while maintaining the ability to add a constraint
        later at runtime, set the bound to an absolute value sufficiently large when you
        create the controller (but different than `±Inf`).

    It is also possible to specify time-varying constraints over ``H_p`` and ``H_c`` 
    horizons. In such a case, they are defined by:
    ```math 
    \begin{alignat*}{3}
        \mathbf{U_{min}  - C_{u_{min}}}  ϵ ≤&&\ \mathbf{U}  &≤ \mathbf{U_{max}  + C_{u_{max}}}  ϵ \\
        \mathbf{ΔU_{min} - C_{Δu_{min}}} ϵ ≤&&\ \mathbf{ΔU} &≤ \mathbf{ΔU_{max} + C_{Δu_{max}}} ϵ \\
        \mathbf{Y_{min}  - C_{y_{min}}}  ϵ ≤&&\ \mathbf{Ŷ}  &≤ \mathbf{Y_{max}  + C_{y_{max}}}  ϵ
    \end{alignat*}
    ```
    For this, use the same keyword arguments as above but with a first capital letter:
    - `Umin`  / `Umax`  / `C_umin`  / `C_umax`  : ``\mathbf{U}`` constraints `(nu*Hp,)`.
    - `ΔUmin` / `ΔUmax` / `C_Δumin` / `C_Δumax` : ``\mathbf{ΔU}`` constraints `(nu*Hc,)`.
    - `Ymin`  / `Ymax`  / `C_ymin`  / `C_ymax`  : ``\mathbf{Ŷ}`` constraints `(ny*Hp,)`.
"""
function setconstraint!(
    mpc::PredictiveController; 
    umin        = nothing, umax        = nothing,
    Deltaumin   = nothing, Deltaumax   = nothing,
    ymin        = nothing, ymax        = nothing,
    xhatmin     = nothing, xhatmax     = nothing,
    c_umin      = nothing, c_umax      = nothing,
    c_Deltaumin = nothing, c_Deltaumax = nothing,
    c_ymin      = nothing, c_ymax      = nothing,
    c_xhatmin   = nothing, c_xhatmax   = nothing,
    Umin        = nothing, Umax        = nothing,
    DeltaUmin   = nothing, DeltaUmax   = nothing,
    Ymin        = nothing, Ymax        = nothing,
    C_umax      = nothing, C_umin      = nothing,
    C_Deltaumax = nothing, C_Deltaumin = nothing,
    C_ymax      = nothing, C_ymin      = nothing,
    Δumin   = Deltaumin,   Δumax = Deltaumax,
    x̂min    = xhatmin,     x̂max = xhatmax,
    c_Δumin = c_Deltaumin, c_Δumax = c_Deltaumax,
    c_x̂min  = c_xhatmin,   c_x̂max = c_xhatmax,
    ΔUmin   = DeltaUmin,   ΔUmax = DeltaUmax,
    C_Δumin = C_Deltaumin, C_Δumax = C_Deltaumax,
)
    model, con, optim = mpc.estim.model, mpc.con, mpc.optim
    nu, ny, nx̂, Hp, Hc, nϵ = model.nu, model.ny, mpc.estim.nx̂, mpc.Hp, mpc.Hc, mpc.nϵ
    notSolvedYet = (JuMP.termination_status(optim) == JuMP.OPTIMIZE_NOT_CALLED)
    if isnothing(Umin) && !isnothing(umin)
        size(umin) == (nu,) || throw(ArgumentError("umin size must be $((nu,))"))
        for i = 1:nu*Hp
            con.U0min[i] = umin[(i-1) % nu + 1] - mpc.Uop[i]
        end
    elseif !isnothing(Umin)
        size(Umin) == (nu*Hp,) || throw(ArgumentError("Umin size must be $((nu*Hp,))"))
        con.U0min .= Umin .- mpc.Uop
    end
    if isnothing(Umax) && !isnothing(umax)
        size(umax) == (nu,) || throw(ArgumentError("umax size must be $((nu,))"))
        for i = 1:nu*Hp
            con.U0max[i] = umax[(i-1) % nu + 1] - mpc.Uop[i]
        end
    elseif !isnothing(Umax)
        size(Umax)   == (nu*Hp,) || throw(ArgumentError("Umax size must be $((nu*Hp,))"))
        con.U0max .= Umax .- mpc.Uop
    end
    if isnothing(ΔUmin) && !isnothing(Δumin)
        size(Δumin) == (nu,) || throw(ArgumentError("Δumin size must be $((nu,))"))
        for i = 1:nu*Hc
            con.ΔŨmin[i] = Δumin[(i-1) % nu + 1]
        end
    elseif !isnothing(ΔUmin)
        size(ΔUmin)  == (nu*Hc,) || throw(ArgumentError("ΔUmin size must be $((nu*Hc,))"))
        con.ΔŨmin[1:nu*Hc] .= ΔUmin
    end
    if isnothing(ΔUmax) && !isnothing(Δumax)
        size(Δumax) == (nu,) || throw(ArgumentError("Δumax size must be $((nu,))"))
        for i = 1:nu*Hc
            con.ΔŨmax[i] = Δumax[(i-1) % nu + 1]
        end
    elseif !isnothing(ΔUmax)
        size(ΔUmax)  == (nu*Hc,) || throw(ArgumentError("ΔUmax size must be $((nu*Hc,))"))
        con.ΔŨmax[1:nu*Hc] .= ΔUmax
    end
    if isnothing(Ymin) && !isnothing(ymin)
        size(ymin) == (ny,) || throw(ArgumentError("ymin size must be $((ny,))"))
        for i = 1:ny*Hp
            con.Y0min[i] = ymin[(i-1) % ny + 1] - mpc.Yop[i]
        end
    elseif !isnothing(Ymin)
        size(Ymin)   == (ny*Hp,) || throw(ArgumentError("Ymin size must be $((ny*Hp,))"))
        con.Y0min .= Ymin .- mpc.Yop
    end
    if isnothing(Ymax) && !isnothing(ymax)
        size(ymax) == (ny,) || throw(ArgumentError("ymax size must be $((ny,))"))
        for i = 1:ny*Hp
            con.Y0max[i] = ymax[(i-1) % ny + 1] - mpc.Yop[i]
        end
    elseif !isnothing(Ymax)
        size(Ymax)   == (ny*Hp,) || throw(ArgumentError("Ymax size must be $((ny*Hp,))"))
        con.Y0max .= Ymax .- mpc.Yop
    end
    if !isnothing(x̂min)
        size(x̂min) == (nx̂,) || throw(ArgumentError("x̂min size must be $((nx̂,))"))
        con.x̂0min .= x̂min .- mpc.estim.x̂op
    end
    if !isnothing(x̂max)
        size(x̂max) == (nx̂,) || throw(ArgumentError("x̂max size must be $((nx̂,))"))
        con.x̂0max .= x̂max .- mpc.estim.x̂op
    end
    allECRs = (
        c_umin, c_umax, c_Δumin, c_Δumax, c_ymin, c_ymax,
        C_umin, C_umax, C_Δumin, C_Δumax, C_ymin, C_ymax, c_x̂min, c_x̂max,
    )
    if any(ECR -> !isnothing(ECR), allECRs)
        nϵ == 1 || throw(ArgumentError("Slack variable weight Cwt must be finite to set softness parameters"))
        notSolvedYet || error("Cannot set softness parameters after calling moveinput!")
    end
    if notSolvedYet
        isnothing(C_umin)   && !isnothing(c_umin)   && (C_umin  = repeat(c_umin,  Hp))
        isnothing(C_umax)   && !isnothing(c_umax)   && (C_umax  = repeat(c_umax,  Hp))
        isnothing(C_Δumin)  && !isnothing(c_Δumin)  && (C_Δumin = repeat(c_Δumin, Hc))
        isnothing(C_Δumax)  && !isnothing(c_Δumax)  && (C_Δumax = repeat(c_Δumax, Hc))
        isnothing(C_ymin)   && !isnothing(c_ymin)   && (C_ymin  = repeat(c_ymin,  Hp))
        isnothing(C_ymax)   && !isnothing(c_ymax)   && (C_ymax  = repeat(c_ymax,  Hp))
        if !isnothing(C_umin)
            size(C_umin) == (nu*Hp,) || throw(ArgumentError("C_umin size must be $((nu*Hp,))"))
            any(C_umin .< 0) && error("C_umin weights should be non-negative")
            con.A_Umin[:, end] .= -C_umin
        end
        if !isnothing(C_umax)
            size(C_umax) == (nu*Hp,) || throw(ArgumentError("C_umax size must be $((nu*Hp,))"))
            any(C_umax .< 0) && error("C_umax weights should be non-negative")
            con.A_Umax[:, end] .= -C_umax
        end
        if !isnothing(C_Δumin)
            size(C_Δumin) == (nu*Hc,) || throw(ArgumentError("C_Δumin size must be $((nu*Hc,))"))
            any(C_Δumin .< 0) && error("C_Δumin weights should be non-negative")
            con.A_ΔŨmin[1:end-1, end] .= -C_Δumin 
        end
        if !isnothing(C_Δumax)
            size(C_Δumax) == (nu*Hc,) || throw(ArgumentError("C_Δumax size must be $((nu*Hc,))"))
            any(C_Δumax .< 0) && error("C_Δumax weights should be non-negative")
            con.A_ΔŨmax[1:end-1, end] .= -C_Δumax
        end
        if !isnothing(C_ymin)
            size(C_ymin) == (ny*Hp,) || throw(ArgumentError("C_ymin size must be $((ny*Hp,))"))
            any(C_ymin .< 0) && error("C_ymin weights should be non-negative")
            con.C_ymin .= C_ymin
            size(con.A_Ymin, 1) ≠ 0 && (con.A_Ymin[:, end] .= -con.C_ymin) # for LinModel
        end
        if !isnothing(C_ymax)
            size(C_ymax) == (ny*Hp,) || throw(ArgumentError("C_ymax size must be $((ny*Hp,))"))
            any(C_ymax .< 0) && error("C_ymax weights should be non-negative")
            con.C_ymax .= C_ymax
            size(con.A_Ymax, 1) ≠ 0 && (con.A_Ymax[:, end] .= -con.C_ymax) # for LinModel
        end
        if !isnothing(c_x̂min)
            size(c_x̂min) == (nx̂,) || throw(ArgumentError("c_x̂min size must be $((nx̂,))"))
            any(c_x̂min .< 0) && error("c_x̂min weights should be non-negative")
            con.c_x̂min .= c_x̂min
            size(con.A_x̂min, 1) ≠ 0 && (con.A_x̂min[:, end] .= -con.c_x̂min) # for LinModel
        end
        if !isnothing(c_x̂max)
            size(c_x̂max) == (nx̂,) || throw(ArgumentError("c_x̂max size must be $((nx̂,))"))
            any(c_x̂max .< 0) && error("c_x̂max weights should be non-negative")
            con.c_x̂max .= c_x̂max
            size(con.A_x̂max, 1) ≠ 0 && (con.A_x̂max[:, end] .= -con.c_x̂max) # for LinModel
        end
    end
    i_Umin,  i_Umax  = .!isinf.(con.U0min), .!isinf.(con.U0max)
    i_ΔŨmin, i_ΔŨmax = .!isinf.(con.ΔŨmin), .!isinf.(con.ΔŨmax)
    i_Ymin,  i_Ymax  = .!isinf.(con.Y0min), .!isinf.(con.Y0max)
    i_x̂min,  i_x̂max  = .!isinf.(con.x̂0min), .!isinf.(con.x̂0max)
    if notSolvedYet
        con.i_b[:], con.i_g[:], con.A[:] = init_matconstraint_mpc(model,
            i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, 
            i_Ymin, i_Ymax, i_x̂min, i_x̂max,
            con.A_Umin, con.A_Umax, con.A_ΔŨmin, con.A_ΔŨmax, 
            con.A_Ymin, con.A_Ymax, con.A_x̂min, con.A_x̂max
        )
        A = con.A[con.i_b, :]
        b = con.b[con.i_b]
        ΔŨvar::Vector{JuMP.VariableRef} = optim[:ΔŨvar]
        JuMP.delete(optim, optim[:linconstraint])
        JuMP.unregister(optim, :linconstraint)
        @constraint(optim, linconstraint, A*ΔŨvar .≤ b)
        setnonlincon!(mpc, model, optim)
    else
        i_b, i_g = init_matconstraint_mpc(model, 
            i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, 
            i_Ymin, i_Ymax, i_x̂min, i_x̂max
        )
        if i_b ≠ con.i_b || i_g ≠ con.i_g
            error("Cannot modify ±Inf constraints after calling moveinput!")
        end
    end
    return mpc
end


@doc raw"""
    init_matconstraint_mpc(model::LinModel,
        i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ymin, i_Ymax, i_x̂min, i_x̂max, args...
    ) -> i_b, i_g, A

Init `i_b`, `i_g` and `A` matrices for the linear and nonlinear inequality constraints.

The linear and nonlinear inequality constraints are respectively defined as:
```math
\begin{aligned} 
    \mathbf{A ΔŨ } &≤ \mathbf{b} \\ 
    \mathbf{g(ΔŨ)} &≤ \mathbf{0}
\end{aligned}
```
`i_b` is a `BitVector` including the indices of ``\mathbf{b}`` that are finite numbers. 
`i_g` is a similar vector but for the indices of ``\mathbf{g}`` (empty if `model` is a 
[`LinModel`](@ref)). The method also returns the ``\mathbf{A}`` matrix if `args` is
provided. In such a case, `args`  needs to contain all the inequality constraint matrices: 
`A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, A_Ymin, A_Ymax, A_x̂min, A_x̂max`.
"""
function init_matconstraint_mpc(::LinModel{NT}, 
    i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ymin, i_Ymax, i_x̂min, i_x̂max, args...
) where {NT<:Real}
    i_b = [i_Umin; i_Umax; i_ΔŨmin; i_ΔŨmax; i_Ymin; i_Ymax; i_x̂min; i_x̂max]
    i_g = BitVector()
    if isempty(args)
        A = nothing
    else
        A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, A_Ymin, A_Ymax, A_x̂min, A_x̂max = args
        A = [A_Umin; A_Umax; A_ΔŨmin; A_ΔŨmax; A_Ymin; A_Ymax; A_x̂min; A_x̂max]
    end
    return i_b, i_g, A
end

"Init `i_b, A` without outputs and terminal constraints if `model` is not a [`LinModel`](@ref)."
function init_matconstraint_mpc(::SimModel{NT},
    i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ymin, i_Ymax, i_x̂min, i_x̂max, args...
) where {NT<:Real}
    i_b = [i_Umin; i_Umax; i_ΔŨmin; i_ΔŨmax]
    i_g = [i_Ymin; i_Ymax; i_x̂min; i_x̂max]
    if isempty(args)
        A = nothing
    else
        A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, _ , _ , _ , _ = args
        A = [A_Umin; A_Umax; A_ΔŨmin; A_ΔŨmax]
    end
    return i_b, i_g, A
end

"By default, there is no nonlinear constraint, thus do nothing."
setnonlincon!(::PredictiveController, ::SimModel, ::JuMP.GenericModel) = nothing

"""
    default_Hp(model::LinModel)

Estimate the default prediction horizon `Hp` for [`LinModel`](@ref).
"""
default_Hp(model::LinModel) = DEFAULT_HP0 + estimate_delays(model)
"Throw an error when model is not a [`LinModel`](@ref)."
function default_Hp(::SimModel)
    throw(ArgumentError("Prediction horizon Hp must be explicitly specified if model is not a LinModel."))
end

"""
    estimate_delays(model::LinModel)

Estimate the number of delays in `model` with a security margin.
"""
function estimate_delays(model::LinModel)
    # TODO: also check for settling time (poles)
    # TODO: also check for non minimum phase systems (zeros)
    # TODO: replace sum with max delay between all the I/O
    # TODO: use this nk value for default N value in sim!
    poles = eigvals(model.A)
    # atol=1e-3 to overestimate the number of delays : for closed-loop stability, it is
    # better to overestimate the default value of Hp, as a security margin.
    nk = sum(isapprox.(abs.(poles), 0.0, atol=1e-3)) # number of delays
    return nk
end
"Return `0` when model is not a [`LinModel`](@ref)."
estimate_delays(::SimModel) = 0

"""
    validate_args(mpc::PredictiveController, ry, d, D̂, R̂y, R̂u)

Check the dimensions of the arguments of [`moveinput!`](@ref).
"""
function validate_args(mpc::PredictiveController, ry, d, D̂, R̂y, R̂u)
    ny, nd, nu, Hp = mpc.estim.model.ny, mpc.estim.model.nd, mpc.estim.model.nu, mpc.Hp
    size(ry) ≠ (ny,)    && throw(DimensionMismatch("ry size $(size(ry)) ≠ output size ($ny,)"))
    size(d)  ≠ (nd,)    && throw(DimensionMismatch("d size $(size(d)) ≠ measured dist. size ($nd,)"))
    size(D̂)  ≠ (nd*Hp,) && throw(DimensionMismatch("D̂ size $(size(D̂)) ≠ measured dist. size × Hp ($(nd*Hp),)"))
    size(R̂y) ≠ (ny*Hp,) && throw(DimensionMismatch("R̂y size $(size(R̂y)) ≠ output size × Hp ($(ny*Hp),)"))
    if ~mpc.noR̂u
        size(R̂u) ≠ (nu*Hp,) && throw(DimensionMismatch("R̂u size $(size(R̂u)) ≠ manip. input size × Hp ($(nu*Hp),)"))
    end
end


@doc raw"""
    init_ΔUtoU(model, Hp, Hc) -> S, T

Init manipulated input increments to inputs conversion matrices.

The conversion from the input increments ``\mathbf{ΔU}`` to manipulated inputs over ``H_p`` 
are calculated by:
```math
\mathbf{U} = \mathbf{S} \mathbf{ΔU} + \mathbf{T} \mathbf{u}(k-1) \\
```
"""
function init_ΔUtoU(model::SimModel{NT}, Hp, Hc) where {NT<:Real}
    # S and T are `Matrix{NT}` since conversion is faster than `Matrix{Bool}` or `BitMatrix`
    I_nu = Matrix{NT}(I, model.nu, model.nu)
    S_Hc = LowerTriangular(repeat(I_nu, Hc, Hc))
    S = [S_Hc; repeat(I_nu, Hp - Hc, Hc)]
    T = repeat(I_nu, Hp)
    return S, T
end


@doc raw"""
    init_predmat(estim, model::LinModel, Hp, Hc) -> E, G, J, K, V, ex̂, gx̂, jx̂, kx̂, vx̂

Construct the prediction matrices for [`LinModel`](@ref) `model`.

The model predictions are evaluated from the deviation vectors (see [`setop!`](@ref)) and:
```math
\begin{aligned}
    \mathbf{Ŷ_0} &= \mathbf{E ΔU} + \mathbf{G d_0}(k) + \mathbf{J D̂_0} 
                                  + \mathbf{K x̂_0}(k) + \mathbf{V u_0}(k-1) 
                                  + \mathbf{B}        + \mathbf{Ŷ_s}                      \\
                 &= \mathbf{E ΔU} + \mathbf{F}
\end{aligned}
```
in which ``\mathbf{x̂_0}(k) = \mathbf{x̂}_i(k) - \mathbf{x̂_{op}}``, with ``i = k`` if 
`estim.direct==true`, otherwise ``i = k - 1``. The predicted outputs ``\mathbf{Ŷ_0}`` and
measured disturbances ``\mathbf{D̂_0}`` respectively include ``\mathbf{ŷ_0}(k+j)`` and 
``\mathbf{d̂_0}(k+j)`` values with ``j=1`` to ``H_p``, and input increments ``\mathbf{ΔU}``,
``\mathbf{Δu}(k+j)`` from ``j=0`` to ``H_c-1``. The vector ``\mathbf{B}`` contains the
contribution for non-zero state ``\mathbf{x̂_{op}}`` and state update ``\mathbf{f̂_{op}}``
operating points (for linearization at non-equilibrium point, see [`linearize`](@ref)). The
stochastic predictions ``\mathbf{Ŷ_s=0}`` if `estim` is not a [`InternalModel`](@ref), see
[`init_stochpred`](@ref). The method also computes similar matrices for the predicted
terminal states at ``k+H_p``:
```math
\begin{aligned}
    \mathbf{x̂_0}(k+H_p) &= \mathbf{e_x̂ ΔU} + \mathbf{g_x̂ d_0}(k)   + \mathbf{j_x̂ D̂_0} 
                                           + \mathbf{k_x̂ x̂_0}(k) + \mathbf{v_x̂ u_0}(k-1)
                                           + \mathbf{b_x̂}                                 \\
                        &= \mathbf{e_x̂ ΔU} + \mathbf{f_x̂}
\end{aligned}
```
The matrices ``\mathbf{E, G, J, K, V, B, e_x̂, g_x̂, j_x̂, k_x̂, v_x̂, b_x̂}`` are defined in the
Extended Help section. The ``\mathbf{F}`` and ``\mathbf{f_x̂}`` vectors are  recalculated at
each control period ``k``, see [`initpred!`](@ref) and [`linconstraint!`](@ref).

# Extended Help
!!! details "Extended Help"
    Using the augmented matrices ``\mathbf{Â, B̂_u, Ĉ, B̂_d, D̂_d}`` in `estim` (see 
    [`augment_model`](@ref)), and the function ``\mathbf{W}(j) = ∑_{i=0}^j \mathbf{Â}^i``,
    the prediction matrices are computed by :
    ```math
    \begin{aligned}
    \mathbf{E} &= \begin{bmatrix}
        \mathbf{Ĉ W}(0)\mathbf{B̂_u}     & \mathbf{0}                      & \cdots & \mathbf{0}                                        \\
        \mathbf{Ĉ W}(1)\mathbf{B̂_u}     & \mathbf{Ĉ W}(0)\mathbf{B̂_u}     & \cdots & \mathbf{0}                                        \\
        \vdots                          & \vdots                          & \ddots & \vdots                                            \\
        \mathbf{Ĉ W}(H_p-1)\mathbf{B̂_u} & \mathbf{Ĉ W}(H_p-2)\mathbf{B̂_u} & \cdots & \mathbf{Ĉ W}(H_p-H_c+1)\mathbf{B̂_u} \end{bmatrix} \\
    \mathbf{G} &= \begin{bmatrix}
        \mathbf{Ĉ}\mathbf{Â}^{0} \mathbf{B̂_d}     \\ 
        \mathbf{Ĉ}\mathbf{Â}^{1} \mathbf{B̂_d}     \\ 
        \vdots                                    \\
        \mathbf{Ĉ}\mathbf{Â}^{H_p-1} \mathbf{B̂_d} \end{bmatrix} \\
    \mathbf{J} &= \begin{bmatrix}
        \mathbf{D̂_d}                              & \mathbf{0}                                & \cdots & \mathbf{0}   \\ 
        \mathbf{Ĉ}\mathbf{Â}^{0} \mathbf{B̂_d}     & \mathbf{D̂_d}                              & \cdots & \mathbf{0}   \\ 
        \vdots                                    & \vdots                                    & \ddots & \vdots       \\
        \mathbf{Ĉ}\mathbf{Â}^{H_p-2} \mathbf{B̂_d} & \mathbf{Ĉ}\mathbf{Â}^{H_p-3} \mathbf{B̂_d} & \cdots & \mathbf{D̂_d} \end{bmatrix} \\
    \mathbf{K} &= \begin{bmatrix}
        \mathbf{Ĉ}\mathbf{Â}^{1}        \\
        \mathbf{Ĉ}\mathbf{Â}^{2}        \\
        \vdots                          \\
        \mathbf{Ĉ}\mathbf{Â}^{H_p}      \end{bmatrix} \\
    \mathbf{V} &= \begin{bmatrix}
        \mathbf{Ĉ W}(0)\mathbf{B̂_u}     \\
        \mathbf{Ĉ W}(1)\mathbf{B̂_u}     \\
        \vdots                          \\
        \mathbf{Ĉ W}(H_p-1)\mathbf{B̂_u} \end{bmatrix} \\
    \mathbf{B} &= \begin{bmatrix}
        \mathbf{Ĉ W}(0)                 \\
        \mathbf{Ĉ W}(1)                 \\
        \vdots                          \\
        \mathbf{Ĉ W}(H_p-1)             \end{bmatrix}   \mathbf{\big(f̂_{op} - x̂_{op}\big)} 
    \end{aligned}
    ```
    For the terminal constraints, the matrices are computed with:
    ```math
    \begin{aligned}
    \mathbf{e_x̂} &= \begin{bmatrix} 
                        \mathbf{W}(H_p-1)\mathbf{B̂_u} & 
                        \mathbf{W}(H_p-2)\mathbf{B̂_u} & 
                        \cdots & 
                        \mathbf{W}(H_p-H_c+1)\mathbf{B̂_u} \end{bmatrix} \\
    \mathbf{g_x̂} &= \mathbf{Â}^{H_p-1} \mathbf{B̂_d} \\
    \mathbf{j_x̂} &= \begin{bmatrix} 
                        \mathbf{Â}^{H_p-2} \mathbf{B̂_d} & 
                        \mathbf{Â}^{H_p-3} \mathbf{B̂_d} & 
                        \cdots & 
                        \mathbf{0} 
                    \end{bmatrix} \\
    \mathbf{k_x̂} &= \mathbf{Â}^{H_p} \\
    \mathbf{v_x̂} &= \mathbf{W}(H_p-1)\mathbf{B̂_u} \\
    \mathbf{b_x̂} &= \mathbf{W}(H_p-1)    \mathbf{\big(f̂_{op} - x̂_{op}\big)}
    \end{aligned}
    ```
"""
function init_predmat(estim::StateEstimator{NT}, model::LinModel, Hp, Hc) where {NT<:Real}
    Â, B̂u, Ĉ, B̂d, D̂d = estim.Â, estim.B̂u, estim.Ĉ, estim.B̂d, estim.D̂d
    nu, nx̂, ny, nd = model.nu, estim.nx̂, model.ny, model.nd
    # --- pre-compute matrix powers ---
    # Apow 3D array : Apow[:,:,1] = A^0, Apow[:,:,2] = A^1, ... , Apow[:,:,Hp+1] = A^Hp
    Âpow = Array{NT}(undef, nx̂, nx̂, Hp+1)
    Âpow[:,:,1] = I(nx̂)
    for j=2:Hp+1
        Âpow[:,:,j] = @views Âpow[:,:,j-1]*Â
    end
    # Apow_csum 3D array : Apow_csum[:,:,1] = A^0, Apow_csum[:,:,2] = A^1 + A^0, ...
    Âpow_csum  = cumsum(Âpow, dims=3)
    # helper function to improve code clarity and be similar to eqs. in docstring:
    getpower(array3D, power) = @views array3D[:,:, power+1]
    # --- state estimates x̂ ---
    kx̂ = getpower(Âpow, Hp)
    K  = Matrix{NT}(undef, Hp*ny, nx̂)
    for j=1:Hp
        iRow = (1:ny) .+ ny*(j-1)
        K[iRow,:] = Ĉ*getpower(Âpow, j)
    end    
    # --- manipulated inputs u ---
    vx̂ = getpower(Âpow_csum, Hp-1)*B̂u
    V  = Matrix{NT}(undef, Hp*ny, nu)
    for j=1:Hp
        iRow = (1:ny) .+ ny*(j-1)
        V[iRow,:] = Ĉ*getpower(Âpow_csum, j-1)*B̂u
    end
    ex̂ = Matrix{NT}(undef, nx̂, Hc*nu)
    E  = zeros(NT, Hp*ny, Hc*nu) 
    for j=1:Hc # truncated with control horizon
        iRow = (ny*(j-1)+1):(ny*Hp)
        iCol = (1:nu) .+ nu*(j-1)
        E[iRow, iCol] = V[iRow .- ny*(j-1),:]
        ex̂[:  , iCol] = getpower(Âpow_csum, Hp-j)*B̂u
    end
    # --- measured disturbances d ---
    gx̂ = getpower(Âpow, Hp-1)*B̂d
    G  = Matrix{NT}(undef, Hp*ny, nd)
    jx̂ = Matrix{NT}(undef, nx̂, Hp*nd)
    J  = repeatdiag(D̂d, Hp)
    if nd ≠ 0
        for j=1:Hp
            iRow = (1:ny) .+ ny*(j-1)
            G[iRow,:] = Ĉ*getpower(Âpow, j-1)*B̂d
        end
        for j=1:Hp
            iRow = (ny*j+1):(ny*Hp)
            iCol = (1:nd) .+ nd*(j-1)
            J[iRow, iCol] = G[iRow .- ny*j,:]
            jx̂[:  , iCol] = j < Hp ? getpower(Âpow, Hp-j-1)*B̂d : zeros(NT, nx̂, nd)
        end
    end
    # --- state x̂op and state update f̂op operating points ---
    coef_bx̂ = getpower(Âpow_csum, Hp-1)
    coef_B  = Matrix{NT}(undef, ny*Hp, nx̂)
    for j=1:Hp
        iRow = (1:ny) .+ ny*(j-1)
        coef_B[iRow,:] = Ĉ*getpower(Âpow_csum, j-1)
    end
    f̂op_n_x̂op = estim.f̂op - estim.x̂op
    bx̂ = coef_bx̂ * f̂op_n_x̂op
    B  = coef_B  * f̂op_n_x̂op
    return E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂
end

"Return empty matrices if `model` is not a [`LinModel`](@ref)"
function init_predmat(estim::StateEstimator{NT}, model::SimModel, Hp, Hc) where {NT<:Real}
    nu, nx̂, nd = model.nu, estim.nx̂, model.nd
    E  = zeros(NT, 0, nu*Hc)
    G  = zeros(NT, 0, nd)
    J  = zeros(NT, 0, nd*Hp)
    K  = zeros(NT, 0, nx̂)
    V  = zeros(NT, 0, nu)
    B  = zeros(NT, 0)
    ex̂, gx̂, jx̂, kx̂, vx̂, bx̂ = E, G, J, K, V, B
    return E, G, J, K, V, B, ex̂, gx̂, jx̂, kx̂, vx̂, bx̂
end

@doc raw"""
    init_quadprog(model::LinModel, Ẽ, S, M_Hp, N_Hc, L_Hp) -> H̃

Init the quadratic programming Hessian `H̃` for MPC.

The matrix appear in the quadratic general form:
```math
    J = \min_{\mathbf{ΔŨ}} \frac{1}{2}\mathbf{(ΔŨ)'H̃(ΔŨ)} + \mathbf{q̃'(ΔŨ)} + r 
```
The Hessian matrix is constant if the model and weights are linear and time invariant (LTI): 
```math
    \mathbf{H̃} = 2 (  \mathbf{Ẽ}'\mathbf{M}_{H_p}\mathbf{Ẽ} + \mathbf{Ñ}_{H_c} 
                    + \mathbf{S̃}'\mathbf{L}_{H_p}\mathbf{S̃} )
```
The vector ``\mathbf{q̃}`` and scalar ``r`` need recalculation each control period ``k``, see
[`initpred!`](@ref). ``r`` does not impact the minima position. It is thus useless at
optimization but required to evaluate the minimal ``J`` value.
"""
function init_quadprog(::LinModel, Ẽ, S̃, M_Hp, Ñ_Hc, L_Hp)
    H̃ = Hermitian(2*(Ẽ'*M_Hp*Ẽ + Ñ_Hc + S̃'*L_Hp*S̃), :L)
    return H̃
end
"Return empty matrices if `model` is not a [`LinModel`](@ref)."
function init_quadprog(::SimModel{NT}, Ẽ, S̃, M_Hp, Ñ_Hc, L_Hp) where {NT<:Real}
    H̃ = Hermitian(zeros(NT, 0, 0), :L)
    return H̃
end

"""
    init_defaultcon_mpc(
        estim, C, S, N_Hc, E, ex̂, fx̂, gx̂, jx̂, kx̂, vx̂, 
        gc!=(_,_,_,_,_,_)->nothing, nc=0
    ) -> con, S̃, Ñ_Hc, Ẽ

Init `ControllerConstraint` struct with default parameters based on estimator `estim`.

Also return `S̃`, `Ñ_Hc` and `Ẽ` matrices for the the augmented decision vector `ΔŨ`.
"""
function init_defaultcon_mpc(
    estim::StateEstimator{NT}, 
    Hp, Hc, C, S, N_Hc, E, ex̂, fx̂, gx̂, jx̂, kx̂, vx̂, bx̂, 
    gc!::GCfunc=(_,_,_,_,_,_)->nothing, nc=0
) where {NT<:Real, GCfunc<:Function}
    model = estim.model
    nu, ny, nx̂ = model.nu, model.ny, estim.nx̂
    nϵ = isinf(C) ? 0 : 1
    u0min,      u0max   = fill(convert(NT,-Inf), nu), fill(convert(NT,+Inf), nu)
    Δumin,      Δumax   = fill(convert(NT,-Inf), nu), fill(convert(NT,+Inf), nu)
    y0min,      y0max   = fill(convert(NT,-Inf), ny), fill(convert(NT,+Inf), ny)
    x̂0min,      x̂0max   = fill(convert(NT,-Inf), nx̂), fill(convert(NT,+Inf), nx̂)
    c_umin,     c_umax  = fill(zero(NT), nu), fill(zero(NT), nu)
    c_Δumin,    c_Δumax = fill(zero(NT), nu), fill(zero(NT), nu)
    c_ymin,     c_ymax  = fill(one(NT),  ny), fill(one(NT),  ny)
    c_x̂min,     c_x̂max  = fill(zero(NT), nx̂), fill(zero(NT), nx̂)
    U0min, U0max, ΔUmin, ΔUmax, Y0min, Y0max = 
        repeat_constraints(Hp, Hc, u0min, u0max, Δumin, Δumax, y0min, y0max)
    C_umin, C_umax, C_Δumin, C_Δumax, C_ymin, C_ymax = 
        repeat_constraints(Hp, Hc, c_umin, c_umax, c_Δumin, c_Δumax, c_ymin, c_ymax)
    A_Umin,  A_Umax, S̃  = relaxU(model, nϵ, C_umin, C_umax, S)
    A_ΔŨmin, A_ΔŨmax, ΔŨmin, ΔŨmax, Ñ_Hc = relaxΔU(
        model, nϵ, C, C_Δumin, C_Δumax, ΔUmin, ΔUmax, N_Hc
    )
    A_Ymin,  A_Ymax, Ẽ  = relaxŶ(model, nϵ, C_ymin, C_ymax, E)
    A_x̂min,  A_x̂max, ẽx̂ = relaxterminal(model, nϵ, c_x̂min, c_x̂max, ex̂)
    i_Umin,  i_Umax  = .!isinf.(U0min),  .!isinf.(U0max)
    i_ΔŨmin, i_ΔŨmax = .!isinf.(ΔŨmin), .!isinf.(ΔŨmax)
    i_Ymin,  i_Ymax  = .!isinf.(Y0min),  .!isinf.(Y0max)
    i_x̂min,  i_x̂max  = .!isinf.(x̂0min),  .!isinf.(x̂0max)
    i_b, i_g, A = init_matconstraint_mpc(
        model, 
        i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ymin, i_Ymax, i_x̂min, i_x̂max,
        A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, A_Ymin, A_Ymax, A_x̂max, A_x̂min
    )
    b = zeros(NT, size(A, 1)) # dummy b vector (updated just before optimization)
    con = ControllerConstraint{NT, GCfunc}(
        ẽx̂      , fx̂    , gx̂     , jx̂       , kx̂     , vx̂     , bx̂     ,
        U0min   , U0max , ΔŨmin  , ΔŨmax    , Y0min  , Y0max  , x̂0min  , x̂0max,
        A_Umin  , A_Umax, A_ΔŨmin, A_ΔŨmax  , A_Ymin , A_Ymax , A_x̂min , A_x̂max,
        A       , b     , i_b    , C_ymin   , C_ymax , c_x̂min , c_x̂max , i_g,
        gc!     , nc
    )
    return con, nϵ, S̃, Ñ_Hc, Ẽ
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
    relaxU(model, nϵ, C_umin, C_umax, S) -> A_Umin, A_Umax, S̃

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
    - \mathbf{(U_{min} - U_{op}) + T} \mathbf{u_0}(k-1) \\
    + \mathbf{(U_{max} - U_{op}) - T} \mathbf{u_0}(k-1)
\end{bmatrix}
```
in which ``\mathbf{U_{min}, U_{max}}`` and ``\mathbf{U_{op}}`` vectors respectively contains
``\mathbf{u_{min}, u_{max}}`` and ``\mathbf{u_{op}}`` repeated ``H_p`` times.
"""
function relaxU(::SimModel{NT}, nϵ, C_umin, C_umax, S) where NT<:Real
    if nϵ == 1 # ΔŨ = [ΔU; ϵ]
        # ϵ impacts ΔU → U conversion for constraint calculations:
        A_Umin, A_Umax = -[S  C_umin], [S -C_umax] 
        # ϵ has no impact on ΔU → U conversion for prediction calculations:
        S̃ = [S zeros(NT, size(S, 1))]
    else # ΔŨ = ΔU (only hard constraints)
        A_Umin, A_Umax = -S,  S
        S̃ = S
    end
    return A_Umin, A_Umax, S̃
end

@doc raw"""
    relaxΔU(
        model, nϵ, C, C_Δumin, C_Δumax, ΔUmin, ΔUmax, N_Hc
    ) -> A_ΔŨmin, A_ΔŨmax, ΔŨmin, ΔŨmax, Ñ_Hc

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
function relaxΔU(::SimModel{NT}, nϵ, C, C_Δumin, C_Δumax, ΔUmin, ΔUmax, N_Hc) where NT<:Real
    nΔU = size(N_Hc, 1)
    if nϵ == 1 # ΔŨ = [ΔU; ϵ]
        # 0 ≤ ϵ ≤ ∞  
        ΔŨmin, ΔŨmax = [ΔUmin; NT[0.0]], [ΔUmax; NT[Inf]]
        A_ϵ = [zeros(NT, 1, length(ΔUmin)) NT[1.0]]
        A_ΔŨmin, A_ΔŨmax = -[I  C_Δumin; A_ϵ], [I -C_Δumax; A_ϵ]
        Ñ_Hc = Hermitian([N_Hc zeros(NT, nΔU, 1);zeros(NT, 1, nΔU) C], :L)
    else # ΔŨ = ΔU (only hard constraints)
        ΔŨmin, ΔŨmax = ΔUmin, ΔUmax
        I_Hc = Matrix{NT}(I, nΔU, nΔU)
        A_ΔŨmin, A_ΔŨmax = -I_Hc,  I_Hc
        Ñ_Hc = N_Hc
    end
    return A_ΔŨmin, A_ΔŨmax, ΔŨmin, ΔŨmax, Ñ_Hc
end

@doc raw"""
    relaxŶ(::LinModel, nϵ, C_ymin, C_ymax, E) -> A_Ymin, A_Ymax, Ẽ

Augment linear output prediction constraints with slack variable ϵ for softening.

Denoting the input increments augmented with the slack variable 
``\mathbf{ΔŨ} = [\begin{smallmatrix} \mathbf{ΔU} \\ ϵ \end{smallmatrix}]``, it returns the 
``\mathbf{Ẽ}`` matrix that appears in the linear model prediction equation 
``\mathbf{Ŷ_0 = Ẽ ΔŨ + F}``, and the ``\mathbf{A}`` matrices for the inequality constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{Y_{min}}} \\ 
    \mathbf{A_{Y_{max}}}
\end{bmatrix} \mathbf{ΔŨ} ≤
\begin{bmatrix}
    - \mathbf{(Y_{min} - Y_{op}) + F} \\
    + \mathbf{(Y_{max} - Y_{op}) - F} 
\end{bmatrix}
```
in which ``\mathbf{Y_{min}, Y_{max}}`` and ``\mathbf{Y_{op}}`` vectors respectively contains
``\mathbf{y_{min}, y_{max}}`` and ``\mathbf{y_{op}}`` repeated ``H_p`` times.
"""
function relaxŶ(::LinModel{NT}, nϵ, C_ymin, C_ymax, E) where NT<:Real
    if nϵ == 1 # ΔŨ = [ΔU; ϵ]
        # ϵ impacts predicted output constraint calculations:
        A_Ymin, A_Ymax = -[E  C_ymin], [E -C_ymax] 
        # ϵ has no impact on output predictions:
        Ẽ = [E zeros(NT, size(E, 1), 1)] 
    else # ΔŨ = ΔU (only hard constraints)
        Ẽ = E
        A_Ymin, A_Ymax = -E,  E
    end
    return A_Ymin, A_Ymax, Ẽ
end

"Return empty matrices if model is not a [`LinModel`](@ref)"
function relaxŶ(::SimModel{NT}, nϵ, C_ymin, C_ymax, E) where NT<:Real
    Ẽ = [E zeros(NT, 0, nϵ)]
    A_Ymin, A_Ymax = -Ẽ,  Ẽ 
    return A_Ymin, A_Ymax, Ẽ
end

@doc raw"""
    relaxterminal(::LinModel, nϵ, c_x̂min, c_x̂max, ex̂) -> A_x̂min, A_x̂max, ẽx̂

Augment terminal state constraints with slack variable ϵ for softening.

Denoting the input increments augmented with the slack variable 
``\mathbf{ΔŨ} = [\begin{smallmatrix} \mathbf{ΔU} \\ ϵ \end{smallmatrix}]``, it returns the 
``\mathbf{ẽ_{x̂}}`` matrix that appears in the terminal state equation 
``\mathbf{x̂_0}(k + H_p) = \mathbf{ẽ_x̂ ΔŨ + f_x̂}``, and the ``\mathbf{A}`` matrices for 
the inequality constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{x̂_{min}}} \\ 
    \mathbf{A_{x̂_{max}}}
\end{bmatrix} \mathbf{ΔŨ} ≤
\begin{bmatrix}
    - \mathbf{(x̂_{min} - x̂_{op}) + f_x̂} \\
    + \mathbf{(x̂_{max} - x̂_{op}) - f_x̂}
\end{bmatrix}
```
"""
function relaxterminal(::LinModel{NT}, nϵ, c_x̂min, c_x̂max, ex̂) where {NT<:Real}
    if nϵ == 1 # ΔŨ = [ΔU; ϵ]
        # ϵ impacts terminal state constraint calculations:
        A_x̂min, A_x̂max = -[ex̂ c_x̂min], [ex̂ -c_x̂max]
        # ϵ has no impact on terminal state predictions:
        ẽx̂ = [ex̂ zeros(NT, size(ex̂, 1), 1)] 
    else # ΔŨ = ΔU (only hard constraints)
        ẽx̂ = ex̂
        A_x̂min, A_x̂max = -ex̂,  ex̂
    end
    return A_x̂min, A_x̂max, ẽx̂
end

"Return empty matrices if model is not a [`LinModel`](@ref)"
function relaxterminal(::SimModel{NT}, nϵ, c_x̂min, c_x̂max, ex̂) where {NT<:Real}
    ẽx̂ = [ex̂ zeros(NT, 0, nϵ)]
    A_x̂min, A_x̂max = -ẽx̂,  ẽx̂
    return A_x̂min, A_x̂max, ẽx̂
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

[^2]: Desbiens, A., D. Hodouin & É. Plamondon. 2000, "Global predictive control: a unified
    control structure for decoupling setpoint tracking, feedforward compensation and 
    disturbance rejection dynamics", *IEE Proceedings - Control Theory and Applications*, 
    vol. 147, no 4, <https://doi.org/10.1049/ip-cta:20000443>, p. 465–475, ISSN 1350-2379.
"""
function init_stochpred(estim::InternalModel{NT}, Hp) where NT<:Real
    As, B̂s, Cs = estim.As, estim.B̂s, estim.Cs
    ny  = estim.model.ny
    nxs = estim.nxs
    Ks = Matrix{NT}(undef, ny*Hp, nxs)
    Ps = Matrix{NT}(undef, ny*Hp, ny)
    for i = 1:Hp
        iRow = (1:ny) .+ ny*(i-1)
        Ms = Cs*As^(i-1)*B̂s
        Ks[iRow,:] = Cs*As^i - Ms*Cs
        Ps[iRow,:] = Ms
    end
    return Ks, Ps 
end

"Return empty matrices if `estim` is not a [`InternalModel`](@ref)."
function init_stochpred(estim::StateEstimator{NT}, _ ) where NT<:Real
    return zeros(NT, 0, estim.nxs), zeros(NT, 0, estim.model.ny)
end

"Validate predictive controller weight and horizon specified values."
function validate_weights(model, Hp, Hc, M_Hp, N_Hc, L_Hp, C=Inf, E=nothing)
    nu, ny = model.nu, model.ny
    nM, nN, nL = ny*Hp, nu*Hc, nu*Hp
    Hp < 1  && throw(ArgumentError("Prediction horizon Hp should be ≥ 1"))
    Hc < 1  && throw(ArgumentError("Control horizon Hc should be ≥ 1"))
    Hc > Hp && throw(ArgumentError("Control horizon Hc should be ≤ prediction horizon Hp"))
    size(M_Hp) ≠ (nM,nM) && throw(ArgumentError("M_Hp size $(size(M_Hp)) ≠ (ny*Hp, ny*Hp) ($nM,$nM)"))
    size(N_Hc) ≠ (nN,nN) && throw(ArgumentError("N_Hc size $(size(N_Hc)) ≠ (nu*Hc, nu*Hc) ($nN,$nN)"))
    size(L_Hp) ≠ (nL,nL) && throw(ArgumentError("L_Hp size $(size(L_Hp)) ≠ (nu*Hp, nu*Hp) ($nL,$nL)"))
    (isdiag(M_Hp) && any(diag(M_Hp) .< 0)) && throw(ArgumentError("Mwt values should be nonnegative"))
    (isdiag(N_Hc) && any(diag(N_Hc) .< 0)) && throw(ArgumentError("Nwt values should be nonnegative"))
    (isdiag(L_Hp) && any(diag(L_Hp) .< 0)) && throw(ArgumentError("Lwt values should be nonnegative"))
    !ishermitian(M_Hp) && throw(ArgumentError("M_Hp should be hermitian"))
    !ishermitian(N_Hc) && throw(ArgumentError("N_Hc should be hermitian"))
    !ishermitian(L_Hp) && throw(ArgumentError("L_Hp should be hermitian"))
    size(C) ≠ ()    && throw(ArgumentError("Cwt should be a real scalar"))
    C < 0           && throw(ArgumentError("Cwt weight should be ≥ 0"))
    !isnothing(E) && size(E) ≠ () && throw(ArgumentError("Ewt should be a real scalar"))
end