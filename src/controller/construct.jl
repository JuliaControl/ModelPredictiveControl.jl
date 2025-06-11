const MSG_LINMODEL_ERR = "estim.model type must be a LinModel, see ManualEstimator docstring "*
                         "to use a nonlinear state estimator with a linear controller"

struct PredictiveControllerBuffer{NT<:Real}
    u ::Vector{NT}
    Z̃ ::Vector{NT}
    D̂ ::Vector{NT}
    Ŷ ::Vector{NT}
    U ::Vector{NT}
    Ẽ ::Matrix{NT}
    P̃u::Matrix{NT}
    empty::Vector{NT}
end

@doc raw"""
    PredictiveControllerBuffer(estim, transcription, Hp, Hc, nϵ)

Create a buffer for `PredictiveController` objects.

The buffer is used to store intermediate results during computation without allocating.
"""
function PredictiveControllerBuffer(
    estim::StateEstimator{NT}, transcription::TranscriptionMethod, Hp::Int, Hc::Int, nϵ::Int
) where NT <: Real
    nu, ny, nd, nx̂ = estim.model.nu, estim.model.ny, estim.model.nd, estim.nx̂
    nZ̃ = get_nZ(estim, transcription, Hp, Hc) + nϵ
    u  = Vector{NT}(undef, nu)
    Z̃  = Vector{NT}(undef, nZ̃)
    D̂  = Vector{NT}(undef, nd*Hp)
    Ŷ  = Vector{NT}(undef, ny*Hp)
    U  = Vector{NT}(undef, nu*Hp)
    Ẽ  = Matrix{NT}(undef, ny*Hp, nZ̃)
    P̃u = Matrix{NT}(undef, nu*Hp, nZ̃)
    empty = Vector{NT}(undef, 0)
    return PredictiveControllerBuffer{NT}(u, Z̃, D̂, Ŷ, U, Ẽ, P̃u, empty)
end

"Include all the objective function weights of [`PredictiveController`](@ref)"
struct ControllerWeights{
    NT<:Real,
    # parameters to support both dense and Diagonal matrices (with specialization):
    MW<:AbstractMatrix{NT}, 
    NW<:AbstractMatrix{NT},  
    LW<:AbstractMatrix{NT}, 
}
    M_Hp::Hermitian{NT, MW}
    Ñ_Hc::Hermitian{NT, NW}
    L_Hp::Hermitian{NT, LW}
    E   ::NT
    iszero_M_Hp::Vector{Bool}
    iszero_Ñ_Hc::Vector{Bool}
    iszero_L_Hp::Vector{Bool}
    iszero_E::Bool
    isinf_C ::Bool
    function ControllerWeights{NT}(
        model, Hp, Hc, M_Hp::MW, N_Hc::NW, L_Hp::LW, Cwt=Inf, Ewt=0
    ) where {
        NT<:Real, 
        MW<:AbstractMatrix{NT}, 
        NW<:AbstractMatrix{NT}, 
        LW<:AbstractMatrix{NT}
    }
        validate_weights(model, Hp, Hc, M_Hp, N_Hc, L_Hp, Cwt, Ewt)
        nΔU = size(N_Hc, 1)
        C = Cwt
        isinf_C = isinf(C)
        if !isinf_C  
            # ΔŨ = [ΔU; ϵ] (ϵ is the slack variable)
            Ñ_Hc = [N_Hc zeros(NT, nΔU, 1); zeros(NT, 1, nΔU) C]
            isdiag(N_Hc) && (Ñ_Hc = Diagonal(Ñ_Hc)) # NW(Ñ_Hc) does not work on Julia 1.10
        else
            # ΔŨ = ΔU (only hard constraints)
            Ñ_Hc = N_Hc
        end
        M_Hp = Hermitian(M_Hp, :L)
        Ñ_Hc = Hermitian(Ñ_Hc, :L)
        L_Hp = Hermitian(L_Hp, :L)
        E = Ewt
        iszero_M_Hp = [iszero(M_Hp)]
        iszero_Ñ_Hc = [iszero(Ñ_Hc)]
        iszero_L_Hp = [iszero(L_Hp)]
        iszero_E = iszero(E)
        return new{NT, MW, NW, LW}(
            M_Hp, Ñ_Hc, L_Hp, E, 
            iszero_M_Hp, iszero_Ñ_Hc, iszero_L_Hp, iszero_E, isinf_C
        )
    end
end

"Outer constructor to convert weight matrix number type to `NT` if necessary."
function ControllerWeights{NT}(
        model, Hp, Hc, M_Hp::MW, N_Hc::NW, L_Hp::LW, Cwt=Inf, Ewt=0
    ) where {NT<:Real, MW<:AbstractMatrix, NW<:AbstractMatrix, LW<:AbstractMatrix}
    return ControllerWeights{NT}(model, Hp, Hc, NT.(M_Hp), NT.(N_Hc), NT.(L_Hp), Cwt, Ewt)
end

"Include all the data for the constraints of [`PredictiveController`](@ref)"
struct ControllerConstraint{NT<:Real, GCfunc<:Union{Nothing, Function}}
    # matrices for the terminal constraints:
    ẽx̂      ::Matrix{NT}
    fx̂      ::Vector{NT}
    gx̂      ::Matrix{NT}
    jx̂      ::Matrix{NT}
    kx̂      ::Matrix{NT}
    vx̂      ::Matrix{NT}
    bx̂      ::Vector{NT}
    # matrices for the zero defect constraints (N/A for single shooting transcriptions):
    Ẽŝ      ::Matrix{NT}
    Fŝ      ::Vector{NT}
    Gŝ      ::Matrix{NT}
    Jŝ      ::Matrix{NT}
    Kŝ      ::Matrix{NT}
    Vŝ      ::Matrix{NT}
    Bŝ      ::Vector{NT}
    # bounds over the prediction horizon (deviation vectors from operating points):
    U0min   ::Vector{NT}
    U0max   ::Vector{NT}
    ΔŨmin   ::Vector{NT}
    ΔŨmax   ::Vector{NT}
    Y0min   ::Vector{NT}
    Y0max   ::Vector{NT}
    x̂0min   ::Vector{NT}
    x̂0max   ::Vector{NT}
    # A matrices for the linear inequality constraints:
    A_Umin  ::Matrix{NT}
    A_Umax  ::Matrix{NT}
    A_ΔŨmin ::Matrix{NT}
    A_ΔŨmax ::Matrix{NT}
    A_Ymin  ::Matrix{NT}
    A_Ymax  ::Matrix{NT}
    A_x̂min  ::Matrix{NT}
    A_x̂max  ::Matrix{NT}
    A       ::Matrix{NT}
    # b vector for the linear inequality constraints:
    b       ::Vector{NT}
    # indices of finite numbers in the b vector (linear inequality constraints):
    i_b     ::BitVector
    # A matrices for the linear equality constraints:
    A_ŝ     ::Matrix{NT}
    Aeq     ::Matrix{NT}
    # b vector for the linear equality constraints:
    beq     ::Vector{NT}
    # nonlinear equality constraints:
    neq     ::Int
    # constraint softness parameter vectors for the nonlinear inequality constraints:
    C_ymin  ::Vector{NT}
    C_ymax  ::Vector{NT}
    c_x̂min  ::Vector{NT}
    c_x̂max  ::Vector{NT}
    # indices of finite numbers in the g vector (nonlinear inequality constraints):
    i_g     ::BitVector
    # custom nonlinear inequality constraints:
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
    model, con =  mpc.estim.model, mpc.con
    transcription, optim = mpc.transcription, mpc.optim
    nu, ny, nx̂, Hp, Hc = model.nu, model.ny, mpc.estim.nx̂, mpc.Hp, mpc.Hc
    nϵ, nc = mpc.nϵ, con.nc
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
        con.i_b[:], con.i_g[:], con.A[:] = init_matconstraint_mpc(
            model, transcription, nc,
            i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, 
            i_Ymin, i_Ymax, i_x̂min, i_x̂max,
            con.A_Umin, con.A_Umax, con.A_ΔŨmin, con.A_ΔŨmax, 
            con.A_Ymin, con.A_Ymax, con.A_x̂min, con.A_x̂max,
            con.A_ŝ
        )
        A = con.A[con.i_b, :]
        b = con.b[con.i_b]
        Z̃var::Vector{JuMP.VariableRef} = optim[:Z̃var]
        JuMP.delete(optim, optim[:linconstraint])
        JuMP.unregister(optim, :linconstraint)
        @constraint(optim, linconstraint, A*Z̃var .≤ b)
        set_nonlincon!(mpc, model, transcription, optim)
    else
        i_b, i_g = init_matconstraint_mpc(
            model, transcription, nc,
            i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, 
            i_Ymin, i_Ymax, i_x̂min, i_x̂max
        )
        if i_b ≠ con.i_b || i_g ≠ con.i_g
            error("Cannot modify ±Inf constraints after calling moveinput!")
        end
    end
    return mpc
end

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


@doc raw"""
    move_blocking(Hp::Int, Hc::Vector{Int}) -> nb

Get the move blocking vector `nb` from the `Hc` argument, and modify it to match `Hp`.

This feature is also known as manipulated variable blocking. The argument `Hc` is
interpreted as the move blocking vector `nb`. It specifies the length of each step (or
"block") in the ``\mathbf{ΔU}`` vector, to customize the pattern (in time steps, thus
strictly positive integers):
```math
    \mathbf{n_b} = \begin{bmatrix} n_1 & n_2 & \cdots & n_{H_c} \end{bmatrix}'
```
The vector that includes all the manipulated input increments ``\mathbf{Δu}`` is then
defined as:
```math
\mathbf{ΔU} = \begin{bmatrix}
    \mathbf{Δu}(k + 0)                                  \\[0.1em]
    \mathbf{Δu}(k + ∑_{i=1}^1 n_i)                      \\[0.1em]
    \mathbf{Δu}(k + ∑_{i=1}^2 n_i)                      \\[0.1em]
    \vdots                                              \\[0.1em]
    \mathbf{Δu}(k + ∑_{i=1}^{H_c-1} n_i)   
\end{bmatrix}
```
The provided `nb` vector is modified to ensure `sum(nb) == Hp`:
- If `sum(nb) < Hp`, a new element is pushed to `nb` with the value `Hp - sum(nb)`.
- If `sum(nb) > Hp`, the intervals are truncated until `sum(nb) == Hp`. For example, if
  `Hp = 10` and `nb = [1, 2, 3, 6, 7]`, then `nb` is truncated to `[1, 2, 3, 4]`.
""" 
function move_blocking(Hp_arg::Int, Hc_arg::AbstractVector{Int})
    Hp = Hp_arg
    nb = Hc_arg
    all(>(0), nb) || throw(ArgumentError("Move blocking vector must be strictly positive integers."))
    if sum(nb) < Hp
        newblock = [Hp - sum(nb)]
        nb = [nb; newblock]
    elseif sum(nb) > Hp
        nb = nb[begin:findfirst(≥(Hp), cumsum(nb))]
        if sum(nb) > Hp
            # if the last block is too large, it is truncated to fit Hp:
            nb[end] = Hp - @views sum(nb[begin:end-1])
        end
    end
    return nb
end

"""
    move_blocking(Hp::Int, Hc::Int) -> nb

Construct a move blocking vector `nb` that match the provided `Hp` and `Hc` integers.

The vector is filled with `1`s, except for the last element which is `Hp - Hc + 1`.
"""
function move_blocking(Hp_arg::Int, Hc_arg::Int)
    Hp, Hc = Hp_arg, Hc_arg
    nb = fill(1, Hc)
    if Hc > 0 # if Hc < 1, it will crash later with a clear error message
        nb[end] = Hp - Hc + 1
    end
    return nb
end

"Get the actual control Horizon `Hc` (integer) from the move blocking vector `nb`."
get_Hc(nb::AbstractVector{Int}) = length(nb)


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
    size(R̂u) ≠ (nu*Hp,) && throw(DimensionMismatch("R̂u size $(size(R̂u)) ≠ manip. input size × Hp ($(nu*Hp),)"))
end

@doc raw"""
    init_quadprog(model::LinModel, weights::ControllerWeights, Ẽ, P̃Δu, P̃u) -> H̃

Init the quadratic programming Hessian `H̃` for MPC.

The matrix appear in the quadratic general form:
```math
    J = \min_{\mathbf{Z̃}} \frac{1}{2}\mathbf{Z̃' H̃ Z̃} + \mathbf{q̃' Z̃} + r 
```
The Hessian matrix is constant if the model and weights are linear and time invariant (LTI): 
```math
    \mathbf{H̃} = 2 (   \mathbf{Ẽ'}      \mathbf{M}_{H_p} \mathbf{Ẽ} 
                     + \mathbf{P̃_{Δu}'} \mathbf{Ñ}_{H_c} \mathbf{P̃_{Δu}} 
                     + \mathbf{P̃_{u}'}  \mathbf{L}_{H_p} \mathbf{P̃_{u}}     )
```
in which ``\mathbf{Ẽ}``, ``\mathbf{P̃_{Δu}}`` and ``\mathbf{P̃_{u}}`` matrices are defined
at [`relaxŶ`](@ref), [`relaxΔU`](@ref) and [`relaxU`](@ref) documentation, respectively. The
vector ``\mathbf{q̃}`` and scalar ``r`` need recalculation each control period ``k``, see
[`initpred!`](@ref). ``r`` does not impact the minima position. It is thus useless at
optimization but required to evaluate the minimal ``J`` value.
"""
function init_quadprog(::LinModel, weights::ControllerWeights, Ẽ, P̃Δu, P̃u)
    M_Hp, Ñ_Hc, L_Hp = weights.M_Hp, weights.Ñ_Hc, weights.L_Hp
    H̃ = Hermitian(2*(Ẽ'*M_Hp*Ẽ + P̃Δu'*Ñ_Hc*P̃Δu + P̃u'*L_Hp*P̃u), :L)
    return H̃
end
"Return empty matrix if `model` is not a [`LinModel`](@ref)."
function init_quadprog(::SimModel{NT}, weights::ControllerWeights, _, _, _) where {NT<:Real}
    H̃ = Hermitian(zeros(NT, 0, 0), :L)
    return H̃
end

"""
    init_defaultcon_mpc(
        estim::StateEstimator, 
        weights::ControllerWeights
        transcription::TranscriptionMethod,
        Hp, Hc, 
        PΔu, Pu, E, 
        ex̂, fx̂, gx̂, jx̂, kx̂, vx̂, bx̂, 
        Eŝ, Fŝ, Gŝ, Jŝ, Kŝ, Vŝ, Bŝ,
        gc!=nothing, nc=0
    ) -> con, nϵ, P̃Δu, P̃u, Ẽ, Ẽŝ

Init `ControllerConstraint` struct with default parameters based on estimator `estim`.

Also return `P̃Δu`, `P̃u`, `Ẽ` and `Ẽŝ` matrices for the the augmented decision vector `Z̃`.
"""
function init_defaultcon_mpc(
    estim::StateEstimator{NT}, 
    weights::ControllerWeights,
    transcription::TranscriptionMethod,
    Hp,  Hc, 
    PΔu, Pu, E, 
    ex̂, fx̂, gx̂, jx̂, kx̂, vx̂, bx̂, 
    Eŝ, Fŝ, Gŝ, Jŝ, Kŝ, Vŝ, Bŝ,
    gc!::GCfunc = nothing, nc = 0
) where {NT<:Real, GCfunc<:Union{Nothing, Function}}
    model = estim.model
    nu, ny, nx̂ = model.nu, model.ny, estim.nx̂
    nϵ = weights.isinf_C ? 0 : 1
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
    A_Umin,  A_Umax, P̃u  = relaxU(Pu, C_umin, C_umax, nϵ)
    A_ΔŨmin, A_ΔŨmax, ΔŨmin, ΔŨmax, P̃Δu = relaxΔU(PΔu, C_Δumin, C_Δumax, ΔUmin, ΔUmax, nϵ)
    A_Ymin,  A_Ymax, Ẽ  = relaxŶ(E, C_ymin, C_ymax, nϵ)
    A_x̂min,  A_x̂max, ẽx̂ = relaxterminal(ex̂, c_x̂min, c_x̂max, nϵ)
    A_ŝ, Ẽŝ = augmentdefect(Eŝ, nϵ)
    i_Umin,  i_Umax  = .!isinf.(U0min), .!isinf.(U0max)
    i_ΔŨmin, i_ΔŨmax = .!isinf.(ΔŨmin), .!isinf.(ΔŨmax)
    i_Ymin,  i_Ymax  = .!isinf.(Y0min), .!isinf.(Y0max)
    i_x̂min,  i_x̂max  = .!isinf.(x̂0min), .!isinf.(x̂0max)
    i_b, i_g, A, Aeq, neq = init_matconstraint_mpc(
        model, transcription, nc,
        i_Umin, i_Umax, i_ΔŨmin, i_ΔŨmax, i_Ymin, i_Ymax, i_x̂min, i_x̂max,
        A_Umin, A_Umax, A_ΔŨmin, A_ΔŨmax, A_Ymin, A_Ymax, A_x̂max, A_x̂min,
        A_ŝ
    )
    # dummy b and beq vectors (updated just before optimization)
    b, beq = zeros(NT, size(A, 1)), zeros(NT, size(Aeq, 1))
    con = ControllerConstraint{NT, GCfunc}(
        ẽx̂      , fx̂     , gx̂     , jx̂       , kx̂     , vx̂     , bx̂     ,
        Ẽŝ      , Fŝ     , Gŝ     , Jŝ       , Kŝ     , Vŝ     , Bŝ     ,
        U0min   , U0max  , ΔŨmin  , ΔŨmax    , Y0min  , Y0max  , x̂0min  , x̂0max,
        A_Umin  , A_Umax , A_ΔŨmin, A_ΔŨmax  , A_Ymin , A_Ymax , A_x̂min , A_x̂max,
        A       , b      , i_b    , 
        A_ŝ     ,
        Aeq     , beq    ,
        neq     ,
        C_ymin  , C_ymax , c_x̂min , c_x̂max , i_g,
        gc!     , nc
    )
    return con, nϵ, P̃Δu, P̃u, Ẽ, Ẽŝ
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
    relaxU(Pu, C_umin, C_umax, nϵ) -> A_Umin, A_Umax, P̃u

Augment manipulated inputs constraints with slack variable ϵ for softening.

Denoting the decision variables augmented with the slack variable
``\mathbf{Z̃} = [\begin{smallmatrix} \mathbf{Z} \\ ϵ \end{smallmatrix}]``, it returns the
augmented conversion matrix ``\mathbf{P̃_u}``, similar to the one described at
[`init_ZtoU`](@ref). It also returns the ``\mathbf{A}`` matrices for the inequality
constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{U_{min}}} \\ 
    \mathbf{A_{U_{max}}} 
\end{bmatrix} \mathbf{Z̃} ≤
\begin{bmatrix}
    - \mathbf{U_{min} + T_u u}(k-1) \\
    + \mathbf{U_{max} - T_u u}(k-1)
\end{bmatrix}
```
in which ``\mathbf{U_{min}}`` and ``\mathbf{U_{max}}`` vectors respectively contains
``\mathbf{u_{min}}`` and ``\mathbf{u_{max}}`` repeated ``H_p`` times.
"""
function relaxU(Pu::AbstractMatrix{NT}, C_umin, C_umax, nϵ) where NT<:Real
    if nϵ == 1 # Z̃ = [Z; ϵ]
        # ϵ impacts Z → U conversion for constraint calculations:
        A_Umin, A_Umax = -[Pu  C_umin], [Pu -C_umax] 
        # ϵ has no impact on Z → U conversion for prediction calculations:
        P̃u = [Pu zeros(NT, size(Pu, 1))]
    else # Z̃ = Z (only hard constraints)
        A_Umin, A_Umax = -Pu,  Pu
        P̃u = Pu
    end
    return A_Umin, A_Umax, P̃u
end

@doc raw"""
    relaxΔU(PΔu, C_Δumin, C_Δumax, ΔUmin, ΔUmax, nϵ) -> A_ΔŨmin, A_ΔŨmax, ΔŨmin, ΔŨmax, P̃Δu

Augment input increments constraints with slack variable ϵ for softening.

Denoting the decision variables augmented with the slack variable 
``\mathbf{Z̃} = [\begin{smallmatrix} \mathbf{Z} \\ ϵ \end{smallmatrix}]``, it returns the
augmented conversion matrix ``\mathbf{P̃_{Δu}}``, similar to the one described at
[`init_ZtoΔU`](@ref), but extracting the input increments augmented with the slack variable
``\mathbf{ΔŨ} = [\begin{smallmatrix} \mathbf{ΔU} \\ ϵ \end{smallmatrix}] = \mathbf{P̃_{Δu} Z̃}``.
Also, knowing that ``0 ≤ ϵ ≤ ∞``, it also returns the augmented bounds 
``\mathbf{ΔŨ_{min}} = [\begin{smallmatrix} \mathbf{ΔU_{min}} \\ 0 \end{smallmatrix}]`` and
``\mathbf{ΔŨ_{max}} = [\begin{smallmatrix} \mathbf{ΔU_{min}} \\ ∞ \end{smallmatrix}]``,
and the ``\mathbf{A}`` matrices for the inequality constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{ΔŨ_{min}}} \\ 
    \mathbf{A_{ΔŨ_{max}}}
\end{bmatrix} \mathbf{Z̃} ≤
\begin{bmatrix}
    - \mathbf{ΔŨ_{min}} \\
    + \mathbf{ΔŨ_{max}}
\end{bmatrix}
```
Note that strictly speaking, the lower bound on the slack variable ϵ is a decision variable
bound, which is more precise than a linear inequality constraint. However, it is more
convenient to treat it as a linear inequality constraint since the optimizer `OSQP.jl` does
not support pure bounds on the decision variables.
"""
function relaxΔU(PΔu::AbstractMatrix{NT}, C_Δumin, C_Δumax, ΔUmin, ΔUmax, nϵ) where NT<:Real
    nZ = size(PΔu, 2)
    if nϵ == 1 # Z̃ = [Z; ϵ]
        ΔŨmin, ΔŨmax = [ΔUmin; NT[0.0]], [ΔUmax; NT[Inf]] # 0 ≤ ϵ ≤ ∞
        A_ϵ = [zeros(NT, 1, nZ) NT[1.0]]
        A_ΔŨmin, A_ΔŨmax = -[PΔu  C_Δumin; A_ϵ], [PΔu -C_Δumax; A_ϵ]
        P̃Δu = [PΔu zeros(NT, size(PΔu, 1), 1); zeros(NT, 1, size(PΔu, 2)) NT[1.0]]
    else # Z̃ = Z (only hard constraints)
        ΔŨmin, ΔŨmax = ΔUmin, ΔUmax
        A_ΔŨmin, A_ΔŨmax = -PΔu,  PΔu
        P̃Δu = PΔu
    end
    return A_ΔŨmin, A_ΔŨmax, ΔŨmin, ΔŨmax, P̃Δu
end

@doc raw"""
    relaxŶ(E, C_ymin, C_ymax, nϵ) -> A_Ymin, A_Ymax, Ẽ

Augment linear output prediction constraints with slack variable ϵ for softening.

Denoting the decision variables augmented with the slack variable 
``\mathbf{Z̃} = [\begin{smallmatrix} \mathbf{Z} \\ ϵ \end{smallmatrix}]``, it returns the 
``\mathbf{Ẽ}`` matrix that appears in the linear model prediction equation 
``\mathbf{Ŷ_0 = Ẽ Z̃ + F}``, and the ``\mathbf{A}`` matrices for the inequality constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{Y_{min}}} \\ 
    \mathbf{A_{Y_{max}}}
\end{bmatrix} \mathbf{Z̃} ≤
\begin{bmatrix}
    - \mathbf{(Y_{min} - Y_{op}) + F} \\
    + \mathbf{(Y_{max} - Y_{op}) - F} 
\end{bmatrix}
```
in which ``\mathbf{Y_{min}, Y_{max}}`` and ``\mathbf{Y_{op}}`` vectors respectively contains
``\mathbf{y_{min}, y_{max}}`` and ``\mathbf{y_{op}}`` repeated ``H_p`` times.
"""
function relaxŶ(E::AbstractMatrix{NT}, C_ymin, C_ymax, nϵ) where NT<:Real
    if nϵ == 1 # Z̃ = [Z; ϵ]
        if iszero(size(E, 1))
            # model is not a LinModel, thus Ŷ constraints are not linear:
            C_ymin = C_ymax = zeros(NT, 0, 1)
        end
        # ϵ impacts predicted output constraint calculations:
        A_Ymin, A_Ymax = -[E  C_ymin], [E -C_ymax] 
        # ϵ has no impact on output predictions:
        Ẽ = [E zeros(NT, size(E, 1), 1)] 
    else # Z̃ = Z (only hard constraints)
        Ẽ = E
        A_Ymin, A_Ymax = -E,  E
    end
    return A_Ymin, A_Ymax, Ẽ
end

@doc raw"""
    relaxterminal(ex̂, c_x̂min, c_x̂max, nϵ) -> A_x̂min, A_x̂max, ẽx̂

Augment terminal state constraints with slack variable ϵ for softening.

Denoting the decision variables augmented with the slack variable 
``\mathbf{Z̃} = [\begin{smallmatrix} \mathbf{Z} \\ ϵ \end{smallmatrix}]``, it returns the 
``\mathbf{ẽ_{x̂}}`` matrix that appears in the terminal state equation 
``\mathbf{x̂_0}(k + H_p) = \mathbf{ẽ_x̂ Z̃ + f_x̂}``, and the ``\mathbf{A}`` matrices for 
the inequality constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{x̂_{min}}} \\ 
    \mathbf{A_{x̂_{max}}}
\end{bmatrix} \mathbf{Z̃} ≤
\begin{bmatrix}
    - \mathbf{(x̂_{min} - x̂_{op}) + f_x̂} \\
    + \mathbf{(x̂_{max} - x̂_{op}) - f_x̂}
\end{bmatrix}
```
"""
function relaxterminal(ex̂::AbstractMatrix{NT}, c_x̂min, c_x̂max, nϵ) where {NT<:Real}
    if nϵ == 1 # Z̃ = [Z; ϵ]
        if iszero(size(ex̂, 1))
            # model is not a LinModel and transcription is a SingleShooting, thus terminal
            # state constraints are not linear:
            c_x̂min = c_x̂max = zeros(NT, 0, 1)
        end
        # ϵ impacts terminal state constraint calculations:
        A_x̂min, A_x̂max = -[ex̂ c_x̂min], [ex̂ -c_x̂max]
        # ϵ has no impact on terminal state predictions:
        ẽx̂ = [ex̂ zeros(NT, size(ex̂, 1), 1)] 
    else # Z̃ = Z (only hard constraints)
        ẽx̂ = ex̂
        A_x̂min, A_x̂max = -ex̂,  ex̂
    end
    return A_x̂min, A_x̂max, ẽx̂
end

@doc raw"""
    augmentdefect(Eŝ, nϵ) -> A_ŝ, Ẽŝ

Augment defect equality constraints with slack variable ϵ if `nϵ == 1`.

It returns the ``\mathbf{Ẽŝ}`` matrix that appears in the defect equation 
``\mathbf{Ŝ = Ẽ_ŝ Z̃ + F_ŝ}`` and the ``\mathbf{A}`` matrix for the equality constraints:
```math
\mathbf{A_ŝ Z̃} = - \mathbf{F_ŝ}
```
"""
function augmentdefect(Eŝ::AbstractMatrix{NT}, nϵ) where NT<:Real
    if nϵ == 1 # Z̃ = [Z; ϵ]
        Ẽŝ = [Eŝ zeros(NT, size(Eŝ, 1), 1)]
    else # Z̃ = Z (only hard constraints)
        Ẽŝ = Eŝ
    end
    A_ŝ = Ẽŝ
    return A_ŝ, Ẽŝ
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