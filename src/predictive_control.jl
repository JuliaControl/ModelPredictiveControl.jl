abstract type PredictiveController end

struct LinMPC <: PredictiveController
    model::LinModel
    estim::StateEstimator
    Hp::Int
    Hc::Int
    M_Hp::Diagonal{Float64}
    Ñ_Hc::Diagonal{Float64}
    L_Hp::Diagonal{Float64}
    C::Float64
    R̂u::Vector{Float64}
    Umin   ::Vector{Float64}
    Umax   ::Vector{Float64}
    ΔŨmin  ::Vector{Float64}
    ΔŨmax  ::Vector{Float64}
    Ŷmin   ::Vector{Float64}
    Ŷmax   ::Vector{Float64}
    c_Umin ::Vector{Float64}
    c_Umax ::Vector{Float64}
    c_ΔUmin::Vector{Float64}
    c_ΔUmax::Vector{Float64}
    c_Ŷmin ::Vector{Float64}
    c_Ŷmax ::Vector{Float64}
    S̃_Hp::Matrix{Bool}
    T_Hp::Matrix{Bool}
    S̃_Hc::Matrix{Bool}
    T_Hc::Matrix{Bool}
    A_umin::Matrix{Float64}
    A_umax::Matrix{Float64}
    A_ŷmin::Matrix{Float64}
    A_ŷmax::Matrix{Float64}
    Ẽl::Matrix{Float64}
    Gl::Matrix{Float64}
    Hl::Matrix{Float64}
    Jl::Matrix{Float64}
    Kl::Matrix{Float64}
    Q̃l::Matrix{Float64}
    Ks::Matrix{Float64}
    Hs::Matrix{Float64}
    function LinMPC(estim, Hp, Hc, Mwt, Nwt, Lwt, Cwt, ru)
        model = estim.model
        nu = model.nu
        ny = model.ny
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
        M_Hp = Diagonal(repeat(Mwt, Hp))
        N_Hc = Diagonal(repeat(Nwt, Hc)) 
        L_Hp = Diagonal(repeat(Lwt, Hp))
        C = Cwt
        # TODO: quick boolean test for no u setpoints (for NonLinMPC)
        R̂u = repeat(ru, Hp) # constant over Hp
        umin,  umax      = fill(-Inf, nu), fill(+Inf, nu)
        Δumin, Δumax     = fill(-Inf, nu), fill(+Inf, nu)
        ŷmin,  ŷmax      = fill(-Inf, ny), fill(+Inf, ny)
        c_umin, c_umax   = fill(0.0, nu),  fill(0.0, nu)
        c_Δumin, c_Δumax = fill(0.0, nu),  fill(0.0, nu)
        c_ŷmin, c_ŷmax   = fill(1.0, ny),  fill(1.0, ny)
        Umin, Umax, ΔUmin, ΔUmax, Ŷmin, Ŷmax = 
            repeat_constraints(Hp, Hc, umin, umax, Δumin, Δumax, ŷmin, ŷmax)
        c_Umin, c_Umax, c_ΔUmin, c_ΔUmax, c_Ŷmin, c_Ŷmax = 
            repeat_constraints(Hp, Hc, c_umin, c_umax, c_Δumin, c_Δumax, c_ŷmin, c_ŷmax)
        S_Hp, T_Hp, S_Hc, T_Hc = init_ΔUtoU(nu, Hp, Hc)
        El, Gl, Hl, Jl, Kl = init_deterpred(model, Hp, Hc)
        # TODO: move this part to augment_slack function :
        if !isinf(C) # ΔU vector is augmented with slack variable ϵ :
            Ñ_Hc = Diagonal([diag(N_Hc); C])
        else # no constraint softening nor slack variable ϵ : 
            Ñ_Hc = N_Hc
        end
        ΔŨmin, ΔŨmax, S̃_Hp, S̃_Hc, Ẽl, A_umin, A_umax, A_ŷmin, A_ŷmax = augment_slack(
            Hp, Hc, ΔUmin, ΔUmax, El, S_Hp, S_Hc, C, c_Umin, c_Umax, c_Ŷmin, c_Ŷmax
        )
        Q̃l = init_quadprog(Ẽl, S̃_Hp, M_Hp, Ñ_Hc, L_Hp)
        Ks, Hs = init_stochpred(estim, Hp) 
        return new(
            model, estim, 
            Hp, Hc, 
            M_Hp, Ñ_Hc, L_Hp, C, R̂u,
            Umin,   Umax,   ΔŨmin,   ΔŨmax,   Ŷmin,   Ŷmax, 
            c_Umin, c_Umax, c_ΔUmin, c_ΔUmax, c_Ŷmin, c_Ŷmax, 
            S̃_Hp, T_Hp, S̃_Hc, T_Hc, 
            A_umin, A_umax, A_ŷmin, A_ŷmax,
            Ẽl, Gl, Hl, Jl, Kl, Q̃l,
            Ks, Hs
        )
    end
end

@doc raw"""
    LinMPC(model::LinModel; <keyword arguments>)

Construct a linear predictive controller `LinMPC` based on [`LinModel`](@ref) `model`.

The controller minimizes the following objective function at each discrete time ``k``:
```math
\min_{\mathbf{ΔU}, ϵ}   \mathbf{(R̂_y - Ŷ)}' \mathbf{M}_{H_p} \mathbf{(R̂_y - Ŷ)}  + 
                             \mathbf{(ΔU)}' \mathbf{N}_{H_c} \mathbf{(ΔU)}  +
                        \mathbf{(R̂_u - U)}' \mathbf{L}_{H_p} \mathbf{(R̂_u - U)}  + Cϵ^2
```
in which :

- ``H_p`` : prediction horizon 
- ``H_c`` : control horizon
- ``\mathbf{ΔU}`` : manipulated input increments over ``H_c``
- ``\mathbf{Ŷ}`` : predicted outputs over ``H_p``
- ``\mathbf{U}`` : manipulated inputs over ``H_p``
- ``\mathbf{R̂_y}`` : predicted output setpoints over ``H_p``
- ``\mathbf{R̂_u}`` : predicted manipulated input setpoints over ``H_p``
- ``\mathbf{M}_{H_p} = \text{diag}\mathbf{(M,M,...,M)}`` : output setpoint tracking weights
- ``\mathbf{N}_{H_c} = \text{diag}\mathbf{(N,N,...,N)}`` : manipulated input increment weights
- ``\mathbf{L}_{H_p} = \text{diag}\mathbf{(L,L,...,L)}`` : manipulated input setpoint tracking weights
- ``C`` : slack variable weight
- ``ϵ`` : slack variable for constraint softening

The ``\mathbf{ΔU}`` vector includes the manipulated input increments ``\mathbf{Δu}(k+j) = 
\mathbf{u}(k + j) - \mathbf{u}(k + j - 1)`` from ``j = 0`` to ``H_c - 1``. The
manipulated input setpoint predictions ``\mathbf{R̂_u}`` are constant at ``\mathbf{r_u}``.

This method uses the default state estimator, a [`SteadyKalmanFilter`](@ref) with default
arguments.

# Arguments
- `model::LinModel` : model used for controller predictions and state estimations.
- `Hp=10+nk`: prediction horizon ``H_p``, `nk` is the number of delays in `model`.
- `Hc=2` : control horizon ``H_c``.
- `Mwt=fill(1.0,model.ny)` : main diagonal of ``\mathbf{M}`` weight matrix (vector)
- `Nwt=fill(0.1,model.nu)` : main diagonal of ``\mathbf{N}`` weight matrix (vector)
- `Lwt=fill(0.0,model.nu)` : main diagonal of ``\mathbf{L}`` weight matrix (vector)
- `Cwt=1e5` : slack variable weight ``C`` (scalar), use `Cwt=Inf` for hard constraints only
- `ru=model.uop`: manipulated input setpoints ``\mathbf{r_u}`` (vector)

"""
LinMPC(model::LinModel; kwargs...) = LinMPC(SteadyKalmanFilter(model); kwargs...)


"""
    LinMPC(estim::StateEstimator; <keyword arguments>)

Use custom state estimator `estim` to construct `LinMPC`.

`estim.model` must be a [`LinModel`](@ref). Else, a [`NonLinMPC`](@ref) is required. 
"""
function LinMPC(
    estim::StateEstimator;
    Hp::Union{Int, Nothing} = nothing,
    Hc::Int = 2,
    Mwt = fill(1.0, estim.model.ny),
    Nwt = fill(0.1, estim.model.nu),
    Lwt = fill(0.0, estim.model.nu),
    Cwt = 1e5,
    ru  = estim.model.uop
)
    isa(estim.model, LinModel) || error("estim.model type must be LinModel") 
    poles = eigvals(estim.model.A)
    nk = sum(poles .≈ 0)
    if isnothing(Hp)
        Hp = 10 + nk
    end
    if Hp ≤ nk
        @warn("prediction horizon Hp ($Hp) ≤ number of delays in model "*
              "($nk), the closed-loop system may be zero-gain (unresponsive) or unstable")
    end
    return LinMPC(estim, Hp, Hc, Mwt, Nwt, Lwt, Cwt, ru)
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
"""
function setconstraint!(
    mpc::PredictiveController; 
    umin = nothing,    umax  = nothing,
    Δumin = nothing,   Δumax = nothing,
    ŷmin = nothing,    ŷmax  = nothing,
    c_umin = nothing,  c_umax = nothing,
    c_Δumin = nothing, c_Δumax = nothing,
    c_ŷmin = nothing,  c_ŷmax = nothing
)
    model = mpc.model
    nu, ny = model.ny, model.ny
    if !isnothing(umin)
        size(umin)   == (nu,) || error("umin size must be $((nu,))")
        mpc.umin[:] = umin
    end
    if !isnothing(umax)
        size(umax)   == (nu,) || error("umax size must be $((nu,))")
        mpc.umax[:] = umax
    end
    if !isnothing(Δumin)
        size(Δumin)  == (nu,) || error("Δumin size must be $((nu,))")
        mpc.Δumin[:] = Δumin
    end
    if !isnothing(Δumax)
        size(Δumax)  == (nu,) || error("Δumax size must be $((nu,))")
        mpc.Δumax[:] = Δumax
    end
    if !isnothing(ŷmin)
        size(ŷmin)   == (ny,) || error("ŷmin size must be $((ny,))")
        mpc.ŷmin[:] = ŷmin
    end
    if !isnothing(ŷmax)
        size(ŷmax)   == (ny,) || error("ŷmax size must be $((ny,))")
        mpc.ŷmax[:] = ŷmax
    end
    if !isnothing(c_umin)
        size(c_umin) == (nu,) || error("c_umin size must be $((nu,))")
        any(c_umin .< 0) && error("c_umin weights should be non-negative")
        mpc.umin[:] = umin
    end
    if !isnothing(c_umax)
        size(c_umax) == (nu,) || error("c_umax size must be $((nu,))")
        any(c_umax .< 0) && error("c_umax weights should be non-negative")
        mpc.c_umax[:] = c_umax
    end
    if !isnothing(c_Δumin)
        size(c_Δumin) == (nu,) || error("c_Δumin size must be $((nu,))")
        any(c_Δumin .< 0) && error("c_Δumin weights should be non-negative")
        mpc.c_Δumin[:] = c_Δumin
    end
    if !isnothing(c_Δumax)
        size(c_Δumax) == (nu,) || error("c_Δumax size must be $((nu,))")
        any(c_Δumax .< 0) && error("c_Δumax weights should be non-negative")
        mpc.c_Δumax[:] = c_Δumax
    end
    if !isnothing(c_ŷmin)
        size(c_ŷmin) == (ny,) || error("c_ŷmin size must be $((ny,))")
        any(c_ŷmin .< 0) && error("c_ŷmin weights should be non-negative")
        mpc.c_ŷmin[:] = c_ŷmin
    end
    if !isnothing(c_ŷmax)
        size(c_ŷmax) == (ny,) || error("c_ŷmax size must be $((ny,))")
        any(c_ŷmax .< 0) && error("c_ŷmax weights should be non-negative")
        mpc.c_ŷmax[:] = c_ŷmax
    end
    Hp, Hc = mpc.Hp, mpc.Hc
    umin, umax = mpc.umin, mpc.umax
    Δumin, Δumax = mpc.Δumin, mpc.Δumax
    ŷmin, ŷmax = mpc.ŷmin, mpc.ŷmax
    c_umin, c_umax = mpc.c_umin, mpc.c_umax
    c_Δumin, c_Δumax = mpc.c_Δumin, mpc.c_Δumax
    c_ŷmin, c_ŷmax = mpc.c_ŷmin, mpc.c_ŷmax
    Umin, Umax, ΔUmin, ΔUmax, Ŷmin, Ŷmax = repeat_constraints(
        Hp, Hc, umin, umax, Δumin, Δumax, ŷmin, ŷmax
    )
    c_Umin, c_Umax, c_ΔUmin, c_ΔUmax, c_Ŷmin, c_Ŷmax = repeat_constraints(
        Hp, Hc, c_umin, c_umax, c_Δumin, c_Δumax, c_ŷmin, c_ŷmax
    )
    mpc.Umin[:] = Umin
    mpc.Umax[:] = Umax
    mpc.ΔUmin[:] = ΔUmin
    mpc.ΔUmax[:] = ΔUmax
    mpc.Ŷmin[:] = Ŷmin
    mpc.Ŷmax[:] = Ŷmax
    mpc.c_Umin[:] = c_Umin
    mpc.c_Umax[:] = c_Umax
    mpc.c_ΔUmin[:] = c_ΔUmin
    mpc.c_ΔUmax[:] = c_ΔUmax
    mpc.c_Ŷmin[:] = c_Ŷmin
    mpc.c_Ŷmax[:] = c_Ŷmax
    if !all(isnothing.((c_umin, c_umax, c_ŷmin, c_ŷmax)))
        _,_,_,_,_, A_umin, A_umax, A_ŷmin, A_ŷmax = augment_slack(
            Hp, Hc, ΔUmin, ΔUmax, El, S_Hp, S_Hc, C, c_Umin, c_Umax, c_Ŷmin, c_Ŷmax
        )
        mpc.A_umin[:] = A_umin
        mpc.A_umax[:] = A_umax
        mpc.A_ŷmin[:] = A_ŷmin  
        mpc.A_ŷmax[:] = A_ŷmax
    end
    return mpc
end

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
    \mathbf{Ŷ} &= \mathbf{E_l ΔU} + \mathbf{G_l u}(k-1) + \mathbf{H_l d}(k) + \mathbf{J_l D̂} 
                                  + \mathbf{K_l x̂_d}(k) + \mathbf{Ŷ_s}
               &= \mathbf{E_l ΔU} + \mathbf{F_l}
\end{aligned}
```
where predicted outputs ``\mathbf{Ŷ}``, stochastic outputs ``\mathbf{Ŷ_s}``, and 
disturbances ``\mathbf{D̂}`` are from ``k + 1`` to ``k + H_p``. Input increments 
``\mathbf{ΔU}`` are from ``k`` to ``k + H_c - 1``. 

Deterministic state estimates ``\mathbf{x̂_d}(k)`` are extracted from current estimates 
``\mathbf{x̂}_{k-1}(k)``. Operating points on `u`, `d` and `y` are omitted in above equation.
Stochastic predictions ``\mathbf{Ŷ_s}`` are evaluated separately to support internal model 
structure. It also allows troubleshooting by inspecting the stochastic model integrator 
values `info.Ŷs`.
"""
function init_deterpred(model::LinModel, Hp, Hc)
    A, Bu, C, Bd, Dd = model.A, model.Bu, model.C, model.Bd, model.Dd
    nu, nx, ny, nd = model.nu, model.nx, model.ny, model.nd

    ## === manipulated inputs u ===
    # Apow 3D array : Apow[:,:,1] = A^0, Apow[:,:,2] = A^1, ...
    Apow = Array{Float64}(undef, size(A,1), size(A,2), Hp+1)
    Apow[:,:,1] = I(nx)
    Kl = Matrix{Float64}(undef, Hp*ny, nx)
    for i=1:Hp
        Apow[:,:,i+1] = A^i
        iRow           = (1:ny) .+ ny*(i-1)
        Kl[iRow,:]     = C*Apow[:,:,i+1]
    end 
    # Apow_csum 3D array : Apow_csum[:,:,1] = A^0, Apow_csum[:,:,2] = A^1 + A^0, ...
    Apow_csum  = cumsum(Apow, dims=3)
    Gl = Matrix{Float64}(undef, Hp*ny, nu)
    for i=1:Hp
        iRow        = (1:ny) .+ ny*(i-1)
        Gl[iRow,:]  = C*Apow_csum[:,:,i]*Bu
    end
    El = zeros(Hp*ny, Hc*nu) 
    for i=1:Hc # truncated with control horizon
        iRow            = (ny*(i-1)+1):(ny*Hp)
        iCol            = (1:nu) .+ nu*(i-1)
        El[iRow,iCol]   = Gl[iRow .- ny*(i-1),:]
    end

    ## === measured disturbances d ===
    Hl = Matrix{Float64}(undef, Hp*ny, nd)
    Jl = repeatdiag(Dd, Hp)
    if nd ≠ 0
        for i=1:Hp
            iRow        = (1:ny) .+ ny*(i-1)
            Hl[iRow,:]  = C*Apow[:,:,i]*Bd
        end
        for i=1:Hp
            iRow            = (ny*i+1):(ny*Hp)
            iCol            = (1:nd) .+ nd*(i-1)
            Jl[iRow,iCol]   = Hl[iRow-ny*i,:]
        end
    end
    return El, Gl, Hl, Jl, Kl
end

@doc raw"""
    augment_slack(El, c_Ŷmin, c_Ŷmax)

Augment linear model deterministic prediction matrices with slack variable ϵ.

Denoting the input increments augmented with the slack variable 
``\mathbf{ΔŨ} = [\begin{smallmatrix} \mathbf{ΔU} \\ ϵ \end{smallmatrix}]``, 
it returns the augmented conversion matrices ``\mathbf{S̃}_{H_p}`` and ``\mathbf{S̃}_{H_c}``,
similar to the ones described at [`init_ΔUtoU`](@ref). It also returns ``\mathbf{Ẽ_l}`` 
to predict the outputs ``\mathbf{Ŷ = Ẽ_l ΔŨ + F_l}``, and the ``\mathbf{A}`` matrices for 
the inequality constraints:
```math
\begin{bmatrix} 
    \mathbf{A_{u_{min}}} \\ 
    \mathbf{A_{u_{max}}} \\
    \mathbf{A_{ŷ_{min}}} \\ 
    \mathbf{A_{ŷ_{max}}}
\end{bmatrix} \mathbf{ΔŨ} ≤
\begin{bmatrix}
    + \mathbf{T}_{H_c} \mathbf{u}(k-1) - \mathbf{U_{min}} \\
    - \mathbf{T}_{H_c} \mathbf{u}(k-1) + \mathbf{U_{max}} \\
    + \mathbf{F_l} - \mathbf{Ŷ_{min}} \\
    - \mathbf{F_l} + \mathbf{Ŷ_{max}}
\end{bmatrix}
```
"""
function augment_slack(
    Hp, Hc, ΔUmin, ΔUmax, El, S_Hp, S_Hc, C, c_Umin, c_Umax, c_Ŷmin, c_Ŷmax
)
    if !isinf(C) # ΔŨ = [ΔU; ϵ]
        # 0 ≤ ϵ ≤ ∞ :
        ΔŨmin = [ΔUmin; 0.0]
        ΔŨmax = [ΔUmax; Inf]
        # ϵ has no impact on ΔU → U conversion for prediction calculations:
        S̃_Hp = [S_Hp falses(size(S_Hp, 1))]
        S̃_Hc = [S_Hc falses(size(S_Hc, 1))] 
        # ϵ impacts ΔU → U conversion for constraint calculations:
        A_umin = -[S_Hc +c_Umin]
        A_umax = +[S_Hc -c_Umax]
        # ϵ has not impact on output predictions
        Ẽl = [El zeros(size(El, 1), 1)] 
        # ϵ impacts predicted output constraint calculations:
        A_ŷmin = -[El +c_Ŷmin] 
        A_ŷmax = +[El -c_Ŷmax]
    else # ΔŨ = ΔU (only hard constraints)
        ΔŨmin = ΔUmin
        ΔŨmax = ΔUmax
        A_umin = -S_Hc
        A_umax = +S_Hc
        A_ŷmin = -El
        A_ŷmax = +El
        Ẽl = El
    end
    return ΔŨmin, ΔŨmax, S̃_Hp, S̃_Hc, Ẽl, A_umin, A_umax, A_ŷmin, A_ŷmax
end


@doc raw"""
    init_stochpred(estim::StateEstimator, Hp)

Init the stochastic prediction matrix `Ks` from `estim` estimator for predictive control.

``\mathbf{K_s}`` is the prediction matrix of the stochastic model (composed exclusively of 
integrators):
```math
    \mathbf{Ŷ_s} = \mathbf{P_s}[\mathbf{M_s x̂_s}(k) + \mathbf{N_s Ŵ_s}]
                 = \mathbf{K_s x̂_s}(k)
```
since the stochastic process noises ``\mathbf{Ŵ_s = 0}`` during MPC predictions. The 
stochastic predictions ``\mathbf{Ŷ_s}`` are the integrator outputs (from ``k+1`` 
to ``k+H_p``). ``\mathbf{x̂_s}`` is extracted from the current estimate ``\mathbf{x̂}``.

!!! note
    Stochastic predictions are calculated separately and added to ``\mathbf{F_l}``
    or ``\mathbf{F_{nl}}`` matrix to reduce MPC optimization computational costs.
"""
function init_stochpred(estim::StateEstimator, Hp)
    As, Cs = estim.As, estim.Cs
    nxs = estim.nxs
    Ms = Matrix{Float64}(undef, Hp*nxs, nxs) 
    for i = 1:Hp
        iRow = (1:nxs) .+ nxs*(i-1)
        Ms[iRow, :] = As^i
    end
    Ps = repeatdiag(Cs, Hp)
    Ks = Ps*Ms
    return Ks, zeros(estim.model.ny*Hp, 0)
end


@doc raw"""
    init_stochpred(estim::InternalModel, Hp)

Init the stochastic prediction matrices for [`InternalModel`](@ref).

`Ks` and `Hs` matrices are defined as:
```math
    \mathbf{Ŷ_s} = \mathbf{K_s x̂_s}(k) + \mathbf{H_s ŷ_s}(k)
```
with ``\mathbf{Ŷ_s}`` as stochastic predictions from ``k+1`` to ``k+H_p``, current 
stochastic states ``\mathbf{x̂_s}(k)`` and outputs ``\mathbf{ŷ_s}(k)``. ``\mathbf{ŷ_s}(k)``
comprises the measured outputs ``\mathbf{ŷ_s^m}(k) = \mathbf{y^m}(k) - \mathbf{ŷ_d}(k)``
and unmeasured ``\mathbf{ŷ_s^u(k) = 0}``. See [^1].

[^1]: Desbiens, A., D. Hodouin & É. Plamondon. 2000, "Global predictive control : a unified
    control structure for decoupling setpoint tracking, feedforward compensation and 
    disturbance rejection dynamics", *IEE Proceedings - Control Theory and Applications*, 
    vol. 147, no 4, https://doi.org/10.1049/ip-cta:20000443, p. 465–475, ISSN 1350-2379.
"""
function init_stochpred(estim::InternalModel, Hp) 
    As, B̂s, Cs = estim.As, estim.B̂s, estim.Cs
    ny  = estim.model.ny
    nxs = estim.nxs
    Ks = Matrix{Float64}(undef, ny*Hp, nxs)
    Hs = Matrix{Float64}(undef, ny*Hp, ny)
    for i = 1:Hp
        iRow = (1:ny) .+ ny*(i-1)
        Ms = Cs*As^(i-1)*B̂s
        Ks[iRow, :] = Cs*As^i - Ms*Cs
        Hs[iRow, :] = Ms
    end
    return Ks, Hs 
end


"""
    init_quadprog(El, S_Hp, M_Hp, N_Hc, L_Hp)

Init quadratic programming (optimization) matrix.

`Ql` is the quadratic programming `Q` matrix. It is constant if the model and weights 
are LTI. The quadratic programming `p` vector needs recalculation each control iteration.  
"""
function init_quadprog(El, S_Hp, M_Hp, N_Hc, L_Hp)
    Ql = 2*(El'*M_Hp*El + N_Hc + S_Hp'*L_Hp*S_Hp)
    # TODO: verify if necessary or use special matrix (symmetric ?)
    # Ql = (Ql + Ql')/2
    return Ql
end


"Generate a block diagonal matrix repeating `n` times the matrix `A`."
repeatdiag(A, n::Int) = kron(I(n), A)


function Base.show(io::IO, mpc::PredictiveController)
    println(io, "$(typeof(mpc)) predictive controller with a sample time "*
                "Ts = $(mpc.model.Ts) s, $(typeof(mpc.estim)) estimator and:")
    println(io, " $(mpc.model.nu) manipulated inputs u")
    println(io, " $(mpc.estim.nx̂) states x̂")
    println(io, " $(mpc.estim.nym) measured outputs ym")
    println(io, " $(mpc.estim.nyu) unmeasured outputs yu")
    print(io,   " $(mpc.estim.model.nd) measured disturbances d")
end