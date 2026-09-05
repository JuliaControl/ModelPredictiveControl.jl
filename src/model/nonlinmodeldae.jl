const DEFAULT_NONLINDAE_HESSIAN = AutoSparse(
    AutoForwardDiff();
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(ALL_COLORING_ORDERS, postprocessing=true),
)

struct NonLinModelDAE{
    NT<:Real, 
    TM<:CollocationMethod,
    JM<:JuMP.GenericModel,
    JB<:AbstractADType,
    HB<:Union{AbstractADType, Nothing}, 
    FQ <:Function,
    H <:Function, 
    PT<:Any, 
} <: SimModelDAE{NT}
    x0::Vector{NT}
    a0::Vector{NT}
    u0::Vector{NT}
    d0::Vector{NT}
    transcription::TM
    # note: `NT` and the number type `JNT` in `JuMP.GenericModel{JNT}` can be
    # different since solvers that support non-Float64 are scarce.
    optim::JM
    jacobian::JB
    hessian::HB
    Z::Vector{NT}
    fq!::FQ
    h!::H
    p::PT
    Mo::SparseMatrixCSC{NT, Int}
    Co::SparseMatrixCSC{NT, Int}
    λo::NT
    Ks::Matrix{NT}
    Es::Matrix{NT}
    Fs::Vector{NT}
    Aeq::Matrix{NT}
    beq::Vector{NT}
    neq::Int
    Ts::NT
    t::Vector{NT}
    nu::Int
    nx::Int
    na::Int
    ny::Int
    nd::Int
    uop::Vector{NT}
    yop::Vector{NT}
    dop::Vector{NT}
    xop::Vector{NT}
    fop::Vector{NT}
    uname::Vector{String}
    yname::Vector{String}
    dname::Vector{String}
    xname::Vector{String}
    buffer::SimModelBuffer{NT}
    function NonLinModelDAE{NT}(
        fq!::FQ, h!::H, Ts, nu, nx, na, ny, nd, 
        p::PT, 
        transcription::TM, 
        optim::JM, 
        jacobian::JB, hessian::HB
    ) where {
            NT<:Real, 
            TM<:CollocationMethod,
            JM<:JuMP.GenericModel,
            JB<:AbstractADType,
            HB<:Union{AbstractADType, Nothing},
            FQ<:Function,
            H<:Function, 
            PT<:Any
        }
        Ts > 0 || error("Sampling time Ts must be positive")
        uop = zeros(NT, nu)
        yop = zeros(NT, ny)
        dop = zeros(NT, nd)
        xop = zeros(NT, nx)
        fop = zeros(NT, nx)
        uname = ["\$u_{$i}\$" for i in 1:nu]
        yname = ["\$y_{$i}\$" for i in 1:ny]
        dname = ["\$d_{$i}\$" for i in 1:nd]
        xname = ["\$x_{$i}\$" for i in 1:nx]
        x0, a0, u0, d0 = zeros(NT, nx), zeros(NT, na), zeros(NT, nu), zeros(NT, nd)
        t  = zeros(NT, 1)
        # the updatestate!(model, u, d) API does not know the input `u` of the next time 
        # step k+1, so only piecewise constant input `u` is supported here:
        transcription.h > 0 && error("Only zero-order hold (h=0) is supported for simulations of DAEs")
        Mo, Co, λo = init_orthocolloc(NT, transcription, nx, Ts)
        nZ = get_nZ_dae(transcription, nx, na)
        Z = zeros(NT, get_nZ_dae(transcription, nx, na))
        Es, Ks, Aeq = init_defectmat_dae(NT, transcription, nx, na, Co, λo)
        Fs  = zeros(NT, size(Aeq, 1))
        beq = zeros(NT, size(Aeq, 1))
        neq = nZ - size(Aeq, 1) # number of nonlinear equality constraints
        buffer = SimModelBuffer{NT}(nu, nx, ny, nd)
        model = new{NT, TM, JM, JB, HB, FQ, H, PT}(
            x0, a0, u0, d0,
            transcription,
            optim, jacobian, hessian,
            Z,
            fq!, h!,
            p,
            Mo, Co, λo,
            Ks, Es, Fs, 
            Aeq, beq, neq,
            Ts, t,
            nu, nx, na, ny, nd, 
            uop, yop, dop, xop, fop,
            uname, yname, dname, xname,
            buffer
        )
        init_optimization!(model, model.optim)
        return model
    end
end

@doc raw"""
    NonLinModelDAE{NT}(fq::Function,  h::Function,  Ts, nu, nx, na, ny, nd=0; <kw args>)
    NonLinModelDAE{NT}(fq!::Function, h!::Function, Ts, nu, nx, na, ny, nd=0; <kw args>)

Construct a nonlinear DAE model from state-space functions `fq`/`fq!` and `h`/`h!`.

It supports continuous differential and algebraic equations (DAE). The functions are
provided in the semi-explicit form:
```math
\begin{aligned}
    \mathbf{ẋ}(t) &= \mathbf{f}\Big( \mathbf{x}(t), \mathbf{a}(t), \mathbf{u}(t), \mathbf{d}(t), \mathbf{p} \Big) \\
    \mathbf{0}    &= \mathbf{q}\Big( \mathbf{x}(t), \mathbf{a}(t), \mathbf{u}(t), \mathbf{d}(t), \mathbf{p} \Big) \\
    \mathbf{y}(t) &= \mathbf{h}\Big( \mathbf{x}(t), \mathbf{a}(t), \mathbf{d}(t), \mathbf{p} \Big)
\end{aligned}
```
where ``\mathbf{x}``, ``\mathbf{y}``, ``\mathbf{u}``, ``\mathbf{d}`` and ``\mathbf{p}`` are
defined in [`NonLinModel`](@ref), and ``\mathbf{a}`` is the algebraic variable with `na`
elements. The ``\mathbf{f}`` and ``\mathbf{q}`` functions are combined into a single method
`fq`/`fq!` since they typically share common computations. If `RHS` represents the result of
the right-hand side in ``\mathbf{0 = q(x, a, u, d, p)}``, the functions can be implemented
in two possible ways:

1. **Non-mutating functions** (out-of-place): define them as `fq(x, a, u, d, p) -> ẋ, RHS`
   and `h(x, a, d, p) -> y`. This syntax is simple and intuitive but it allocates more memory.
2. **Mutating functions** (in-place): define them as `fq!(ẋ, RHS, x, a, u, d, p) -> nothing`
   and `h!(y, x, a, d, p) -> nothing`. This syntax reduces the allocations and potentially
   the computational burden as well.

!!! tip
    Replace the `a`, `d` or `p` argument with `_` in your functions if not needed (see
    Examples below).
    
The optional parameter `NT` explicitly set the number type of vectors (default to `Float64`).
Open loop simulations rely on a [`CollocationMethod`](@ref) and `JuMP.jl` as a root solver
to avoid new dependencies, and also to provide a similar solving environnement as
[`NonLinMPC`](@ref), for troubleshooting. 

!!! warning
    The two functions must be in pure Julia to use the model in [`NonLinMPC`](@ref) and
    [`MovingHorizonEstimator`](@ref), except if a finite difference backend is used (e.g. 
    [`AutoFiniteDiff`](@extref DifferentiationInterface List)).

See also [`NonLinModel`](@ref) for ODEs.

# Arguments
- `fq::Function` or `fq!`: combined state and algebraic function of the model.
- `h::Function` or `h!`: output function of the model.
- `Ts`: sampling time of the model in seconds.
- `nu`: number of manipulated inputs.
- `nx`: number of states.
- `na`: number of algebraic variables.
- `ny`: number of outputs.
- `nd=0`: number of measured disturbances.
- `p=[]`: parameters of the model (any type).
- `transcription=OrthogonalCollocation()` : a [`TrapezoidalCollocation`](@ref) or 
   [`OrthogonalCollocation`](@ref) instance for open-loop simulations.
- `optim=JuMP.Model(Ipopt.Optimizer)` : nonlinear optimizer for open-loop simulations,
   provided as a [`JuMP.Model`](@extref) object (default to [`Ipopt`](https://github.com/jump-dev/Ipopt.jl) optimizer).
- `jacobian=default_jacobian(transcription)` : an `AbstractADType` backend for the Jacobian
   of the nonlinear constraints, see [`DifferentiationInterface` doc](@extref DifferentiationInterface List)
- `hessian=false` : an `AbstractADType` backend or `Bool` for the Hessian of the Lagrangian, 
   see `jacobian` above for the options. The default `false` skip it and use the
   quasi-Newton method of `optim` (see Extended Help).

# Examples
```jldoctest
julia> fq!(ẋ, RHS, x, a, u, _ , p) = (ẋ .= p*x .+ a; RHS .= a .- u; nothing);

julia> h!(y, x, _ , _ , _ ) = (y .= 0.1x; nothing);

julia> model1 = NonLinModelDAE(fq!, h!, 5.0, 1, 1, 1, 1, p=-0.2)
NonLinModelDAE with a sample time Ts = 5.0 s:
├ optimizer: Ipopt
├ transcription: OrthogonalCollocation (3 collocation points)
├ jacobian: AutoSparse (AutoForwardDiff, TracerSparsityDetector, GreedyColoringAlgorithm)
├ hessian: nothing
└ dimensions:
  ├ 1 manipulated inputs u
  ├ 1 states x
  ├ 1 algebraic variables a
  ├ 1 outputs y
  └ 0 measured disturbances d
```

# Extended Help
!!! details "Extended Help"
    If the dynamics are a function of the time, simply add a measured disturbance defined as
    ``d(t) = t``. This object does not support the ``\mathbf{u}`` argument in ``\mathbf{h}``
    function, see the Extended Help of [`LinModel`](@ref) for the justification.

    The default `jacobian` backend is [sparse](@extref DifferentiationInterface AutoSparse-object):
    ```julia
    AutoSparse(
        AutoForwardDiff(); 
        sparsity_detector  = TracerSparsityDetector(), 
        coloring_algorithm = GreedyColoringAlgorithm(
            (
                NaturalOrder(),
                LargestFirst(),
                SmallestLast(),
                IncidenceDegree(),
                DynamicLargestFirst(),
                RandomOrder(StableRNG(0), 0)
            ), 
        postprocessing = true
        )
    )
    ```
    This is also the default differentiation backend for the Hessian if `hessian=true`.
"""
function NonLinModelDAE{NT}(
    fq::Function, h::Function, Ts::Real, nu::Int, nx::Int, na::Int, ny::Int, nd::Int=0;
    p=NT[], 
    transcription = OrthogonalCollocation(), 
    optim = JuMP.Model(DEFAULT_NLP_OPTIMIZER, add_bridges=false),
    jacobian = DEFAULT_JACSPARSE,
    hessian = false,
) where {NT<:Real}
    fq!, h! = get_mutating_functions_dae(NT, fq, h)
    hessian = validate_hessian(hessian, DEFAULT_NONLINDAE_HESSIAN)
    return NonLinModelDAE{NT}(
        fq!, h!, Ts, nu, nx, na, ny, nd, p, 
        transcription, optim, jacobian, hessian
    )
end

function NonLinModelDAE(
    fq::Function, h::Function, Ts::Real, 
    nu::Int, nx::Int, na::Int, ny::Int, nd::Int=0;
    p=Float64[], 
    transcription = OrthogonalCollocation(), 
    optim = JuMP.Model(DEFAULT_NLP_OPTIMIZER, add_bridges=false),
    jacobian = DEFAULT_JACSPARSE,
    hessian = false,
)
    return NonLinModelDAE{Float64}(
        fq, h, Ts, nu, nx, na, ny, nd; 
        p, transcription, optim, jacobian, hessian
    )
end

"Get the mutating versions of the functions `fq` and `h` for a DAE model."
function get_mutating_functions_dae(NT, fq, h)
    ismutating_f_q = validate_fq_dae(NT, fq)
    fq! = if ismutating_f_q
        fq
    else
        function fq!(ẋ, RHS, x, a, u, d, p)
            ẋ_ret, RHS_ret = fq(x, a, u, d, p)
            ẋ   .= ẋ_ret
            RHS .= RHS_ret
            return nothing
        end
    end
    ismutating_h = validate_h_dae(NT, h)
    h! = if ismutating_h
        h
    else
        function h!(y, x, a, d, p)
            y .= h(x, a, d, p)
            return nothing
        end
    end
    return fq!, h!
end

"""
    validate_fq_dae(NT, fq) -> ismutating

Validate `fq` function argument signature for DAEs and return `true` if mutating.
"""
function validate_fq_dae(NT, fq)
    ismutating = hasmethod(
        fq, 
        #       ẋ         , RHS       , x         , a         , u         , d         , p    
        Tuple{  Vector{NT}, Vector{NT}, Vector{NT}, Vector{NT}, Vector{NT}, Vector{NT}, Any}
    )
    isnonmutating = hasmethod(
        fq, 
        #     x,        , a         ,  u         , d         , p    
        Tuple{Vector{NT}, Vector{NT},  Vector{NT}, Vector{NT}, Any}
    )
    if !(ismutating || isnonmutating)
        error(
            "the state function has no method with type signature "*
            "fq(x::Vector{$(NT)}, a::Vector{$(NT)}, u::Vector{$(NT)}, d::Vector{$(NT)}, p::Any) or mutating form "*
            "fq!(ẋ::Vector{$(NT)}, RHS::Vector{$(NT)}, x::Vector{$(NT)}, a::Vector{$(NT)}, u::Vector{$(NT)}, d::Vector{$(NT)}, p::Any)"
        )
    end
    return ismutating
end

"""
    validate_h_dae(NT, h) -> ismutating

Validate `h` function argument signature for DAEs and return `true` if mutating.
"""
function validate_h_dae(NT, h)
    ismutating = hasmethod(
        h, 
        #     y         , x         , a         , d         , p
        Tuple{Vector{NT}, Vector{NT}, Vector{NT}, Vector{NT}, Any}
    )
    isnonmutating = hasmethod(
        h, 
        #     x         , a         , d         , p
        Tuple{Vector{NT}, Vector{NT}, Vector{NT}, Any}
    )
    if !(ismutating || isnonmutating)
        error(
            "the output function has no method with type signature "*
             "h(x::Vector{$(NT)}, a::Vector{$(NT)}, d::Vector{$(NT)}, p::Any) or mutating form "*
            "h!(y::Vector{$(NT)}, x::Vector{$(NT)}, a::Vector{$(NT)}, d::Vector{$(NT)}, p::Any)"
        )
    end
    return ismutating
end

"Get the number of elements in the optimization decision vector `Z` for DAE solving."
function get_nZ_dae(transcription::OrthogonalCollocation, nx, na)
    return nx + na + transcription.no*nx + transcription.no*na
end
get_nZ_dae(::TrapezoidalCollocation, nx, na) = nx + na

"Get the number of elements in the algebraic variable over the collocation points `ā`."
get_nā(model::SimModelDAE, transcription::OrthogonalCollocation) = transcription.no*model.na

@doc raw"""
    init_defectmat_dae(NT, ::OrthogonalCollocation, nx, na, Co, λo) -> Es, Ks, Aeq

Init the matrices for computing the defect of the next state.

Knowing that the decision vector ``\mathbf{Z}`` contain ``\mathbf{x̂_0}(k+1)``, 
``\mathbf{k̄}(k+0)``, ``\mathbf{ā}(k+0)`` and ``\mathbf{a}(k+1)`` vectors with an 
[`OrthogonalCollocation`](@ref), this linear equation compute the defect of the states at
time ``k+1``:
```math
\begin{aligned}
    \mathbf{s}(k+1) &= \mathbf{E_s Z + K_s x_0}(k)                                      \\
                    &= \mathbf{E_s Z + F_s}
\end{aligned}
```   
It is forced to be ``\mathbf{s}(k+1) = \mathbf{0}`` using the optimization equality
constraints.
"""
function init_defectmat_dae(NT, transcription::OrthogonalCollocation, nx, na, Co, λo)
    nā = transcription.no*na
    Ks = λo*I(nx)
    Esx = -I
    Esk̄ = Co
    Esā = zeros(NT, nx, nā)
    Esa = zeros(NT, nx, na)
    Es = [Esx Esk̄ Esā Esa]
    Aeq = Es
    return Es, Ks, Aeq
end

"""
    init_defectmat_dae(NT, ::CollocationMethod, nx, na, _ , _ ) -> Es, Ks, Aeq

No linear equality constraint for other [`CollocationMethod`](@ref)s, return empty matrices.
"""
function init_defectmat_dae(NT, ::CollocationMethod, nx, na, _ , _ ) 
    Ks = zeros(NT, 0, nx)
    Es = zeros(NT, 0, nx + na)
    Aeq = Es
    return Es, Ks, Aeq
end

"""
    init_optimization!(model::NonLinModelDAE, optim::JuMP.GenericModel) -> nothing

Init the nonlinear optimization for [`NonLinModelDAE`](@ref) model.
"""
function init_optimization!(model::NonLinModelDAE, optim::JuMP.GenericModel)  
    # --- variables and linear constraints ---
    nZ = length(model.Z)
    JuMP.num_variables(optim) == 0 || JuMP.empty!(optim)
    JuMP.set_silent(optim)
    @variable(optim, Zvar[i=1:nZ])
    Aeq = model.Aeq
    beq = model.beq
    @constraint(optim, linconstrainteq, Aeq*Zvar .== beq)
    # --- nonlinear optimization init ---
    geq_oracle = get_nonlincon_oracle(model, optim)
    @constraint(optim, nonlinconstrainteq, Zvar in geq_oracle)
    return nothing
end

"""
    get_nonlincon_oracle(model::NonLinModelDAE, optim::JuMP.GenericModel) -> geq_oracle

Return the nonlinear constraint oracle for [`NonLinModelDAE`](@ref) `model`.

Return `geq_oracle`, the equality [`VectorNonlinearOracle`](@extref MathOptInterface MathOptInterface.VectorNonlinearOracle)
for the the nonlinear constraints. This method is really intricate because the oracles are
used inside the nonlinear optimization, so they must be type-stable and as efficient as
possible. All the function outputs and derivatives are cached and updated in-place if
required to use the efficient [`value_and_jacobian!`](@extref DifferentiationInterface DifferentiationInterface.value_and_jacobian!).
"""
function get_nonlincon_oracle(model::NonLinModelDAE, ::JuMP.GenericModel{JNT}) where JNT<:Real
    transcription = model.transcription
    jac, hess = model.jacobian, model.hessian
    nk̄, nā = get_nk̄(model, transcription), get_nā(model, transcription)
    neq = model.neq
    nZ = length(model.Z)
    strict = Val(true) 
    myNaN                              = convert(JNT, NaN)
    k̄::Vector{JNT},   q̄::Vector{JNT}   = zeros(JNT, nk̄),  zeros(JNT, nā)
    geq::Vector{JNT}, λeq::Vector{JNT} = zeros(JNT, neq), rand(JNT, neq)
    function geq!(geq, Z, k̄, q̄) 
        update_predictions!(k̄, q̄, geq, model, Z)
        return nothing
    end
    function ℓ_geq(Z, λeq, k̄, q̄, geq)
        update_predictions!(k̄, q̄, geq, model, Z)
        return dot(λeq, geq)
    end
    Z_∇geq = fill(myNaN, nZ)    # NaN to force update at first call
    ∇geq_cache = (
        Cache(k̄), Cache(q̄)
    )
    ∇geq_prep = prepare_jacobian(geq!, geq, jac, Z_∇geq, ∇geq_cache...; strict)
    ∇geq    = init_diffmat(JNT, jac, ∇geq_prep, nZ, neq)
    ∇geq_structure  = init_diffstructure(∇geq)
    if !isnothing(hess)
        ∇²geq_cache = (
            Cache(k̄), Cache(q̄), Cache(geq)
        )
        ∇²geq_prep = prepare_hessian(
            ℓ_geq, hess, Z_∇geq, Constant(λeq), ∇²geq_cache...; strict
        )
        ∇²ℓ_geq = init_diffmat(JNT, hess, ∇²geq_prep, nZ, nZ)
        ∇²geq_structure = lowertriangle_indices(init_diffstructure(∇²ℓ_geq))
    end
    function update_con_eq!(geq, ∇geq, Z̃_∇geq, Z̃_arg)
        if isdifferent(Z̃_arg, Z̃_∇geq)
            Z̃_∇geq .= Z̃_arg
            value_and_jacobian!(geq!, geq, ∇geq, ∇geq_prep, jac, Z̃_∇geq, ∇geq_cache...)
        end
        return nothing
    end
    function geq_func!(geq_arg, Z_arg)
        update_con_eq!(geq, ∇geq, Z_∇geq, Z_arg)
        return geq_arg .= geq
    end
    function ∇geq_func!(∇geq_arg, Z_arg)
        update_con_eq!(geq, ∇geq, Z_∇geq, Z_arg)
        return fill_diffstructure!(∇geq_arg, ∇geq, ∇geq_structure)
    end
    function ∇²geq_func!(∇²ℓ_arg, Z_arg, λ_arg)
        Z_∇geq .= Z_arg
        λeq    .= λ_arg
        hessian!(ℓ_geq, ∇²ℓ_geq, ∇²geq_prep, hess, Z_∇geq, Constant(λeq), ∇²geq_cache...)
        return fill_diffstructure!(∇²ℓ_arg, ∇²ℓ_geq, ∇²geq_structure)
    end
    geq_min = geq_max = zeros(JNT, neq)
    geq_oracle = MOI.VectorNonlinearOracle(;
        dimension = nZ,
        l = geq_min,
        u = geq_max,
        eval_f = geq_func!,
        jacobian_structure = ∇geq_structure,
        eval_jacobian = ∇geq_func!,
        hessian_lagrangian_structure = isnothing(hess) ? Tuple{Int,Int}[] : ∇²geq_structure,
        eval_hessian_lagrangian      = isnothing(hess) ? nothing          : ∇²geq_func!
    )
    return geq_oracle
end

"""
    update_predictions!(k̄, q̄, geq, model, Z)

TBW
"""
function update_predictions!(k̄, q̄, geq, model, Z)
    nx, na = model.nx, model.na
    transcription = model.transcription
    Mo, no =  model.Mo, transcription.no
    nk̄, nā = get_nk̄(model, transcription), get_nā(model, transcription)
    x0, u0, d0 = model.x0, model.u0, model.d0
    x0next_Z, a0next_Z = @views Z[1:nx],                 Z[(nx+1):(nx+na)]
    k̄_Z,      ā_Z      = @views Z[(nx+na+1):(nx+na+nk̄)], Z[(nx+na+nk̄+1):(nx+na+nk̄+nā)]
    sk̄     = @views geq[1:nk̄]
    sā     = @views geq[(nk̄+1):(nk̄+nā)] 
    sanext = @views geq[(nk̄+nā+1):(nk̄+nā+na)]
    Δk = k̄
    for i=1:no
        Δk[(1 + (i-1)*nx):(i*nx)] = @views k̄_Z[(1 + (i-1)*nx):(i*nx)] .- x0
    end
    mul!(sk̄, Mo, Δk)
    for i=1:no
        k̇i   = @views   k̄[(1 + (i-1)*nx):(i*nx)]
        qi   = @views   q̄[(1 + (i-1)*na):(i*na)]
        ki_Z = @views k̄_Z[(1 + (i-1)*nx):(i*nx)]
        ai_Z = @views ā_Z[(1 + (i-1)*na):(i*na)]
        model.fq!(k̇i, qi, ki_Z, ai_Z, u0, d0, model.p)
    end
    sk̄ .-= k̄
    sā  .= q̄
    k̇next, qnext = @views k̄[1:nx], q̄[1:na]
    model.fq!(k̇next, qnext, x0next_Z, a0next_Z, u0, d0, model.p)
    sanext .= qnext
    return nothing
end

"Warm start `model.Z` at zero if `model` is a [`NonLinModelDAE`](@ref)."
steadystate!(model::NonLinModelDAE, _ , _ ) = (model.Z .= 0; nothing)

@doc raw"""
    f!(x0next, _ , model::NonLinModelDAE, x0, u0, d0, _ ) -> nothing

Solve the optimization `model.optim` problem for [`NonLinModelDAE`](@ref).

After solving, the next state ``\mathbf{x_0}(k+1)`` will be stored in-place in `x0next`
argument. The next algebraic variable ``\mathbf{a_0}(k+1)`` will be also internally stored
in `model.a0`.
"""
function f!(x0next, _ , model::NonLinModelDAE, x0, u0, d0, _ )
    nx, na = model.nx, model.na
    model.u0 .= u0
    model.d0 .= d0
    Fs = model.Fs
    mul!(Fs, model.Ks, x0)
    model.beq .= @. -Fs
    linconeq = model.optim[:linconstrainteq]
    JuMP.set_normalized_rhs(linconeq, model.beq)
    Z = solve!(model)
    x0next   .= @views Z[1:nx]
    model.a0 .= @views Z[(nx+1):(nx+na)]
    return nothing
end

function solve!(model::NonLinModelDAE)
    optim = model.optim
    Zvar::Vector{JuMP.VariableRef} = optim[:Zvar]
    Zs = zeros(get_nZ_dae(model.transcription, model.nx, model.na)) #set_warmstart_mpc!(mpc, mpc.transcription, Zvar)
    JuMP.optimize!(optim)
    #=if !issolved(optim)
        status = JuMP.termination_status(optim)
        if iserror(optim)
            @error(
                "MPC terminated without solution: returning last solution shifted "*
                "(more info in debug log)",
                status
            )
        else
            @warn(
                "MPC termination status not OPTIMAL or LOCALLY_SOLVED: keeping solution "*
                "anyway (more info in debug log)", 
                status
            )
        end
        @debug info2debugstr(getinfo(mpc))
    end=#
    if iserror(optim)
        model.Z .= Zs
    else
        model.Z .= JuMP.value.(Zvar)
    end
    return model.Z
end


"""
    h!(y0, model::NonLinModelDAE, x0, d0, p) -> nothing

Call `model.h!` with algebraic variables stored in `model.a0` for [`NonLinModelDAE`](@ref).
"""
h!(y0, model::NonLinModelDAE, x0, d0, p) = model.h!(y0, x0, model.a0, d0, p)

function Base.show(io::IO, model::NonLinModelDAE)
    nu, nd = model.nu, model.nd
    nx, ny = model.nx, model.ny
    na = model.na
    n = maximum(ndigits.((nu, nx, ny, nd))) + 1
    println(io, "$(nameof(typeof(model))) with a sample time Ts = $(model.Ts) s:")
    println(io, "├ optimizer: $(JuMP.solver_name(model.optim))")
    println(io, "├ transcription: $(transcription_str(model.transcription))")
    println(io, "├ jacobian: $(backend_str(model.jacobian))")
    println(io, "├ hessian: $(backend_str(model.hessian))")
    println(io, "└ dimensions:")
    println(io, "  ├$(lpad(nu, n)) manipulated inputs u")
    println(io, "  ├$(lpad(nx, n)) states x")
    println(io, "  ├$(lpad(na, n)) algebraic variables a")
    println(io, "  ├$(lpad(ny, n)) outputs y")
    println(io, "  └$(lpad(nd, n)) measured disturbances d")
    nZ = length(model.Z)
    nAeq = size(model.Aeq, 1)
    neq  = model.neq
    m = maximum(ndigits.((nZ, nAeq, neq))) + 1
    println(io, "  └ optimization:")
    println(io, "    ├$(lpad(nZ, m)) decision variables Z")
    println(io, "    ├$(lpad(nAeq, m)) linear equality constraints Aeq")
    print(io,   "    └$(lpad(neq, m)) nonlinear equality constraints geq")
end