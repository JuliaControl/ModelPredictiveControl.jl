const DEFAULT_NONLINDAE_HESSIAN = AutoForwardDiff()

struct NonLinModelDAE{
    NT<:Real, 
    TM<:CollocationMethod,
    JMS<:JuMP.GenericModel,
    JMO<:JuMP.GenericModel,
    JB<:AbstractADType,
    HB<:Union{AbstractADType, Nothing}, 
    FQ <:Function,
    H <:Function, 
    PT<:Any, 
} <: SimModelDAE{NT}
    x0::Vector{NT}
    a0::Vector{NT}
    transcription::TM
    # note: `NT` and the number type `JNT` in `JuMP.GenericModel{JNT}` can be
    # different since solvers that support non-Float64 are scarce.
    optim_state::JMS
    optim_output::JMO
    jacobian::JB
    hessian::HB
    force∇geq::Vector{Bool}
    force∇q::Vector{Bool}
    Z::Vector{NT}
    fq!::FQ
    h!::H
    p::PT
    Mo::Matrix{NT}
    Co::Matrix{NT}
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
    x0_optim::Vector{NT}
    u0_optim::Vector{NT}
    d0_optim::Vector{NT}
    iszero_Ha::Bool
    buffer::SimModelBuffer{NT}
    function NonLinModelDAE{NT}(
        fq!::FQ, h!::H, Ts, nu, nx, na, ny, nd, 
        p::PT, 
        transcription::TM, 
        optim_state::JMS,
        optim_output::JMO,
        jacobian::JB, hessian::HB
    ) where {
            NT<:Real, 
            TM<:CollocationMethod,
            JMS<:JuMP.GenericModel,
            JMO<:JuMP.GenericModel,
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
        x0, a0 = zeros(NT, nx), zeros(NT, na)
        t  = zeros(NT, 1)
        # the updatestate!(model, u, d) API does not know the input `u` of the next time 
        # step k+1, so only piecewise constant input `u` is supported here:
        transcription.h > 0 && error("Only zero-order hold (h=0) is supported for simulations of DAEs")
        iszero_Ha = validate_strictly_proper(NT, fq!, h!, nu, nx, na, ny, nd, p)
        Mo, Co, λo = init_orthocolloc(NT, transcription, nx, Ts)
        nZ = get_nZ_dae(transcription, nx, na)
        Z = zeros(NT, get_nZ_dae(transcription, nx, na))
        Es, Ks, Aeq = init_defectmat_dae(NT, transcription, nx, na, Co, λo)
        Fs  = zeros(NT, size(Aeq, 1))
        beq = zeros(NT, size(Aeq, 1))
        neq = nZ - size(Aeq, 1) # number of nonlinear equality constraints
        x0_optim, u0_optim, d0_optim = zeros(NT, nx), zeros(NT, nu), zeros(NT, nd)
        # force computation of derivatives for the first NLP iteration:
        force∇q, force∇geq = [true], [true]
        buffer = SimModelBuffer{NT}(nu, nx, ny, nd, 0, na)
        model = new{NT, TM, JMS, JMO, JB, HB, FQ, H, PT}(
            x0, a0,
            transcription,
            optim_state, optim_output, 
            jacobian, hessian,
            force∇q, force∇geq,
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
            x0_optim, u0_optim, d0_optim,
            iszero_Ha,
            buffer
        )
        init_optimization!(model, model.optim_state, model.optim_output)
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
`fq`/`fq!` since they typically share common computations. If `res` represents the result of
``\mathbf{q(x, a, u, d, p)}`` (or residuals), the functions can be implemented in two
possible ways:

1. **Non-mutating functions** (out-of-place): define them as `fq(x, a, u, d, p) -> (ẋ, res)`
   and `h(x, a, d, p) -> y`. This syntax is simple and intuitive but it allocates more memory.
2. **Mutating functions** (in-place): define them as `fq!(ẋ, res, x, a, u, d, p) -> nothing`
   and `h!(y, x, a, d, p) -> nothing`. This syntax reduces the allocations and potentially
   the computational burden as well.

!!! tip
    Replace the `a`, `d` or `p` argument with `_` in your functions if not needed (see
    Examples below).
    
The optional parameter `NT` explicitly set the number type of vectors (default to `Float64`).
Open loop simulations rely on a [`CollocationMethod`](@ref) and `JuMP.jl` as a root solver
to avoid new dependencies, and also to provide a similar solving environnement as
[`NonLinMPC`](@ref), for troubleshooting. Computing the current model output ``\mathbf{y}(t)``
also require solving the algebraic equation ``\mathbf{q}`` using `JuMP.jl`. 

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
- `optim_state=JuMP.Model(Ipopt.Optimizer)` : nonlinear optimizer for [`updatestate!`](@ref),
   provided as a [`JuMP.Model`](@extref) object (default to [`Ipopt`](https://github.com/jump-dev/Ipopt.jl) optimizer).
- `optim_output=JuMP.Model(Ipopt.Optimizer)` : nonlinear optimizer for [`evaloutput`](@ref),
   provided as a [`JuMP.Model`](@extref) object (default to [`Ipopt`](https://github.com/jump-dev/Ipopt.jl) optimizer).
- `jacobian=AutoForwardDiff()` : an `AbstractADType` backend for the Jacobian of the
   nonlinear constraints, see [`DifferentiationInterface` doc](@extref DifferentiationInterface List)
- `hessian=false` : an `AbstractADType` backend or `Bool` for the Hessian of the Lagrangian, 
   see `jacobian` above for the options. The default `false` skip it and use the
    quasi-Newton method of `optim` (see Extended Help).

# Examples
```jldoctest
julia> fq!(ẋ, res, x, a, u, _ , p) = (ẋ .= p*x .+ a; res .= a .- u; nothing);

julia> h!(y, x, _ , _ , _ ) = (y .= 0.1x; nothing);

julia> model = NonLinModelDAE(fq!, h!, 5.0, 1, 1, 1, 1, p=-0.2)
NonLinModelDAE with a sample time Ts = 5.0 s:
├ state optimizer: Ipopt
├ output optimizer: Ipopt
├ transcription: OrthogonalCollocation (3 collocation points)
├ jacobian: AutoForwardDiff
├ hessian: nothing
└ dimensions:
  │ ├ 1 manipulated inputs u
  │ ├ 1 states x
  │ ├ 1 algebraic variables a
  │ ├ 1 outputs y
  │ └ 0 measured disturbances d
  └ optimization:
    ├ 8 decision variables Z
    ├ 1 linear equality constraints Aeq
    └ 7 nonlinear equality constraints geq
```

# Extended Help
!!! details "Extended Help"
    If the dynamics are a function of the time, simply add a measured disturbance defined as
    ``d(t) = t``. This object does not support the ``\mathbf{u}`` argument in ``\mathbf{h}``
    function, see the Extended Help of [`LinModel`](@ref) for the justification. More
    precisely, it only supports strictly proper DAEs, so the constructor will verify there
    are no global direct transmissions from ``\mathbf{u}`` to ``\mathbf{y}`` with the
    functions ``mathbf{q}`` and ``\mathbf{h}`` using [`SparseConnectivityTracer.jl`](@extref SparseConnectivityTracer.jl).

    By default, a dense [`ForwardDiff`](@extref ForwardDiff) backend is used for the 
    Jacobians of the nonlinear equality constraints. This is also the default backend for
    the Hessians if `hessian=true`.
"""
function NonLinModelDAE{NT}(
    fq::Function, h::Function, Ts::Real, nu::Int, nx::Int, na::Int, ny::Int, nd::Int=0;
    p=NT[], 
    transcription = OrthogonalCollocation(), 
    optim_state   = JuMP.Model(DEFAULT_NLP_OPTIMIZER, add_bridges=false),
    optim_output  = JuMP.Model(DEFAULT_NLP_OPTIMIZER, add_bridges=false),
    jacobian = DEFAULT_JACDENSE,
    hessian = false,
) where {NT<:Real}
    fq!, h! = get_mutating_functions_dae(NT, fq, h)
    hessian = validate_hessian(hessian, DEFAULT_NONLINDAE_HESSIAN)
    return NonLinModelDAE{NT}(
        fq!, h!, Ts, nu, nx, na, ny, nd, p, 
        transcription, optim_state, optim_output, jacobian, hessian
    )
end

function NonLinModelDAE(
    fq::Function, h::Function, Ts::Real, 
    nu::Int, nx::Int, na::Int, ny::Int, nd::Int=0;
    p=Float64[], 
    transcription = OrthogonalCollocation(), 
    optim_state   = JuMP.Model(DEFAULT_NLP_OPTIMIZER, add_bridges=false),
    optim_output  = JuMP.Model(DEFAULT_NLP_OPTIMIZER, add_bridges=false),
    jacobian = DEFAULT_JACDENSE,
    hessian = false,
)
    return NonLinModelDAE{Float64}(
        fq, h, Ts, nu, nx, na, ny, nd; 
        p, transcription, optim_state, optim_output, jacobian, hessian
    )
end

"Get the mutating versions of the functions `fq` and `h` for a DAE model."
function get_mutating_functions_dae(NT, fq, h)
    ismutating_f_q = validate_fq_dae(NT, fq)
    fq! = if ismutating_f_q
        fq
    else
        function fq!(ẋ, res, x, a, u, d, p)
            ẋ_ret, res_ret = fq(x, a, u, d, p)
            ẋ   .= ẋ_ret
            res .= res_ret
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
        #       ẋ         , res       , x         , a         , u         , d         , p    
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
            "fq!(ẋ::Vector{$(NT)}, res::Vector{$(NT)}, x::Vector{$(NT)}, a::Vector{$(NT)}, u::Vector{$(NT)}, d::Vector{$(NT)}, p::Any)"
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

"""
    validate_strictly_proper(NT, fq!, h!, nu, nx, na, ny, nd, p) -> iszero_Ha

Validate if the DAE model is strictly proper with `SparseConnectivityTracer.jl`. 

It also returns `iszero_Ha` indicating wether or not that algebraic variable is used in `h!`
function.
"""
function validate_strictly_proper(NT, fq!, h!, nu, nx, na, ny, nd, p)
    msg = """
    This package does not support a direct transmission from the input u to the output y. 
    See the Extended Help of LinModel for the justification.
    """
    detector = TracerSparsityDetector()
    ẋ, q, y = jacobian_buffer(zeros(nx), detector), zeros(na), zeros(ny)
    x0, a0, u0, d0 = zeros(NT, nx), zeros(NT, na), zeros(NT, nu), zeros(NT, nd)
    funcQu! = (q, u) -> fq!(ẋ, q, x0, a0, u,  d0, p)
    funcQa! = (q, a) -> fq!(ẋ, q, x0, a,  u0, d0, p)
    funcHa! = (y, a) ->  h!(y, x0, a, d0, p)
    S_∂y∂u, iszero_Ha = try
        S_Qu = jacobian_sparsity(funcQu!, q, u0, detector)
        S_Qa = jacobian_sparsity(funcQa!, q, a0, detector)
        S_Ha = jacobian_sparsity(funcHa!, y, a0, detector)
        S_∂y∂u = S_Ha/S_Qa*S_Qu
        S_∂y∂u, iszero(S_Ha)
    catch 
        @warn(
        """
        Could not validate if the DAE is strictly proper with SparseConnectivityTracer.jl.
        $msg"""
        )
        spzeros(ny, nu), false
    end
    if !iszero(S_∂y∂u)
        error(
        """
        The DAE is not globally strictly proper according to SparseConnectivityTracer.jl.
        $(msg)The resulting sparsity structure of ∂h/∂u is provided below (should be all zeros).
        """,
        sprint(show, MIME"text/plain"(), S_∂y∂u), 
        )
    end
    return iszero_Ha
end

"Get the number of elements in the optimization decision vector `Z` for DAE solving."
function get_nZ_dae(transcription::OrthogonalCollocation, nx, na)
    return nx + transcription.no*nx + na + transcription.no*na
end
get_nZ_dae(::TrapezoidalCollocation, nx, na) = nx + 2na

@doc raw"""
    init_defectmat_dae(NT, ::OrthogonalCollocation, nx, na, Co, λo) -> Es, Ks, Aeq

Init the matrices for computing the defect of the next state.

Knowing that the decision vector ``\mathbf{Z}`` contain ``\mathbf{x̂_0}(k+1)``, 
``\mathbf{a_0}(k+0)``, ``\mathbf{k̄}(k+0)`` and ``\mathbf{ā}(k+0)`` vectors with an 
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
    Esa = zeros(NT, nx, na)
    Esā = zeros(NT, nx, nā)
    Es = [Esx Esa Esk̄ Esā]
    Aeq = Es
    return Es, Ks, Aeq
end

"""
    init_defectmat_dae(NT, ::CollocationMethod, nx, na, _ , _ ) -> Es, Ks, Aeq

No linear equality constraint for other [`CollocationMethod`](@ref)s, return empty matrices.
"""
function init_defectmat_dae(NT, ::CollocationMethod, nx, na, _ , _ ) 
    Ks = zeros(NT, 0, nx)
    Es = zeros(NT, 0, nx + 2na)
    Aeq = Es
    return Es, Ks, Aeq
end

"""
    init_optimization!(
        model::NonLinModelDAE, optim_state::JuMP.GenericModel, optim_output::JuMP.GenericModel
    ) -> nothing

Init the two nonlinear optimization problems for [`NonLinModelDAE`](@ref) model.
"""
function init_optimization!(
    model::NonLinModelDAE, optim_state::JuMP.GenericModel, optim_output::JuMP.GenericModel
)
    if optim_state === optim_output
        throw(ArgumentError("optim_state and optim_output must be different JuMP models"))
    end
    geq_oracle, q_oracle = get_nonlincon_oracle(model, optim_state, optim_output)
    # --- collocation problem: optim_state ---
    JuMP.num_variables(optim_state) == 0 || JuMP.empty!(optim_state)
    JuMP.set_silent(optim_state)
    nZ = length(model.Z)
    @variable(optim_state, Zvar[i=1:nZ])
    Aeq = model.Aeq
    beq = model.beq
    @constraint(optim_state, linconstrainteq,    Aeq*Zvar .== beq)
    @constraint(optim_state, nonlinconstrainteq, Zvar in geq_oracle)
    # --- algebraic equation: optim_output ---
    JuMP.num_variables(optim_output) == 0 || JuMP.empty!(optim_output)
    JuMP.set_silent(optim_output)
    na = model.na
    @variable(optim_output, a0var[i=1:na])
    @constraint(optim_output, nonlinconstraintq, a0var in q_oracle)
    return nothing
end

"""
    get_nonlincon_oracle(
        model::NonLinModelDAE, optim_state::JuMP.GenericModel, optim_output::JuMP.GenericModel
    ) -> geq_oracle, q_oracle

Return the nonlinear equality constraint oracles for [`NonLinModelDAE`](@ref) `model`.

Return `geq_oracle` and `q_oracle`, the equality [`VectorNonlinearOracle`](@extref MathOptInterface MathOptInterface.VectorNonlinearOracle)
for the collocation problem algebraic equation, respectively. This method is really
intricate because the oracles are used inside the nonlinear optimization, so they must be
type-stable and as efficient as possible. All the function outputs and derivatives are
cached and updated in-place if required to use the efficient [`value_and_jacobian!`](@extref DifferentiationInterface DifferentiationInterface.value_and_jacobian!).
"""
function get_nonlincon_oracle(
    model::NonLinModelDAE, ::JuMP.GenericModel{JNT}, ::JuMP.GenericModel{JNT}
) where JNT<:Real
    transcription = model.transcription
    jac, hess = model.jacobian, model.hessian
    nx, na, neq, nk̄ = model.nx, model.na, model.neq, get_nk̄(model, transcription)
    nZ = length(model.Z)
    strict = Val(true) 
    k̄::Vector{JNT}                     = zeros(JNT, nk̄)
    geq::Vector{JNT}, λeq::Vector{JNT} = zeros(JNT, neq), rand(JNT, neq)
    q::Vector{JNT},   λq::Vector{JNT}  = zeros(JNT, na),  rand(JNT, na)
    ẋ::Vector{JNT}                     = zeros(JNT, nx)
    # -------------- collocation constraint: nonlinear oracle -------------------------
    function geq!(geq, Z, k̄) 
        update_predictions!(k̄, geq, model, Z)
        return nothing
    end
    function ℓ_geq(Z, λeq, k̄, geq)
        update_predictions!(k̄, geq, model, Z)
        return dot(λeq, geq)
    end
    Z_∇geq = zeros(JNT, nZ)
    ∇geq_prep = prepare_jacobian(geq!, geq, jac, Z_∇geq, Cache(k̄); strict)
    ∇geq    = init_diffmat(JNT, jac, ∇geq_prep, nZ, neq)
    ∇geq_structure  = init_diffstructure(∇geq)
    if !isnothing(hess)
        ∇²geq_prep = prepare_hessian(
            ℓ_geq, hess, Z_∇geq, Constant(λeq), Cache(k̄), Cache(geq); strict
        )
        ∇²ℓ_geq = init_diffmat(JNT, hess, ∇²geq_prep, nZ, nZ)
        ∇²geq_structure = lowertriangle_indices(init_diffstructure(∇²ℓ_geq))
    end
    function update_con_eq!(geq, ∇geq, Z_∇geq, Z_arg)
        if isdifferent(Z_arg, Z_∇geq) || model.force∇geq[]
            model.force∇geq[] = false
            Z_∇geq .= Z_arg
            value_and_jacobian!(geq!, geq, ∇geq, ∇geq_prep, jac, Z_∇geq, Cache(k̄))
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
        hessian!(
            ℓ_geq, ∇²ℓ_geq, ∇²geq_prep, hess, Z_∇geq, Constant(λeq), Cache(k̄), Cache(geq)
        )
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
    # -------------- algebraic equation: nonlinear oracle -------------------------
    function q!(q, a, ẋ)
        return model.fq!(ẋ, q, model.x0_optim, a, model.u0_optim, model.d0_optim, model.p)
    end
    function ℓ_q(a, λq, ẋ, q)
        model.fq!(ẋ, q, model.x0_optim, a, model.u0_optim, model.d0_optim, model.p)
        return dot(λq, q)
    end
    a_∇q = zeros(JNT, na)
    ∇q_prep      = prepare_jacobian(q!, q, jac, a_∇q, Cache(ẋ); strict)
    ∇q           = init_diffmat(JNT, jac, ∇q_prep, na, na)
    ∇q_structure = init_diffstructure(∇q)
    if !isnothing(hess)
        ∇²q_prep = prepare_hessian(ℓ_q, hess, a_∇q, Constant(λq), Cache(ẋ), Cache(q); strict)
        ∇²ℓ_q         = init_diffmat(JNT, hess, ∇²q_prep, na, na)
        ∇²q_structure = lowertriangle_indices(init_diffstructure(∇²ℓ_q))
    end
    function update_con_q!(q, ∇q, a_∇q, a_arg)
        if isdifferent(a_arg, a_∇q) || model.force∇q[]
            model.force∇q[] = false
            a_∇q .= a_arg
            value_and_jacobian!(q!, q, ∇q, ∇q_prep, jac, a_∇q, Cache(ẋ))
        end
        return nothing
    end
    function q_func!(q_arg, a_arg)
        update_con_q!(q, ∇q, a_∇q, a_arg)
        return q_arg .= q
    end
    function ∇q_func!(∇q_arg, a_arg)
        update_con_q!(q, ∇q, a_∇q, a_arg)
        return fill_diffstructure!(∇q_arg, ∇q, ∇q_structure)
    end
    function ∇²q_func!(∇²ℓ_arg, a_arg, λ_arg)
        a_∇q .= a_arg
        λq   .= λ_arg
        hessian!(ℓ_q, ∇²ℓ_q, ∇²q_prep, hess, a_∇q, Constant(λq), Cache(ẋ), Cache(q))
        return fill_diffstructure!(∇²ℓ_arg, ∇²ℓ_q, ∇²q_structure)
    end
    q_min = q_max = zeros(JNT, na)
    q_oracle = MOI.VectorNonlinearOracle(;
        dimension = na,
        l = q_min,
        u = q_max,
        eval_f = q_func!,
        jacobian_structure = ∇q_structure,
        eval_jacobian = ∇q_func!,
        hessian_lagrangian_structure = isnothing(hess) ? Tuple{Int,Int}[] : ∇²q_structure,
        eval_hessian_lagrangian      = isnothing(hess) ? nothing          : ∇²q_func!
    )
    return geq_oracle, q_oracle
end

"""
    update_predictions!(k̄, geq, model, Z)

TBW
"""
function update_predictions!(k̄, geq, model, Z)
    x0, u0, d0 = model.x0_optim, model.u0_optim, model.d0_optim
    con_nonlinprogeq!(geq, k̄, model, model.transcription, x0, u0, d0, Z)
    return nothing
end

function con_nonlinprogeq!(
    geq, k̄, model::NonLinModelDAE, ::TrapezoidalCollocation, x0, u0, d0, Z
)
    nx, na = model.nx, model.na
    Ts = model.Ts
    x0next_Z, a0_Z, a0next_Z = @views Z[1:nx], Z[(nx+1):(nx+na)], Z[(nx+na+1):(nx+2na)]
    sknext, q1, q2  = @views geq[1:nx], geq[(nx+1):(nx+na)], geq[(nx+na+1):(nx+2na)]
    k̇1, k̇2 = @views k̄[1:nx], k̄[(nx+1):(2nx)]
    model.fq!(k̇1, q1, x0,       a0_Z,     u0, d0, model.p)
    model.fq!(k̇2, q2, x0next_Z, a0next_Z, u0, d0, model.p)
    sknext .= @. x0 - x0next_Z + 0.5*Ts*(k̇1 + k̇2)
    return geq
end

function con_nonlinprogeq!(
    geq, k̄, model::NonLinModelDAE, transcription::OrthogonalCollocation, x0, u0, d0, Z
)
    nx, na = model.nx, model.na
    Mo, no =  model.Mo, transcription.no
    nk̄, nā = get_nk̄(model, transcription), no*na
    a0_Z, k̄_Z, ā_Z = @views Z[(nx+1):(nx+na)], Z[(nx+na+1):(nx+na+nk̄)], Z[(nx+na+nk̄+1):end]
    q0, sk̄, q̄  = @views geq[1:na], geq[(na+1):(na+nk̄)], geq[(na+nk̄+1):(na+nk̄+nā)] 
    @views model.fq!(k̄[1:nx], q0, x0, a0_Z, u0, d0, model.p)
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
    return geq
end

@doc raw"""
    initstate_core!(model::NonLinModelDAE, u0, d0)

Warm-start `model.Z` and `model.a0` at zero if `model` is a [`NonLinModelDAE`](@ref).

The field `model.a0` and `model.Z` respectively warm-start [`evaloutput`](@ref) and
[`updatestate!`](@ref) solving. The method also set `model.optim_u0` and `model.optim_d0` at
`u0` and `d0` values. The `model.u0` field is used to solve the algebraic equation 
``\mathbf{q}`` in [`evaloutput`](@ref) method, but it should not impact the result in theory
since `model` is strictly proper w.r.t. `u0`.
"""
function initstate_core!(model::NonLinModelDAE, u0, d0) 
    model.Z  .= 0
    model.a0 .= 0
    model.x0_optim .= model.x0
    model.u0_optim .= u0
    model.d0_optim .= d0
    return nothing
end

@doc raw"""
    f!(x0next, _ , model::NonLinModelDAE, x0, u0, d0, _ ) -> nothing

Solve the optimization `model.optim` problem for [`NonLinModelDAE`](@ref).

After solving, the next state ``\mathbf{x_0}(k+1)`` will be stored in-place in the `x0next`
argument. The next algebraic variable ``\mathbf{a_0}(k+1)`` will be also stored at
`model.a0`.
"""
function f!(x0next, _ , model::NonLinModelDAE, x0, u0, d0, _ )
    nx, na = model.nx, model.na
    model.x0_optim .= x0
    model.u0_optim .= u0
    model.d0_optim .= d0
    linconstrainteq!(model, model.transcription)
    Zvar = model.optim_state[:Zvar]
    Z = solve!(model, model.optim_state, Zvar, model.Z)
    x0next       .= @views Z[1:nx]
    model.a0     .= @views Z[(nx + 1):(nx + na)]
    model.Z      .= Z
    return nothing
end

"""
    h!(y0, model::NonLinModelDAE, x0, d0, p) -> nothing

Solve the algebraic equation to get `a0` and call `model.h!` for [`NonLinModelDAE`](@ref).

If `model.iszero_Ha` is `true`, the algebraic variable is not used in `model.h!` according
to [`SparseConnectivityTracer.jl`](@extref SparseConnectivityTracer.jl), the algebraic
equation solving is thus skipped and `model.h!` is called directly.
"""
function h!(y0, model::NonLinModelDAE, x0, d0, p)
    if !model.iszero_Ha
        model.x0_optim .= x0
        model.d0_optim .= d0
        # model.u0_optim is not updated since u0 not available, but model is strictly proper
        # hence possible impacts on a0 vector but no direct impacts on y0 vector in the end.
        a0var = model.optim_output[:a0var]
        a0 = solve!(model, model.optim_output, a0var, model.a0)
        model.a0 .= a0
    else # model.h! is not a function of a0, this vector is not needed here:
        a0 = model.buffer.a
    end
    model.h!(y0, x0, a0, d0, p)
    return nothing
end

function linconstrainteq!(model::NonLinModelDAE, ::OrthogonalCollocation)
    mul!(model.Fs, model.Ks, model.x0_optim)
    model.beq .= @. -model.Fs
    linconeq = model.optim_state[:linconstrainteq]
    JuMP.set_normalized_rhs(linconeq, model.beq)
    return nothing
end
linconstrainteq!(::NonLinModelDAE, ::CollocationMethod) = nothing

"""
    solve!(model::NonLinModelDAE, optim, Zvar, Zs)

Solve optimization problem `optim` with the JuMP variable `Zvar` warm-started at `Zs`.
"""
function solve!(model::NonLinModelDAE, optim, Zvar, Zs)
    JuMP.set_start_value.(Zvar, Zs)
    set_force∇!(model)
    JuMP.optimize!(optim)
    if !issolved(optim)
        status = JuMP.termination_status(optim)
        if iserror(optim)
            @error(
                "DAE terminated without solution: returning last solution "*
                "(more info in debug log)",
                status
            )
        else
            @warn(
                "DAE termination status not OPTIMAL or LOCALLY_SOLVED: keeping solution "*
                "anyway (more info in debug log)", 
                status
            )
        end
        @debug info2debugstr(getinfo(model))
    end
    Z = iserror(optim) ? Zs : JuMP.value.(Zvar)
    return Z
end

"Force the computation of the derivatives for the first NLP iteration."
function set_force∇!(model::NonLinModelDAE)
    model.force∇geq[] = true
    model.force∇q[]   = true
    return nothing
end

@doc raw"""
    getinfo(model::NonLinModelDAE) -> info

Get additional info about `model` [`NonLinModelDAE`](@ref) solution for troubleshooting.

The function should be called after calling [`updatestate!`](@ref) on `model` object. It
returns the dictionary `info` with the following fields:

- `:xnext` : next state, ``\mathbf{x}(k+1)``
- `:q` : current algebraic equation residuals `res`, ``\mathbf{q(x, a, u, d, p)}`` 
- `:y` : current output, ``\mathbf{y}(k)``
- `:x` : current state, ``\mathbf{x}(k)``
- `:a` : current algebraic variable, ``\mathbf{a}(k)``
- `:u` : current manipulated input, ``\mathbf{u}(k)``
- `:d` : current measured disturbances, ``\mathbf{u}(k)``

The following two fields are also available if the related method is called at least once:

- `:sol_state` : solution summary of [`updatestate!`](@ref) optimizer for printing
- `:sol_output` : solution summary of [`evaloutput`](@ref) optimizer for printing

# Examples
```jldoctest
julia> fq!(ẋ, res, x, a, u, _ , p) = (ẋ .= p*x .+ a; res .= a .- u; nothing);

julia> h!(y, x, _ , _ , _ ) = (y .= 0.1x; nothing);

julia> model = NonLinModelDAE(fq!, h!, 5.0, 1, 1, 1, 1, p=-0.2);

julia> u = [7]; updatestate!(model, u);

julia> a = round.(getinfo(model)[:a], digits=6)
1-element Vector{Float64}:
 7.0
```
"""
function getinfo(model::NonLinModelDAE{NT}) where NT<:Real
    x0, u0, d0 = model.x0_optim, model.u0_optim, model.d0_optim
    a0, p =  model.a0, model.p
    buffer = model.buffer
    ẋ, q, y0 = buffer.x, buffer.a, buffer.y
    model.fq!(ẋ, q, x0, a0, u0, d0, p)
    model.h!(y0, x0, a0, d0, p)
    y = y0
    y .+ model.yop
    x, u, d = buffer.x, buffer.u, buffer.d
    x .= x0 .+ model.xop
    u .= u0 .+ model.uop
    d .= d0 .+ model.dop
    a  = a0
    info = Dict{Symbol, Any}()
    info[:xnext] = model.x0 + model.xop
    info[:q] = q
    info[:y] = y
    info[:x] = x
    info[:a] = a
    info[:u] = u
    info[:d] = d
    if JuMP.termination_status(model.optim_state) ≠ JuMP.OPTIMIZE_NOT_CALLED
        info[:sol_state]  = JuMP.solution_summary(model.optim_state,  verbose=true)
    end
    if JuMP.termination_status(model.optim_output) ≠ JuMP.OPTIMIZE_NOT_CALLED
        info[:sol_output] = JuMP.solution_summary(model.optim_output, verbose=true)
    end
    return info
end

function Base.show(io::IO, model::NonLinModelDAE)
    nu, nd = model.nu, model.nd
    nx, ny = model.nx, model.ny
    na = model.na
    n = maximum(ndigits.((nu, nx, ny, nd))) + 1
    println(io, "$(nameof(typeof(model))) with a sample time Ts = $(model.Ts) s:")
    println(io, "├ state optimizer: $(JuMP.solver_name(model.optim_state))")
    println(io, "├ output optimizer: $(JuMP.solver_name(model.optim_output))")
    println(io, "├ transcription: $(transcription_str(model.transcription))")
    println(io, "├ jacobian: $(backend_str(model.jacobian))")
    println(io, "├ hessian: $(backend_str(model.hessian))")
    println(io, "└ dimensions:")
    println(io, "  │ ├$(lpad(nu, n)) manipulated inputs u")
    println(io, "  │ ├$(lpad(nx, n)) states x")
    println(io, "  │ ├$(lpad(na, n)) algebraic variables a")
    println(io, "  │ ├$(lpad(ny, n)) outputs y")
    println(io, "  │ └$(lpad(nd, n)) measured disturbances d")
    nZ = length(model.Z)
    nAeq = size(model.Aeq, 1)
    neq  = model.neq
    m = maximum(ndigits.((nZ, nAeq, neq))) + 1
    println(io, "  └ optimization:")
    println(io, "    ├$(lpad(nZ, m)) decision variables Z")
    println(io, "    ├$(lpad(nAeq, m)) linear equality constraints Aeq")
    print(io,   "    └$(lpad(neq, m)) nonlinear equality constraints geq")
end