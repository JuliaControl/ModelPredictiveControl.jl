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
    F_Q <:Function,
    H <:Function, 
    PT<:Any, 
} <: SimModelDAE{NT}
    x0::Vector{NT}
    z0::Vector{NT}
    transcription::TM
    # note: `NT` and the number type `JNT` in `JuMP.GenericModel{JNT}` can be
    # different since solvers that support non-Float64 are scarce.
    optim::JM
    jacobian::JB
    hessian::HB
    f_q!::F_Q
    h!::H
    p::PT
    Ts::NT
    t::Vector{NT}
    nu::Int
    nx::Int
    nz::Int
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
        f_q!::F_Q, h!::H, Ts, 
        nu, nx, nz, ny, nd, p::PT, 
        transcription::TM, optim::JM, 
        jacobian::JB, hessian::HB
    ) where {
            NT<:Real, 
            TM<:CollocationMethod,
            JM<:JuMP.GenericModel,
            JB<:AbstractADType,
            HB<:Union{AbstractADType, Nothing},
            F_Q<:Function,
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
        x0 = zeros(NT, nx)
        z0 = zeros(NT, nz)
        t  = zeros(NT, 1)
        buffer = SimModelBuffer{NT}(nu, nx, ny, nd)
        return new{NT, TM, JM, JB, HB, F_Q, H, PT}(
            x0, z0,
            transcription,
            optim, jacobian, hessian,
            f_q!, h!,
            p, 
            Ts, t,
            nu, nx, nz, ny, nd, 
            uop, yop, dop, xop, fop,
            uname, yname, dname, xname,
            buffer
        )
    end
end

@doc raw"""
    NonLinModelDAE{NT}(f_q::Function,  h::Function,  Ts, nu, nx, nz, ny, nd=0; <kw args>)
    NonLinModelDAE{NT}(f_q!::Function, h!::Function, Ts, nu, nx, nz, ny, nd=0; <kw args>)

Construct a nonlinear DAE model from state-space functions `f_q`/`f_q!` and `h`/`h!`.

It supports continuous differential and algebraic equations (DAE). The functions are:
```math
\begin{aligned}
    \mathbf{ẋ}(t) &= \mathbf{f}\Big( \mathbf{x}(t), \mathbf{z}(t), \mathbf{u}(t), \mathbf{d}(t), \mathbf{p} \Big) \\
    \mathbf{0}    &= \mathbf{q}\Big( \mathbf{x}(t), \mathbf{z}(t), \mathbf{u}(t), \mathbf{d}(t), \mathbf{p} \Big) \\
    \mathbf{y}(t) &= \mathbf{h}\Big( \mathbf{x}(t), \mathbf{z}(t), \mathbf{d}(t), \mathbf{p} \Big)
\end{aligned}
```
where ``\mathbf{x}``, ``\mathbf{y}``, ``\mathbf{u}``, ``\mathbf{d}`` and ``\mathbf{p}`` are
defined in [`NonLinModel`](@ref), and ``\mathbf{z}`` comprises the algebraic variables. The
``\mathbf{f}`` and ``\mathbf{q}`` functions are combined into a single method since they
typically share common computations. If `RHS` represents the result of the right-hand side
in ``\mathbf{0 = q(x, z, u, d, p)}``, the functions can be implemented in two possible ways:

1. **Non-mutating functions** (out-of-place): define them as `f_q(x, z, u, d, p) -> ẋ, RHS`
   and `h(x, z, d, p) -> y`. This syntax is simple and intuitive but it allocates memory.
2. **Mutating functions** (in-place): define them as `f_q!(ẋ, RHS, x, z, u, d, p) -> nothing`
   and `h!(y, x, z, d, p) -> nothing`. This syntax reduces the allocations and potentially
   the computational burden as well.

!!! tip
    Replace the `z`, `d` or `p` argument with `_` in your functions if not needed (see Examples below).
    
The optional parameter `NT` explicitly set the number type of vectors (default to `Float64`).

!!! warning
    The two functions must be in pure Julia to use the model in [`NonLinMPC`](@ref) and
    [`MovingHorizonEstimator`](@ref), except if a finite difference backend is used (e.g. 
    [`AutoFiniteDiff`](@extref DifferentiationInterface List)).

See also [`NonLinModel`](@ref) for ODEs.

# Arguments
- `f_q::Function` or `f_q!`: state and algebraic function of the model.
- `h::Function` or `h!`: output function of the model.
- `Ts`: sampling time of the model in seconds.
- `nu`: number of manipulated inputs.
- `nx`: number of states.
- `nz`: number of algebraic variables.
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
julia> f_q!(ẋ, RHS, x, z, u, _ , p) = (ẋ .= p*x .+ z; RHS .= z .- u; nothing);

julia> h!(y, x, _ , _ , _ ) = (y .= 0.1x; nothing);

julia> model1 = NonLinModelDAE(f_q!, h!, 5.0, 1, 1, 1, 1, p=-0.2)
NonLinModelDAE with a sample time Ts = 5.0 s:
├ optimizer: Ipopt
├ transcription: OrthogonalCollocation (3 collocation points)
├ jacobian: AutoSparse (AutoForwardDiff, TracerSparsityDetector, GreedyColoringAlgorithm)
├ hessian: nothing
└ dimensions:
  ├ 1 manipulated inputs u
  ├ 1 states x
  ├ 1 algebraic variables z
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
    f_q::Function, h::Function, Ts::Real, 
    nu::Int, nx::Int, nz::Int, ny::Int, nd::Int=0;
    p=NT[], 
    transcription = OrthogonalCollocation(), 
    optim = JuMP.Model(DEFAULT_NLP_OPTIMIZER, add_bridges=false),
    jacobian = DEFAULT_JACSPARSE,
    hessian = false,
) where {NT<:Real}
    f_q!, h! = get_mutating_functions_dae(NT, f_q, h)
    hessian = validate_hessian(hessian, DEFAULT_NONLINDAE_HESSIAN)
    return NonLinModelDAE{NT}(
        f_q!, h!, Ts, nu, nx, nz, ny, nd, p, 
        transcription, optim, jacobian, hessian
    )
end

function NonLinModelDAE(
    f_q::Function, h::Function, Ts::Real, 
    nu::Int, nx::Int, nz::Int, ny::Int, nd::Int=0;
    p=Float64[], 
    transcription = OrthogonalCollocation(), 
    optim = JuMP.Model(DEFAULT_NLP_OPTIMIZER, add_bridges=false),
    jacobian = DEFAULT_JACSPARSE,
    hessian = false,
)
    return NonLinModelDAE{Float64}(
        f_q, h, Ts, nu, nx, nz, ny, nd; 
        p, transcription, optim, jacobian, hessian
    )
end

"Get the mutating versions of the functions `f_q` and `h` for a DAE model."
function get_mutating_functions_dae(NT, f_q, h)
    ismutating_f_q = validate_f_q_dae(NT, f_q)
    f_q! = if ismutating_f_q
        f_q
    else
        function f_q!(ẋ, RHS, x, z, u, d, p)
            ẋ_ret, RHS_ret = f_q(x, z, u, d, p)
            ẋ   .= ẋ_ret
            RHS .= RHS_ret
            return nothing
        end
    end
    ismutating_h = validate_h_dae(NT, h)
    h! = if ismutating_h
        h
    else
        function h!(y, x, z, d, p)
            y .= h(x, z, d, p)
            return nothing
        end
    end
    return f_q!, h!
end

"""
    validate_f_q(NT, f_q) -> ismutating

Validate `f_q` function argument signature for DAEs and return `true` if mutating.
"""
function validate_f_q_dae(NT, f_q)
    ismutating = hasmethod(
        f_q, 
        #       ẋ         , RHS       , x         , z         , u         , d         , p    
        Tuple{  Vector{NT}, Vector{NT}, Vector{NT}, Vector{NT}, Vector{NT}, Vector{NT}, Any}
    )
    isnonmutating = hasmethod(
        f_q, 
        #     x,        , z         ,  u         , d         , p    
        Tuple{Vector{NT}, Vector{NT},  Vector{NT}, Vector{NT}, Any}
    )
    if isnonmutating

    end
    if !(ismutating || isnonmutating)
        error(
            "the state function has no method with type signature "*
            "f_q(x::Vector{$(NT)}, z::Vector{$(NT)}, u::Vector{$(NT)}, d::Vector{$(NT)}, p::Any) or mutating form "*
            "f_q!(ẋ::Vector{$(NT)}, RHS::Vector{$(NT)}, x::Vector{$(NT)}, z::Vector{$(NT)}, u::Vector{$(NT)}, d::Vector{$(NT)}, p::Any)"
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
        #     y         , x         , z         , d         , p
        Tuple{Vector{NT}, Vector{NT}, Vector{NT}, Vector{NT}, Any}
    )
    isnonmutating = hasmethod(
        h, 
        #     x         , z         , d         , p
        Tuple{Vector{NT}, Vector{NT}, Vector{NT}, Any}
    )
    if !(ismutating || isnonmutating)
        error(
            "the output function has no method with type signature "*
             "h(x::Vector{$(NT)}, z::Vector{$(NT)}, d::Vector{$(NT)}, p::Any) or mutating form "*
            "h!(y::Vector{$(NT)}, x::Vector{$(NT)}, z::Vector{$(NT)}, d::Vector{$(NT)}, p::Any)"
        )
    end
    return ismutating
end

"""
    init_optimization!(model::NonLinModelDAE, optim::JuMP.GenericModel) -> nothing

Init the nonlinear optimization for [`NonLinModelDAE`](@ref) model.
"""
function init_optimization!(model::NonLinModelDAE, optim::JuMP.GenericModel)  
    # --- variables and linear constraints ---
    nZ̃ = length(model.Z̃)
    JuMP.num_variables(optim) == 0 || JuMP.empty!(optim)
    JuMP.set_silent(optim)
    @variable(optim, Z̃var[i=1:nZ̃])
    Aeq = model.Aeq
    beq = model.beq
    @constraint(optim, linconstrainteq, Aeq*Z̃var .== beq)
    # --- nonlinear optimization init ---
    geq_oracle = get_nonlincon_oracle(model, optim)
    # set_nonlincon!(model, geq_oracle)
    return nothing
end


function Base.show(io::IO, model::NonLinModelDAE)
    nu, nd = model.nu, model.nd
    nx, ny = model.nx, model.ny
    n = maximum(ndigits.((nu, nx, ny, nd))) + 1
    println(io, "$(nameof(typeof(model))) with a sample time Ts = $(model.Ts) s:")
    println(io, "├ optimizer: $(JuMP.solver_name(model.optim))")
    println(io, "├ transcription: $(transcription_str(model.transcription))")
    println(io, "├ jacobian: $(backend_str(model.jacobian))")
    println(io, "├ hessian: $(backend_str(model.hessian))")
    println(io, "└ dimensions:")
    println(io, "  ├$(lpad(nu, n)) manipulated inputs u")
    println(io, "  ├$(lpad(nx, n)) states x")
    println(io, "  ├$(lpad(nx, n)) algebraic variables z")
    println(io, "  ├$(lpad(ny, n)) outputs y")
    print(io,   "  └$(lpad(nd, n)) measured disturbances d")
end