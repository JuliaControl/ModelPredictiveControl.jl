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
    F <:Function, 
    Q <:Function,
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
    f!::F
    q!::Q
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
        f!::F, q!::Q, h!::H, Ts, 
        nu, nx, nz, ny, nd, p::PT, 
        transcription::TM, optim::JM, 
        jacobian::JB, hessian::HB
    ) where {
            NT<:Real, 
            TM<:CollocationMethod,
            JM<:JuMP.GenericModel,
            JB<:AbstractADType,
            HB<:Union{AbstractADType, Nothing},
            F<:Function, 
            Q<:Function,
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
        return new{NT, TM, JM, JB, HB, F, Q, H, PT}(
            x0, z0,
            transcription,
            optim, jacobian, hessian,
            f!, q!, h!,
            p, 
            Ts, t,
            nu, nx, nz, ny, nd, 
            uop, yop, dop, xop, fop,
            uname, yname, dname, xname,
            buffer
        )
    end
end

function NonLinModelDAE{NT}(
    f::Function, q::Function, h::Function, Ts::Real, 
    nu::Int, nx::Int, nz::Int, ny::Int, nd::Int=0;
    p=NT[], 
    transcription = OrthogonalCollocation(), 
    optim = JuMP.Model(DEFAULT_NLP_OPTIMIZER, add_bridges=false),
    jacobian = DEFAULT_JACSPARSE,
    hessian = false,
) where {NT<:Real}
    f!, q!, h! = get_mutating_functions_dae(NT, f, q, h)
    hessian = validate_hessian(hessian, DEFAULT_NONLINDAE_HESSIAN)
    return NonLinModelDAE{NT}(
        f!, q!, h!, Ts, nu, nx, nz, ny, nd, p, 
        transcription, optim, jacobian, hessian
    )
end

function NonLinModelDAE(
    f::Function, q::Function, h::Function, Ts::Real, 
    nu::Int, nx::Int, nz::Int, ny::Int, nd::Int=0;
    p=Float64[], 
    transcription = OrthogonalCollocation(), 
    optim = JuMP.Model(DEFAULT_NLP_OPTIMIZER, add_bridges=false),
    jacobian = DEFAULT_JACSPARSE,
    hessian = false,
)
    return NonLinModelDAE{Float64}(
        f, q, h, Ts, nu, nx, nz, ny, nd; 
        p, transcription, optim, jacobian, hessian
    )
end

"Get the mutating versions of the functions `f`, `q`, and `h` for a DAE model."
function get_mutating_functions_dae(NT, f, q, h)
    ismutating_f = validate_f_q_dae(NT, f, "f")
    f! = if ismutating_f
        f
    else
        function f!(ẋ, x, z, u, d, p)
            ẋ .= f(x, z, u, d, p)
            return nothing
        end
    end
    ismutating_q = validate_f_q_dae(NT, q, "q")
    q! = if ismutating_q
        q
    else
        function q!(RHS, x, z, u, d, p)
            RHS .= q(x, z, u, d, p)
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
    return f!, q!, h!
end

"""
    validate_f_q(NT, f_q, name) -> ismutating

Validate `f` or `q` function argument signature for DAEs and return `true` if mutating.
"""
function validate_f_q_dae(NT, f_q, name)
    ismutating = hasmethod(
        f_q, 
        #       ẋ or RHS  , x         , z         , u         , d         , p    
        Tuple{  Vector{NT}, Vector{NT}, Vector{NT}, Vector{NT}, Vector{NT}, Any}
    )
    isnonmutating = hasmethod(
        f_q, 
        #     x,        , z         ,  u         , d         , p    
        Tuple{Vector{NT}, Vector{NT},  Vector{NT}, Vector{NT}, Any}
    )
    if !(ismutating || isnonmutating)
        error(
            "the $(name) function has no method with type signature "*
            "$(name)(x::Vector{$(NT)}, z::Vector{$(NT)}, u::Vector{$(NT)}, d::Vector{$(NT)}, p::Any) or mutating form "*
            "$(name)!(RHS::Vector{$(NT)}, x::Vector{$(NT)}, z::Vector{$(NT)}, u::Vector{$(NT)}, d::Vector{$(NT)}, p::Any)"
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