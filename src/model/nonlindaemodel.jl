const DEFAULT_NONLINDAE_HESSIAN = AutoSparse(
    AutoForwardDiff();
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(ALL_COLORING_ORDERS, postprocessing=true),
)

struct NonLinDAEmodel{
    NT<:Real, 
    TM<:CollocationMethod,
    JM<:JuMP.GenericModel,
    JB<:AbstractADType,
    HB<:Union{AbstractADType, Nothing}, 
    F <:Function, 
    Q <:Function,
    H <:Function, 
    PT<:Any, 
} <: DAEmodel{NT}
    x0::Vector{NT}
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
    function NonLinDAEmodel{NT}(
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
        t  = zeros(NT, 1)
        buffer = SimModelBuffer{NT}(nu, nx, ny, nd)
        return new{NT, TM, JM, JB, HB, F, Q, H, PT}(
            x0, 
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

function NonLinDAEmodel{NT}(
    f::Function, q::Function, h::Function, Ts::Real, 
    nu::Int, nx::Int, nz::Int, ny::Int, nd::Int=0;
    p=NT[], 
    transcription = OrthogonalCollocation(), 
    optim = JuMP.Model(DEFAULT_NLP_OPTIMIZER, add_bridges=false),
    jacobian = DEFAULT_JACSPARSE,
    hessian = false,
) where {NT<:Real}
    #TODO: MODIF THIS
    f! = f
    q! = q
    h! = h
    hessian = validate_hessian(hessian, DEFAULT_NONLINDAE_HESSIAN)
    return NonLinDAEmodel{NT}(
        f!, q!, h!, Ts, nu, nx, nz, ny, nd, p, 
        transcription, optim, jacobian, hessian
    )
end

function NonLinDAEmodel(
    f::Function, q::Function, h::Function, Ts::Real, 
    nu::Int, nx::Int, nz::Int, ny::Int, nd::Int=0;
    p=Float64[], 
    transcription = OrthogonalCollocation(), 
    optim = JuMP.Model(DEFAULT_NLP_OPTIMIZER, add_bridges=false),
    jacobian = DEFAULT_JACSPARSE,
    hessian = false,
)
    return NonLinDAEmodel{Float64}(
        f, q, h, Ts, nu, nx, nz, ny, nd; 
        p, transcription, optim, jacobian, hessian
    )
end