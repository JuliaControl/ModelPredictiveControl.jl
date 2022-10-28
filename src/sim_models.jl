using ControlSystemsBase

abstract type SimModel end

struct LinModel <: SimModel
    A   ::Matrix{Float64}
    Bu  ::Matrix{Float64}
    C   ::Matrix{Float64}
    Bd  ::Matrix{Float64}
    Dd  ::Matrix{Float64}
    f   ::Function
    h   ::Function
    Ts  ::Float64
    nu  ::Int
    nx  ::Int
    ny  ::Int
    nd  ::Int
    u_op::Vector{Float64}
    y_op::Vector{Float64}
    d_op::Vector{Float64}
    function LinModel(A,Bu,C,Bd,Dd,Ts,nu,nx,ny,nd,u_op,y_op,d_op)
        size(A)  == (nx,nx) || error("A size must be $((nx,nx))")
        size(Bu) == (nx,nu) || error("Bu size must be $((nx,nu))")
        size(C)  == (ny,nx) || error("C size must be $((ny,nx))")
        size(Bd) == (nx,nd) || error("Bd size must be $((nx,nd))")
        size(Dd) == (ny,nd) || error("Dd size must be $((ny,nd))")
        Ts > 0 || error("Sampling time Ts must be positive")
        f(x,u,d) = A*x + Bu*u + Bd*d
        h(x,d) = C*x + Dd*d
        validate_op!(u_op,y_op,d_op,nu,ny,nd)
        return new(A,Bu,C,Bd,Dd,f,h,Ts,nu,nx,ny,nd,u_op,y_op,d_op)
    end
end


IntRangeOrVector = Union{UnitRange{Int}, Vector{Int}}

"""
    LinModel(G::StateSpace, Ts::Real; kwargs...)

Construct a LinModel from state-state model `G`.

If `G` is continuous, it is dicretized using `c2d` and `:zoh` for manipulated inputs, 
and `:tustin`, for measured disturbances.

# Arguments
- `G::StateSpace`: state-space model including manipulated inputs and measured disturbances
- `Ts::Real`: model sampling time in second
- `i_u::IntRangeOrVector = 1:size(G,2)`: index of `G` inputs that are 
    manipulated
- `i_d::IntRangeOrVector = Int[]`: index of `G` inputs that are measured
    disturbances
- `u_op::Vector{<:Real} = Float64[]`: manipulated input operating points
- `y_op::Vector{<:Real} = Float64[]`: outputs operating points
- `d_op::Vector{<:Real} = Float64[]`: measured disturbances operating points

# Examples
```jldoctest
julia> LinModel(tf(3, [10,1]), 2)
```
"""
function LinModel(
    G::StateSpace,
    Ts::Real;
    i_u::IntRangeOrVector = 1:size(G,2),
    i_d::IntRangeOrVector = Int[],
    u_op::Vector{<:Real} = Float64[],
    y_op::Vector{<:Real} = Float64[],
    d_op::Vector{<:Real} = Float64[]
    )
    if ~isempty(i_d)
        # common indexes in i_u and i_d are interpreted as measured disturbances d :
        i_u = collect(i_u);
        map(i -> deleteat!(i_u, i_u .== i), i_d);
    end
    if length(unique(i_u)) ≠ length(i_u)
        error("Manipulated input indices i_u should contains valid and unique indices")
    end
    if length(unique(i_d)) ≠ length(i_d)
        error("Measured disturbances indices i_d should contains valid and unique indices")
    end
    Gu = sminreal(G[:,i_u])  # remove states associated to measured disturbances d
    Gd = sminreal(G[:,i_d])  # remove states associated to manipulates inputs u
    if ~iszero(Gu.D)
        error("State matrix D must be 0 for columns associated to manipulated inputs u")
    end
    if iscontinuous(G)
        # manipulated inputs : zero-order hold discretization 
        Gu_dis = c2d(Gu,Ts,:zoh);
        # measured disturbances : tustin discretization (continous signals with ADCs)
        Gd_dis = c2d(Gd,Ts,:tustin)
    else
        #TODO: Resample discrete system instead of throwing an error
        G.Ts == Ts || error("Sample time Ts must be identical to model.Ts")
        Gu_dis = Gu
        Gd_dis = Gd     
    end
    G_dis = [Gu_dis Gd_dis]
    nx = size(G_dis.A,1)
    nu = length(i_u)
    ny = size(G_dis,1)
    nd = length(i_d)
    A   = G_dis.A
    Bu  = G_dis.B[:,1:nu]
    Bd  = G_dis.B[:,nu+1:end]
    C   = G_dis.C;
    Dd  = G_dis.D[:,nu+1:end]
    return LinModel(A,Bu,C,Bd,Dd,Ts,nu,nx,ny,nd,u_op,y_op,d_op)
end

"""
    LinModel(G::TransferFunction, Ts::Real; kwargs...)

Convert to minimal realization state-space when `G` is a transfer function.
"""
function LinModel(G::TransferFunction, Ts::Real; kwargs...)
    G_min = minreal(ss(G)) # remove useless states with pole-zero cancelation
    return LinModel(G_min,Ts;kwargs...)
end

struct NonLinModel <: SimModel
    f::Function
    h::Function
    Ts::Float64
    nu::Int
    nx::Int
    ny::Int
    nd::Int
    u_op::Vector{Float64}
    y_op::Vector{Float64}
    d_op::Vector{Float64}
    function NonLinModel(
        f::Function,
        h::Function,
        Ts::Real,
        nu::Int,
        nx::Int,
        ny::Int,   
        nd::Int = 0;
        u_op::Vector{<:Real} = Float64[],
        y_op::Vector{<:Real} = Float64[],
        d_op::Vector{<:Real} = Float64[]
        )
        Ts > 0 || error("Sampling time Ts must be positive")
        validate_fcts(f,h,Ts,nd)
        validate_op!(u_op,y_op,d_op,nu,ny,nd)
        return new(f,h,Ts,nu,nx,ny,nd,u_op,y_op,d_op)
    end
end

function validate_fcts(f::Function, h::Function, Ts::Float64, nd::Int)
    fargsvalid1 = hasmethod(f,
        Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    )
    fargsvalid2 = hasmethod(f,
        Tuple{Vector{ComplexF64}, Vector{Float64}, Vector{Float64}}
    )
    if ~fargsvalid1 && ~fargsvalid2
        error("state function has no method of type "*
            "f(x::Vector{Float64}, u::Vector{Float64}, d::Vector{Float64}) or "*
            "f(x::Vector{ComplexF64}, u::Vector{Float64}, d::Vector{Float64})")
    end
    hargsvalid1 = hasmethod(h,Tuple{Vector{Float64}, Vector{Float64}})
    hargsvalid2 = hasmethod(h,Tuple{Vector{ComplexF64}, Vector{Float64}})
    if ~hargsvalid1 && ~hargsvalid2
        error("output function has no method of type "*
            "h(x::Vector{Float64}, d::Vector{Float64}) or "*
            "h(x::Vector{ComplexF64}, d::Vector{Float64})")
    end
end

function validate_op!(u_op, y_op, d_op, nu::Int, ny::Int, nd::Int)
    isempty(u_op) && append!(u_op,zeros(nu,))
    isempty(y_op) && append!(y_op,zeros(ny,))
    isempty(d_op) && append!(d_op,zeros(nd,))
    size(u_op)  == (nu,) || error("u_op size must be $((nu,))")
    size(y_op)  == (ny,) || error("y_op size must be $((ny,))")
    size(d_op)  == (nd,) || error("d_op size must be $((nd,))")
    return nothing
end

function Base.show(io::IO, model::SimModel)
    println(io,   "Discrete-time $(typestr(model)) model with "*
                "a sample time Ts = $(model.Ts) s and:")
    println(io, "- $(model.nu) manipulated inputs u")
    println(io, "- $(model.nx) states x")
    println(io, "- $(model.ny) outputs y")
    print(io,   "- $(model.nd) measured disturbances d")
end
typestr(model::LinModel) = "linear"
typestr(model::NonLinModel) = "nonlinear"

#=
function update_x(mMPC,x,u,d)
#UPDATE_X Update |mMPC| model states with current states |x|, manipulated
#input |u| and measured disturbance |d|.
    
        if ~mMPC.nd
            d = zeros(0,1); # d argument ignored
        end
    
        d0 = d - mMPC.d_op;
        u0 = u - mMPC.u_op;
     
        if mMPC.linModel
            xNext = mMPC.A*x + mMPC.B*u0 + mMPC.Bd*d0;
        else
            if mMPC.SimulFuncHasW   
                # processe noise w = 0 (only used for MHE observer)
                [~,Xmat] = mMPC.SimulFunc(x,u0,d0,zeros(mMPC.nx,1));
            else                              
                [~,Xmat] = mMPC.SimulFunc(x,u0,d0);
            end
            return Xmat(:,end);
        end
        
    end
=#