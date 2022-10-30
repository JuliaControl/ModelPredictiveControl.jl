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
    LinModel(sys::StateSpace, Ts::Real; <keyword arguments>)

Construct a `LinModel` from state-state model `sys`.

If `sys` is continuous, it is dicretized with [`c2d`](https://juliacontrol.github.io/ControlSystems.jl/latest/lib/constructors/#ControlSystemsBase.c2d)
and `:zoh` for manipulated inputs, and `:tustin`, for measured disturbances.

See also [`ss`](https://juliacontrol.github.io/ControlSystems.jl/latest/lib/constructors/#ControlSystemsBase.ss),
[`tf`](https://juliacontrol.github.io/ControlSystems.jl/latest/lib/constructors/#ControlSystemsBase.tf).

# Arguments

- `sys::StateSpace`: state-space model incl. manipulated inputs and measured disturbances
- `Ts::Real = NaN`: model sampling time in second, can be ommited if sys is discrete
- `i_u::IntRangeOrVector = 1:size(sys,2)`: indices of `sys` inputs that are 
    manipulated
- `i_d::IntRangeOrVector = Int[]`: indices of `sys` inputs that are measured
    disturbances
- `u_op::Vector{<:Real} = Float64[]`: manipulated input operating points
- `y_op::Vector{<:Real} = Float64[]`: outputs operating points
- `d_op::Vector{<:Real} = Float64[]`: measured disturbances operating points

# Examples

```jldoctest
julia> LinModel(tf(3, [10, 1]), 2, u_op=[50], y_op=[20])
Discrete-time linear model with a sample time Ts = 2.0 s and:
- 1 manipulated inputs u
- 1 states x
- 1 outputs y
- 0 measured disturbances d
```
"""
function LinModel(
    sys::StateSpace,
    Ts::Real = NaN;
    i_u::IntRangeOrVector = 1:size(sys,2),
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
    sysu = sminreal(sys[:,i_u])  # remove states associated to measured disturbances d
    sysd = sminreal(sys[:,i_d])  # remove states associated to manipulates inputs u
    if ~iszero(sysu.D)
        error("State matrix D must be 0 for columns associated to manipulated inputs u")
    end
    if iscontinuous(sys)
        isnan(Ts) && error("Sample time Ts must be specified if sys is continuous")
        # manipulated inputs : zero-order hold discretization 
        sysu_dis = c2d(sysu,Ts,:zoh);
        # measured disturbances : tustin discretization (continous signals with ADCs)
        sysd_dis = c2d(sysd,Ts,:tustin)
    else
        if ~isnan(Ts)
            #TODO: Resample discrete system instead of throwing an error
            sys.Ts == Ts || error("Sample time Ts must be identical to sys.Ts")
        else
            Ts = sys.Ts
        end
        sysu_dis = sysu
        sysd_dis = sysd     
    end
    sys_dis = [sysu_dis sysd_dis]
    nx = size(sys_dis.A,1)
    nu = length(i_u)
    ny = size(sys_dis,1)
    nd = length(i_d)
    A   = sys_dis.A
    Bu  = sys_dis.B[:,1:nu]
    Bd  = sys_dis.B[:,nu+1:end]
    C   = sys_dis.C;
    Dd  = sys_dis.D[:,nu+1:end]
    return LinModel(A,Bu,C,Bd,Dd,Ts,nu,nx,ny,nd,u_op,y_op,d_op)
end

"""
    LinModel(sys::TransferFunction, Ts::Real; <keyword arguments>)

Convert to minimal realization state-space when `sys` is a transfer function.
"""
function LinModel(sys::TransferFunction, Ts::Real = NaN; kwargs...)
    sys_min = minreal(ss(sys)) # remove useless states with pole-zero cancelation
    return LinModel(sys_min,Ts;kwargs...)
end


@doc raw"""
    NonLinModel(f, h, Ts::Real, nu::Int, nx::Int, ny::Int, nd::Int=0; <keyword arguments>)

Construct a `NonLinModel` from discrete state-state functions `f` and `h`.

The discrete state-space functions are:
```math
\begin{align*}
\mathbf{x}(k+1) &= \mathbf{f}\big( \mathbf{x}(k), \mathbf{u}(k), \mathbf{d}(k) \big) \\
\mathbf{y}(k)   &= \mathbf{h}\big( \mathbf{x}(k), \mathbf{d}(k) \big)
\end{align*}
```
Replace the `d` argument with `_` if `nd = 0` (see [Examples](@ref nonlinmodel_ex))

# Arguments

- `f`: state update function ``\mathbf{f}(\mathbf{x}(k), \mathbf{u}(k), \mathbf{d}(k))``
- `h`: output function ``\mathbf{h}(\mathbf{x}(k), \mathbf{d}(k))``
- `Ts::Real`: model sampling time in second
- `nu::Int`: number of manipulated inputs
- `nx::Int`: number of states
- `ny::Int`: number of outpus
- `nd::Int = 0`: number of measured distrubances
- `u_op::Vector{<:Real} = Float64[]`: manipulated input operating points
- `y_op::Vector{<:Real} = Float64[]`: outputs operating points
- `d_op::Vector{<:Real} = Float64[]`: measured disturbances operating points

# [Examples](@id nonlinmodel_ex)

```jldoctest
julia> NonLinModel((x,u,_)->-x+u, (x,_)->2x, 10, 1 , 1 , 1)
Discrete-time nonlinear model with a sample time Ts = 10.0 s and:
- 1 manipulated inputs u
- 1 states x
- 1 outputs y
- 0 measured disturbances d
```
"""
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
        f,
        h,
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

function validate_fcts(f, h, Ts::Real, nd::Int)
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