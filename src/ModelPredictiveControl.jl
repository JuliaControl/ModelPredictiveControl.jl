module ModelPredictiveControl

export greet, LinModel, NonLinModel

greet() = "Hello World!"

abstract type SimModel end

struct LinModel <: SimModel
    Ts  ::Float64
    nx  ::Int
    nu  ::Int
    ny  ::Int
    nd  ::Int
    A   ::Matrix{Float64}
    Bu  ::Matrix{Float64}
    C   ::Matrix{Float64}
    Bd  ::Matrix{Float64}
    Dd  ::Matrix{Float64}
    function LinModel(Ts,nx,nu,ny,nd,A,Bu,C,Bd,Dd)
        size(A)     == (nx,nx) || error("A size must be $((nx,nx))")
        size(Bu)    == (nx,nu) || error("Bu size must be $((nx,nu))")
        size(C)     == (ny,nx) || error("C size must be $((ny,nx))")
        size(Bd)    == (nx,nd) || error("Bd size must be $((nx,nd))")
        size(Dd)    == (ny,nd) || error("Dd size must be $((ny,nd))")
        return new(Ts,nx,nu,ny,nd,A,Bu,C,Bd,Dd)
    end
end

function LinModel(Ts,A::Matrix,B::Matrix,C::Matrix)
    nx = size(A,1)
    nu = size(B,2)
    nd = 0
    ny = size(C,1)
    Bd = zeros(nx,nd)
    Dd = zeros(ny,nd)
    return LinModel(Ts,nx,nu,ny,nd,A,B,C,Bd,Dd)
end 

function LinModel(Ts,A::Matrix,Bu::Matrix,C::Matrix,Bd::Matrix,Dd=[])
    nx = size(A,1)
    nu = size(Bu,2)
    nd = size(Bd,2)
    ny = size(C,1)
    isempty(Dd) && (Dd = zeros(ny,nd));
    return LinModel(Ts,nx,nu,ny,nd,A,Bu,C,Bd,Dd)
end

struct NonLinModel <: SimModel
    Ts::Float64
    nx  ::Int
    nu  ::Int
    ny  ::Int
    nd  ::Int
    SimulFunc::Function
end

end