using JuMP
import DifferentiationInterface
import ForwardDiff
import Ipopt
import Test

f(x::T...) where {T} = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

function di_∇f(
    g::AbstractVector{T},
    x::Vararg{T,N};
    backend = DifferentiationInterface.AutoForwardDiff(),
) where {T,N}
    DifferentiationInterface.gradient!(splat(f), g, backend, collect(x))
    return
end

function di_∇²f(
    H::AbstractMatrix{T},
    x::Vararg{T,N};
    backend = DifferentiationInterface.AutoForwardDiff(),
) where {T,N}
    H_dense = DifferentiationInterface.hessian(splat(f), backend, collect(x))
    for i in 1:N, j in 1:i
        H[i, j] = H_dense[i, j]
    end
    return
end

"""
    di_derivatives(f::Function; backend) -> Tuple{Function,Function}

Return a tuple of functions that evaluate the gradient and Hessian of `f` using
DifferentiationInterface.jl with any given `backend`.
"""
function di_derivatives(f::Function; backend)
    function ∇f(g::AbstractVector{T}, x::Vararg{T,N}) where {T,N}
        DifferentiationInterface.gradient!(splat(f), g, backend, collect(x))
        return
    end
    function ∇²f(H::AbstractMatrix{T}, x::Vararg{T,N}) where {T,N}
        H_dense =
            DifferentiationInterface.hessian(splat(f), backend, collect(x))
        for i in 1:N, j in 1:i
            H[i, j] = H_dense[i, j]
        end
        return
    end
    return ∇f, ∇²f
end

function di_rosenbrock(; backend)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    @operator(model, op_rosenbrock, 2, f, di_derivatives(f; backend)...)
    @objective(model, Min, op_rosenbrock(x[1], x[2]))
    optimize!(model)
    assert_is_solved_and_feasible(model)
    return value.(x)
end

di_rosenbrock(; backend = DifferentiationInterface.AutoForwardDiff())