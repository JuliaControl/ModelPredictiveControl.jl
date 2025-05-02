const DEFAULT_HP0 = 10
const DEFAULT_HC  = 2
const DEFAULT_MWT = 1.0
const DEFAULT_NWT = 0.1
const DEFAULT_LWT = 0.0
const DEFAULT_CWT = 1e5
const DEFAULT_EWT = 0.0

"Termination status that means 'no solution available'."
const ERROR_STATUSES = (
    JuMP.INFEASIBLE, JuMP.DUAL_INFEASIBLE, JuMP.LOCALLY_INFEASIBLE, 
    JuMP.INFEASIBLE_OR_UNBOUNDED, JuMP.NUMERICAL_ERROR, JuMP.INVALID_MODEL, 
    JuMP.INVALID_OPTION, JuMP.INTERRUPTED, JuMP.OTHER_ERROR
)

"Verify that `optim` termination status is `OPTIMAL` or `LOCALLY_SOLVED`."
function issolved(optim::JuMP.GenericModel)
    status = JuMP.termination_status(optim)
    return (status == JuMP.OPTIMAL || status == JuMP.LOCALLY_SOLVED)
end

"Verify that `optim` termination status means 'no solution available'."
function iserror(optim::JuMP.GenericModel) 
    status = JuMP.termination_status(optim)
    return any(errstatus->isequal(status, errstatus), ERROR_STATUSES)
end

"Convert getinfo dictionary to a debug string (without any truncation)."
function info2debugstr(info)
    mystr = "Content of getinfo dictionary:\n"
    for (key, value) in info
        (key == :sol) && continue
        mystr *= "  :$key => $value\n"
    end
    if haskey(info, :sol)
        split_sol = split(string(info[:sol]), "\n")
        solstr = join((lpad(line, length(line) + 2) for line in split_sol), "\n", "")
        mystr *= "  :sol => \n"*solstr
    end
    return mystr
end

"Evaluate the quadratic programming objective function `0.5x'*H*x + q'*x` at `x`."
obj_quadprog(x, H, q) = 0.5*dot(x, H, x) + q'*x  # dot(x, H, x) is faster than x'*H*x

"Limit the solving time to `Ts` if supported by `optim` optimizer."
function limit_solve_time(optim::GenericModel, Ts)
    try
        JuMP.set_time_limit_sec(optim, Ts)
    catch err
        if isa(err, MOI.UnsupportedAttribute{MOI.TimeLimitSec})
            @warn "Solving time limit is not supported by the optimizer."
        else
            rethrow()
        end
    end
end

"Verify that provided 1st and 2nd order differentiation backends are possible and efficient."
validate_backends(firstOrder::AbstractADType, secondOrder::Nothing) = nothing
validate_backends(firstOrder::Nothing, secondOrder::AbstractADType) = nothing
function validate_backends(firstOrder::AbstractADType, secondOrder::AbstractADType) 
    @warn(
        """
        Two AbstractADType backends were provided for the 1st and 2nd order differentiations,
        meaning that 1st order derivatives will be computed twice. Use nothing for the 1st
        order backend to retrieve results from the hessian backend, which is more efficient.
        """
    )
    return nothing 
end
function validate_backends(firstOrder::Nothing, secondOrder::Nothing)
    throw(ArgumentError("1st and 2nd order differentiation backends cannot be both nothing."))
end

"Init a differentiation result matrix as dense or sparse matrix, as required by `backend`."
init_diffmat(T, backend::AbstractADType, _  , nx , ny) = Matrix{T}(undef, ny, nx)
init_diffmat(T, backend::AutoSparse    ,prep , _ , _ ) = similar(sparsity_pattern(prep), T)
init_diffmat(T, backend::Nothing       , _  , nx , ny) = Matrix{T}(undef, ny, nx)

"""
    update_memoized_diff!(
        x, y, ∇f, ∇²f, prep_∇f, prep_∇²f, context,
        gradient::AbstractADType, hessian::Nothing, f!, xarg
    ) -> nothing

Update `f!` value `y` and and its gradient `∇f` in-place if `x ≠ xarg`.

The method mutates all the arguments before `gradient`. This function is used for the
memoization of the `f!` function derivatives, to avoid redundant computations with the
splatting syntax of `JuMP.@operator`.
"""
function update_memoized_diff!(
    x, y, ∇f, _ , prep_∇f, _ , context,
    gradient::AbstractADType, hessian::Nothing, f!::F, xarg
) where F <: Function
    if isdifferent(xarg, x)
        x .= xarg # more efficient than individual f! and gradient! calls:
        y[], _ = value_and_gradient!(f!, ∇f, prep_∇f, gradient, x, context...)
    end
    return nothing
end

"Also update the Hessian `∇²f` if `hessian isa AbstractADType` and `isnothing(gradient)`."
function update_memoized_diff!(
    x, y, ∇f, ∇²f, _ , prep_∇²f, context,
    gradient::Nothing, hessian::AbstractADType, f!::F, xarg
) where F <: Function
    if isdifferent(xarg, x)
        x .= xarg # more efficient than individual f!, gradient! and hessian! calls:
        y[], _ = value_gradient_and_hessian!(f!, ∇f, ∇²f, prep_∇²f, hessian, x, context...)
    end
    return nothing
end 

"Update `∇f` and `∇²f` individually if both backends are `AbstractADType`."
function update_memoized_diff!(
    x, y, ∇f, ∇²f, prep_∇f, prep_∇²f, context,
    gradient::AbstractADType, hessian::AbstractADType, f!::F, xarg
) where F <: Function
    if isdifferent(xarg, x)
        x .= xarg # inefficient, as warned by validate_backends(), but still possible:
        hessian!(f!, ∇²f, prep_∇²f, hessian, x, context...)
        y[], _ = value_and_gradient!(f!, ∇f, prep_∇f, gradient, x, context...)
    end
    return nothing
end

"""
    update_memoized_diff!(x, y, ∇f, prep_∇f, context, jacobian::AbstractADType, f!, xarg) 

Update `f!` value `y` (vector) and and its jacobian `∇f` in-place if `x ≠ xarg`.

This method mutates all the arguments before `jacobian`.
"""
function update_memoized_diff!(
    x, y, ∇f, prep_∇f, context, jacobian::AbstractADType, f!::F, xarg
) where F <: Function
    if isdifferent(xarg, x)
        x .= xarg # more efficient than individual f! and jacobian! calls:
        value_and_jacobian!(f!, y, ∇f, prep_∇f, jacobian, x, context...)
    end
    return nothing
end

"Verify that x and y elements are different using `!==`."
isdifferent(x, y) = any(xi !== yi for (xi, yi) in zip(x, y))

"Generate a block diagonal matrix repeating `n` times the matrix `A`."
repeatdiag(A, n::Int) = kron(I(n), A)

"In-place version of `repeat` but for vectors only."
function repeat!(Y::Vector, a::Vector, n::Int)
    na = length(a)
    for i=0:n-1
        # stop if Y is too short, another clearer error is thrown later in the code:
        na*(i+1) > length(Y) && break 
        Y[(1+na*i):(na*(i+1))] = a
    end
    return Y
end

"Convert 1-element vectors and normal matrices to Hermitians."
to_hermitian(A::AbstractVector) = Hermitian(reshape(A, 1, 1), :L)
to_hermitian(A::AbstractMatrix) = Hermitian(A, :L)
to_hermitian(A::Hermitian) = A
to_hermitian(A) = A

"""
Compute the inverse of a the Hermitian positive definite matrix `A` using `cholesky`.

Builtin `inv` function uses LU factorization which is not the best choice for Hermitian
positive definite matrices. The function will mutate `buffer` to reduce memory allocations.
"""
function inv_cholesky!(buffer::Matrix, A::Hermitian)
    Achol  = Hermitian(buffer, :L)
    Achol .= A
    chol_obj = cholesky!(Achol)
    invA = Hermitian(inv(chol_obj), :L)
    return invA
end