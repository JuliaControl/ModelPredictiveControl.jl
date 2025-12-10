const DEFAULT_HP0 = 10
const DEFAULT_HC  = 2
const DEFAULT_MWT = 1.0
const DEFAULT_NWT = 0.1
const DEFAULT_LWT = 0.0
const DEFAULT_CWT = 1e5
const DEFAULT_EWT = 0.0

"All deterministic algorithms for matrix coloring order in `SparseMatrixColoring.jl`."
const ALL_COLORING_ORDERS = (
    NaturalOrder(),
    LargestFirst(),
    SmallestLast(),
    IncidenceDegree(),
    DynamicLargestFirst(),
    RandomOrder(StableRNG(0), 0)
)

const HIDDEN_GETINFO_KEYS_MHE = (
    :What, :xhatarr, :epsilon, :Xhat, :xhat, :Vhat, :Pbar, :xbar, :Yhat, :Yhatm, :ϵ,
    :nablaJ, :nabla2J, :nablag, :nabla2lg, :nablageq, :nabla2lgeq
)

const HIDDEN_GETINFO_KEYS_MPC = (
    :DeltaU, :epsilon, :Dhat, :yhat, :Yhat, :xhatend, :Yhats, :Rhaty, :Rhatu,
    :nablaJ, :nabla2J, :nablag, :nabla2lg, :nablageq, :nabla2lgeq
)

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
        if key in HIDDEN_GETINFO_KEYS_MHE || key in HIDDEN_GETINFO_KEYS_MPC
            # skip the redundant non-Unicode keys
            continue
        end
        mystr *= "  :$key => $value\n"
    end
    if haskey(info, :sol)
        split_sol = split(string(info[:sol]), "\n")
        # Add the treeview prefix to each line
        solstr = join(("   " * line for line in split_sol), "\n")
        mystr *= "  :sol => \n" * solstr * "\n"  # Ensure a trailing newline
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
            @warn "Solving time limit is not supported by the $(JuMP.solver_name(optim)) "*
                  "optimizer."
        else
            rethrow()
        end
    end
end

"Init a differentiation result matrix as dense or sparse matrix, as required by `backend`."
init_diffmat(T, ::AbstractADType, _ , nx, ny) = zeros(T, ny, nx)
function init_diffmat(T, ::AutoSparse, prep , _ , _ )
    A = similar(sparsity_pattern(prep), T)
    return A .= 0
end

"Init the sparsity structure of matrix `A` as required by `JuMP.jl`."
function init_diffstructure(A::AbstractSparseMatrix)
    I, J = findnz(A)
    return collect(zip(I, J))
end
init_diffstructure(A::AbstractMatrix) = Tuple.(CartesianIndices(A))[:]

"Get the lower-triangular indices from the differentiation matrix structure `i_vec`."
function lowertriangle_indices(i_vec::Vector{Tuple{Int, Int}})
    return [(i,j) for (i,j) in i_vec if i ≥ j]
end

"Store the diff. matrix `A` in the vector `v` with list of nonzero indices `i_vec`." 
function fill_diffstructure!(
    v::AbstractVector, A::AbstractMatrix, i_vec::Vector{Tuple{Int, Int}}
)
    for i in eachindex(i_vec)
        i_A, j_A = i_vec[i]
        v[i] = A[i_A, j_A]
    end
    return v
end

"Store the diff. matrix `A` in the matrix `T` with list of nonzero indices `i_vec`." 
function fill_diffstructure!(
    T::AbstractMatrix, A::AbstractMatrix, i_vec::Vector{Tuple{Int, Int}}
)
    for i in eachindex(i_vec)
        i_A, j_A = i_vec[i]
        T[i_A, j_A] = A[i_A, j_A]
    end
    return T
end

backend_str(backend::AbstractADType) = string(nameof(typeof(backend)))
backend_str(backend::Nothing) = "nothing"
function backend_str(backend::AutoSparse)
    str =   "AutoSparse ($(nameof(typeof(backend.dense_ad))),"*
            " $(nameof(typeof(backend.sparsity_detector))),"*
            " $(nameof(typeof(backend.coloring_algorithm))))"
    return str
end
function backend_str(backend::SecondOrder)
    str =   "SecondOrder ($(nameof(typeof(backend.outer))),"*
                        " $(nameof(typeof(backend.inner))))"
    return str
end
dense_backend(backend::AbstractADType) = backend
dense_backend(backend::AutoSparse) = backend.dense_ad
dense_backend(backend::SecondOrder) = backend.inner

"Validate `hessian` keyword argument and return the differentiation `backend`."
function validate_hessian(hessian, gradient, oracle, default)
    if hessian == true
        backend = default
    elseif hessian == false || isnothing(hessian)
        backend = nothing
    else
        backend = hessian
    end
    if oracle == false && !isnothing(backend)
        error("Second order derivatives are only supported with oracle=true.")
    end
    if oracle == true && !isnothing(backend)
        hess = dense_backend(backend)
        grad = dense_backend(gradient)
        if hess != grad
            @info "The objective function gradient will be computed with the hessian "*
                "backend ($(backend_str(hess)))\n instead of the one in gradient "*
                "argument ($(backend_str(grad))) for efficiency."
        end
    end
    return backend
end

"Verify that x and y elements are different using `!==`."
isdifferent(x, y) = any(xi !== yi for (xi, yi) in zip(x, y))

"Generate a block diagonal matrix repeating `n` times the matrix `A`."
repeatdiag(A, n::Int) = kron(I(n), A)
function repeatdiag(A::Hermitian{NT, Diagonal{NT, Vector{NT}}}, n::Int) where {NT<:Real}
    return Hermitian(repeatdiag(A.data, n), :L) # to return hermitian of a `Diagonal`
end

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
Compute the inverse of a the Hermitian positive definite matrix `A` in-place and return it.

There is 3 methods for this function:
- If `A` is a `Hermitian{<Real, Matrix{<:Real}}`, it uses `LAPACK.potrf!` and 
  `LAPACK.potri!` functions to compute the Cholesky factor and then the inverse. This is
  allocation-free. See <https://tinyurl.com/4pwdwbcj> for the source.
- If `A` is a `Hermitian{<Real, Diagonal{<:Real, Vector{<:Real}}}`, it computes the 
  inverse of the diagonal elements in-place (allocation-free).
- Else if `A` is a `Hermitian{<:Real, <:AbstractMatrix}`, it computes the Cholesky factor
  with `cholesky!` and then the inverse with `inv`, which allocates memory.
"""
function inv!(A::Hermitian{NT, Matrix{NT}}) where {NT<:Real}
    _, info = LAPACK.potrf!(A.uplo, A.data)
    (info == 0) || throw(PosDefException(info))
    LAPACK.potri!(A.uplo, A.data)
    return A
end
function inv!(A::Hermitian{NT, Diagonal{NT, Vector{NT}}}) where {NT<:Real}
    A.data.diag .= 1 ./ A.data.diag
    return A
end
function inv!(A::Hermitian{<:Real, <:AbstractMatrix})
    Achol = cholesky!(A)
    invA = inv(Achol)
    A .= Hermitian(invA, :L)
    return A
end

"Add `Threads.@threads` to a `for` loop if `flag==true`, else leave the loop as is."
macro threadsif(flag, expr)
    quote
        if $(flag)
            Threads.@threads $expr
        else
            $expr
        end
    end |> esc
end

"Add `ProgressLogging.@progress` with the name `name`  to a `for` loop if `flag==true`"
macro progressif(flag, name, expr)
    quote
        if $(flag)
            ProgressLogging.@progress $name $expr
        else
            $expr
        end
    end |> esc
end