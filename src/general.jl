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
            rethrow(err)
        end
    end
end

"Generate a block diagonal matrix repeating `n` times the matrix `A`."
repeatdiag(A, n::Int) = kron(I(n), A)

"Convert 1-element vectors and normal matrices to Hermitians."
to_hermitian(A::AbstractVector) = Hermitian(reshape(A, 1, 1), :L)
to_hermitian(A::AbstractMatrix) = Hermitian(A, :L)
to_hermitian(A::Hermitian) = A