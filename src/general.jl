"Termination status that means 'no solution available'."
const FATAL_STATUSES = [
    INFEASIBLE, DUAL_INFEASIBLE, LOCALLY_INFEASIBLE, INFEASIBLE_OR_UNBOUNDED, 
    NUMERICAL_ERROR, INVALID_MODEL, INVALID_OPTION, INTERRUPTED, 
    OTHER_ERROR
]

"Verify that the solver termination status means 'no solution available'."
isfatal(status::TerminationStatusCode) = any(status .== FATAL_STATUSES)

"Evaluate the quadratic programming objective function `0.5x'*H*x + q'*x` at `x`."
obj_quadprog(x, H, q) = 0.5*dot(x, H, x) + q'*x  # dot(x, H, x) is faster than x'*H*x

"Limit the solving time to `Ts` if supported by `optim` optimizer."
function limit_solve_time(optim, Ts)
    try
        set_time_limit_sec(optim,Ts)
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