#ifndef DAQP_TYPES_H
# define DAQP_TYPES_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

#ifdef DAQP_SINGLE_PRECISION
typedef float c_float;
#else
typedef double c_float;
#endif

typedef struct{

    // Data for the QP problem
    //
    // min  0.5 x'*H*x + f'x
    // s.t   lb  <=  x  <= ub
    //       lbA <= A*x <= ubA
    //
    // n  - dimension of x
    // m  - total number of constraints
    // ms - number of simple bounds
    // blower = [lb; lbA];
    // bupper = [ub; ubA];
    // (The number of rows in A is hence m-ms)

    // sense define the state of the constraints
    // (active, immutable, upper/lower, soft, binary).

    int n;
    int m;
    int ms;

    c_float* H;
    c_float* f;

    c_float* A;
    c_float* bupper;
    c_float* blower;

    int* sense;

    // Hierarchical QP
    int* break_points;
    int nh;
    // Extra flags for problem
    int problem_type; // 1 == AVI otherwise QP
}DAQPProblem;

typedef struct{
    c_float primal_tol;
    c_float dual_tol;
    c_float zero_tol;
    c_float pivot_tol;
    c_float progress_tol;

    int cycle_tol;
    int iter_limit;
    c_float fval_bound;

    c_float eps_prox;
    c_float eta_prox;

    c_float rho_soft;

    c_float rel_subopt;
    c_float abs_subopt;

    c_float sing_tol;
    c_float refactor_tol;
    c_float time_limit;
}DAQPSettings;


typedef struct{
    int bin_id;
    int depth;
    int WS_start;
    int WS_end;
}DAQPNode;

typedef struct{
    int* bin_ids;
    int nb;
    int neq;

    DAQPNode* tree;
    int  n_nodes;

    int* tree_WS;
    int nWS;
    int n_clean;
    int* fixed_ids;

    int nodecount;
    int itercount;
}DAQPBnB;

typedef struct{
    int is_symmetric;
    int retry_rho_needed;

    c_float* Hsym;
    c_float* Hs_rho;
    c_float* H_rho;
    int* P_H2;

    c_float* LU_H;
    int* P_H;

    c_float* kkt_buffer;
    int* P_S;

    c_float* xtemp;
    c_float* Hx;
    c_float* x;
    c_float* y;

    c_float rho;
}DAQPAVI;

/*
 * Elimination of equality constraints.
 *
 * The LDP
 *   min 0.5||u||^2  s.t.  dlower <= M u <= dupper,
 * in which the rows E are equalities (M_E u = d_E), is reduced with the QR
 * factorization M_E' = [Q1 Q2] [R; 0] by the change of variables
 *   u = Q1 y1 + Q2 w,   R' y1 = d_E.
 * Since ||u||^2 = ||y1||^2+||w||^2, the reduced problem in w is again an LDP,
 * with M_r = M Q2 and d_r = d - M Q1 y1, so no refactorization is needed.
 * The reduced problem has no simple bounds (ms = 0) and keeps the constraint
 * indexing of the original problem, with the eliminated rows marked immutable.
 */
typedef struct{
    int n;  // Number of primal variables before elimination
    int m;  // Number of constraints before elimination
    int ms; // Number of simple bounds before elimination
    int neq; // Number of eliminated equality constraints
    int nign; // Number of equalities that were linearly dependent (ignored)
    int ndrop; // Number of constraints that the equalities imply
    int ncand; // Number of equality candidates that memory is allocated for
    int nz; // Reduced number of primal variables (n-neq)
    int m_r; // Reduced number of constraints
    int installed; // Whether the reduced problem is currently in the workspace
    int expanded; // Whether the reduced solution has already been expanded

    int* eq_ids; // The neq eliminated, followed by the nign ignored, equalities
    int* drop_ids; // Constraints that are implied by the equalities
    int* map; // Constraint index in the original problem of each reduced one
    c_float* Q; // Householder vectors (leading neq columns) and Q2 (trailing)
    c_float* R; // Upper triangular factor of M_E' (packed by columns)
    c_float* tau; // Householder scalars
    c_float* s_eq; // Normalization of the eliminated equality constraints
    c_float* y1; // Q1-part of u (solves R'y1 = d_E)
    c_float* lam_eq; // Multipliers of the eliminated constraints
    c_float* W; // Rinv*Q2 (n x nz, row major)
    c_float* xp; // Part of the solution that the equalities determine
    c_float* tmp; // Scratch of size 2n

    c_float up_norm2; // ||up||^2 (offset in the objective function)

    // Reduced problem (swapped into the workspace while solving)
    c_float* M;
    c_float* dupper;
    c_float* dlower;
    c_float* scaling;
    c_float* Mu;
    int* sense;
    // Full problem (restored into the workspace when it is updated). The
    // reduced problem has an identity Hessian, so the Hessian factor and the
    // linear term are set aside as well: the solver then sees an ordinary
    // least-distance problem.
    c_float* M_full;
    c_float* dupper_full;
    c_float* dlower_full;
    c_float* scaling_full;
    c_float* Mu_full;
    c_float* Rinv_full;
    c_float* RinvD_full;
    c_float* v_full;
    int* sense_full;
}DAQPEqElim;

typedef struct{
    DAQPProblem* qp;
    // LDP data
    int n; // Number of primal variables
    int m; // Number of constraints
    int ms; // Number of simple bounds
    c_float *M; // M' M is the Hessian of the dual objective function (dimensions: n x m)
    c_float *dupper; // Linear part of dual objective function (dimensions: m x 1)
    c_float *dlower; // Linear part of dual objective function (dimensions: m x 1)
    c_float *Rinv; // Inverse of upper cholesky factor of primal Hessian
    c_float *v; // v = R'\f (used to transform QP to LDP
    int *sense; // State of constraints
    c_float *scaling; // normalizations
    c_float *RinvD; // in case Rinv is diagonal


    // Iterates
    c_float *x; // The final primal solution
    c_float *xold; // The latest primal solution (used for proximal-point iteratios)

    c_float* lam; // Dual iterate
    c_float* lam_star; // Current constrained stationary point
    c_float* u; // Stores Mk' lam_star
    c_float fval;

    // LDL factors (Mk Mk' = L D L')
    c_float *L;
    c_float *D;
    // Intermittent variables (LDL')
    c_float* xldl; // Solution to L xdldl = -dk
    c_float* zldl; // zldl_i = xldl_i/D_i
    int reuse_ind; // How much work that can be saved when solving Mk Mk' lam* = -dk

    int *WS; // Working set, size: maximum number of constraints (n+ns+1)
    int n_active; // Number of active contraints

    int iterations;
    int sing_ind; // Flag for denoting whether Mk Mk' is singular or not

    // Proximal support. Diagonal Hessians can regularize individual
    // directions; dense singular Hessians use a full shift for stability.
    int* prox_mask;
    int  n_prox; // Number of directions that needed regularization


    // Soft constraint
    c_float soft_slack;
#ifdef SOFT_WEIGHTS
    // The softened objective is given by
    //    min  0.5 x'*H*x + f'x + 0.5 su'su + 0.5 sl'sl,
    // and the softened constraints are given by (similarly for simple bounds)
    //    lbA-rho_ls*sl <= A*x <= ubA+rho_us*su,
    // with the bounds sl >= d_ls, su >= d_us.
    // The bounds are assumed to include the contribution from d_ls/d_us,
    // since the slacks start active at their bounds.
    c_float *d_ls;
    c_float *d_us;
    c_float *rho_ls;
    c_float *rho_us;
#endif

    // Settings
    DAQPSettings* settings;

    // BnB
    DAQPBnB* bnb;
    // Hierarchical QP
    int nh;
    int* break_points;
    // AVI
    DAQPAVI* avi;
    // Equality elimination (NULL if the LDP is not reduced)
    DAQPEqElim* eq;
    // Timer (used for time limit checking, set externally by daqp_solve)
    void *timer;
    // M*u from the latest feasibility scan (length m-ms); NULL disables batching
    c_float *Mu;
}DAQPWorkspace;

#define DAQP_IS_HIERARCHICAL(work) \
    ((work)->break_points != NULL && (work)->nh > 1)

# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif //ifndef DAQP_TYPES_H
