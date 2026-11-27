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

    // Semi-proximal support: prox_mask[i] == 1 iff direction i needed eps
    // regularization to make the Cholesky factor non-singular.
    int* prox_mask;
    int  n_prox; // Number of directions that needed regularization


    // Soft constraint
    c_float soft_slack;

    // Settings
    DAQPSettings* settings;

    // BnB
    DAQPBnB* bnb;
    // Hierarchical QP
    int nh;
    int* break_points;
    // AVI
    DAQPAVI* avi;
    // Timer (used for time limit checking, set externally by daqp_solve)
    void *timer;
}DAQPWorkspace;

# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif //ifndef DAQP_TYPES_H
