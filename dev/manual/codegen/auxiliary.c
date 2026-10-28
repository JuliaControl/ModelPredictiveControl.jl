#include "auxiliary.h"
#include <float.h>
#include "factorization.h"
#include "daqp.h"

/* Soft constraints (see types.h for the penalty and its weights, which are
 * uniform if DAQP_NO_SOFT_WEIGHTS is set). A soft constraint contributes
 * 0.5*rho*(|lam|-w)_+^2 to the dual, so its slack is zero while |lam| <= w
 * (DAQP_SLACK_FIXED) and rho*(|lam|-w) beyond that (DAQP_SLACK_FREE); only the
 * latter adds rho to the diagonal of the dual Hessian and shifts the dual
 * linear term by rho*w. The penalty is continuously differentiable, so |lam|
 * passing w is an ordinary blocking event (see daqp_remove_blocking).
 */

// Nonzero unless every soft constraint has the same, purely quadratic penalty,
// in which case the branches below stay out of the hot loops
#ifdef DAQP_SOFT_WEIGHTS
#define DAQP_HAS_L1(work) \
    ((work)->settings->w_soft != 0 || (work)->rho_ls != NULL)
#else
#define DAQP_HAS_L1(work) ((work)->settings->w_soft != 0)
#endif

// Reciprocal quadratic weight of the active side of constraint id (zero
// selects settings->rho_soft, which is given in the normalized formulation)
static inline c_float daqp_soft_rho(DAQPWorkspace *work, const int id){
#ifdef DAQP_SOFT_WEIGHTS
    if(work->rho_ls != NULL){
        const c_float rho = DAQP_IS_LOWER(id) ? work->rho_ls[id] : work->rho_us[id];
        if(rho != 0)
            return work->scaling ? rho*work->scaling[id]*work->scaling[id] : rho;
    }
#else
    (void)id;
#endif
    return work->settings->rho_soft;
}

// Linear weight of the active side of constraint id, i.e. what the multiplier
// has to exceed for the slack to become nonzero (zero selects settings->w_soft)
static inline c_float daqp_soft_w(DAQPWorkspace *work, const int id){
#ifdef DAQP_SOFT_WEIGHTS
    if(work->w_ls != NULL){
        const c_float w = DAQP_IS_LOWER(id) ? work->w_ls[id] : work->w_us[id];
        if(w != 0)
            return work->scaling ? w/work->scaling[id] : w;
    }
#else
    (void)id;
#endif
    return work->settings->w_soft;
}

// Signed violation of a free soft constraint for multiplier lam. At lam = 0,
// its negative is the shift of the dual linear term.
static inline c_float daqp_soft_residual(DAQPWorkspace *work, const int id,
        const c_float lam){
    const c_float w = daqp_soft_w(work,id);
    if(w == 0)
        return lam == 0 ? 0 : daqp_soft_rho(work,id)*lam;
    return daqp_soft_rho(work,id)
        *(lam + (DAQP_IS_LOWER(id) ? w : -w));
}

// Contribution to the objective from the slack of constraint id
static inline c_float daqp_soft_penalty(DAQPWorkspace *work, const int id,
        const c_float lam){
    const c_float w = daqp_soft_w(work,id);
    if(w == 0) return daqp_soft_rho(work,id)*lam*lam; // Plain quadratic penalty
    if(DAQP_IS_SLACK_FIXED(id)) return 0; // The slack is zero
    return daqp_soft_rho(work,id)*(lam*lam-w*w);
}

// Largest violation of a soft constraint, in the units of the original
// problem (only needed once a solution has been found)
c_float daqp_max_soft_slack(DAQPWorkspace *work){
    int i;
    c_float smax = 0;
    for(i = 0; i < work->n_active; i++){
        const int id = work->WS[i];
        if(!DAQP_IS_SOFT(id)) continue;
        c_float s = daqp_soft_slack(work,i);
        if(s < 0) s = -s;
        if(work->scaling != NULL) s /= work->scaling[id]; // Undo the normalization
        if(s > smax) smax = s;
    }
    return smax;
}

/*
 * Cycling that persists after the working set has been refactorized: constraints
 * are then only added if their violation also exceeds the rounding level of
 * u = -M'lam, and daqp_solve checks the solution with primal_tol afterwards
 * (daqp_violates_hard).
 * Returns 0 (the cycling is reported) for the proximal method and hierarchical
 * problems, which handle cycling themselves, for a nonsymmetric AVI, or if the
 * floor is already used.
 */
int daqp_set_noise_floor(DAQPWorkspace *work){
    if(work->n_prox > 0 || DAQP_IS_HIERARCHICAL(work) ||
            (work->avi != NULL && !work->avi->is_symmetric) ||
            (work->state & DAQP_STATE_NOISE_FLOOR)) return 0;
    work->state |= DAQP_STATE_NOISE_FLOOR;
    return 1;
}

// The (negative) rounding level of u = -M'lam, below which violations are not
// added (0 if the floor is not used; the rows of M are normalized)
c_float daqp_noise_floor(DAQPWorkspace *work){
    int i;
    c_float s = 0;
    if(!(work->state & DAQP_STATE_NOISE_FLOOR)) return 0;
    for(i = 0; i < work->n_active; i++) s += work->lam_star[i] < 0 ? -work->lam_star[i] : work->lam_star[i];
    return -DAQP_ADD_NOISE_GAIN*(sizeof(c_float) == sizeof(float) ? FLT_EPSILON : DBL_EPSILON)*s;
}

// Slack of the soft constraint that is active at working set index i
c_float daqp_soft_slack(DAQPWorkspace *work, const int i){
    const int id = work->WS[i];
    if(DAQP_IS_SLACK_FIXED(id)) return 0;
    return daqp_soft_residual(work,id,work->lam_star[i]);
}

void daqp_remove_constraint(DAQPWorkspace* work, const int rm_ind){
    int i;
    // Update data structures
    DAQP_SET_INACTIVE(work->WS[rm_ind]);
    daqp_update_LDL_remove(work,rm_ind);
    (work->n_active)--;

    for(i=rm_ind;i<work->n_active;i++){
        work->WS[i] = work->WS[i+1];
        work->lam[i] = work->lam[i+1];
    }
    // Can only reuse work less than the ind that was removed
    if(rm_ind < work->reuse_ind)
        work->reuse_ind = rm_ind;

    // Check if the removal lead to singularity (can happen due to numerics).
    if(work->n_active > 0 && work->D[work->n_active-1] < work->settings->sing_tol)
        work->sing_ind = work->n_active-1;
    else{ // Pivot for improved numerics
        daqp_pivot_last(work);
    }
}
// Add a constraint, keeping the slack state that is marked in sense
static void daqp_add_constraint_keep_slack(DAQPWorkspace *work,
        const int add_ind, c_float lam){
    // Update data structures
    DAQP_SET_ACTIVE(add_ind);
    const c_float rho = DAQP_IS_SOFT(add_ind) && DAQP_IS_SLACK_FREE(add_ind)
        ? daqp_soft_rho(work,add_ind) : 0;
    daqp_update_LDL_add(work,add_ind,rho);
    work->WS[work->n_active] = add_ind;
    work->lam[work->n_active] = lam;
    work->n_active++;

    // Pivot for improved numerics
    daqp_pivot_last(work);
}

void daqp_add_constraint(DAQPWorkspace *work, const int add_ind, c_float lam){
    work->sense[add_ind] &= ~DAQP_SLACK_SWITCHED;
    // Mark whether the slack is zero, given the multiplier
    if(DAQP_IS_SOFT(add_ind)){
        const c_float w = daqp_soft_w(work,add_ind);
        if(w > 0){
            // The slack switches when the constraint is removed (a soft equality
            // with only a quadratic penalty stays immutable)
            DAQP_SET_MUTABLE(add_ind);
            const c_float magnitude = (DAQP_IS_LOWER(add_ind) ? -lam : lam) >= w ? w : 0;
            lam = DAQP_IS_LOWER(add_ind) ? -magnitude : magnitude;
        }
        if(w > 0 && (DAQP_IS_LOWER(add_ind) ? -lam : lam) < w)
            DAQP_SET_SLACK_FIXED(add_ind);
        else
            DAQP_SET_SLACK_FREE(add_ind);
    }
    daqp_add_constraint_keep_slack(work,add_ind,lam);
}

void daqp_compute_primal_and_fval(DAQPWorkspace *work){
    int i,j,disp,id;
    c_float fval=0;
    const int has_l1 = DAQP_HAS_L1(work);
    // Reset u
    for(j=0;j<work->n;j++)
        work->u[j]=0;
    //u[m] <-- Mk'*lam_star (zero if empty set)
    for(i=0;i<work->n_active;i++){
        id = work->WS[i];
        const c_float li = work->lam_star[i]; // hoist: invariant across the inner loop
        if(id < work->ms){
            // Simple constraint
            if(work->Rinv!=NULL){ // Hessian is not identity
                for(j=id, disp=DAQP_R_OFFSET(id,work->n);j<work->n;++j)
                    work->u[j]-=work->Rinv[disp+j]*li;
            }
            else work->u[id]-=li; // Hessian is identity
        }
        else{ // General constraint
            for(j=0,disp=work->n*(id-work->ms);j<work->n;j++)
                work->u[j]-=work->M[disp++]*li;
        }
        if(DAQP_IS_SOFT(id)){
            fval += has_l1 ? daqp_soft_penalty(work,id,li)
                : work->settings->rho_soft*li*li;
            work->sense[id] &= ~DAQP_SLACK_SWITCHED;
        }
    }
    for(j=0;j<work->n;j++)
        fval+=work->u[j]*work->u[j];
    work->fval = fval;
}
int daqp_add_infeasible(DAQPWorkspace *work){
    int j,disp;
    c_float ep = -work->settings->primal_tol;
    c_float min_val = 0.0;
    c_float bound;
    c_float Mu,min_cand;
    int isupper=0, add_ind=DAQP_EMPTY_IND;
    const c_float noise = daqp_noise_floor(work); // 0 unless cycling persisted
    // Simple bounds
    for(j=0, disp=0;j<work->ms;j++){
        // Never activate immutable or already active constraints
        if(work->sense[j]&(DAQP_ACTIVE+DAQP_IMMUTABLE)){
            disp+=work->n-j;
            continue;
        }
        if(work->Rinv==NULL){// Hessian is identify
            Mu=work->u[j];
        }
        else{
            Mu = daqp_dot_inline(work->Rinv+disp,work->u+j,work->n-j);
        }
        disp+=work->n-j;
        bound = (work->scaling == NULL) ? ep : ep*work->scaling[j];
        if(bound > noise) bound = noise;
        min_cand = work->dupper[j]-Mu;
        if(min_cand < min_val && min_cand < bound){
            add_ind = j; isupper = 1;
            min_val = min_cand;
        }
        else{
            min_cand = Mu - work->dlower[j];
            if(min_cand < min_val && min_cand < bound){
                add_ind = j; isupper = 0;
                min_val = min_cand;
            }
        }
    }
    /* General two-sided constraints */
    daqp_compute_Mu(work);
    for(j=work->ms, disp=0;j<work->m;j++){
        // Never activate immutable or already active constraints
        if(work->sense[j]&(DAQP_ACTIVE+DAQP_IMMUTABLE)){
            disp+=work->n;// Skip ahead in M
            continue;
        }
        Mu = work->Mu == NULL
            ? daqp_dot_inline(work->M+disp,work->u,work->n)
            : work->Mu[j-work->ms];
        disp+=work->n;
        bound = (work->scaling == NULL) ? ep : ep*work->scaling[j];
        if(bound > noise) bound = noise;

        min_cand = work->dupper[j]-Mu;
        if(min_cand < min_val &&  min_cand < bound){
            add_ind = j; isupper = 1;
            min_val = min_cand;
        }
        else{
            min_cand = Mu - work->dlower[j];
            if(min_cand < min_val && min_cand < bound){
                add_ind = j; isupper = 0;
                min_val = min_cand;
            }
        }
    }
    // No constraint is infeasible => return
    if(add_ind == DAQP_EMPTY_IND) return 0;
    // Otherwise add infeasible constraint to working set
    if(isupper)
        DAQP_SET_UPPER(add_ind);
    else
        DAQP_SET_LOWER(add_ind);
    // Set lam = lam_star
    c_float *swp_ptr;
    swp_ptr=work->lam; work->lam = work->lam_star; work->lam_star=swp_ptr;
    daqp_add_constraint(work,add_ind,isupper ? -min_val : min_val);
    return 1;
}

// Compute all general constraint products from the existing row-major M.
// Four rows are accumulated together to reuse u and expose independent
// accumulation chains.  A NULL Mu buffer keeps the row-wise fallback active.
void daqp_compute_Mu(DAQPWorkspace *work){
    if(work->Mu == NULL) return;

    const int rows = work->m-work->ms;
    const int n = work->n;
    int row = 0;
    for(; row+3 < rows; row+=4){
        const c_float *m0 = work->M+(row+0)*n;
        const c_float *m1 = work->M+(row+1)*n;
        const c_float *m2 = work->M+(row+2)*n;
        const c_float *m3 = work->M+(row+3)*n;
        c_float sum0=0, sum1=0, sum2=0, sum3=0;
        for(int k=0; k<n; k++){
            const c_float uk = work->u[k];
            sum0 += m0[k]*uk;
            sum1 += m1[k]*uk;
            sum2 += m2[k]*uk;
            sum3 += m3[k]*uk;
        }
        work->Mu[row+0] = sum0;
        work->Mu[row+1] = sum1;
        work->Mu[row+2] = sum2;
        work->Mu[row+3] = sum3;
    }
    for(; row<rows; row++)
        work->Mu[row] = daqp_dot_inline(work->M+row*n,work->u,n);
}
// Lower bound sum rho*p_i^2 (free soft slacks) on the dual curvature along the
// singular direction p = lam_star; zero if the slacks only carry rounding
static c_float daqp_soft_curvature(DAQPWorkspace *work){
    int i;
    c_float c = 0, ps = 0, pp = 0;
    for(i = 0; i <= work->sing_ind; i++){
        const int id = work->WS[i];
        const c_float p2 = work->lam_star[i]*work->lam_star[i];
        pp += p2;
        if(DAQP_IS_SOFT(id) && DAQP_IS_SLACK_FREE(id)){
            c += daqp_soft_rho(work,id)*p2;
            ps += p2;
        }
    }
    return ps > DAQP_SOFT_CURV_TOL*pp ? c : 0;
}

// Step lam toward lam_star (along lam_star if singular) until a multiplier
// reaches zero (removal) or passes w (slack switch). Returns 0 if full step
int daqp_remove_blocking(DAQPWorkspace *work){
    int i, ind, rm_ind = DAQP_EMPTY_IND;
    const int singular = work->sing_ind != DAQP_EMPTY_IND;
    const int has_l1 = DAQP_HAS_L1(work);
    c_float alpha = DAQP_INF, alpha_cand, y, ystar, p, target, rm_target = 0;
    // Blocking beyond dual_tol, or beyond zero_tol for a singular direction (which
    // is not a multiplier, but scaled to 1 for the constraint that made it singular)
    const c_float tol = singular ? work->settings->zero_tol : work->settings->dual_tol;

    for(i = 0; i < work->n_active; i++){
        ind = work->WS[i];
        if(DAQP_IS_IMMUTABLE(ind)) continue;
        // Fold the sign of the multiplier (dual feasibility <=> y >= 0).
        // ystar is the end point of the step, or its direction if singular.
        const int lower = DAQP_IS_LOWER(ind);
        ystar = lower ? -work->lam_star[i] : work->lam_star[i];

        if(!has_l1 || !DAQP_IS_SOFT(ind)){ // Blocked when the multiplier reaches zero
            if(ystar >= -tol) continue;
            target = 0;
        }
        else{
            // The multiplier is confined to [0,w] while the slack is zero and
            // to [w,inf) otherwise; the state switches when it leaves its range
            const c_float w = daqp_soft_w(work,ind);
            const int fixed = DAQP_IS_SLACK_FIXED(ind);
            target = fixed ? 0 : w; // Blocked from below
            if(ystar >= (singular ? 0 : target) - tol){
                if(!fixed || w == 0 || ystar <= (singular ? 0 : w) + tol)
                    continue;
                target = w; // A zero slack is released
            }
            // Switching back at w without a primal step is rounding
            if(!singular && w > 0 && target == w && (work->sense[ind] & DAQP_SLACK_SWITCHED) &&
                    (lower ? -work->lam[i] : work->lam[i]) == w)
                continue;
        }

        y = lower ? -work->lam[i] : work->lam[i];
        p = singular ? ystar : ystar-y;
        alpha_cand = (target-y)/p;
        if(target != 0 && alpha_cand < 0) alpha_cand = 0;
        if(alpha_cand < alpha){
            alpha = alpha_cand;
            rm_ind = i;
            rm_target = target;
        }
    }
    if(rm_ind == DAQP_EMPTY_IND){
        if(singular){
            const c_float c = daqp_soft_curvature(work);
            if(c > 0){ // Blocked by the slacks
                if(work->D[work->sing_ind] < c) work->D[work->sing_ind] = c;
                if(work->reuse_ind > work->sing_ind) work->reuse_ind = work->sing_ind;
                work->sing_ind = DAQP_EMPTY_IND;
                return 1;
            }
        }
        return 0; // Either dual feasible or primal infeasible
    }

    // A zero-length transition cannot make progress when the CSP is singular:
    // the zero slack is what makes the working set rank deficient
    if(singular && alpha <= 0) rm_target = 0;

    // Update lambda
    if(singular)
        for(i = 0; i < work->n_active; i++)
            work->lam[i] += alpha*work->lam_star[i];
    else
        for(i = 0; i < work->n_active; i++)
            work->lam[i] += alpha*(work->lam_star[i]-work->lam[i]);

    work->sing_ind = DAQP_EMPTY_IND;
    ind = work->WS[rm_ind];
    if(rm_target == 0){ // The constraint leaves the working set
        daqp_remove_constraint(work,rm_ind);
        // Earlier slack switches are now genuine
        if(has_l1)
            for(i = 0; i < work->n_active; i++)
                work->sense[work->WS[i]] &= ~DAQP_SLACK_SWITCHED;
        return 1;
    }

    // The slack switches state, which only adds or removes rho on the diagonal
    const c_float lam = DAQP_IS_LOWER(ind) ? -rm_target : rm_target;
    const int release = DAQP_IS_SLACK_FIXED(ind);
    if(release) DAQP_SET_SLACK_FREE(ind);
    else DAQP_SET_SLACK_FIXED(ind);
    work->sense[ind] |= DAQP_SLACK_SWITCHED;

    // Nothing in the factorization depends on the diagonal of the last row,
    // so a slack there can switch without forming its row of M*M' again
    if(rm_ind == work->n_active-1 && !singular){
        const c_float rho = daqp_soft_rho(work,ind);
        work->D[rm_ind] += release ? rho : -rho;
        work->lam[rm_ind] = lam;
        // The shift in this row's RHS changed, so it has to be recomputed
        if(work->reuse_ind > rm_ind) work->reuse_ind = rm_ind;
        // Singularity as in daqp_update_LDL_add: a free slack adds rho to the
        // diagonal, and so relaxes the rank condition
        int ns_active = 0;
        for(i = 0; i < work->n_active; i++)
            if(DAQP_IS_SOFT(work->WS[i]) && DAQP_IS_SLACK_FREE(work->WS[i]))
                ns_active++;
        if(work->D[rm_ind] < work->settings->sing_tol ||
                rm_ind >= work->n + ns_active)
            work->sing_ind = rm_ind;
        else
            daqp_pivot_last(work); // The new diagonal may be a worse pivot
    }
    else{
        daqp_remove_constraint(work,rm_ind);
        if(work->sing_ind == DAQP_EMPTY_IND)
            daqp_add_constraint_keep_slack(work,ind,lam);
    }
    return 1;
}

// x <-- L'\x for the unit lower triangular L 
static void daqp_solve_Lt(const c_float *L, c_float *x, const int n){
    int i, j;
    for(i = n-1; i > 0; i--){
        const c_float xi = x[i];
        const c_float *Li = L + DAQP_ARSUM(i);
        for(j = 0; j+4 <= i; j += 4){
            const c_float a0 = x[j]-Li[j]*xi, a1 = x[j+1]-Li[j+1]*xi;
            const c_float a2 = x[j+2]-Li[j+2]*xi, a3 = x[j+3]-Li[j+3]*xi;
            x[j] = a0; x[j+1] = a1; x[j+2] = a2; x[j+3] = a3;
        }
        for(; j < i; j++) x[j] -= Li[j]*xi;
    }
}
// Row id of M (rows id < ms are rows of Rinv) times v
static c_float daqp_row_dot(const DAQPWorkspace *work, const int id, const c_float* v){
    int j, disp;
    c_float s = 0;
    if(id >= work->ms)
        for(j = 0, disp = work->n*(id-work->ms); j < work->n; j++) s += work->M[disp++]*v[j];
    else if(work->Rinv != NULL)
        for(j = id, disp = DAQP_R_OFFSET(id,work->n); j < work->n; j++) s += work->Rinv[disp+j]*v[j];
    else s = v[id];
    return s;
}

void daqp_compute_CSP(DAQPWorkspace *work){
    int i,j,disp;
    c_float sum;
    const int has_l1 = DAQP_HAS_L1(work);
    // Forward substitution (xi <-- L\d)
    for(i=work->reuse_ind,disp=DAQP_ARSUM(work->reuse_ind); i<work->n_active; i++){
        // Setup RHS
        const int id = work->WS[i];
        sum = DAQP_IS_LOWER(id) ? -work->dlower[id] : -work->dupper[id];
        // Linear weight of a nonzero slack
        if(has_l1 && DAQP_IS_SOFT(id) && DAQP_IS_SLACK_FREE(id))
            sum -= daqp_soft_residual(work,id,0);
        for(j=0; j<i; j++)
            sum -= work->L[disp++]*work->xldl[j];
        disp++; //Skip 1 in L
        work->xldl[i] = sum;
    }
    // Scale with D  (zi = xi/di)
    for(i=work->reuse_ind; i<work->n_active; i++)
        work->zldl[i] = work->xldl[i]/work->D[i];
    //Backward substitution  (lam_star <-- L'\z)
    for(i = 0; i < work->n_active; i++) work->lam_star[i] = work->zldl[i];
    daqp_solve_Lt(work->L, work->lam_star, work->n_active);
    work->reuse_ind = work->n_active; // Save forward substitution information
}

//TODO this could probably be directly calculated in L
void daqp_compute_singular_direction(DAQPWorkspace *work){
    // Step direction is stored in lam_star
    int i,offset_L= DAQP_ARSUM(work->sing_ind);

    // Backwards substitution (p_tidle <-- L'\(-l))
    for(i = 0; i < work->sing_ind; i++) work->lam_star[i] = -work->L[offset_L+i];
    daqp_solve_Lt(work->L, work->lam_star, work->sing_ind);
    work->lam_star[work->sing_ind]=1;

    // Orient the direction such that is is a descent direction
    const int id = work->WS[work->sing_ind];
    const int lower = DAQP_IS_LOWER(id);
    int flip = lower;
    if(DAQP_IS_SOFT(id) && DAQP_IS_SLACK_FIXED(id)){
        const c_float w = daqp_soft_w(work,id);
        const c_float y = lower ? -work->lam[work->sing_ind]
                                :  work->lam[work->sing_ind];
        if(w > 0 && y >= w-work->settings->dual_tol)
            flip = !lower;
    }
    if(flip)
        for(i=0;i<=work->sing_ind;i++)
            work->lam_star[i] =-work->lam_star[i];
}


// Whether -d_W'lam exceeds fval = ||u||^2 (its value at a solution) by more
// than fval and the active residuals allowed by primal_tol
int daqp_inconsistent_dual(DAQPWorkspace *work){
    int i, id;
    c_float dl = 0, tol = 0, l;
    for(i = 0; i < work->n_active; i++){
        id = work->WS[i];
        if(DAQP_IS_SOFT(id)) return 0;
        l = work->lam_star[i];
        dl += l*(DAQP_IS_LOWER(id) ? work->dlower[id] : work->dupper[id]);
        tol += (l < 0 ? -l : l)*(work->scaling != NULL ? work->scaling[id] : 1);
    }
    return -dl > 2*work->fval + work->settings->primal_tol*tol;
}

void daqp_pivot_last(DAQPWorkspace *work){
    const int rm_ind = work->n_active-2;
    if(work->n_active > 1 && work->sing_ind == DAQP_EMPTY_IND &&
            work->D[rm_ind] < work->settings->pivot_tol && // element in D small enough
            work->D[rm_ind] < work->D[work->n_active-1]){ // element in D smallar than neighbor
        const int ind_old = work->WS[rm_ind];
        // Ensure that binaries never swap order (since this order is exploited)
        if(DAQP_IS_BINARY(ind_old) && DAQP_IS_BINARY(work->WS[work->n_active-1])) return;
        if(work->bnb != NULL && rm_ind < work->bnb->n_clean) return;

        c_float lam_old = work->lam[rm_ind];
        daqp_remove_constraint(work,rm_ind); // pivot_last might be recursively called here

        if(work->sing_ind!=DAQP_EMPTY_IND) return; // Abort if D becomes singular

        // Reordering only: the slack state of the constraint is unchanged
        daqp_add_constraint_keep_slack(work,ind_old,lam_old);
    }
}

// Add constraint id to the working set, with a multiplier that reproduces the
// slack state that is marked in sense
static void daqp_activate_constraint(DAQPWorkspace *work, const int id){
    c_float lam = 1.0;
    const c_float w = DAQP_IS_SOFT(id) ? daqp_soft_w(work,id) : 0;
    if(w > 0) lam = DAQP_IS_SLACK_FREE(id) ? w+1 : 0.9*w;
    daqp_add_constraint(work,id, DAQP_IS_LOWER(id) ? -lam : lam);
}

// Remove the last constraint of a singular working set (not necessarily the
// latest added one, due to pivoting). Equalities become mutable (re-added if violated)
int daqp_drop_singular_last(DAQPWorkspace *work){
    const int id = work->WS[--work->n_active];
    DAQP_SET_INACTIVE(id);
    if(!DAQP_IS_BINARY(id)) work->sense[id] &= ~DAQP_IMMUTABLE;
    work->sing_ind = DAQP_EMPTY_IND;
    if(work->reuse_ind > work->n_active)
        work->reuse_ind = work->n_active;
    return id;
}

// Activate the constraints that are marked active in sense
// Equalities are activated before inequalities
int daqp_activate_constraints(DAQPWorkspace *work){
    int i, j, first_mutable = work->m, exitflag = 1;
    for(i = 0; i < work->m; i++){
        if(!DAQP_IS_ACTIVE(i)) continue;
        if(!DAQP_IS_IMMUTABLE(i)){
            if(i < first_mutable) first_mutable = i;
            continue;
        }
        daqp_activate_constraint(work,i);
        if(work->sing_ind != DAQP_EMPTY_IND){
            c_float dependency_residual = 0.0;
            c_float dependency_scale = 1.0;

            //  The new equality is linearly dependent on the active equalities
            daqp_compute_singular_direction(work);
            for(j = 0; j < work->n_active; j++){
                int id = work->WS[j];
                c_float bound = DAQP_IS_LOWER(id)
                    ? work->dlower[id] : work->dupper[id];
                c_float term = work->lam_star[j] * bound;
                dependency_residual += term;
                dependency_scale += term < 0 ? -term : term;
            }

            // Dependency might only be numerical => keep as mutable constraint
            daqp_drop_singular_last(work);
            if(dependency_residual >
                        work->settings->primal_tol*dependency_scale ||
                    dependency_residual <
                        -work->settings->primal_tol*dependency_scale)
                exitflag = DAQP_EXIT_OVERDETERMINED_INITIAL;
        }
    }

    // Activate active inequalities
    for(i = first_mutable; i < work->m; i++){
        if(!DAQP_IS_ACTIVE(i) || DAQP_IS_IMMUTABLE(i)) continue;
        daqp_activate_constraint(work,i);
        if(work->sing_ind != DAQP_EMPTY_IND){
            // Drop dependent constraint and leave remaining mutable ones inactive
            daqp_drop_singular_last(work);
            for(j = i+1; j < work->m; j++){
                if(!DAQP_IS_IMMUTABLE(j)) DAQP_SET_INACTIVE(j);
            }
            return exitflag;
        }
    }
    return exitflag;
}

// Rebuild the working set from sense. For hierarchical problems, only the
// constraints of the first level are activated (since those are hard)
int daqp_reactivate_constraints(DAQPWorkspace *work){
    int error_flag;
    const int m = work->m;
    reset_daqp_workspace(work);
    if(DAQP_IS_HIERARCHICAL(work)) work->m = work->hierarchy->break_points[0];
    error_flag = daqp_activate_constraints(work);
    work->m = m;
    return error_flag;
}

// Deactivate all active constraints that are mutable (i.e., not equality constraints)
void daqp_deactivate_constraints(DAQPWorkspace *work){
    int i;
    if(work->bnb != NULL) work->bnb->n_root_WS = 0; // Also drop the BnB warm start
    for(i =0;i<work->n_active;i++){
        if(DAQP_IS_IMMUTABLE(work->WS[i])) continue;
        DAQP_SET_INACTIVE(work->WS[i]);
    }
    reset_daqp_workspace(work); // The next update activates the remaining ones
}

// Solve L*D*L'*dlam = r (r and dlam in xldl, zldl used as scratch)
void daqp_solve_working_set(DAQPWorkspace *work){
    int i, j, disp;
    const int na = work->n_active;
    c_float sum;
    // Forward substitution L*y = r
    for(i = 0, disp = 0; i < na; i++){
        sum = work->xldl[i];
        for(j = 0; j < i; j++)
            sum -= work->L[disp++]*work->xldl[j];
        disp++; // Skip the stored diagonal (= 1)
        work->xldl[i] = sum;
    }
    // Scale by D^{-1}
    for(i = 0; i < na; i++)
        work->zldl[i] = work->xldl[i]/work->D[i];
    // Backward substitution L'*dlam = zldl
    for(i = 0; i < na; i++) work->xldl[i] = work->zldl[i];
    daqp_solve_Lt(work->L, work->xldl, na);
}

// y <-- y - M_W'*dlam for the working set W
void daqp_sub_working_set_rows(DAQPWorkspace *work, const c_float* dlam, c_float* y){
    int i, j, disp, id;
    for(i = 0; i < work->n_active; i++){
        const c_float dl = dlam[i];
        id = work->WS[i];
        if(id < work->ms){
            if(work->Rinv != NULL){
                for(j = id, disp = DAQP_R_OFFSET(id,work->n); j < work->n; j++)
                    y[j] -= work->Rinv[disp+j]*dl;
            }
            else y[id] -= dl;
        }
        else
            for(j = 0, disp = work->n*(id-work->ms); j < work->n; j++)
                y[j] -= work->M[disp++]*dl;
    }
}

// Recompute a suspicious pivot when row add_ind is added (the new row of L is at L+L_start):
// c solves M_W M_W' c = M_W m (refined once) and the pivot is ||m - M_W'c||^2. If the residual
// still decreases substantially in the refinement, the working set is singular.
// (c is stored in the new row of L, and the residual r in zldl)
// Only a singular verdict changes the factorization: otherwise the original row of L and pivot
// are kept (the row is saved in the unused next row of L, which exists since n_active < n)
void daqp_refine_pivot(DAQPWorkspace *work, const int add_ind, const int L_start){
    int i, j, disp, it;
    const int k = work->n_active, n = work->n;
    c_float *c = work->L+L_start, *l = c+k+1, *r = work->zldl, rr[2];
    work->reuse_ind = 0; // xldl and zldl are used as scratch
    for(i = 0; i < k; i++) l[i] = c[i];
    daqp_solve_Lt(work->L, c, k); // c = L'\l
    for(it = 0; it < 2; it++){
        for(j = 0; j < n; j++) r[j] = 0; // r = m - M_W'c
        if(add_ind >= work->ms)
            for(j = 0, disp = n*(add_ind-work->ms); j < n; j++) r[j] = work->M[disp++];
        else if(work->Rinv != NULL)
            for(j = add_ind, disp = DAQP_R_OFFSET(add_ind,n); j < n; j++) r[j] = work->Rinv[disp+j];
        else r[add_ind] = 1;
        daqp_sub_working_set_rows(work, c, r);
        for(j = 0, rr[it] = 0; j < n; j++) rr[it] += r[j]*r[j];
        if(it == 1) break;
        for(i = 0; i < k; i++) work->xldl[i] = -daqp_row_dot(work, work->WS[i], r);
        daqp_solve_working_set(work);
        for(i = 0; i < k; i++) c[i] -= work->xldl[i];
    }
    if(rr[1] >= DAQP_PIVOT_REFINE_DECREASE*rr[0]){ // Nonsingular: keep the original row
        for(i = 0; i < k; i++) c[i] = l[i];
        return;
    }
    work->D[k] = 0;
    for(j = 0; j < k; j++) // New row of L: l = L'c (in place, since row j only uses c[i], i >= j)
        for(i = j+1, disp = DAQP_ARSUM(i)+j; i < k; disp += ++i)
            c[j] += work->L[disp]*c[i];
}

// One step of iterative refinement for active constraints.
// After computing u = -M'*lam_star, numerical errors in the LDL solve cause
// active constraint residuals r[i] = M_i*u - d_i to be nonzero. These errors
// are amplified by 1/scaling[j] in the original space, potentially exceeding
// primal_tol for near-singular factorizations with small scaling values.
// This function solves (L*D*L') * delta_lam = r using the existing factorization
// and updates u -= M'*delta_lam to cancel the residual exactly:
//   M*(u - M'*delta_lam) = M*u - M*M'*delta_lam = M*u - r = d.
void daqp_refine_active(DAQPWorkspace *work){
    int i, j, disp, id;
    c_float Mu, d;

    // Refinement uses xldl and zldl as scratch, invalidating the cached CSP
    // forward substitution independently of whether the active set changes.
    work->reuse_ind = 0;

    // Compute r[i] = M_i*u - d_i and store in xldl[i].
    for(i = 0; i < work->n_active; i++){
        id = work->WS[i];
        if(id < work->ms){
            if(work->Rinv != NULL){
                Mu = 0;
                for(j=id, disp=DAQP_R_OFFSET(id,work->n); j<work->n; j++)
                    Mu += work->Rinv[disp+j] * work->u[j];
            } else {
                Mu = work->u[id];
            }
        } else {
            Mu = 0;
            for(j=0, disp=work->n*(id-work->ms); j<work->n; j++)
                Mu += work->M[disp++] * work->u[j];
        }
        d = DAQP_IS_LOWER(id) ? work->dlower[id] : work->dupper[id];
        work->xldl[i] = Mu - d; // RHS: +r[i] (positive, so L*D*L'*dlam=r gives u-=M'*dlam zeroes residual)
        // For a nonzero soft slack the CSP system has a diagonal
        // reciprocal-weight term. Account for it (and for the linear weight)
        // when forming the refinement residual.
        if(DAQP_IS_SOFT(id) && DAQP_IS_SLACK_FREE(id))
            work->xldl[i] -= daqp_soft_residual(work,id,work->lam_star[i]);
    }

    daqp_solve_working_set(work); // xldl = delta_lam

    // Update lam_star += delta_lam and u -= M'*delta_lam.
    for(i=0; i<work->n_active; i++)
        work->lam_star[i] += work->xldl[i];
    daqp_sub_working_set_rows(work,work->xldl,work->u);

    // Recompute fval since both u and lam_star changed
    c_float fval = 0;
    for(i=0; i<work->n_active; i++){
        id = work->WS[i];
        if(DAQP_IS_SOFT(id)) fval += daqp_soft_penalty(work,id,work->lam_star[i]);
    }
    for(j=0; j<work->n; j++)
        fval += work->u[j] * work->u[j];
    work->fval = fval;
}

// Residual of active constraint id at x: A_id x - b_id (b the active side)
static c_float daqp_active_residual(const DAQPWorkspace *work, const int id){
    const DAQPProblem* qp = work->qp;
    const int n = work->n, ms = work->ms;
    c_float val;
    int j;
    if(id < ms) val = work->x[id];
    else{
        const c_float* a = qp->A+(size_t)(id-ms)*n;
        for(j = 0, val = 0; j < n; j++) val += a[j]*work->x[j];
    }
    return val - (DAQP_IS_LOWER(id) ? qp->blower[id] : qp->bupper[id]);
}

// y <-- Rinv*y (undoing the normalization of the simple bounds)
void daqp_apply_Rinv(DAQPWorkspace *work, c_float* y){
    int i, j, disp;
    const int n = work->n;
    if(work->Rinv != NULL){
        for(i = 0, disp = 0; i < n; i++){
            c_float sum = y[i]*work->Rinv[disp++];
            for(j = i+1; j < n; j++) sum += work->Rinv[disp++]*y[j];
            y[i] = sum;
        }
        // Generated workspaces store a normalized Rinv without the scaling array
        if((work->state & DAQP_STATE_RINV_NORMALIZED) && work->scaling != NULL)
            for(i = 0; i < work->ms; i++) y[i] /= work->scaling[i];
    }
    else if(work->RinvD != NULL)
        for(i = 0; i < n; i++) y[i] *= work->RinvD[i];
}

// One step of iterative refinement of x on the active constraints, done
// directly in x to avoid the cancellation in x = Rinv*(u-v)
void daqp_refine_primal(DAQPWorkspace *work){
    int i, j, id;
    const int n = work->n, na = work->n_active;
    const DAQPProblem* qp = work->qp;
    c_float *r = work->xldl, *du = work->zldl, dfval = 0;
    if(na == 0 || qp == NULL || work->sing_ind != DAQP_EMPTY_IND) return;

    // Skip if x is accurate: a well-conditioned (unregularized) Hessian and rounding
    // errors (about DAQP_REFINE_GAIN*eps*max(|u|,|v|)/min(D), with |u|^2 = fval >= |d_W|^2)
    // below primal_tol
    if(!(work->state & DAQP_STATE_ILL_CONDITIONED) && work->n_prox == 0){
        c_float scale2 = work->fval, min_D = 1, err;
        for(i = 0; i < na; i++) if(work->D[i] < min_D) min_D = work->D[i];
        if(work->v != NULL)
            for(i = 0; i < n; i++) if(work->v[i]*work->v[i] > scale2) scale2 = work->v[i]*work->v[i];
        err = DAQP_REFINE_GAIN*(sizeof(c_float) == sizeof(float) ? FLT_EPSILON : DBL_EPSILON);
        if(err*err*scale2 <= work->settings->primal_tol*work->settings->primal_tol*min_D*min_D) return;
    }
    for(i = 0; i < na; i++) if(DAQP_IS_SOFT(work->WS[i])) return;

    // xldl and zldl are used as scratch
    work->reuse_ind = 0;

    // r = S*(A_W x - b_W)
    for(i = 0; i < na; i++){
        id = work->WS[i];
        const c_float val = daqp_active_residual(work,id);
        dfval += work->lam_star[i]*val;
        r[i] = val*(work->scaling != NULL ? work->scaling[id] : 1);
    }

    daqp_solve_working_set(work); // r = dlam

    // du = -M_W'*dlam
    for(j = 0; j < n; j++) du[j] = 0;
    daqp_sub_working_set_rows(work,r,du);
    for(j = 0; j < n; j++) dfval += 0.5*du[j]*du[j];

    daqp_apply_Rinv(work,du); // dx
    for(i = 0; i < n; i++) work->x[i] += du[i];

    for(i = 0; i < na; i++)
        work->lam_star[i] += r[i]*(work->scaling != NULL ? work->scaling[work->WS[i]] : 1);
    work->fval += 2*dfval; // fval is twice the objective function value
}
