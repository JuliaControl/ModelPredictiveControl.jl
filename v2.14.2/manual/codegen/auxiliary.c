#include "auxiliary.h"
#include "factorization.h"

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

// The weights are indexed by the original problem, which is renumbered if
// equalities have been eliminated (work->scaling always refers to the
// problem that is currently installed)
static inline int daqp_soft_ind(DAQPWorkspace *work, const int id){
    return (work->eq != NULL && work->eq->installed) ? work->eq->map[id] : id;
}

// Reciprocal quadratic weight of the active side of constraint id (zero
// selects settings->rho_soft, which is given in the normalized formulation)
static inline c_float daqp_soft_rho(DAQPWorkspace *work, const int id){
#ifdef DAQP_SOFT_WEIGHTS
    if(work->rho_ls != NULL){
        const int i = daqp_soft_ind(work,id);
        const c_float rho = DAQP_IS_LOWER(id) ? work->rho_ls[i] : work->rho_us[i];
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
        const int i = daqp_soft_ind(work,id);
        const c_float w = DAQP_IS_LOWER(id) ? work->w_ls[i] : work->w_us[i];
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
    // Mark whether the slack is zero, given the multiplier
    if(DAQP_IS_SOFT(add_ind)){
        DAQP_SET_MUTABLE(add_ind);
        const c_float w = daqp_soft_w(work,add_ind);
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
        if(DAQP_IS_SOFT(id))
            fval += has_l1 ? daqp_soft_penalty(work,id,li)
                : work->settings->rho_soft*li*li;
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
    // Add the constraint
    if(isupper)
        daqp_add_constraint(work,add_ind,1);
    else
        daqp_add_constraint(work,add_ind,-1);
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
/* Take the step lam <- lam + alpha*(lam_star-lam) (lam + alpha*lam_star if the
 * CSP is singular), stopping at the first multiplier that reaches zero, which
 * removes the constraint, or that passes w, which switches the state of its
 * slack. Returns 0 if the full step can be taken. */
int daqp_remove_blocking(DAQPWorkspace *work){
    int i, ind, rm_ind = DAQP_EMPTY_IND;
    const int singular = work->sing_ind != DAQP_EMPTY_IND;
    const int has_l1 = DAQP_HAS_L1(work);
    const c_float dual_tol = work->settings->dual_tol;
    c_float alpha = DAQP_INF, alpha_cand, y, ystar, p, target, rm_target = 0;

    for(i = 0; i < work->n_active; i++){
        ind = work->WS[i];
        if(DAQP_IS_IMMUTABLE(ind)) continue;
        // Fold the sign of the multiplier (dual feasibility <=> y >= 0).
        // ystar is the end point of the step, or its direction if singular.
        const int lower = DAQP_IS_LOWER(ind);
        ystar = lower ? -work->lam_star[i] : work->lam_star[i];

        if(!has_l1 || !DAQP_IS_SOFT(ind)){ // Blocked when the multiplier reaches zero
            if(ystar >= -dual_tol) continue;
            target = 0;
        }
        else{
            // The multiplier is confined to [0,w] while the slack is zero and
            // to [w,inf) otherwise; the state switches when it leaves its range
            const c_float w = daqp_soft_w(work,ind);
            const int fixed = DAQP_IS_SLACK_FIXED(ind);
            target = fixed ? 0 : w; // Blocked from below
            if(ystar >= (singular ? 0 : target) - dual_tol){
                if(!fixed || w == 0 || ystar <= (singular ? 0 : w) + dual_tol)
                    continue;
                target = w; // A zero slack is released
            }
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
    if(rm_ind == DAQP_EMPTY_IND) return 0; // Either dual feasible or primal infeasible

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
        return 1;
    }

    // The slack switches state, which only adds or removes rho on the diagonal
    const c_float lam = DAQP_IS_LOWER(ind) ? -rm_target : rm_target;
    const int release = DAQP_IS_SLACK_FIXED(ind);
    if(release) DAQP_SET_SLACK_FREE(ind);
    else DAQP_SET_SLACK_FIXED(ind);

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

void daqp_compute_CSP(DAQPWorkspace *work){
    int i,j,disp,start_disp;
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
    start_disp = DAQP_ARSUM(work->n_active)-1;
    for(i = work->n_active-1;i>=0;i--){
        sum=work->zldl[i];
        disp = start_disp--;
        for(j=work->n_active-1;j>i;j--){
            sum-=work->lam_star[j]*work->L[disp];
            disp-=j;
        }
        work->lam_star[i] = sum;
    }
    work->reuse_ind = work->n_active; // Save forward substitution information
}

//TODO this could probably be directly calculated in L
void daqp_compute_singular_direction(DAQPWorkspace *work){
    // Step direction is stored in lam_star
    int i,j,disp,offset_L= DAQP_ARSUM(work->sing_ind);
    int start_disp= offset_L-1;

    // Backwards substitution (p_tidle <-- L'\(-l))
    for(i = work->sing_ind-1;i>=0;i--){
        work->lam_star[i] = -work->L[offset_L+i];
        disp = start_disp--;
        for(j=work->sing_ind-1;j>i;j--){
            work->lam_star[i]-=work->lam_star[j]*work->L[disp];
            disp-=j;
        }
    }
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

// Activate the constraints that are marked active in sense
// Equalities are activated before inequalities
int daqp_activate_constraints(DAQPWorkspace *work){
    int i, j, first_mutable = work->m;
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

            DAQP_SET_INACTIVE(i);
            work->n_active--;
            work->sing_ind = DAQP_EMPTY_IND;
            if(work->reuse_ind > work->n_active)
                work->reuse_ind = work->n_active;

            if(dependency_residual >
                        work->settings->primal_tol*dependency_scale ||
                    dependency_residual <
                        -work->settings->primal_tol*dependency_scale)
                return DAQP_EXIT_OVERDETERMINED_INITIAL;
            // Consistent redundant equality: safely ignore.
        }
    }

    // Activate active inequalities
    for(i = first_mutable; i < work->m; i++){
        if(!DAQP_IS_ACTIVE(i) || DAQP_IS_IMMUTABLE(i)) continue;
        daqp_activate_constraint(work,i);
        if(work->sing_ind != DAQP_EMPTY_IND){
            // Leave this constraint and the remaining mutable ones inactive.
            for(j = i; j < work->m; j++){
                if(!DAQP_IS_IMMUTABLE(j)) DAQP_SET_INACTIVE(j);
            }
            work->n_active--;
            work->sing_ind = DAQP_EMPTY_IND;
            return 1;
        }
    }
    return 1;
}

// Deactivate all active constraints that are mutable (i.e., not equality constraints)
void daqp_deactivate_constraints(DAQPWorkspace *work){
    int i;
    if(work->eq != NULL) work->eq->working_set_valid = 0;
    if(work->bnb != NULL) work->bnb->n_root_WS = 0; // Also drop the BnB warm start
    for(i =0;i<work->n_active;i++){
        if(DAQP_IS_IMMUTABLE(work->WS[i])) continue;
        DAQP_SET_INACTIVE(work->WS[i]);
    }
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
    c_float sum, Mu, d;

    // Refinement uses xldl and zldl as scratch, invalidating the cached CSP
    // forward substitution independently of whether the active set changes.
    work->reuse_ind = 0;

    // Compute -r[i] = -(M_i*u - d_i) and store in xldl[i].
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

    // Forward substitution L * y = xldl
    for(i=0, disp=0; i<work->n_active; i++){
        sum = work->xldl[i];
        for(j=0; j<i; j++)
            sum -= work->L[disp++] * work->xldl[j];
        disp++; // skip stored diagonal (= 1)
        work->xldl[i] = sum;
    }

    // Scale by D^{-1}: zldl[i] = xldl[i] / D[i].
    for(i=0; i<work->n_active; i++)
        work->zldl[i] = work->xldl[i] / work->D[i];

    // Backward substitution L' * delta_lam = zldl -> stored in xldl.
    {
        int start_disp = DAQP_ARSUM(work->n_active) - 1;
        for(i=work->n_active-1; i>=0; i--){
            sum = work->zldl[i];
            disp = start_disp--;
            for(j=work->n_active-1; j>i; j--){
                sum -= work->xldl[j] * work->L[disp];
                disp -= j;
            }
            work->xldl[i] = sum; // xldl[i] = delta_lam[i]
        }
    }

    // Update lam_star += delta_lam.
    // The residual r = M*u - d was computed from the exact constraint matrix,
    for(i=0; i<work->n_active; i++)
        work->lam_star[i] += work->xldl[i];

    // Update u -= M'*delta_lam and recompute fval.
    for(i=0; i<work->n_active; i++){
        c_float dlam = work->xldl[i];
        id = work->WS[i];
        if(id < work->ms){
            if(work->Rinv != NULL){
                for(j=id, disp=DAQP_R_OFFSET(id,work->n); j<work->n; j++)
                    work->u[j] -= work->Rinv[disp+j] * dlam;
            } else {
                work->u[id] -= dlam;
            }
        } else {
            for(j=0, disp=work->n*(id-work->ms); j<work->n; j++)
                work->u[j] -= work->M[disp++] * dlam;
        }
    }

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
