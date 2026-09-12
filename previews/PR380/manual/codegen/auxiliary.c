#include "auxiliary.h"
#include "factorization.h"
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

    // Check if the removal lead to singularity (can happen due to numerics)
    if(work->n_active > 0 && work->D[work->n_active-1] < work->settings->sing_tol){
        work->sing_ind = work->n_active-1;
        work->D[work->n_active-1] = 0;
    }
    else{ // Pivot for improved numerics
        daqp_pivot_last(work);
    }
}
void daqp_add_constraint(DAQPWorkspace *work, const int add_ind, c_float lam){
    // Update data structures
    DAQP_SET_ACTIVE(add_ind);
    daqp_update_LDL_add(work, add_ind);
    work->WS[work->n_active] = add_ind;
    work->lam[work->n_active] = lam;
    work->n_active++;

    // Pivot for improved numerics
    daqp_pivot_last(work);
}

void daqp_compute_primal_and_fval(DAQPWorkspace *work){
    int i,j,disp,id;
    c_float fval=0;
    // Reset u & soft slack
    for(j=0;j<work->n;j++)
        work->u[j]=0;
    work->soft_slack = 0;
    //u[m] <-- Mk'*lam_star (zero if empty set)
    for(i=0;i<work->n_active;i++){
        id = work->WS[i];
        if(id < work->ms){
            // Simple constraint
            if(work->Rinv!=NULL){ // Hessian is not identity
                for(j=id, disp=DAQP_R_OFFSET(id,work->n);j<work->n;++j)
                    work->u[j]-=work->Rinv[disp+j]*work->lam_star[i];
            }
            else work->u[id]-=work->lam_star[i]; // Hessian is identity
        }
        else{ // General constraint
            for(j=0,disp=work->n*(id-work->ms);j<work->n;j++)
                work->u[j]-=work->M[disp++]*work->lam_star[i];
        }
        if(DAQP_IS_SOFT(id)){
            fval+= work->lam_star[i]*work->lam_star[i];
        }
    }
    // Check for progress 
    fval=fval*work->settings->rho_soft;
    work->soft_slack=fval;// XXX: keep this for now to return SOFT_OPTIMAL
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
            Mu = daqp_dot(work->Rinv+disp,work->u+j,work->n-j);
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
    for(j=work->ms, disp=0;j<work->m;j++){
        // Never activate immutable or already active constraints
        if(work->sense[j]&(DAQP_ACTIVE+DAQP_IMMUTABLE)){
            disp+=work->n;// Skip ahead in M
            continue;
        }
        Mu = daqp_dot(work->M+disp,work->u,work->n);
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
int daqp_remove_blocking(DAQPWorkspace *work){
    int i,rm_ind = DAQP_EMPTY_IND;
    c_float alpha=DAQP_INF;
    c_float alpha_cand;
    const c_float dual_tol = work->settings->dual_tol;
    for(i=0;i<work->n_active;i++){
        if(DAQP_IS_IMMUTABLE(work->WS[i])) continue;
        if(DAQP_IS_LOWER(work->WS[i])){
            if(work->lam_star[i]<dual_tol) continue; //lam <= 0 for lower -> dual feasible
        }
        else if(work->lam_star[i]>-dual_tol) continue; //lam* >= 0 for upper-> dual feasible

        if(work->sing_ind == DAQP_EMPTY_IND)
            alpha_cand= -work->lam[i]/(work->lam_star[i]-work->lam[i]);
        else
            alpha_cand= -work->lam[i]/work->lam_star[i];
        if(alpha_cand < alpha){
            alpha = alpha_cand;
            rm_ind = i;
        }
    }
    if(rm_ind == DAQP_EMPTY_IND) return 0; // Either dual feasible or primal infeasible
    // If blocking constraint -> update lambda
    if(work->sing_ind == DAQP_EMPTY_IND)
        for(i=0;i<work->n_active;i++)
            work->lam[i]+=alpha*(work->lam_star[i]-work->lam[i]);
    else
        for(i=0;i<work->n_active;i++)
            work->lam[i]+=alpha*work->lam_star[i];

    // Remove the constraint from the working set and update LDL
    work->sing_ind=DAQP_EMPTY_IND;
    daqp_remove_constraint(work,rm_ind);
    return 1;
}

void daqp_compute_CSP(DAQPWorkspace *work){
    int i,j,disp,start_disp;
    c_float sum;
    // Forward substitution (xi <-- L\d)
    for(i=work->reuse_ind,disp=DAQP_ARSUM(work->reuse_ind); i<work->n_active; i++){
        // Setup RHS
        if(DAQP_IS_LOWER(work->WS[i])){
            sum = -work->dlower[work->WS[i]];
        }
        else{
            sum = -work->dupper[work->WS[i]];
        }
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

    if(DAQP_IS_LOWER(work->WS[work->sing_ind])) //Flip to ensure descent direction
        for(i=0;i<=work->sing_ind;i++)
            work->lam_star[i] =-work->lam_star[i];
}


void daqp_pivot_last(DAQPWorkspace *work){
    const int rm_ind = work->n_active-2;
    if(work->n_active > 1 &&
            work->D[rm_ind] < work->settings->pivot_tol && // element in D small enough
            work->D[rm_ind] < work->D[work->n_active-1]){ // element in D smallar than neighbor
        const int ind_old = work->WS[rm_ind];
        // Ensure that binaries never swap order (since this order is exploited)
        if(DAQP_IS_BINARY(ind_old) && DAQP_IS_BINARY(work->WS[work->n_active-1])) return;
        if(work->bnb != NULL && rm_ind < work->bnb->n_clean) return;

        c_float lam_old = work->lam[rm_ind];
        daqp_remove_constraint(work,rm_ind); // pivot_last might be recursively called here

        if(work->sing_ind!=DAQP_EMPTY_IND) return; // Abort if D becomes singular

        daqp_add_constraint(work,ind_old,lam_old);
    }
}

// Activate constrainte that are marked active in sense
int daqp_activate_constraints(DAQPWorkspace *work){
    //TODO prioritize inequalities?
    int i;
    for(i =0;i<work->m;i++){
        if(DAQP_IS_ACTIVE(i)){
            if(DAQP_IS_LOWER(i))
                daqp_add_constraint(work,i, -1.0);
            else
                daqp_add_constraint(work,i, 1.0);
        }
        if(work->sing_ind != DAQP_EMPTY_IND){
            int exitflag = 1;
            for(;i<work->m;i++){
                // 1. Check if there are equalities that couldn't be activated
                // 2. Make sure that sense is clean for unactivated constraints
                if(DAQP_IS_ACTIVE(i)){
                    if(DAQP_IS_IMMUTABLE(i))
                        exitflag = DAQP_EXIT_OVERDETERMINED_INITIAL;
                    else
                        DAQP_SET_INACTIVE(i);
                }
            }
            // Remove the last constraint that lead to singularity
            work->n_active--;
            work->sing_ind = DAQP_EMPTY_IND;
            return exitflag;
        }
    }
    return 1;
}

// Deactivate all active constraints that are mutable (i.e., not equality constraints)
void daqp_deactivate_constraints(DAQPWorkspace *work){
    int i;
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
        // For soft constraints the CSP system is (MM'+rho*I)*lam = -d,
        if(DAQP_IS_SOFT(id))
            work->xldl[i] -= work->settings->rho_soft * work->lam_star[i];
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

    // Recompute fval = soft_slack + ||u||^2 since u changed.
    c_float fval = work->soft_slack;
    for(j=0; j<work->n; j++)
        fval += work->u[j] * work->u[j];
    work->fval = fval;
}
