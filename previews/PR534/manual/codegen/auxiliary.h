#ifndef DAQP_AUX_H
# define DAQP_AUX_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

#include "types.h"
#include "constants.h"

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

void daqp_remove_constraint(DAQPWorkspace* work, const int rm_ind);
void daqp_add_constraint(DAQPWorkspace *work, const int add_ind, c_float lam);
void daqp_compute_primal_and_fval(DAQPWorkspace *work);
// Slack of the soft constraint active at working set index i, and the largest
// slack among them (in the units of the original problem)
c_float daqp_soft_slack(DAQPWorkspace *work, const int i);
c_float daqp_max_soft_slack(DAQPWorkspace *work);
int daqp_set_noise_floor(DAQPWorkspace *work);
c_float daqp_noise_floor(DAQPWorkspace *work);
void daqp_compute_Mu(DAQPWorkspace *work);
int daqp_add_infeasible(DAQPWorkspace *work);
int daqp_remove_blocking(DAQPWorkspace *work);
int daqp_implied_last(DAQPWorkspace *work);
void daqp_compute_CSP(DAQPWorkspace *work);
void daqp_refine_active(DAQPWorkspace *work);
void daqp_refine_primal(DAQPWorkspace *work);
void daqp_solve_working_set(DAQPWorkspace *work);
void daqp_sub_working_set_rows(DAQPWorkspace *work, const c_float* dlam, c_float* y);
void daqp_refine_pivot(DAQPWorkspace *work, const int add_ind, const int L_start);
void daqp_apply_Rinv(DAQPWorkspace *work, c_float* y);
void daqp_compute_singular_direction(DAQPWorkspace *work);
int daqp_inconsistent_dual(DAQPWorkspace *work);

void daqp_pivot_last(DAQPWorkspace *work);

int daqp_drop_singular_last(DAQPWorkspace *work);
int daqp_activate_constraints(DAQPWorkspace *work);
int daqp_reactivate_constraints(DAQPWorkspace *work);
void daqp_deactivate_constraints(DAQPWorkspace *work);

# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif //ifndef DAQP_AUX_H
