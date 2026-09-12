#ifndef DAQP_CONSTANTS_H
#define DAQP_CONSTANTS_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

#include <stddef.h>

#define DAQP_EMPTY_IND -1
#define DAQP_UNCONSTRAINED_OPTIMAL -2
#define DAQP_INF ((c_float)1e30)

// DEFAULT SETTINGS
#define DAQP_DEFAULT_PRIM_TOL 1e-6
#define DAQP_DEFAULT_DUAL_TOL 1e-12
#define DAQP_DEFAULT_ZERO_TOL 1e-11
#define DAQP_DEFAULT_PROG_TOL 1e-14
#define DAQP_DEFAULT_LP_PROG_TOL 1e-10
#define DAQP_DEFAULT_PIVOT_TOL 1e-6
#define DAQP_DEFAULT_CYCLE_TOL 10
#define DAQP_DEFAULT_ETA 1e-6
#define DAQP_DEFAULT_ITER_LIMIT 10000
#define DAQP_DEFAULT_RHO_SOFT 1e-6
#define DAQP_DEFAULT_REL_SUBOPT 0
#define DAQP_DEFAULT_ABS_SUBOPT 0
#define DAQP_DEFAULT_SING_TOL (3.7e-11)
#define DAQP_DEFAULT_REFACTOR_TOL 1e-9
#define DAQP_DEFAULT_EPS_PROX 1e-6


// MACROS
#define DAQP_ARSUM(x) ((x)*(x+1)/2)
#define DAQP_R_OFFSET(X,Y) (((2*Y-X-1)*X)/2)

// EXIT FLAGS
#define DAQP_EXIT_SOFT_OPTIMAL 2
#define DAQP_EXIT_OPTIMAL 1
#define DAQP_EXIT_INFEASIBLE -1
#define DAQP_EXIT_CYCLE -2
#define DAQP_EXIT_UNBOUNDED -3
#define DAQP_EXIT_ITERLIMIT -4
#define DAQP_EXIT_NONCONVEX -5
#define DAQP_EXIT_OVERDETERMINED_INITIAL -6
#define DAQP_EXIT_TIMELIMIT -7

// UPDATE LDP MASKS
#define DAQP_UPDATE_Rinv 1
#define DAQP_UPDATE_M 2
#define DAQP_UPDATE_v 4
#define DAQP_UPDATE_d 8
#define DAQP_UPDATE_sense 16
#define DAQP_UPDATE_hierarchy 32
#define DAQP_UPDATE_unconstrained 64

// CONSTRAINT MASKS
#define DAQP_ACTIVE 1
#define DAQP_IS_ACTIVE(x) (work->sense[x]&1)
#define DAQP_SET_ACTIVE(x) (work->sense[x]|=1)
#define DAQP_SET_INACTIVE(x) (work->sense[x]&=~1)

// marks if a constraints is active at its lower bound
#define DAQP_LOWER 2
#define DAQP_IS_LOWER(x) (work->sense[x]&2)
#define DAQP_SET_LOWER(x) (work->sense[x]|=2)
#define DAQP_SET_UPPER(x) (work->sense[x]&=~2)

// marks if a constraint cannot be activated/deactivated
#define DAQP_IMMUTABLE 4
#define DAQP_IS_IMMUTABLE(x) (work->sense[x]&4)
#define DAQP_SET_IMMUTABLE(x) (work->sense[x]|=4)
#define DAQP_SET_MUTABLE(x) (work->sense[x]&=~4)

// marks that a constraint might be violated (but the slack is penalized)
#define DAQP_SOFT 8
#define DAQP_IS_SOFT(x) (work->sense[x]&8)
#define DAQP_SET_SOFT(x) (work->sense[x]|=8)
#define DAQP_SET_HARD(x) (work->sense[x]&=~8)

// marks that a constraint has to be active at either its upper or lower bound
#define DAQP_BINARY 16
#define DAQP_IS_BINARY(x) (work->sense[x]&16)

// marks that the soft slack is at its lower bound (d_ls or d_us)
#define DAQP_SLACK_FIXED 32
#define DAQP_IS_SLACK_FIXED(x) (work->sense[x]&32)
#define DAQP_IS_SLACK_FREE(x) ((work->sense[x]&32)==0)
#define DAQP_SET_SLACK_FIXED(x) (work->sense[x]|=32)
#define DAQP_SET_SLACK_FREE(x) (work->sense[x]&=~32)

# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif //ifndef DAQP_CONSTANTS_H
