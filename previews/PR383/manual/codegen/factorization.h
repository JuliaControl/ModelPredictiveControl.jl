#ifndef DAQP_FACTORIZATION_H
# define DAQP_FACTORIZATION_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

#include "types.h"
#include "constants.h"

c_float daqp_dot(const c_float* v1, const c_float* v2, const int n);
void daqp_update_LDL_add(DAQPWorkspace *work, const int add_ind);
void daqp_update_LDL_remove(DAQPWorkspace *work, const int rm_ind);

# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif //ifndef DAQP_FACTORIZATION_H
