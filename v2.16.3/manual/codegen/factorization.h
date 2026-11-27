#ifndef DAQP_FACTORIZATION_H
# define DAQP_FACTORIZATION_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

#include "types.h"
#include "constants.h"

static inline c_float daqp_dot_inline(const c_float* v1, const c_float* v2, const int n) {
    c_float sum = 0.0;
    for (int i = 0; i < n; i++) sum += v1[i] * v2[i];
    return sum;
}
// y <-- M*x for the row-major rows x n matrix M, or, if y2 is not NULL, both
// y and y2 are incremented by M*x. Four rows are accumulated together to
// reuse x and expose independent accumulation chains.
static inline void daqp_M_times(const c_float *M, const c_float *x, const int rows, const int n,
        c_float *y, c_float *y2){
    int row = 0;
    for(; row+3 < rows; row+=4){
        const c_float *m0 = M+(size_t)(row+0)*n;
        const c_float *m1 = M+(size_t)(row+1)*n;
        const c_float *m2 = M+(size_t)(row+2)*n;
        const c_float *m3 = M+(size_t)(row+3)*n;
        c_float sum0=0, sum1=0, sum2=0, sum3=0;
        for(int k=0; k<n; k++){
            const c_float xk = x[k];
            sum0 += m0[k]*xk;
            sum1 += m1[k]*xk;
            sum2 += m2[k]*xk;
            sum3 += m3[k]*xk;
        }
        if(y2 == NULL){
            y[row+0] = sum0; y[row+1] = sum1;
            y[row+2] = sum2; y[row+3] = sum3;
        }
        else{
            y[row+0] += sum0; y[row+1] += sum1;
            y[row+2] += sum2; y[row+3] += sum3;
            y2[row+0] += sum0; y2[row+1] += sum1;
            y2[row+2] += sum2; y2[row+3] += sum3;
        }
    }
    for(; row<rows; row++){
        const c_float s = daqp_dot_inline(M+(size_t)row*n,x,n);
        if(y2 == NULL) y[row] = s;
        else{ y[row] += s; y2[row] += s; }
    }
}
// rho is the reciprocal quadratic weight for a free soft slack, zero otherwise
void daqp_update_LDL_add(DAQPWorkspace *work, const int add_ind, const c_float rho);
void daqp_update_LDL_remove(DAQPWorkspace *work, const int rm_ind);

# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif //ifndef DAQP_FACTORIZATION_H
