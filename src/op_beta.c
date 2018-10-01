#include "op_beta.h"

void cvxop_beta_init(cvxop_beta *opC, int N, double dt, int verbose) {
    opC->active = 1;
    opC->N = N;
    opC->dt = dt;
    opC->verbose = verbose;

    cvxmat_alloc(&opC->C, N, 1);

    double tt;
    for (int i = 0; i < N; i++) {
        tt = N-i;
        opC->C.vals[i] = tt*(tt+1)/2.0;
    }

    double norm = 0.0;

    for (int i = 0; i < N; i++) {
        norm += opC->C.vals[i] * opC->C.vals[i];
    }

    norm = sqrt(norm);

    for (int i = 0; i < N; i++) {
        opC->C.vals[i] /= norm;
    }
}

void cvxop_beta_add2taumx(cvxop_beta *opC, cvx_mat *taumx)
{
    if (opC->active > 0) {

        for (int i = 1; i < taumx->N; i++) {
            taumx->vals[i] -= opC->C.vals[i];
        }

    }
}