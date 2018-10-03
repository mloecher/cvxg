#ifndef CVX_OPBETA_H
#define CVX_OPBETA_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "cvx_matrix.h"

typedef struct {
    int active;
    int verbose;

    int N;
    double dt;

    cvx_mat C;
} cvxop_beta;

void cvxop_beta_init(cvxop_beta *opC, int N, double dt, int verbose);
void cvxop_beta_add2taumx(cvxop_beta *opC, cvx_mat *taumx);
void cvxop_beta_destroy(cvxop_beta *opC);

#endif /* CVX_OPBETA_H */