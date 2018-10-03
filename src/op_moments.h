#ifndef CVX_OPMOMENTS_H
#define CVX_OPMOMENTS_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "cvx_matrix.h"

typedef struct {
    int active;
    int verbose;

    int N;
    int Nm;
    int ind_inv;
    double dt;

    cvx_mat moment_tol;
    cvx_mat norms;
    cvx_mat Q;
    cvx_mat sigQ;
    cvx_mat zQ;
    cvx_mat zQbuff;
    cvx_mat zQbar;
    cvx_mat Qx;

} cvxop_moments;


void cvxop_moments_init(cvxop_moments *opQ, int N, int ind_inv, double dt,
                        double m0_tol, double m1_tol, double m2_tol, int verbose);

void cvxop_moments_add2tau(cvxop_moments *opQ, cvx_mat *tau_mat);
void cvxop_moments_add2taumx(cvxop_moments *opQ, cvx_mat *taumx);
void cvxop_moments_update(cvxop_moments *opQ, cvx_mat *txmx, double relax);
int cvxop_moments_check(cvxop_moments *opQ, cvx_mat *G);
void cvxop_moments_destroy(cvxop_moments *opQ);

#endif /* CVX_OPMOMENTS_H */