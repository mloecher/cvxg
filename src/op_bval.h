#ifndef CVX_OPBVAL_H
#define CVX_OPBVAL_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "cvx_matrix.h"

typedef struct {
    int active;
    int verbose;

    int N;
    int ind_inv;
    double dt;

    double mod;

    double d_norm;
    double weight;

    cvx_mat Binit;
    cvx_mat B0;
    cvx_mat B;
    cvx_mat sigBdenom;
    cvx_mat sigB;

    cvx_mat Btau;
    cvx_mat Bvaltemp;

    cvx_mat zB;
    cvx_mat zBbuff;
    cvx_mat zBbar;
    cvx_mat Bx;

    cvx_mat norm_helper;

} cvxop_bval;

void cvxop_bval_init(cvxop_bval *opB, int N, int ind_inv, double dt, double init_weight, int verbose);
void cvxop_bval_add2tau(cvxop_bval *opB, cvx_mat *tau_mat);
void cvxop_bval_add2taumx(cvxop_bval *opB, cvx_mat *taumx);
void cvxop_bval_update(cvxop_bval *opB, cvx_mat *txmx, double relax);
double cvxop_bval_getbval(cvxop_bval *opB, cvx_mat *G, cvx_mat *tau);
void cvxop_bval_destroy(cvxop_bval *opB);
void cvxop_bval_reweight(cvxop_bval *opB, double weight_mod);


#endif /* CVX_OPBVAL_H */