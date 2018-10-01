#include "op_gradient.h"

void cvxop_gradient_init(cvxop_gradient *opG, int N, double dt, double gmax, int ind_inv, int verbose) {
    opG->active = 1;
    opG->N = N;
    opG->dt = dt;
    opG->gmax = gmax;
    opG->ind_inv = ind_inv;
    opG->verbose = verbose;
    

    cvxmat_alloc(&opG->Gfix, N, 1);

    for (int i = 0; i < N; i++) {
        opG->Gfix.vals[i] = -999999.0;
    }
    opG->Gfix.vals[0] = 0.0;
    opG->Gfix.vals[N-1] = 0.0;
}

void cvxop_gradient_setFixRange(cvxop_gradient *opG, int start, int end, double val)
{
    if (end > opG->Gfix.N) {end = opG->Gfix.N;}
    for (int i = start; i < end; i++) {
        opG->Gfix.vals[i] = val;
    }
}


void cvxop_gradient_limiter(cvxop_gradient *opG, cvx_mat *xbar)
{
    for (int i = 0; i < xbar->N; i++) {
        if (opG->Gfix.vals[i] > -9999.0) {
            xbar->vals[i] = opG->Gfix.vals[i];
        }
    }

    for (int i = 0; i < xbar->N; i++) {
        if (xbar->vals[i] > 0.99*opG->gmax) {
            xbar->vals[i] = 0.99*opG->gmax;
        } else if (xbar->vals[i] < -0.99*opG->gmax) {
            xbar->vals[i] = -0.99*opG->gmax;
        }
    }

}

void cvxop_init_G(cvxop_gradient *opG, cvx_mat *G)
{
    for (int i = 0; i < G->N; i++) {
        if (i <= opG->ind_inv) {
            G->vals[i] = 0.5 * opG->gmax;
        } else {
            G->vals[i] = -0.5 * opG->gmax;
        }
    }

}

int cvxop_gradient_check(cvxop_gradient *opG, cvx_mat *G)
{

    int grad_too_high = 0;
    int grad_bad = 0;

    for (int i = 0; i < G->N; i++) {
        if (fabs(G->vals[i]) > opG->gmax) {
            grad_too_high++;
            grad_bad = 1;
        }
    }
    
    if (opG->verbose>0) {  
        printf("    Gradient check: (%d)  %d\n", grad_bad, grad_too_high);
    }

    return grad_bad;
}