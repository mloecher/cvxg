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
    if (start < 0) {start = 0;}
    if (end > (opG->Gfix.N-1)) {end = (opG->Gfix.N-1);}
    for (int i = start; i <= end; i++) {
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
            G->vals[i] = -0.01 * opG->gmax;
        } else {
            G->vals[i] = 0.01 * opG->gmax;
        }
    }

}

int cvxop_gradient_check(cvxop_gradient *opG, cvx_mat *G)
{

    int grad_too_high = 0;
    int grad_bad = 0;

    double norm0 = 0.0;
    for (int i = 0; i < G->N; i++) {
        norm0 += G->vals[i] * G->vals[i];
        if (fabs(G->vals[i]) > opG->gmax) {
            grad_too_high++;
            grad_bad = 1;
        }
    }
    norm0 = sqrt(norm0);
    opG->d_norm = norm0;

    
    if (opG->verbose>0) {  
        printf("    Gradient check: (%d)  %d      norm = %.2e \n", grad_bad, grad_too_high, norm0);
    }

    return grad_bad;
}

double cvxop_gradient_getbval(cvxop_gradient *opG, cvx_mat *G)
{
    double Gt = 0;
    double bval = 0;
    double mod = 71576597699.4529; // (GAMMA*2*pi)^2


    for (int i = 0; i < opG->N; i++) {
        if (i < opG->ind_inv) {
            Gt += G->vals[i];
        } else {
            Gt -= G->vals[i];
        }
        bval += Gt*Gt;
    }    
    bval *= (mod * opG->dt * opG->dt * opG->dt);

    return bval;
}   

void cvxop_gradient_destroy(cvxop_gradient *opG)
{
    free(opG->Gfix.vals);
}