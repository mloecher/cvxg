#include "op_bval.h"

void cvxop_bval_init(cvxop_bval *opB, int N, int ind_inv, double dt, int verbose) {
    opB->active = 1;
    opB->N = N;
    opB->dt = dt;
    opB->ind_inv = ind_inv;
    opB->verbose = verbose;

    cvxmat_alloc(&opB->B, N, N);
    cvxmat_alloc(&opB->Binit, N, N);
    cvxmat_alloc(&opB->sigBdenom, N, 1);
    cvxmat_alloc(&opB->sigB, N, 1);

    cvxmat_alloc(&opB->Bvaltemp, N, 1);

    cvxmat_alloc(&opB->Btau, N, 1);

    cvxmat_alloc(&opB->zB, N, 1);
    cvxmat_alloc(&opB->zBbuff, N, 1);
    cvxmat_alloc(&opB->zBbar, N, 1);
    cvxmat_alloc(&opB->Bx, N, 1);

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            
            if (i >= j) {
                if (j >= ind_inv) {
                    cvxmat_set(&(opB->Binit), i, j, -dt); 
                } else {
                    cvxmat_set(&(opB->Binit), i, j, dt); 
                }
            }

        }
    }

    cvxmat_multAtA(&opB->B, &opB->Binit);

    for (int i = 0; i < opB->B.N; i++) {
        opB->B.vals[i] *= 1.0;
    }

    double sum;
    for (int j = 0; j < N; j++) {
        sum = 0.0;
        for (int i = 0; i < N; i++) {
            double temp = cvxmat_get(&(opB->B), i, j);
            sum += fabs(temp);
        }
        opB->sigBdenom.vals[j] = sum;
    }

    for (int i = 0; i < N; i++) {
        opB->sigB.vals[i] = 1.0/opB->sigBdenom.vals[i];
    }

    // for (int j = 0; j < N; j++) {
    //     printf("\n row %d \n", j);
    //     for (int i = 0; i < N; i++) {
            
    //             double temp = cvxmat_get(&(opB->B), j, i); 
    //             printf("%.2e  ", temp);

    //     }
    // }
}

void cvxop_bval_add2tau(cvxop_bval *opB, cvx_mat *tau_mat)
{
    if (opB->active > 0) {
        for (int i = 0; i < opB->N; i++) {
            tau_mat->vals[i] += fabs(opB->sigBdenom.vals[i]);
        }
    }
}

void cvxop_bval_add2taumx(cvxop_bval *opB, cvx_mat *taumx)
{
    if (opB->active > 0) {
        
        cvxmat_setvals(&(opB->Btau), 0.0);
        cvxmat_multAx2(&opB->Btau, &opB->B, &opB->zB);

        for (int i = 1; i < taumx->N; i++) {
            taumx->vals[i] -= opB->Btau.vals[i];
        }

    }
}


void cvxop_bval_update(cvxop_bval *opB, cvx_mat *txmx, double relax)
{
    if (opB->active > 0) {
        cvxmat_setvals(&(opB->Bx), 0.0);
        cvxmat_multAx2(&opB->Bx, &opB->B, txmx);

        // zBbuff  = zB + sigB.*(B*txmx);

        for (int i = 0; i < opB->Bx.N; i++) {
            opB->Bx.vals[i] *= opB->sigB.vals[i];
        }

        for (int i = 0; i < opB->zBbuff.N; i++) {
            opB->zBbuff.vals[i] = opB->zB.vals[i] + opB->Bx.vals[i];
        }

        // zD=p*zDbar+(1-p)*zD;
        for (int i = 0; i < opB->zB.N; i++) {
            opB->zB.vals[i] = relax * opB->zBbuff.vals[i] + (1 - relax) * opB->zB.vals[i];
        }
    }
}

double cvxop_bval_getbval(cvxop_bval *opB, cvx_mat *G)
{
    double mod = 71576597699.4529; // (GAMMA*2*pi)^2
    double bval = 0.0;
    cvxmat_multAx(&opB->Bvaltemp, &opB->B, G);

    for (int i = 0; i < opB->Bvaltemp.N; i++) {
        bval += G->vals[i] * opB->Bvaltemp.vals[i] * mod * opB->dt;
    }

    return bval;


}


void cvxop_bval_destroy(cvxop_bval *opB)
{
    free(opB->Binit.vals);
    free(opB->B.vals);
    free(opB->sigBdenom.vals);
    free(opB->sigB.vals);

    free(opB->Btau.vals);
    free(opB->Bvaltemp.vals);

    free(opB->zB.vals);
    free(opB->zBbuff.vals);
    free(opB->zBbar.vals);
    free(opB->Bx.vals);
}
