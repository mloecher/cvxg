#include "op_moments.h"

void cvxop_moments_init(cvxop_moments *opQ, int N, int ind_inv, double dt,
                        double m0_tol, double m1_tol, double m2_tol, int verbose)
{
    opQ->active = 1;
    opQ->N = N;
    opQ->ind_inv = ind_inv;
    opQ->dt = dt;
    opQ->verbose = verbose;

    opQ->Nm = 3; // Hard code maximum number of moments for now

    cvxmat_alloc(&opQ->Q, N, opQ->Nm);
    cvxmat_alloc(&opQ->moment_tol, opQ->Nm, 1);

    cvxmat_alloc(&opQ->norms, opQ->Nm, 1);
    cvxmat_alloc(&opQ->sigQ, opQ->Nm, 1);

    cvxmat_alloc(&opQ->zQ, opQ->Nm, 1);
    cvxmat_alloc(&opQ->zQbuff, opQ->Nm, 1);
    cvxmat_alloc(&opQ->zQbar, opQ->Nm, 1);
    cvxmat_alloc(&opQ->Qx, opQ->Nm, 1);


    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = 0; i < N; i++) {
            double temp = pow((dt*i), (double)j);
            cvxmat_set(&(opQ->Q), i, j, temp);
        }
    }

    double gamma = 42.58e3;

    // for (int j = 0; j < opQ->Nm; j++) {
    //     for (int i = 0; i < N; i++) {
    //         double temp = cvxmat_get(&(opQ->Q), i, j) * gamma * dt;
    //         cvxmat_set(&(opQ->Q), i, j, temp);
    //     }
    // }

    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = ind_inv; i < N; i++) {
            double temp = cvxmat_get(&(opQ->Q), i, j) * -1.0;
            cvxmat_set(&(opQ->Q), i, j, temp);
        }
    }


    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = 0; i < N; i++) {
            double temp = cvxmat_get(&(opQ->Q), i, j);
            opQ->norms.vals[j] += temp*temp;
        }
    }

    for (int j = 0; j < opQ->Nm; j++) {
        opQ->norms.vals[j] = sqrt(opQ->norms.vals[j]);
    }

    if (opQ->verbose>0) {   
        printf("Q norms = %.2e  %.2e  %.2e\n", opQ->norms.vals[0], opQ->norms.vals[1], opQ->norms.vals[2]);
    }

    // Relative weighting of moment norms is more important than this.
    opQ->norms.vals[0] *= 0.5;

    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = 0; i < N; i++) {
            double temp = cvxmat_get(&(opQ->Q), i, j) / opQ->norms.vals[j];
            cvxmat_set(&(opQ->Q), i, j, temp);
        }
    }

    opQ->moment_tol.vals[0] = m0_tol/opQ->norms.vals[0];
    opQ->moment_tol.vals[1] = m1_tol/opQ->norms.vals[1];
    opQ->moment_tol.vals[2] = m2_tol/opQ->norms.vals[2];

    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = 0; i < N; i++) {
            double temp = cvxmat_get(&(opQ->Q), i, j);
            opQ->sigQ.vals[j] += fabs(temp);
        }
    }

    for (int j = 0; j < opQ->Nm; j++) {
        opQ->sigQ.vals[j] = 1.0 / opQ->sigQ.vals[j];
    }
    
    if (opQ->verbose>0) {   
        printf("sigQ = %.2e  %.2e  %.2e\n", opQ->sigQ.vals[0], opQ->sigQ.vals[1], opQ->sigQ.vals[2]);
    }

}

void cvxop_moments_add2tau(cvxop_moments *opQ, cvx_mat *tau_mat)
{
    for (int j = 0; j < opQ->Nm; j++) {
        if (opQ->moment_tol.vals[j] >= 0) {
            for (int i = 0; i < opQ->N; i++) {
                double temp = cvxmat_get(&(opQ->Q), i, j);
                tau_mat->vals[i] += fabs(temp);
            }
        }
    }
}

void cvxop_moments_add2taumx(cvxop_moments *opQ, cvx_mat *taumx)
{
    for (int j = 0; j < opQ->Nm; j++) {
        if (opQ->moment_tol.vals[j] >= 0) {
            for (int i = 0; i < opQ->N; i++) {
                double temp = cvxmat_get(&(opQ->Q), i, j);
                taumx->vals[i] += (temp * opQ->zQ.vals[j]);
            }
        }
    }

}


void cvxop_moments_update(cvxop_moments *opQ, cvx_mat *txmx, double relax)
{

    cvxmat_setvals(&(opQ->Qx), 0.0);

    // zDbuff  = zD + sigD.*(D*txmx);
    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = 0; i < opQ->N; i++) {
            double temp = cvxmat_get(&(opQ->Q), i, j) * txmx->vals[i];
            opQ->Qx.vals[j] += temp;
        }
    }

    for (int j = 0; j < opQ->Nm; j++) {
        opQ->Qx.vals[j] *= opQ->sigQ.vals[j];
    }

    for (int j = 0; j < opQ->Nm; j++) {
        opQ->zQbuff.vals[j] = opQ->zQ.vals[j] + opQ->Qx.vals[j];
    }

    // zDbar = zDbuff - sigD.*min(SRMAX,max(-SRMAX,zDbuff./sigD));
    // Use zDbar as temp storage for the second half of the above equation
    for (int i = 0; i < opQ->Nm; i++) {
        double temp = opQ->zQbuff.vals[i];
        if (temp/opQ->sigQ.vals[i] > 0.99*opQ->moment_tol.vals[i]) {
            opQ->zQbar.vals[i] = opQ->sigQ.vals[i]*opQ->moment_tol.vals[i];
        } else if (temp/opQ->sigQ.vals[i] < -0.99*opQ->moment_tol.vals[i]) {
            opQ->zQbar.vals[i] = -opQ->sigQ.vals[i]*opQ->moment_tol.vals[i];
        } else {
            opQ->zQbar.vals[i] = temp;
        }
    }

    for (int i = 0; i < opQ->Nm; i++) {
        opQ->zQbar.vals[i] = opQ->zQbuff.vals[i] - opQ->zQbar.vals[i];
    }

    // zD=p*zDbar+(1-p)*zD;
    for (int i = 0; i < opQ->Nm; i++) {
        opQ->zQ.vals[i] = relax * opQ->zQbar.vals[i] + (1 - relax) * opQ->zQ.vals[i];
    }
}


int cvxop_moments_check(cvxop_moments *opQ, cvx_mat *G)
{
    cvxmat_setvals(&(opQ->Qx), 0.0);

    // zDbuff  = zD + sigD.*(D*txmx);
    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = 0; i < opQ->N; i++) {
            double temp = cvxmat_get(&(opQ->Q), i, j) * G->vals[i];
            opQ->Qx.vals[j] += temp;
        }
    }

    int moments_bad = 0;
    double moment_tol = 1.0e-2;

    for (int j = 0; j < opQ->Nm; j++) {
        if (opQ->moment_tol.vals[j] >= 0) {
            if (fabs(opQ->Qx.vals[0]) > moment_tol) {
                moments_bad = 1;
            }
        }
    }

    if (opQ->verbose>0) {   
        printf("    Moments check:  (%d)  %.2e  %.2e  %.2e\n", moments_bad, opQ->Qx.vals[0], opQ->Qx.vals[1], opQ->Qx.vals[2]);\
    }

    return moments_bad;
}

/*
int main (void)
{
    printf ("In op_moments.c main function\n");
    
    cvxop_moments opQ;
    cvxop_moments_init(&opQ, 40, 24, 0.1e-3,
                        0.0, 0.0, 0.0);

    printf("\n\n");
    for (int i = 0; i < opQ.N; i++) {
        printf("%.2e  ", opQ.Q0.vals[i]);
    }
    printf("\n\n");
    for (int i = 0; i < opQ.N; i++) {
        printf("%.2e  ", opQ.Q1.vals[i]);
    }
    printf("\n\n");
    for (int i = 0; i < opQ.N; i++) {
        printf("%.2e  ", opQ.Q2.vals[i]);
    }


    return 0;
}
*/


void cvxop_moments_destroy(cvxop_moments *opQ)
{
    free(opQ->moment_tol.vals);
    free(opQ->norms.vals);
    free(opQ->Q.vals);
    free(opQ->sigQ.vals);
    free(opQ->zQ.vals);
    free(opQ->zQbuff.vals);
    free(opQ->zQbar.vals);
    free(opQ->Qx.vals);
}