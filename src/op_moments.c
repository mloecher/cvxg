#include "op_moments.h"

/**
 * Initialize the opB struct
 * This is the operator that sets
 */
void cvxop_moments_init(cvxop_moments *opQ, int N, int ind_inv, double dt,
                        double *moment_tols_in, double init_weight, int verbose)
{
    opQ->active = 1;
    opQ->N = N;
    opQ->ind_inv = ind_inv;
    opQ->dt = dt;
    opQ->verbose = verbose;
    opQ->weight = init_weight;

    opQ->Nm = 3; // Hard code maximum number of moments for now

    cvxmat_alloc(&opQ->Q0, N, opQ->Nm);
    cvxmat_alloc(&opQ->Q, N, opQ->Nm);
    cvxmat_alloc(&opQ->moment_tol0, opQ->Nm, 1);
    cvxmat_alloc(&opQ->moment_tol, opQ->Nm, 1);

    cvxmat_alloc(&opQ->norms, opQ->Nm, 1);
    cvxmat_alloc(&opQ->sigQ, opQ->Nm, 1);

    cvxmat_alloc(&opQ->zQ, opQ->Nm, 1);
    cvxmat_alloc(&opQ->zQbuff, opQ->Nm, 1);
    cvxmat_alloc(&opQ->zQbar, opQ->Nm, 1);
    cvxmat_alloc(&opQ->Qx, opQ->Nm, 1);

    cvxmat_alloc(&opQ->norm_helper, N, 1);
    cvxmat_alloc(&opQ->tau_helper, N, 1);

    // Copy moment tolerances into cvx_mat array
    for (int j = 0; j < opQ->Nm; j++) {
        opQ->moment_tol0.vals[j] = moment_tols_in[j];
    }

    // Set Q array rows to dt^momentnum
    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = 0; i < N; i++) {
            double temp = pow((dt*i), (double)j);
            cvxmat_set(&(opQ->Q0), i, j, temp);
        }
    }
    
    // Scale so that Q0 returns true moments
    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = 0; i < N; i++) {
            double temp = cvxmat_get(&(opQ->Q0), i, j) * gamma * dt;
            cvxmat_set(&(opQ->Q0), i, j, temp);
        }
    }
    
    // Gradient values need to be reversed after the 180
    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = ind_inv; i < N; i++) {
            double temp = cvxmat_get(&(opQ->Q0), i, j) * -1.0;
            cvxmat_set(&(opQ->Q0), i, j, temp);
        }
    }

    // Count the number of moment constraints that are "on"
    double active_tols = 0.0;
    for (int j = 0; j < opQ->Nm; j++) {
        if (opQ->moment_tol.vals[j] >= 0) {
            active_tols += 1.0;
        }
    }

    // Calculate the row norms of the moment array and store
    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = 0; i < N; i++) {
            double temp = cvxmat_get(&(opQ->Q0), i, j);
            opQ->norms.vals[j] += temp*temp;
        }
        opQ->norms.vals[j] = sqrt(opQ->norms.vals[j]);
        // opQ->norms.vals[j] = 1.0;
    }

    // Scale the Q0 array and copy into Q
    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = 0; i < N; i++) {
            double temp = cvxmat_get(&(opQ->Q0), i, j);
            cvxmat_set(&(opQ->Q), i, j, opQ->weight * temp / opQ->norms.vals[j]);
        }
        opQ->moment_tol.vals[j] = opQ->weight *  opQ->moment_tol0.vals[j] / opQ->norms.vals[j];

    }

    if (opQ->verbose>0) {   
        printf("Q norms = %.2e  %.2e  %.2e    active norms = %.1f\n", opQ->norms.vals[0], opQ->norms.vals[1], opQ->norms.vals[2], active_tols);
    }

    // Calculate sigQ
    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = 0; i < N; i++) {
            double temp = cvxmat_get(&(opQ->Q), i, j);
            opQ->sigQ.vals[j] += fabs(temp);
        }
        opQ->sigQ.vals[j] = 1.0 / opQ->sigQ.vals[j];
    }
    
    if (opQ->verbose > 0) {   
        printf("sigQ = %.2e  %.2e  %.2e\n", opQ->sigQ.vals[0], opQ->sigQ.vals[1], opQ->sigQ.vals[2]);
    }
}


void cvxop_moments_reweight(cvxop_moments *opQ, double weight_mod)
{
    opQ->weight *= weight_mod;

    // Calculate the row norms of the moment array and store
    for (int j = 0; j < opQ->Nm; j++) {
        opQ->norms.vals[j] = 0.0;
        for (int i = 0; i < opQ->N; i++) {
            double temp = cvxmat_get(&(opQ->Q0), i, j);
            opQ->norms.vals[j] += temp*temp;
        }
        opQ->norms.vals[j] = sqrt(opQ->norms.vals[j]);
        // opQ->norms.vals[j] = 1.0;
    }

    // Scale the Q0 array and copy into Q
    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = 0; i < opQ->N; i++) {
            double temp = cvxmat_get(&(opQ->Q0), i, j);
            cvxmat_set(&(opQ->Q), i, j, opQ->weight * temp / opQ->norms.vals[j]);
        }
        opQ->moment_tol.vals[j] = opQ->weight *  opQ->moment_tol0.vals[j] / opQ->norms.vals[j];
    }

    // Calculate sigQ
    for (int j = 0; j < opQ->Nm; j++) {
        opQ->sigQ.vals[j] = 0.0;
        for (int i = 0; i < opQ->N; i++) {
            double temp = cvxmat_get(&(opQ->Q), i, j);
            opQ->sigQ.vals[j] += fabs(temp);
        }
        opQ->sigQ.vals[j] = 1.0 / opQ->sigQ.vals[j];
    }

    // zQ is usually reset anyways, but this is needed to maintain the existing relaxation
    for (int i = 0; i < opQ->zQ.N; i++) {
        opQ->zQ.vals[i] *= weight_mod;
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


int cvxop_moments_check(cvxop_moments *opQ, cvx_mat *G, cvx_mat *tau)
{
    
    double AzX = 0.0;
    for (int j = 0; j < opQ->Nm; j++) {
        if (opQ->moment_tol.vals[j] >= 0) {
            for (int i = 0; i < opQ->N; i++) {
                double temp = cvxmat_get(&(opQ->Q), i, j);
                temp = (temp * opQ->zQ.vals[j]);
                AzX += temp*temp;
            }
        }
    }
    AzX = sqrt(AzX);
    
    
    
    cvxmat_setvals(&(opQ->Qx), 0.0);
    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = 0; i < opQ->N; i++) {
            double temp = cvxmat_get(&(opQ->Q), i, j) * G->vals[i];
            opQ->Qx.vals[j] += temp;
        }
        opQ->Qx.vals[j] *= opQ->sigQ.vals[j];
    }
    
    for (int j = 0; j < opQ->Nm; j++) {
        if (opQ->moment_tol.vals[j] >= 0) {
            for (int i = 0; i < opQ->N; i++) {
                double temp = cvxmat_get(&(opQ->Q), i, j);
                opQ->norm_helper.vals[i] += (temp * opQ->Qx.vals[j]);
            }
        }
    }


    double norm0 = 0.0;
    for (int j = 0; j < opQ->Nm; j++) {
        if (opQ->moment_tol.vals[j] >= 0) {
            for (int i = 0; i < opQ->N; i++) {
                double temp = cvxmat_get(&(opQ->Q), i, j) * G->vals[i];
                norm0 += temp * temp;
            }
        }
    }
    norm0 = sqrt(norm0);

    opQ->d_norm = norm0;


    double nh_1 = 0.0;
    double nh_2 = 0.0;
    double nh_inf = 0.0;
    for (int i = 0; i < opQ->norm_helper.N; i++) {
        double val = opQ->norm_helper.vals[i] * tau->vals[i];
        nh_2 += val * val;
        nh_1 += fabs(val);
        if (fabs(val) > nh_inf) {
            nh_inf = fabs(val);
        }
    }
    nh_2 = sqrt(nh_2);

    cvxmat_setvals(&(opQ->Qx), 0.0);
    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = 0; i < opQ->N; i++) {
            double temp = cvxmat_get(&(opQ->Q0), i, j) * G->vals[i];
            opQ->Qx.vals[j] += temp;
        }
    }

    int moments_bad = 0;
    double moment_tol = 1.0e-2;

    for (int j = 0; j < opQ->Nm; j++) {
        if (opQ->moment_tol0.vals[j] >= 0) {
            if (fabs(opQ->Qx.vals[j]) > moment_tol) {
                moments_bad = 1;
            }
        }
    }


    if (opQ->verbose>0) {   
        printf("    Moments check:  (%d)  %.2e  %.2e  %.2e              norma = %.2e\n", moments_bad, opQ->Qx.vals[0], opQ->Qx.vals[1], opQ->Qx.vals[2], AzX);
        // printf("  ^^^  norm_helper momt  nh_2 = %.2e    nh_inf = %.2e    nh_1 = %.2e\n", nh_2, nh_inf, nh_1);
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
    free(opQ->moment_tol0.vals);
    free(opQ->norms.vals);
    free(opQ->Q.vals);
    free(opQ->Q0.vals);
    free(opQ->sigQ.vals);
    free(opQ->zQ.vals);
    free(opQ->zQbuff.vals);
    free(opQ->zQbar.vals);
    free(opQ->Qx.vals);

    free(opQ->norm_helper.vals);
    free(opQ->tau_helper.vals);
}






void cvxop_moments_init_old(cvxop_moments *opQ, int N, int ind_inv, double dt,
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
 
    cvxmat_alloc(&opQ->norm_helper, N, 1);
    cvxmat_alloc(&opQ->tau_helper, N, 1);
 
    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = 0; i < N; i++) {
            double temp = pow((dt*i), (double)j);
            cvxmat_set(&(opQ->Q), i, j, temp);
        }
    }
 
    opQ->moment_tol.vals[0] = m0_tol;
    opQ->moment_tol.vals[1] = m1_tol;
    opQ->moment_tol.vals[2] = m2_tol;
 
    double gamma = 42.58e3;
 
    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = 0; i < N; i++) {
            double temp = cvxmat_get(&(opQ->Q), i, j) * gamma * dt;
            cvxmat_set(&(opQ->Q), i, j, temp);
        }
    }
 
    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = ind_inv; i < N; i++) {
            double temp = cvxmat_get(&(opQ->Q), i, j) * -1.0;
            cvxmat_set(&(opQ->Q), i, j, temp);
        }
    }
 
    double active_tols = 0.0;
    for (int j = 0; j < opQ->Nm; j++) {
        if (opQ->moment_tol.vals[j] >= 0) {
            active_tols += 1.0;
        }
    }
 
 
    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = 0; i < N; i++) {
            double temp = cvxmat_get(&(opQ->Q), i, j);
            opQ->norms.vals[j] += temp*temp;
        }
    }
 
    for (int j = 0; j < opQ->Nm; j++) {
        opQ->norms.vals[j] = sqrt(opQ->norms.vals[j]) * dt / pow(2.0, j) / 1000;
    }
 
    for (int j = 0; j < opQ->Nm; j++) {
        for (int i = 0; i < N; i++) {
            double temp = cvxmat_get(&(opQ->Q), i, j) / opQ->norms.vals[j];
            cvxmat_set(&(opQ->Q), i, j, temp);
        }
    }
 
    opQ->moment_tol.vals[0] = m0_tol/opQ->norms.vals[0];
    opQ->moment_tol.vals[1] = m1_tol/opQ->norms.vals[1];
    opQ->moment_tol.vals[2] = m2_tol/opQ->norms.vals[2];
 
    if (opQ->verbose>0) {   
        printf("Q norms = %.2e  %.2e  %.2e    active norms = %.1f\n", opQ->norms.vals[0], opQ->norms.vals[1], opQ->norms.vals[2], active_tols);
    }
 
 
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
 
    for (int j = 0; j < opQ->Nm; j++) {
        if (opQ->moment_tol.vals[j] >= 0) {
            for (int i = 0; i < opQ->N; i++) {
                double temp = cvxmat_get(&(opQ->Q), i, j);
                opQ->tau_helper.vals[i] += fabs(temp);
            }
        }
    }
 
}