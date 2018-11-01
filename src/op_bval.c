#include "op_bval.h"

void cvxop_bval_init(cvxop_bval *opB, int N, int ind_inv, double dt, double init_weight, int verbose) {
    opB->N = N;
    opB->dt = dt;
    opB->ind_inv = ind_inv;
    opB->verbose = verbose;
    opB->mod = 1.0;
    opB->weight = init_weight;
    

    cvxmat_alloc(&opB->B0, N, N);
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

    cvxmat_alloc(&opB->norm_helper, N, 1);

    cvxmat_alloc(&opB->C, N, 1);


    if (opB->active > 0) {
    
        double tt;
        for (int i = 0; i < N; i++) {
            tt = N-i;
            opB->C.vals[i] = tt*(tt+1)/2.0;
        }


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

        cvxmat_multAtA(&opB->B0, &opB->Binit);
        
        // double nsum;
        // for (int i = 0; i < N; i++) {
            
        //     nsum = 0.0;
        //     for (int j = 0; j < N; j++) {
        //         double temp = cvxmat_get(&(opB->B0), i, j);
        //         nsum += (temp*temp);
        //     }
        //     nsum = sqrt(nsum);

        //     for (int j = 0; j < N; j++) {
        //         double temp = cvxmat_get(&(opB->B0), i, j);
        //         cvxmat_set(&(opB->B0), i, j, temp);
        //     }

        // }


        double mat_norm = 0.0;
        for (int i = 0; i < opB->B0.N; i++) {
            mat_norm += (opB->B0.vals[i] * opB->B0.vals[i]);
        }
        mat_norm = sqrt(mat_norm);
        opB->mat_norm = mat_norm;

        for (int i = 0; i < opB->B0.N; i++) {
            opB->B0.vals[i] /= mat_norm;
        }

        for (int i = 0; i < opB->B.N; i++) {
            opB->B.vals[i] = opB->weight * opB->B0.vals[i];
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
}

void cvxop_bval_reweight(cvxop_bval *opB, double weight_mod)
{
    opB->weight *= weight_mod;

    for (int i = 0; i < opB->B.N; i++) {
        opB->B.vals[i] = opB->weight * opB->B0.vals[i];
    }

    double sum;
    for (int j = 0; j < opB->N; j++) {
        sum = 0.0;
        for (int i = 0; i < opB->N; i++) {
            double temp = cvxmat_get(&(opB->B), i, j);
            sum += fabs(temp);
        }
        opB->sigBdenom.vals[j] = sum;
    }

    for (int i = 0; i < opB->N; i++) {
        opB->sigB.vals[i] = 1.0/opB->sigBdenom.vals[i];
    }

    for (int i = 0; i < opB->zB.N; i++) {
        opB->zB.vals[i] *= weight_mod;
    }
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

        double rr = relax;

        // zD=p*zDbar+(1-p)*zD;
        for (int i = 0; i < opB->zB.N; i++) {
            opB->zB.vals[i] = rr * opB->zBbuff.vals[i] + (1 - rr) * opB->zB.vals[i];
        }
    }
}

double cvxop_bval_getbval(cvxop_bval *opB, cvx_mat *G, cvx_mat *tau)
{   
    double AzX = 0.0;
    if (opB->active > 0) {
        cvxmat_setvals(&(opB->Btau), 0.0);
        cvxmat_multAx2(&opB->Btau, &opB->B, &opB->zB);
        for (int i = 1; i < opB->Btau.N; i++) {
            AzX += opB->Btau.vals[i] * opB->Btau.vals[i];
        }
        AzX = sqrt(AzX);
    }


    double mod = 71576597699.4529; // (GAMMA*2*pi)^2
    double bval = 0.0;
    
    
    cvxmat_setvals(&(opB->Bvaltemp), 0.0);
    cvxmat_multAx(&opB->Bvaltemp, &opB->B, G);
    for (int i = 0; i < opB->Bvaltemp.N; i++) {
        opB->Bvaltemp.vals[i] *= opB->sigB.vals[i];
    }
    cvxmat_setvals(&(opB->norm_helper), 0.0);
    cvxmat_multAx2(&opB->norm_helper, &opB->B, &opB->Bvaltemp);


    double norm0 = 0.0;
    for (int i = 0; i < opB->Bvaltemp.N; i++) {
        norm0 += opB->Bvaltemp.vals[i] * opB->Bvaltemp.vals[i];
        // norm0 += fabs(opB->Bvaltemp.vals[i]);
    }
    // norm0 = sqrt(norm0);
    // opB->d_norm = norm0 * opB->N;
    opB->d_norm = norm0; 

    double nh_1 = 0.0;
    double nh_2 = 0.0;
    double nh_inf = 0.0;
    for (int i = 0; i < opB->norm_helper.N; i++) {
        double val = opB->norm_helper.vals[i] * tau->vals[i];
        nh_2 += val * val;
        nh_1 += fabs(val);
        if (fabs(val) > nh_inf) {
            nh_inf = fabs(val);
        }
    }
    nh_2 = sqrt(nh_2);

    if (opB->verbose>0) {   
        printf("    bval calc:      weight = %.2e                                norma = %.2e\n", opB->weight, AzX);
        printf("  ^^^  norm_helper bval  nh_2 = %.2e    nh_inf = %.2e    nh_1 = %.2e\n", nh_2, nh_inf, nh_1);
    }

    for (int i = 0; i < opB->Bvaltemp.N; i++) {
        bval += G->vals[i] * opB->Bvaltemp.vals[i] * mod * opB->dt / opB->weight / opB->sigB.vals[i];
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
