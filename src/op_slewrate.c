#include "op_slewrate.h"

void cvxop_slewrate_init(cvxop_slewrate *opD, int N, double dt, double smax, double init_weight, int verbose)
{
    opD->active = 1;
    opD->N = N;
    opD->dt = dt;
    opD->weight = init_weight;
    opD->base_smax = smax*dt;
    opD->smax = smax*dt*opD->weight;
    opD->sigD = 1.0/(2.0*opD->weight);
    opD->verbose = verbose;

    if (opD->verbose>0) {   
        printf("opD->smax = %.2e\n", opD->smax);
    }

    cvxmat_alloc(&opD->zD, N-1, 1);
    cvxmat_alloc(&opD->zDbuff, N-1, 1);
    cvxmat_alloc(&opD->zDbar, N-1, 1);
    cvxmat_alloc(&opD->Dx, N-1, 1);
    cvxmat_alloc(&opD->norm_helper, N, 1);
}

void cvxop_slewrate_reweight(cvxop_slewrate *opD, double weight_mod)
{
    opD->weight *= weight_mod;
    opD->smax = opD->base_smax*opD->weight;
    opD->sigD = 1.0/(2.0*opD->weight);

    // zD is usually reset anyways, but this is needed to maintain the existing relaxation
    for (int i = 0; i < opD->zD.N; i++) {
        opD->zD.vals[i] *= weight_mod;
    }
}


void cvxop_slewrate_add2tau(cvxop_slewrate *opD, cvx_mat *tau_mat)
{
    int N = tau_mat->rows;
    for (int i = 0; i < N; i++) {
        if ((i == 0) || (i == N-1)) {
            tau_mat->vals[i] += opD->weight;
        } else {
            tau_mat->vals[i] += 2.0*opD->weight;
        }
    }
}

void cvxop_slewrate_add2taumx(cvxop_slewrate *opD, cvx_mat *taumx)
{
    taumx->vals[0] += -opD->weight*opD->zD.vals[0];
    for (int i = 1; i < opD->zD.N; i++) {
        taumx->vals[i] += opD->weight*(opD->zD.vals[i-1] - opD->zD.vals[i]);
    }
    taumx->vals[taumx->N-1] += opD->weight*opD->zD.vals[opD->zD.N-1];

}

void cvxop_slewrate_update(cvxop_slewrate *opD, cvx_mat *txmx, double relax)
{

    // zDbuff  = zD + sigD.*(D*txmx);

    cvxmat_setvals(&(opD->Dx), 0.0);

    for (int i = 0; i < opD->Dx.N; i++) {
        opD->Dx.vals[i] += opD->weight*opD->sigD*(txmx->vals[i+1] - txmx->vals[i]);
    }

    for (int i = 0; i < opD->zDbuff.N; i++) {
        opD->zDbuff.vals[i] = opD->zD.vals[i] + opD->Dx.vals[i];
    }

    

    // zDbar = zDbuff - sigD.*min(SRMAX,max(-SRMAX,zDbuff./sigD));
    // Use zDbar as temp storage for the second half of the above equation
    for (int i = 0; i < opD->zDbar.N; i++) {
        double temp = opD->zDbuff.vals[i] / opD->sigD;
        if (temp > 0.99*opD->smax) {
            opD->zDbar.vals[i] = 0.99*opD->smax;
        } else if (temp < -0.99*opD->smax) {
            opD->zDbar.vals[i] = -0.99*opD->smax;
        } else {
            opD->zDbar.vals[i] = temp;
        }
    }

    for (int i = 0; i < opD->zDbar.N; i++) {
        opD->zDbar.vals[i] = opD->zDbuff.vals[i] - opD->sigD*opD->zDbar.vals[i];
    }

    double rr = relax;

    // zD=p*zDbar+(1-p)*zD;
    for (int i = 0; i < opD->zD.N; i++) {
        opD->zD.vals[i] = rr * opD->zDbar.vals[i] + (1.0 - rr) * opD->zD.vals[i];
    }
}

int cvxop_slewrate_check(cvxop_slewrate *opD, cvx_mat *G, cvx_mat *tau)
{

    double temp = 0.0;
    double AzX = 0.0;
    temp = opD->weight*opD->zD.vals[0];
    AzX += temp*temp;
    for (int i = 1; i < opD->zD.N; i++) {
        temp = opD->weight*(opD->zD.vals[i-1] - opD->zD.vals[i]);
        AzX += temp * temp;
    }
    temp = opD->weight*opD->zD.vals[opD->zD.N-1];
    AzX += temp*temp;
    AzX = sqrt(AzX);

    cvxmat_setvals(&(opD->Dx), 0.0);
    for (int i = 0; i < opD->Dx.N; i++) {
        opD->Dx.vals[i] += opD->weight*opD->sigD*(G->vals[i+1] - G->vals[i]);
    }
    opD->norm_helper.vals[0] += -opD->weight*opD->Dx.vals[0];
    for (int i = 1; i < opD->Dx.N; i++) {
        opD->norm_helper.vals[i] += opD->weight*(opD->Dx.vals[i-1] - opD->Dx.vals[i]);
    }
    opD->norm_helper.vals[opD->norm_helper.N-1] += opD->weight*opD->Dx.vals[opD->Dx.N-1];



    double norm0 = 0.0;
    double norm1 = 0.0;
    for (int i = 0; i < opD->Dx.N; i++) {
        norm0 += opD->Dx.vals[i] * opD->Dx.vals[i] ;
        norm1 += opD->zDbar.vals[i] * opD->zDbar.vals[i] ;
    }
    norm0 = sqrt(norm0);
    norm1 = sqrt(norm1);

    opD->d_norm = norm0;

    int slew_too_high = 0;
    int slew_bad = 0;

    for (int i = 0; i < opD->Dx.N; i++) {
        if (fabs(opD->Dx.vals[i]) > opD->smax*opD->sigD) {
            slew_too_high++;
            slew_bad = 1;
        }
    }

    double nh_1 = 0.0;
    double nh_2 = 0.0;
    double nh_inf = 0.0;
    for (int i = 0; i < opD->norm_helper.N; i++) {
        double val = opD->norm_helper.vals[i]  * tau->vals[i];
        nh_2 += val * val;
        nh_1 += fabs(val);
        if (fabs(val) > nh_inf) {
            nh_inf = fabs(val);
        }
    }
    nh_2 = sqrt(nh_2);

    if (opD->verbose>0) {  
        printf("    Slew check:     (%d)  %d                                           norma = %.2e \n", slew_bad, slew_too_high, AzX);
        printf("  ^^^  norm_helper slew  nh_2 = %.2e    nh_inf = %.2e    nh_1 = %.2e\n", nh_2, nh_inf, nh_1);
    }

    // double weight_mod = norm1/norm0;
    // opD->weight = weight_mod;
    // opD->smax = opD->base_smax * weight_mod;
    // opD->sigD = 1.0/(2.0*opD->weight);
    // printf("    Slew check:     %.2e  %.2e  %.2e \n", norm0, norm1, opD->weight);

    return slew_bad;
}

void cvxop_slewrate_destroy(cvxop_slewrate *opD)
{
    free(opD->zD.vals);
    free(opD->zDbuff.vals);
    free(opD->zDbar.vals);
    free(opD->Dx.vals);
}