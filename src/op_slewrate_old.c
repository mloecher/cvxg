#include "op_slewrate.h"

void cvxop_slewrate_init(cvxop_slewrate *opD, int N, double dt, double smax, int verbose)
{
    opD->active = 1;
    opD->N = N;
    opD->dt = dt;
    opD->smax = smax*dt;
    opD->verbose = verbose;

    if (opD->verbose>0) {   
        printf("opD->smax = %.2e\n", opD->smax);
    }

    cvxmat_alloc(&opD->zD, N-1, 1);
    cvxmat_alloc(&opD->zDbuff, N-1, 1);
    cvxmat_alloc(&opD->zDbar, N-1, 1);
    cvxmat_alloc(&opD->Dx, N-1, 1);
}

void cvxop_slewrate_add2tau(cvxop_slewrate *opD, cvx_mat *tau_mat)
{
    int N = tau_mat->rows;
    for (int i = 0; i < N; i++) {
        if ((i == 0) || (i == N-1)) {
            tau_mat->vals[i] += 0.5;
        } else {
            tau_mat->vals[i] += 1.0;
        }
    }
}

void cvxop_slewrate_add2taumx(cvxop_slewrate *opD, cvx_mat *taumx)
{
    taumx->vals[0] += -0.5*opD->zD.vals[0];
    for (int i = 1; i < opD->zD.N; i++) {
        taumx->vals[i] += 0.5*(opD->zD.vals[i-1] - opD->zD.vals[i]);
    }
    taumx->vals[taumx->N-1] += 0.5*opD->zD.vals[opD->zD.N-1];

}

void cvxop_slewrate_update(cvxop_slewrate *opD, cvx_mat *txmx, double relax)
{

    // zDbuff  = zD + sigD.*(D*txmx);
    cvxmat_setvals(&(opD->Dx), 0.0);

    for (int i = 0; i < opD->Dx.N; i++) {
        opD->Dx.vals[i] += 0.5 * (txmx->vals[i+1] - txmx->vals[i]);
    }

    for (int i = 0; i < opD->zDbuff.N; i++) {
        opD->zDbuff.vals[i] = opD->zD.vals[i] + opD->Dx.vals[i];
    }

    // zDbar = zDbuff - sigD.*min(SRMAX,max(-SRMAX,zDbuff./sigD));
    // Use zDbar as temp storage for the second half of the above equation
    for (int i = 0; i < opD->zDbar.N; i++) {
        if (opD->zDbuff.vals[i] > 0.99*opD->smax) {
            opD->zDbar.vals[i] = 0.99*opD->smax;
        } else if (opD->zDbuff.vals[i] < -0.99*opD->smax) {
            opD->zDbar.vals[i] = -0.99*opD->smax;
        } else {
            opD->zDbar.vals[i] = opD->zDbuff.vals[i];
        }
    }

    for (int i = 0; i < opD->zDbar.N; i++) {
        opD->zDbar.vals[i] = opD->zDbuff.vals[i] - opD->zDbar.vals[i];
    }

    // zD=p*zDbar+(1-p)*zD;
    for (int i = 0; i < opD->zD.N; i++) {
        opD->zD.vals[i] = relax * opD->zDbar.vals[i] + (1 - relax) * opD->zD.vals[i];
    }
}

int cvxop_slewrate_check(cvxop_slewrate *opD, cvx_mat *G)
{
    cvxmat_setvals(&(opD->Dx), 0.0);

    for (int i = 0; i < opD->Dx.N; i++) {
        opD->Dx.vals[i] += 0.5 * (G->vals[i+1] - G->vals[i]);
    }

    int slew_too_high = 0;
    int slew_bad = 0;

    for (int i = 0; i < opD->Dx.N; i++) {
        if (fabs(opD->Dx.vals[i]) > opD->smax) {
            slew_too_high++;
            slew_bad = 1;
        }
    }

    if (opD->verbose>0) {  
        printf("    Slew check:     (%d)  %d\n", slew_bad, slew_too_high);
    }

    return slew_bad;
}

// cvxop_slewrate_add2taumx(&opD, &tau_mat);
// cvxop_slewrate_update(&opD, &txmx);