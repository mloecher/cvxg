#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "cvx_matrix.h"
#include "op_slewrate.h"
#include "op_moments.h"
#include "op_beta.h"
#include "op_bval.h"
#include "op_gradient.h"


void cvx_optimize_kernel(cvx_mat *G, cvxop_gradient *opG, cvxop_slewrate *opD, cvxop_moments *opQ, cvxop_beta *opC, cvxop_bval *opB, int N, double relax, int verbose)
{
    
    cvxmat_alloc(G, N, 1);
    cvxmat_setvals(G, 0.0);

    cvx_mat xbar;
    copyNewMatrix(G, &xbar);

    cvx_mat taumx;
    copyNewMatrix(G, &taumx);

    cvx_mat txmx;
    copyNewMatrix(G, &txmx);

    // tau scaling vector
    cvx_mat tau;
    copyNewMatrix(G, &tau);

    cvxop_slewrate_add2tau(opD, &tau);
    cvxop_moments_add2tau(opQ, &tau);
    cvxop_bval_add2tau(opB, &tau);
    cvxmat_EWinvert(&tau);

    double obj0 = 1.0;
    double obj1 = 1.0;

//     cvxmat_setvals(G, 0.5*opG->gmax);
    cvxop_init_G(opG, G);
    
    char str[256];
    FILE *fp;

    int count = 0;
    while (count < 40000) {
    
        // xbar = G-tau.*((D'*zD)+(Q'*zQ)+C')
        cvxmat_setvals(&taumx, 0.0);

        cvxop_slewrate_add2taumx(opD, &taumx);
        cvxop_moments_add2taumx(opQ, &taumx);
        cvxop_beta_add2taumx(opC, &taumx);
        cvxop_bval_add2taumx(opB, &taumx);

        cvxmat_EWmultIP(&taumx, &tau);
        cvxmat_subractMat(&xbar, G, &taumx);

        
        // xbar = gradient_limits(xbar)
        cvxop_gradient_limiter(opG, &xbar);

        // txmx = 2*xbar-G;
        cvxmat_subractMatMult1(&txmx, 2.0, &xbar, G);


        // zDbuff  = zD + sigD.*(D*txmx);
        // zQbuff  = zQ + sigQ.*(Q*txmx);

        // zDbar = zDbuff - sigD.*min(SRMAX,max(-SRMAX,zDbuff./sigD));
        // zQbar = zQbuff - sigQ.*min(mvec,max(-mvec,zQbuff./sigQ));

        // zD=p*zDbar+(1-p)*zD;
        // zQ=p*zQbar+(1-p)*zQ;
        cvxop_slewrate_update(opD, &txmx, relax);
        cvxop_moments_update(opQ, &txmx, relax);
        cvxop_bval_update(opB, &txmx, relax);

        // for (int i = 0; i < opB->B.N; i++) {
        //     opB->B.vals[i] *= 0.80;
        // }

        // G=p*xbar+(1-p)*G;
        cvxmat_updateG(G, relax, &xbar);

        // Need checks here
        if ( count % 500 == 0 ) {
            // double obj1 = 0.0;
            // for (int ii = 0; ii < N; ii++) {
            //     obj1 += opC->C.vals[ii] * G->vals[ii];    
            // }
            // obj1 = fabs(obj1);

            obj1 = cvxop_bval_getbval(opB, G);

            double percent_increase = (obj1-obj0)/obj0;

            if (verbose>0) {printf("count = %d   obj = %.1f   increase = %.2e\n", count, obj1, percent_increase);}
            int limit_break = 0;
            limit_break += cvxop_moments_check(opQ, G);
            limit_break += cvxop_slewrate_check(opD, G);
            limit_break += cvxop_gradient_check(opG, G);

            // This might happen before convergence, maybe check if it happens twice in a row?
            if ( (count > 0) && (fabs(percent_increase) < 2.0e-3) && (limit_break == 0)) {
                if (verbose > 0) {
                    printf("** Early termination at count = %d   bval = %.1f\n", count, obj1);
                }
                break;
            }

            obj0 = obj1;

            // sprintf(str, "./raw_v1/G_count_%06d.raw", count);
            // fp = fopen (str, "wb");
            // fwrite(G->vals, sizeof(double), G->N, fp);
            // fwrite(opB->Btau.vals, sizeof(double), G->N, fp);
            // fclose(fp);
        }
    
        count++;
    }
    
    // printf("\n --- G:\n");
    // for (int i = 0; i < G.N; i++) {
    //     printf("%.2e  ", G.vals[i]);
    // }
    // printf("\n\n");

    // printf("\n --- zQ:\n");
    // for (int i = 0; i < opQ->zQ.N; i++) {
    //     printf("%.2e  ", opQ->zQ.vals[i]);
    // }
    // printf("\n\n");

    // printf("\n --- zD:\n");
    // for (int i = 0; i < opD->zD.N; i++) {
    //     printf("%.2e  ", opD->zD.vals[i]);
    // }
    // printf("\n\n");
}

void run_kernel_diff(double **G_out, int *N_out, double gmax, double smax, double m0_tol, double m1_tol, double m2_tol, double TE, double T_readout, double T_90, double T_180, double dt, int diffmode)
{
    struct timespec ts0;
    struct timespec ts1;

    double relax = 1.9;
    int verbose = 0;

    int N = round((TE-T_readout) * 1.0e-3/dt);
    if (N < 5) {
        printf ("\nWARNING: N = %d looks too small, setting to 5\n\n", N);
        N = 5;
    }
    int ind_inv = round((N + T_readout/(dt*1.0e3))/2.0);
    
    // I think these should be ceil instead of floor, but just matching matlab code for now
    int ind_end90 = floor(T_90*(1e-3/dt));
    int ind_start180 = ind_inv - floor(T_180*(1e-3/dt/2));
    int ind_end180 = ind_inv + floor(T_180*(1e-3/dt/2));

    if (verbose > 0) {
        printf ("\nN = %d  ind_inv = %d\n90_zeros = %d:%d    180_zeros = %d:%d\n\n", N, ind_inv, 0, ind_end90, ind_start180, ind_end180);
    }

    cvxop_gradient opG;
    cvxop_gradient_init(&opG, N, dt, gmax, ind_inv, verbose);
    cvxop_gradient_setFixRange(&opG, 0, ind_end90, 0.0);
    cvxop_gradient_setFixRange(&opG, ind_start180, ind_end180, 0.0);

    cvxop_slewrate opD;
    cvxop_slewrate_init(&opD, N, dt, smax, verbose);

    cvxop_moments opQ;
    cvxop_moments_init(&opQ, N, ind_inv, dt,
                        m0_tol, m1_tol, m2_tol, 
                        verbose);
    
    cvxop_beta opC;
    cvxop_beta_init(&opC, N, dt, verbose);
    

    cvxop_bval opB;
    cvxop_bval_init(&opB, N, ind_inv, dt, verbose);

    if (diffmode == 1) {
        opB.active = 0; 
    } else if (diffmode == 2) {
        opC.active = 0; 
    }
    

    cvx_mat G;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    cvx_optimize_kernel(&G, &opG, &opD, &opQ, &opC, &opB, N, relax, verbose);
    clock_gettime(CLOCK_MONOTONIC, &ts1);

    *N_out = G.rows;
    *G_out = G.vals;

    double elapsed;
    elapsed = (ts1.tv_sec - ts0.tv_sec);
    elapsed += (ts1.tv_nsec - ts0.tv_nsec) / 1000000000.0;

    if (verbose > 0) {
        printf ("Elapsed Time = %.4f ms\n", 1000.0*elapsed);
    }

}

int main (void)
{
    printf ("In optimize_kernel.c main function\n");
    

    // 1 = betamax
    // 2 = bval max
    int diffmode = 2;

    double *G;
    int N;

    run_kernel_diff(&G, &N, 0.074, 50.0, 0.0, 0.0, 0.0, 60.0, 10.0, 3.0, 6.0, 0.3e-3, diffmode);


    // run_kernel_diff(0.074, 100.0, 0.0, 0.0, 0.0, 5.0, 1.0, 0.5, 0.5, 0.1e-3);


    // cvxop_gradient opG;
    // cvxop_gradient_init(&opG, 40, 0.1e-3, .074);

    // cvxop_slewrate opD;
    // cvxop_slewrate_init(&opD, 40, 0.1e-3, 100.0, relax);

    // cvxop_moments opQ;
    // cvxop_moments_init(&opQ, 40, 24, 0.1e-3,
    //                     0.0, 0.0, 0.0,
    //                     relax);

    // cvxop_beta opC;
    // cvxop_beta_init(&opC, 40, 0.1e-3);

    // cvx_optimize_kernel(&opG, &opD, &opQ, &opC, 40, relax);



    return 0;
}
