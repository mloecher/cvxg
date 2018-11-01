#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "cvx_matrix.h"
#include "op_slewrate.h"
#include "op_moments.h"
#include "op_beta.h"
#include "op_bval.h"
#include "op_gradient.h"


void cvx_optimize_kernel(cvx_mat *G, cvxop_gradient *opG, cvxop_slewrate *opD, cvxop_moments *opQ, cvxop_beta *opC, cvxop_bval *opB, 
                         int N, double relax, int verbose, double bval_reduction, double *ddebug, int N_converge)
{
    int max_iter = 40000;
    int check_amount = 100;
    int i_check = 0;
    int N_backlog = max_iter / check_amount;
    double *bval_backlog = (double *)malloc(N_backlog*sizeof(double));
    for (int i = 0; i < N_backlog; i++) {
        bval_backlog[i] = 0.0;
    }
    
    
    double stop_increase = 1.0e-1;
    double stop_increase2 = 1.0e-4;
    double stop_increase3 = 2.0e-3;
    int converge_count = 0;
    int limit_count = 0;

    int N_balance = N_converge/2;
    int rebalance_count = 0;
    int raise_allowed = 1;

    int bad_slew_sum = 0;
    int bad_moment_sum = 0;
    
    int is_balanced = 0;
    if (bval_reduction <= 0) {
        is_balanced = 1;
    }

    // int is_balanced = 1;

    for (int i = 0; i < opB->zB.N; i++) {
        opB->zB.vals[i] = 0.0;
    }
    for (int i = 0; i < opD->zD.N; i++) {
        opD->zD.vals[i] = 0.0;
    }
    for (int i = 0; i < opQ->zQ.N; i++) {
        opQ->zQ.vals[i] = 0.0;
    }

    cvx_mat G0;
    copyNewMatrix(G, &G0);

    cvx_mat xbar;
    copyNewMatrix(G, &xbar);
    cvxmat_setvals(&xbar, 0.0);

    cvx_mat taumx;
    copyNewMatrix(G, &taumx);
    cvxmat_setvals(&taumx, 0.0);

    cvx_mat txmx;
    copyNewMatrix(G, &txmx);
    cvxmat_setvals(&txmx, 0.0);

    // tau scaling vector
    cvx_mat tau;
    copyNewMatrix(G, &tau);
    cvxmat_setvals(&tau, 0.0);

    cvxop_slewrate_add2tau(opD, &tau);
    cvxop_moments_add2tau(opQ, &tau);
    cvxop_bval_add2tau(opB, &tau);
    cvxop_beta_add2tau(opC, &tau);
    cvxmat_EWinvert(&tau);

    if (verbose > 0) {
        double tau_norm = 0.0;
        for (int i = 0; i < tau.N; i++) {
            tau_norm += tau.vals[i] * tau.vals[i];
        }
        tau_norm = sqrt(tau_norm);
        printf("  Tau norm = %.2e\n", tau_norm);
    }

    double obj0 = 1.0;
    double obj1 = 1.0;    
    
    char str[256];
    FILE *fp;

    int count = 0;
    while (count < 40000) {
    
        // xbar = G-tau.*((D'*zD)+(Q'*zQ)+C'+(B'*zB))
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

        if (count == 0) {
            double tnorm = 0.0;
            for (int i = 0; i < opB->zB.N; i++) {
                tnorm += opB->zB.vals[i] * opB->zB.vals[i];
            }
            tnorm = sqrt(tnorm);

            printf("\ncount0 norm B = %.2e\n", tnorm); 

            tnorm = 0.0;
            for (int i = 0; i < opQ->zQ.N; i++) {
                tnorm += opQ->zQ.vals[i] * opQ->zQ.vals[i];
            }
            tnorm = sqrt(tnorm);
            printf("\ncount0 norm Q = %.2e\n", tnorm); 

            tnorm = 0.0;
            for (int i = 0; i < opD->zD.N; i++) {
                tnorm += opD->zD.vals[i] * opD->zD.vals[i];
            }
            tnorm = sqrt(tnorm);
            printf("\ncount0 norm D = %.2e\n", tnorm); 
        }
        

        // Need checks here
        if ( count % 100 == 0 ) {
            double resid_diff = 0.0;
            for (int ii = 0; ii < N; ii++) {
                resid_diff += pow( ((G0.vals[ii] - G->vals[ii]) / opG->gmax), 2.0 );    
            }
            resid_diff = sqrt(resid_diff);
            for (int ii = 0; ii < N; ii++) {
                G0.vals[ii] = G->vals[ii];    
            }


            // obj1 = cvxop_bval_getbval(opB, G, &tau);
            obj1 = cvxop_gradient_getbval(opG, G);
            bval_backlog[i_check] = obj1;
            

            double ii_backlog = 0;
            double bval_backlog0 = 0.0;
            for (int i = 0; i < 5; i++) {
                if ((i_check - i) >= 0) {
                    bval_backlog0 += bval_backlog[ (i_check - i) ];
                } else {
                    bval_backlog0 += 999999999999.0;
                }
                ii_backlog += 1.0;
            }
            bval_backlog0 /= ii_backlog;
            

            ii_backlog = 0;
            double bval_backlog1 = 0.0;
            for (int i = 5; i < 10; i++) {
                if ((i_check - i) >= 0) {
                    bval_backlog1 += bval_backlog[ (i_check - i) ];
                } else {
                    bval_backlog1 += 9999999999.0;
                }
                ii_backlog += 1.0;
            }
            bval_backlog1 /= ii_backlog;

            i_check += 1;

            double backlog_diff = (sqrt(bval_backlog0) - sqrt(bval_backlog1)) / sqrt(bval_backlog1);

            bval_backlog0 = (sqrt(obj1) - sqrt(bval_backlog1)) / sqrt(obj1);
            bval_backlog1 = (sqrt(obj1) - sqrt(bval_backlog1)) / sqrt(obj1);

            int is_converged3 = 0;
            if (fabs(backlog_diff) < stop_increase3) {
                is_converged3 = 1;
            }

            int is_converged4 = 0;
            if (fabs(backlog_diff) < 2*stop_increase3) {
                is_converged4 = 1;
            }

            printf("bval5 = %.2e   bval10 = %.2e   diff = %.2e\n", bval_backlog0, bval_backlog1, backlog_diff);

            double percent_increase = ( sqrt(obj1)-sqrt(obj0) )/sqrt(obj0);

            if (fabs(percent_increase) < stop_increase) {
                converge_count += 1;
            } else {
                converge_count = 0;
                limit_count = 0;
            }

            rebalance_count += 1;
            // if (resid_diff < stop_increase) {
            //     converge_count += 1;
            // } else {
            //     converge_count = 0;
            //     limit_count = 0;
            // }
            
            int is_converged2 = 0;
            if (fabs(percent_increase) < stop_increase2) {
                is_converged2 = 1;
            }
            // if (percent_increase < 0.0) {
            //     is_converged2 = 1;
            // }


            if (verbose>0) {printf("count = %d   cc = %d   obj = %.1f   increase = %.2e  resid_diff = %.2e\n", count, converge_count, obj1, percent_increase, resid_diff);}
            int bad_slew = cvxop_slewrate_check(opD, G, &tau);
            int bad_moments = cvxop_moments_check(opQ, G, &tau);
            int bad_gradient = cvxop_gradient_check(opG, G);
            
            // double trash = cvxop_bval_getbval(opB, G, &tau);

            if (bad_slew > 0)    {
                ddebug[5] += 1.0;
                bad_slew_sum += 1;
            }
            if (bad_moments > 0) {
                ddebug[6] += 1.0;
                bad_moment_sum += 1;
            }

            int limit_break = 0;
            limit_break += bad_slew;
            limit_break += bad_moments;
            limit_break += bad_gradient;

            if (limit_break == 0) {
                limit_count += 1;
            } else {
                limit_count = 0;
            }

            // printf("bval = %.2e  slew = %.2e  moment = %.2e  gradient = %.2e\n", opB->d_norm, opD->d_norm, opQ->d_norm, opG->d_norm);
            // double temp_scale = opG->d_norm;
            // printf("bval = %.2e  slew = %.2e  moment = %.2e  gradient = %.2e\n", 
            //         opB->d_norm/temp_scale, opD->d_norm/temp_scale, opQ->d_norm/temp_scale, opG->d_norm/temp_scale);

            // printf("bval / moment = %.2e  \n", 
            //         opB->d_norm/opQ->d_norm);

            // printf("bval / g = %.2e  \n", 
            //         opB->d_norm/opG->d_norm);

            // if ( (count > 0) && (converge_count > N_converge)  && (is_converged2 > 0) && (limit_break == 0)) {
            if ( (count > 0) && (rebalance_count > N_converge)  && (is_converged3 > 0) && (limit_break == 0)) {
                
                if (verbose > 0) {
                    printf("** Early termination at count = %d   bval = %.1f\n", count, obj1);
                }
                break;
            }

            /*
            if ( (bval_reduction > 0.0) && (count > 0) && (rebalance_count > N_balance) && (limit_break == 0) && (raise_allowed > 0)) {
                
                if (verbose > 0) {
                    printf("\n\n !-!-!-!-!-!-! Raising bvalue !-!-!-!-!-!-! \n\n");
                }

                rebalance_count = 0;
                converge_count = 0;

                cvxop_bval_reweight(opB, 2.0*bval_reduction);
                cvxop_beta_reweight(opC, 2.0*bval_reduction);

                cvxmat_setvals(&tau, 0.0);
                cvxop_slewrate_add2tau(opD, &tau);
                cvxop_moments_add2tau(opQ, &tau);
                cvxop_bval_add2tau(opB, &tau);
                cvxop_beta_add2tau(opC, &tau);
                cvxmat_EWinvert(&tau);
                            

                for (int i = 0; i < opB->zB.N; i++) {
                    opB->zB.vals[i] = 0.0; 
                }
                for (int i = 0; i < opD->zD.N; i++) {
                    opD->zD.vals[i] = 0.0;
                }
                for (int i = 0; i < opQ->zQ.N; i++) {
                    opQ->zQ.vals[i] = 0.0;
                }

            }
            */

           float bval_mod = 1.0;
           int needs_rebalancing = 0;
           if ( (is_converged3 > 0) && (rebalance_count > N_converge) ) {
               needs_rebalancing = 1;
           }
        //    if ( (rebalance_count > 4*N_converge) ) {
        //        needs_rebalancing = 1;
        //        bval_mod = 1.1 / bval_reduction;
        //        printf("\n\n !-!-!-!-!-!-!      1.1 reweight      !-!-!-!-!-!-! \n\n");
        //    }

        //    if (rebalance_count > 10*N_converge) {
        //        needs_rebalancing = 1;
        //    }

            if ( (bval_reduction > 0.0) && (count > 0) && (needs_rebalancing > 0) ) {

            
                raise_allowed = 0;
                rebalance_count = 0;

                if (verbose > 0) {
                    printf("\n\n !-!-!-!-!-!-! Converged to an inadequate waveform, reweighting !-!-!-!-!-!-! \n\n");
                }

                converge_count = 0;
                
                // cvxop_bval_reweight(opB, bval_reduction);

                if (bad_moments > 0) {
                    cvxop_moments_reweight(opQ, bval_mod*bval_reduction);
                    // cvxop_slewrate_reweight(opD, 0.25*bval_reduction);
                }
                if (bad_slew > 0) {
                    cvxop_slewrate_reweight(opD, bval_mod*bval_reduction);
                    // cvxop_moments_reweight(opQ, 0.25*bval_reduction);
                }
                
                if ((bad_slew < 1) && (bad_moments < 1)) {
                    cvxop_bval_reweight(opB, bval_mod*bval_reduction);
                    cvxop_beta_reweight(opC, bval_mod*bval_reduction);
                    // is_balanced = 1;
                } else {
                    // cvxop_bval_reweight(opB, (1.0/bval_reduction));
                    // cvxop_beta_reweight(opC, (1.0/bval_reduction));
                    is_balanced = 1;
                }

                bad_moment_sum = 0;
                bad_slew_sum = 0;
                

                cvxmat_setvals(&tau, 0.0);
                cvxop_slewrate_add2tau(opD, &tau);
                cvxop_moments_add2tau(opQ, &tau);
                cvxop_bval_add2tau(opB, &tau);
                cvxop_beta_add2tau(opC, &tau);
                cvxmat_EWinvert(&tau);
                            

                for (int i = 0; i < opB->zB.N; i++) {
                    opB->zB.vals[i] = 0.0; 
                }
                for (int i = 0; i < opD->zD.N; i++) {
                    opD->zD.vals[i] = 0.0;
                }
                for (int i = 0; i < opQ->zQ.N; i++) {
                    opQ->zQ.vals[i] = 0.0;
                }
            }

            // if ( (count > 0) && (converge_count > 4*N_converge)) {
            //     if (verbose > 0) {
            //         printf("** Broke for failing to stay in limits at count = %d   bval = %.1f\n", count, obj1);
            //     }
            //     break;
            // }

            obj0 = obj1;

            printf("WEIGHTS betaval = %.2e  bval = %.2e  slew = %.2e  moment = %.2e  B norm = %.2e\n", 
                    opC->weight, opB->weight, opD->weight, opQ->weight, opB->mat_norm);



            printf("\n");

            // sprintf(str, "./raw_v1/G_count_%06d.raw", count);
            // fp = fopen (str, "wb");
            // fwrite(G->vals, sizeof(double), G->N, fp);
            // fwrite(opB->Btau.vals, sizeof(double), G->N, fp);
            // fclose(fp);
        }
    
        count++;
        ddebug[0] = count;
        fflush(stdout);
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

    ddebug[13] = cvxop_gradient_getbval(opG, G);

    free(xbar.vals);
    free(taumx.vals);
    free(txmx.vals);
    free(tau.vals);
    free(G0.vals);

    ddebug[1] = opB->weight;

    int bad_slew = cvxop_slewrate_check(opD, G, &tau);
    int bad_moments = cvxop_moments_check(opQ, G, &tau);
    int bad_gradient = cvxop_gradient_check(opG, G);

    ddebug[7] = bad_slew;
    ddebug[8] = bad_moments;
    ddebug[9] = bad_gradient;
}

void interp_down(cvx_mat *G, double dt_in, double dt_out, double TE, double T_readout) {
    int N1 = round((TE-T_readout) * 1.0e-3/dt_out);

    double *new_vals;
    double *temp_free = G->vals;
    new_vals = malloc(N1 * sizeof(double));


    double tt;
    double ti;
    int i0, i1;
    double d0, d1;
    double v0, v1;
    for (int i = 0; i < N1; i++) {
        ti = (dt_out * i) / dt_in;
        
        i0 = floor(ti);
        if (i0 < 0) {i0 = 0;} // Shouldn't happen unless some weird rounding and floor?
        
        i1 = i0+1;

        if (i1 < G->N) {
            d0 = fabs(ti-i1);
            d1 = 1.0 - d0;

            v0 = d0 * temp_free[i0];
            v1 = d1 * temp_free[i1];

            new_vals[i] = v0 + v1;
        } else {
            d0 = fabs(ti-i1);
            v0 = d0 * temp_free[i0];
            new_vals[i] = v0;
        }
    }

    G->vals = new_vals;
    G->N = N1;
    G->rows = N1;
    free(temp_free);
}


void run_kernel_refine(cvx_mat *G, double *ddebug, double gmax, double smax, double *moment_tols, double dt, double TE, 
                        double T_readout, double T_90, double T_180, int diffmode,
                        double bval_weight, double slew_weight, double moments_weight, 
                        double bval_reduce)
{
    double relax = 1.8;
    int verbose = 1;

    int N = G->N;

    int ind_inv = round((N + T_readout/(dt*1.0e3))/2.0);
    
    // I think these should be ceil instead of floor, but just matching matlab code for now
    int ind_end90 = ceil(T_90*(1e-3/dt));
    int ind_start180 = ind_inv - ceil(T_180*(1e-3/dt/2));
    int ind_end180 = ind_inv + ceil(T_180*(1e-3/dt/2));

    if (verbose > 0) {
        printf ("\nN = %d  ind_inv = %d\n90_zeros = %d:%d    180_zeros = %d:%d\n\n", N, ind_inv, 0, ind_end90, ind_start180, ind_end180);
    }

    cvxop_gradient opG;
    cvxop_gradient_init(&opG, N, dt, gmax, ind_inv, verbose);
    cvxop_gradient_setFixRange(&opG, 0, ind_end90, 0.0);
    cvxop_gradient_setFixRange(&opG, ind_start180, ind_end180, 0.0);

    cvxop_slewrate opD;
    cvxop_slewrate_init(&opD, N, dt, smax, slew_weight, verbose);

    cvxop_moments opQ;
    cvxop_moments_init(&opQ, N, ind_inv, dt,
                        moment_tols, moments_weight,
                        verbose);
    
    cvxop_beta opC;
    cvxop_beta_init(&opC, N, dt, bval_weight, verbose);
    
    cvxop_bval opB;
    cvxop_bval_init(&opB, N, ind_inv, dt, bval_weight, verbose);

    if (diffmode == 1) {
        opB.active = 0; 
    } else if (diffmode == 2) {
        opC.active = 0; 
    }

    cvx_optimize_kernel(G, &opG, &opD, &opQ, &opC, &opB, N, relax, verbose, bval_reduce, ddebug, 6);
    
    if (verbose > 0) {
        printf ("\n****************************************\n");
        printf ("--- Finished diff kernel4 refiner in %d iterations  b weight = %.1e", (int)(ddebug)[0], (ddebug)[1]);
        printf ("\n****************************************\n");
    }
}


void run_kernel_diff4(double **G_out, int *N_out, double **ddebug,
                      double gmax, double smax, 
                      double *moment_tols, double TE, 
                      double T_readout, double T_90, double T_180, int diffmode,
                      double bval_weight, double slew_weight, double moments_weight, 
                      double bval_reduce, int N0)
{
    double relax = 1.9;
    int verbose = 1;

    int N = N0;
    double dt = (TE-T_readout) * 1.0e-3 / (double) N;

    if (verbose > 0) {
        printf ("\nFirst pass, N = %d    dt = %.2e\n\n", N, dt);
    }

    if (N < 5) {
        printf ("\nWARNING: N = %d looks too small, setting to 5\n\n", N);
        N = 5;
    }
    int ind_inv = round((N + T_readout/(dt*1.0e3))/2.0);
    
    // I think these should be ceil instead of floor, but just matching matlab code for now
    int ind_end90 = ceil(T_90*(1e-3/dt));
    int ind_start180 = ind_inv - ceil(T_180*(1e-3/dt/2));
    int ind_end180 = ind_inv + ceil(T_180*(1e-3/dt/2));

    if (verbose > 0) {
        printf ("\nN = %d  ind_inv = %d\n90_zeros = %d:%d    180_zeros = %d:%d\n\n", N, ind_inv, 0, ind_end90, ind_start180, ind_end180);
    }

    cvxop_gradient opG;
    cvxop_gradient_init(&opG, N, dt, gmax, ind_inv, verbose);
    cvxop_gradient_setFixRange(&opG, 0, ind_end90, 0.0);
    cvxop_gradient_setFixRange(&opG, ind_start180, ind_end180, 0.0);

    cvxop_slewrate opD;
    cvxop_slewrate_init(&opD, N, dt, smax, slew_weight, verbose);
    // cvxop_slewrate_init(&opD, N, dt, smax, 1.0, verbose);

    cvxop_moments opQ;
    cvxop_moments_init(&opQ, N, ind_inv, dt,
                        moment_tols, moments_weight,
                        verbose);
    // cvxop_moments_init(&opQ, N, ind_inv, dt,
    //                     moment_tols, 10.0,
    //                     verbose);
    
    cvxop_beta opC;
    cvxop_bval opB;
    
    if (diffmode == 1) {
        opB.active = 0; 
        opC.active = 1; 
    } else if (diffmode == 2) {
        opC.active = 0; 
        opB.active = 1; 
    }

    cvxop_beta_init(&opC, N, dt, bval_weight, verbose);
    // cvxop_beta_init(&opC, N, dt, 1.0, verbose);    
    
    cvxop_bval_init(&opB, N, ind_inv, dt, bval_weight, verbose);
    // cvxop_bval_init(&opB, N, ind_inv, dt, 0.1, verbose);

    cvx_mat G;
    
    cvxmat_alloc(&G, N, 1);
    cvxmat_setvals(&G, 0.0);
    cvxop_init_G(&opG, &G);

	*ddebug = (double *)malloc(48*sizeof(double));
    for (int i = 0; i < 48; i++) {
        (*ddebug)[i] = 0.0;
    }

    cvx_optimize_kernel(&G, &opG, &opD, &opQ, &opC, &opB, N, relax, verbose, bval_reduce, *ddebug, 10);
    
    if (verbose > 0) {
        printf ("\n****************************************\n");
        printf ("--- Finished diff kernel4 #1 in %d iterations  b weight = %.1e", (int)(*ddebug)[0], (*ddebug)[1]);
        printf ("\n****************************************\n");
    }

    double dt2 = 0.1e-3;

    // interp_down(&G, dt, dt2, TE, T_readout);

    // run_kernel_refine(&G, *ddebug, gmax, smax, moment_tols, dt2, TE, 
    //                     T_readout, T_90, T_180, diffmode,
    //                      bval_weight, slew_weight, moments_weight, 
    //                     10.0);

    // cvxop_bval_reweight(&opB, 5.0);
    // cvx_optimize_kernel(&G, &opG, &opD, &opQ, &opC, &opB, N, relax, verbose, 0.5, *ddebug);
    
    // if (verbose > 0) {
    //     printf ("\n****************************************\n");
    //     printf ("--- Finished diff kernel4 #2 in %d iterations  b weight = %.1e", (int)(*ddebug)[0], (*ddebug)[1]);
    //     printf ("\n****************************************\n");
    // }

    // cvx_optimize_kernel(&G, &opG, &opD, &opQ, &opC, &opB, N, relax, verbose, 0.7, *ddebug);
    
    // if (verbose > 0) {
    //     printf ("\n****************************************\n");
    //     printf ("--- Finished diff kernel4 #3 in %d iterations  b weight = %.1e", (int)(*ddebug)[0], (*ddebug)[1]);
    //     printf ("\n****************************************\n");
    // }


    *N_out = G.rows;
    *G_out = G.vals;

    (*ddebug)[2] = opB.weight;
    (*ddebug)[3]  = opQ.weight;
    (*ddebug)[4]  = opD.weight;

    (*ddebug)[10] = opQ.norms.vals[0];
    (*ddebug)[11] = opQ.norms.vals[1];
    (*ddebug)[12] = opQ.norms.vals[2];

    cvxop_gradient_destroy(&opG);
    cvxop_slewrate_destroy(&opD);
    cvxop_moments_destroy(&opQ);
    cvxop_beta_destroy(&opC);
    cvxop_bval_destroy(&opB);

    fflush(stdout);

}




void run_kernel_diff(double **G_out, int *N_out, double gmax, double smax, double m0_tol, double m1_tol, double m2_tol, double TE, double T_readout, double T_90, double T_180, double dt, int diffmode)
{
    // struct timespec ts0;
    // struct timespec ts1;

    double relax = 1.9;
    int verbose = 1;

    int N = round((TE-T_readout) * 1.0e-3/dt);
    if (N < 5) {
        printf ("\nWARNING: N = %d looks too small, setting to 5\n\n", N);
        N = 5;
    }
    int ind_inv = round((N + T_readout/(dt*1.0e3))/2.0);
    
    // I think these should be ceil instead of floor, but just matching matlab code for now
    int ind_end90 = ceil(T_90*(1e-3/dt));
    int ind_start180 = ind_inv - ceil(T_180*(1e-3/dt/2));
    int ind_end180 = ind_inv + ceil(T_180*(1e-3/dt/2));

    if (verbose > 0) {
        printf ("\nN = %d  ind_inv = %d\n90_zeros = %d:%d    180_zeros = %d:%d\n\n", N, ind_inv, 0, ind_end90, ind_start180, ind_end180);
    }

    cvxop_gradient opG;
    cvxop_gradient_init(&opG, N, dt, gmax, ind_inv, verbose);
    cvxop_gradient_setFixRange(&opG, 0, ind_end90, 0.0);
    cvxop_gradient_setFixRange(&opG, ind_start180, ind_end180, 0.0);

    cvxop_slewrate opD;
    cvxop_slewrate_init(&opD, N, dt, smax, 1.0, verbose);

    cvxop_moments opQ;
    cvxop_moments_init_old(&opQ, N, ind_inv, dt,
                        m0_tol, m1_tol, m2_tol, 
                        verbose);
    
    cvxop_beta opC;
    cvxop_beta_init(&opC, N, dt, 1.0, verbose);
    
    // clock_gettime(CLOCK_MONOTONIC, &ts0);
    cvxop_bval opB;
    cvxop_bval_init(&opB, N, ind_inv, dt, 1.0, verbose);
    // clock_gettime(CLOCK_MONOTONIC, &ts1);

    if (diffmode == 1) {
        opB.active = 0; 
    } else if (diffmode == 2) {
        opC.active = 0; 
    }
    

    cvx_mat G;

    double *ddebug;
	ddebug = (double *)malloc(48*sizeof(double));
    
    cvx_optimize_kernel(&G, &opG, &opD, &opQ, &opC, &opB, N, relax, verbose, 0.75, ddebug, 8);
    

    *N_out = G.rows;
    *G_out = G.vals;

    // double elapsed;
    // elapsed = (ts1.tv_sec - ts0.tv_sec);
    // elapsed += (ts1.tv_nsec - ts0.tv_nsec) / 1000000000.0;
    // if (verbose > 0) {
    //     printf ("Elapsed Time = %.4f ms\n", 1000.0*elapsed);
    // }

    cvxop_gradient_destroy(&opG);
    cvxop_slewrate_destroy(&opD);
    cvxop_moments_destroy(&opQ);
    cvxop_beta_destroy(&opC);
    cvxop_bval_destroy(&opB);

    free(ddebug);

    fflush(stdout);

}


void run_kernel_diff2(double **G_out, int *N_out, double gmax, double smax, 
                      double m0_tol, double m1_tol, double m2_tol, double TE, 
                      double T_readout, double T_90, double T_180, double dt, int diffmode,
                      double bval_weight)
{
    double relax = 1.9;
    int verbose = 1;

    int N = round((TE-T_readout) * 1.0e-3/dt);
    if (N < 5) {
        printf ("\nWARNING: N = %d looks too small, setting to 5\n\n", N);
        N = 5;
    }
    int ind_inv = round((N + T_readout/(dt*1.0e3))/2.0);
    
    // I think these should be ceil instead of floor, but just matching matlab code for now
    int ind_end90 = ceil(T_90*(1e-3/dt));
    int ind_start180 = ind_inv - ceil(T_180*(1e-3/dt/2));
    int ind_end180 = ind_inv + ceil(T_180*(1e-3/dt/2));

    if (verbose > 0) {
        printf ("\nN = %d  ind_inv = %d\n90_zeros = %d:%d    180_zeros = %d:%d\n\n", N, ind_inv, 0, ind_end90, ind_start180, ind_end180);
    }

    cvxop_gradient opG;
    cvxop_gradient_init(&opG, N, dt, gmax, ind_inv, verbose);
    cvxop_gradient_setFixRange(&opG, 0, ind_end90, 0.0);
    cvxop_gradient_setFixRange(&opG, ind_start180, ind_end180, 0.0);

    cvxop_slewrate opD;
    cvxop_slewrate_init(&opD, N, dt, smax, 1.0, verbose);

    cvxop_moments opQ;
    cvxop_moments_init_old(&opQ, N, ind_inv, dt,
                        m0_tol, m1_tol, m2_tol, 
                        verbose);
    
    cvxop_beta opC;
    cvxop_beta_init(&opC, N, dt, 1.0, verbose);
    
    cvxop_bval opB;
    cvxop_bval_init(&opB, N, ind_inv, dt, bval_weight, verbose);

    if (diffmode == 1) {
        opB.active = 0; 
    } else if (diffmode == 2) {
        opC.active = 0; 
    }
    

    cvx_mat G;
    
    double *ddebug;
	ddebug = (double *)malloc(48*sizeof(double));

    cvx_optimize_kernel(&G, &opG, &opD, &opQ, &opC, &opB, N, relax, verbose, 0.75, ddebug, 8);
    

    *N_out = G.rows;
    *G_out = G.vals;

    // double elapsed;
    // elapsed = (ts1.tv_sec - ts0.tv_sec);
    // elapsed += (ts1.tv_nsec - ts0.tv_nsec) / 1000000000.0;
    // if (verbose > 0) {
    //     printf ("Elapsed Time = %.4f ms\n", 1000.0*elapsed);
    // }

    cvxop_gradient_destroy(&opG);
    cvxop_slewrate_destroy(&opD);
    cvxop_moments_destroy(&opQ);
    cvxop_beta_destroy(&opC);
    cvxop_bval_destroy(&opB);
    free(ddebug);

    fflush(stdout);

}



void run_kernel_diff3(double **G_out, int *N_out, double gmax, double smax, 
                      double m0_tol, double m1_tol, double m2_tol, double TE, 
                      double T_readout, double T_90, double T_180, double dt, int diffmode,
                      double bval_weight)
{
    double relax = 1.9;
    int verbose = 1;

    int N = round((TE-T_readout) * 1.0e-3/dt);
    if (N < 5) {
        printf ("\nWARNING: N = %d looks too small, setting to 5\n\n", N);
        N = 5;
    }
    int ind_inv = round((N + T_readout/(dt*1.0e3))/2.0);
    
    // I think these should be ceil instead of floor, but just matching matlab code for now
    int ind_end90 = ceil(T_90*(1e-3/dt));
    int ind_start180 = ind_inv - ceil(T_180*(1e-3/dt/2));
    int ind_end180 = ind_inv + ceil(T_180*(1e-3/dt/2));

    if (verbose > 0) {
        printf ("\nN = %d  ind_inv = %d\n90_zeros = %d:%d    180_zeros = %d:%d\n\n", N, ind_inv, 0, ind_end90, ind_start180, ind_end180);
    }

    cvxop_gradient opG;
    cvxop_gradient_init(&opG, N, dt, gmax, ind_inv, verbose);
    cvxop_gradient_setFixRange(&opG, 0, ind_end90, 0.0);
    cvxop_gradient_setFixRange(&opG, ind_start180, ind_end180, 0.0);

    cvxop_slewrate opD;
    cvxop_slewrate_init(&opD, N, dt, smax, 1.0, verbose);

    cvxop_moments opQ;
    cvxop_moments_init_old(&opQ, N, ind_inv, dt,
                        m0_tol, m1_tol, m2_tol, 
                        verbose);
    
    cvxop_beta opC;
    cvxop_beta_init(&opC, N, dt, 1.0, verbose);
    
    cvxop_bval opB;
    cvxop_bval_init(&opB, N, ind_inv, dt, bval_weight, verbose);

    if (diffmode == 1) {
        opB.active = 0; 
    } else if (diffmode == 2) {
        opC.active = 0; 
    }
    

    cvx_mat G;
    
    double *ddebug;
	ddebug = (double *)malloc(48*sizeof(double));

    cvx_optimize_kernel(&G, &opG, &opD, &opQ, &opC, &opB, N, relax, verbose, -1.0, ddebug, 8);
    
    if (verbose > 0) {
        printf ("\n--- Finished diff kernel3 in %d iterations\n\n", (int)ddebug[0]);
    }

    *N_out = G.rows;
    *G_out = G.vals;

    cvxop_gradient_destroy(&opG);
    cvxop_slewrate_destroy(&opD);
    cvxop_moments_destroy(&opQ);
    cvxop_beta_destroy(&opC);
    cvxop_bval_destroy(&opB);
    free(ddebug);

    fflush(stdout);

}



int main (void)
{
    printf ("In optimize_kernel.c main function\n");
    

    // 1 = betamax
    // 2 = bval max
    int diffmode = 1;

    double *G;
    int N;
    double *debug;

    double m_tol[3]={0.0, 0.0, 0.0};

    run_kernel_diff4(&G, &N, &debug, 0.04, 10.0, m_tol, 60.0, 12.0, 4.0, 8.0, diffmode, 0.1, 1.0, 10.0, 20.0, 512);

    // run_kernel_diff4(&G, &N, &debug, 0.04, 100.0, m_tol, 200.0, 12.0, 4.0, 8.0, diffmode, 0.1, 1.0, 10.0, 20.0, 512);

    return 0;
}
