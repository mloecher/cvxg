#include <mex.h>
#include "matrix.h"
#include "optimize_kernel.c"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *temp;
    double *G;
    int N;
    
    double gmax = mxGetScalar(prhs[0]);
    double smax = mxGetScalar(prhs[1]);
    double m0_tol = mxGetScalar(prhs[2]);
    double m1_tol = mxGetScalar(prhs[3]);
    double m2_tol = mxGetScalar(prhs[4]);
    double TE = mxGetScalar(prhs[5]);
    double T_readout = mxGetScalar(prhs[6]);
    double T_90 = mxGetScalar(prhs[7]);
    double T_180 = mxGetScalar(prhs[8]);
    double dt = mxGetScalar(prhs[9]);
    int diffmode = mxGetScalar(prhs[10]);
    
    run_kernel_diff(&G, &N, gmax, smax, m0_tol, m1_tol, m2_tol, TE, T_readout, T_90, T_180, dt, diffmode);
    
    plhs[0] = mxCreateDoubleMatrix(1,N,mxREAL);

    // Why doesnt this work?: mxSetPr(plhs[0], G);
    temp = mxGetPr(plhs[0]);
    for (int i = 0; i < N; i++) {
        temp[i] = G[i];
    }
        

}