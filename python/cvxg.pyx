import numpy as np
cimport numpy as np
import time

cdef extern from "../src/optimize_kernel.c":
    void _run_kernel_diff "run_kernel_diff"(double **G_out, int *N_out, double gmax, double smax, double m0_tol, double m1_tol, double m2_tol, 
    double TE, double T_readout, double T_90, double T_180, double dt, int diffmode)

    void _run_kernel_diff2 "run_kernel_diff2"(double **G_out, int *N_out, double gmax, double smax, double m0_tol, double m1_tol, double m2_tol, 
    double TE, double T_readout, double T_90, double T_180, double dt, int diffmode, double bval_weight)

    void _run_kernel_diff3 "run_kernel_diff3"(double **G_out, int *N_out, double gmax, double smax, double m0_tol, double m1_tol, double m2_tol, 
    double TE, double T_readout, double T_90, double T_180, double dt, int diffmode, double bval_weight)

    void _run_kernel_diff4 "run_kernel_diff4"(double **G_out, int *N_out, double **ddebug, double gmax, double smax, double *moment_tols, 
    double TE, double T_readout, double T_90, double T_180, int diffmode, double bval_weight, double slew_weight, double moments_weight, double bval_reduce, int N0)

def run_kernel(gmax, smax, MMT, TE, T_readout, T_90, T_180, dt, diffmode, bval_weight = 1.0):

    m0_tol = 0.0
    m1_tol = -1.0
    m2_tol = -1.0
    if MMT > 0:
        m1_tol = 0.0
    if MMT > 1:
        m2_tol = 0.0

    cdef double *G_out
    cdef int N_out

    _run_kernel_diff2(&G_out, &N_out, gmax, smax, m0_tol, m1_tol, m2_tol, TE, T_readout, T_90, T_180, dt, diffmode, bval_weight)

    G_return = np.zeros(N_out)
    for i in range(N_out):
        G_return[i] = G_out[i]

    return G_return

def run_kernel3(gmax, smax, MMT, TE, T_readout, T_90, T_180, dt, diffmode, bval_weight = 1.0):

    m0_tol = 0.0
    m1_tol = -1.0
    m2_tol = -1.0
    if MMT > 0:
        m1_tol = 0.0
    if MMT > 1:
        m2_tol = 0.0

    cdef double *G_out
    cdef int N_out

    _run_kernel_diff3(&G_out, &N_out, gmax, smax, m0_tol, m1_tol, m2_tol, TE, T_readout, T_90, T_180, dt, diffmode, bval_weight)

    G_return = np.zeros(N_out)
    for i in range(N_out):
        G_return[i] = G_out[i]

    return G_return


def run_kernel4(gmax, smax, MMT, TE, T_readout, T_90, T_180, diffmode, bval_weight = 1.0, slew_weight = 1.0, moments_weight = 1.0, N0 = 64, bval_reduce = 10.0):

    m_tol = np.array([0.0, -1.0, -1.0])
    if MMT > 0:
        m_tol[1] = 0.0
    if MMT > 1:
        m_tol[2] = 0.0

    cdef double *G_out
    cdef int N_out  
    cdef double *ddebug

    m_tol = np.ascontiguousarray(np.ravel(m_tol), np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] m_tol_c = m_tol

    _run_kernel_diff4(&G_out, &N_out, &ddebug, gmax, smax, &m_tol_c[0], TE, T_readout, T_90, T_180, diffmode, bval_weight, slew_weight, moments_weight, bval_reduce, N0)

    G_return = np.zeros(N_out)
    for i in range(N_out):
        G_return[i] = G_out[i]

    debug_out = np.zeros(48)
    for i in range(48):
        debug_out[i] = ddebug[i]

    return G_return, debug_out