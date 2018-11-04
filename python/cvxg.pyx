import numpy as np
cimport numpy as np
import time

cdef extern from "../src/optimize_kernel.c":
    void _run_kernel_diff_fixeddt "run_kernel_diff_fixeddt"(double **G_out, int *N_out, double **ddebug,
                                                        double dt0, double gmax, double smax, 
                                                        double *moment_tols, double TE, 
                                                        double T_readout, double T_90, double T_180, int diffmode,
                                                        double bval_weight, double slew_weight, double moments_weight, 
                                                        double bval_reduce,  double dt_out)

    void _run_kernel_diff_fixedN "run_kernel_diff_fixedN"(double **G_out, int *N_out, double **ddebug,
                                                        int N0, double gmax, double smax, 
                                                        double *moment_tols, double TE, 
                                                        double T_readout, double T_90, double T_180, int diffmode,
                                                        double bval_weight, double slew_weight, double moments_weight, 
                                                        double bval_reduce,  double dt_out)


def run_kernel_fixN(gmax, smax, MMT, TE, T_readout, T_90, T_180, diffmode, bval_weight = 1.0, slew_weight = 1.0, moments_weight = 10.0, N0 = 64, bval_reduce = 10.0, dt_out = -1.0):

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

    _run_kernel_diff_fixedN(&G_out, &N_out, &ddebug, N0, gmax, smax, &m_tol_c[0], TE, T_readout, T_90, T_180, diffmode, bval_weight, slew_weight, moments_weight, bval_reduce, dt_out)

    G_return = np.zeros(N_out)
    for i in range(N_out):
        G_return[i] = G_out[i]

    debug_out = np.zeros(48)
    for i in range(48):
        debug_out[i] = ddebug[i]

    return G_return, debug_out


def run_kernel_fixdt(gmax, smax, MMT, TE, T_readout, T_90, T_180, diffmode, bval_weight = 1.0, slew_weight = 1.0, moments_weight = 10.0, dt = 0.4e-3, bval_reduce = 10.0, dt_out = -1.0):

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

    _run_kernel_diff_fixeddt(&G_out, &N_out, &ddebug, dt, gmax, smax, &m_tol_c[0], TE, T_readout, T_90, T_180, diffmode, bval_weight, slew_weight, moments_weight, bval_reduce, dt_out)

    G_return = np.zeros(N_out)
    for i in range(N_out):
        G_return[i] = G_out[i]

    debug_out = np.zeros(48)
    for i in range(48):
        debug_out[i] = ddebug[i]

    return G_return, debug_out