import numpy as np
import matplotlib.pyplot as plt

import os
import pickle

import sys

from timeit import default_timer as timer

import cvxg


for MMT in [0, 1, 2]:
    if MMT == 0:
        TE = 70
    elif MMT == 1:
        TE = 90
    else:
        TE = 130

    gmax = .08
    smax = 80


    T_readout = 16
    T_90 = 4
    T_180 = 6

    params = (gmax, smax, MMT, TE, T_readout, T_90, T_180)

    print(params)
    sys.stdout.flush()
    all_res = []

    for diffmode in [1, 2]:

        for N0 in [48, 64, 128, 192, 256, 512, 1024]:
            
            start = timer()
            G, dd = cvxg.run_kernel_fixN(gmax, smax, MMT, TE, T_readout, T_90, T_180, diffmode, N0 = N0)
            end = timer()
            res = (diffmode, 'N', N0, (end-start), G, dd)
            all_res.append(res)

        for dt in [2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.1]:
            dt *= 1e-3
            
            start = timer()
            G, dd = cvxg.run_kernel_fixdt(gmax, smax, MMT, TE, T_readout, T_90, T_180, diffmode, dt = dt)
            end = timer()
            res = (diffmode, 'dt', dt, (end-start), G, dd)
            all_res.append(res)
            
    output = (params, all_res)
    suffix = 0
    filename = 'res4/%.4f_%.4f_%d_%.4f_%.4f_%.4f_%.4f_%d.p' % (gmax, smax, MMT, TE, T_readout, T_90, T_180, suffix)
    while os.path.isfile(filename):
        suffix += 1
        filename = 'res4/%.4f_%.4f_%d_%.4f_%.4f_%.4f_%.4f_%d.p' % (gmax, smax, MMT, TE, T_readout, T_90, T_180, suffix)

    pickle.dump( output, open( filename, "wb" ) )