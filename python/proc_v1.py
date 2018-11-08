import numpy as np
import matplotlib.pyplot as plt

import os
import pickle

import sys

from timeit import default_timer as timer

import cvxg


for ii in range(10000):

    MMT = np.random.randint(0, 2 + 1)
    if MMT == 0:
        TE = np.random.uniform(50.0, 110.0)
    elif MMT == 1:
        TE = np.random.uniform(55.0, 170.0)
    else:
        TE = np.random.uniform(60.0, 200.0)



    gmax = np.random.uniform(.04, .16)
    smax = np.random.uniform(20.0, 400.0)
    
    
    T_readout = np.random.uniform(8, 24)
    T_90 = np.random.uniform(3, 6)
    T_180 = np.random.uniform(4, 8)

    params = (gmax, smax, MMT, TE, T_readout, T_90, T_180)

    print(ii, params)
    sys.stdout.flush()
    all_res = []

    for diffmode in [1, 2, 3]:
        for N0 in [48, 64, 128, 192, 256, 512, 768, 1024, 2048]:
            
            if diffmode == 2 and N0 > 600:
                continue

            start = timer()
            G, dd = cvxg.run_kernel_fixN(gmax, smax, MMT, TE, T_readout, T_90, T_180, diffmode, N0 = N0)
            end = timer()
            res = (diffmode, 'N', N0, (end-start), G, dd)
            all_res.append(res)

        for dt in [2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.1, .05]:
            
            if diffmode == 2 and dt < 0.4:
                continue
            
            dt *= 1e-3
            
            start = timer()
            G, dd = cvxg.run_kernel_fixdt(gmax, smax, MMT, TE, T_readout, T_90, T_180, diffmode, dt = dt)
            end = timer()
            res = (diffmode, 'dt', dt, (end-start), G, dd)
            all_res.append(res)
            
    output = (params, all_res)
    suffix = 0
    filename = 'res6/%.4f_%.4f_%d_%.4f_%.4f_%.4f_%.4f_%d.p' % (gmax, smax, MMT, TE, T_readout, T_90, T_180, suffix)
    while os.path.isfile(filename):
        suffix += 1
        filename = 'res6/%.4f_%.4f_%d_%.4f_%.4f_%.4f_%.4f_%d.p' % (gmax, smax, MMT, TE, T_readout, T_90, T_180, suffix)

    pickle.dump( output, open( filename, "wb" ) )