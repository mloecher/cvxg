%%
src_files = {'cvx_matrix.c'; 'op_gradient.c'; 'op_bval.c'; 'op_beta.c'; 'op_slewrate.c'; 'op_moments.c'};
src_files = sprintf('../src/%s ' ,src_files{:});
src_files = strtrim(src_files);
src_files = ['mex_CVXG.c' ' ' src_files];

inc_path = ['-I' '../src/'];
command = ['mex -v CFLAGS="$CFLAGS -std=c11" ' src_files ' ' inc_path];

eval(command);

%% This is how you run it for a fixed TE
gmax = 0.074;
smax = 50.0;
m0_tol = 0.0;
m1_tol = 0.0;
m2_tol = 0.0;
TE = 60.0;
T_readout = 10.0;
T_90 = 3.0;
T_180 = 12.0;
dt = 0.2e-3;
diffmode = 2;

G = mex_CVXG(gmax, smax, m0_tol, m1_tol, m2_tol, TE, T_readout, T_90, T_180, dt, diffmode);

bval = get_bval(G, T_readout, dt)
moments = get_moments(G, T_readout, dt);


%% 
% This is how you run it for a fixed bval, note that this is a different
% line search than used previously, it should be faster
target_bval = 800;
MMT = 2.0;

[TE0,G0,b0] = design_symmetric_gradients(target_bval,T_readout,T_90,gmax,MMT);

min_TE = T_90 + T_180 + 1.0;
max_TE = TE0;

G = get_min_TE( target_bval, min_TE, max_TE, gmax, smax, m0_tol, m1_tol, m2_tol, T_readout, T_90, T_180, dt, diffmode );

TE_out = numel(G)*dt*1e3+T_readout;
