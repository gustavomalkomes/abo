clear all;

% rng(0);

%% parameters
g_tol   = 1E-5;      % gradient/hessian tiny step
e_tol   = 5E-3;      % error tolerance
n       = 10;        % number of points
xx      = rand(n,1); % random points
zz      = rand(n,1);

%% checking linear_covariance
n_param = 1;
fun_fx = @(hyp,i)   linear_covariance(hyp,xx,zz,i);
fun_dx = @(hyp,i,j) linear_covariance(hyp,xx,zz,i,j);

[diff,a,b] = check_hessian(fun_fx,fun_dx,randn(n_param,1),g_tol,n);
assert(diff<e_tol,'linear_covariance Hessian doesnt match');

%% checking isotropic_rq_covariance
n_param = 3;
fun_fx = @(hyp,i)   isotropic_rq_covariance(hyp,xx,zz,i);
fun_dx = @(hyp,i,j) isotropic_rq_covariance(hyp,xx,zz,i,j);

[diff,a,b] = check_hessian(fun_fx,fun_dx,randn(n_param,1),g_tol,n);
assert(diff<e_tol,'isotropic_rq_covariance Hessian doesnt match');


%% checking sum_covariance
n_param = 9;
K1 = {@sum_covariance,{@isotropic_sqdexp_covariance,@isotropic_matern_covariance12}};
K2 = {@sum_covariance,{@isotropic_matern_covariance12,@periodic_covariance}};
K  = {@sum_covariance,{K1,K2}};

fun_fx = @(hyp,i)   feval(K{:}, hyp, xx, zz, i);
fun_dx = @(hyp,i,j) feval(K{:}, hyp, xx, zz, i, j);

[diff,a,b] = check_hessian(fun_fx,fun_dx,randn(n_param,1),g_tol,n);
assert(diff<e_tol,'sum_covariance Hessian doesnt match');

%% checking prod_covariance
n_param = 9;
K1 = {@prod_covariance,{@isotropic_sqdexp_covariance,@isotropic_matern_covariance12}};
K2 = {@prod_covariance,{@isotropic_matern_covariance12,@periodic_covariance}};
K  = {@prod_covariance,{K1,K2}};

fun_fx = @(hyp,i)   feval(K{:}, hyp, xx, zz, i);
fun_dx = @(hyp,i,j) feval(K{:}, hyp, xx, zz, i, j);

[diff,a,b] = check_hessian(fun_fx,fun_dx,randn(n_param,1),g_tol,n);
assert(diff<e_tol,'prod_covariance Hessian doesnt match');

%% checking periodic_covariance
n_param = 3;
fun_fx = @(hyp,i)   periodic_covariance(hyp,xx,zz,i);
fun_dx = @(hyp,i,j) periodic_covariance(hyp,xx,zz,i,j);

[diff,a,b] = check_hessian(fun_fx,fun_dx,randn(n_param,1),g_tol,n);
assert(diff<e_tol,'periodic_covariance Hessian doesnt match');

%% checking isotropic_matern_covariance
n_param = 2;
fun_fx = @(hyp,i)   isotropic_matern_covariance12(hyp,xx,zz,i);
fun_dx = @(hyp,i,j) isotropic_matern_covariance12(hyp,xx,zz,i,j);

diff = check_hessian(fun_fx,fun_dx,randn(n_param,1),g_tol,n);
assert(diff<e_tol,'isotropic_matern_covariance12 Hessian doesnt match');

%% checking isotropic_matern_covariance
n_param = 2;
fun_fx = @(hyp,i)   isotropic_matern_covariance52(hyp,xx,zz,i);
fun_dx = @(hyp,i,j) isotropic_matern_covariance52(hyp,xx,zz,i,j);

diff = check_hessian(fun_fx,fun_dx,randn(n_param,1),g_tol,n);
assert(diff<e_tol,'isotropic_matern_covariance12 Hessian doesnt match');


%% checking isotropic_sqdexp_covariance
n_param = 2;
fun_fx = @(hyp,i)   isotropic_sqdexp_covariance(hyp,xx,zz,i);
fun_dx = @(hyp,i,j) isotropic_sqdexp_covariance(hyp,xx,zz,i,j);

diff = check_hessian(fun_fx,fun_dx,randn(n_param,1),g_tol,n);
assert(diff<e_tol,'isotropic_sqdexp_covariance Hessian doesnt match');