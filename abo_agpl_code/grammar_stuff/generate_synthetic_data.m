function [y, true_theta] = generate_synthetic_data(x,gp, varargin)

if isempty(varargin)
    true_theta = independent_prior(gp.parameters.priors);
else
   true_theta  = varargin{1};
end

true_mean_function        = gp.parameters.mean_function;
true_covariance_function  = gp.parameters.covariance_function;

mu = feval(true_mean_function{:},true_theta.mean, x);
K  = feval(true_covariance_function{:}, true_theta.cov,  x);

K  = fix_pd_matrix(K);

n = size(x,1);
y  = mu + chol(K)' * randn(n, 1) + exp(true_theta.lik) * randn(n, 1);
