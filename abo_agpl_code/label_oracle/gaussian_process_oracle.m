function label = gaussian_process_oracle(problem, x, y, x_star, gp)

if ~isempty(problem)
    x = [problem.label_initial_x; x];
    y = [problem.label_initial_y; y];
end

if isempty(y)
    m     = feval(gp.mean_function{:}, gp.theta.mean, x_star);
    K     = feval(gp.covariance_function{:}, gp.theta.cov, x_star);
    label = normrnd(m,sqrt(K));    
    return
end

% mean function
m = feval(gp.mean_function{:}, gp.theta.mean, x);
ms = feval(gp.mean_function{:}, gp.theta.mean, x_star);

% K(X, X)
Kxx = feval(gp.covariance_function{:}, gp.theta.cov, x);
Kxx = fix_pd_matrix(Kxx);
% K(X, X_*)
Kxs = feval(gp.covariance_function{:}, gp.theta.cov,  x, x_star);
% K(X_*, K_*)
Kss = feval(gp.covariance_function{:}, gp.theta.cov,  x_star);

% posterior distribution for y*
posterior_mu = ms + Kxs' / Kxx * (y - m);
posterior_K  = Kss - Kxs' / Kxx * Kxs;
posterior_K  = (posterior_K + posterior_K') / 2;
posterior_K  = fix_pd_matrix(posterior_K);

label = mvnrnd(posterior_mu, posterior_K, 1)';
end
