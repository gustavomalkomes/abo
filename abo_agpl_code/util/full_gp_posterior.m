% Get latent mean and (possibly approximate) full latent covariance
% for a set of test points using GPML.

function [mu, Sigma, posterior] = full_gp_posterior(hyperparameters, inference_method, ...
          mean_function, covariance_function, likelihood, x, y, ...
          x_star)

  K      = feval(covariance_function{:}, hyperparameters.cov, x_star);
  K_star = feval(covariance_function{:}, hyperparameters.cov, x, x_star);

  % find posterior latent mean and posterior structure
  [~, ~, mu, ~, ~, posterior] = gp(hyperparameters, inference_method, ...
          mean_function, covariance_function, likelihood, x, y, x_star);

  if (nargout == 1)
    return;
  end

  % check whether posterior.L is a Cholesky decomposition
  is_chol = (isreal(diag(posterior.L))  && ...
             all(diag(posterior.L) > 0) && ...
             all(all(tril(posterior.L, -1) == 0)));

  % find posterior covariance
  if (is_chol)
    % posterior.L contains chol(sqrt(W) * K * sqrt(W) + I)
    K_star = bsxfun(@times, K_star, posterior.sW);
    Sigma = K - K_star' * solve_chol(posterior.L, K_star);
  else
    % posterior.L contains -inv(K + inv(W))
    Sigma = K + K_star' * posterior.L * K_star;
  end

end