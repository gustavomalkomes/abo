function ei = compute_ei(y_max, mu, sigma)

sigma(sigma<0) = 0;

% compute expected improvement
delta = (mu - y_max);
u     = delta./sigma;
u_pdf = normpdf(u);
u_cdf = normcdf(u);

ei  = delta .* u_cdf + sigma .* u_pdf;
ei(ei<0) = 0;

end
