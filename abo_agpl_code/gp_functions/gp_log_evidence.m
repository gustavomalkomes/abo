% Laplace approximation to model evidence is
%
%   log Z ~ L(\hat{\theta}) + (d / 2) log(2\pi) - (1 / 2) log det H,
%
% where d is the dimension of \theta and H is the negative Hessian
% of L evaluated at \hat{\theta}

function log_evidence = gp_log_evidence(model)

% model L is the cholesky decomposition of the Hessian matrix
d = size(model.L,1);

% computing the log evidence
% applying the following trick: log(det(HnlZ.H))/2 = sum(log(diag(chol(HnlZ.H))))
% log_evidence = -nlZ + (d/2)*log(2*pi) - sum(log(diag(chol(HnlZ.value))));

% precompting log(2\pi) / 2
half_log_2pi = 0.918938533204673;
log_evidence = -model.nlZ + d * half_log_2pi - sum(log(diag(model.L)));

end