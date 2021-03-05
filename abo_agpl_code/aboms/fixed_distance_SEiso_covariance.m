% ....
function result = fixed_distance_SEiso_covariance(K, theta, x, z, i, j)

persistent fixed_cov;

% call covSEiso for everything but Hessian calculation
if (nargin == 1 && ~isempty(K))
    fixed_cov = K;
elseif (nargin <= 2)
    result = fixed_SEiso;
elseif (nargin == 3)
    result = fixed_SEiso(fixed_cov,theta, x);
elseif (nargin == 4)
    result = fixed_SEiso(fixed_cov, theta, x, z);
elseif (nargin == 5)
    result = fixed_SEiso(fixed_cov, theta, x, z, i);
    % Hessian with respect to \theta_i \theta_j
else
    
    % ensure i <= j by exploiting symmetry
    if (i > j)
        result = fixed_distance_SEiso_covariance(fixed_cov, theta, x, z, j, i);
        return;
    end
    
    % Hessians involving the log output scale
    if (j == 2)
        result = 2 * fixed_SEiso(fixed_cov, theta, x, z, i);
        return;
    end
    
    K = fixed_SEiso(fixed_cov, theta, x, z, 1);
    
    if (isempty(z))
        z = x;
    end
    
    ell = exp(-2*theta(1));
    
    factor = fixed_cov(x,z)*ell;
    
    result = (factor - 2) .* K;
end

end

function K = fixed_SEiso(fixed_cov, hyp, x, z, i)

if nargin<3, K = '2'; return; end                  % report number of parameters
if nargin<4, z = []; end
% make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

ell = exp(2*hyp(1));                                 % characteristic length scale
sf2 = exp(2*hyp(2));                                           % signal variance

% precompute squared distances
if dg                                                               % vector kxx
    K = fixed_cov(sub2ind(size(fixed_cov), x, x));
else
    if xeqz                                                 % symmetric matrix Kxx
        K = fixed_cov(x,x);
    else                                                   % cross covariances Kxz
        K = fixed_cov(x,z);
    end
end

K = K/ell;

if nargin<5                                                        % covariances
    K = sf2*exp(-K/2);
else                                                               % derivatives
    if i==1
        K = sf2*exp(-K/2).*K;
    elseif i==2
        K = 2*sf2*exp(-K/2);
    else
        error('Unknown hyperparameter')
    end
end

end