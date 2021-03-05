function K = linear_covariance(hyp, x, z, i, j)

% hyp(1) output scale
% hyp(2) offset

if nargin<2, K = '1'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

% compute inner products
if dg                                                               % vector kxx
    K = sum(x.*x,2);
else
    if xeqz                                                 % symmetric matrix Kxx
        K = x*x';
    else                                                   % cross covariances Kxz
        K = x*z';
    end
end

s2 = exp(2*hyp(1));                                          % output scale
% b2 = exp(2*hyp(2));                                          % offset

if nargin<4
    K = s2*K;
elseif nargin==4                                                  % derivatives
    if i == 1
        K = 2.*s2.*K;
%     elseif i == 2
%         K = 2.*b2.*ones(size(K));
    else
        error('Unknown hyperparameter')
    end
elseif  (nargin==5)                                                        % hessian
    
    if (i > j) % ensure i <= j by exploiting symmetry
        K = linear_covariance(hyp, x, z, j, i);
    elseif (i==1 && j==1)
        K = 4.*s2.*K;
%     elseif (i==1 && j==2)
%         K = 0.*ones(size(K));
%     elseif (i==2 && j==2)
%         K = 4.*b2.*ones(size(K));
    else
        error('Unknown hyperparameter')
    end
    
end
