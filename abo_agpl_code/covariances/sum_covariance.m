function K = sum_covariance(cov, hyp, x, z, i, j)

if (nargin <= 1)
    K = covSum(cov);
elseif (nargin == 2)
    K = covSum(cov,hyp);
elseif (nargin == 3)
    K = covSum(cov,hyp,x);
elseif (nargin == 4)
    K = covSum(cov,hyp,x,z);
elseif (nargin == 5)
    K = covSum(cov,hyp,x,z,i);
else
    
    if (i > j)
        K = sum_covariance(cov,hyp, x, z, j, i);
        return;
    end
    
    for ii = 1:numel(cov)                          % iterate over covariance functions
        f = cov(ii); if iscell(f{:}), f = f{:}; end   % expand cell array if necessary
        nh(ii) = cellstr(feval(f{:}));                         % collect number hypers
    end
    
    v = [];                 % v vector indicates to which covariance parameters belong
    for ii = 1:length(cov),
        v = [v repmat(ii, 1, eval(char(nh(ii))))];
    end
    
    % hessian
    
    if j<=length(v)
        vi = v(i);                                        % which covariance function
        vj = v(j);                                        % which covariance function
        ki = sum(v(1:i)==vi);                    % which parameter in that covariance
        kj = sum(v(1:j)==vj);                    % which parameter in that covariance
        
        if (vi~=vj)
            K = zeros(size(x,1));
            return;
        end
        f  = cov(vj);
        if iscell(f{:}), f = f{:}; end          % dereference cell array if necessary
        K = feval(f{:}, hyp(v==vj), x, z, ki,kj);                % compute derivative
    else
        error('Unknown hyperparameter')
    end
        
    
end

end