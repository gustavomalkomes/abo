function K = prod_covariance_fast(cov, hyp, x, z, i, j)

if (nargin <= 1)
    K = covProd_fast(cov);
elseif (nargin == 2)
    K = covProd_fast(cov,hyp);
elseif (nargin == 3)
    K = covProd_fast(cov,hyp,x);
elseif (nargin == 4)
    K = covProd_fast(cov,hyp,x,z);
elseif (nargin == 5)
    K = covProd_fast(cov,hyp,x,z,i);
else
    
    if (i > j)
        K = prod_covariance_fast(cov,hyp, x, z, j, i);
        return;
    end
    
    v = cov{2}{2};
    cov = cov{1};
    
    % hessian
    if j<=length(v)
        K = 1; 
        vi = v(i);                                        % which covariance function
        vj = v(j);                                        % which covariance function
        ki = sum(v(1:i)==vi);                    % which parameter in that covariance
        kj = sum(v(1:j)==vj);                    % which parameter in that covariance
        for ii = 1:length(cov)                      % iteration over factor functions
            f = cov(ii); if iscell(f{:}), f = f{:}; end    % expand cell if necessary
            if (vi==vj)
                if ii==vj
                    K = K .* feval(f{:}, hyp(v==ii), x, z, ki, kj); 
                else
                    K = K .* feval(f{:}, hyp(v==ii), x, z);         
                end
            else
                if ii==vi
                    K = K .* feval(f{:}, hyp(v==ii), x, z, ki); % accumulate covariances
                else
                    K = K .* feval(f{:}, hyp(v==ii), x, z, kj);        
                end 
            end
        end
    else
        error('Unknown hyperparameter')
    end
    
    
end

end