function result = isotropic_rq_covariance(hyperparameters, x, z, i,j)

if (nargin <= 1)
    result = covRQiso();
elseif (nargin == 2)
    result = covRQiso(hyperparameters, x);
elseif (nargin == 3)
    result = covRQiso(hyperparameters, x, z);
elseif (nargin == 4)
    result = covRQiso(hyperparameters, x, z, i);
else
    
    % ensure i <= j by exploiting symmetry
    if (i > j)
        result = isotropic_rq_covariance(hyperparameters, x, z, j, i);
        return;
    end
    
    % Hessians involving the log output scale
    if (i == 2 || j == 2)
        if (i == 2)
            result = 2 * covRQiso(hyperparameters, x, z, j);
            return;
        else
            result = 2 * covRQiso(hyperparameters, x, z, i);
            return; 
        end
    end
    
    ell   = exp(hyperparameters(1));
    sf2   = exp(2*hyperparameters(2));
    alpha = exp(hyperparameters(3));
    
    if (isempty(z))
        D2  = sq_dist(x'/ell);
        d2_n  = sq_dist(x');
    else
        D2   = sq_dist(x'/ell,z'/ell);
        d2_n = sq_dist(x',z');
    end
    
    K = covRQiso(hyperparameters, x, z, 1);
    L = (1+0.5*D2/alpha);
    
    if (j==1) % H(1,1)
        % Hessians involving ell
        result = -2.*K + ((alpha+1)*sf2/alpha)*(L.^(-alpha-2)).*(D2.^2);
        
    elseif j==3
        % Hessians involving alpha
        if i==1 % H(1,3)
            result = K.*(D2.*(alpha+1)./(2*alpha*L)-alpha*log(L));
        else    % H(3,3)
            a      = (0.5*D2./L - alpha*log(L)).^2;            
            b      = (0.25*(D2./L).^2./alpha + 0.5*D2./L - alpha*log(L));
            result = sf2*L.^(-alpha).*(a+b);
        end
    else
        error('Unknown hyperparameter')
    end
    
end

end