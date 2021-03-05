function result = periodic_covariance(hyperparameters, x, z, i, j)

if (nargin <= 1)
    result = covPeriodic();
elseif (nargin == 2)
    result = covPeriodic(hyperparameters, x);
elseif (nargin == 3)
    result = covPeriodic(hyperparameters, x, z);
elseif (nargin == 4)
    result = covPeriodic(hyperparameters, x, z, i);
else
    
    % ensure i <= j by exploiting symmetry
    if (i > j)
        result = periodic_covariance(hyperparameters, x, z, j, i);
        return;
    end
    
    % Hessians involving the log output scale
    if (j == 3)
        result = 2 * covPeriodic(hyperparameters, x, z, i);
        return;
    end
    
    if (isempty(z))
        z = x;
    end
    
    ell   = exp(hyperparameters(1));
    p     = exp(hyperparameters(2));
    
    dist  = sqrt(sq_dist(x', z'));
    alpha = (pi.*dist)./p;
    
    dk = covPeriodic(hyperparameters, x, z, j);
    
    if (i == 2 && j==2)

        K           = covPeriodic(hyperparameters, x, z);
        const       = 4*(pi*dist).^2;
        
        factor      = (dk./(K + eps) - 1); %% FIX THIS
        
        extra_term  = const.*K.*cos(2*alpha)./(ell*p)^2;
        result      = dk.*factor - extra_term;
    else
        factor = 2.*sin(alpha)./ell;
        result = dk.*(factor.*factor-2);
    end
    
    
end

end