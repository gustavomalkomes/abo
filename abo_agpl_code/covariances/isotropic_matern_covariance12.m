function result = isotropic_matern_covariance12(hyperparameters, x, z, i, j)

% TODO fix this for multiple values of d
d = 1;

if (nargin <= 1)
    result = covMaterniso(d);
elseif (nargin == 2)
    result = covMaterniso(d,hyperparameters, x);
elseif (nargin == 3)
    result = covMaterniso(d,hyperparameters, x, z);
elseif (nargin == 4)
    result = covMaterniso(d,hyperparameters, x, z, i);
else
    
    % ensure i <= j by exploiting symmetry
    if (i > j)
        result = isotropic_matern_covariance12(hyperparameters, x, z, j, i);
        return;
    end
    
    % Hessians involving the log output scale
    if (j == 2)
        result = 2 * covMaterniso(d,hyperparameters, x, z, i);
        return;
    end

    if (isempty(z))
        dist   = sqrt(sq_dist(x'));
    else
        dist   = sqrt(sq_dist(x',z'));
    end

    % Hessians involving the log length scale
    dK = covMaterniso(d,hyperparameters, x, z, 1);

    factor = dist*exp(-hyperparameters(1));
    result = (factor - 1) .* dK;
    
end


end