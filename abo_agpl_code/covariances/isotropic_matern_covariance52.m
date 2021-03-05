function result = isotropic_matern_covariance52(hyperparameters, x, z, i, j)

d = 5;

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
        result = isotropic_matern_covariance52(hyperparameters, x, z, j, i);
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
    theta_one = hyperparameters(1);
    theta_two = hyperparameters(2);
    exponent  = -sqrt(5)*dist.*exp(-theta_one)-3*theta_one+2*theta_two;
    result    = (sqrt(5)*dist+exp(theta_one))*(5/3).*(dist.^2).*exp(exponent);
end


end