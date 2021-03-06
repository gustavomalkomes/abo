function [theta,best_iter] = minimize_cks_v2_restart(model,x,y,varargin)

% parse optional inputs
parser = inputParser;

addParamValue(parser, 'initial_hyperparameters', []);
addParamValue(parser, 'num_restarts', 3);
addParamValue(parser, 'length', -200);
addParamValue(parser, 'verbosity', 0);
addParamValue(parser, 'method', 'LBFGS');

parse(parser, varargin{:});
options = parser.Results;

if (isempty(options.initial_hyperparameters))
    initial_hyperparameters = sample_hyperparameters(model);
else
    initial_hyperparameters = options.initial_hyperparameters;
end

[best_theta, best_nlZ,best_iter] = ...
    minimize_v2(initial_hyperparameters, @gp, options, model.inference_method, ...
    model.mean_function, model.covariance_function, model.likelihood, x, y);

for i = 1:options.num_restarts
    theta = sample_hyperparameters(model);
    
    [theta_values, nlZ,iter] = ...
        minimize_v2(theta, @gp, options, model.inference_method, ...
        model.mean_function, model.covariance_function, model.likelihood, x, y);
    
    if (nlZ(end) < best_nlZ(end))
        best_nlZ = nlZ;
        best_theta = theta_values;
        best_iter  = iter;
    end
end

theta = best_theta;
end


function hyp = sample_hyperparameters(model)
fixed_hyp     = model.fixed_hyperparameters.cov;
hyp           = model.prior();
ind           = fixed_hyp ~= inf;
hyp.cov(ind)  = fixed_hyp(ind);
end