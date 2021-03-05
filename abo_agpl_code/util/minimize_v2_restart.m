function [best_hyperparameters, best_nlZ, best_iter] = minimize_v2_restart(model,x,y,varargin)

% parse optional inputs
parser = inputParser;

addParamValue(parser, 'initial_hyperparameters', []);
addParamValue(parser, 'num_restarts', 3);
addParamValue(parser, 'minimize_options', ...
    struct('verbosity', 0, ...
    'method', 'LBFGS', ...
    'length', -500));


parse(parser, varargin{:});
options = parser.Results;

if (isempty(options.initial_hyperparameters))
    initial_hyperparameters = model.prior();
else
    initial_hyperparameters = options.initial_hyperparameters;
end

f = @(hyperparameter_values) gp_optimizer_wrapper(hyperparameter_values, ...
    initial_hyperparameters, model.inference_method, ...
    model.mean_function, model.covariance_function, model.likelihood, ...
    x, y);

[best_hyperparameter_values, best_nlZ, best_iter] = ...
    minimize_v2(unwrap(initial_hyperparameters), f, options.minimize_options);

for i = 1:options.num_restarts
    hyperparameters = model.prior();
    
    [hyperparameter_values, nlZ, iter] = ...
        minimize_v2(unwrap(hyperparameters), f, options.minimize_options);
    
    if (nlZ(end) < best_nlZ)
        best_nlZ = nlZ;
        best_hyperparameter_values = hyperparameter_values;
        best_iter = iter;
    end
end

best_hyperparameters = rewrap(initial_hyperparameters, ...
    best_hyperparameter_values);
