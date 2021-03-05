% MINIMIZE_MINFUNC optimize GP hyperparameters with random restart.
%
% This implements GP hyperparameter optimization with random
% restart. Each optimization is accomplished using Mark Schmidt's
% minFunc function:
%
%   http://www.di.ens.fr/~mschmidt/Software/minFunc.html
%
% For each restart, a new set of hyperparameters will be drawn from a
% specified hyperparameter prior p(\theta). The user may optionally
% specify the initial hyperparameters to use for the first
% optimization attempt.
%
% Usage
% -----
%
%   [best_hyperparameters, best_nlZ] = minimize_minFunc(model, x, y, varargin)
%
% Required inputs:
%
%   model: a struct describing the GP model, containing fields:
%
%        inference_method: a GPML inference method
%           mean_function: a GPML mean function
%     covariance_function: a GPML covariance function
%              likelihood: a GPML likelihood
%                   prior: a function handle to a hyperparameter prior
%                          p(\theta) (see priors.m in gpml_extensions)
%
%       x: the observation locations (N x D)
%       y: the observation values (N x 1)
%
% Optional inputs (specified as name/value pairs):
%
%   'initial_hyperparameters': a GPML hyperparameter struct specifying
%                              the intial hyperparameters for the
%                              first optimization attempt (if not
%                              specified, will be drawn from the prior)
%
%           'minFunc_options': a struct containing options to pass to
%                              minFunc when optimizing the log
%                              posterior, default:
%
%                               .Display     = 'off'
%                               .MaxFunEvals = 300
%
%             'num_restarts': the number of random restarts to use
%                             when optimizing the log posterior,
%                             default: 1
%
%                             Note: this specifies the number of
%                             _re_starts; at least one optimization
%                             call will always be made.
%
% See also MINFUNC.

% Copyright (c) 2014 Roman Garnett

function [best_hyperparameters, best_nlZ, best_minFunc_output] = minimize_minFunc(model, ...
    x, y, varargin)

% parse optional inputs
parser = inputParser;

addParamValue(parser, 'initial_hyperparameters', []);
addParamValue(parser, 'num_restarts', 3);
addParamValue(parser, 'minFunc_options', ...
    struct('Display', 'off', ...
    'MaxIter', 500,   ...
    'method', 'lbfgs'));

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

[best_hyperparameter_values, best_nlZ, exitflag, best_minFunc_output] = ...
    minFunc(f, unwrap(initial_hyperparameters), options.minFunc_options);

if exitflag < 0
    warning('MinFunc failed trying to optimize initial_hyperparameters. Random restart\n');
    options.num_restarts = options.num_restarts + 1;
    best_nlZ = +inf;
end

for i = 1:options.num_restarts
    hyperparameters = model.prior();
    
    [hyperparameter_values, nlZ, exitflag, minFunc_output] = ...
        minFunc(f, unwrap(hyperparameters), options.minFunc_options);
    
    try
        theta = rewrap(initial_hyperparameters, hyperparameter_values);
        K = feval(model.covariance_function{:}, theta.cov, x);
        L = chol(fix_pd_matrix(K));
    catch
        nlZ = +inf;
        exitflag = -1;
    end
    
    % if minFunc have failed
    if exitflag < 0
        for j = 1:10
            hyperparameters = model.prior();            
            
            warning('MinFunc failed trying to optimize the hyperparameters. N# failures is %d out of 10\n', j);
            [hyperparameter_values, nlZ, exitflag, minFunc_output] = ...
                minFunc(f, unwrap(hyperparameters), options.minFunc_options);
            
            try
                theta = rewrap(initial_hyperparameters, hyperparameter_values);
                K = feval(model.covariance_function{:}, theta.cov, x);
                L = chol(fix_pd_matrix(K));
            catch
                nlZ = +inf;
                exitflag = -1;
            end
            if exitflag > 0
                break;
            end
        end
    end
    
    % saving best values
    if (nlZ < best_nlZ && abs(nlZ-best_nlZ) > 1e-6)        
        best_nlZ = nlZ;
        best_hyperparameter_values = hyperparameter_values;
        best_minFunc_output = minFunc_output;
    end
end

best_hyperparameters = rewrap(initial_hyperparameters, ...
    best_hyperparameter_values);

if isnan(best_nlZ) || isinf(best_nlZ)
    error('Optimization failured')
end

end