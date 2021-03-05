function [chosen_x_star, context] = expected_improvement_limited_optimization_discrete(...
    problem,models, x_train, y_train, context)

model_posterior = context.model_posterior;
if isfield(context, 'used')
    used = context.used;
else
    used = problem.used;
end

name            = problem.name;
n_cand          = size(problem.x_pool,1);
n_models        = numel(models);

y_min           = min(y_train);
ei              = nan(n_cand, n_models);

exclude         = context.exclude;

% computing ei for each model
for i = 1:n_models
    % expected improvement, notice that we assume that the
    % observation noise is very small therefore f is very close to
    ei_obj = @(x_star) ei_objective(x_star, models{i}, ...
        x_train, y_train, y_min, exclude);
    
    ei(:,i) = ei_obj(problem.x_pool);
end
t = toc;

% marginal expected improvement
marginal_ei        = ei*model_posterior';

% select maximum
[ei_val, ei_index] = max(marginal_ei);
chosen_x_star      = ei_index;

fprintf(' %s >>> Optimization %f in s. X chosen was %d\n', name, t, ei_index);

%a = find(~used);
%used(a(ei_index)) = true;

% saving data in context
context.used    = used;
%context.x_pool  = x_pool;
context.max_acq = ei_val;
%context.acq     = ei;
end
