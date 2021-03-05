function [chosen_x_star, context] = expected_improvement_optimization(...
    problem,models, x_train, y_train, context)

model_posterior = context.model_posterior;
if isfield(context, 'used')
    used = context.used;
else
    used = problem.used;
end

x_cand          = problem.x_pool;
x_pool          = x_cand(~used,:);

n_cand          = size(x_pool,1);
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
    
    ei(:,i) = ei_obj(x_pool);
end

% marginal expected improvement
marginal_ei        = ei*model_posterior';

% select maximum
[ei_val_initial, ei_index] = max(marginal_ei);
chosen_x_star      = x_pool(ei_index,:);

a = find(~used);
used(a(ei_index)) = true;

d = size(x_train,2);
cmaes_opts = cmaes('defaults');
cmaes_opts.DispFinal = 0;
cmaes_opts.DispModulo = 100;
cmaes_opts.SaveVariables = 0;
cmaes_opts.LogModulo = 0;

cmaes_opts.LBounds = zeros(d,1);
cmaes_opts.UBounds = ones(d,1);
cmaes_opts.MaxIter = 5000;
cmaes_opts.EvalParallel = 'yes';
cmaes_opts.TolFun = 1e-6;

%t = tic;
[chosen_x_star, ei_val] = ...
    cmaes('nmei_objective', chosen_x_star', 1/5, cmaes_opts, ...
    models, x_train, y_train, y_min, exclude, model_posterior);
%toc(t)

chosen_x_star = chosen_x_star';
ei_val = -ei_val;

if ei_val > ei_val_initial
   fprintf('Improved\n'); 
end

% saving data in context
context.used    = used;
context.x_pool  = x_pool;
context.max_acq = ei_val;
context.acq     = ei;
end