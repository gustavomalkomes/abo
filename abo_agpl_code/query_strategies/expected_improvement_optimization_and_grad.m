function [chosen_x_star, context] = ...
    expected_improvement_optimization_and_grad(...
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

% marginal expected improvement
marginal_ei        = mei_objective(x_pool, ...
    models, x_train, y_train, y_min, exclude, model_posterior);

% select maximum
[~, ei_index]      = max(marginal_ei);
chosen_x_star      = x_pool(ei_index,:);

a = find(~used);
used(a(ei_index)) = true;

d = size(x_train,2);

options = optimoptions('fmincon', 'Display', 'iter', ...
    'SpecifyObjectiveGradient',true);

fun = @(x) nmei_objective(x, models, x_train, y_train, y_min, ...
    exclude, model_posterior);

x0  = chosen_x_star;

% checkgrad('nmei_objective', rand(1,2), 1e-6, models, x_train, y_train, y_min, ...
%    exclude, model_posterior)

% [chosen_x_star, ei_val, ~, output] = fmincon(fun,x0,[],[],[],[], ...
%     zeros(d,1),ones(d,1), [], options);

tic
[chosen_x_star, ei_val] = fmincon(fun,x0,[],[],[],[], ...
    zeros(d,1),ones(d,1), [], options);
toc

ei_val = -ei_val;

disp(chosen_x_star)
%fprintf(' >>>> iter %d funcCount %d\n', output.iterations, output.funcCount);

% saving data in context
context.used    = used;
context.x_pool  = x_pool;
context.max_acq = ei_val;
context.acq     = ei;

end