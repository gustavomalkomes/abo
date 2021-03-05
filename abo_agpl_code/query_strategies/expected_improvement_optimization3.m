function [chosen_x_star, context] = expected_improvement_optimization3(...
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

options = optimoptions('fmincon', 'Display', 'none');

fun = @(x) nmei_objective(x, models, x_train, y_train, y_min, ...
    exclude, model_posterior);

x0  = chosen_x_star;

t = tic;

[chosen_x_star, ei_val, ~, output] = fmincon(fun,x0,[],[],[],[], ...
    zeros(d,1),ones(d,1), [], options);

acq_time = toc(t);
disp(chosen_x_star)
% [chosen_x_star, ei_val] = fmincon(fun,x0,[],[],[],[], ...
%     zeros(d,1),ones(d,1), [], options);

ei_val = -ei_val;

fprintf(' >>>> iter %d funcCount %d in %f s\n', output.iterations, output.funcCount, acq_time);

% saving data in context
context.used    = used;
context.x_pool  = x_pool;
context.max_acq = ei_val;
context.acq     = ei;
end