function [chosen_x_star, context] = ...
    expected_improvement_limited_optimization(...
    problem, models, x_train, y_train, context)

model_posterior = context.model_posterior;
if isfield(context, 'used')
    used = context.used;
else
    used = problem.used;
end

name            = problem.name;
x_cand          = problem.x_pool;
x_pool          = x_cand(~used,:);

n_cand          = size(x_pool,1);
n_models        = numel(models);

y_min           = min(y_train);
ei              = nan(n_cand, n_models);

exclude         = context.exclude;

fprintf(' %s >>> First Started there are %d points\n', name, n_cand);

tic
% computing ei for each model
for i = 1:n_models
    % expected improvement, notice that we assume that the
    % observation noise is very small therefore f is very close to
    ei_obj = @(x_star) ei_objective(x_star, models{i}, ...
        x_train, y_train, y_min, exclude);
    
    ei(:,i) = ei_obj(x_pool);
end

t = toc;

fprintf(' %s >>> First optimization %f in s\n', name, t);

% marginal expected improvement
marginal_ei        = ei*model_posterior';

% select maximum
[~, ei_index]      = max(marginal_ei);
chosen_x_star      = x_pool(ei_index,:);

a = find(~used);
used(a(ei_index)) = true;

d = size(x_train,2);

options = optimoptions('fmincon', 'Display', 'none', ...
    'MaxFunctionEvaluations', 1000, 'Algorithm','sqp');

options = optimoptions('fmincon', 'Display', 'none', ...
    'MaxFunctionEvaluations', 1000);

fun = @(x) nmei_objective(x, models, x_train, y_train, y_min, ...
    exclude, model_posterior);

x0  = chosen_x_star;

A = [];
b = [];
Aeq = [];
beq = [];

t = tic;
[chosen_x_star, ei_val, exitflag, output] = fmincon(fun,x0,A,b,Aeq,beq, ...
    zeros(d,1),ones(d,1), [], options);
time = toc(t);
ei_val = -ei_val;

fprintf(' %s >>> rounds %d funcCount %d -- %f s\n', name, output.iterations, output.funcCount, time);

% saving data in context
context.used    = used;
context.x_pool  = x_pool;
context.max_acq = ei_val;
context.acq     = ei;
end
