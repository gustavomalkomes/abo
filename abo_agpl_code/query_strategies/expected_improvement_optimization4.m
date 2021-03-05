function [chosen_x_star, context] = expected_improvement_optimization4(...
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

% select 3 initial points out of the top 20 to do a fine grid-search
top_range = 20;
total_restarts = 2;

[ei_vals, ei_index] = sort(marginal_ei, 'descend');
permutation_index = randperm(top_range,total_restarts) + 1;
chosen_x_star_list = x_pool(ei_index(permutation_index),:);

permutation_index  = [1, permutation_index];
chosen_x_star_list = [x_pool(ei_index(1),:); chosen_x_star_list];

d = size(x_train,2);

options = optimoptions('fmincon','FiniteDifferenceType','central', ...
    'Display', 'none');

fun = @(x) neg_marginal_ei(x, models, x_train, y_train, y_min, ...
    exclude, model_posterior);


chosen_x_star_best = chosen_x_star_list(1,:);
ei_val_best = ei_vals(1);
index = 1;
for i = 1:size(chosen_x_star_list,1)
    chosen_x_star = chosen_x_star_list(i,:);
    x0            = chosen_x_star;
    
    [chosen_x_star, ei_val] = fmincon(fun,x0,[],[],[],[], ...
        zeros(d,1),ones(d,1), [], options);

    ei_val = -ei_val;

    if ei_val > ei_val_best
        ei_val_best = ei_val;
        chosen_x_star_best = chosen_x_star;
        index = i;
    end
end

ei_val = ei_val_best;
chosen_x_star = chosen_x_star_best;

a = find(~used);
used(a(ei_index(permutation_index(index)))) = true;

% saving data in context
context.used    = used;
context.x_pool  = x_pool;
context.max_acq = ei_val;
context.acq     = ei;
end