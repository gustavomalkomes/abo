function [chosen_x_star, context] = expected_improvement_optimization5(...
    problem,models, x_train, y_train, context)

model_posterior = context.model_posterior;

[~, order]      = sort(y_train);
x_cand          = x_train(order(1:3),:);

jitter          = rand(size(x_cand));
jitter          = jitter./sqrt(sum(jitter.^2,2)); %normalized
jitter          = jitter.*0.03;

x_cand          = x_cand + jitter;

total_n         = size(problem.x_pool,1);
index_cand      = randperm(total_n, 2);
x_cand          = [x_cand; problem.x_pool(index_cand,:)];

y_min           = y_train(order(1));

d               = size(x_train,2);
exclude         = context.exclude;
options         = optimoptions('fmincon','FiniteDifferenceType','central', ...
    'Display', 'none');

fun = @(x) neg_marginal_ei(x, models, x_train, y_train, y_min, ...
    exclude, model_posterior);

chosen_x_star_best = x_cand(1,:);
ei_val_best        = -inf;

for i = 1:size(x_cand,1)
    x0            = x_cand(i,:);
   
    [chosen_x_star, ei_val] = fmincon(fun,x0,[],[],[],[], ...
        zeros(d,1),ones(d,1), [], options);

    ei_val = -ei_val;

    if ei_val > ei_val_best
        ei_val_best = ei_val;
        chosen_x_star_best = chosen_x_star;
    end
end

ei_val = ei_val_best;
chosen_x_star = chosen_x_star_best;

% saving data in context
context.x_pool  = x_cand;
context.max_acq = ei_val;
context.acq     = ei_val;
end