function [chosen_x_star, context] = egreedy(...
    problem,models, x_train, y_train, context)

model_posterior = context.model_posterior;
if isfield(context, 'used')
    used = context.used;
else
    used = problem.used;
end


is_explore = isfield(context, 'explore') && context.explore;


x_cand          = problem.x_pool;
x_pool          = x_cand(~used,:);
d               = size(x_cand,2);

if is_explore
    context.explore = false;
else
    
    ObjectiveFunction = @(x) problem.label_oracle(problem,x_train,y_train,x);
    
    [~, index]               = min(y_train);
    X0                       = x_train(index,:);
    
    problem.label_oracle(problem,x_train,y_train,X0);
    problem.lb = zeros(d,1);
    
    options = saoptimset('simulannealbnd');
    options.MaxFunEvals = problem.budget - size(problem.initial_y,1);
    
    [x,fval,exitFlag,output] = simulannealbnd(ObjectiveFunction,X0, zeros(1,d), ones(1,d), options);

    context.explore = false;

end
    


n_cand          = size(x_pool,1);
rand_index      = randi(n_cand);

y_min           = min(y_train);



chosen_x_star      = x;

a = find(~used);
used(a(rand_index)) = true;

% saving data in context
context.used    = used;
context.x_pool  = x_pool;
context.acq     = 0;

if is_explore
    context.max_acq = 1;
else
    context.max_acq = 0;
end

end