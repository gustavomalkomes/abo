function [chosen_x_star, context] = ...
    multi_model_ei(problem, models, x_train, y_train, context)

model_posterior = context.model_posterior;
if isfield(context, 'used')
    used = context.used;
else
    used = problem.used;
end

name            = problem.name;
y_min           = min(y_train);

exclude         = context.exclude;

n_models        = numel(models);
[n_points, n_features] = size(problem.x_pool(~used,:));
fprintf(' %s >>> Optimization there are %d points\n', name, n_points);

chosen_x_star = NaN(n_models,n_features);
% computing ei for each model
for i = 1:n_models
    % expected improvement, notice that we assume that the
    % observation noise is very small therefore f is very close to
    ei_obj = @(x_star) ei_objective(x_star, models{i}, ...
        x_train, y_train, y_min, exclude);
    
%     tic;
    ei_start = ei_obj(problem.x_pool(~used,:));
%     t = toc;
%     fprintf(' %s >>> First optimization %f in s\n', name, t);
    
    [~, ei_index] = max(ei_start);
    x0            = problem.x_pool(ei_index,:);
    
    a = find(~used);
    used(a(ei_index)) = true;
    
    d = size(x_train,2);
    
    options = optimoptions('fmincon', 'Display', 'none', ...
        'MaxFunctionEvaluations', 100);
    
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    
    t = tic;
    [chosen_x_star(i,:), ei_val, exitflag, output] = ...
        fmincon(ei_obj,x0,A,b,Aeq,beq,zeros(d,1),ones(d,1),[],options);
    time = toc(t);
    ei_val = -ei_val;
    
%     fprintf(' %s >>> rounds %d funcCount %d -- %f s\n', ...
%         name, output.iterations, output.funcCount, time);
    
    
end


mei = mei_objective(chosen_x_star, models, x_train, y_train, ...
    y_min, exclude, model_posterior);

% select maximum
[ei_val, ei_index] = max(mei);
chosen_x_star      = chosen_x_star(ei_index,:);

% saving data in context
context.used    = used;
context.max_acq = ei_val;
end
