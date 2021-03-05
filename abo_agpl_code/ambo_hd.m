function [x_star, y_star, context, models] = ambo_hd(problem, models, ...
    update_models, query_strategy, callback)

warning('off');

% set initial points
x = problem.initial_x;
y = problem.initial_y;

x_star = NaN(problem.budget, size(x,2));
y_star = NaN(problem.budget, size(y,2));

context = [];
context.exclude = [];

% Updating initial bag of models
%fprintf('Updating initial bag of %d models \n', numel(models))
[models, log_evidence, context] = ...
    update_models(problem,models,x,y,context);

for j = 1:numel(models)
    models{j}.parameters.log_evidence = log_evidence(j);
    models{j}.parameters.number_points = numel(y);
    model_name = covariance2str(models{j}.parameters.covariance_function);
%    fprintf('%s log_evidence/n %f  n = %d\n', model_name, log_evidence(j)/numel(y), numel(y));
end

boms_models = models;

% candidate points
theta               = gpr_hyperparameters(problem.data_noise);
covariances_root    = problem.covariance_root;
covariances         = covariance_grammar_started(covariances_root,theta);
d                   = problem.d;

for i=1:numel(covariances)
    covariances{i}.fixed_hyps = inf*ones(numel(covariances{i}.priors),1);
end

if d > 1
    covariances = mask_kernels(covariances, d);
end

starting_depth       = 2;
max_candidates       = problem.max_candidates;
[covariances, names] = fully_expand_tree(covariances, starting_depth, max_candidates);

total_time_update = [];
total_time_acquisition = [];
total_time_oracle = [];

for i = 1:problem.budget
    %     try
    tstart_update = tic;
    
%    if mod(i,5) == 1
%        fprintf('Running BOMS\n')
        [boms_models, boms_log_evidence, covariances] = ...
            boms_wrapper_hd(problem,boms_models,covariances,x,y);
        
        if size(boms_log_evidence,1) > size(boms_log_evidence,2)
            boms_log_evidence = boms_log_evidence';
        end
        
        new_models = boms_models(end+1-problem.eval_budget:end);
        new_log_evidence = boms_log_evidence(end+1-problem.eval_budget:end);
        new_log_evidence = new_log_evidence*numel(y);
%    end
    
    if i > 1
        [models, log_evidence, context] = update_models(problem,models,x,y, context);
    end
    
%    if mod(i,5) == 1
        models       = [models, new_models];
        log_evidence = [log_evidence, new_log_evidence];
%    end
    
    
    [~, order] = sort(log_evidence, 'descend');
    models = models(order);
    log_evidence = log_evidence(order);
    
    % computing model_posterior
    % exp and model evidence normalization
    model_posterior = exp(log_evidence-max(log_evidence));
    model_posterior = model_posterior/sum(model_posterior);
    
%    fprintf('Current top 5 out of %d in the bag of models\n', numel(models));
    for j = 1:numel(model_posterior)
        if j <= 5
%            fprintf('%50s \t %2.3f\n', covariance2str(models{j}.parameters.covariance_function), model_posterior(j))
        end
        if model_posterior(j) < 0.001 || j > problem.max_num_models
            final_model = j-1;
%            fprintf('Pruning models. Using just %d\n', final_model);
            models = models(1:final_model);
            model_posterior = model_posterior(1:final_model);
            model_posterior = model_posterior/sum(model_posterior);
            break;
        end
    end
    
    boms_models = models;
    
    time_update = toc(tstart_update);
    total_time_update = [total_time_update; time_update];
    
    % saving model_posterior in the context
    context.model_posterior = model_posterior;
    
%    fprintf('Start acquisition function\n')
    tstart_acquisition = tic;
    % select location(s) of next observation(s) from the given list
    [chosen_x_star, context] = query_strategy(problem,models,x,y,context);
    
    time_acquisition = toc(tstart_acquisition);
    total_time_acquisition = [total_time_acquisition; time_acquisition];
    
    tstart_oracle = tic;
    % observe label(s) at chosen location(s)
    this_chosen_label = problem.label_oracle(problem,x,y,chosen_x_star);
    
    time_oracle = toc(tstart_oracle);
    total_time_oracle = [total_time_oracle; time_oracle];
    
    
    %     catch ME
    %         if strcmp(ME.identifier, 'MATLAB:posdef')
    %             fprintf('MATLAB:posdef error. Finishing earlier.\n')
    %             return;
    %         else
    %             rethrow(ME)
    %         end
    %    end
    
    % update lists with new observation(s)
    
    if problem.isdiscrete
       chosen_x_star =  problem.x_pool(chosen_x_star,:);
    end

    x_star(i,:) = chosen_x_star;
    x = [x; chosen_x_star];
    
    y_star(i,:) = this_chosen_label;
    y = [y; this_chosen_label];
    
    % call callback, if defined
    if (nargin > 4) && ~isempty(callback)
        callback(problem, models, x, y, i, context);
    end
end


time.total_time_update = total_time_update;
time.total_time_acquisition = total_time_acquisition;
time.total_time_oracle = total_time_oracle;

context.time = time;
