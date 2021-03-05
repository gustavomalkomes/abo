function [x_star, y_star, context, models] = abo(problem, models, ...
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

for i=1:numel(covariances)
   base_names{i} = covariance2str(covariances{i}.fun); 
end


starting_depth       = 2;
max_candidates       = problem.max_candidates;
[covariances] = fully_expand_tree(covariances, ...
    starting_depth, max_candidates);

% initial models
init_models = boms_models;
for i=1:numel(init_models)
    init_cov{i}.fun = init_models{i}.parameters.covariance_function;
    init_cov{i}.priors = init_models{i}.parameters.priors.cov;
    init_names{i} = covariance2str(init_cov{i}.fun);
end

[new_cov, new_names] = remove_duplicate_candidates(covariances, ...
    init_names, base_names);

covariances = init_cov;
names = init_names;
init_num_models = numel(covariances);
for i = 1:numel(new_names)
    covariances{init_num_models + i} = new_cov{i};
    names{init_num_models + i} = new_names{i};
end

samples = sobol_sample(problem.total_hyp_samples, 300);
prior = create_bo_prior(x, samples, 0.001, 20);
prior = prior.update(prior, covariances, names, []);
for i=1:init_num_models
    prior = prior.update(prior, [], [], i);
end

total_time_update = [];
total_time_acquisition = [];
total_time_oracle = [];
total_time_model_search = [];
not_optimal = true;

for i = 1:problem.budget
    %     try
    if not_optimal

        tstart_model_search = tic;
        [boms_models, boms_log_evidence, prior] = ...
            aboms_wrapper(problem,boms_models,prior,x,y);
        
        boms_log_evidence = boms_log_evidence';
        
        new_models       = boms_models(end+1-problem.eval_budget:end);
        new_log_evidence = boms_log_evidence(end+1-problem.eval_budget:end);
        new_log_evidence = new_log_evidence*numel(y);
        
        time_model_search = toc(tstart_model_search);
        total_time_model_search = [total_time_model_search; time_model_search];
        
        fprintf('Total time model search %f\n', time_model_search);
        
        tstart_update = tic;
        if i > 1
            [models, log_evidence, context] = ...
                update_models(problem,models,x,y, context);
        end
        models       = [models, new_models];
        log_evidence = [log_evidence, new_log_evidence];
        
        [~, order] = sort(log_evidence, 'descend');
        models = models(order);
        log_evidence = log_evidence(order);
        
        % computing model_posterior
        % exp and model evidence normalization
        model_posterior = exp(log_evidence-max(log_evidence));
        model_posterior = model_posterior/sum(model_posterior);
        
        %    fprintf('Current top 5 out of %d in the bag of models\n', numel(models));
        for j = 1:numel(model_posterior)
            if (model_posterior(j) < 0.01) || (j > problem.max_num_models)
                final_model = j-1;
                %            fprintf('Pruning models. Using just %d\n', final_model);
                models = models(1:final_model);
                model_posterior = model_posterior(1:final_model);
                model_posterior = model_posterior/sum(model_posterior);
                break;
            end
        end
        
        time_update = toc(tstart_update);
        total_time_update = [total_time_update; time_update];
        
        fprintf('Total time model update %f\n', time_update);
        
        % saving model_posterior in the context
        context.model_posterior = model_posterior;
        
        %    fprintf('Start acquisition function\n')
        tstart_acquisition = tic;
        % select location(s) of next observation(s) from the given list
        [chosen_x_star, context] = query_strategy(problem,models,x,y,context);
        
        time_acquisition = toc(tstart_acquisition);
        total_time_acquisition = [total_time_acquisition; time_acquisition];
        
        fprintf('Total time acquistion function %f\n', time_acquisition);
        
        tstart_oracle = tic;
        % observe label(s) at chosen location(s)
        this_chosen_label = problem.label_oracle(problem,x,y,chosen_x_star);
        
        time_oracle = toc(tstart_oracle);
        total_time_oracle = [total_time_oracle; time_oracle];
        
        
        % update lists with new observation(s)
        
        if problem.isdiscrete
            chosen_x_star =  problem.x_pool(chosen_x_star,:);
        end
    end
    
    y_first = min(problem.initial_y);
    y_best  = min(y);
    gap     = NaN;
    if isfield(problem, 'optimum') && ~isnan(problem.optimum)
        gap = (y_first - y_best)/(y_first - problem.optimum);
    end
    
    if gap > 0.995 && abs(y_best - problem.optimum) < 0.0001
        not_optimal = false;
    end
    
    % update lists with new observation(s)
    x_star(i,:) = chosen_x_star;
    x = [x; chosen_x_star];
    
    y_star(i,:) = this_chosen_label;
    y = [y; this_chosen_label];
    
    
    context.time_acquisition = time_acquisition;
    context.time_model_search = time_model_search;
    context.time_update = time_update;
    context.time_oracle = time_oracle;
    
    
    % call callback, if defined
    if (nargin > 4) && ~isempty(callback)
        callback(problem, models, x, y, i, context);
    end
end

time.total_time_update = total_time_update;
time.total_time_acquisition = total_time_acquisition;
time.total_time_oracle = total_time_oracle;

context.time = time;
