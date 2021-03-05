function [x_star, y_star, context] = agpl(problem, models, ...
    update_models, query_strategy, callback)

warning('off');

% set initial points
x = problem.initial_x;
y = problem.initial_y;

x_star = NaN(problem.budget, size(x,2));
y_star = NaN(problem.budget, size(y,2));

context = [];
context.exclude = [];

total_time_update = [];
total_time_acquisition = [];
total_time_oracle = [];

is_discrete = isfield(problem, 'isdiscrete') && problem.isdiscrete;
not_optimal = true;

for i = 1:problem.budget
   
    if not_optimal
        
        tstart_update = tic;
        % update the models with current observations
        [models, log_evidence, context] = update_models(problem,models,x,y, context);
        
        [~, order]   = sort(log_evidence, 'descend');
        models       = models(order);
        log_evidence = log_evidence(order);
        
        % computing model_posterior
        % exp and model evidence normalization
        model_posterior = exp(log_evidence-max(log_evidence));
        model_posterior = model_posterior/sum(model_posterior);
        
        for j = 1:numel(model_posterior)
            %            fprintf('%s %f\n', covariance2str(models{j}.parameters.covariance_function), model_posterior(j))
            if (model_posterior(j) < 0.01) || (j > problem.max_num_models)
                final_model = j-1;
                %                fprintf('Pruning models. Using just %d\n', final_model);
                models = models(1:1:final_model);
                model_posterior = model_posterior(1:final_model);
                model_posterior = model_posterior/sum(model_posterior);
                break;
            end
        end
        
        time_update = toc(tstart_update);
        total_time_update = [total_time_update; time_update];
        
        % saving model_posterior in the context
        context.model_posterior = model_posterior;
        
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
        
        
        if is_discrete
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
    context.time_model_search = 0;
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