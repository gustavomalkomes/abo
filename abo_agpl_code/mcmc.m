function [x_star, y_star, context, models] = mcmc(problem, models, ...
    update_models, query_strategy, callback)

warning('off');

% set initial points
x = problem.initial_x;
y = problem.initial_y;

x_star = NaN(problem.budget, size(x,2));
y_star = NaN(problem.budget, size(y,2));

context = [];
context.exclude = [];

% initial setup
cov_root     = problem.covariance_root;
theta        = gpr_hyperparameters(problem.data_noise);
covariances  = covariance_grammar_started(cov_root,theta);
pred         = problem.prediction_function;
num_models   = numel(covariances);
base_names   = cell(1,num_models);
for i = 1:num_models
    base_names{i} = covariance2str(covariances{i}.fun);
end
base_covs    = covariances;

last_cov              = covariances{1};
last_model            = gpr_model_builder(last_cov,theta,pred);

total_time_update = [];
total_time_acquisition = [];
total_time_oracle = [];
total_time_model_search = [];
not_optimal = true;

for i = 1:problem.budget
    
    if not_optimal
        
        tstart_update = tic;
        % update the models with current observations
        [models, log_evidence, context] = update_models(problem,models,x,y, context);
        
        time_update = toc(tstart_update);
        total_time_update = [total_time_update; time_update];
        fprintf('Total time model update %f\n', time_update);
        
        tstart_model_search = tic;
        %    if mod(i,5) == 1
        chain          = {};
        chain{1}.cov   = last_cov;
        chain{1}.model = last_model;
        
        [last_model, last_cov_log_evidence] = gp_update(problem, {last_model}, x, y, []);
        chain{1}.model = last_model{:};
        chain{1}.evidence = last_cov_log_evidence;
        
        for j = 1:problem.eval_budget
            chain = update_mcmc(chain, base_covs, base_names, problem, x, y, theta, pred);
        end
        
        last_cov = chain{end}.cov;
        last_model = chain{end}.model;
        
        for j = 1:numel(chain)
            models(end+1) = {chain{j}.model};
            log_evidence(end+1) = chain{j}.evidence;
        end
        %    end
        
        
        time_model_search = toc(tstart_model_search);
        total_time_model_search = [total_time_model_search; time_model_search];
        
        fprintf('Total time model search %f\n', time_model_search);
        
        
        [~, order]   = sort(log_evidence, 'descend');
        models       = models(order);
        log_evidence = log_evidence(order);
        
        % computing model_posterior
        % exp and model evidence normalization
        model_posterior = exp(log_evidence-max(log_evidence));
        model_posterior = model_posterior/sum(model_posterior);
        
        for j = 1:numel(model_posterior)
            %        fprintf('%s %f\n', covariance2str(models{j}.parameters.covariance_function), model_posterior(j))
            if (model_posterior(j) < 0.01) || (j > problem.max_num_models)
                final_model = j-1;
                %            fprintf('Pruning models. Using just %d\n', final_model);
                models = models(1:final_model);
                model_posterior = model_posterior(1:final_model);
                model_posterior = model_posterior/sum(model_posterior);
                break;
            end
        end
        
        
        % saving molde_posterior in the context
        context.model_posterior = model_posterior;
        
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


end

function chain = update_mcmc(chain, base_covs, base_names, problem, x, y, theta, pred)

model              = chain{end}.cov;
model_log_evidence = chain{end}.evidence;

model_neighbors    = expand_covariance(model, base_covs, base_names, 10);

proposal_cov       = model_neighbors(randi(numel(model_neighbors)));
proposal_neighbors = expand_covariance(proposal_cov{1}, base_covs, base_names, 10);

proposal           = gpr_model_builder(proposal_cov{1},theta,pred);
[proposal, proposal_log_evidence] = gp_update(problem, {proposal}, x, y, []);

log_r = proposal_log_evidence - model_log_evidence ...
    - log(numel(proposal_neighbors)) + log(numel(model_neighbors));

r = min(1,exp(log_r));

%fileID = fopen('samples.txt','a');

%fprintf(fileID,'m_star %f, last_cov %f, m_star_neigh %f and last_cov_neig %f\n', ...
%    proposal_log_evidence, model_log_evidence, log(numel(proposal_neighbors)), ...
%    log(numel(model_neighbors)));

rand_number = unifrnd(0,1,1);

%fprintf('R is %f. Rand number is %f \n', r, rand_number);


if rand_number  < r
    new_entry.cov = proposal_cov{1};
    new_entry.model = proposal{:};
    new_entry.evidence = proposal_log_evidence;
    chain = [chain, new_entry];
    %fprintf('Accepting %s log_evidence: %f\n', covariance2str(proposal_cov{1}.fun), proposal_log_evidence)
    %fprintf(fileID, 'A %f\n', proposal_log_evidence);
else
    %fprintf('Rejecting %s log_evidence: %f\n', covariance2str(proposal_cov{1}.fun), proposal_log_evidence)
    %fprintf(fileID, 'R %f\n', proposal_log_evidence);
end

%fclose(fileID);
end