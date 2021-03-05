function [x_star, y_star, context, models] = mcmc_add(problem, models, ...
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

for i = 1:problem.budget
    
    tstart_update = tic;
    % update the models with current observations
    [models, log_evidence, context] = update_models(problem,models,x,y, context);
    
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
    
    [~, order]   = sort(log_evidence, 'descend');
    models       = models(order);
    log_evidence = log_evidence(order);
    
    % computing model_posterior
    % exp and model evidence normalization
    model_posterior = exp(log_evidence-max(log_evidence));
    model_posterior = model_posterior/sum(model_posterior);
    
    for j = 1:numel(model_posterior)
%        fprintf('%s %f\n', covariance2str(models{j}.parameters.covariance_function), model_posterior(j))
        if (model_posterior(j) < 0.001) || j > problem.max_num_models
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
    
    % saving molde_posterior in the context
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
    
    % update lists with new observation(s)
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


end

function chain = update_mcmc(chain, base_covs, base_names, problem, x, y, theta, pred)

last_cov              = chain{end}.cov;
last_cov_log_evidence = chain{end}.evidence;

last_cov_neighbors = expand_covariance(last_cov, base_covs, base_names, 10);
m_star             = last_cov_neighbors(randi(numel(last_cov_neighbors)));
m_star_neighbors   = expand_covariance(m_star{1}, base_covs, base_names, 10);

m_star_model       = gpr_model_builder(m_star{1},theta,pred);
[m_star_model, m_star_log_evidence] = gp_update(problem, {m_star_model}, x, y, []);

r = m_star_log_evidence - last_cov_log_evidence ...
    - log(numel(m_star_neighbors)) + log(numel(last_cov_neighbors));

r = min(1,exp(r));

if unifrnd(0,1,1)  < r
    new_entry.cov = m_star{1};
    new_entry.model = m_star_model{:};
    new_entry.evidence = m_star_log_evidence;
    chain = [chain, new_entry];
    fprintf('Accepting %s log_evidence: %f\n', covariance2str(m_star{1}.fun), m_star_log_evidence)
else
    fprintf('Rejecting %s log_evidence: %f\n', covariance2str(m_star{1}.fun), m_star_log_evidence)
end

end