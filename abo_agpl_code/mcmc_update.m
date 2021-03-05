function belief = mcmc_update(problem, belief, x, y, ...
    base_covs, base_names, theta, pred)

% initial setup
if ~isfield(belief, 'candidates')
    
    cov_root     = problem.covariance_root;
    covariances  = covariance_grammar_started(cov_root,theta);
    pred         = problem.prediction_function;

    base_covs    = covariances;
    last_cov     = covariances{1};
    
    belief.candidates.last_cov   = last_cov;
    belief.candidates.last_model = gpr_model_builder(last_cov,theta,pred);
    
end

% update your current belief
belief         = gp_update(problem, belief, x, y);
models         = belief.models;
log_evidence   = belief.log_evidence;

% Start MCMC
last_cov       = belief.candidates.last_cov;
last_model     = belief.candidates.last_model;
chain          = {};
chain{1}.cov   = last_cov;
chain{1}.model = last_model;

% Update last model from the chain
[last_model, last_cov_log_evidence] = ...
    gp_update(problem, {last_model}, x, y);

chain{1}.model    = last_model{:};
chain{1}.evidence = last_cov_log_evidence;

% Get new samples 
for j = 1:problem.eval_budget
    chain = mcmc_samples(chain, base_covs, ...
        base_names, problem, x, y, theta, pred);
end

last_cov   = chain{end}.cov;
last_model = chain{end}.model;

belief.candidates.last_cov   = last_cov;
belief.candidates.last_model = last_model;

% Add new models to our belief
for j = 1:numel(chain)
    models(end+1) = {chain{j}.model};
    log_evidence(end+1) = chain{j}.evidence;
end

[~, order]   = sort(log_evidence, 'descend');
models       = models(order);
log_evidence = log_evidence(order);

% computing model_posterior
% exp and model evidence normalization
model_posterior = exp(log_evidence-max(log_evidence));
model_posterior = model_posterior/sum(model_posterior);

for j = 1:numel(model_posterior)
    fprintf('%s %f\n', covariance2str(models{j}.parameters.covariance_function), model_posterior(j))
    if (model_posterior(j) < 0.001) || j > problem.max_num_models
        final_model = j-1;
        fprintf('Pruning models. Using just %d\n', final_model);
        models = models(1:final_model);
        model_posterior = model_posterior(1:final_model);
        model_posterior = model_posterior/sum(model_posterior);
        break;
    end
end


end

function chain = mcmc_samples(chain, base_covs, base_names, ...
    problem, x, y, theta, pred)

last_cov              = chain{end}.cov;
last_cov_log_evidence = chain{end}.evidence;

last_cov_neighbors = expand_covariance(last_cov, base_covs, base_names, 10);
m_star             = last_cov_neighbors(randi(numel(last_cov_neighbors)));
m_star_neighbors   = expand_covariance(m_star{1}, base_covs, base_names, 10);

m_star_model       = gpr_model_builder(m_star{1},theta,pred);
[m_star_model, m_star_log_evidence] = gp_update(problem, {m_star_model}, x, y, []);

% Metropolis-Hastings acceptance probability
% log(evidence_m_star) + log(1/neighbors(m_star))
% - log(evidence_last) - log(1/neighbors(last))
% which is the same that:
% log(evidence_m_star) - log(neighbors(m_star))
% - log(evidence_last) + log(neighbors(last))

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