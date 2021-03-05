function [models, context] = gp_update_no_train(problem,models,x,y, context)

num_models   = numel(models);
log_evidence = zeros(1,num_models);
options      = problem.optimization;

% training each model m
for m = 1:num_models,
    % training gp hyperparameters
    models{m}.parameters = gp_no_train(x,y,models{m}.parameters,options);
    
    % computing log evidence
    log_evidence(m) = gp_log_evidence(models{m}.parameters);    
end

% computing model_posterior
% exp and model evidence normalization
model_posterior = exp(log_evidence-max(log_evidence));
model_posterior = model_posterior/sum(model_posterior);

context.model_posterior = model_posterior;

end

