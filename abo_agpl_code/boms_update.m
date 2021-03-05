function belief = gp_update(problem, belief, x, y)

models       = belief.models;
num_models   = numel(models);
log_evidence = zeros(1,num_models);
options      = problem.optimization;

% training each model m
learn_theta = size(y,1) == size(problem.initial_y,1);

for m = 1:num_models
    % loading model from belief
    model = models{m}.parameters;
    
    % training gp hyperparameters
    if learn_theta || mod(size(y,1),problem.update_interval) == 0
        model = gp_train(x, y, model, options);
        %fprintf(' >>>> Training <<<<< \n');
    else
        model = gp_train_rank_one_update(x, y, model);
    end
    
    % computing log evidence
    log_evidence(m) = gp_log_evidence(model);
    
    % saving model into belief
    models{m}.parameters = model;
end

[~, order]   = sort(log_evidence, 'descend');
models       = models(order);
log_evidence = log_evidence(order);

% computing model_posterior
% exp and model evidence normalization
model_posterior = exp(log_evidence-max(log_evidence));
model_posterior = model_posterior/sum(model_posterior);

last_model = num_models;
for j = 1:num_models
    %     fprintf('%s %f\n', ...
    %         covariance2str(models{j}.parameters.covariance_function), ...
    %         model_posterior(j))
    
    if model_posterior(j) < 0.001
        last_model = j-1;
        fprintf('Pruning models. Using just %d\n', last_model);
        break;
    end
end

models          = models(1:last_model);
model_posterior = model_posterior(1:last_model);
model_posterior = model_posterior/sum(model_posterior);

belief.models          = models;
belief.model_posterior = model_posterior;

end