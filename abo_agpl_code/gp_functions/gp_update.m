function [models, log_evidence, context] = ...
    gp_update(problem,models,x,y, context)

warning('off');
num_models   = numel(models);
log_evidence = zeros(1,num_models);
options      = problem.optimization;

% training each model m
for m = 1:num_models
    start_clock = tic();
    % training gp hyperparameters
    try
        models{m}.parameters = gp_train(x,y,models{m}.parameters,options);
    catch
        % Removing points that are too close
        fprintf('Error during training model %d. ', m);
        fprintf('Removing points that are too close to each other.\n');
        n = size(x,1);
        exclude = false(n,1);
        for i = 1:n
            for j = i+1:n
                if(norm(x(i,:)-x(j,:)) < 0.05)
                    exclude(i) = true;
                end
            end
        end
        x = x(~exclude,:);
        y = y(~exclude,:);
        context.exclude = exclude;
        models{m}.parameters = gp_train(x,y,models{m}.parameters,options);
    end
    
    % computing log evidence
    log_evidence(m) = gp_log_evidence(models{m}.parameters);
    
    fprintf('Model %d trained in %.3f s\n', m, toc(start_clock));
end

end