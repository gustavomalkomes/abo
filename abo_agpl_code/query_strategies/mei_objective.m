function [mei, mei_grad] = mei_objective(x, models, x_train, y_train, ...
    y_min, exclude, model_posterior)

n_models = numel(models);

if size(x,2) ~= size(x_train,2)
    x = x';
end

ei       = nan(size(x,1), n_models);
mei_grad = zeros(size(x,1), size(x,2));

% computing ei for each model
for i = 1:n_models
    if nargout < 2
        ei(:,i) = ei_objective(x, models{i}, ...
            x_train, y_train, y_min, exclude);
    else
        func = @(xx) ei_objective(xx, models{i}, ...
            x_train, y_train, y_min, exclude);
        [ei_grad_i,ei(:,i)] = AutoDiffJacobianFiniteDiff(func,x);
        mei_grad = mei_grad + ei_grad_i*model_posterior(i);
    end
end

mei = ei*model_posterior';

end
