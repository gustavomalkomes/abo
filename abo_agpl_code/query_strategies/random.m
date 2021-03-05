function [chosen_x_star,context] = random(...
    problem,models, x_train, y_train, context)

model_posterior = context.model_posterior;
x_pool   = problem.points(:,:);
n_cand   = size(x_pool,1);
n_models = numel(models);

n_train  = size(x_train,1);
remove_ind = [];
for i = 1:n_cand,
    for j = 1:n_train,
        if norm(x_pool(i,:) - x_train(j,:)) < 1e-6,
            remove_ind = [remove_ind, i];
        end
    end
end

x_pool = x_pool(setdiff(1:n_cand, remove_ind), :);

chosen_x_star = datasample(x_pool,1);

end