function [chosen_x_star, context] = ens_bo(...
    problem,models, x_train, y_train, context)

%model_posterior = context.model_posterior;

if isfield(context, 'used')
    used = context.used;
else
    used = problem.used;
end

rem_budget      = problem.budget - size(y_train,1) - 1 + size(problem.initial_y,1);
x_cand          = problem.x_pool;
x_pool          = x_cand(~used,:);

n_cand          = size(x_pool,1);
n_models        = numel(models);

y_min           = min(y_train);

i               = 1;
model           = models{i};
gp_model        = model.parameters;

% gp predictions
[~, ~, mu,cov] = model.prediction(gp_model.theta, ...
    gp_model.inference_method, ...
    gp_model.mean_function, gp_model.covariance_function, ...
    gp_model.likelihood, x_train, gp_model.posterior, x_pool);

num_samples = 5;

n_total_blocks = 20;
n_block_size = floor(n_cand/n_total_blocks);
last_block_size = mod(n_cand, n_total_blocks);

blocks = reshape(1:(n_cand - last_block_size), n_total_blocks, n_block_size);
last_block = (n_cand-last_block_size+1:n_cand);

blocks_ens = zeros(n_total_blocks, n_block_size);
parfor j = 1:n_total_blocks,     
    blocks_ens(j, :) = batch_ens(blocks(j,:), mu, cov, num_samples, ...
        model, gp_model, x_train, x_pool, rem_budget);
end

% last block
last_ens = zeros(last_block_size,1);
parfor j = 1:last_block_size
    last_ens(j) = batch_ens(last_block(j), mu, cov, num_samples, ...
        model, gp_model, x_train, x_pool, rem_budget);
end

unwrapped_ens = reshape(blocks_ens, n_cand-last_block_size,1);
ens = [unwrapped_ens; last_ens];


% no par for implementation
% ens = zeros(n_cand,1);
% for j = 1:n_cand
%     ens(j) = compute_ens(x_pool(j,:),mu(j),cov(j), num_samples, ...
%     model, gp_model, x_train, x_pool, rem_budget);
% end

% select maximum
[ens_val, ens_index] = min(ens);
chosen_x_star        = x_pool(ens_index,:);

a = find(~used);
used(a(ens_index))   = true;

context.used         = used;

if problem.verbose
    fprintf(' > f_min: %.8f, a(x): %.8f\n', ...
        y_min, ens_val);
    %disp(chosen_x_star);
end

end

%% batch wrapper to compute ens
function ens = batch_ens(indexes, mu, cov, num_samples, ...
    model, gp_model, x_train, x_star, rem_budget)
    ens = zeros(1, numel(indexes));
    ii = 0;
    for i = indexes
        ii = ii + 1;
        ens(ii) = compute_ens(x_star(i,:),mu(i),cov(i), num_samples, ...
        model, gp_model, x_train, x_star, rem_budget);
    end
end

%% compute ens
function ens = compute_ens(x_fake,y_fake_mu,y_fake_cov, num_samples, ...
    model, gp_model, x_train, x_star, rem_budget)

future_steps = 0;
[y_fakes, w] = bq_sigma_points(y_fake_mu, y_fake_cov, num_samples);
w            = w./(sum(w));

for i = 1:num_samples
    y_fake = y_fakes(i);
    
    new_posterior = update_posterior(gp_model.theta, gp_model.mean_function, ...
        gp_model.covariance_function, x_train, gp_model.posterior, x_fake, y_fake);
    
    % conditioning in the fake_observation
    [~, ~, mu_fake,~] = model.prediction(gp_model.theta, ...
        gp_model.inference_method, ...
        gp_model.mean_function, gp_model.covariance_function, ...
        gp_model.likelihood, [x_train; x_fake], new_posterior, x_star);
    
    bot_prob = sort(mu_fake, 'ascend');
    
    future_steps = future_steps  + w(i)*sum(bot_prob(1:rem_budget));
end

ens = y_fake_mu + future_steps;

end