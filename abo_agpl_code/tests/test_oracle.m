clear all;

root             = {'SE', 'M1', 'RQ'};
true_model_index = 3;
level            = 0;
data_noise       = 0.00001;
n_pool           = 200;
x_pool           = linspace(-5,5,n_pool)';
n_samples        = 5;
type             = 'mgp';

% creating covariances
theta                = gpr_hyperparameters(data_noise);
covariances_root     = covariance_grammar_started(root,theta);
[covariances,names]  = covariance_builder_grammar_random(covariances_root,level);

% creating models based on the covariances
n_models = numel(covariances);
models   = cell(1,n_models);
for i = 1:n_models,
    models{i} = gpr_model_builder(covariances{i}, theta, type);
end

true_model = models{true_model_index};
theta = independent_prior(true_model.parameters.priors);

theta.cov(1) = log(0.5);
theta.cov(2) = log(0.3);
theta.mean   = 10;

true_model.parameters.theta = theta;

x_star = x_pool;
mu     = true_model.parameters.mean_function;
cov    = true_model.parameters.covariance_function;
hyp    = true_model.parameters.theta;

m = feval(mu{:}, hyp.mean, x_star);
K  = feval(cov{:}, hyp.cov,  x_star);

K  = fix_pd_matrix(K);
n  = size(x_star,1);
y = chol(K)' * randn(n, n_samples) + exp(hyp.lik) * randn(n, n_samples);
y  = bsxfun(@plus, m, y);

figure(1)
plot(x_star,y)

label_oracle = @(x, y, x_star) ...
    gaussian_process_oracle([], x, y, x_star, true_model.parameters);

y_plot = [];

for j = 1:n_samples
    
    x  = [];
    y  = [];
    clear functions    
    
    x_star = linspace(-5,5, 20)';
    y_star = [];
    
    x = x_star(1,:);
    y = label_oracle(x, y, x);
    
    for i = 2:numel(x_star)
        y_star = label_oracle(x, y, x_star(i,:));
        x = [x; x_star(i,:)];
        y = [y; y_star];
    end
    
    x_star = x_pool;
    y_star = label_oracle(x, y, x_star);
    y_plot = [y_plot, y_star];
end

figure(2)
clf;
plot(x_pool,y_plot)
