

clear all;

root         =  {'SE','RQ'};
num_features = 2;
level        = 0;
max_depth    = 10;

hyper                   = gpr_hyperparameters();
covariances_root        = covariance_grammar_started(root,hyper);
for i=1:numel(covariances_root)
    num_hyps = numel(covariances_root{i}.priors);
    covariances_root{i}.fixed_hyps = inf*ones(num_hyps,1);
end

[base_covs] = covariance_builder_grammar(covariances_root,level);
[base_covs_masked,base_names] = mask_kernels(base_covs, num_features);

fprintf('Expanding around %s\n', covariance2str(base_covs_masked{1}.fun))
covariances = expand_covariance(base_covs_masked{1},base_covs_masked,base_names,max_depth);

names_expanded = cell(size(covariances));
for i = 1:numel(covariances)
    names_expanded{i} = covariance2str(covariances{i}.fun);
end


fprintf('Expanding around %s\n', covariance2str(covariances{10}.fun))
covariances = expand_covariance(covariances{10},base_covs_masked,base_names,max_depth);

names_expanded'

names_expanded = cell(size(covariances));
for i = 1:numel(covariances)
    names_expanded{i} = covariance2str(covariances{i}.fun);
end

names_expanded'

