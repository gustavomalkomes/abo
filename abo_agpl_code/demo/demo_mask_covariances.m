

data_noise       = 0.01;
d                = 2;
theta            = gpr_hyperparameters(data_noise);
covariances_root = covariance_grammar_started({'SE', 'RQ'}, theta, d);
[base_cov_masked] = mask_kernels(covariances_root, d);
[covariances, names] = fully_expand_tree(base_cov_masked, 1);
