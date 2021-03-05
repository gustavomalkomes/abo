


hyper_param            = gpr_hyperparameters();
level                  = 2;
covariances_root_names = {'SE','RQ'};
true_covariance_name   = '(SE+RQ)';
covariances_root       = covariance_grammar_started(covariances_root_names,hyper_param);
[covariances,covariances_names] = covariance_builder_grammar(covariances_root,level);


problem.data_noise   = 0.01;
problem.theta_models = gpr_hyperparameters();
problem.d            = 10;

explore_budget       = 20;
exploit_budget       = 5;

names                = covariances_names(1:7);
neighborhoods        = {{covariances{10:12}}};
starting_depth       = 5;

% [new_covs,new_names] = select_new_candidates(problem, ...
%     explore_budget, exploit_budget, names, neighborhoods, starting_depth);
% new_names


[new_covs,new_names] = select_new_candidates(problem, explore_budget, 0, [], [], 1);
new_names'