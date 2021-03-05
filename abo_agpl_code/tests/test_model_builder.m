clear all;
clc;

hyper_param = gpr_hyperparameters();
level       = 1;
covariances_root_names = {'SE','RQ'};
true_covariance_name   = '(SE+RQ)';
covariances_root       = covariance_grammar_started(covariances_root_names,hyper_param);
[covariances,covariances_names] = covariance_builder_grammar(covariances_root,level);

covariances_names'
find(cellfun(@(x) strcmp(x,true_covariance_name), covariances_names, 'UniformOutput', 1))