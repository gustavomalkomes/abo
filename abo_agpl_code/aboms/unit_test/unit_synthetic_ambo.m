
rng(0);

covariances_root    =  {'SE','RQ'};
d                   = 2;
eval_budget         = 2;
explore_budget      = 5;
exploit_budget      = 3;
k                   = 3; % top k best models for exploitation
max_candidates      = 200;
random_seed         = 0;
total_hyp_samples   = 50;
data_noise          = 0.0001;
n_points            = 100;
prediction_function = 'bbq';
theta               = gpr_hyperparameters(data_noise);

covariances         = covariance_grammar_started(covariances_root,theta);
for i=1:numel(covariances)
    covariances{i}.fixed_hyps = inf*ones(numel(covariances{i}.priors),1);
end

if d > 1
   covariances = mask_kernels(covariances, d);
end

[tmp_cov, tmp_names] = covariance_builder_grammar_random(covariances,3);
fprintf('True model is %s\n', tmp_names{end});

true_model = gpr_model_builder(tmp_cov{end}, theta, ...
    prediction_function);

x = sobol_sample(n_points,d);
[y, true_theta] = generate_synthetic_data(x,true_model);

opt.num_restarts             = 5;
opt.minFunc_options.Display  = 'off';
opt.minFunc_options.MaxIter  = 1000;
opt.display                  = 0;

problem.n                    = size(x,1);
problem.d                    = size(x,2);

problem.points               = x;
problem.y                    = y;
problem.data_noise           = data_noise;
problem.optimization         = opt;
problem.covariances          = covariances;
problem.total_hyp_samples    = total_hyp_samples;
problem.eval_budget          = eval_budget;
problem.explore_budget       = explore_budget;
problem.exploit_budget       = exploit_budget;
problem.max_candidates       = max_candidates;
problem.theta_models         = theta;
problem.prediction_function  = prediction_function;
problem.k                    = k;


%[best_model, models] = aboms(problem);
[best_model, models, prior, ~] = aboms(problem);
model_name      = covariance2str(best_model.parameters.covariance_function);
fprintf('Best Model: %s \n',model_name);
% 
% [best_model, models, prior, ~] = ...
%                         aboms(problem, models, prior);
% model_name      = covariance2str(best_model.parameters.covariance_function);
% fprintf('Best Model: %s \n',model_name);
% 


[models, log_evidence, prior] = ...
    aboms_wrapper(problem,models,prior,x,y);
