

x                   = randn(10,2);
y                   = randn(10,1);

d                   = 2;
eval_budget         = 2;
explore_budget      = 5;
exploit_budget      = 8;
k                   = 10; % top k best models for exploitation
max_candidates      = 200;
total_hyp_samples   = 50;
data_noise          = 0.0001;
max_num_models      = 30;
prediction_function = 'bbq';
theta               = gpr_hyperparameters(data_noise);

opt.num_restarts             = 2;
opt.minFunc_options.Display  = 'off';
opt.minFunc_options.MaxIter  = 1000;
opt.display                  = 0;

problem.data_noise           = data_noise;
problem.optimization.options = opt;
problem.total_hyp_samples    = total_hyp_samples;
problem.eval_budget          = eval_budget;
problem.explore_budget       = explore_budget;
problem.exploit_budget       = exploit_budget;
problem.max_candidates       = max_candidates;
problem.theta_models         = theta;
problem.prediction_function  = prediction_function;
problem.k                    = k;
problem.max_num_models       = max_num_models;
problem.d                    = d;

d                            = problem.d;
max_num_models               = problem.max_num_models;

theta                        = gpr_hyperparameters(problem.data_noise);

problem.covariances          = covariances;
problem.n                    = size(x,1);
problem.points               = x;
problem.y                    = y;
problem.theta_models         = theta;

[~,models,log_evidence, covariances] = boms(problem);