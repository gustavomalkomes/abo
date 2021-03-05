% default parameters 
d                    = numel(lb);
save_folder          = '180205_cosmo';
boms_eval_budget     = 5;
num_queries          = 50;
num_exp              = 1;
num_init_points      = 5;
verbose              = 1;
num_grid_points      = (d^2)*2000; % number of grid points per dimension
update_models        = @gp_update;
data_noise           = 0.01;
prediction_function  = 'bbq';

covariances_root     = {'SE','RQ'};
save_interval        = 1;
total_hyp_samples    = 100;
explore_budget       = 10;
exploit_budget       = 5;
max_candidates       = 200;
param_k              = 3;
max_num_models       = 30;

% optimization parameters
opt.num_restarts             = 2;
opt.minFunc_options.Display  = 'off';
opt.minFunc_options.MaxIter  = 1000;
opt.display                  = 0;
