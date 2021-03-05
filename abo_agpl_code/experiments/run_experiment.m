function run_experiment(exp_name, method, fun, exp_initial_seed)

addpath(genpath('../../'))

f        = @(x) feval(fun, x);
fun_info = f([]);

if contains(fun, 'grid')
    lb = fun_info.x_min;
else
    lb = fun_info.lb;
end

if exist(['./config/', exp_name, '_configurations.m'],'file')==2
    feval([exp_name, '_configurations'])
else
    default_configurations;
end

% if numel(method) > 4 && (strcmp(method(1:4),'ambo') || strcmp(method(1:4),'mcmc'))
%     boms_eval_budget  = str2num(method(5:end));
% end

if contains(fun, 'grid')
    query_strategy = @expected_improvement_limited_optimization_discrete;
else
    query_strategy = @expected_improvement_limited_optimization;
end

if contains(exp_name, 'mm')
   query_strategy = @multi_model_ei; 
end

theta = gpr_hyperparameters(data_noise);

if strcmp(method, 'SE')
    root = {'SE'};
elseif strcmp(method, 'RQ')
    root = {'RQ'};
elseif strcmp(method, 'SEard')
    root = {'SEard'};
elseif strcmp(method, 'SEfactor')
    root = {'SEfactor'};    
elseif strcmp(method, 'SEadd')
    root = {'SE'};
else    
    root = [];
end

% Start the optimization
if contains(fun,'grid')
    f            = @(x) feval(fun, x);
    fun_info     = f([]);
    x_pool       = fun_info.x_pool;
    optimum      = fun_info.min;
    label_oracle = @(problem,x_train, y_train, x_star) f(x_star);
    
    lb           = min(x_pool);
    ub           = max(x_pool);
else
    fprintf('Optimizing function %s ...\n', fun);
    f        = @(x) feval(fun, x);
    fun_info = f([]);
    try
        optimum  = fun_info.min;
    catch
        optimum  = NaN;
    end
    
    lb       = fun_info.lb;
    ub       = fun_info.ub;
    
    map_to_problem    = @(x) bsxfun(@plus, bsxfun(@times, x, (ub-lb)), lb);
    label_oracle      = @(problem,x_train, y_train, x_star) ...
        f(map_to_problem(x_star));
end

problem.budget               = num_queries;
problem.function_name        = fun;
problem.optimization         = opt;
problem.verbose              = verbose;
problem.label_oracle         = label_oracle;
problem.lb                   = lb;
problem.ub                   = ub;
problem.optimum              = optimum;
problem.save_interval        = save_interval;
problem.d                    = d;
problem.max_num_models       = max_num_models;
problem.prediction_function  = prediction_function;
problem.max_candidates       = max_candidates;
problem.eval_budget          = boms_eval_budget;
problem.exploit_budget       = exploit_budget;
problem.explore_budget       = explore_budget;
problem.total_hyp_samples    = total_hyp_samples;
problem.k                    = param_k;
problem.covariance_root      = covariances_root;
problem.data_noise           = data_noise;

folder_log = ['./', save_folder, '/', problem.function_name, '/log/'];
if ~exist(folder_log, 'dir')
    mkdir(folder_log)
end

folder = ['./', save_folder, '/', problem.function_name, '/'];
if ~exist(folder, 'dir')
    mkdir(folder)
end

folder_extra = ['./', save_folder, '_extra/', problem.function_name, '/'];
if ~exist(folder_extra, 'dir')
    mkdir(folder_extra)
end


y_initial = [];
for j = 1:num_exp
    
    exp_num = exp_initial_seed + j;
    rng(exp_num, 'twister');
    

    initial_models = get_initial_models(d, data_noise, prediction_function);

    fprintf('Experiment number %d\n', exp_num);
    
    problem.name = [fun, '_', method, '_', num2str(exp_num)];
    
    save_file_name   = [fun, '_', exp_name, '_', method, ...
        '_s', num2str(exp_num)];
    
    % initilization with random points and a sobol x_pool
    fprintf('Initialization. Num of initial points %d\n', num_init_points);
    
    if contains(fun, 'grid')
        
        n_grid_points  = size(x_pool,1);
        used           = false(size(x_pool,1),1);
        init_idx       = randperm(n_grid_points,num_init_points);
        used(init_idx) = true;
        
        x              = x_pool(init_idx,:);
        y              = zeros(num_init_points,1);
        
        for k = 1:num_init_points
            y(k) = problem.label_oracle([], [], [], init_idx(k));
        end
        
        problem.x_pool               = x_pool;
        problem.used                 = used;
        problem.initial_x            = x;
        problem.initial_y            = y;
        problem.isdiscrete           = true;
    else
        problem = initilization_random_points(problem, d, ...
            num_grid_points, num_init_points);
        problem.isdiscrete            = false;
    end
    
    
    fprintf('initial ys: %f ... %f\n', ...
        problem.initial_y(1), problem.initial_y(num_init_points));
    
    % Run Active GP Learning
    if ~isempty(root)
        covariances_root = covariance_grammar_started(root,theta,d);
        initial_models           = gpr_model_builder(covariances_root{1},...
            theta,prediction_function);
        initial_models           = {initial_models};
    end
    
    if strcmp(method,'SEadd')
       [cov,~] = mask_kernels(covariances_root, d);
       fully_additive = cov{1};
        for ii = 2:d
            fully_additive = combine_tokens('+', fully_additive, cov{ii});
        end
        initial_models           = gpr_model_builder(fully_additive,...
                                    theta,prediction_function);
        initial_models           = {initial_models};
    end
    
    models = initial_models;
    
    log_filename         = [folder_log, save_file_name,'_log.txt'];
    callback             = @(problem, models, x, y, i, context)  ...
    save_tracker_callback(log_filename, problem, models, x, y, i, context);


    tstart = tic;
    if numel(method) > 2 && strcmp(method(1:3),'abo')
        [x_star , y_star, context, models] = ...
            abo(problem, models, update_models, query_strategy, callback);
    elseif numel(method) > 2 && strcmp(method(1:3),'sbo')
        [x_star , y_star, context, models] = ...
            sbo(problem, models, update_models, query_strategy, callback);
        
    elseif numel(method) > 3 && strcmp(method(1:4),'mcmc')
        [x_star , y_star, context] = ...
            mcmc(problem, models, update_models, query_strategy, callback);
    else
        [x_star , y_star, context] = ...
            agpl(problem, models, update_models, query_strategy, callback);
    end
    
    telapsed = toc(tstart);
    stopwatch = telapsed;
    
    problem = rmfield(problem,'x_pool');
    problem = rmfield(problem,'used');
    time = context.time;
    
    output_file = [folder, save_file_name,'_complete'];
    save(output_file, 'problem', 'y_star', 'exp_num', 'stopwatch', 'time');
    
    output_file = [folder_extra, save_file_name,'_extra_data'];
    save(output_file, 'problem', 'context', 'models');
    
end

end

function models = get_initial_models(d, data_noise, pred_function)

problem.d            = d;
problem.theta_models = gpr_hyperparameters(data_noise);
explore_budget       = 20;

[new_covs,new_names] = ...
    select_new_candidates(problem, explore_budget, 0, [], [], 1);

models = {};
for i = 1:numel(new_covs)
    models{i} = gpr_model_builder(new_covs{i},problem.theta_models,...
        pred_function);
    fprintf('Model: %s \n', new_names{i})
end

end
