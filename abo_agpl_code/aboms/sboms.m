function [best_model, models, prior, y] = sboms(problem, varargin)

%fprintf('%s BOMS... \n', datestr(now,'yy-mmm-dd-HH:MM'))

noise_old_models_constant = 0;

% Constants
total_hyp_samples   = problem.total_hyp_samples;
covariances         = problem.covariances;
num_models          = numel(covariances);
bo_data_noise       = 0.001;
base_covs           = covariances;
max_hyps            = 200;
max_depth           = 10;
eval_budget         = problem.eval_budget;
exploit_budget      = problem.exploit_budget;
explore_budget      = problem.explore_budget;
starting_depth      = 2;
max_candidates      = problem.max_candidates;
top_k               = problem.k;
data_subsample_size = 20;
expand_best         = false;

% optimization parameters
minFunc_options.Display    = 'off';
minFunc_options.Method     = 'lbfgs';
problem.optimization.options.minFunc_options = minFunc_options;

do_BOMS             = isempty(varargin);

% initial setup
models       = {};
base_names   = cell(1,num_models);
for i = 1:num_models
    base_names{i}  = covariance2str(covariances{i}.fun);
    base_hyps(i)   = numel(covariances{i}.priors);
end

train_order    = {};

num_models = numel(covariances);
if do_BOMS
    [covariances, names] = fully_expand_tree(base_covs, ...
        starting_depth, max_candidates);
    num_models = numel(covariances);
    
    samples = sobol_sample(total_hyp_samples, max_hyps);
    prior = create_bo_prior(problem.points, samples, problem.data_noise, data_subsample_size);
    prior = prior.update(prior, covariances, names, []);
else
    % SBOMS
    init_models = varargin{1};
    for i=1:numel(init_models)
        init_cov{i}.fun = init_models{i}.parameters.covariance_function;
        init_cov{i}.priors = init_models{i}.parameters.priors.cov;
        init_names{i} = covariance2str(init_cov{i}.fun);
        noise_old_models(i) = 0;
    end
    
    models = varargin{1};
    prior = varargin{2};
end


x = [];
y = [];

x_star = 1:num_models;
next_x = 1; % train SE model first

hyps           = {};

time.cov       = [];
time.eval      = [];
time.bo_eval   = [];
time.expand    = [];
time.ei        = [];
time.total     = [];

best_log_evidence = -inf;
best_model_index  = 1;
best_scores       = -inf*ones(1,top_k);
best_indices     = zeros(1,top_k);


if ~do_BOMS
    init_num_models = numel(models);
    x      = 1:init_num_models;
    x_star = prior.candidates;
    next_x = randsample(prior.candidates,1);
    y = zeros(init_num_models,1);
    
    for i = 1:init_num_models
        [init_models(i), next_y] = ...
            gp_update(problem, init_models(i), problem.points, problem.y, []);
        y(i) = next_y;
        y(i) = y(i)/numel(problem.y);
        t(i) = init_models{i}.parameters.optimization_time;
        models(i) = init_models(i);
    end
    
    for i = 1:init_num_models
        time.eval = [time.eval, t(i)];
    end
    
    [values, order]   = sort(y, 'descend');
    best_log_evidence = values(1);
    best_model_index  = order(1);
    
    best_scores   = -inf*ones(1,top_k);
    best_indices  = zeros(1,top_k);
    
    for i = 1:(min(top_k, numel(values))-1)
        best_scores(i)  = values(i+1);
        best_indices(i) = order(i+1);
    end
    
    neighborhoods = cell(1,top_k);
end


%% Begin Search
time_start = tic;
for b = 1:eval_budget
    % Train next model.
    eval_start = toc(time_start);
    model_name = covariance2str(prior.covariances{next_x}.fun);
    %fprintf('\n%d: Training %s ... \n', b, model_name);
    new_model = gpr_model_builder(prior.covariances{next_x}, ...
        problem.theta_models, problem.prediction_function);
    
    models                = [models, new_model];
    [models(end), next_y] = gp_update(problem, models(end), problem.points, problem.y, []);
    
    models{end}.parameters.log_evidence = next_y;
    models{end}.parameters.number_points = problem.n;
    
    next_y = next_y/problem.n;
    fprintf('SBOMS. Query > Log evidence/n %f Model %s n = %d\n', ...
        next_y, covariance2str(models{end}.parameters.covariance_function), ...
        problem.n);
    
    num_models            = numel(models);
    time.eval = [time.eval, models{end}.parameters.optimization_time];
    
    % Update best models and results
    new_best = false;
    if best_log_evidence < next_y
        best_log_evidence = next_y;
        best_model_index  = num_models;
        new_best          = true;
    end
    best_scores       = [best_scores, next_y];
    best_indices      = [best_indices, next_x];
    neighborhoods{top_k +1} = [];
    [~,ind]           = sort(best_scores, 'descend');
    
    best_scores       = best_scores(ind(1:(end-1)));
    best_indices      = best_indices(ind(1:(end-1)));
    neighborhoods     = neighborhoods(ind(1:(end-1)));
    train_order       = [train_order, model_name];
    
    % Expand candidate pool
    expand_start = toc(time_start);
    
    %    fprintf('\n%s Expanding Covariance Pool ... \n', datestr(now,'yy-mmm-dd-HH:MM'))
    if any(next_x == best_indices)
        neighborhoods{best_indices == next_x} = ...
            expand_covariance(prior.covariances{next_x}, ...
            base_covs, base_names, max_depth);
    end
    
    %    fprintf('%s Selecting new candidates... \n', datestr(now,'yy-mmm-dd-HH:MM'))
    [new_covs, new_names] = select_new_candidates(problem, ...
        explore_budget, exploit_budget, prior.names, ...
        neighborhoods, starting_depth);
    
    %    fprintf('%s Removing candidates... \n', datestr(now,'yy-mmm-dd-HH:MM'))
    if new_best && expand_best
        [covs_best,names_best] = remove_duplicate_candidates(...
            neighborhoods{1},[prior.names,new_names],base_names);
        
        new_covs  = [new_covs, covs_best];
        new_names = [new_names, names_best];
    end
    time.expand           = [time.expand, (toc(time_start) - expand_start)];
    
    % Update datasets and cov names
    x       = (1:num_models)';
    y       = [y; next_y];
    
    if ~isempty(new_covs)
        x_star  = (num_models+1):(prior.n + numel(new_covs));
    else
        x_star  = (num_models+1):prior.n;
    end
    
    % Update BO model
    cov_start = toc(time_start);
    %    fprintf('%s Updating BO Model ... \n', datestr(now,'yy-mmm-dd-HH:MM'));
    
    prior    = prior.update(prior, new_covs, new_names, next_x);
    best_indices(best_indices == next_x) = num_models;
    time.cov = [time.cov, (toc(time_start) - cov_start)];
    
    bo_eval_start    = toc(time_start);
    kernel_kernel    = boms_model_builder(prior.cov, bo_data_noise, prior.mean);
    
    if ~do_BOMS
        kernel_kernel.parameters.covariance_function = ...
            {@add_noise_covariance, {1:numel(init_models), ...
            noise_old_models, kernel_kernel.parameters.covariance_function}};
    end
    
    try
        kernel_kernel.parameters = ...
            gp_train(x, y, kernel_kernel.parameters, problem.optimization);
    catch
        % Let' try to fix it but removing models that are very similiar
        nn = size(y,1);
        exclude = false(nn,1);
        for i = 1:nn
            for j = i+1:nn
                if(norm(y(i,:)-y(j,:)) < 0.01)
                    exclude(i) = true;
                end
            end
        end
        x = x(~exclude,:);
        y = y(~exclude,:);
        time.eval = time.eval(~exclude);
        kernel_kernel.parameters = gp_train(x, y, ...
            kernel_kernel.parameters, problem.optimization);
    end
    
    kernel_kernel_gp = kernel_kernel.parameters;
    
    % update timing model (OLS)
    num_hyps_x = get_num_hyps(prior.names(x),base_names,base_hyps)';
    if b < 4
        theta = [0,.1];
    else
        x_t   = [ones(size(num_hyps_x)),num_hyps_x];
        y_t   = log(time.eval)';
        theta = (x_t' * x_t + 0.01*eye(size(x_t,2))) \ x_t' * y_t;
    end
    
    time.bo_eval     = [time.bo_eval, (toc(time_start) - bo_eval_start)];
    
    
    % Select next model to evalute
    ei_start = toc(time_start);
    [mu,s2]  = gp(kernel_kernel_gp.theta, @infExact,...
        kernel_kernel_gp.mean_function,...
        kernel_kernel_gp.covariance_function, ...
        kernel_kernel_gp.likelihood, x, y, x_star');
    
    num_hyps = get_num_hyps(prior.names(x_star'),base_names,base_hyps);
    times    = exp(theta(1) + theta(2) * num_hyps);
    %times    = ones(size(times));
    
    if b < eval_budget
        %       fprintf('\nComputing EI ... \n');
        [next_x, scores] = select_best_candidate(max(y), x_star', mu, s2, times);
        [~,ind] = sort(scores);
        ii = 0;
        for i=ind
            ii = ii + 1;
            %            fprintf('ei/s: %.8f model %s (%.4f +- %.4f)\n', scores(i), covariance2str(prior.covariances{x_star(i)}.fun), mu(i), 2*sqrt(s2(i)));
            if ii > 5
                break;
            end
        end
    end
    
    % reduce to max # of candidates
    num_candidates = numel(ind);
    if num_candidates > max_candidates
        cand_to_remove = x_star(ind(1:(num_candidates - max_candidates)));
        prior          = prior.update(prior, {}, {}, [], cand_to_remove);
        next_x         = next_x - sum(cand_to_remove < next_x);
    end
    
    time.ei    = [time.ei, (toc(time_start) - ei_start)];
    time.ei(end);
    time.total = [time.total, toc(time_start)];
end

best_model      = models{best_model_index};

end