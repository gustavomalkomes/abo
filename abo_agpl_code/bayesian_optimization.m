function [x_star, y_star, context] = bayesian_optimization(problem, ...
    models, ~, callback)

% set verbose to false if not defined
verbose = isfield(problem, 'verbose') && problem.verbose;

% set initial points
x = problem.initial_x;
y = problem.initial_y;

x_star = NaN(problem.budget, size(x,2));
y_star = NaN(problem.budget, size(y,2));

context = [];

x_cand   = problem.x_pool;
used     = problem.used;
n_models = numel(models);

models{1}.parameters.theta = models{1}.parameters.prior();

for i = 1:problem.budget
    if (verbose)
        fprintf('point %i:\n', i);
    end
    
    y_min  = min(y);
    x_pool = x_cand(~used,:);
    n_cand = size(x_pool,1);
    ei = nan(n_cand, n_models);
    
    for m = 1:n_models,        
        gp_model = models{m}.parameters;
        
%         [map_theta, ~, minFunc_output] = ...
%             minimize_minFunc(gp_model, x, y, ...
%             'initial_hyperparameters', [], ...
%             'num_restarts', problem.optimization.num_restarts, ...
%             'minFunc_options', problem.optimization.minFunc_options);
        
%       [map_theta,best_iter,best_nlZ] = minimize_restart(gp_model,x,y);

        hyp = gp_model.theta;
        
        hyp.cov = zeros(size(hyp.cov));
        hyp.mean = zeros(size(hyp.mean));
        hyp.lik = log(0.1);
        
        hyp = minimize(hyp, ...
            @gp,-100,@infExact, ...
            gp_model.mean_function, gp_model.covariance_function, ...
            @likGauss,x,y);
        
        gp_model.theta = hyp;
        
        ei_obj = @(x_star) ei_objective(x_star, gp_model, ...
            x, y, y_min);
        
        ei(:,m) = ei_obj(x_pool);
        
        models{m}.parameters = gp_model;
    end
    
    [ei_val, ei_index] = max(ei);
    used(ei_index)     = true;
    chosen_x_star      = x_pool(ei_index,:);
    
    if problem.verbose
        fprintf(' > f_min: %.8f, a(x): %.8f\n', ...
            y_min, ei_val);
        %disp(chosen_x_star);
    end

    % observe label(s) at chosen location(s)
    this_chosen_label = problem.label_oracle(problem,chosen_x_star);
    
    % update lists with new observation(s)
    x_star(i,:) = chosen_x_star;
    x = [x; chosen_x_star];
    
    y_star(i,:) = this_chosen_label;
    y = [y; this_chosen_label];
    
    % call callback, if defined
    if (nargin > 5)
        callback(problem, models, x, y, i, context);
    end
end

end

% expected improvement as a objective function
function ei = ei_objective(x_star, gp_model, x_train, y_train, y_min)

% gp predictions
[mu,cov] = bbq(gp_model.theta, gp_model.inference_method, ...
    gp_model.mean_function, gp_model.covariance_function, ...
    gp_model.likelihood, x_train, y_train, x_star);

% make sure that cov > 0
cov((cov<0)) = 0;
sigma = sqrt(cov);

% compute expected improvement
delta = (y_min - mu);
u     = delta./sigma;
u_pdf = normpdf(u);
u_cdf = normcdf(u);

ei  = delta .* u_cdf + sigma .* u_pdf;

end


