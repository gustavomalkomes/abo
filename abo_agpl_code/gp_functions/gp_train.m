% simple gp_update using minimize_minFunc

function [model] = gp_train(x,y,model,options)
% update the model with the values training points (x,y)
% and compute the model evidence

if isfield(model,'theta')
    initial_theta = model.theta;
else
    initial_theta = [];
    if isfield(options, 'num_restarts')
        options.num_restarts = options.num_restarts + 1;
    else
        options.num_restarts = 2;
    end
end

if isfield(options, 'display') && options.display > 0
    fprintf('%s data %-3d x %-3d cov_hyp: %-3d     cov_name: %-50s\n', ...
        datestr(now,'yy-mmm-dd-HH:MM'), size(x,1), size(x,2), ...
        str2num(feval(model.covariance_function{:})), ...
        covariance2str(model.covariance_function));
end

 try
    tic;
    [map_theta, ~, minFunc_output] = ...
        minimize_minFunc(model, x, y, ...
        'initial_hyperparameters', initial_theta, ...
        'num_restarts', options.num_restarts, ...
        'minFunc_options', options.minFunc_options);
    optimization_time=toc;
catch
    fprintf('minFunc failed\n');
    minFunc_output = [];
    tic;
    [map_theta] = minimize_restart(model,x,y) ;
    optimization_time=toc;
end

% TODO: fix this hack
if ~iscell(model.inference_method) 
    % hack to catch infExact
    [post, nlZ, ~] = feval(model.inference_method, map_theta, ...
        model.mean_function, model.covariance_function, model.likelihood, x, y);
    hessian_computation_time = 0;
    L = [];
    HnlZ = [];
else % here we doing Exact_inference
    tic;
    % computing the Hessian
    [post, nlZ, ~, ~, ~, HnlZ] = feval(model.inference_method{:}, map_theta, ...
        model.mean_function, model.covariance_function, model.likelihood, x, y);
    % check if the Hessian is positive definite
    [L,p] = chol(HnlZ.value);
    
    if (p > 0)
        warning('Hessian is not positive definite. Fixing matrix');
        fixed_H = fix_pd_matrix(HnlZ.value);
        [L,p] = chol(fixed_H);
        if p > 0 
            [n,~] = size(fixed_H);
            fixed_H = fixed_H + 1e-4*eye(n);
            L = chol(fixed_H);
        end
    end
    hessian_computation_time = toc;
end

% updating (hyp)parameters and saving Hessian informations
model.theta             = map_theta;
model.HnlZ              = HnlZ;
model.L                 = L;
model.nlZ               = nlZ;
model.optimization_time = optimization_time;
model.hessian_time      = hessian_computation_time;
model.posterior         = post;

if isfield(options, 'display') && options.display > 0
    fprintf('%s lZ: %-4.2f     cov_hyp: %-3d\tcov_name: %-50s\n', ...
        datestr(now,'yy-mmm-dd-HH:MM'), -nlZ, ...
        str2num(feval(model.covariance_function{:})), ...
        covariance2str(model.covariance_function));
end

if isfield(options, 'display') && options.display > 1 && ~isempty(minFunc_output)
    
    if isfield(minFunc_output, 'iterations')
    fprintf('\t\tcov_hyp: %-3d   iter:%-4d        fun_count:%-4d \t\t %4.2f seconds\n', ...
        str2num(feval(model.covariance_function{:})), ...
        minFunc_output.iterations, minFunc_output.funcCount, optimization_time);
    end
    
    fprintf('\t\tcov_hyp: %-3d   H_time:%-4.2f seconds \t\t\t total_time: %4.2f seconds\n', ...
        str2num(feval(model.covariance_function{:})), ...
        hessian_computation_time, hessian_computation_time+optimization_time);
    
    if isfield(minFunc_output, 'firstorderopt')
    fprintf('\t\tcov_hyp: %-3d   opt: %-2.8f \t\t\t\t %s\n\n', ...
        str2num(feval(model.covariance_function{:})), ...
        minFunc_output.firstorderopt, minFunc_output.message);
    end
end