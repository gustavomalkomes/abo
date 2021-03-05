% simple gp_update using minimize_minFunc

function [model] = gp_no_train(x,y,model,options)
% update the model with the values training points (x,y) 
% and compute the model evidence


theta = model.theta;

% computing the Hessian
[~, nlZ, ~, ~, ~, HnlZ] = feval(model.inference_method{:}, theta, ...
    model.mean_function, model.covariance_function, model.likelihood, x, y);

% check if the Hessian is positive definite
[L,p] = chol(HnlZ.value);

if (p > 0)
    warning('Hessian is not positive definite. Fixing matrix');
    fixed_H = fix_pd_matrix(HnlZ.value);
    L = chol(fixed_H);
end


% updating (hyp)parameters and saving Hessian informations
model.theta             = theta;
model.L                 = L;
model.nlZ               = nlZ;

end