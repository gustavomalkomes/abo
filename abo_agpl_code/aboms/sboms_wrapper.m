function [models, log_evidence, prior] = ...
    sboms_wrapper(problem,models,prior,x,y)

warning('off') 

covariances_root    =  {'SE','RQ'};
data_noise          = 0.001;
theta               = gpr_hyperparameters(data_noise);
covariances         = covariance_grammar_started(covariances_root,theta);
for i=1:numel(covariances)
    covariances{i}.fixed_hyps = inf*ones(numel(covariances{i}.priors),1);
end
d                   = problem.d;
if d > 1
    covariances = mask_kernels(covariances, d);
end

problem.theta_models         = theta;
problem.n                    = size(x,1);
problem.points               = x;
problem.y                    = y;
problem.covariances          = covariances;
 [~, models, prior, log_evidence] = sboms(problem, models, prior);

end
