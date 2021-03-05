function model = gpr_model_builder(covariance_model,hyper_param,type)

% lik specification
lik_function  = @likGauss;
priors.lik    = {get_prior(@gaussian_prior, ...
    hyper_param.lik_noise_std, hyper_param.lik_noise_std_var)};

% mean function specification
mean_function = {@constant_mean};
priors.mean   = {get_prior(@gaussian_prior, ...
    hyper_param.mean_offset, hyper_param.mean_var)};

% covariance function specifications
cov_function     = covariance_model.fun;
priors.cov       = covariance_model.priors;

prior            = get_prior(@independent_prior, priors);

inference_method = @exact_inference;
inference_method = {@inference_with_prior, inference_method, prior};

if strcmp(type, 'gp')
    prediction       = @gp;
    %inference_method = @infExact;
elseif strcmp(type, 'bbq')
    prediction       = @bbq;
elseif strcmp(type, 'ut')
    prediction       = @mgp_ut;
end

param.priors              = priors;
param.prior               = prior;
param.mean_function       = mean_function;
param.covariance_function = cov_function;
param.likelihood          = lik_function;
param.inference_method    = inference_method;

model.parameters   = param;
model.log_evidence = @gp_log_evidence;
model.update       = @gp_update;
model.entropy      = @gpr_entropy;
model.prediction   = prediction;