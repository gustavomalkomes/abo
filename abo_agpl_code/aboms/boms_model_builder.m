function model = boms_model_builder(covariance_model, hyper_or_noise, mean_model)

if isstruct(hyper_or_noise)
    hyper_param = hyper_or_noise;
else
    if hyper_or_noise > 0
        hyper_param = gpr_hyperparameters(hyper_or_noise);
    else
        hyper_param = gpr_hyperparameters();
    end
end

% mean function specification
if nargin == 3
    if isstruct(mean_model)
        mean_function = mean_model.fun;
        priors.mean   = mean_model.priors;
    else
        hyper_param.mean_offset = mean_model;
        mean_function = {@constant_mean};
        priors.mean   = {get_prior(@gaussian_prior, hyper_param.mean_offset, hyper_param.mean_var)};
    end
else
    mean_function = {@constant_mean};
    priors.mean   = {get_prior(@gaussian_prior, hyper_param.mean_offset, hyper_param.mean_var)};
end


% lik specification
lik_function  = @likGauss;
priors.lik    = {get_prior(@gaussian_prior, hyper_param.lik_noise_std, hyper_param.lik_noise_std_var)};

% covariance function specifications
cov_function     = covariance_model.fun;
priors.cov       = covariance_model.priors;

prior            = get_prior(@independent_prior, priors);
inference_method = @exact_inference;
inference_method = {@inference_with_prior, inference_method, prior};

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