function prior = create_bo_prior(x, samples, noise, data_subsample_size)
    prior.n             = 0;
    prior.K             = [];
    prior.num_hyps      = [];
    prior.candidates    = [];
    prior.evaluated     = [];
    prior.covariances   = {};
    prior.names         = {};
    ind                 = randsample(1:size(x,1),min(size(x,1),data_subsample_size));
    prior.x             = x(ind,:);
    prior.samples       = samples;
    prior.update        = @update_bo_prior;
    prior.cov_matrices  = {};
    
    % create prior mean and covariance
    hyps = boms_hyperparameters(noise);
    % mean.priors = {get_prior(@gaussian_prior, hyps.model_temperature_mean, hyps.model_temperature_var), ...
    %                get_prior(@gaussian_prior, hyps.model_offset_mean, hyps.model_offset_var)};
    % mean.fun    = {@(varargin) fixed_exp_decay_mean(prior.num_hyps,varargin{:})}; 
    mean.priors = {get_prior(@gaussian_prior, hyps.mean_offset, hyps.mean_var)};
    mean.fun    = {@constant_mean}; 
    cov.priors  = {get_prior(@gaussian_prior, hyps.model_length_scale_mean, hyps.model_length_scale_var), ...
                   get_prior(@gaussian_prior, hyps.output_scale_mean, hyps.output_scale_var)};
    %cov.fun     = {@(varargin) fixed_distance_Materniso_covariance(prior.K,varargin{:})};
    
    cov.fun     = {@(varargin) fixed_distance_SEiso_covariance(prior.K,varargin{:})};
    
    prior.noise_prior = {get_prior(@gaussian_prior, hyps.lik_noise_std, hyps.lik_noise_std_var)};
    
    prior.mean = mean;
    prior.cov = cov;
end