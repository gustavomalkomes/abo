function theta = gpr_hyperparameters(varargin)

if nargin == 0
    data_noise = 0.01;
else
    data_noise = varargin{1};
end


theta.length_scale_mean  = log(0.1);
theta.length_scale_var   = 1;

theta.output_scale_mean  = log(0.4);
theta.output_scale_var   = 1;

theta.p_length_scale_mean  = log(2);
theta.p_length_scale_var   = 0.5;

theta.p_mean               = log(0.1);
theta.p_var                = 0.5;

theta.alpha_mean           = log(0.05);
theta.alpha_var            = 0.5;

theta.lik_noise_std        = log(data_noise);
theta.lik_noise_std_var    = 1;

theta.mean_offset          = 0;
theta.mean_var             = 1;

end
