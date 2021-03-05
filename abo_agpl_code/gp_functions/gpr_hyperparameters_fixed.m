function theta = gpr_hyperparameters_fixed(varargin)

if nargin == 0
    data_noise = 0.01;
else
    data_noise = varargin{1};
end

variance  = 0.00000001;

theta.length_scale_mean  = log(0.1);
theta.length_scale_var   = variance;

theta.output_scale_mean  = log(0.4);
theta.output_scale_var   = variance;

theta.p_length_scale_mean  = log(2);
theta.p_length_scale_var   = variance;

theta.p_mean               = log(0.1);
theta.p_var                = variance;

theta.alpha_mean           = log(0.05);
theta.alpha_var            = variance;

theta.lik_noise_std        = log(data_noise);
theta.lik_noise_std_var    = variance;

theta.mean_offset          = 0;
theta.mean_var             = variance;

end
