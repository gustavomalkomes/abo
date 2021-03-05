function param = boms_hyperparameters(varargin)

if nargin == 0
    data_noise = 0.5;
else
    data_noise = varargin{1};
end


param.length_scale_mean    = log(0.1);
param.length_scale_var     = 0.5;
param.p_mean               = log(0.1);
param.p_var                = 0.5;

param.output_scale_mean    = log(0.4);
param.output_scale_var     = 0.5;

param.p_length_scale_mean  = log(2);
param.p_length_scale_var   = 0.5;

param.alpha_mean           = log(0.05);
param.alpha_var            = 0.5;

param.lik_noise_std        = log(data_noise);
param.lik_noise_std_var    = 1;

param.offset_mean          = 0;
param.offset_var           = 10*param.output_scale_var;

param.mean_offset          = 0;
param.mean_var             = 1;

param.model_temperature_mean = log(25);
param.model_temperature_var  = log(25);

param.model_offset_mean      = 0;
param.model_offset_var       = log(1.5);

param.model_length_scale_mean = log(.5);
param.model_length_scale_var  = 1;

param.model_output_scale_mean = log(.4);
param.model_output_scale_var  = 1;


end