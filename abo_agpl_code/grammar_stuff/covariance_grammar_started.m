function covariances = covariance_grammar_started(covariances_names,hyp,varargin)

n           = numel(covariances_names);
covariances = cell(1,n);

for i=1:n
    switch covariances_names{i}
        case 'SEfactor'
            m.fun          = {@factor_sqdexp_covariance, 2};
            d              = 2*varargin{1};
            p              = cell(d,1);
            for ii = 1:d
               p{ii} = get_prior(@gaussian_prior, hyp.length_scale_mean, hyp.length_scale_var);
            end
            m.priors       = p;
            
        case 'SEard'
            m.fun          = {@ard_sqdexp_covariance};
            d              = varargin{1};
            p              = cell(d+1,1);
            for ii = 1:d
               p{ii} = get_prior(@gaussian_prior, hyp.length_scale_mean, hyp.length_scale_var);
            end
            
            p{end} = get_prior(@gaussian_prior, hyp.output_scale_mean, hyp.output_scale_var);
            m.priors       = p;
            
        case 'SE'
            m.fun          = {@isotropic_sqdexp_covariance};
            m.priors       = {get_prior(@gaussian_prior, hyp.length_scale_mean, hyp.length_scale_var), ...
                get_prior(@gaussian_prior, hyp.output_scale_mean, hyp.output_scale_var)};
            
        case 'M1'
            m.fun          = {@isotropic_matern_covariance};
            m.priors       = {get_prior(@gaussian_prior, hyp.length_scale_mean, hyp.length_scale_var), ...
                get_prior(@gaussian_prior, hyp.output_scale_mean, hyp.output_scale_var)};
            
        case 'PER'
            m.fun          = {@periodic_covariance};
            m.priors       = {get_prior(@gaussian_prior, hyp.p_length_scale_mean, hyp.p_length_scale_var), ...
                get_prior(@gaussian_prior, hyp.p_mean, hyp.p_var), ...
                get_prior(@gaussian_prior, hyp.output_scale_mean, hyp.output_scale_var)};
            
        case 'LIN'
            m.fun          = {@linear_covariance};
            m.priors       = {get_prior(@gaussian_prior, hyp.output_scale_mean, hyp.output_scale_var)};
                %get_prior(@gaussian_prior, param.offset_mean, param.offset_var)};

        case 'RQ'
            m.fun     = {@isotropic_rq_covariance};
            m.priors  = {get_prior(@gaussian_prior, hyp.length_scale_mean, hyp.length_scale_var), ...
                get_prior(@gaussian_prior, hyp.output_scale_mean, hyp.output_scale_var), ...
                get_prior(@gaussian_prior, hyp.alpha_mean, hyp.alpha_var), ...
            };
    end
    
    covariances{i} = m;
    
end


end