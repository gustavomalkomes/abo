function covariance_name = covariance2str(covariance_handle)

if numel(covariance_handle) > 1
    fname = func2str(covariance_handle{1});
else
    if iscell(covariance_handle)
        fname = func2str(covariance_handle{:});  
    else
        fname = func2str(covariance_handle);  
    end
end

if(strcmp(fname,'isotropic_sqdexp_covariance'))
    covariance_name = 'SE';
elseif(strcmp(fname,'isotropic_matern_covariance'))
    covariance_name = 'M1';
elseif(strcmp(fname,'periodic_covariance'))
    covariance_name = 'PER';
elseif(strcmp(fname,'linear_covariance'))
    covariance_name = 'LIN';
elseif(strcmp(fname,'isotropic_rq_covariance'))
    covariance_name = 'RQ';
elseif(strcmp(fname,'ard_sqdexp_covariance'))
    covariance_name = 'SEard';
elseif(strcmp(fname,'sum_covariance') )
    covariance_name = ['(',covariance2str(covariance_handle{2}{1}),'+',covariance2str(covariance_handle{2}{2}),')'];
elseif(strcmp(fname,'prod_covariance'))
    covariance_name = ['(',covariance2str(covariance_handle{2}{1}),'*',covariance2str(covariance_handle{2}{2}),')'];
elseif(strcmp(fname,'sum_covariance_fast'))
    covariance_name = ['(',covariance2str(covariance_handle{2}{1}{1}),'+',covariance2str(covariance_handle{2}{1}{2}),')'];
elseif(strcmp(fname,'prod_covariance_fast'))
    covariance_name = ['(',covariance2str(covariance_handle{2}{1}{1}),'*',covariance2str(covariance_handle{2}{1}{2}),')'];
elseif(strcmp(fname,'mask_covariance'))
    covariance_name = covariance2str(covariance_handle{2}{2});
    index = num2str(covariance_handle{2}{1});
    covariance_name = [covariance_name, '_', index];
else
    covariance_name = 'Unk';
end

end