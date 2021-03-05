function indx = find_covariance(cov_name,names)

indx = find(cellfun(@(x) strcmp(x,cov_name), names, 'UniformOutput', 1));