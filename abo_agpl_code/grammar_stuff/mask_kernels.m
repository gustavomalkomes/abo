function [new_set_of_covariances, new_set_of_names] = mask_kernels(covariances, num_features)

new_set_of_covariances = cell(1,numel(covariances)*num_features);
new_set_of_names       = cell(1,numel(covariances)*num_features);
c = 0;

for i = 1:numel(covariances)
    for j = 1:num_features
        %fprintf('Applying mask %d to %s\n', j, covariance2str(covariances{i}.fun));
        mask_features = j;
        
%        {@sum_covariance, {m1.fun,m2.fun}}
        new_cov.fun = {@mask_covariance, {mask_features,covariances{i}.fun}};
        
        new_cov.priors = covariances{i}.priors;
%        new_cov.fixed_hyps = covariances{i}.fixed_hyps;
        
        c = c + 1;
        new_set_of_covariances{c} = new_cov;
        new_set_of_names{c} = covariance2str(new_cov.fun);
    end
end

end
