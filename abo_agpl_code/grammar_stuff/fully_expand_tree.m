function [covs, names, level_sizes] = fully_expand_tree(base_covs, depth, varargin)

max_number_models = inf;
if length(varargin) == 1
    max_number_models = varargin{1};
end

covs = base_covs;

% get names of base covariances
base_names = cell(1,numel(base_covs));
for i=1:numel(base_covs)
    base_names{i} = covariance2str(base_covs{i}.fun);
end
covs_to_expand = covs;
names = base_names;
level_sizes = zeros(1,depth+1);
level_sizes(1) = numel(base_names);

%expand tree
for i=1:depth
    new_covs ={};
    for j=1:numel(covs_to_expand)
        temp_covs              = expand_covariance(covs_to_expand{j}, base_covs, base_names, depth+2);
        [temp_covs, new_names] = remove_duplicate_candidates(temp_covs, names, base_names);
        names                  = [names, new_names];
        new_covs               = [new_covs, temp_covs];
        
        if numel(covs) + numel(new_covs) >= max_number_models
            break;
        end
    end
    
    covs           = [covs, new_covs];
    covs_to_expand = new_covs;
    level_sizes(i+1) = numel(covs);
    
    if numel(covs) >= max_number_models
        break;
    end
    
end
