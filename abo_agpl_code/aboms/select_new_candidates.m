function [new_covs,new_names] = select_new_candidates(problem, ...
    explore_budget, exploit_budget, names, neighborhoods, starting_depth)

new_covs  = {};
new_names = {};

depth_prob = 1/2;

%fprintf('%s Exploration... \n', datestr(now,'yy-mmm-dd-HH:MM'))

covariances_root  = covariance_grammar_started({'SE', 'RQ'}, problem.theta_models, problem.d);
[base_cov_masked, base_cov_masked_names] = mask_kernels(covariances_root, problem.d);

% Exploration
while numel(new_covs) < explore_budget
    % allow kernels up to depth 10
    depth = min(10, starting_depth + geornd(depth_prob));
    cov   = base_cov_masked{randi(numel(base_cov_masked))};
%    fprintf('%s Exploration... depth %d \n', datestr(now,'yy-mmm-dd-HH:MM'), depth)
    for i = 1:(depth-1)
        covs = get_next_covariances(cov, base_cov_masked);
        cov  = covs{randi(numel(covs))};
    end
%    fprintf('%s Exploration... cov %s \n', datestr(now,'yy-mmm-dd-HH:MM'), covariance2str(cov.fun))
    [cov,name] = remove_duplicate_candidates({cov}, [names, new_names], base_cov_masked_names);
    if numel(cov) > 0
        new_covs   = [new_covs,  cov];
        new_names  = [new_names, name];
    end
%    fprintf('%s Exploration... %d \n', datestr(now,'yy-mmm-dd-HH:MM'), numel(new_covs))
end

% Exploitation
%fprintf('%s Exploitation... \n', datestr(now,'yy-mmm-dd-HH:MM'))
if exploit_budget > 0
for i=1:numel(neighborhoods)
    if isempty(neighborhoods{i})
        break
    end
    n_neighborhoods_to_sample = min(exploit_budget,numel(neighborhoods{i}));
    covs          = randsample(neighborhoods{i},n_neighborhoods_to_sample);
    [covs,nnames] = remove_duplicate_candidates(covs, [names, new_names], base_cov_masked_names);
    if numel(covs) > 0
        new_covs  = [new_covs,  covs];
        new_names = [new_names, nnames];
    end
end
end

end