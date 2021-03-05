function covs = get_next_covariances(cov, base_covs)
    covs = get_next_covariances_recur(cov, base_covs, ' ');
end

function covs = get_next_covariances_recur(cov, base_covs, mode)
% Find next kernels by:
%   - Replacing S with S + B or S * B, where 
%     S is a subexpression of kernel and B is a base kernel.
%   - Replacing any B with B', where B' is another base kernel.

covs = {};

% append new kernel to existing kernel.
for i=1:length(base_covs)
    if mode ~= '+'
        covs = [covs, combine_tokens('+', base_covs{i}, cov)];
    end
    if mode ~= '*'
        covs = [covs, combine_tokens('*', base_covs{i}, cov)];
    end
end

isBase = false;
if all(size(cov.fun) == [1,1])
    isBase = true;
else
    [op, ~] = cov.fun{:};
    if strcmp(func2str(op), 'mask_covariance')
        isBase = true;
    end
end

% If kernel is a base kernel, replace it with the other base kernels
if isBase
    for i=1:length(base_covs)
        if ~isequal(base_covs{i}.fun, cov.fun)
            covs = [covs, base_covs{i}];
        end
    end
else
    
% If kernel is a composition of kernels, recur on sub-expressions.
    [op, args] = cov.fun{:};
    
    if isequal(op, @prod_covariance_fast) || isequal(op, @sum_covariance_fast)
        args = args{1};
    end
    
    num_hyp = eval(feval(args{1}{:}));
    
    token_1.fun = args{1};
    token_2.fun = args{2};
    if num_hyp == 0
        token_1.priors = {};
        token_2.priors = {};
    else
        token_1.priors = cov.priors(1:num_hyp);
        token_2.priors = cov.priors(num_hyp+1:end);
    end
    
    %token_1.fixed_hyps = cov.fixed_hyps(1:num_hyp);
    %token_2.fixed_hyps = cov.fixed_hyps(num_hyp+1:end);
    
    %recur on first arguement
    if isequal(op, @prod_covariance) || isequal(op, @prod_covariance_fast)
        next_candidates = get_next_covariances_recur(token_1, base_covs, '*');
        for nc=next_candidates
            covs = [covs, combine_tokens('*', nc{:}, token_2)];
        end
    elseif isequal(op, @sum_covariance) || isequal(op, @sum_covariance_fast)
        next_candidates = get_next_covariances_recur(token_1, base_covs, '+');
        for nc=next_candidates
            covs = [covs, combine_tokens('+', nc{:}, token_2)];
        end
    else
       error('Operation not defined.'); 
    end

    
    %if arguments are different, recur on second argument
    if isequal(args{1}{:}, args{2}{:})
        return;
    end
    if isequal(op, @prod_covariance) || isequal(op, @prod_covariance_fast)
        next_candidates = get_next_covariances_recur(token_2, base_covs, '*');
        for nc=next_candidates
            covs = [covs, combine_tokens('*', token_1, nc{:})];
        end
    elseif isequal(op, @sum_covariance) || isequal(op, @sum_covariance_fast)
        next_candidates = get_next_covariances_recur(token_2, base_covs, '+');
        for nc=next_candidates
            covs = [covs, combine_tokens('+', token_1, nc{:})];
        end
    else
        error('Operation not defined.')
    end
end

end