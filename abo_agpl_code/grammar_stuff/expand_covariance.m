function covs = expand_covariance(cov, base_covs, base_names, max_depth)
count = 0;
name  = covariance2str(cov.fun);
% check if a candidate has the same combination of base
% covariances as an already used covariance.
for k=1:length(base_names)
    count = count + length(strfind(name,base_names{k}));
end
if count > max_depth
    covs = {};
elseif count == max_depth
    covs = expand_covs_recur(cov, base_covs);
else
    covs = get_next_covariances(cov, base_covs);
    covs = [covs, expand_covs_recur(cov, base_covs)];
end
end


function covs = expand_covs_recur(cov, base_covs)
covs = {};

% If kernel is a base kernel, return
if all(size(cov.fun) == [1,1])
    return;
else
    [op, ~] = cov.fun{:};
    if strcmp(func2str(op), 'mask_covariance')
        return;
    end
end

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

% switch operator
if isequal(op, @prod_covariance) || isequal(op, @prod_covariance_fast)
    covs = [covs, combine_tokens('*', token_1, token_2)];
elseif isequal(op, @sum_covariance) || isequal(op, @sum_covariance_fast)
    covs = [covs, combine_tokens('+', token_1, token_2)];
else
    error('Operation not defined')
end

% remove a base kernel
if all(size(token_1.fun) == [1,1])
    covs = [covs, token_2];
end
if all(size(token_2.fun) == [1,1])
    covs = [covs, token_1];
end

% recur on arguements
new_covs_1 = expand_covs_recur(token_1, base_covs);
new_covs_2 = expand_covs_recur(token_2, base_covs);

if isequal(op, @prod_covariance) || isequal(op, @prod_covariance_fast)
    for nc=new_covs_1
        covs = [covs, combine_tokens('*', nc{:}, token_2)];
    end
    for nc=new_covs_2
        covs = [covs, combine_tokens('*', token_1, nc{:})];
    end
elseif isequal(op, @sum_covariance) || isequal(op, @sum_covariance_fast)
    for nc=new_covs_1
        covs = [covs, combine_tokens('+', nc{:}, token_2)];
    end
    for nc=new_covs_2
        covs = [covs, combine_tokens('+', token_1, nc{:})];
    end
else
    error('Operation not defined!');
end
end
