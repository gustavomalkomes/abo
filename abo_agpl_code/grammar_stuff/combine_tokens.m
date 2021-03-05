function m = combine_tokens(op, m1,m2)

result = {{m1.fun, m2.fun} , precompute_stuff({m1.fun, m2.fun})};

switch op
    case '+'
        m.fun = {@sum_covariance_fast, result};
    case '*'
        m.fun = {@prod_covariance_fast, result};
    otherwise
        error('Unknown operation');
end
m.priors = {m1.priors{:}, m2.priors{:}};
if isfield(m1, 'fixed_hyps') && isfield(m2, 'fixed_hyps')
    m.fixed_hyps = [m1.fixed_hyps; m2.fixed_hyps];
end
end

function stuff = precompute_stuff(cov)

for ii = 1:numel(cov)                        % iterate over covariance functions
  f = cov(ii); if iscell(f{:}), f = f{:}; end   % expand cell array if necessary
  j(ii) = cellstr(feval(f{:}));                          % collect number hypers
end

v = [];               % v vector indicates to which covariance parameters belong
for ii = 1:length(cov)
    v = [v repmat(ii, 1, eval(char(j(ii))))]; 
end

stuff = {j, v};
end