function result = add_noise_covariance(K, theta, x, z, i, j)

indices = fix(K{1}(:));
noise = K{2}';
K = K(3);
% expand cell
if (iscell(K{:}))
    K = K{:};
end

if (nargin == 0)
    error('gpml_extensions:missing_argument', ...
        'covariance input K is required!');
elseif (nargin <= 2)
    result = feval(K{:});
elseif (nargin == 3)
    mask = ismember(x,indices);
    result = feval(K{:},theta,x);
    result(mask,mask) = result(mask,mask) + diag(noise(x(mask)));
elseif (nargin == 4)
    mask = ismember(x,indices);
    result = feval(K{:},theta,x, z);
    if sum(mask) > 0
        if strcmp(z, 'diag')
            result(mask) = result(mask) + noise*ones(sum(mask));
        else
            if isempty(z)
                result(mask,mask) = result(mask,mask) + diag(noise(x(mask)));
            end
        end
    end
elseif (nargin == 5)
    result = feval(K{:}, theta, x, z, i);
elseif  (nargin == 6)
    result = feval(K{:}, theta, x, z, i, j);
end

end