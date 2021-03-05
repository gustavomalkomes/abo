function A = fix_pd_matrix(A,epsilon)

[n,m] = size(A);

% set a epi
if (nargin)<2
    epsilon = 1E-6;
end

% is A a square matrix?
if (n~=m) || (length(size(A))>2)
    error('This matrix is not square');
end

[~,W] = eig(A);

% make it symetric
A = (A + A')/2;
% make it not so will-conditioned
%A = A + epsilon*eye(n);
[V,W] = eig(A);

% fix the eig values of the A
new_diag = diag(W);
new_diag(new_diag<epsilon) = epsilon;
W = diag(new_diag);
A = V*W*V';

% make sure that it is still symmetric
A = (A + A')/2;
%A = A + epsilon*eye(n);
%A = A + eye(size(A)) * max(0,1e-12 - min(real(eig(A))))

end