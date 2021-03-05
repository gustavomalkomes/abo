function prior = update_bo_prior(prior, new_covariances, new_names, new_evaluated, removed_candidates)

% Computes the necessary entries of K
% Inputs:
%   prior              - previous bo prior
%   new_covariances    - covariances functions of new candidates
%   new_evaluated      - indices of newly evaluated covariances
%   removed_candidates - indices of candidates to be removed from
%                        consideration

if nargin < 5
    removed_candidates = [];
end
if nargin < 4
    new_evaluated = [];
end
if nargin < 3
    new_covariances = {};
    new_names       = {};
end

if numel(new_covariances) ~= numel(new_names)
    error('# of new covs must match # of new names');
end

prior.candidates = setdiff(prior.candidates, removed_candidates);

n_new = numel(new_covariances);

% caculate number of hyperparameters
% if n_new > 0
%     prior.num_hyps = [prior.num_hyps; zeros(n_new,1)];
%     for i=1:numel(new_covariances)
%         prior.num_hyps(prior.n+i) = eval(feval(new_covariances{i}.fun{:}));
%     end
% end

if n_new > 0
    % add new candidates
    n = prior.n;
    prior.candidates  = [prior.candidates; ((n+1):(n+n_new))'];
    prior.covariances = [prior.covariances, new_covariances];
    prior.names       = [prior.names, new_names];
    % expand prior size
    K = 1 - eye(n+n_new);
    %K = sparse(n+n_new);
    K(1:n,1:n) = prior.K;
    prior.K = K;
    prior.n = n+n_new;
end

covariances   = prior.covariances;
samples       = prior.samples;
x             = prior.x;
noise_prior   = prior.noise_prior;
noise_samples = exp(prior_sample(noise_prior,samples(:,end)));
if n_new > 0
    % precompute covariances matrices
    for i=(n+1):(n+n_new)
        saved_covs = cell(1,size(samples,1));
        cov = covariances{i};
        hyps  = prior_sample(cov.priors,samples);
        for s = 1:size(samples,1)
            k = feval(cov.fun{:}, hyps(s,:),  x);
            k = k + noise_samples(s)*eye(size(k,1));
            saved_covs{s} = k;
        end
        prior.cov_matrices{i} = saved_covs;
    end
end

if n_new > 0
    % compute alignment between new candidate models and old evaluated
    % models.
    for i=(n+1):(n+n_new)
        for j=prior.evaluated'
            alignment = compute_alignment(prior.cov_matrices{i}, ...
                                          prior.cov_matrices{j});
            prior.K(i,j) = alignment;
            prior.K(j,i) = alignment;
        end
    end
end

% compute alignment for newly evaluated models;
n_eval = numel(new_evaluated);
if n_eval > 0
    prior.evaluated  = [prior.evaluated; new_evaluated];
    prior.candidates = setdiff(prior.candidates, new_evaluated);
    for i=new_evaluated'
        % compute alignemnt between newly evaluated models
        for j=new_evaluated'
            if i < j
                alignment = compute_alignment(prior.cov_matrices{i}, ...
                                              prior.cov_matrices{j});
                prior.K(i,j) = alignment;
                prior.K(j,i) = alignment;
            end
        end
        
        % compute alignment between newly evaluated models and candidates
        for j=prior.candidates'
            alignment = compute_alignment(prior.cov_matrices{i}, ...
                                          prior.cov_matrices{j});
            prior.K(i,j) = alignment;
            prior.K(j,i) = alignment;
        end
    end
end

num_eval = numel(prior.evaluated);
num_cand = numel(prior.candidates);
K = 1 - eye(num_eval + num_cand);
K(1:num_eval, 1:num_eval) = prior.K(prior.evaluated, prior.evaluated);
K(1:num_eval, (num_eval+1):(num_eval+num_cand)) = prior.K(prior.evaluated, prior.candidates);
K((num_eval+1):(num_eval+num_cand), 1:num_eval) = prior.K(prior.candidates, prior.evaluated);

covariances = cell(1,num_eval + num_cand);
covariances(1:num_eval) = prior.covariances(prior.evaluated);
covariances((num_eval+1):(num_eval+num_cand)) = prior.covariances(prior.candidates);

names = cell(1,num_eval + num_cand);
names(1:num_eval) = prior.names(prior.evaluated);
names((num_eval+1):(num_eval+num_cand)) = prior.names(prior.candidates);

cov_matrices = cell(1,num_eval + num_cand);
cov_matrices(1:num_eval) = prior.cov_matrices(prior.evaluated);
cov_matrices((num_eval+1):(num_eval+num_cand)) = prior.cov_matrices(prior.candidates);

prior.covariances  = covariances;
prior.names        = names;
prior.cov_matrices = cov_matrices;
prior.evaluated    = (1:num_eval)';
prior.candidates   = ((num_eval+1):(num_eval+num_cand))';
prior.K            = K;
prior.n            = num_eval+num_cand;

% update prior
%prior.mean.fun = {@(varargin) fixed_exp_decay_mean(prior.num_hyps,varargin{:})};
%prior.cov.fun  = {@(varargin) fixed_distance_Materniso_covariance(prior.K,varargin{:})};

fixed_distance_SEiso_covariance(prior.K);
prior.cov.fun  = {@fixed_distance_SEiso_covariance, []};


end

%% Computes the average prior alignment over hyperparameter samples
function alignment = compute_alignment(covs1, covs2)
    alignment = 0;
    n = numel(covs1);
    for s = 1:n
        k1 = covs1{s};
        k2 = covs2{s};
        m = zeros(size(k1,1),1);
        alignment = alignment + hellinger_distance(m,k1,m,k2);
    end
    alignment = alignment/n;
end