function hyps = prior_sample(priors,samples)
    n = size(samples,1);
    prior_mean = zeros(1,numel(priors));
    prior_std  = zeros(1,numel(priors));
    for i=1:numel(priors)
        func    = functions(priors{i});
        prior_mean(i) = func.workspace{1}.extra_arguments{1}(1);
        prior_std(i)  = sqrt(func.workspace{1}.extra_arguments{2}(1));
    end
    hyps = norminv(samples(:,1:numel(priors)),repmat(prior_mean,[n,1]),repmat(prior_std,[n,1]));
end