function hyp = get_hyperparameters(prior)

n_hyp = numel(prior);
hyp = zeros(n_hyp,1);

for i = 1:n_hyp,
    hyp(i) = prior{i}();
end
