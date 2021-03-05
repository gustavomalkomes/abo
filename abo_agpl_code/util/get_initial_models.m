function models = get_initial_models(d, data_noise, pred_function)

theta             = gpr_hyperparameters(data_noise);
covariances_root  = covariance_grammar_started({'SE', 'RQ'}, theta, d);
[base_cov_masked] = mask_kernels(covariances_root, d);

covariances       = fully_expand_tree(base_cov_masked, 1);

if d > 2
    
    models_d1     = covariances(2*d+1:end);
    covariances   = covariances(1:2*d);
    
    random_index  = randperm(numel(models_d1), d*4);
    for i = 1:numel(random_index)
        covariances(end+1) = models_d1(random_index(i));
    end
    
    fully_additive = covariances{1};
    
    %% Just for SE %%
    for i = 2:d
        fully_additive = combine_tokens('+', fully_additive, covariances{i});
    end
    covariance2str(fully_additive.fun);
    
    
    %     for i = 1:d
    %         for j = i+1:d
    %             pair = combine_tokens('*', covariances{i}, covariances{j});
    %             if i == 1 && j == 2
    %                 pairwise_additive = pair;
    %             else
    %                 pairwise_additive = combine_tokens('+', pairwise_additive, pair);
    %             end
    %             % covariance2str(pairwise_additive.fun)
    %         end
    %     end
    %     covariance2str(pairwise_additive.fun)
    
    covariances{end + 1} = fully_additive;
    %     covariances{end + 1} = pairwise_additive;
    
    %% Just for RQ %%
    
    fully_additive = covariances{d+1};
    for i = 2:d
        fully_additive = combine_tokens('+', fully_additive, covariances{d+i});
    end
    covariance2str(fully_additive.fun);
    
    %     for i = 1:d
    %         for j = i+1:d
    %             pair = combine_tokens('*', covariances{d+i}, covariances{d+j});
    %             if i == 1 && j == 2
    %                 pairwise_additive = pair;
    %             else
    %                 pairwise_additive = combine_tokens('+', pairwise_additive, pair);
    %             end
    %             % covariance2str(pairwise_additive.fun)
    %         end
    %     end
    %     covariance2str(pairwise_additive.fun)
    
    covariances{end + 1} = fully_additive;
    %     covariances{end + 1} = pairwise_additive;
    
end

covariances(end+1) = covariance_grammar_started({'SEard'}, theta, d);

models = {};
for i = 1:numel(covariances)
    models{i} = gpr_model_builder(covariances{i},theta,pred_function);
    fprintf('%s \n', covariance2str(covariances{i}.fun))
end

end
