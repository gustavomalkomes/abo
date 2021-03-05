function [tokens,tokens_name] = covariance_builder_grammar_random(tokens,level)

for i=1:numel(tokens)
    tokens_name{i} = covariance2str(tokens{i}.fun);
end

for i=1:level
    new_tokens = {};
    new_tokens_name = {};
    c = 1;
    for t=1:numel(tokens)
        for tt = 1:numel(tokens)
            prospective_name = ['(',covariance2str(tokens{t}.fun),'+',covariance2str(tokens{tt}.fun),')'];
            if (t < tt && ~ismember(prospective_name, tokens_name))
                new_tokens_name{c} = prospective_name;
                new_tokens{c} = combine_tokens('+',tokens{t},tokens{tt});
                c = c + 1;
            end

            prospective_name = ['(',covariance2str(tokens{t}.fun),'*',covariance2str(tokens{tt}.fun),')'];
            if (t < tt && ~ismember(prospective_name, tokens_name))
                new_tokens_name{c} = prospective_name;
                new_tokens{c} = combine_tokens('*',tokens{t},tokens{tt});
                c = c + 1;
            end
            
        end
    end
    

    random_indx = randperm(numel(new_tokens_name),1);
    
    new_tokens_name = new_tokens_name{random_indx};
    new_tokens      = new_tokens{random_indx};
    
    tokens_name = [tokens_name, new_tokens_name];
    tokens      = [tokens, new_tokens];
    
end

end

function m = combine_tokens(op, m1,m2)
    switch op
        case '+'
            m.fun = {@sum_covariance, {m1.fun,m2.fun}};
        case '*'
            m.fun = {@prod_covariance, {m1.fun,m2.fun}};
        otherwise 
            error('Unknown operation');
    end
        m.priors = {m1.priors{:}, m2.priors{:}};
end