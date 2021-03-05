function [candidates, new_names] = remove_duplicate_candidates(...
    possible_candidates, used_names, base_cov_names)

candidates = {}; new_names = {};
names = used_names;

for i=1:length(possible_candidates)
    valid = 1;
    name = covariance2str(possible_candidates{i}.fun);
    for j=1:length(names)
        name_2 = names{j};
        counts_1 = zeros([5,1]); counts_2 = zeros([5,1]);
         % check if a candidate has the same combination of base
         % covariances as an already used covariance.
        for k=1:length(base_cov_names)
            counts_1(k) = length(strfind(name,base_cov_names{k}));
            counts_2(k) = length(strfind(name_2,base_cov_names{k}));
        end
        if ~all(counts_1 == counts_2)
            continue;
        end
        % if so, check to see if they have the same algebraic expression.
        equal = 1;
        for k=1:2
            SE = rand; RQ = rand; M1 = rand; LIN = rand; PER = rand;
            SEard = rand;
            name_exp = strrep(name, '_', '*');
            name2_exp = strrep(names{j}, '_', '*');
            if abs(eval(name_exp) - eval(name2_exp)) > 1e-6
                equal = 0;
                break;
            end
        end
        % if so, this candidate is a duplicate and can be discarded.
        if equal
            valid = 0;
            break;
        end     
    end
    if valid
        candidates = [candidates, possible_candidates{i}];
        names      = [names, name];
        new_names  = [new_names, name];
    end
end
